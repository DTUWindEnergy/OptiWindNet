"""Refresh notebook outputs in docs/notebooks via nbclient.

Executes each notebook with the kernel given via ``--kernel`` from cwd
``docs/notebooks`` so relative data paths resolve, then strips transient
metadata so the result matches the clean state checked into the repository.

Cells whose accumulated text output exceeds ``--scrolled-line-threshold``
lines get ``metadata.scrolled = true`` (aimed at MILP solver logs).

By default, notebooks that invoke a MILP / OR-tools solver are skipped
because they can take several minutes; pass ``--milp`` (or explicit paths)
to include them. A notebook is classified as MILP if any code cell mentions
``MILPRouter``, ``solver_factory``, ``ortools.cp_sat``, or other MILP
solver tags from ``MILP_PATTERN``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

HERE = Path(__file__).resolve().parent
NB_DIR = HERE / 'notebooks'
DEFAULT_TIMEOUT = 600
DEFAULT_SCROLLED_THRESHOLD = 24

MILP_PATTERN = re.compile(
    r'\b(?:MILPRouter|solver_factory|ortools\.cp_sat|gurobipy|'
    r'cplex|pyscipopt|highspy|cbcpy|MILP_router)\b'
)
MPL_PATTERN = re.compile(
    r'\b(?:gplot|pplot|matplotlib|pyplot|plt\.[a-zA-Z_]|mpl\.rcParams)'
)
MPL_SETUP_CODE = """\
# transient setup injected by docs/run_notebooks.py for stable SVG output
import matplotlib as mpl
mpl.rcParams.update({"svg.hashsalt": "fixed-salt-for-this-project"})
%config InlineBackend.print_figure_kwargs = {"metadata": {"Date": None, "Creator": "OptiWindNet"}}
"""


def cell_source(cell: nbformat.NotebookNode) -> str:
    src = cell.get('source', '')
    return ''.join(src) if isinstance(src, list) else src


def is_milp(path: Path) -> bool:
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.get('cell_type') == 'code' and MILP_PATTERN.search(cell_source(cell)):
            return True
    return False


def uses_matplotlib(nb: nbformat.NotebookNode) -> bool:
    return any(
        cell.get('cell_type') == 'code' and MPL_PATTERN.search(cell_source(cell))
        for cell in nb.cells
    )


def renumber_execution_counts(nb: nbformat.NotebookNode) -> None:
    """Reset code-cell execution_counts (and output counts) to 1..N."""
    n = 0
    for cell in nb.cells:
        if cell.get('cell_type') != 'code':
            continue
        if cell.get('execution_count') is None:
            continue
        n += 1
        cell['execution_count'] = n
        for out in cell.get('outputs', ()):
            if 'execution_count' in out:
                out['execution_count'] = n


def count_text_lines(cell: nbformat.NotebookNode) -> int:
    """Count lines of text-ish output across all outputs of a code cell."""
    if cell.get('cell_type') != 'code':
        return 0
    total = 0
    for out in cell.get('outputs', ()):
        otype = out.get('output_type')
        if otype == 'stream':
            text = out.get('text', '')
            if isinstance(text, list):
                text = ''.join(text)
            total += text.count('\n') + (1 if text and not text.endswith('\n') else 0)
        elif otype in ('execute_result', 'display_data'):
            data = out.get('data', {})
            text = data.get('text/plain', '')
            if isinstance(text, list):
                text = ''.join(text)
            total += text.count('\n') + (1 if text and not text.endswith('\n') else 0)
        elif otype == 'error':
            total += len(out.get('traceback', []))
    return total


def merge_adjacent_stream_outputs(cell: nbformat.NotebookNode) -> None:
    """Coalesce stream chunks split nondeterministically by Jupyter."""
    outputs = cell.get('outputs')
    if not outputs:
        return

    merged = []
    for out in outputs:
        if (
            merged
            and out.get('output_type') == 'stream'
            and merged[-1].get('output_type') == 'stream'
            and out.get('name') == merged[-1].get('name')
        ):
            merged[-1]['text'] = ''.join(merged[-1].get('text', '')) + ''.join(
                out.get('text', '')
            )
        else:
            merged.append(out)
    cell['outputs'] = merged


def clean_notebook(nb: nbformat.NotebookNode, scrolled_threshold: int) -> None:
    """Strip transient metadata in-place; mark long-output cells as scrolled."""
    nb.metadata = nbformat.from_dict({'language_info': {'name': 'python'}})
    for cell in nb.cells:
        merge_adjacent_stream_outputs(cell)
        meta = cell.get('metadata', {}) or {}
        meta.pop('execution', None)
        if cell.get('cell_type') == 'code':
            if count_text_lines(cell) > scrolled_threshold:
                meta['scrolled'] = True
            else:
                meta.pop('scrolled', None)
        cell['metadata'] = meta


def select_notebooks(args: argparse.Namespace) -> list[Path]:
    if args.notebooks:
        paths = [Path(p).resolve() for p in args.notebooks]
    else:
        paths = sorted(NB_DIR.glob('*.ipynb'))
        if not args.milp:
            paths = [p for p in paths if not is_milp(p)]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        sys.exit(f'not found: {", ".join(str(p) for p in missing)}')
    return paths


def run(args: argparse.Namespace) -> int:
    paths = select_notebooks(args)
    if not paths:
        print('no notebooks selected')
        return 0

    failures: list[tuple[Path, str]] = []
    for path in paths:
        rel = path.relative_to(NB_DIR) if path.is_relative_to(NB_DIR) else path.name
        label = f'[{"DRY" if args.dry_run else "RUN"}] {rel}'
        t0 = time.monotonic()
        injected = False
        try:
            nb = nbformat.read(path, as_version=4)
            if not args.no_mpl_setup and uses_matplotlib(nb):
                nb.cells.insert(0, nbformat.v4.new_code_cell(MPL_SETUP_CODE))
                injected = True
            client = NotebookClient(
                nb,
                timeout=args.timeout,
                kernel_name=args.kernel,
                resources={'metadata': {'path': str(NB_DIR)}},
            )
            client.execute()
        except CellExecutionError as exc:
            elapsed = time.monotonic() - t0
            print(f'{label}  FAIL  {elapsed:6.1f}s  {exc.ename}: {exc.evalue}')
            failures.append((path, f'{exc.ename}: {exc.evalue}'))
            continue
        except Exception as exc:
            elapsed = time.monotonic() - t0
            print(f'{label}  FAIL  {elapsed:6.1f}s  {type(exc).__name__}: {exc}')
            failures.append((path, f'{type(exc).__name__}: {exc}'))
            continue

        if injected:
            nb.cells.pop(0)
            renumber_execution_counts(nb)
        clean_notebook(nb, args.scrolled_line_threshold)
        elapsed = time.monotonic() - t0

        if args.dry_run:
            print(f'{label}   OK   {elapsed:6.1f}s  (not written)')
        else:
            tmp = path.with_suffix('.ipynb.tmp')
            nbformat.write(nb, tmp)
            os.replace(tmp, path)
            print(f'{label}   OK   {elapsed:6.1f}s')

    if failures:
        print(f'\n{len(failures)} failure(s):')
        for path, msg in failures:
            print(f'  {path.name}: {msg}')
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description='Refresh notebook outputs in docs/notebooks via nbclient.'
    )
    p.add_argument(
        '--kernel',
        required=True,
        help='Jupyter kernel spec name to execute notebooks with',
    )
    p.add_argument(
        'notebooks',
        nargs='*',
        help='specific .ipynb paths (default: all in docs/notebooks)',
    )
    p.add_argument(
        '--milp',
        action='store_true',
        help='include MILP notebooks (skipped by default; ignored when paths are given)',
    )
    p.add_argument(
        '-n',
        '--dry-run',
        action='store_true',
        help='execute and validate, but do not write results back',
    )
    p.add_argument(
        '--timeout',
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f'per-cell timeout in seconds (default: {DEFAULT_TIMEOUT})',
    )
    p.add_argument(
        '--no-mpl-setup',
        action='store_true',
        help='skip injection of matplotlib SVG-stability setup',
    )
    p.add_argument(
        '--scrolled-line-threshold',
        type=int,
        default=DEFAULT_SCROLLED_THRESHOLD,
        help=(
            'cells with more text-output lines than this get '
            f'metadata.scrolled=true (default: {DEFAULT_SCROLLED_THRESHOLD})'
        ),
    )
    return run(p.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
