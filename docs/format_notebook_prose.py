"""Run Prettier over the *markdown cells* of the notebooks.

Prettier formats Markdown files, and a notebook is not one: it is JSON that
happens to carry Markdown inside string arrays. Prettier has no parser for
``.ipynb`` at all -- handed one, it fails -- so the prose in the notebooks is
reachable only by taking the markdown cells out, which is what this script does.

Each markdown cell is written to a scratch ``.md`` file, the whole batch is
handed to Prettier in one call, and the results are read back into the notebook.
The project's ``.prettierrc`` is passed explicitly, since the scratch files sit
outside the repository and Prettier would otherwise find no configuration::

    python docs/format_notebook_prose.py --check
    python docs/format_notebook_prose.py --write

Code and raw cells, and cell outputs, are never touched. The JSON is written the
way JupyterLab writes it -- ``indent=1``, ``source`` as a list of lines, trailing
newline -- so an unchanged notebook is byte-identical afterwards.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB_DIR = HERE / 'notebooks'
CONFIG = HERE.parent / '.prettierrc'


def notebook_paths() -> list[Path]:
    return sorted(NB_DIR.glob('*.ipynb'))


def run_prettier(texts: list[str]) -> list[str]:
    """Format each markdown fragment, in one Prettier invocation."""
    if not texts:
        return []
    prettier = shutil.which('prettier')
    if prettier is None:
        sys.exit('prettier not found on PATH; install it or adjust PATH')

    with tempfile.TemporaryDirectory(prefix='nbprose-') as tmp:
        paths = []
        for index, text in enumerate(texts):
            path = Path(tmp) / f'cell{index:05d}.md'
            path.write_text(text, encoding='utf8')
            paths.append(path)
        result = subprocess.run(
            [prettier, '--config', str(CONFIG), '--write', *map(str, paths)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            sys.exit(f'prettier failed:\n{result.stderr.strip()}')
        return [path.read_text(encoding='utf8') for path in paths]


def markdown_cells(notebooks: list[Path]) -> tuple[list[dict], list[str]]:
    """Return every markdown cell and its current text, across the notebooks."""
    cells, texts = [], []
    for path in notebooks:
        nb = json.loads(path.read_text(encoding='utf8'))
        for cell in nb['cells']:
            if cell['cell_type'] == 'markdown':
                cells.append({'path': path, 'nb': nb, 'cell': cell})
                texts.append(''.join(cell['source']))
    return cells, texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('paths', nargs='*', type=Path, help='notebooks (default: all)')
    parser.add_argument('--write', action='store_true', help='rewrite the notebooks')
    parser.add_argument('--check', action='store_true', help='report and exit 1')
    args = parser.parse_args()

    notebooks = args.paths or notebook_paths()
    cells, texts = markdown_cells(notebooks)
    formatted = run_prettier(texts)

    # Prettier ends a file with a newline; a notebook's last source line does not
    # carry one.
    changed: dict[Path, int] = {}
    for entry, new in zip(cells, formatted):
        new = new.rstrip('\n')
        if new == ''.join(entry['cell']['source']):
            continue
        changed[entry['path']] = changed.get(entry['path'], 0) + 1
        lines = new.split('\n')
        entry['cell']['source'] = [line + '\n' for line in lines[:-1]] + lines[-1:]

    for path in sorted(changed):
        verb = 'reformatted' if args.write else 'would reformat'
        print(
            f'{path.relative_to(HERE.parent)}: {verb} {changed[path]} markdown cell(s)'
        )

    if args.write:
        written = {
            entry['path']: entry['nb'] for entry in cells if entry['path'] in changed
        }
        for path, nb in written.items():
            path.write_text(
                json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf8'
            )

    if not changed:
        print('every markdown cell is already formatted')
        return 0
    print(
        f'\n{len(changed)} of {len(notebooks)} notebooks, {sum(changed.values())} cells'
    )
    return 1 if args.check else 0


if __name__ == '__main__':
    raise SystemExit(main())
