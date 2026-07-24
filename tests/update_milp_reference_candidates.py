"""Solve provisional MILP reference cases and retain proven optima temporarily.

This is deliberately not the final test artifact generator. Until deployment
storage is agreed, successful candidates are stored as readable, gitignored
JSON under ``artifacts/``. :mod:`tests.update_milp_references` validates and
promotes that file without rerunning any solver.

Run one candidate from the repository root, for example::

    python -m tests.update_milp_reference_candidates neart 5

For batches, import :func:`update_candidates` and pass multiple
``MILPCase`` values. Site bundles are reused by handle, so cases on
the same location share their ``L``, ``P``, and ``A`` objects.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from optiwindnet.MILP import ModelOptions

from .cases import MILPCase
from .paths import REPO_ROOT
from .probe_milp_generation import (
    milp_case_from_args,
    milp_reference_parser,
    solve_milp_reference,
)
from .sitecache import get_bundle, SiteBundle

DEFAULT_OUTPUT = REPO_ROOT / 'artifacts' / 'milp_reference_candidates.json'
_SCHEMA_VERSION = 1


def reference_candidate_matrix(
    *,
    solver_name: str = 'ortools.cp_sat',
    time_limit: float = 30.0,
    mip_gap: float = 1e-8,
) -> tuple[MILPCase, ...]:
    """Return the provisional 16-case matrix currently under review."""
    return (
        MILPCase('neart', solver_name, 5, time_limit=time_limit, mip_gap=mip_gap),
        MILPCase(
            'neart',
            solver_name,
            8,
            ModelOptions(feeder_route='straight', feeder_limit='min_plus2'),
            time_limit,
            mip_gap,
        ),
        MILPCase(
            'neart',
            solver_name,
            7,
            ModelOptions(
                topology='radial',
                feeder_limit='minimum',
                balanced=True,
            ),
            time_limit,
            mip_gap,
        ),
        MILPCase(
            'galloper',
            solver_name,
            4,
            ModelOptions(topology='ringed'),
            time_limit,
            mip_gap,
        ),
        MILPCase(
            'neart',
            solver_name,
            3,
            ModelOptions(topology='ringed'),
            time_limit,
            mip_gap,
        ),
        MILPCase(
            'borkum2',
            solver_name,
            5,
            time_limit=time_limit,
            mip_gap=mip_gap,
        ),
        MILPCase(
            'borkum2',
            solver_name,
            8,
            ModelOptions(
                topology='radial',
                feeder_route='straight',
                feeder_limit='minimum',
            ),
            time_limit,
            mip_gap,
        ),
        MILPCase(
            'borkum2',
            solver_name,
            2,
            ModelOptions(topology='ringed'),
            time_limit,
            mip_gap,
        ),
        MILPCase(
            'cazzaro_2022',
            solver_name,
            5,
            time_limit=time_limit,
            mip_gap=mip_gap,
        ),
        MILPCase(
            'cazzaro_2022',
            solver_name,
            6,
            ModelOptions(
                topology='radial',
                feeder_route='straight',
                feeder_limit='specified',
                max_feeders=10,
            ),
            time_limit,
            mip_gap,
        ),
        MILPCase(
            'cazzaro_2022',
            solver_name,
            3,
            ModelOptions(topology='ringed'),
            time_limit,
            mip_gap,
        ),
        MILPCase(
            'galloper',
            solver_name,
            5,
            ModelOptions(
                topology='radial',
                feeder_limit='exactly',
                max_feeders=9,
                balanced=True,
            ),
            time_limit,
            mip_gap,
        ),
        MILPCase(
            'galloper',
            solver_name,
            8,
            ModelOptions(feeder_limit='min_plus2'),
            time_limit,
            mip_gap,
        ),
        MILPCase(
            'mermaid',
            solver_name,
            5,
            time_limit=time_limit,
            mip_gap=mip_gap,
        ),
        MILPCase(
            'mermaid',
            solver_name,
            3,
            ModelOptions(topology='ringed', feeder_route='straight'),
            time_limit,
            mip_gap,
        ),
        MILPCase(
            'albatros',
            solver_name,
            5,
            time_limit=time_limit,
            mip_gap=mip_gap,
        ),
    )


def reference_problem_key(case: MILPCase) -> str:
    """Return a solver/control-independent key for one candidate problem."""
    options = case.model_options
    tokens = [
        f'site-{case.site}',
        f'capacity-{case.capacity}',
        *(
            f'{name.replace("_", "-")}-{str(getattr(value, "value", value)).lower()}'
            for name, value in options.items()
        ),
    ]
    return '__'.join(tokens)


def _load_candidates(output: Path) -> dict[str, Any]:
    if not output.exists():
        return {'schema_version': _SCHEMA_VERSION, 'cases': {}}
    data = json.loads(output.read_text())
    if not isinstance(data, dict) or data.get('schema_version') != _SCHEMA_VERSION:
        raise ValueError(f'{output}: unsupported candidate storage schema')
    if not isinstance(data.get('cases'), dict):
        raise TypeError(f'{output}: "cases" must be a dictionary')
    return data


def _write_candidates(output: Path, data: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temporary.replace(output)


def update_candidates(
    cases: Iterable[MILPCase],
    *,
    output: Path = DEFAULT_OUTPUT,
    cold: bool = False,
    warmstart_time_limit: float = 0.3,
    warmstart_seed: int = 0,
    warmstart_max_retries: int = 2,
    verbose: bool = False,
) -> dict[str, Any]:
    """Solve candidates, cache site meshes, and store only proven optima."""
    storage = _load_candidates(output)
    stored_cases: dict[str, Any] = storage['cases']
    bundles: dict[str, SiteBundle] = {}
    results: dict[str, dict[str, object]] = {}
    stored: list[str] = []
    not_stored: list[str] = []

    for case in cases:
        bundle = bundles.get(case.site)
        if bundle is None:
            bundle = bundles[case.site] = get_bundle(case.site)
        key = reference_problem_key(case)
        result: dict[str, object]
        try:
            result = solve_milp_reference(
                case,
                bundle=bundle,
                cold=cold,
                warmstart_time_limit=warmstart_time_limit,
                warmstart_seed=warmstart_seed,
                warmstart_max_retries=warmstart_max_retries,
                verbose=verbose,
            )
        except Exception as exc:
            result = {
                'case': {
                    'site': case.site,
                    'capacity': case.capacity,
                    'solver_name': case.solver_name,
                    'model_options': {
                        name: getattr(value, 'value', value)
                        for name, value in case.model_options.items()
                    },
                },
                'solution': {
                    'termination': f'error:{type(exc).__name__}',
                    'bound': None,
                    'objective': None,
                    'runtime': None,
                },
                'error': str(exc),
            }
        results[key] = result
        solution = result['solution']
        if (
            isinstance(solution, dict)
            and str(solution.get('termination', '')).lower() == 'optimal'
        ):
            stored_cases[key] = result
            stored.append(key)
        else:
            not_stored.append(key)

    if stored:
        _write_candidates(output, storage)
    return {
        'output': str(output),
        'stored': stored,
        'not_stored': not_stored,
        'results': results,
    }


def main() -> None:
    parser = milp_reference_parser(positional_required=False)
    parser.description = __doc__
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        '--matrix',
        action='store_true',
        help='run the provisional 16-case candidate matrix',
    )
    args = parser.parse_args()
    if args.matrix:
        if args.site is not None or args.capacity is not None:
            parser.error('site and capacity cannot be combined with --matrix')
        cases = reference_candidate_matrix(
            solver_name=args.solver_name,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
        )
    else:
        if args.site is None or args.capacity is None:
            parser.error('site and capacity are required unless --matrix is used')
        cases = (milp_case_from_args(args),)
    print(
        json.dumps(
            update_candidates(
                cases,
                output=args.output,
                cold=args.cold,
                warmstart_time_limit=args.warmstart_time_limit,
                warmstart_seed=args.warmstart_seed,
                warmstart_max_retries=args.warmstart_max_retries,
                verbose=args.verbose,
            ),
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
