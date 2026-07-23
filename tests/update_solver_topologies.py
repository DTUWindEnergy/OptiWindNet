"""Explicitly regenerate deterministic undetoured solver topology goldens.

Run from the repository root with::

    NUMBA_CACHE_DIR="$PWD/.numba_cache" python -m tests.update_solver_topologies
"""

import pickle

from optiwindnet.heuristics import constructor
from optiwindnet.terse import TerseLinks

from .cases import (
    CONSTRUCTOR_CASES,
    MILP_FORMULATION_CASES,
    expected_topology,
    topology_golden_key,
)
from .solver_topologies import SOLVER_TOPOLOGIES_FILE, solve_milp_case
from .sitecache import get_bundle
from .topology_assertions import assert_topology


def generate() -> dict[str, TerseLinks]:
    goldens = {}
    for case in CONSTRUCTOR_CASES:
        if not case.exact_golden:
            continue
        A = get_bundle(case.site).A
        S = constructor(
            A,
            capacity=case.capacity,
            method=case.method,
            bias_margin=case.bias_margin,
            weigh_detours=case.feeder_route.value == 'segmented',
            straight_feeder_route=case.feeder_route.value == 'straight',
        )
        assert_topology(S, expected_topology(case), case.capacity)
        goldens[topology_golden_key(case)] = TerseLinks.from_topology(S)

    # The primary required solver owns the shared adapter problem golden.
    case = next(case for case in MILP_FORMULATION_CASES if case.exact_golden)
    _, S = solve_milp_case(case)
    assert_topology(S, case.model_options['topology'], case.capacity)
    goldens[topology_golden_key(case)] = TerseLinks.from_topology(S)
    return goldens


def main() -> None:
    goldens = generate()
    with SOLVER_TOPOLOGIES_FILE.open('wb') as file:
        pickle.dump(goldens, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved {len(goldens)} topology goldens to {SOLVER_TOPOLOGIES_FILE}')


if __name__ == '__main__':
    main()
