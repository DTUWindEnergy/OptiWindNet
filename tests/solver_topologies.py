"""Strict loader and low-level solve helpers for topology goldens."""

from __future__ import annotations

import pickle
from pathlib import Path

from optiwindnet.MILP import solver_factory
from optiwindnet.terse import LinkScope, TerseLinks

from .cases import (
    CONSTRUCTOR_CASES,
    MILP_ADAPTER_CASES,
    MILP_FORMULATION_CASES,
    MILPCase,
    expected_topology,
    golden_keys,
    topology_golden_key,
)
from .sitecache import get_bundle, get_location

SOLVER_TOPOLOGIES_FILE = Path(__file__).with_name('solver_topologies.pkl')
GoldenValue = TerseLinks | tuple[TerseLinks, ...]


def solve_milp_case(case: MILPCase):
    """Solve a typed case and return only its undetoured incumbent topology."""
    bundle = get_bundle(case.site)
    solver = solver_factory(case.solver_name)
    solver.set_problem(
        bundle.P,
        bundle.A,
        capacity=case.capacity,
        model_options=case.model_options,
    )
    info = solver.solve(time_limit=case.time_limit, mip_gap=case.mip_gap)
    S = solver.get_incumbent_topology()
    return info, S


def _expected_metadata() -> dict[str, tuple[int, int, object]]:
    expected = {}
    for case in CONSTRUCTOR_CASES:
        if not case.exact_golden:
            continue
        L = get_location(case.site)
        key = topology_golden_key(case)
        expected[key] = (L.graph['T'], L.graph['R'], expected_topology(case))
    for case in (*MILP_FORMULATION_CASES, *MILP_ADAPTER_CASES):
        if not case.exact_golden:
            continue
        L = get_location(case.site)
        metadata = (L.graph['T'], L.graph['R'], case.model_options['topology'])
        key = topology_golden_key(case)
        previous = expected.setdefault(key, metadata)
        if previous != metadata:
            raise ValueError(f'golden key {key!r} has conflicting cases')
    return expected


def load_solver_topologies() -> dict[str, GoldenValue]:
    """Load and fully validate the topology-scoped golden artifact."""
    if not SOLVER_TOPOLOGIES_FILE.exists():
        raise FileNotFoundError(
            f'Missing solver topology goldens: {SOLVER_TOPOLOGIES_FILE}\n'
            'Regenerate with: python -m tests.update_solver_topologies'
        )
    with SOLVER_TOPOLOGIES_FILE.open('rb') as file:
        data = pickle.load(file)
    if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
        raise TypeError('solver topology goldens must be a dict keyed by problem')

    expected = _expected_metadata()
    if data.keys() != expected.keys() or data.keys() != golden_keys():
        missing = sorted(golden_keys() - data.keys())
        stale = sorted(data.keys() - golden_keys())
        raise ValueError(
            f'solver topology golden keys differ: missing={missing}, stale={stale}'
        )

    for key, value in data.items():
        values = value if isinstance(value, tuple) else (value,)
        if not values or not all(isinstance(item, TerseLinks) for item in values):
            raise TypeError(
                f'{key}: value must be TerseLinks or a non-empty tuple of them'
            )
        T, R, topology = expected[key]
        for item in values:
            if item.scope is not LinkScope.TOPOLOGY:
                raise ValueError(f'{key}: golden scope must be topology')
            if (item.T, item.R, item.topology) != (T, R, topology):
                raise ValueError(f'{key}: golden metadata differs from live case')
    return data


def assert_matches_golden(S, value: GoldenValue) -> None:
    """Compare topology ``S`` with all explicitly accepted tied optima."""
    accepted = value if isinstance(value, tuple) else (value,)
    assert any(
        TerseLinks.from_topology(S, nodeset_digest=item.nodeset_digest) == item
        for item in accepted
    )
