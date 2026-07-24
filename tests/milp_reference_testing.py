"""Reference loading and execution helpers for MILP regression tests."""

from __future__ import annotations

import json
import math
import pickle
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import networkx as nx

from optiwindnet.fingerprint import fingerprint_coordinates
from optiwindnet.MILP import SolutionInfo
from optiwindnet.terse import LinkScope, TerseLinks

from .cases import MILP_ADAPTER_CASES, MILPCase, case_node_id
from .helpers import run_milp_solve_with_retry
from .paths import REPO_ROOT
from .sitecache import get_bundle_from_nodeset_digest, get_location
from .topology_assertions import assert_topology
from .update_milp_reference_candidates import (
    reference_candidate_matrix,
    reference_problem_key,
)

MILP_REFERENCES_FILE = Path(__file__).with_name('milp_references.pkl')
PROVISIONAL_REFERENCES_FILE = REPO_ROOT / 'artifacts' / 'milp_reference_candidates.json'
TEST_TIME_LIMIT = 1.0


@dataclass(frozen=True, slots=True)
class MILPReference:
    """A proven bound/objective pair and its topology-scoped optimum."""

    bound: float
    objective: float
    topology: TerseLinks


@dataclass(frozen=True, slots=True)
class MILPReferenceExecution:
    """One backend, one problem case, and one warm-start mode."""

    case: MILPCase
    warmstart: bool


# Indices in reference_candidate_matrix() that CPLEX proved optimal. The three
# omitted ringed cases remain candidates, not reference data.
_REFERENCE_CASE_INDICES = (0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14, 15)

# Each backend occurs exactly once cold and once warm. Thirteen distinct reference
# problems are covered; the remaining five executions repeat cases across
# different adapters.
_EXECUTION_ASSIGNMENTS = (
    ('ortools.cp_sat', 6, False),
    ('ortools.cp_sat', 1, True),
    ('ortools.gscip', 13, False),
    ('ortools.gscip', 2, True),
    ('ortools.highs', 14, False),
    ('ortools.highs', 3, True),
    ('highs', 0, False),
    ('highs', 5, True),
    ('scip', 13, False),
    ('scip', 8, True),
    ('gurobi', 3, False),
    ('gurobi', 9, True),
    ('cplex', 13, False),
    ('cplex', 15, True),
    ('cbc', 15, False),
    ('cbc', 12, True),
    ('fscip', 13, False),
    ('fscip', 11, True),
)


def _make_reference_executions() -> tuple[MILPReferenceExecution, ...]:
    candidates = reference_candidate_matrix(time_limit=TEST_TIME_LIMIT)
    executions = tuple(
        MILPReferenceExecution(
            replace(
                candidates[candidate_index],
                solver_name=solver_name,
                time_limit=TEST_TIME_LIMIT,
            ),
            warmstart,
        )
        for solver_name, candidate_index, warmstart in _EXECUTION_ASSIGNMENTS
    )

    expected_backends = {case.solver_name for case in MILP_ADAPTER_CASES}
    backend_counts = Counter(item.case.solver_name for item in executions)
    expected_counts = {backend: 2 for backend in expected_backends}
    if backend_counts != expected_counts:
        raise ValueError(
            f'MILP reference backend counts differ: '
            f'expected={expected_counts}, actual={dict(backend_counts)}'
        )
    modes = Counter((item.case.solver_name, item.warmstart) for item in executions)
    expected_modes = {
        (backend, warmstart): 1
        for backend in expected_backends
        for warmstart in (False, True)
    }
    if modes != expected_modes:
        raise ValueError('each MILP backend must run exactly once cold and once warm')
    if any(item.case.time_limit != TEST_TIME_LIMIT for item in executions):
        raise ValueError('every MILP reference execution must use a 1 s time limit')

    expected_problem_keys = {
        reference_problem_key(candidates[index]) for index in _REFERENCE_CASE_INDICES
    }
    execution_problem_keys = {reference_problem_key(item.case) for item in executions}
    if execution_problem_keys != expected_problem_keys:
        missing = sorted(expected_problem_keys - execution_problem_keys)
        stale = sorted(execution_problem_keys - expected_problem_keys)
        raise ValueError(
            f'MILP reference execution coverage differs: '
            f'missing={missing}, stale={stale}'
        )
    return executions


MILP_REFERENCE_EXECUTIONS = _make_reference_executions()


def reference_execution_id(execution: MILPReferenceExecution) -> str:
    """Return a readable pytest id including the warm-start mode."""
    mode = 'warm' if execution.warmstart else 'cold'
    return f'{case_node_id(execution.case)}-{mode}'


def _case_options(case: MILPCase) -> dict[str, Any]:
    return {
        name: getattr(value, 'value', value)
        for name, value in case.model_options.items()
    }


def _expected_cases_by_key() -> dict[str, MILPCase]:
    cases_by_key = {
        reference_problem_key(execution.case): execution.case
        for execution in MILP_REFERENCE_EXECUTIONS
    }
    return cases_by_key


def _validate_milp_references(
    references: object,
    *,
    source: Path,
) -> dict[str, MILPReference]:
    if not isinstance(references, dict) or not all(
        isinstance(key, str) for key in references
    ):
        raise TypeError(f'{source}: references must be a dictionary keyed by problem')

    cases_by_key = _expected_cases_by_key()
    if references.keys() != cases_by_key.keys():
        missing = sorted(cases_by_key.keys() - references.keys())
        stale = sorted(references.keys() - cases_by_key.keys())
        raise ValueError(
            f'{source}: MILP reference keys differ: missing={missing}, stale={stale}'
        )

    for key, case in cases_by_key.items():
        reference = references[key]
        if not isinstance(reference, MILPReference):
            raise TypeError(f'{source}: {key}: value must be MILPReference')
        if not math.isfinite(reference.objective) or not math.isfinite(reference.bound):
            raise ValueError(f'{source}: {key}: objective and bound must be finite')
        tolerance = max(1e-8, abs(reference.objective) * 1e-9)
        if not math.isclose(
            reference.bound,
            reference.objective,
            rel_tol=1e-9,
            abs_tol=tolerance,
        ):
            raise ValueError(f'{source}: {key}: optimal bound and objective differ')

        topology = reference.topology
        if topology.nodeset_digest is None:
            raise ValueError(f'{source}: {key}: topology has no nodeset digest')
        bundle = get_bundle_from_nodeset_digest(topology.nodeset_digest)
        if bundle.handle != case.site:
            raise ValueError(
                f'{source}: {key}: digest resolves to {bundle.handle!r}, '
                f'expected {case.site!r}'
            )
        L = bundle.L
        expected_metadata = (
            LinkScope.TOPOLOGY,
            case.model_options['topology'],
            L.graph['T'],
            L.graph['R'],
        )
        actual_metadata = (topology.scope, topology.topology, topology.T, topology.R)
        if actual_metadata != expected_metadata:
            raise ValueError(f'{source}: {key}: topology metadata differs')
        reconstructed = topology.to_topology(
            capacity=case.capacity,
            creator='milp-reference',
        )
        assert_topology(
            reconstructed,
            case.model_options['topology'],
            case.capacity,
        )
    return references


def load_milp_references(
    path: Path = MILP_REFERENCES_FILE,
) -> dict[str, MILPReference]:
    """Load and strictly validate the deployed MILP reference artifact."""
    if not path.exists():
        raise FileNotFoundError(
            f'Missing MILP references: {path}\n'
            'Deploy them with: python -m tests.update_milp_references'
        )
    with path.open('rb') as file:
        references = pickle.load(file)
    return _validate_milp_references(references, source=path)


def load_provisional_milp_references(
    path: Path = PROVISIONAL_REFERENCES_FILE,
) -> dict[str, MILPReference]:
    """Load temporary CPLEX results for deployment without solving again."""
    if not path.exists():
        raise FileNotFoundError(
            f'Missing provisional MILP references: {path}\n'
            'Generate them with: python -m tests.update_milp_reference_candidates'
        )
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise ValueError(f'{path}: unsupported provisional reference schema')
    stored = data.get('cases')
    if not isinstance(stored, dict):
        raise TypeError(f'{path}: "cases" must be a dictionary')

    cases_by_key = _expected_cases_by_key()
    if stored.keys() != cases_by_key.keys():
        missing = sorted(cases_by_key.keys() - stored.keys())
        stale = sorted(stored.keys() - cases_by_key.keys())
        raise ValueError(
            f'{path}: provisional reference keys differ: '
            f'missing={missing}, stale={stale}'
        )

    references = {}
    for key, case in cases_by_key.items():
        record = stored[key]
        if not isinstance(record, dict):
            raise TypeError(f'{path}: {key}: record must be a dictionary')
        stored_case = record.get('case')
        if not isinstance(stored_case, dict):
            raise TypeError(f'{path}: {key}: case must be a dictionary')
        expected_case = {
            'site': case.site,
            'capacity': case.capacity,
            'model_options': _case_options(case),
        }
        for name, expected in expected_case.items():
            if stored_case.get(name) != expected:
                raise ValueError(f'{path}: {key}: stored {name} differs from live case')

        solution = record.get('solution')
        if not isinstance(solution, dict):
            raise TypeError(f'{path}: {key}: solution must be a dictionary')
        if str(solution.get('termination', '')).lower() != 'optimal':
            raise ValueError(f'{path}: {key}: reference was not proven optimal')
        topology_data = record.get('solution_topology')
        if not isinstance(topology_data, dict):
            raise TypeError(f'{path}: {key}: solution_topology must be a dictionary')
        topology = TerseLinks.from_dict(topology_data)
        site = record.get('site')
        if not isinstance(site, dict):
            raise TypeError(f'{path}: {key}: site must be a dictionary')
        digest_hex = site.get('nodeset_digest')
        if digest_hex is None:
            L = get_location(case.site)
            nodeset_digest = fingerprint_coordinates(L.graph['VertexC'])[0]
        elif isinstance(digest_hex, str):
            nodeset_digest = bytes.fromhex(digest_hex)
        else:
            raise TypeError(f'{path}: {key}: nodeset_digest must be hexadecimal')
        if topology.nodeset_digest is None:
            topology = replace(topology, nodeset_digest=nodeset_digest)
        elif topology.nodeset_digest != nodeset_digest:
            raise ValueError(f'{path}: {key}: topology and site digests differ')
        references[key] = MILPReference(
            bound=float(solution['bound']),
            objective=float(solution['objective']),
            topology=topology,
        )
    return _validate_milp_references(references, source=path)


def solve_milp_reference_execution(
    execution: MILPReferenceExecution,
    reference: MILPReference,
) -> tuple[SolutionInfo, nx.Graph, str]:
    """Run one 1 s cold or warm MILP reference execution."""
    case = execution.case
    if case.time_limit != TEST_TIME_LIMIT:
        raise ValueError('MILP reference tests require exactly a 1 s time limit')
    nodeset_digest = reference.topology.nodeset_digest
    if nodeset_digest is None:
        raise ValueError('MILP reference topology has no nodeset digest')
    bundle = get_bundle_from_nodeset_digest(nodeset_digest)
    if bundle.handle != case.site:
        raise ValueError(
            f'reference digest resolves to {bundle.handle!r}, expected {case.site!r}'
        )
    warmstart = None
    if execution.warmstart:
        warmstart = reference.topology.to_topology(
            capacity=case.capacity,
            creator='milp-reference-optimum',
        )
        assert_topology(warmstart, case.model_options['topology'], case.capacity)

    return run_milp_solve_with_retry(
        bundle.P,
        bundle.A,
        solver_name=case.solver_name,
        capacity=case.capacity,
        model_options=case.model_options,
        time_limit=TEST_TIME_LIMIT,
        mip_gap=case.mip_gap,
        warmstart=warmstart,
    )
