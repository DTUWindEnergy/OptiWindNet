"""Cross-backend MILP checks against deployed proven-optimal references."""

from __future__ import annotations

import math

import pytest

from .helpers import solver_unavailable
from .isolation import should_isolate
from .milp_reference_testing import (
    MILP_REFERENCE_EXECUTIONS,
    load_milp_references,
    reference_execution_id,
    solve_milp_reference_execution,
)
from .solver_topologies import assert_matches_golden
from .topology_assertions import assert_topology
from .update_milp_reference_candidates import reference_problem_key


@pytest.fixture(scope='module')
def milp_references():
    return load_milp_references()


@pytest.mark.parametrize(
    'execution',
    MILP_REFERENCE_EXECUTIONS,
    ids=reference_execution_id,
)
def test_milp_reference_execution(execution, milp_references, ortools_worker):
    case = execution.case
    reference = milp_references[reference_problem_key(case)]
    if should_isolate(case.solver_name):
        result = ortools_worker.run(
            solve_milp_reference_execution,
            (execution, reference),
            timeout=30 + case.time_limit,
        )
    else:
        try:
            result = solve_milp_reference_execution(execution, reference)
        except BaseException as exc:
            result = exc

    if isinstance(result, BaseException) and solver_unavailable(result):
        pytest.skip(f'{case.solver_name} unavailable: {result}')
    if isinstance(result, BaseException):
        raise result

    info, S, warmed_by = result
    assert bool(warmed_by) is execution.warmstart
    assert_topology(S, case.model_options['topology'], case.capacity)

    if execution.warmstart or info.termination.lower() == 'optimal':
        assert_matches_golden(S, reference.topology)
    if info.termination.lower() == 'optimal':
        return

    assert math.isfinite(info.bound)
    assert math.isfinite(info.objective)
    assert info.bound <= info.objective
    tolerance = max(1e-8, abs(reference.objective) * 1e-9)
    assert info.bound <= reference.objective + tolerance
    assert reference.objective <= info.objective + tolerance
