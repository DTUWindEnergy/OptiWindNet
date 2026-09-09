# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Cross-backend MILP checks against deployed proven-optimal references."""

import math

import pytest

from .helpers import solver_unavailable
from .milp_reference_testing import (
    MILP_REFERENCE_EXECUTIONS,
    load_milp_references,
    reference_execution_id,
    reference_topology_id,
    solve_milp_reference_execution,
)
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
def test_milp_reference_execution(execution, milp_references, run_isolated):
    case = execution.case
    reference = milp_references[reference_problem_key(case)]
    result = run_isolated(
        case.solver_name,
        solve_milp_reference_execution,
        (execution, reference),
        30 + case.time_limit,
    )
    if isinstance(result, BaseException) and solver_unavailable(result):
        pytest.skip(f'{case.solver_name} unavailable: {result}')
    if isinstance(result, BaseException):
        raise result

    info, S, warmed_by = result
    assert bool(warmed_by) is execution.warmstart
    assert_topology(S, case.model_options['topology'], case.capacity)

    if execution.warmstart or info.termination.lower() == 'optimal':
        # the incumbent, not the routed pool pick: the reference is the optimum
        # of the model objective, which is what solve() stamps on SolutionInfo
        assert info.topology_id == reference_topology_id(reference, case)
    if info.termination.lower() == 'optimal':
        return

    assert math.isfinite(info.bound)
    assert math.isfinite(info.objective)
    assert info.bound <= info.objective
    tolerance = max(1e-8, abs(reference.objective) * 1e-9)
    assert info.bound <= reference.objective + tolerance
    assert reference.objective <= info.objective + tolerance
