import logging
import math

import numpy as np
import pytest

import optiwindnet.MILP as MILP
import optiwindnet.MILP._core as core
from optiwindnet.interarraylib import terse_links_from_S
from optiwindnet.mesh import make_planar_embedding
from optiwindnet.MILP import ModelOptions, solver_factory
from optiwindnet.synthetic import toyfarm

# topology in terse links for toy_farm at capacity=5
_terse_toy_farm_5 = np.array([2, -1, 1, 2, -1, -1, 3, 4, -1, 5, 8, 8])
_CAPACITY = 5
_RUNTIME = 10
_GAP = 0.001


# ortools.math_opt bundles its own copies of HiGHS/SCIP that collide with the
# standalone highspy/pyscipopt packages if loaded into the same process (DLL
# hell). Only 'ortools*' work is dispatched to the shared `ortools_worker`
# subprocess fixture (see tests/conftest.py); every other solver runs directly
# in this process.
#
# Note: this module must not import optiwindnet.MILP.ortools (or anything else
# that transitively loads ortools' native libraries) at module scope --
# `ortools_worker` re-imports this module in its persistent worker process to
# unpickle whichever job function it's given, and that worker process must
# stay ortools-only (never load standalone highspy/pyscipopt).
def _solve_toy(solver_name, P, A):
    solver = solver_factory(solver_name)
    solver.set_problem(P, A, capacity=_CAPACITY, model_options=ModelOptions())
    solution_info = solver.solve(time_limit=_RUNTIME, mip_gap=_GAP)
    return solution_info, solver.get_solution()


def _solver_is_unavailable(exc: BaseException) -> bool:
    if isinstance(exc, (FileNotFoundError, ModuleNotFoundError)):
        return True

    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            'unable to create gurobi model',
            'token.gurobi.com',
            'not licensed',
            'license',
        )
    )


def _patch_cpu_topology(monkeypatch, affinity, mapping=None):
    class FakePath:
        def __init__(self, path):
            self.path = path

        def __truediv__(self, child):
            return type(self)(f'{self.path}/{child}')

        def read_text(self):
            if mapping is None:
                raise OSError('sysfs unavailable')
            return mapping[self.path]

    monkeypatch.setattr(core.sys, 'platform', 'linux')
    monkeypatch.setattr(
        core.os, 'sched_getaffinity', lambda pid: affinity, raising=False
    )
    monkeypatch.setattr(core, 'Path', FakePath)


@pytest.fixture(scope='module')
def P_A_toy():
    L = toyfarm()
    P, A = make_planar_embedding(L)
    return P, A


@pytest.mark.parametrize(
    'solver_name',
    ['ortools.cp_sat', 'gurobi', 'cplex', 'highs', 'scip', 'cbc', 'fscip'],
)
def test_MILP_solvers(P_A_toy, solver_name, ortools_worker):
    P, A = P_A_toy
    if solver_name.startswith('ortools'):
        result = ortools_worker.run(_solve_toy, (solver_name, P, A), 30 + _RUNTIME)
    else:
        try:
            result = _solve_toy(solver_name, P, A)
        except BaseException as exc:
            result = exc

    if isinstance(result, BaseException) and _solver_is_unavailable(result):
        pytest.skip(f'{solver_name} not available')
    if isinstance(result, BaseException):
        raise result

    solution_info, (S, _) = result
    assert solution_info.termination.lower() == 'optimal'
    assert (terse_links_from_S(S) == _terse_toy_farm_5).all()


def _solve_toy_balanced(solver_name, P, A, max_feeders):
    """Solve toyfarm with the feeder count pinned to ``max_feeders`` and balanced."""
    solver = solver_factory(solver_name)
    solver.set_problem(
        P,
        A,
        capacity=_CAPACITY,
        model_options=ModelOptions(
            feeder_limit='exactly', max_feeders=max_feeders, balanced=True
        ),
    )
    solver.solve(time_limit=_RUNTIME, mip_gap=_GAP)
    S, _ = solver.get_solution()
    R = S.graph['R']
    return sorted(S.nodes[t]['load'] for r in range(-R, 0) for t in S.neighbors(r))


@pytest.mark.parametrize('solver_name', ['ortools.cp_sat', 'highs', 'scip'])
@pytest.mark.parametrize(
    ('max_feeders', 'expected_loads'),
    [
        (5, [2, 2, 2, 3, 3]),
        # this one is the regression guard: the balanced lower bound is 12 // 7 = 1,
        # so it constrains nothing and only the upper bound of 2 keeps the loads
        # together. A model carrying just the lower bound minimizes to [1,1,1,1,1,3,4].
        (7, [1, 1, 2, 2, 2, 2, 2]),
    ],
)
def test_balanced_pins_loads_to_floor_and_ceil(
    P_A_toy, solver_name, max_feeders, expected_loads, ortools_worker
):
    # toyfarm has T=12, so pinning the feeder count to a non-divisor makes the
    # loads span the two values {T // F, ceil(T / F)}.
    P, A = P_A_toy
    args = (solver_name, P, A, max_feeders)
    if solver_name.startswith('ortools'):
        result = ortools_worker.run(_solve_toy_balanced, args, 30 + _RUNTIME)
    else:
        try:
            result = _solve_toy_balanced(*args)
        except BaseException as exc:
            result = exc

    if isinstance(result, BaseException) and _solver_is_unavailable(result):
        pytest.skip(f'{solver_name} not available')
    if isinstance(result, BaseException):
        raise result

    assert result == expected_loads


def test_feeder_and_load_bounds_weighted_power():
    # weighted terminals: W=18 at capacity 5 needs 4 feeders (T=12 would need 3)
    assert core.feeder_and_load_bounds(
        12, 5, core.FeederLimit.MINIMUM, 0, False, total_power=18
    ) == (4, 4, None, None)
    # balanced bounds derive from W: floor(18/4) = 4; ceil(18/4) = 5 is already
    # implied by the flow variable's upper bound (the capacity), hence None
    assert core.feeder_and_load_bounds(
        12, 5, core.FeederLimit.MINIMUM, 0, True, total_power=18
    ) == (4, 4, 4, None)

    # EXACTLY is constrained by the number of terminals, not their total power.
    with pytest.raises(ValueError, match='above the number of terminals'):
        core.feeder_and_load_bounds(
            12, 5, core.FeederLimit.EXACTLY, 13, False, total_power=18
        )


def _solve_toy_weighted(solver_name, P, A):
    A = A.copy()
    A.graph['power_scale'] = 2
    for t in range(3):
        A.nodes[t]['power'] = 3
    solver = solver_factory(solver_name)
    solver.set_problem(P, A, capacity=_CAPACITY, model_options=ModelOptions())
    solver.solve(time_limit=_RUNTIME, mip_gap=_GAP)
    return solver.get_solution()


@pytest.mark.parametrize('solver_name', ['ortools.cp_sat', 'highs'])
def test_weighted_power_solve(P_A_toy, solver_name, ortools_worker):
    # toyfarm with terminals 0-2 at power 3: W = 3*3 + 9*1 = 18, capacity 5
    P, A = P_A_toy
    args = (solver_name, P, A)
    if solver_name.startswith('ortools'):
        result = ortools_worker.run(_solve_toy_weighted, args, 30 + _RUNTIME)
    else:
        try:
            result = _solve_toy_weighted(*args)
        except BaseException as exc:
            result = exc

    if isinstance(result, BaseException) and _solver_is_unavailable(result):
        pytest.skip(f'{solver_name} not available')
    if isinstance(result, BaseException):
        raise result

    S, G = result
    assert (S.graph.get('power_scale'), G.graph.get('power_scale')) == (2, 2)
    T = S.graph['T']
    expected_power = [3, 3, 3] + [1] * 9
    assert [S.nodes[t].get('power', 1) for t in range(T)] == expected_power
    assert [G.nodes[t].get('power', 1) for t in range(T)] == expected_power

    R = S.graph['R']
    subtree_loads = sorted(
        S.nodes[t]['load'] for r in range(-R, 0) for t in S.neighbors(r)
    )
    assert sum(subtree_loads) == 18
    assert all(load <= _CAPACITY for load in subtree_loads)
    # the model's feeder lower bound must count power, not terminals:
    # ceil(W / k) = 4, while ceil(T / k) would allow only 3
    assert len(subtree_loads) >= 4


def _job_solver_factory_name(solver_name):
    return solver_factory(solver_name).name


@pytest.mark.parametrize(
    ('solver_name', 'expected_name'),
    [
        ('ortools', 'ortools.cp_sat'),
        ('ortools.gscip', 'ortools.gscip'),
        ('ortools.highs', 'ortools.highs'),
    ],
)
def test_solver_factory_ortools_backends(solver_name, expected_name, ortools_worker):
    result = ortools_worker.run(_job_solver_factory_name, (solver_name,), 30)
    assert result == expected_name


@pytest.mark.parametrize(
    ('solver_name', 'find_spec_result', 'which_result', 'error', 'message'),
    [
        (
            'ortools.highs',
            None,
            'unused',
            ModuleNotFoundError,
            "Package 'ortools' not found",
        ),
        ('fscip', object(), None, FileNotFoundError, "Executable 'fscip' not found"),
    ],
)
def test_solver_factory_missing_dependencies(
    monkeypatch, solver_name, find_spec_result, which_result, error, message
):
    monkeypatch.setattr(MILP, 'find_spec', lambda name: find_spec_result)
    monkeypatch.setattr(MILP.shutil, 'which', lambda name: which_result)

    with pytest.raises(error, match=message):
        solver_factory(solver_name)


def test_solver_factory_rejects_unknown_solver():
    with pytest.raises(ValueError, match='Unsupported solver: unknown'):
        solver_factory('unknown')


def test_physical_core_count_linux_counts_unique_physical_cores(monkeypatch):
    _patch_cpu_topology(
        monkeypatch,
        {0, 1, 2},
        {
            '/sys/devices/system/cpu/cpu0/topology/physical_package_id': '0',
            '/sys/devices/system/cpu/cpu0/topology/core_id': '0',
            '/sys/devices/system/cpu/cpu1/topology/physical_package_id': '0',
            '/sys/devices/system/cpu/cpu1/topology/core_id': '0',
            '/sys/devices/system/cpu/cpu2/topology/physical_package_id': '0',
            '/sys/devices/system/cpu/cpu2/topology/core_id': '1',
        },
    )

    assert core.physical_core_count() == 2


def test_physical_core_count_falls_back_to_psutil(monkeypatch):
    import psutil

    _patch_cpu_topology(monkeypatch, {0})
    monkeypatch.setattr(psutil, 'cpu_count', lambda logical=False: 7)

    assert core.physical_core_count() == 7


def _job_make_solve_params(backend, physical_core_count_value, kwargs):
    from unittest import mock

    import optiwindnet.MILP.ortools as ortools_milp

    with mock.patch.object(
        ortools_milp, 'physical_core_count', lambda: physical_core_count_value
    ):
        solve_params = ortools_milp.SolverORTools(backend)._make_solve_parameters(
            **kwargs
        )
    return solve_params, kwargs['applied_options']


@pytest.mark.parametrize(
    ('backend', 'verbose', 'time_limit', 'mip_gap', 'expected_threads'),
    [
        ('cp_sat', True, 1.5, 0.01, None),
        ('highs', False, 2.0, 0.05, None),
    ],
)
def test_make_solve_parameters_thread_defaults(
    ortools_worker, backend, verbose, time_limit, mip_gap, expected_threads
):
    kwargs = dict(
        time_limit=time_limit,
        mip_gap=mip_gap,
        applied_options={},
        verbose=verbose,
    )
    solve_params, applied_options = ortools_worker.run(
        _job_make_solve_params, (backend, 6, kwargs), 30
    )

    assert solve_params.threads is expected_threads
    assert applied_options == {}
    if backend == 'cp_sat':
        assert solve_params.cp_sat.log_search_progress is True
    else:
        assert solve_params.highs.int_options['threads'] == 6


def _job_gscip_native_params(applied_options):
    import optiwindnet.MILP.ortools as ortools_milp

    solver = ortools_milp.SolverORTools('gscip')
    return solver._make_solve_parameters(
        time_limit=3.0,
        mip_gap=0.001,
        applied_options=applied_options,
        verbose=False,
    )


def test_make_solve_parameters_gscip_routes_native_parameter_types(ortools_worker):
    applied_options = {
        'limits/nodes': 12,
        'display/verblevel': 4,
        'parallel/mode': True,
        'numerics/epsilon': 0.25,
        'misc/branchdir': 'u',
        'visual/vbcfilename': 'trace.vbc',
    }

    solve_params = ortools_worker.run(_job_gscip_native_params, (applied_options,), 30)

    assert solve_params.gscip.int_params['limits/nodes'] == 12
    assert solve_params.gscip.int_params['display/verblevel'] == 4
    assert solve_params.gscip.bool_params['parallel/mode'] is True
    assert solve_params.gscip.real_params['numerics/epsilon'] == 0.25
    assert solve_params.gscip.char_params['misc/branchdir'] == 'u'
    assert solve_params.gscip.string_params['visual/vbcfilename'] == 'trace.vbc'


def _job_gscip_rejects_unsupported():
    import optiwindnet.MILP.ortools as ortools_milp

    solver = ortools_milp.SolverORTools('gscip')
    solver._make_solve_parameters(
        time_limit=3.0,
        mip_gap=0.001,
        applied_options={'randomization/permutationseed': [1, 2, 3]},
        verbose=False,
    )


def test_make_solve_parameters_gscip_rejects_unsupported_native_param_type(
    ortools_worker,
):
    result = ortools_worker.run(_job_gscip_rejects_unsupported, (), 30)
    assert isinstance(result, TypeError)
    assert 'Unsupported type list' in str(result)


def _bounds(T, capacity, feeder_limit, max_feeders=0, balanced=True):
    return core.feeder_and_load_bounds(
        T, capacity, core.FeederLimit(feeder_limit), max_feeders, balanced
    )


@pytest.mark.parametrize('capacity', range(2, 20))
def test_feeder_and_load_bounds_balanced_at_minimum(capacity):
    for T in range(1, 200):
        feeders_lb, feeders_ub, load_lb, load_ub = _bounds(T, capacity, 'minimum')
        F = math.ceil(T / capacity)
        # the feeder count is pinned to the minimum
        assert feeders_lb == feeders_ub == F
        # a None bound means the flow variable's own bounds already imply it
        lo = load_lb if load_lb is not None else 1
        hi = load_ub if load_ub is not None else capacity
        # every terminal fits and no subtree exceeds the requested capacity
        assert F * lo <= T <= F * hi
        assert hi <= capacity
        # subtree loads differ at most by one unit
        assert (T // F) == lo or lo == 1
        assert math.ceil(T / F) == hi or hi == capacity


def test_feeder_and_load_bounds_omits_implied_bounds():
    # T = 12 over 3 feeders: loads are exactly 4, and 4 < capacity=5
    assert _bounds(12, 5, 'minimum') == (3, 3, 4, 4)
    # a single feeder carrying everything: only the lower bound says anything
    assert _bounds(12, 12, 'minimum') == (1, 1, 12, None)
    # capacity 1 forces one terminal per feeder: both bounds are already implied
    assert _bounds(3, 1, 'minimum') == (3, 3, None, None)


def test_feeder_and_load_bounds_exactly_pins_the_given_count():
    # 12 terminals over 5 feeders -> loads in {2, 3}, well below capacity 5
    assert _bounds(12, 5, 'exactly', max_feeders=5) == (5, 5, 2, 3)
    assert _bounds(12, 5, 'exactly', max_feeders=5, balanced=False) == (
        5,
        5,
        None,
        None,
    )


def test_feeder_and_load_bounds_exactly_rejects_infeasible_counts():
    with pytest.raises(ValueError, match='below the minimum'):
        _bounds(12, 5, 'exactly', max_feeders=2)
    with pytest.raises(ValueError, match='above the number of terminals'):
        _bounds(12, 5, 'exactly', max_feeders=13)


def test_feeder_and_load_bounds_specified_is_an_upper_bound():
    assert _bounds(12, 5, 'specified', max_feeders=7, balanced=False) == (
        3,
        7,
        None,
        None,
    )
    # ... unless it coincides with the minimum, which pins the count
    assert _bounds(12, 5, 'specified', max_feeders=3) == (3, 3, 4, 4)
    with pytest.raises(ValueError, match='below the minimum'):
        _bounds(12, 5, 'specified', max_feeders=2)


@pytest.mark.parametrize(
    ('feeder_limit', 'feeders_ub'),
    [
        ('unlimited', None),
        ('min_plus1', 4),
        ('min_plus2', 5),
        ('min_plus3', 6),
    ],
)
def test_feeder_and_load_bounds_warns_when_balance_is_unenforceable(
    caplog, feeder_limit, feeders_ub
):
    with caplog.at_level(logging.WARNING, logger=core.__name__):
        assert _bounds(12, 5, feeder_limit) == (3, feeders_ub, None, None)
    assert 'will not enforce balanced subtrees' in caplog.text
