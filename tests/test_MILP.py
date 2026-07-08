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


def _raise_or_skip_unavailable(result, solver_name):
    if isinstance(result, BaseException) and _solver_is_unavailable(result):
        pytest.skip(f'{solver_name} not available')
    if isinstance(result, BaseException):
        raise result
    return result


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


@pytest.mark.parametrize(
    ('total_power', 'capacity', 'expected'),
    [
        (30, 5, 6),
        (33, 5, 7),
        (3015, 500, 7),
        (1450, 100, 15),
    ],
)
def test_minimum_feeder_count_uses_total_power(total_power, capacity, expected):
    assert core.minimum_feeder_count(total_power, capacity) == expected


def test_balanced_feeder_min_load_uses_total_power():
    assert core.balanced_feeder_min_load(total_power=33, feeder_count=7) == 4


def _job_ortools_flow_nonzero_coefficients(A_toy, powers, nonuniform_power=False):
    import optiwindnet.MILP.ortools as ortools_milp

    A = A_toy.copy()
    if nonuniform_power:
        A.graph[core.NONUNIFORM_POWER_ATTR] = True
    for t, power in powers.items():
        A.nodes[t]['power'] = power

    model, _ = ortools_milp.make_min_length_model(A, _CAPACITY)
    constraint = {
        constraint.name: constraint for constraint in model.linear_constraints()
    }['flow_nonzero_0~1']
    return {
        entry.variable.name: entry.coefficient
        for entry in model.linear_constraint_matrix_entries()
        if entry.linear_constraint == constraint
    }


def _job_ortools_make_weighted_model_without_mode(A_toy):
    import optiwindnet.MILP.ortools as ortools_milp

    A = A_toy.copy()
    A.nodes[0]['power'] = 3

    ortools_milp.make_min_length_model(A, _CAPACITY)


def test_ortools_flow_lower_bound_uses_source_power(P_A_toy, ortools_worker):
    _, A_toy = P_A_toy
    result = ortools_worker.run(
        _job_ortools_flow_nonzero_coefficients,
        (A_toy, {0: 3}, True),
        30,
    )
    coeff_by_var_name = _raise_or_skip_unavailable(result, 'ortools')

    assert coeff_by_var_name['flow_0~1'] == 1
    assert coeff_by_var_name['link_0~1'] == -3


def test_pyomo_flow_lower_bound_uses_source_power(P_A_toy):
    import pyomo.environ as pyo

    import optiwindnet.MILP.pyomo as pyomo_milp

    _, A_toy = P_A_toy
    A = A_toy.copy()
    A.graph[core.NONUNIFORM_POWER_ATTR] = True
    A.nodes[0]['power'] = 3

    model, _ = pyomo_milp.make_min_length_model(A, _CAPACITY)
    constraint = model.cons_flow_lb[0, 1]
    model.link_[0, 1].value = 1
    model.flow_[0, 1].value = 2
    assert pyo.value(constraint.body) > pyo.value(constraint.upper)

    model.flow_[0, 1].value = 3
    assert pyo.value(constraint.body) == pyo.value(constraint.upper)


def test_milp_rejects_nonuniform_power_without_mode(P_A_toy, ortools_worker):
    _, A_toy = P_A_toy
    result = ortools_worker.run(
        _job_ortools_make_weighted_model_without_mode,
        (A_toy,),
        30,
    )

    if isinstance(result, BaseException) and _solver_is_unavailable(result):
        pytest.skip('ortools not available')
    assert isinstance(result, ValueError)
    assert 'weighted power mode is not active' in str(result)


def test_milp_ignores_uniform_power_without_mode(P_A_toy, ortools_worker):
    _, A_toy = P_A_toy
    result = ortools_worker.run(
        _job_ortools_flow_nonzero_coefficients,
        (A_toy, {t: 3 for t in range(A_toy.graph['T'])}, False),
        30,
    )
    coeff_by_var_name = _raise_or_skip_unavailable(result, 'ortools')

    assert coeff_by_var_name['link_0~1'] == -1


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
