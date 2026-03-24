import pytest
import numpy as np
import multiprocessing

import optiwindnet.MILP as MILP
import optiwindnet.MILP._core as core
import optiwindnet.MILP.ortools as ortools_milp
from optiwindnet.synthetic import toyfarm
from optiwindnet.mesh import make_planar_embedding
from optiwindnet.MILP import solver_factory, ModelOptions
from optiwindnet.interarraylib import terse_links_from_S


# topology in terse links for toy_farm at capacity=5
_terse_toy_farm_5 = np.array([2, -1, 1, 2, -1, -1, 3, 4, -1, 5, 8, 8])
_CAPACITY = 5
_RUNTIME = 10
_GAP = 0.001


# Loading solvers OR-Tools and scip within the same python instance causes DLL hell
# (ortools contains a SCIP library). Typical usage of OWN is with a single solver.
# Use a workaround for tests: spawn a new python process for each solver.
def _worker_MILP_solver(P, A, solver_name, queue) -> None | tuple:
    try:
        solver = solver_factory(solver_name)
        solver.set_problem(P, A, capacity=_CAPACITY, model_options=ModelOptions())
        solution_info = solver.solve(time_limit=_RUNTIME, mip_gap=_GAP)
    except BaseException as exc:
        queue.put(exc)
        return
    queue.put((solution_info, solver.get_solution()))


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


def _make_solve_params(backend, **kwargs):
    return ortools_milp.SolverORTools(backend)._make_solve_parameters(**kwargs)


@pytest.fixture(scope='module')
def P_A_toy():
    L = toyfarm()
    P, A = make_planar_embedding(L)
    return P, A


@pytest.mark.parametrize(
    'solver_name',
    ['ortools.cp_sat', 'gurobi', 'cplex', 'highs', 'scip', 'cbc', 'fscip'],
)
def test_MILP_solvers(P_A_toy, solver_name):
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue(maxsize=1)
    p = ctx.Process(target=_worker_MILP_solver, args=(*P_A_toy, solver_name, queue))
    p.start()
    p.join()
    result = queue.get(timeout=10 + _RUNTIME)

    if isinstance(result, BaseException) and _solver_is_unavailable(result):
        pytest.skip(f'{solver_name} not available')
    if isinstance(result, BaseException):
        raise result

    solution_info, (S, _) = result
    assert solution_info.termination.lower() == 'optimal'
    assert (terse_links_from_S(S) == _terse_toy_farm_5).all()


@pytest.mark.parametrize(
    ('solver_name', 'expected_name'),
    [
        ('ortools', 'ortools.cp_sat'),
        ('ortools.gscip', 'ortools.gscip'),
        ('ortools.highs', 'ortools.highs'),
    ],
)
def test_solver_factory_ortools_backends(solver_name, expected_name):
    solver = solver_factory(solver_name)
    assert solver.name == expected_name


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


@pytest.mark.parametrize(
    ('backend', 'verbose', 'time_limit', 'mip_gap', 'expected_threads'),
    [
        ('cp_sat', True, 1.5, 0.01, None),
        ('highs', False, 2.0, 0.05, None),
    ],
)
def test_make_solve_parameters_thread_defaults(
    monkeypatch, backend, verbose, time_limit, mip_gap, expected_threads
):
    monkeypatch.setattr(ortools_milp, 'physical_core_count', lambda: 6)
    applied_options = {}

    solve_params = _make_solve_params(
        backend,
        time_limit=time_limit,
        mip_gap=mip_gap,
        applied_options=applied_options,
        verbose=verbose,
    )

    assert solve_params.threads is expected_threads
    assert applied_options == {}
    if backend == 'cp_sat':
        assert solve_params.cp_sat.log_search_progress is True
    else:
        assert solve_params.highs.int_options['threads'] == 6


def test_make_solve_parameters_gscip_routes_native_parameter_types():
    solver = ortools_milp.SolverORTools('gscip')
    applied_options = {
        'limits/nodes': 12,
        'display/verblevel': 4,
        'parallel/mode': True,
        'numerics/epsilon': 0.25,
        'misc/branchdir': 'u',
        'visual/vbcfilename': 'trace.vbc',
    }

    solve_params = solver._make_solve_parameters(
        time_limit=3.0,
        mip_gap=0.001,
        applied_options=applied_options,
        verbose=False,
    )

    assert solve_params.gscip.int_params['limits/nodes'] == 12
    assert solve_params.gscip.int_params['display/verblevel'] == 4
    assert solve_params.gscip.bool_params['parallel/mode'] is True
    assert solve_params.gscip.real_params['numerics/epsilon'] == 0.25
    assert solve_params.gscip.char_params['misc/branchdir'] == 'u'
    assert solve_params.gscip.string_params['visual/vbcfilename'] == 'trace.vbc'


def test_make_solve_parameters_gscip_rejects_unsupported_native_param_type():
    solver = ortools_milp.SolverORTools('gscip')

    with pytest.raises(TypeError, match='Unsupported type list'):
        solver._make_solve_parameters(
            time_limit=3.0,
            mip_gap=0.001,
            applied_options={'randomization/permutationseed': [1, 2, 3]},
            verbose=False,
        )
