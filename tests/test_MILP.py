import importlib
import logging
import math
import pickle

import networkx as nx
import pytest

import optiwindnet.MILP as MILP
import optiwindnet.MILP._core as core
from optiwindnet.interarraylib import terse_links_from_S
from optiwindnet.MILP import ModelOptions, solver_factory
from optiwindnet.terse import TerseLinks
from optiwindnet.types import Topology

from .helpers import solver_unavailable
from .sitecache import get_bundle
from .cases import (
    MILP_ADAPTER_CASES,
    MILP_BOUNDARY_CASES,
    MILP_FAMILY_CASES,
    MILP_FORMULATION_CASES,
    case_node_id,
    topology_golden_key,
)
from .solver_topologies import (
    assert_matches_golden,
    load_solver_topologies,
    solve_milp_case,
)
from .topology_assertions import assert_topology

# topology in terse links for toy_farm at capacity=5
_CAPACITY = 5
_RUNTIME = 10
_GAP = 0.001
_SOLVER_GOLDENS = load_solver_topologies()


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
def _solve_toy_incumbent(solver_name, topology):
    from unittest import mock

    import optiwindnet.MILP.ortools as ortools_milp
    from optiwindnet.interarraylib import validate_topology

    bundle = get_bundle('toy')
    P, A = bundle.P, bundle.A
    solver = solver_factory(solver_name)
    solver.set_problem(
        P,
        A,
        capacity=_CAPACITY,
        model_options=ModelOptions(topology=topology),
    )
    solution_info = solver.solve(time_limit=_RUNTIME, mip_gap=_GAP)
    with mock.patch.object(
        ortools_milp.PathFinder,
        'create_detours',
        side_effect=AssertionError('incumbent retrieval must not route'),
    ):
        S = solver.get_incumbent_topology()
    return dict(
        terse=terse_links_from_S(S),
        violations=validate_topology(S, _CAPACITY),
        graph=dict(S.graph),
        objective=solution_info.objective,
        preserved_objective=solver.solution_info.objective,
    )


def _get_incumbent_before_solve(solver_name):
    bundle = get_bundle('toy')
    P, A = bundle.P, bundle.A
    solver = solver_factory(solver_name)
    solver.set_problem(P, A, capacity=_CAPACITY, model_options=ModelOptions())
    return solver.get_incumbent_topology()


def _exercise_ortools_retrieval_branch(feeder_route):
    """Exercise get_solution() branches with routing and pool work faked out."""
    from unittest import mock

    import optiwindnet.MILP.ortools as ortools_milp

    bundle = get_bundle('toy')
    P, A = bundle.P, bundle.A
    solver = ortools_milp.SolverORTools('cp_sat')
    solver.P, solver.A = P, A
    solver.model_options = ModelOptions(feeder_route=feeder_route)
    calls = []
    S = nx.Graph(source='best-objective')
    G = nx.Graph()

    def incumbent():
        calls.append('incumbent')
        return S

    def investigate(P, A):
        assert P is solver.P
        assert A is solver.A
        calls.append('investigate')
        return S, G

    solver._incumbent_topology_from_pool = incumbent
    solver._investigate_pool = investigate
    solver._make_graph_attributes = lambda: {'solver_details': {}}
    with (
        mock.patch.object(ortools_milp, 'G_from_S', return_value=G),
        mock.patch.object(ortools_milp, 'PathFinder') as pathfinder,
    ):
        pathfinder.return_value.create_detours.return_value = G
        selected, _ = solver.get_solution()
    return calls, selected.graph['source'], pathfinder.call_count


class _FakePool(core.PoolHandler):
    def __init__(self, candidates: list[tuple[float, str]]):
        self.candidates = sorted(candidates, key=lambda candidate: candidate[0])
        self.solution_info = core.SolutionInfo(0.0, 0.0, 1.0, 0.0, 'optimal')
        self.selected = 0
        self.objectives_requested: list[int] = []
        self.decoded: list[str] = []
        self.investigations = 0

    def _objective_at(self, index: int) -> float:
        self.objectives_requested.append(index)
        self.selected = index
        return self.candidates[index][0]

    def _topology_from_mip_pool(self) -> nx.Graph:
        label = self.candidates[self.selected][1]
        self.decoded.append(label)
        return nx.Graph(candidate=label)

    def _investigate_pool(
        self, P: nx.PlanarEmbedding, A: nx.Graph
    ) -> tuple[nx.Graph, nx.Graph]:
        self.investigations += 1
        raise AssertionError('incumbent retrieval must not investigate the pool')


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
    bundle = get_bundle('toy')
    return bundle.P, bundle.A


def test_pool_incumbent_helper_selects_and_decodes_only_best_objective():
    pool = _FakePool([(7.0, 'later'), (1.0, 'best'), (4.0, 'middle')])

    S = pool._incumbent_topology_from_pool()

    assert S.graph['candidate'] == 'best'
    assert pool.objectives_requested == [0]
    assert pool.decoded == ['best']
    assert pool.investigations == 0


def test_solver_graph_attributes_preserve_warmstart_and_feeder_limit():
    from types import SimpleNamespace

    fake = SimpleNamespace(
        name='fake',
        metadata=SimpleNamespace(
            fun_fingerprint={'funhash': b'x'},
            model_options=ModelOptions(feeder_limit='exactly', max_feeders=3),
            warmed_by='constructor',
        ),
        solution_info=core.SolutionInfo(1.0, 1.0, 0.0, 0.1, 'optimal'),
        applied_options={'threads': 1},
        stopping={'time_limit': 1, 'mip_gap': 0.01},
    )

    attributes = core.Solver._make_graph_attributes(
        fake  # pyrefly: ignore[bad-argument-type]
    )

    assert attributes['warmstart'] == 'constructor'
    assert attributes['solver_details']['max_feeders'] == 3
    assert 'max_feeders' not in attributes['method_options']

    fake.metadata.warmed_by = None
    assert 'warmstart' not in core.Solver._make_graph_attributes(
        fake  # pyrefly: ignore[bad-argument-type]
    )


def test_pool_investigation_ranks_routed_candidates(monkeypatch, P_A_toy):
    class Pool(core.PoolHandler):
        num_solutions = 3

        def __init__(self):
            self.index = 0

        def _objective_at(self, index):
            self.index = index
            return (1.0, 2.0, 10.0)[index]

        def _topology_from_mip_pool(self):
            return nx.Graph(candidate=self.index)

    pool = Pool()
    lengths = (5.0, 6.0, 1.0)
    monkeypatch.setattr(core, 'G_from_S', lambda S, A: S)

    class FakePathFinder:
        def __init__(self, S, **kwargs):
            self.S = S

        def create_detours(self):
            G = nx.Graph()
            G.add_edge(0, 1, length=lengths[self.S.graph['candidate']])
            return G

    monkeypatch.setattr(core, 'PathFinder', FakePathFinder)
    P, A = P_A_toy

    S, G = pool._investigate_pool(P, A)

    assert S.graph['candidate'] == 0
    assert G.graph['pool_entry'] == (0, 1.0)
    assert G.graph['pool_count'] == 3


def test_ortools_incumbent_matches_toy_topology_without_routing(ortools_worker):
    result = ortools_worker.run(
        _solve_toy_incumbent,
        ('ortools.cp_sat', 'branched'),
        30 + _RUNTIME,
    )
    if isinstance(result, BaseException):
        raise result

    case = next(case for case in MILP_FORMULATION_CASES if case.exact_golden)
    golden = _SOLVER_GOLDENS[topology_golden_key(case)]
    assert isinstance(golden, TerseLinks)
    assert tuple(result['terse']) == golden.links
    assert result['violations'] == []
    assert result['objective'] == result['preserved_objective']
    assert {
        'R',
        'T',
        'topology',
        'capacity',
        'max_load',
        'has_loads',
        'creator',
    } <= result['graph'].keys()


@pytest.mark.parametrize('topology', ['radial', 'ringed'])
def test_ortools_incumbent_decodes_valid_topology(ortools_worker, topology):
    result = ortools_worker.run(
        _solve_toy_incumbent,
        ('ortools.cp_sat', topology),
        30 + _RUNTIME,
    )
    if isinstance(result, BaseException):
        raise result

    assert result['violations'] == []
    assert result['graph']['topology'] == topology


def test_ortools_incumbent_before_solve_has_useful_error(ortools_worker):
    result = ortools_worker.run(_get_incumbent_before_solve, ('ortools.cp_sat',), 30)

    assert isinstance(result, AttributeError)
    assert '.solve() must be called before solution retrieval' in str(result)


@pytest.mark.parametrize('bridging', (False, True), ids=('one-root', 'bridging'))
@pytest.mark.parametrize('n', range(2, 11))
def test_ringed_warmstart_links_conserve_flow(n, bridging):
    from types import SimpleNamespace

    from optiwindnet.interarraylib import add_ring_to_S
    from optiwindnet.MILP._core import warmstart_links

    terminals = list(range(n))
    R = 2 if bridging else 1
    roots = list(range(-R, 0))
    E = [(u, v) for u in terminals for v in terminals if u < v]
    Ep = [(v, u) for u, v in E]
    stars = [(t, r) for t in terminals for r in roots]
    starsp = [(r, t) for t in terminals for r in roots]
    link_ = {key: key for key in E + Ep + stars + starsp}
    flow_ = {key: key for key in E + Ep + stars}
    metadata = SimpleNamespace(
        R=R,
        link_=link_,
        flow_=flow_,
        model_options={'topology': Topology.RINGED},
    )
    S = nx.Graph(R=R, T=n)
    S.add_nodes_from(roots)
    add_ring_to_S(
        S,
        (-1, -2) if bridging else (-1, -1),
        terminals,
        subtree=0,
        A=None,
    )

    link_values = dict.fromkeys(link_, 0)
    flow_values = dict.fromkeys(flow_, 0)
    for link_var, flow_var, flow in warmstart_links(
        metadata,  # pyrefly: ignore[bad-argument-type]
        S,
    ):
        link_values[link_var] = 1
        if flow_var is not None:
            flow_values[flow_var] = flow

    active_feeders = [key for key in stars if link_values[key]]
    assert len(active_feeders) == 1
    assert flow_values[active_feeders[0]] == n
    assert sum(link_values[key] for key in starsp) == 1
    for terminal in terminals:
        outgoing = [
            key for key, active in link_values.items() if active and key[0] == terminal
        ]
        incoming = [
            key for key, active in link_values.items() if active and key[1] == terminal
        ]
        assert len(outgoing) == len(incoming) == 1
        assert (
            sum(flow_values.get(key, 0) for key in outgoing)
            - sum(flow_values.get(key, 0) for key in incoming)
            == 1
        )


@pytest.mark.parametrize('bridging', (False, True), ids=('one-root', 'bridging'))
@pytest.mark.parametrize('n', range(1, 11))
def test_ringed_mip_decoder_reuses_flow_tree(n, bridging):
    from types import SimpleNamespace

    head_root = -1
    close_root = -2 if bridging else head_root
    R = 2 if bridging else 1
    flows = {(0, head_root): n}
    flows.update({(terminal, terminal - 1): n - terminal for terminal in range(1, n)})
    closing_link = (close_root, n - 1)
    links = dict.fromkeys((*flows, closing_link), True)
    metadata = SimpleNamespace(
        R=R,
        T=n,
        capacity=math.ceil(n / 2),
        model_options={'topology': Topology.RINGED},
        link_=links,
        flow_=flows,
    )
    A = nx.path_graph(n)
    nx.set_edge_attributes(A, {edge: 1.0 for edge in A.edges}, 'length')

    class FakeSolver:
        name = 'fake'
        A: nx.Graph
        metadata: object

        @staticmethod
        def _link_val(value):
            return value

        @staticmethod
        def _flow_val(value):
            return value

    fake = FakeSolver()
    fake.A = A
    fake.metadata = metadata
    S = core.Solver._topology_from_mip_sol(
        fake  # pyrefly: ignore[bad-argument-type]
    )

    assert_topology(S, Topology.RINGED, metadata.capacity)


def test_ringed_warmstart_is_accepted_by_scip():
    bundle = get_bundle('albatros')
    P, A = bundle.P, bundle.A
    options = ModelOptions(topology='ringed')
    try:
        seed = solver_factory('scip')
        seed.set_problem(P, A, capacity=3, model_options=options)
        seed.solve(time_limit=10, mip_gap=0.05)
        S = seed.get_incumbent_topology()
    except BaseException as exc:
        if solver_unavailable(exc):
            pytest.skip(f'scip unavailable: {exc}')
        raise

    solver = solver_factory('scip')
    solver.set_problem(P, A, capacity=3, model_options=options, warmstart=S)
    assert solver.metadata.warmed_by == S.graph['creator']


def test_segmented_get_solution_still_investigates_pool(ortools_worker):
    calls, source, pathfinder_calls = ortools_worker.run(
        _exercise_ortools_retrieval_branch, ('segmented',), 30
    )

    assert calls == ['investigate']
    assert source == 'best-objective'
    assert pathfinder_calls == 0


def test_straight_get_solution_starts_from_best_incumbent(ortools_worker):
    calls, source, pathfinder_calls = ortools_worker.run(
        _exercise_ortools_retrieval_branch, ('straight',), 30
    )

    assert calls == ['incumbent']
    assert source == 'best-objective'
    assert pathfinder_calls == 1


@pytest.mark.parametrize(
    ('module_name', 'class_name'),
    (
        ('optiwindnet.MILP.pyomo', 'SolverPyomo'),
        ('optiwindnet.MILP.pyomo', 'SolverPyomoAppsi'),
    ),
)
def test_pyomo_solution_retrieval_glue(monkeypatch, module_name, class_name, P_A_toy):
    module = importlib.import_module(module_name)
    solver = object.__new__(getattr(module, class_name))
    P, A = P_A_toy
    S, G_tentative, G = nx.Graph(), nx.Graph(), nx.Graph()
    solver.P, solver.A = P, A
    solver._load_incumbent_topology = lambda: S
    solver._make_graph_attributes = lambda: {'retrieved': True}
    monkeypatch.setattr(module, 'G_from_S', lambda actual, available: G_tentative)

    class FakePathFinder:
        def __init__(self, tentative, planar, available):
            assert (tentative, planar, available) == (G_tentative, P, A)

        def create_detours(self):
            return G

    monkeypatch.setattr(module, 'PathFinder', FakePathFinder)

    assert solver.get_solution() == (S, G)
    assert G.graph['retrieved'] is True


@pytest.mark.parametrize(
    ('module_name', 'class_name', 'prepare'),
    (
        ('optiwindnet.MILP.scip', 'SolverSCIP', False),
        ('optiwindnet.MILP.cplex', 'SolverCplex', True),
        ('optiwindnet.MILP.fscip', 'SolverFSCIP', False),
    ),
)
@pytest.mark.parametrize('feeder_route', ('segmented', 'straight'))
def test_pool_backend_solution_retrieval_branches(
    monkeypatch, module_name, class_name, prepare, feeder_route, P_A_toy
):
    module = importlib.import_module(module_name)
    solver = object.__new__(getattr(module, class_name))
    P, A = P_A_toy
    S, G_tentative, G = nx.Graph(), nx.Graph(), nx.Graph()
    calls = []
    solver.P, solver.A = P, A
    solver.model_options = ModelOptions(feeder_route=feeder_route)
    solver._incumbent_topology_from_pool = lambda: calls.append('incumbent') or S
    solver._investigate_pool = lambda planar, available: (
        calls.append('investigate') or (S, G)
    )
    solver._make_graph_attributes = lambda: {'retrieved': True}
    if prepare:
        solver._prepare_solution_pool = lambda: calls.append('prepare')
    monkeypatch.setattr(module, 'G_from_S', lambda actual, available: G_tentative)

    class FakePathFinder:
        def __init__(self, tentative, planar, available):
            assert (tentative, planar, available) == (G_tentative, P, A)

        def create_detours(self):
            return G

    monkeypatch.setattr(module, 'PathFinder', FakePathFinder)

    assert solver.get_solution() == (S, G)
    expected = ['prepare'] if prepare else []
    expected.append('incumbent' if feeder_route == 'straight' else 'investigate')
    assert calls == expected
    assert G.graph['retrieved'] is True


@pytest.mark.parametrize('case', MILP_ADAPTER_CASES, ids=case_node_id)
def test_milp_adapter_topology_golden(case, ortools_worker):
    if case.solver_name.startswith('ortools'):
        result = ortools_worker.run(solve_milp_case, (case,), 30 + case.time_limit)
    else:
        try:
            result = solve_milp_case(case)
        except BaseException as exc:
            result = exc

    if isinstance(result, BaseException) and solver_unavailable(result):
        pytest.skip(f'{case.solver_name} not available')
    if isinstance(result, BaseException):
        raise result

    solution_info, S = result
    assert solution_info.termination.lower() == 'optimal'
    assert_topology(S, case.model_options['topology'], case.capacity)
    assert_matches_golden(S, _SOLVER_GOLDENS[topology_golden_key(case)])


@pytest.mark.parametrize('case', MILP_FORMULATION_CASES, ids=case_node_id)
def test_milp_required_formulation_topologies(case, ortools_worker):
    result = ortools_worker.run(solve_milp_case, (case,), 30 + case.time_limit)
    if isinstance(result, BaseException):
        raise result
    info, S = result
    assert info.termination in ('OPTIMAL', 'FEASIBLE')
    assert_topology(S, case.model_options['topology'], case.capacity)
    if case.exact_golden:
        assert_matches_golden(S, _SOLVER_GOLDENS[topology_golden_key(case)])


@pytest.mark.parametrize('case', MILP_FAMILY_CASES, ids=case_node_id)
def test_milp_distinct_formulation_families(case):
    try:
        info, S = solve_milp_case(case)
    except BaseException as exc:
        if solver_unavailable(exc):
            pytest.skip(f'{case.solver_name} unavailable: {exc}')
        raise
    assert info.termination.lower() in ('optimal', 'feasible', 'gaplimit')
    assert_topology(S, case.model_options['topology'], case.capacity)


@pytest.mark.parametrize('case', MILP_BOUNDARY_CASES, ids=case_node_id)
def test_milp_required_boundary_solves(case, ortools_worker):
    result = ortools_worker.run(solve_milp_case, (case,), 30 + case.time_limit)
    if isinstance(result, BaseException):
        raise result
    info, S = result
    assert info.bound <= info.objective * (1 + 1e-6)
    assert_topology(S, case.model_options['topology'], case.capacity)


def _solve_toy_balanced(solver_name, max_feeders):
    """Solve ``toy`` with the feeder count pinned and balanced."""
    bundle = get_bundle('toy')
    P, A = bundle.P, bundle.A
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
    S = solver.get_incumbent_topology()
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
    solver_name, max_feeders, expected_loads, ortools_worker
):
    # ``toy`` has T=12, so pinning the feeder count to a non-divisor makes the
    # loads span the two values {T // F, ceil(T / F)}.
    args = (solver_name, max_feeders)
    if solver_name.startswith('ortools'):
        result = ortools_worker.run(_solve_toy_balanced, args, 30 + _RUNTIME)
    else:
        try:
            result = _solve_toy_balanced(*args)
        except BaseException as exc:
            result = exc

    if isinstance(result, BaseException) and solver_unavailable(result):
        pytest.skip(f'{solver_name} not available')
    if isinstance(result, BaseException):
        raise result

    assert result == expected_loads


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


# --------------------------------------------------------------------------- #
# ModelOptions is the contract: coerced on the way in, frozen thereafter
# --------------------------------------------------------------------------- #
def test_model_options_coerces_strings_to_enums():
    opts = ModelOptions(topology='ringed', feeder_limit='minimum')
    assert opts['topology'] is core.Topology.RINGED
    assert opts['feeder_limit'] is core.FeederLimit.MINIMUM


def test_model_options_materializes_every_default():
    assert set(ModelOptions()) == {
        'topology',
        'feeder_route',
        'feeder_limit',
        'balanced',
        'max_feeders',
    }


@pytest.mark.parametrize(
    ('kwargs', 'expected'),
    [
        (dict(max_feeders='3'), 'max_feeders must be int'),
        (dict(balanced='yes'), 'balanced must be bool'),
    ],
)
def test_model_options_rejects_wrong_scalar_type(kwargs, expected):
    """A str for an int option is a type error, not an enum lookup."""
    with pytest.raises(TypeError, match=expected):
        ModelOptions(**kwargs)


@pytest.mark.parametrize(
    'mutate',
    [
        lambda o: o.__setitem__('topology', 'ringed'),
        lambda o: o.__delitem__('topology'),
        lambda o: o.update(topology='ringed'),
        lambda o: o.pop('topology'),
        lambda o: o.popitem(),
        lambda o: o.setdefault('topology', 'ringed'),
        lambda o: o.clear(),
    ],
)
def test_model_options_is_immutable(mutate):
    """Mutation would bypass __init__'s coercion, so it is refused."""
    opts = ModelOptions()
    with pytest.raises(TypeError, match='immutable'):
        mutate(opts)
    assert opts['topology'] is core.Topology.BRANCHED


def test_model_options_pickle_roundtrip_preserves_values_and_immutability():
    options = ModelOptions(
        topology='ringed',
        feeder_route='straight',
        feeder_limit='exactly',
        balanced=True,
        max_feeders=3,
    )

    restored = pickle.loads(pickle.dumps(options))

    assert type(restored) is ModelOptions
    assert restored == options
    assert restored['topology'] is Topology.RINGED
    with pytest.raises(TypeError, match='immutable'):
        restored['balanced'] = False


def test_set_problem_coerces_a_plain_mapping(P_A_toy):
    """A plain dict builds the same model as the ModelOptions equivalent.

    Uncoerced, ``{'topology': 'ringed'}`` compares false against
    ``Topology.RINGED`` and silently yields a *branched* model -- one with no
    root-to-terminal ring variables at all.
    """
    P, A = P_A_toy

    def ring_var_count(model_options):
        solver = solver_factory('highs')
        solver.set_problem(P, A, capacity=_CAPACITY, model_options=model_options)
        assert isinstance(solver.model_options, ModelOptions)
        return sum(1 for u, _ in solver.metadata.linkset if u < 0)

    from_mapping = ring_var_count({'topology': 'ringed'})
    from_options = ring_var_count(ModelOptions(topology='ringed'))
    assert from_mapping == from_options > 0


def test_model_options_help(capsys):
    ModelOptions.help()
    captured = capsys.readouterr()
    assert 'topology' in captured.out
    assert 'feeder_route' in captured.out


def test_calculate_bounds_invalid_max_feeders_ringed():
    from optiwindnet.MILP._core import feeder_and_load_bounds, FeederLimit

    with pytest.raises(ValueError, match='multiple of'):
        feeder_and_load_bounds(
            T=10,
            capacity=5,
            feeder_limit=FeederLimit.SPECIFIED,
            balanced=False,
            max_feeders=3,
            feeders_per_subtree=2,
        )


def test_pool_handler_methods():
    class DummyPoolHandler(core.PoolHandler):
        def _objective_at(self, index: int) -> float:
            return 999.0

        def _topology_from_mip_pool(self) -> nx.Graph:
            return nx.Graph()

    handler = DummyPoolHandler()
    with pytest.raises(AttributeError, match='must be called before'):
        handler._incumbent_topology_from_pool()

    handler.solution_info = core.SolutionInfo(
        runtime=0.1, bound=10.0, objective=10.0, relgap=0.0, termination='optimal'
    )
    with pytest.raises(ValueError, match='Best solution-pool objective'):
        handler._incumbent_topology_from_pool()


def test_solver_gurobi_error_branches():
    from optiwindnet.MILP.gurobi import SolverGurobi, OWNSolutionNotFound

    solver = SolverGurobi()
    with pytest.raises(AttributeError, match="has no attribute 'model'"):
        solver.solve(time_limit=1.0, mip_gap=1e-3)

    class FakeGurobiModel:
        def getAttr(self, attr):
            return 0

    term = type('Term', (), {'name': 'infeasible'})()

    class FakeSolver:
        options = {}
        _solver_model = FakeGurobiModel()

        def solve(self, model, **kwargs):
            return {'Solver': [{'Termination condition': term}]}

        def close(self):
            pass

    solver.model = object()  # pyrefly: ignore[bad-assignment]
    solver.solver = FakeSolver()
    solver.options = {}
    solver.solve_kwargs = {}
    with pytest.raises(OWNSolutionNotFound, match='Unable to find a solution'):
        solver.solve(time_limit=1.0, mip_gap=1e-3)


def test_solver_pyomo_error_branches(monkeypatch):
    from optiwindnet.MILP.pyomo import SolverPyomo, OWNSolutionNotFound

    solver = SolverPyomo('highs')
    with pytest.raises(AttributeError, match="has no attribute 'model'"):
        solver.solve(time_limit=1.0, mip_gap=1e-3)

    with pytest.raises(AttributeError, match="has no attribute 'model'"):
        solver._load_incumbent_topology()

    term = type('Term', (), {'name': 'infeasible'})()

    class PyomoResult(dict):
        solution = []

    class FakePyomoSolver:
        options = {}

        def solve(self, model, **kwargs):
            return PyomoResult({'Solver': [{'Termination condition': term}]})

    solver.model = object()  # pyrefly: ignore[bad-assignment]
    solver.solver = FakePyomoSolver()
    solver.options = {}
    solver.solve_kwargs = {}
    with pytest.raises(OWNSolutionNotFound, match='Unable to find a solution'):
        solver.solve(time_limit=1.0, mip_gap=1e-3)
