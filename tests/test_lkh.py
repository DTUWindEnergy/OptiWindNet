# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import math
import shutil
import subprocess
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import optiwindnet.baselines.lkh as lkh_mod
from optiwindnet.clustering import ClusterPlan
from optiwindnet.constraints import NetworkConstraints, _resolved
from optiwindnet.converting import linkbits_from_S
from optiwindnet.identity import linkset_id, topology_id
from optiwindnet.transforming import as_normalized
from optiwindnet.types import OWNWarmupFailed, Topology

from .cases import LKH_CASES, case_node_id, expected_topology
from .producers import lkh_topology
from .sitecache import get_bundle
from .topology_assertions import assert_topology


@pytest.mark.parametrize('case', LKH_CASES, ids=case_node_id)
def test_lkh_real_topology_cases(case):
    """Run a small optional matrix when the external LKH binary is installed."""
    if shutil.which('LKH') is None:
        pytest.skip('LKH executable not on PATH')
    S = lkh_topology(case)
    assert_topology(S, expected_topology(case), case.capacity)


def _make_routeset(branches: list[list[int]], R: int = 1) -> nx.Graph:
    T = sum(len(branch) for branch in branches)
    S = nx.Graph(T=T, R=R)
    for r in range(-R, 0):
        S.add_node(r, load=0)
    # all branches share the first root by default
    for subtree, branch in enumerate(branches):
        predecessor = -1
        for load, node in zip(range(len(branch), 0, -1), branch):
            S.add_node(node, load=load, subtree=subtree)
            S.add_edge(predecessor, node, load=load)
            predecessor = node
    S.nodes[-1]['load'] = T
    return S


def _make_A(T: int = 4, R: int = 1, edges=()) -> nx.Graph:
    A = nx.Graph(
        T=T,
        R=R,
        diagonals={},
        VertexC=np.zeros((T + R, 2)),
        d2roots=np.ones((T + R, R)),
        name='test',
    )
    A.add_nodes_from(range(T))
    A.add_nodes_from(range(-R, 0))
    for u, v, length in edges:
        A.add_edge(u, v, length=length)
    # Add the linkset metadata normally supplied by make_planar_embedding().
    A.graph['_canonical_terminal_links'] = np.array(
        sorted((u, v) if u < v else (v, u) for u, v, _ in edges),
        dtype=np.uint32,
    ).reshape(-1, 2)
    A.graph['_linkset_id'] = linkset_id(A)
    return A


def _fake_output(routes, *, cost=1.0, vehicles=2):
    return {
        'routes': routes,
        'penalty': 0,
        'minimum': str(int(cost * 1e5)),
        'cost': cost,
        'log': '',
        'stderr': '',
        'elapsed_time': 0.01,
        'solution_time': 0.0,
        'vehicles': vehicles,
        'seed': 0,
    }


def test_initial_tours_separate_the_routes_with_depot_clones():
    # T_c=4: depot 5, clones 6 and 7; the unused vehicle is an empty route
    tours = lkh_mod._initial_tours([[[0, 1], [2, 3]]], [[0, 1, 2, 3]], [3])
    assert tours == [[1, 2, 6, 3, 4, 7, 5]]


def test_initial_tours_refer_to_matrix_rows_not_walk_order():
    # nodes 3, 1, 2, 0 sit at rows 3, 1, 2, 0 of the matrix: ids 4, 2, 3, 1
    tours = lkh_mod._initial_tours([[[3, 1, 2, 0]]], [[0, 1, 2, 3]], [1])
    assert tours == [[4, 2, 3, 1, 5]]


def test_initial_tours_index_each_cluster_on_its_own():
    tours = lkh_mod._initial_tours(
        [[[11, 10]], [[20, 21]]], [[10, 11], [20, 21]], [1, 1]
    )
    assert tours == [[2, 1, 3], [1, 2, 3]]


def test_initial_tours_leave_a_root_without_routes_unseeded():
    tours = lkh_mod._initial_tours([[], [[0, 1, 2, 3]]], [[], [0, 1, 2, 3]], [0, 2])
    assert tours == [None, [1, 2, 3, 4, 6, 5]]


def test_lkh3_clusters_multi_root_solve_by_the_warmstart(monkeypatch):
    captured = {}

    def fake_do_lkh(L, **kwargs):
        captured[kwargs['name']] = (L.shape[0] - 1, kwargs['initial_tour_nodes'])
        return _fake_output(routes=[], vehicles=kwargs['vehicles'])

    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    monkeypatch.setattr(lkh_mod, 'cluster_plan', lambda *args: pytest.fail())
    # every terminal is closest to root -2 (d2roots is all ones)
    warmstart = nx.Graph(T=4, R=2)
    warmstart.add_edges_from([(-2, 3), (-1, 0), (0, 2), (2, 1)])

    with pytest.raises(ValueError, match='no solution'):
        lkh_mod.lkh3(
            _make_A(T=4, R=2),
            NetworkConstraints(capacity=3),
            time_limit=0.1,
            repair=False,
            warmstart=warmstart,
        )
    # root -2 holds terminal 3, root -1 the route 0, 2, 1 (rows 1, 3, 2), which
    # fits within capacity: 3 vehicles, so two empty routes (clones 5, 6)
    assert captured == {
        'test_root-2': (1, [1, 2]),
        'test_root-1': (3, [1, 3, 2, 5, 6, 4]),
    }


def test_lkh3_rejects_an_unusable_warmstart(monkeypatch):
    monkeypatch.setattr(lkh_mod, '_do_lkh', lambda L, **kw: pytest.fail('solved'))
    warmstart = nx.Graph(T=4, R=1)
    warmstart.add_edges_from([(-1, 0), (0, 1), (-1, 2)])  # terminal 3 left out
    with pytest.raises(OWNWarmupFailed, match='leaves 1 terminals out'):
        lkh_mod.lkh3(
            _make_A(T=4),
            NetworkConstraints(capacity=2),
            time_limit=0.1,
            repair=False,
            warmstart=warmstart,
        )
    warmstart.add_edge(-1, 3)  # three routes, over a maximum of two
    with pytest.raises(OWNWarmupFailed, match='more than the maximum of 2'):
        lkh_mod.lkh3(
            _make_A(T=4),
            NetworkConstraints(max_feeders=2, capacity=2),
            time_limit=0.1,
            repair=False,
            warmstart=warmstart,
        )


@pytest.mark.skipif(shutil.which('LKH') is None, reason='LKH not on PATH')
def test_lkh3_warmstart_is_its_starting_point():
    A = as_normalized(get_bundle('example_location').A)
    S0 = lkh_mod.lkh3(
        A, NetworkConstraints(capacity=4), time_limit=1.0, seed=0, repair=False
    )

    S1 = lkh_mod.lkh3(
        A,
        NetworkConstraints(capacity=4),
        time_limit=0.5,
        seed=1,
        repair=False,
        warmstart=S0,
    )

    assert S1.graph['objective'] <= S0.graph['objective'] + 1e-9


def test_lkh3_single_root_calls_do_lkh_with_expected_args(monkeypatch):
    A = _make_A(T=4)
    captured = {}

    def fake_do_lkh(L, **kwargs):
        captured['L_shape'] = L.shape
        captured.update(kwargs)
        return _fake_output(routes=[[0, 1], [2, 3]], vehicles=kwargs['vehicles'])

    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', lambda S, A, ringed=False: S)

    S = lkh_mod.lkh3(
        A, NetworkConstraints(capacity=2), time_limit=0.1, seed=42, repair=False
    )

    assert captured['L_shape'] == (5, 5)  # T_c + 1
    assert captured['vehicles'] == 2  # ceil(4/2) = 2
    # balanced=False branch: (T_c % capacity) or capacity = (4 % 2) or 2 = 2
    assert captured['min_route_size'] == 2
    assert captured['seed'] == 42
    assert S.graph['T'] == 4
    assert S.graph['R'] == 1
    assert S.graph['solver_details']['seed'] == 42
    assert S.graph['solver_details']['vehicles'] == 2


def _capture_do_lkh(monkeypatch, routes_):
    """Make _do_lkh() return ``routes_`` (one list per call) and record its calls."""
    calls = []
    routes_iter = iter(routes_)

    def fake_do_lkh(L, **kwargs):
        calls.append({'L': L, **kwargs})
        return _fake_output(routes=next(routes_iter), vehicles=kwargs['vehicles'])

    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    return calls


def test_lkh3_fills_radial_routes_with_slack_nodes(monkeypatch):
    # T=5 in 3 routes of at most 2: one slack node (row 5) opens the short route
    calls = _capture_do_lkh(monkeypatch, [[[5, 0], [1, 2], [3, 4]]])
    S = lkh_mod.lkh3(
        _make_A(T=5),
        NetworkConstraints(feeders='min', balanced=True, capacity=2),
        time_limit=0.1,
        repair=False,
    )
    [call] = calls
    assert call['asymmetric'] and call['L'].shape == (7, 7)
    assert (call['capacity'], call['vehicles']) == (2, 3)
    assert S.degree[-1] == 3
    assert sorted(S.nodes[n]['load'] for n in S.predecessors(-1)) == [1, 2, 2]
    assert S.graph['method_options']['feeders_exact']
    assert S.graph['solver_details']['capacity_effective'] == 2


def test_lkh3_bounds_the_ring_sizes_of_an_exact_count(monkeypatch):
    # 3 rings, above the minimum of 2 for T=6 at a ring capacity of 4
    calls = _capture_do_lkh(monkeypatch, [[[0, 1], [2, 3], [4, 5]]])
    lkh_mod.lkh3(
        _make_A(T=6),
        NetworkConstraints(topology='ringed', feeders=6, capacity=2),
        time_limit=0.1,
        repair=False,
    )
    [call] = calls
    assert not call['asymmetric']
    assert (call['min_route_size'], call['capacity'], call['vehicles']) == (2, 4, 3)


def test_lkh3_pins_only_the_roots_bounded_exactly(monkeypatch):
    # terminals 0, 1 go to root -2 (free), 2, 3, 4 to root -1 (exactly 2 routes)
    A = _make_A(T=5, R=2)
    A.graph['d2roots'] = np.array([[1.0, 2.0]] * 2 + [[2.0, 1.0]] * 3)
    calls = {}

    def fake_do_lkh(L, **kwargs):
        calls[kwargs['name']] = {'L': L, **kwargs}
        # root -1's slack node (row 3) opens its short route
        routes = [[3, 0], [1, 2]] if kwargs['name'] == 'test_root-1' else [[0, 1]]
        return _fake_output(routes=routes, vehicles=kwargs['vehicles'])

    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    S = lkh_mod.lkh3(
        A,
        NetworkConstraints(substation_feeders=(None, 2), capacity=2),
        time_limit=0.1,
        repair=False,
    )
    pinned, free = calls['test_root-1'], calls['test_root-2']
    assert pinned['asymmetric'] and pinned['L'].shape == (5, 5)
    assert (pinned['capacity'], pinned['vehicles']) == (2, 2)
    assert not free['asymmetric']
    assert S.degree[-1] == 2


@pytest.mark.parametrize(
    ('output', 'match'),
    [
        ({'penalty': 3}, 'penalty 3'),
        ({'routes': [[5, 0, 1], [2, 3, 4]]}, '2 routes, not 3'),
        ({'routes': [[5, 0], [1], [2, 3, 4]]}, 'route sizes 1 to 3, not 1 to 2'),
        ({'routes': [[0, 5], [1, 2], [3, 4]]}, '1 slack nodes off a route start'),
    ],
)
def test_lkh3_raises_on_a_pinned_solution_off_the_bounds(monkeypatch, output, match):
    def fake_do_lkh(L, **kwargs):
        return {
            **_fake_output(routes=[[5, 0], [1, 2], [3, 4]], vehicles=3),
            **output,
        }

    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    with pytest.raises(ValueError, match=match):
        lkh_mod.lkh3(
            _make_A(T=5),
            NetworkConstraints(feeders='min', balanced=True, capacity=2),
            time_limit=0.1,
            repair=False,
        )


def test_lkh3_refuses_more_exact_rings_than_terminal_pairs():
    with pytest.raises(ValueError, match='Rings \\(3\\) above the maximum \\(2\\)'):
        lkh_mod.lkh3(
            _make_A(T=5),
            NetworkConstraints(topology='ringed', feeders=6, capacity=3),
            time_limit=0.1,
            repair=False,
        )


def test_with_slack_nodes_lets_only_the_depot_lead_to_a_slack_node():
    L = lkh_mod._build_weight_matrix(
        _make_A(T=3), [0, 1, 2], -1, scale=1.0, complete=True, w_clip=999
    )
    M = lkh_mod._with_slack_nodes(L, 2, 999)
    assert M.shape == (6, 6)
    assert (M[3:5, :3] == L[:3, -1]).all()  # slack -> terminal: the feeder
    assert (M[:3, 3:5] == 999).all()  # terminal -> slack: forbidden
    assert M[3, 4] == M[4, 3] == 999  # slack -> slack: forbidden
    assert (M[-1, 3:5] == 0).all() and (M[3:5, -1] == 0).all()


def test_initial_tours_open_the_routes_with_slack_nodes():
    # T_c=5, 1 slack node (id 6), depot 7, clones 8 and 9; capacity 2
    tours = lkh_mod._initial_tours(
        [[[0], [1, 2], [3, 4]]], [[0, 1, 2, 3, 4]], [3], [1], [2]
    )
    assert tours == [[6, 1, 8, 2, 3, 9, 4, 5, 7]]


def test_lkh3_does_not_keep_feeders_straight():
    with pytest.raises(NotImplementedError, match='straight'):
        lkh_mod.lkh3(
            _make_A(T=4),
            NetworkConstraints(feeder_route='straight', capacity=2),
            time_limit=0.1,
            repair=False,
        )


@pytest.mark.parametrize(
    ('capacity', 'max_feeders', 'vehicles'),
    [
        (2, None, 2),  # the minimum
        (2, 3, 3),  # the maximum is offered whole
        (4, None, 4),  # a cluster within capacity gets an allowance
        (4, 'min', 1),  # ... but not beyond the maximum
    ],
)
def test_lkh3_offers_the_vehicles_the_bounds_allow(
    monkeypatch, capacity, max_feeders, vehicles
):
    captured = {}

    def fake_do_lkh(L, **kwargs):
        captured.update(kwargs)
        return _fake_output(routes=[[0, 1, 2, 3]], vehicles=kwargs['vehicles'])

    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    lkh_mod.lkh3(
        _make_A(T=4),
        NetworkConstraints(max_feeders=max_feeders, capacity=capacity),
        time_limit=0.1,
        repair=False,
    )
    assert captured['vehicles'] == vehicles


@pytest.mark.skipif(shutil.which('LKH') is None, reason='LKH not on PATH')
def test_lkh3_keeps_the_substation_limits():
    from optiwindnet.validating import validate_topology

    A = as_normalized(get_bundle('neart', read_powers=False).A)
    # unlimited, neart's root -1 takes 5 feeders and 33 turbines
    constraints = NetworkConstraints(
        capacity=7, max_substation_feeders=(None, 4), max_substation_loads=(None, 26)
    )

    S = lkh_mod.lkh3(A, constraints, time_limit=1.0, seed=1)

    assert S.degree[-1] <= 4 and S.nodes[-1]['load'] <= 26
    assert validate_topology(S, 7) == []
    assert S.graph['solver_details']['root_feeders_ub'][0] is None


@pytest.mark.skipif(shutil.which('LKH') is None, reason='LKH not on PATH')
@pytest.mark.parametrize(
    ('site', 'capacity', 'constraints'),
    [
        ('mermaid', 5, {'feeders': 'min', 'balanced': True}),
        ('mermaid', 5, {'feeders': 'min+2'}),
        ('mermaid', 5, {'topology': 'ringed', 'feeders': 'min', 'balanced': True}),
        ('mermaid', 5, {'topology': 'ringed', 'feeders': 'min+2'}),
        ('neart', 7, {'feeders': 'min', 'balanced': True}),
        ('neart', 7, {'substation_feeders': (None, 4)}),
        ('neart', 7, {'topology': 'ringed', 'feeders': 'min', 'balanced': True}),
    ],
)
def test_lkh3_holds_exact_counts_and_balance(site, capacity, constraints):
    from optiwindnet.validating import validate_topology

    A = as_normalized(get_bundle(site, read_powers=False).A)
    R = A.graph['R']
    constraints = NetworkConstraints(capacity=capacity, **constraints)
    bounds = _resolved(A, constraints)

    S = lkh_mod.lkh3(A, constraints, time_limit=1.0, seed=1)

    assert validate_topology(S, capacity) == []
    feeders_ = [S.degree[r] for r in range(-R, 0)]
    assert bounds.feeders_lb <= sum(feeders_) <= (bounds.feeders_ub or sum(feeders_))
    for feeders, lo, hi in zip(
        feeders_,
        bounds.root_feeders_lb or (0,) * R,
        bounds.root_feeders_ub or (None,) * R,
    ):
        assert lo <= feeders <= (feeders if hi is None else hi)
    if bounds.balanced and bounds.topology is not Topology.RINGED:
        loads = [S.nodes[n]['load'] for r in range(-R, 0) for n in S.predecessors(r)]
        assert bounds.group_load_lb <= min(loads)
        assert max(loads) <= bounds.group_load_ub


def test_lkh3_stops_lkh_that_hangs(monkeypatch):
    """LKH hangs when its preprocessing takes up the whole time limit."""
    timeouts = []

    def hanging_run(argv, timeout, **kwargs):
        timeouts.append(timeout)
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(lkh_mod.subprocess, 'run', hanging_run)
    with pytest.raises(ValueError, match='shorter than its preprocessing'):
        lkh_mod.lkh3(
            _make_A(T=4), NetworkConstraints(capacity=2), time_limit=0.1, repair=False
        )
    assert timeouts == [1.5 * 0.1 + 2]


def test_lkh3_raises_when_lkh_finds_no_solution(monkeypatch):
    monkeypatch.setattr(
        lkh_mod, '_do_lkh', lambda L, **kw: _fake_output(routes=[], vehicles=1)
    )
    with pytest.raises(ValueError, match='no solution for root -1'):
        lkh_mod.lkh3(
            _make_A(T=4), NetworkConstraints(capacity=4), time_limit=0.1, repair=False
        )


def test_lkh3_caps_each_root_by_its_feeder_limit(monkeypatch):
    A = _make_A(T=4, R=2)
    A.graph['d2roots'] = np.array([[1.0, 2.0], [1.0, 2.0], [2.0, 1.0], [2.0, 1.0]])
    vehicles_ = {}

    def fake_do_lkh(L, **kwargs):
        vehicles_[kwargs['name']] = kwargs['vehicles']
        return _fake_output(routes=[[0, 1]], vehicles=kwargs['vehicles'])

    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    S = lkh_mod.lkh3(
        A,
        NetworkConstraints(max_substation_feeders=(1, None), capacity=2),
        time_limit=0.1,
        repair=False,
    )
    # each cluster fits within capacity: an allowance of its 2 terminals
    assert vehicles_ == {'test_root-2': 1, 'test_root-1': 2}
    assert S.graph['solver_details']['root_feeders_ub'] == (1, None)


def test_lkh3_seed_none_picks_random_seed(monkeypatch):
    A = _make_A(T=4)
    captured = {}

    def fake_do_lkh(L, **kwargs):
        captured.update(kwargs)
        return _fake_output(routes=[[0, 1], [2, 3]], vehicles=2)

    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', lambda S, A, ringed=False: S)

    S = lkh_mod.lkh3(
        A, NetworkConstraints(capacity=2), time_limit=0.1, seed=None, repair=False
    )

    assert isinstance(captured['seed'], int)
    assert captured['seed'] >= 0
    assert S.graph['solver_details']['seed'] == captured['seed']


def test_lkh3_repairs_on_crossings(monkeypatch):
    A = _make_A(T=4, edges=[(0, 1, 1.0), (2, 3, 2.0)])
    A.graph['diagonals'] = {(0, 1): None}

    repaired = [
        # first iteration: outstanding crossing → triggers retry + edge removal
        ('with_cross', [((0, 1), (2, 3))]),
        ('clean', []),
    ]
    repair_iter = iter(repaired)

    def fake_do_lkh(L, **kwargs):
        return _fake_output(routes=[[0, 1], [2, 3]], vehicles=2)

    def fake_repair(S, A_inner, ringed=False):
        label, crossings = next(repair_iter)
        S.graph['outstanding_crossings'] = crossings
        S.graph['_label'] = label
        return S

    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', fake_repair)

    S = lkh_mod.lkh3(A, NetworkConstraints(capacity=2), time_limit=0.1, seed=1)
    assert S.graph['retries'] == 1
    assert S.graph['_label'] == 'clean'


def test_lkh3_max_retries_warns(monkeypatch):
    A = _make_A(T=4, edges=[(0, 1, 1.0), (2, 3, 2.0)])
    A.graph['diagonals'] = {(0, 1): None}

    def fake_do_lkh(L, **kwargs):
        return _fake_output(routes=[[0, 1], [2, 3]], vehicles=2)

    def fake_repair(S, A_inner, ringed=False):
        S.graph['outstanding_crossings'] = [((0, 1), (2, 3))]
        return S

    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', fake_repair)

    warnings_seen = []
    monkeypatch.setattr(lkh_mod, 'warn', warnings_seen.append)

    S = lkh_mod.lkh3(
        A, NetworkConstraints(capacity=2), time_limit=0.1, seed=1, max_retries=1
    )
    assert S.graph['retries'] == 1
    assert any('max_retries reached' in str(w) for w in warnings_seen)


def test_lkh3_repair_false_skips_repair_loop(monkeypatch):
    A = _make_A(T=4)
    repair_calls = []

    def fake_do_lkh(L, **kwargs):
        return _fake_output(routes=[[0, 1], [2, 3]], vehicles=2)

    def fake_repair(S, A_inner, ringed=False):
        repair_calls.append(1)
        return S

    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', fake_repair)

    S = lkh_mod.lkh3(
        A, NetworkConstraints(capacity=2), time_limit=0.1, seed=1, repair=False
    )
    assert repair_calls == []
    assert 'retries' not in S.graph


def test_lkh3_multi_root_runs_one_call_per_cluster(monkeypatch):
    A = _make_A(T=4, R=2)
    A.graph['d2roots'] = np.ones((6, 2))
    captured_calls = []

    def fake_cluster_plan(A_inner, bounds):
        return ClusterPlan([{0, 1}, {2, 3}], (1, 1), (False, False))

    def fake_do_lkh(L, **kwargs):
        captured_calls.append(kwargs.copy())
        return _fake_output(routes=[[0, 1]], vehicles=1)

    monkeypatch.setattr(lkh_mod, 'cluster_plan', fake_cluster_plan)
    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', lambda S, A, ringed=False: S)

    S = lkh_mod.lkh3(
        A, NetworkConstraints(capacity=2), time_limit=0.1, seed=7, repair=False
    )
    assert len(captured_calls) == 2
    # multi-root: solver_details aggregates per root
    assert S.graph['solver_details']['vehicles'] == (1, 1)
    assert S.graph['R'] == 2
    # both clusters used the same seed
    assert all(call['seed'] == 7 for call in captured_calls)


def test_lkh3_multi_root_accepts_vehicles_above_min(monkeypatch, caplog):
    A = _make_A(T=4, R=2)
    A.graph['d2roots'] = np.ones((6, 2))

    def fake_cluster_plan(A_inner, bounds):
        return ClusterPlan([{0, 1}, {2, 3}], (1, 1), (False, False))

    def fake_do_lkh(L, **kwargs):
        return _fake_output(routes=[[0, 1]], vehicles=1)

    monkeypatch.setattr(lkh_mod, 'cluster_plan', fake_cluster_plan)
    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', lambda S, A, ringed=False: S)

    lkh_mod.lkh3(
        A,
        NetworkConstraints(capacity=2, max_feeders=4),
        time_limit=0.1,
        seed=1,
        repair=False,
    )
    assert 'multi-root' not in caplog.text


def test_lkh3_records_identity_over_the_available_links(monkeypatch):
    A = _make_A(T=4, edges=[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (0, 3, 1.0)])

    monkeypatch.setattr(
        lkh_mod, '_do_lkh', lambda L, **kw: _fake_output(routes=[[0, 1], [2, 3]])
    )
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', lambda S, A, ringed=False: S)

    S = lkh_mod.lkh3(A, NetworkConstraints(capacity=2), time_limit=0.1, seed=1)

    assert S.graph['_linkbits'] == linkbits_from_S(A, S)
    assert S.graph['_topology_id'] == topology_id(S.graph['_linkbits'])
    assert S.graph['_linkset_id'] == A.graph['_linkset_id']


def test_lkh3_raises_when_the_solution_leaves_A(monkeypatch):
    """The big-M only discourages a link absent from A: a solution may use it."""
    A = _make_A(T=4, edges=[(0, 1, 1.0)])

    # route [2, 3] activates (2, 3), which A does not have
    monkeypatch.setattr(
        lkh_mod, '_do_lkh', lambda L, **kw: _fake_output(routes=[[0, 1], [2, 3]])
    )
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', lambda S, A, ringed=False: S)

    match = 'rogue links absent from the mesh: \\[\\(2, 3\\)\\]'
    with pytest.raises(ValueError, match=match):
        lkh_mod.lkh3(A, NetworkConstraints(capacity=2), time_limit=0.1, seed=1)


def test_lkh3_complete_that_stays_within_A_is_identified_over_A(monkeypatch):
    """A complete solve using only links in A retains an identity over A."""
    A = _make_A(T=4, edges=[(0, 1, 1.0), (2, 3, 1.0)])

    monkeypatch.setattr(
        lkh_mod, '_do_lkh', lambda L, **kw: _fake_output(routes=[[0, 1], [2, 3]])
    )

    S = lkh_mod.lkh3(
        A,
        NetworkConstraints(capacity=2),
        time_limit=0.1,
        seed=1,
        complete=True,
        repair=False,
    )

    assert S.graph['method_options']['complete']
    # the options that shape the solve are part of the stored method
    assert {
        key: S.graph['method_options'][key]
        for key in ('balanced', 'ringed', 'repair', 'max_retries')
    } == {'balanced': False, 'ringed': False, 'repair': False, 'max_retries': 10}
    assert S.graph['_linkbits'] == linkbits_from_S(A, S)
    assert S.graph['_linkset_id'] == A.graph['_linkset_id'] == linkset_id(A)


def test_lkh3_complete_that_leaves_A_is_not_identified(monkeypatch, caplog):
    A = _make_A(T=4, edges=[(0, 1, 1.0), (2, 3, 1.0)])

    # (0, 2) and (1, 3) are the links the complete graph adds
    monkeypatch.setattr(
        lkh_mod, '_do_lkh', lambda L, **kw: _fake_output(routes=[[0, 2], [1, 3]])
    )

    with caplog.at_level('WARNING'):
        S = lkh_mod.lkh3(
            A,
            NetworkConstraints(capacity=2),
            time_limit=0.1,
            seed=1,
            complete=True,
            repair=False,
        )

    assert '_linkbits' not in S.graph
    assert 'left the available-links set' in caplog.text
    assert A.graph['_linkset_id'] == linkset_id(A)  # input identity is unchanged


def test_lkh3_edgeless_A_implies_complete(monkeypatch, caplog):
    A = _make_A(T=4)

    monkeypatch.setattr(
        lkh_mod, '_do_lkh', lambda L, **kw: _fake_output(routes=[[0, 1], [2, 3]])
    )

    with caplog.at_level('WARNING'):
        S = lkh_mod.lkh3(
            A, NetworkConstraints(capacity=2), time_limit=0.1, seed=1, repair=False
        )

    assert S.graph['method_options']['complete']
    assert '_linkbits' not in S.graph
    assert 'left the available-links set' in caplog.text


def test_lkh3_complete_is_refused_together_with_repair():
    A = _make_A(T=4, edges=[(0, 1, 1.0), (2, 3, 1.0)])
    with pytest.raises(NotImplementedError, match='complete graph over all nodes'):
        lkh_mod.lkh3(
            A, NetworkConstraints(capacity=2), time_limit=0.1, seed=1, complete=True
        )


def test_build_weight_matrix_single_root_shape_and_depot_column():
    A = _make_A(T=3, edges=[(0, 1, 2.5)])
    A.graph['d2roots'] = np.array([[1.0], [2.0], [3.0], [0.0]])
    L = lkh_mod._build_weight_matrix(
        A, terminals=[0, 1, 2], root=-1, scale=10.0, complete=False, w_clip=999
    )
    assert L.shape == (4, 4)
    # edge (0,1) weight=2.5*10=25
    assert L[0, 1] == 25 and L[1, 0] == 25
    # depot column from d2roots
    assert L[0, -1] == 10
    assert L[1, -1] == 20
    assert L[2, -1] == 30
    # missing edges stay at w_clip
    assert L[0, 2] == 999


def test_route_from_tour_parses_tour_file(tmp_path):
    tour_file = tmp_path / 'test.tour'
    tour_file.write_text(
        'NAME : test\nTYPE : TOUR\nDIMENSION : 4\nTOUR_SECTION\n1\n2\n3\n4\n-1\nEOF\n'
    )
    L = np.array(
        [
            [0, 10, 50, 5],
            [10, 0, 10, 50],
            [50, 10, 0, 10],
            [5, 50, 10, 0],
        ]
    )
    route, cost = lkh_mod._route_from_tour(str(tour_file), L)
    assert isinstance(route, list)
    assert isinstance(cost, (int, float))
    assert set(route) == {0, 1, 2}


def test_lkh_lower_level_function(monkeypatch):
    A = _make_A(T=4)
    warnings_seen = []
    monkeypatch.setattr(lkh_mod, 'warn', warnings_seen.append)

    def fake_do_lkh(L, **kwargs):
        return _fake_output(routes=[[0, 1], [2, 3]], vehicles=2)

    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', lambda S, A, ringed=False: S)

    S = lkh_mod._lkh(A, capacity=2, vehicles=1, time_limit=0.1)
    assert any('too low' in str(w) for w in warnings_seen)
    assert S.graph.get('has_loads') is True
    assert S.nodes[-1]['load'] == 4


def _capture_lkh_problem(monkeypatch):
    """Capture the generated LKH-3 problem and parameter files without running LKH.

    The fake process returns no solution, causing lkh3() to raise ValueError
    when checking the result. Callers expect that error and inspect the files
    written before the subprocess call.
    """
    captured = {}

    def fake_run(argv, **kwargs):
        params = Path(argv[1]).read_text()
        captured['params'] = dict(
            line.split(' = ', 1) for line in params.splitlines() if ' = ' in line
        )
        captured['flags'] = {line for line in params.splitlines() if ' = ' not in line}
        captured['problem'] = Path(captured['params']['PROBLEM_FILE']).read_text()
        return subprocess.CompletedProcess(argv, 1, stdout=b'', stderr=b'')

    monkeypatch.setattr(lkh_mod.subprocess, 'run', fake_run)
    return captured


def _demand_section(problem: str) -> dict[int, int]:
    body = problem.split('DEMAND_SECTION\n', 1)[1].split('\nEOF', 1)[0]
    return {int(node): int(demand) for node, demand in
            (line.split() for line in body.splitlines())}  # fmt: skip


def test_lkh3_sends_one_unit_of_demand_per_unitary_terminal(monkeypatch):
    captured = _capture_lkh_problem(monkeypatch)
    A = get_bundle('toy').A
    T = A.graph['T']

    with pytest.raises(ValueError, match='no solution'):
        lkh_mod.lkh3(
            as_normalized(A),
            NetworkConstraints(capacity=5),
            time_limit=0.1,
            repair=False,
        )

    demands = _demand_section(captured['problem'])
    assert demands == {**{t + 1: 1 for t in range(T)}, T + 1: 0}
    assert captured['params']['MTSP_MAX_SIZE'] == '5'


@pytest.mark.parametrize('topology', ['radial', 'ringed'])
def test_lkh3_searches_with_special_moves(monkeypatch, topology):
    captured = _capture_lkh_problem(monkeypatch)

    with pytest.raises(ValueError, match='no solution'):
        lkh_mod.lkh3(
            as_normalized(get_bundle('toy').A),
            NetworkConstraints(topology=topology, capacity=5),
            time_limit=0.1,
            repair=False,
        )

    assert 'SPECIAL' in captured['flags']


def test_lkh3_declares_terminal_inflow_as_demand(monkeypatch):
    captured = _capture_lkh_problem(monkeypatch)
    A = as_normalized(get_bundle('toy').A.copy())
    T = A.graph['T']
    inflow = {t: 1 + (t % 3) for t in range(T)}
    nx.set_node_attributes(A, inflow, 'inflow')

    with pytest.raises(ValueError, match='no solution'):
        lkh_mod.lkh3(A, NetworkConstraints(capacity=8), time_limit=0.1, repair=False)

    demands = _demand_section(captured['problem'])
    assert demands == {**{t + 1: inflow[t] for t in range(T)}, T + 1: 0}
    assert captured['problem'].splitlines()[0].startswith('NAME')
    # The smallest demand is one unit, allowing at most 8 nodes per route.
    assert captured['params']['MTSP_MAX_SIZE'] == '8'
    # Unequal demands disable the minimum node count.
    assert captured['params']['MTSP_MIN_SIZE'] == '0'
    # the feeder minimum follows the total inflow (24), not the terminal count
    assert int(captured['params']['VEHICLES']) >= math.ceil(sum(inflow.values()) / 8)


def test_lkh3_rejects_non_unit_terminal_inflow_it_cannot_honour():
    """Unsupported modes are rejected before invoking the LKH executable."""
    A = as_normalized(get_bundle('toy').A.copy())
    nx.set_node_attributes(A, {0: 2}, 'inflow')

    with pytest.raises(NotImplementedError, match='single-root radial solve'):
        lkh_mod.lkh3(
            A, NetworkConstraints(capacity=5, topology='ringed'), time_limit=1.0
        )
    with pytest.raises(NotImplementedError, match='single-root radial solve'):
        lkh_mod.lkh3(
            A,
            NetworkConstraints(capacity=5, feeders=3, balanced=True),
            time_limit=1.0,
        )

    Amr = as_normalized(get_bundle('neart').A.copy())
    nx.set_node_attributes(Amr, {0: 2}, 'inflow')
    with pytest.raises(NotImplementedError, match='single-root radial solve'):
        lkh_mod.lkh3(Amr, NetworkConstraints(capacity=5), time_limit=1.0)
