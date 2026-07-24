import shutil

import networkx as nx
import numpy as np
import pytest

import optiwindnet.baselines.lkh as lkh_mod
from optiwindnet.interarraylib import as_normalized

from .cases import LKH_CASES, case_node_id, expected_topology
from .sitecache import get_bundle
from .topology_assertions import assert_topology


@pytest.mark.parametrize('case', LKH_CASES, ids=case_node_id)
def test_lkh_real_topology_cases(case):
    """Run a small optional matrix when the external LKH binary is installed."""
    if shutil.which('LKH') is None:
        pytest.skip('LKH executable not on PATH')
    A = get_bundle(case.site).A
    S = lkh_mod.lkh3(
        as_normalized(A),
        capacity=case.capacity,
        time_limit=case.time_limit,
        ringed=case.ringed,
        seed=case.seed,
    )
    assert_topology(S, expected_topology(case), case.capacity)


def _make_routeset(branches: list[list[int]], R: int = 1) -> nx.Graph:
    T = sum(len(branch) for branch in branches)
    S = nx.Graph(T=T, R=R)
    for r in range(-R, 0):
        S.add_node(r, load=0)
    subtree = 0
    # all branches share the first root by default
    for branch in branches:
        predecessor = -1
        for load, node in zip(range(len(branch), 0, -1), branch):
            S.add_node(node, load=load, subtree=subtree)
            S.add_edge(predecessor, node, load=load)
            predecessor = node
        subtree += 1
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
    return A


def _fake_output(routes, *, cost=1.0, vehicles=2):
    return dict(
        routes=routes,
        penalty=0,
        minimum=str(int(cost * 1e5)),
        cost=cost,
        log='',
        stderr='',
        elapsed_time=0.01,
        solution_time=0.0,
        vehicles=vehicles,
        seed=0,
    )


def test_initial_tours_from_warmstart_walked_in_branch_order():
    warmstart = _make_routeset([[0, 1], [2, 3]])
    # R=1 with sorted terminals == node ids; walk order [0,1,2,3] → ids [1,2,3,4]
    tours = lkh_mod._initial_tours_from_warmstart(
        warmstart, terminals_=[[0, 1, 2, 3]], vehicles_=[2]
    )
    # 4 customer ids, then 1 depot clone (vehicles - 1), then depot at the end
    assert tours == [[1, 2, 3, 4, 6, 5]]


def test_initial_tours_from_warmstart_uses_matrix_index_not_walk_order():
    # Walk order [3, 1, 2, 0] differs from sorted-terminal order. The LKH
    # initial tour must reference each customer by its matrix index + 1
    # (i.e. its position in the sorted `terminals` list), NOT by walk
    # rank — otherwise LKH starts from a permutation unrelated to the
    # warmstart structure.
    warmstart = _make_routeset([[3, 1, 2, 0]])
    tours = lkh_mod._initial_tours_from_warmstart(
        warmstart, terminals_=[[0, 1, 2, 3]], vehicles_=[1]
    )
    # nodes 3,1,2,0 → matrix indices 3,1,2,0 → LKH ids 4,2,3,1; depot at T+1=5.
    assert tours == [[4, 2, 3, 1, 5]]


def test_initial_tours_from_warmstart_multi_root_uses_per_cluster_indices():
    # Two roots, each with two terminals. Cluster -2 has terminals {10, 11};
    # cluster -1 has terminals {20, 21}. Each cluster is indexed independently:
    # node 10 → id 1, node 11 → id 2 within cluster -2; etc.
    warmstart = nx.Graph(T=4, R=2)
    for r in (-2, -1):
        warmstart.add_node(r, load=2)
    # Branch under -2: walk 11, 10
    warmstart.add_node(11, load=2, subtree=0)
    warmstart.add_node(10, load=1, subtree=0)
    warmstart.add_edge(-2, 11, load=2)
    warmstart.add_edge(11, 10, load=1)
    # Branch under -1: walk 20, 21
    warmstart.add_node(20, load=2, subtree=1)
    warmstart.add_node(21, load=1, subtree=1)
    warmstart.add_edge(-1, 20, load=2)
    warmstart.add_edge(20, 21, load=1)

    tours = lkh_mod._initial_tours_from_warmstart(
        warmstart,
        terminals_=[[10, 11], [20, 21]],
        vehicles_=[1, 1],
    )
    # Cluster -2 walks 11→10 → ids 2,1; depot=3 (T_c+1).
    # Cluster -1 walks 20→21 → ids 1,2; depot=3.
    assert tours == [[2, 1, 3], [1, 2, 3]]


def test_initial_tours_from_warmstart_empty_root_returns_none():
    warmstart = _make_routeset([[0, 1, 2, 3]], R=2)  # all branches under -1
    tours = lkh_mod._initial_tours_from_warmstart(
        warmstart, terminals_=[[], [0, 1, 2, 3]], vehicles_=[1, 2]
    )
    assert tours[0] is None
    assert tours[1] == [1, 2, 3, 4, 6, 5]


def test_lkh3_single_root_calls_do_lkh_with_expected_args(monkeypatch):
    A = _make_A(T=4)
    captured = {}

    def fake_do_lkh(L, **kwargs):
        captured['L_shape'] = L.shape
        captured.update(kwargs)
        return _fake_output(routes=[[0, 1], [2, 3]], vehicles=kwargs['vehicles'])

    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', lambda S, A, ringed=False: S)

    S = lkh_mod.lkh3(A, capacity=2, time_limit=0.1, seed=42)

    assert captured['L_shape'] == (5, 5)  # T_c + 1
    assert captured['vehicles'] == 2  # ceil(4/2) = 2
    # balanced=False branch: (T_c % capacity) or capacity = (4 % 2) or 2 = 2
    assert captured['min_route_size'] == 2
    assert captured['seed'] == 42
    assert S.graph['T'] == 4
    assert S.graph['R'] == 1
    assert S.graph['solver_details']['seed'] == 42
    assert S.graph['solver_details']['vehicles'] == 2


def test_lkh3_balanced_sets_min_route_size(monkeypatch):
    A = _make_A(T=5)  # capacity=2 -> vehicles=3, leftover=1
    captured = {}

    def fake_do_lkh(L, **kwargs):
        captured.update(kwargs)
        return _fake_output(routes=[[0, 1], [2, 3], [4]], vehicles=3)

    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', lambda S, A, ringed=False: S)

    lkh_mod.lkh3(A, capacity=2, time_limit=0.1, seed=1, balanced=True)

    assert captured['min_route_size'] == 1  # 5 % 2 = 1
    assert captured['vehicles'] == 3


def test_lkh3_seed_none_picks_random_seed(monkeypatch):
    A = _make_A(T=4)
    captured = {}

    def fake_do_lkh(L, **kwargs):
        captured.update(kwargs)
        return _fake_output(routes=[[0, 1], [2, 3]], vehicles=2)

    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', lambda S, A, ringed=False: S)

    S = lkh_mod.lkh3(A, capacity=2, time_limit=0.1, seed=None)

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
    monkeypatch.setattr(lkh_mod, 'add_link_blockmap', lambda A: None)
    monkeypatch.setattr(lkh_mod, '_prune_links', lambda A, limit: None)

    S = lkh_mod.lkh3(A, capacity=2, time_limit=0.1, seed=1)
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
    monkeypatch.setattr(lkh_mod, 'add_link_blockmap', lambda A: None)
    monkeypatch.setattr(lkh_mod, '_prune_links', lambda A, limit: None)

    warnings_seen = []
    monkeypatch.setattr(lkh_mod, 'warn', warnings_seen.append)

    S = lkh_mod.lkh3(A, capacity=2, time_limit=0.1, seed=1, max_retries=1)
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

    S = lkh_mod.lkh3(A, capacity=2, time_limit=0.1, seed=1, repair=False)
    assert repair_calls == []
    assert 'retries' not in S.graph


def test_lkh3_multi_root_runs_one_call_per_cluster(monkeypatch):
    A = _make_A(T=4, R=2)
    A.graph['d2roots'] = np.ones((6, 2))
    captured_calls = []

    def fake_clusterize(A_inner, capacity):
        return [{0, 1}, {2, 3}]

    def fake_do_lkh(L, **kwargs):
        captured_calls.append(kwargs.copy())
        return _fake_output(routes=[[0, 1]], vehicles=1)

    monkeypatch.setattr(lkh_mod, 'clusterize', fake_clusterize)
    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', lambda S, A, ringed=False: S)

    S = lkh_mod.lkh3(A, capacity=2, time_limit=0.1, seed=7)
    assert len(captured_calls) == 2
    # multi-root: solver_details aggregates per root
    assert S.graph['solver_details']['vehicles'] == (1, 1)
    assert S.graph['R'] == 2
    # both clusters used the same seed
    assert all(call['seed'] == 7 for call in captured_calls)


def test_lkh3_multi_root_warns_when_vehicles_above_min(monkeypatch):
    A = _make_A(T=4, R=2)
    A.graph['d2roots'] = np.ones((6, 2))

    def fake_clusterize(A_inner, capacity):
        return [{0, 1}, {2, 3}]

    def fake_do_lkh(L, **kwargs):
        return _fake_output(routes=[[0, 1]], vehicles=1)

    monkeypatch.setattr(lkh_mod, 'clusterize', fake_clusterize)
    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', lambda S, A, ringed=False: S)

    warnings_seen = []
    monkeypatch.setattr(lkh_mod, 'warn', warnings_seen.append)

    lkh_mod.lkh3(A, capacity=2, time_limit=0.1, seed=1, vehicles=4)
    assert any('multi-root' in str(w) for w in warnings_seen)


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
