import networkx as nx
import numpy as np
import pytest

import optiwindnet.baselines.lkh as lkh_mod


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


def test_initial_tour_from_solution_preserves_max_load():
    S = _make_routeset([[0, 1, 2], [3]])
    initial_tour = lkh_mod._initial_tour_from_solution(S, vehicles=2)
    assert initial_tour == [1, 2, 3, 4, 6, 5]


def test_split_warmstart_per_root_single_root():
    warmstart = _make_routeset([[0, 1], [2, 3]])
    tours = lkh_mod._split_warmstart_per_root(warmstart, vehicles_=[2])
    # 4 terminals walked in branch order, plus 1 depot clone, then depot at the end
    assert tours == [[1, 2, 3, 4, 6, 5]]


def test_lkh3_single_root_calls_do_lkh_with_expected_args(monkeypatch):
    A = _make_A(T=4)
    captured = {}

    def fake_do_lkh(L, **kwargs):
        captured['L_shape'] = L.shape
        captured.update(kwargs)
        return _fake_output(routes=[[0, 1], [2, 3]], vehicles=kwargs['vehicles'])

    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', lambda S, A: S)

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
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', lambda S, A: S)

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
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', lambda S, A: S)

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

    def fake_repair(S, A_inner):
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

    def fake_repair(S, A_inner):
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

    def fake_repair(S, A_inner):
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
        return [{0, 1}, {2, 3}], [0, 0]

    def fake_do_lkh(L, **kwargs):
        captured_calls.append(kwargs.copy())
        return _fake_output(routes=[[0, 1]], vehicles=1)

    monkeypatch.setattr(lkh_mod, 'clusterize', fake_clusterize)
    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', lambda S, A: S)

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
        return [{0, 1}, {2, 3}], [0, 0]

    def fake_do_lkh(L, **kwargs):
        return _fake_output(routes=[[0, 1]], vehicles=1)

    monkeypatch.setattr(lkh_mod, 'clusterize', fake_clusterize)
    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', lambda S, A: S)

    warnings_seen = []
    monkeypatch.setattr(lkh_mod, 'warn', warnings_seen.append)

    lkh_mod.lkh3(A, capacity=2, time_limit=0.1, seed=1, vehicles=4)
    assert any('multi-root' in str(w) for w in warnings_seen)


def test_iterative_lkh_emits_deprecation_warning(monkeypatch):
    A = _make_A(T=4)
    A.graph['d2roots'] = np.ones((5, 1))

    def fake_do_lkh(L, **kwargs):
        return _fake_output(routes=[[0, 1], [2, 3]], vehicles=2)

    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', lambda S, A: S)
    monkeypatch.setattr(lkh_mod, 'add_link_blockmap', lambda A: None)
    monkeypatch.setattr(lkh_mod, '_prune_links', lambda A, limit: None)

    with pytest.warns(DeprecationWarning, match='iterative_lkh'):
        lkh_mod.iterative_lkh(A, capacity=2, time_limit=0.1, seed=1)


def test_lkh_emits_deprecation_warning(monkeypatch):
    A = _make_A(T=4)
    A.graph['d2roots'] = np.ones((5, 1))

    def fake_do_lkh(L, **kwargs):
        return _fake_output(routes=[[0, 1], [2, 3]], vehicles=2)

    monkeypatch.setattr(lkh_mod, '_do_lkh', fake_do_lkh)

    with pytest.warns(DeprecationWarning, match='lkh'):
        lkh_mod.lkh(A, capacity=2, time_limit=0.1, seed=1)


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
