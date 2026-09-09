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
from optiwindnet.converting import linkbits_from_S
from optiwindnet.identity import linkset_id, topology_id
from optiwindnet.transforming import as_normalized

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

    S = lkh_mod.lkh3(A, capacity=2, time_limit=0.1, seed=42, repair=False)

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

    lkh_mod.lkh3(A, capacity=2, time_limit=0.1, seed=1, balanced=True, repair=False)

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

    S = lkh_mod.lkh3(A, capacity=2, time_limit=0.1, seed=None, repair=False)

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

    S = lkh_mod.lkh3(A, capacity=2, time_limit=0.1, seed=7, repair=False)
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

    lkh_mod.lkh3(A, capacity=2, time_limit=0.1, seed=1, vehicles=4, repair=False)
    assert any('multi-root' in str(w) for w in warnings_seen)


def test_lkh3_records_identity_over_the_available_links(monkeypatch):
    A = _make_A(T=4, edges=[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (0, 3, 1.0)])

    monkeypatch.setattr(
        lkh_mod, '_do_lkh', lambda L, **kw: _fake_output(routes=[[0, 1], [2, 3]])
    )
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', lambda S, A, ringed=False: S)

    S = lkh_mod.lkh3(A, capacity=2, time_limit=0.1, seed=1)

    assert S.graph['_linkbits'] == linkbits_from_S(A, S)
    assert S.graph['_topology_id'] == topology_id(S.graph['_linkbits'])
    assert S.graph['_linkset_id'] == A.graph['_linkset_id']


def test_lkh3_drops_identity_when_the_solution_leaves_A(monkeypatch, caplog):
    """The big-M only discourages a link absent from A: it cannot forbid it."""
    A = _make_A(T=4, edges=[(0, 1, 1.0)])

    # route [2, 3] activates (2, 3), which A does not have
    monkeypatch.setattr(
        lkh_mod, '_do_lkh', lambda L, **kw: _fake_output(routes=[[0, 1], [2, 3]])
    )
    monkeypatch.setattr(lkh_mod, 'repair_routeset_path', lambda S, A, ringed=False: S)

    with caplog.at_level('WARNING'):
        S = lkh_mod.lkh3(A, capacity=2, time_limit=0.1, seed=1)

    assert '_linkbits' not in S.graph
    assert '_topology_id' not in S.graph
    assert '_linkset_id' not in S.graph
    assert 'left the available-links set' in caplog.text


def test_lkh3_complete_that_stays_within_A_is_identified_over_A(monkeypatch):
    """A complete solve using only links in A retains an identity over A."""
    A = _make_A(T=4, edges=[(0, 1, 1.0), (2, 3, 1.0)])

    monkeypatch.setattr(
        lkh_mod, '_do_lkh', lambda L, **kw: _fake_output(routes=[[0, 1], [2, 3]])
    )

    S = lkh_mod.lkh3(A, capacity=2, time_limit=0.1, seed=1, complete=True, repair=False)

    assert S.graph['method_options']['complete']
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
            A, capacity=2, time_limit=0.1, seed=1, complete=True, repair=False
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
        S = lkh_mod.lkh3(A, capacity=2, time_limit=0.1, seed=1, repair=False)

    assert S.graph['method_options']['complete']
    assert '_linkbits' not in S.graph
    assert 'left the available-links set' in caplog.text


def test_lkh3_complete_is_refused_together_with_repair():
    A = _make_A(T=4, edges=[(0, 1, 1.0), (2, 3, 1.0)])
    with pytest.raises(NotImplementedError, match='complete graph over all nodes'):
        lkh_mod.lkh3(A, capacity=2, time_limit=0.1, seed=1, complete=True)


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

    The fake process returns no solution, causing lkh3() to raise AssertionError
    when checking the result. Callers expect that error and inspect the files
    written before the subprocess call.
    """
    captured = {}

    def fake_run(argv, **kwargs):
        params = Path(argv[1]).read_text()
        captured['params'] = dict(
            line.split(' = ', 1) for line in params.splitlines() if ' = ' in line
        )
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

    with pytest.raises(AssertionError, match='root node load'):
        lkh_mod.lkh3(as_normalized(A), capacity=5, time_limit=0.1, repair=False)

    demands = _demand_section(captured['problem'])
    assert demands == {**{t + 1: 1 for t in range(T)}, T + 1: 0}
    assert captured['params']['MTSP_MAX_SIZE'] == '5'


def test_lkh3_declares_terminal_power_as_demand(monkeypatch):
    captured = _capture_lkh_problem(monkeypatch)
    A = as_normalized(get_bundle('toy').A.copy())
    T = A.graph['T']
    powers = {t: 1 + (t % 3) for t in range(T)}
    nx.set_node_attributes(A, powers, 'power')

    with pytest.raises(AssertionError, match='root node load'):
        lkh_mod.lkh3(A, capacity=8, time_limit=0.1, repair=False)

    demands = _demand_section(captured['problem'])
    assert demands == {**{t + 1: powers[t] for t in range(T)}, T + 1: 0}
    assert captured['problem'].splitlines()[0].startswith('NAME')
    # The smallest demand is one unit, allowing at most 8 nodes per route.
    assert captured['params']['MTSP_MAX_SIZE'] == '8'
    # Unequal demands disable the minimum node count.
    assert captured['params']['MTSP_MIN_SIZE'] == '0'
    # the feeder minimum follows the total power (24), not the terminal count
    assert int(captured['params']['VEHICLES']) >= math.ceil(sum(powers.values()) / 8)


def test_lkh3_rejects_non_unit_terminal_power_it_cannot_honour():
    """Unsupported modes are rejected before invoking the LKH executable."""
    A = as_normalized(get_bundle('toy').A.copy())
    nx.set_node_attributes(A, {0: 2}, 'power')

    with pytest.raises(NotImplementedError, match='single-root radial solve'):
        lkh_mod.lkh3(A, capacity=5, time_limit=1.0, ringed=True)
    with pytest.raises(NotImplementedError, match='single-root radial solve'):
        lkh_mod.lkh3(A, capacity=5, time_limit=1.0, balanced=True)

    Amr = as_normalized(get_bundle('neart').A.copy())
    nx.set_node_attributes(Amr, {0: 2}, 'power')
    with pytest.raises(NotImplementedError, match='single-root radial solve'):
        lkh_mod.lkh3(Amr, capacity=5, time_limit=1.0)
