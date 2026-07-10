import math

import networkx as nx
import numpy as np
import pytest

import optiwindnet.baselines.hgs as hgs_mod
from optiwindnet.baselines.hgs import _balanced_capacity


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


@pytest.mark.parametrize('capacity', range(2, 20))
def test_balanced_capacity_invariants(capacity):
    for T in range(1, 200):
        capacity_eff, vehicles, num_slack = _balanced_capacity(T, capacity)
        # the requested capacity is never exceeded
        assert capacity_eff <= capacity
        # the feeder count stays at the minimum for the requested capacity
        assert vehicles == math.ceil(T / capacity)
        # every terminal fits, and every route comes out exactly full
        assert vehicles * capacity_eff == T + num_slack
        # a slack node is only reachable from the depot, so it must start its
        # own route: there can never be more slack nodes than routes
        assert num_slack < vehicles


def test_balanced_capacity_shrinks_when_slack_would_exceed_routes():
    # T=97, capacity=12 -> 9 vehicles but 11 slack nodes under the naive formula
    assert 9 * 12 - 97 >= math.ceil(97 / 12)
    assert _balanced_capacity(97, 12) == (11, 9, 2)
    # a single decrement is not enough here: capacity 11 still yields slack 9 >= 2
    assert _balanced_capacity(13, 12) == (7, 2, 1)


def test_balanced_capacity_exact_fit_needs_no_slack():
    assert _balanced_capacity(24, 12) == (12, 2, 0)


def test_balanced_capacity_empty_cluster():
    assert _balanced_capacity(0, 12) == (12, 0, 0)


def _capture_do_hgs(monkeypatch, routes):
    captured = {}

    def fake_do_hgs(W, coordinates, vehicles, capacity, hgs_options, log_callback=None):
        captured.update(
            vehicles=vehicles, capacity=capacity, n=coordinates.shape[1], W=W
        )
        return (routes, 0.01, 0.0, 1.0, '', {})

    monkeypatch.setattr(hgs_mod, '_do_hgs', fake_do_hgs)
    return captured


def test_balanced_solve_passes_effective_capacity_and_keeps_requested_one(monkeypatch):
    T, capacity = 13, 12
    # effective capacity 7, 2 vehicles, 1 slack node (index 13 in the matrix)
    routes = [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]]
    captured = _capture_do_hgs(monkeypatch, routes)

    S = hgs_mod.hgs_cvrp(
        _make_A(T=T),
        capacity=capacity,
        time_limit=0.1,
        seed=1,
        balanced=True,
        repair=False,
    )

    # the solver is driven with the shrunken capacity ...
    assert captured['capacity'] == 7
    assert captured['vehicles'] == 2
    assert captured['n'] == T + 1 + 1  # depot + terminals + 1 slack node
    # ... but the solution carries the capacity the caller asked for
    assert S.graph['capacity'] == capacity
    assert S.graph['solver_details']['capacity_effective'] == 7
    # the slack node is stripped, so loads sum back to T
    assert S.nodes[-1]['load'] == T
    assert S.graph['max_load'] <= capacity


def test_unbalanced_solve_uses_requested_capacity(monkeypatch):
    captured = _capture_do_hgs(monkeypatch, [[1, 2], [3, 4]])

    S = hgs_mod.hgs_cvrp(
        _make_A(T=4), capacity=2, time_limit=0.1, seed=1, balanced=False, repair=False
    )

    assert captured['capacity'] == 2
    assert captured['n'] == 5  # no slack nodes
    assert 'capacity_effective' not in S.graph['solver_details']
