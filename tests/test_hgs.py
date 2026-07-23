import math
import re

import networkx as nx
import numpy as np
import pytest

import optiwindnet.baselines.hgs as hgs_mod
from optiwindnet.baselines._core import remove_offending_crossings
from optiwindnet.baselines.hgs import _balanced_capacity
from optiwindnet.interarraylib import as_normalized
from optiwindnet.types import Topology

from .cases import HGS_CASES, case_node_id, expected_topology
from .helpers import terminal_terminal_crossings
from .producers import hgs_topology
from .sitecache import get_bundle
from .topology_assertions import assert_topology


@pytest.mark.parametrize('case', HGS_CASES, ids=case_node_id)
def test_hgs_real_topology_cases(case):
    """A compact seeded matrix validates raw HGS topology output."""
    A = get_bundle(case.site).A
    S = hgs_topology(case)
    assert_topology(S, expected_topology(case), case.capacity)
    assert terminal_terminal_crossings(S, A.graph['VertexC']) == []


@pytest.mark.parametrize('capacity', (1, 12))
def test_hgs_small_site_capacity_boundaries(capacity):
    A = get_bundle('example_location').A
    S = hgs_mod.hgs_cvrp(as_normalized(A), capacity=capacity, time_limit=0.2, seed=0)
    assert_topology(S, Topology.RADIAL, capacity)


@pytest.mark.parametrize('capacity', (2, 3, 5))
def test_hgs_ringed_capacity_boundaries(capacity):
    A = get_bundle('albatros').A
    S = hgs_mod.hgs_cvrp(
        as_normalized(A),
        capacity=capacity,
        time_limit=0.5,
        ringed=True,
        seed=0,
    )
    assert_topology(S, Topology.RINGED, capacity)


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


@pytest.mark.parametrize('capacity', range(2, 20))
def test_balanced_capacity_invariants_with_exact_vehicles(capacity):
    for T in range(1, 60):
        for vehicles in range(math.ceil(T / capacity), T + 1):
            capacity_eff, vehicles_out, num_slack = _balanced_capacity(
                T, capacity, vehicles
            )
            # the requested feeder count is honored ...
            assert vehicles_out == vehicles
            # ... without ever exceeding the requested capacity
            assert capacity_eff <= capacity
            # every terminal fits, and every route comes out exactly full
            assert vehicles * capacity_eff == T + num_slack
            # so no route can be left empty and the feeder count is pinned
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


def test_balanced_exact_vehicles_above_minimum(monkeypatch):
    T, capacity, vehicles = 12, 6, 4
    # exactly 4 feeders -> effective capacity 3, no slack node needed
    routes = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    captured = _capture_do_hgs(monkeypatch, routes)

    S = hgs_mod.hgs_cvrp(
        _make_A(T=T),
        capacity=capacity,
        time_limit=0.1,
        seed=1,
        vehicles=vehicles,
        vehicles_exact=True,
        balanced=True,
        repair=False,
    )

    assert captured['capacity'] == 3
    assert captured['vehicles'] == vehicles
    assert captured['n'] == T + 1  # depot + terminals, no slack node
    # the pinned feeder count is what the root ends up with
    assert S.degree[-1] == vehicles
    assert S.graph['capacity'] == capacity
    assert S.graph['method_options']['feeders_above_min'] == vehicles - math.ceil(
        T / capacity
    )
    assert S.graph['method_options']['feeders_exact']


def test_balanced_exact_vehicles_with_slack(monkeypatch):
    T, capacity, vehicles = 10, 6, 4
    # exactly 4 feeders -> effective capacity 3, 2 slack nodes (indices 11 and 12)
    routes = [[1, 2, 3], [4, 5, 6], [7, 8, 11], [9, 10, 12]]
    captured = _capture_do_hgs(monkeypatch, routes)

    S = hgs_mod.hgs_cvrp(
        _make_A(T=T),
        capacity=capacity,
        time_limit=0.1,
        seed=1,
        vehicles=vehicles,
        vehicles_exact=True,
        balanced=True,
        repair=False,
    )

    assert captured['capacity'] == 3
    assert captured['n'] == T + 1 + 2  # depot + terminals + 2 slack nodes
    # the slack nodes are stripped, but their routes remain feeders
    assert S.degree[-1] == vehicles
    assert S.nodes[-1]['load'] == T
    # balanced: loads differ at most by one unit
    assert sorted(S.nodes[t]['load'] for t in S.neighbors(-1)) == [2, 2, 3, 3]


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'match'),
    [
        (dict(vehicles=None), ValueError, 'requires `vehicles` to be set'),
        (dict(vehicles=3, balanced=False), NotImplementedError, '`balanced`=True'),
        (dict(vehicles=1), ValueError, 'below the minimum necessary'),
        (dict(vehicles=13), ValueError, 'above the number of terminals'),
    ],
)
def test_exact_vehicles_validation(kwargs, exception, match):
    kwargs = dict(balanced=True) | kwargs
    with pytest.raises(exception, match=re.escape(match)):
        hgs_mod.hgs_cvrp(
            _make_A(T=12),
            capacity=6,
            time_limit=0.1,
            vehicles_exact=True,
            repair=False,
            **kwargs,
        )


def test_exact_vehicles_rejected_above_minimum_for_multi_root():
    with pytest.raises(ValueError, match='minimum feasible value'):
        hgs_mod.hgs_cvrp(
            _make_A(T=12, R=2),
            capacity=6,
            time_limit=0.1,
            vehicles=3,
            vehicles_exact=True,
            balanced=True,
            repair=False,
        )


def test_balanced_rejects_non_minimum_vehicles_without_exact_flag():
    with pytest.raises(ValueError, match='vehicles_exact'):
        hgs_mod.hgs_cvrp(
            _make_A(T=12),
            capacity=6,
            time_limit=0.1,
            vehicles=4,
            balanced=True,
            repair=False,
        )


def test_unbalanced_solve_uses_requested_capacity(monkeypatch):
    captured = _capture_do_hgs(monkeypatch, [[1, 2], [3, 4]])

    S = hgs_mod.hgs_cvrp(
        _make_A(T=4), capacity=2, time_limit=0.1, seed=1, balanced=False, repair=False
    )

    assert captured['capacity'] == 2
    assert captured['n'] == 5  # no slack nodes
    assert 'capacity_effective' not in S.graph['solver_details']


def test_solution_time_falls_back_to_total_runtime():
    assert hgs_mod._solution_time('Summary total 1.25', objective=999.0) == 1.25


def test_unbalanced_vehicle_count_is_clamped_up(monkeypatch):
    captured = _capture_do_hgs(monkeypatch, [[1, 2], [3, 4]])

    hgs_mod.hgs_cvrp(
        _make_A(T=4),
        capacity=2,
        time_limit=0.1,
        seed=1,
        vehicles=1,
        repair=False,
    )

    assert captured['vehicles'] == 2


def test_multiroot_nonminimum_vehicle_count_is_reset(monkeypatch, caplog):
    def fake_multi_root(*args):
        inputs = (
            [1, 1],
            [6, 6],
            [np.array([-2, *range(6)]), np.array([-1, *range(6, 12)])],
            [0, 0],
            [6, 6],
        )
        output = ([[1, 2, 3, 4, 5, 6]], 0.1, 0.1, 1.0, '', {})
        return inputs, [output, output]

    monkeypatch.setattr(hgs_mod, '_solve_multi_root', fake_multi_root)
    with caplog.at_level('WARNING'):
        S = hgs_mod.hgs_cvrp(
            _make_A(T=12, R=2),
            capacity=6,
            time_limit=0.1,
            vehicles=3,
            seed=None,
            repair=False,
        )

    assert S.graph['method_options']['feeders_above_min'] == 0
    assert 'setting to the minimum' in caplog.text


def test_remove_offending_crossing_keeps_absent_diagonal_mapping():
    A = nx.Graph()
    A.add_edge(0, 1, length=1.0)
    A.add_edge(2, 3, length=1.0)

    remove_offending_crossings(A, {}, [((0, 1), (2, 3))])

    assert (0, 1) not in A.edges
