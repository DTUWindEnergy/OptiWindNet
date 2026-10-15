# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import itertools
import math
import re

import networkx as nx
import numpy as np
import pytest

import optiwindnet.baselines.hgs as hgs_mod
from optiwindnet.baselines._core import (
    balanced_capacity,
    remove_offending_crossings,
    routes_from_warmstart,
)
from optiwindnet.constraints import NetworkConstraints
from optiwindnet.converting import linkbits_from_S
from optiwindnet.identity import linkset_id, topology_id
from optiwindnet.transforming import as_normalized
from optiwindnet.types import OWNWarmupFailed, Topology

from .cases import (
    DEFAULT_BASELINE_TIME_LIMIT,
    HGS_CASES,
    case_node_id,
    expected_topology,
)
from .helpers import ringed_S, terminal_terminal_crossings
from .producers import hgs_topology
from .sitecache import get_bundle
from .topology_assertions import assert_topology


@pytest.mark.parametrize('case', HGS_CASES, ids=case_node_id)
def test_hgs_real_topology_cases(case):
    """A compact seeded matrix validates raw HGS topology output."""
    A = get_bundle(case.site).A
    S = hgs_topology(case)
    assert_topology(S, expected_topology(case), case.capacity)
    assert S.graph['_linkbits'] == linkbits_from_S(A, S)
    assert S.graph['_topology_id'] == topology_id(S.graph['_linkbits'])
    assert S.graph['_linkset_id'] == A.graph['_linkset_id']
    assert terminal_terminal_crossings(S, A.graph['VertexC']) == []


@pytest.mark.parametrize('capacity', (1, 12))
def test_hgs_small_site_capacity_boundaries(capacity):
    A = get_bundle('example_location').A
    S = hgs_mod.hgs_cvrp(
        as_normalized(A),
        capacity=capacity,
        time_limit=DEFAULT_BASELINE_TIME_LIMIT,
        seed=0,
    )
    assert_topology(S, Topology.RADIAL, capacity)


@pytest.mark.parametrize('capacity', (2, 3, 5))
def test_hgs_ringed_capacity_boundaries(capacity):
    A = get_bundle('albatros').A
    S = hgs_mod.hgs_cvrp(
        as_normalized(A),
        capacity=capacity,
        time_limit=0.5,
        seed=0,
        constraints=NetworkConstraints(topology='ringed'),
    )
    assert_topology(S, Topology.RINGED, capacity)
    assert S.graph['method_options']['ringed']


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


@pytest.mark.parametrize('capacity', range(2, 20))
def test_balanced_capacity_invariants(capacity):
    for T in range(1, 200):
        capacity_eff, vehicles, num_slack = balanced_capacity(T, capacity)
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
            capacity_eff, vehicles_out, num_slack = balanced_capacity(
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
    assert balanced_capacity(97, 12) == (11, 9, 2)
    # a single decrement is not enough here: capacity 11 still yields slack 9 >= 2
    assert balanced_capacity(13, 12) == (7, 2, 1)


def test_balanced_capacity_exact_fit_needs_no_slack():
    assert balanced_capacity(24, 12) == (12, 2, 0)


def test_balanced_capacity_empty_cluster():
    assert balanced_capacity(0, 12) == (12, 0, 0)


def _capture_do_hgs(monkeypatch, routes):
    captured = {}

    def fake_do_hgs(
        W,
        coordinates,
        vehicles,
        capacity,
        hgs_options,
        log_callback=None,
        demands=None,
        initial_routes=None,
    ):
        captured.update(
            vehicles=vehicles,
            capacity=capacity,
            n=coordinates.shape[1],
            W=W,
            demands=demands,
        )
        captured.setdefault('initial_routes', []).append(initial_routes)
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
        repair=False,
        constraints=NetworkConstraints(topology='radial', feeders='min', balanced=True),
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
        repair=False,
        constraints=NetworkConstraints(
            topology='radial', feeders=vehicles, balanced=True
        ),
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
    # the options that shape the solve are part of the stored method
    assert {
        key: S.graph['method_options'][key]
        for key in ('balanced', 'ringed', 'repair', 'max_retries')
    } == {'balanced': True, 'ringed': False, 'repair': False, 'max_retries': 10}


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
        repair=False,
        constraints=NetworkConstraints(
            topology='radial', feeders=vehicles, balanced=True
        ),
    )

    assert captured['capacity'] == 3
    assert captured['n'] == T + 1 + 2  # depot + terminals + 2 slack nodes
    # the slack nodes are stripped, but their routes remain feeders
    assert S.degree[-1] == vehicles
    assert S.nodes[-1]['load'] == T
    # balanced: loads differ at most by one unit
    assert sorted(S.nodes[t]['load'] for t in S.neighbors(-1)) == [2, 2, 3, 3]


@pytest.mark.parametrize(
    ('feeders', 'match'),
    [(1, 'below the minimum necessary'), (13, 'above the maximum possible (12)')],
)
def test_exact_feeder_count_validation(feeders, match):
    with pytest.raises(ValueError, match=re.escape(match)):
        hgs_mod.hgs_cvrp(
            _make_A(T=12),
            capacity=6,
            time_limit=0.1,
            constraints=NetworkConstraints(feeders=feeders, balanced=True),
            repair=False,
        )


@pytest.mark.parametrize('vehicles', (3, 5, 8))
def test_ringed_exact_vehicles_pins_ring_count(vehicles):
    # albatros: T=16, so capacity 3 needs at least 3 rings and allows at most 8
    A = get_bundle('albatros').A
    S = hgs_mod.hgs_cvrp(
        as_normalized(A),
        capacity=3,
        time_limit=0.5,
        seed=0,
        constraints=NetworkConstraints(
            topology='ringed', feeders=2 * vehicles, balanced=True
        ),
    )
    assert_topology(S, Topology.RINGED, 3)
    # two feeders per ring
    assert S.degree[-1] == 2 * vehicles
    # balanced: ring sizes differ by at most one terminal
    ring_sizes = np.bincount([S.nodes[t]['subtree'] for t in range(16)])
    assert len(ring_sizes) == vehicles
    assert ring_sizes.max() - ring_sizes.min() <= 1


def test_ringed_exact_vehicles_rejects_rings_below_two_terminals():
    with pytest.raises(ValueError, match=re.escape('above the maximum (6)')):
        hgs_mod.hgs_cvrp(
            _make_A(T=12),
            capacity=6,
            time_limit=0.1,
            repair=False,
            constraints=NetworkConstraints(
                topology='ringed', feeders=14, balanced=True
            ),
        )


def test_unbalanced_solve_uses_requested_capacity(monkeypatch):
    captured = _capture_do_hgs(monkeypatch, [[1, 2], [3, 4]])

    S = hgs_mod.hgs_cvrp(
        _make_A(T=4),
        capacity=2,
        time_limit=0.1,
        seed=1,
        repair=False,
    )

    assert captured['capacity'] == 2
    assert captured['n'] == 5  # no slack nodes
    # The worker supplies unit demands when no array is passed.
    assert captured['demands'] is None
    assert 'capacity_effective' not in S.graph['solver_details']


def test_hgs_edgeless_A_solves_complete_and_is_not_identified(monkeypatch, caplog):
    _capture_do_hgs(monkeypatch, [[1, 2], [3, 4]])
    A = _make_A(T=4)

    with caplog.at_level('WARNING'):
        S = hgs_mod.hgs_cvrp(A, capacity=2, time_limit=0.1, seed=1, repair=False)

    assert A.number_of_edges() == 0
    assert S.graph['method_options']['complete']
    # the solution's links are not in A, so they have no bit position
    assert '_linkbits' not in S.graph
    assert '_topology_id' not in S.graph
    assert '_linkset_id' not in S.graph
    assert 'left the available-links set' in caplog.text


def test_hgs_complete_is_refused_together_with_repair():
    A = _make_A(T=4)
    with pytest.raises(NotImplementedError, match='complete graph over all nodes'):
        hgs_mod.hgs_cvrp(A, capacity=2, time_limit=0.1, seed=1)


def test_solution_time_falls_back_to_total_runtime():
    assert hgs_mod._solution_time('Summary total 1.25', objective=999.0) == 1.25


def test_vehicles_below_the_minimum_raise():
    with pytest.raises(ValueError, match=re.escape('below the minimum necessary (2)')):
        hgs_mod.hgs_cvrp(
            _make_A(T=4),
            capacity=2,
            time_limit=0.1,
            seed=1,
            repair=False,
            constraints=NetworkConstraints(topology='radial', max_feeders=1),
        )


def test_remove_offending_crossing_keeps_absent_diagonal_mapping():
    A = nx.Graph()
    A.add_edge(0, 1, length=1.0)
    A.add_edge(2, 3, length=1.0)
    A.graph.update(R=1, T=4)
    A.graph['_canonical_terminal_links'] = np.array(((0, 1), (2, 3)), np.uint32)
    A.graph['_linkset_id'] = linkset_id(A)

    remove_offending_crossings(A, {}, [((0, 1), (2, 3))])

    assert (0, 1) not in A.edges
    assert '_canonical_terminal_links' not in A.graph
    assert '_linkset_id' not in A.graph


def test_hgs_honours_non_unit_terminal_inflow_on_a_single_root_radial_solve():
    """HGS-CVRP respects declared terminal inflow when filling each route."""
    from optiwindnet.validating import validate_topology

    A = as_normalized(get_bundle('toy').A.copy())
    inflow = {t: 1 + (t % 3) for t in range(A.graph['T'])}
    nx.set_node_attributes(A, inflow, 'inflow')

    S = hgs_mod.hgs_cvrp(A, capacity=8, time_limit=DEFAULT_BASELINE_TIME_LIMIT, seed=1)

    subtree_loads = [S.nodes[t]['load'] for t in S.neighbors(-1)]
    assert max(subtree_loads) <= 8  # capacity is a inflow budget, not a count
    assert sum(subtree_loads) == sum(inflow.values())
    # the solution carries the inflow its loads were derived from
    assert {t: S.nodes[t].get('inflow') for t in (0, 1, 2)} == {0: None, 1: 2, 2: 3}
    assert validate_topology(S, 8) == []


@pytest.mark.parametrize(
    ('site', 'constraints'),
    (
        ('toy', {'topology': 'ringed'}),
        ('toy', {'feeders': 2, 'balanced': True}),
        ('neart', {}),  # multi-root: clusterize() partitions by terminal count
    ),
)
def test_hgs_rejects_non_unit_terminal_inflow_it_cannot_honour(site, constraints):
    A = as_normalized(get_bundle(site).A.copy())
    nx.set_node_attributes(A, {0: 2}, 'inflow')

    with pytest.raises(NotImplementedError, match='single-root radial solve'):
        hgs_mod.hgs_cvrp(
            A,
            capacity=8,
            time_limit=DEFAULT_BASELINE_TIME_LIMIT,
            constraints=NetworkConstraints(**constraints),
        )


def _radial_S(R: int, T: int, routes_: dict[int, list[list[int]]]) -> nx.Graph:
    """Path-form solution topology: ``{root: [route, ...]}``, routes from the gate."""
    S = nx.Graph(R=R, T=T, topology=Topology.RADIAL)
    S.add_nodes_from(range(-R, T))
    for r, routes in routes_.items():
        for route in routes:
            nx.add_path(S, [r, *route])
    return S


def test_routes_from_warmstart_walks_paths_and_branches_from_the_gate():
    S = _radial_S(1, 5, {-1: [[1, 0], [2, 3]]})
    S.add_edge(2, 4)  # branch: 2 -> {3, 4}

    routes_ = routes_from_warmstart(S, R=1, T=5)

    assert len(routes_) == 1
    path, branched = sorted(routes_[0])
    assert path == [1, 0]
    # depth first from the gate, whichever branch comes first
    assert branched[0] == 2
    assert sorted(branched) == [2, 3, 4]


def test_routes_from_warmstart_walks_a_ring_from_gate_to_gate():
    S = ringed_S(1, [(-1, [3, 1, 0, 2])])

    routes_ = routes_from_warmstart(S, R=1, T=4)

    assert len(routes_[0]) == 1
    route = routes_[0][0]
    # a single route around the ring, starting and ending at its two gates
    assert sorted(route) == [0, 1, 2, 3]
    assert {route[0], route[-1]} == set(S.neighbors(-1))
    assert all(S.has_edge(u, v) for u, v in itertools.pairwise(route))


def test_routes_from_warmstart_groups_routes_by_root():
    S = _radial_S(2, 4, {-2: [[3], [0]], -1: [[2, 1]]})

    routes_ = routes_from_warmstart(S, R=2, T=4)

    assert sorted(routes_[0]) == [[0], [3]]
    assert routes_[1] == [[2, 1]]


@pytest.mark.parametrize(
    ('mutate', 'match'),
    [
        (lambda S: S.remove_edge(1, 2), 'leaves 1 terminals out'),
        (lambda S: S.add_edge(2, -2), 'reaches root'),
        (lambda S: S.add_edge(-2, -1), 'links root'),
        (lambda S: S.add_edge(2, 4), 'neither terminals nor roots'),
        (lambda S: S.graph.update(T=5), 'but the problem has'),
    ],
)
def test_routes_from_warmstart_rejects_unusable_warmstart(mutate, match):
    S = _radial_S(2, 4, {-2: [[0, 3]], -1: [[1, 2]]})
    mutate(S)

    with pytest.raises(OWNWarmupFailed, match=match):
        routes_from_warmstart(S, R=2, T=4)


def test_initial_routes_refer_to_distance_matrix_rows():
    n_from_i = np.array([-2, 10, 20, 30])

    assert hgs_mod._initial_routes([[30, 10], [20]], n_from_i, 0, 3) == [
        [3, 1],
        [2],
    ]


def test_initial_routes_open_balanced_routes_with_slack_nodes():
    # T=5, effective capacity 3: 2 routes, 1 slack node (row 6)
    n_from_i = np.array([-1, 0, 1, 2, 3, 4, -1])

    initial = hgs_mod._initial_routes([[0, 1, 2], [3, 4]], n_from_i, 1, 3)

    assert initial == [[1, 2, 3], [6, 4, 5]]


def test_initial_routes_reject_balanced_route_count_mismatch():
    n_from_i = np.array([-1, 0, 1, 2, 3, 4, -1])

    with pytest.raises(
        OWNWarmupFailed, match='needs 2 routes, but the warm start has 3'
    ):
        hgs_mod._initial_routes([[0, 1], [2], [3, 4]], n_from_i, 1, 3)


def test_hgs_seeds_only_the_first_solve_of_the_repair_loop(monkeypatch):
    captured = _capture_do_hgs(monkeypatch, [[1, 2], [3, 4]])
    calls = []

    def fake_repair(S, A, ringed=False):
        calls.append(None)
        S = S.copy()
        if len(calls) == 1:
            S.graph['outstanding_crossings'] = [((0, 1), (2, 3))]
        return S

    monkeypatch.setattr(hgs_mod, 'repair_routeset_path', fake_repair)
    monkeypatch.setattr(hgs_mod, 'remove_offending_crossings', lambda *args: None)
    A = _make_A(T=4, edges=[(0, 1, 1.0), (2, 3, 1.0), (1, 2, 1.0)])
    warmstart = _radial_S(1, 4, {-1: [[3, 2], [0, 1]]})

    hgs_mod.hgs_cvrp(A, capacity=2, time_limit=0.1, seed=1, warmstart=warmstart)

    assert captured['initial_routes'] == [[[4, 3], [1, 2]], None]


def test_hgs_clusters_multi_root_solve_by_the_warmstart(monkeypatch):
    def echo_do_hgs(W, coordinates, vehicles, capacity, hgs_options, log_callback=None,
                    demands=None, initial_routes=None):  # fmt: skip
        return (initial_routes, 0.01, 0.0, 1.0, '', {})

    monkeypatch.setattr(hgs_mod, '_do_hgs', echo_do_hgs)
    monkeypatch.setattr(hgs_mod, 'cluster_plan', lambda *args: pytest.fail())
    # every terminal is closest to root -1 (d2roots is all ones: argmin picks -2)
    warmstart = _radial_S(2, 4, {-2: [[3]], -1: [[0, 2, 1]]})

    S = hgs_mod.hgs_cvrp(
        _make_A(T=4, R=2),
        capacity=3,
        time_limit=0.1,
        seed=1,
        repair=False,
        warmstart=warmstart,
    )

    assert {frozenset(e) for e in S.edges} == {frozenset(e) for e in warmstart.edges}


def test_hgs_rejects_warmstart_clustering_off_the_exact_vehicle_count():
    # T=4 at capacity 2 needs 2 routes, but clusters of 1 and 3 terminals need 3
    warmstart = _radial_S(2, 4, {-2: [[3]], -1: [[0, 2], [1]]})

    with pytest.raises(OWNWarmupFailed, match='3 routes, outside the range of 2 to 2'):
        hgs_mod.hgs_cvrp(
            _make_A(T=4, R=2),
            capacity=2,
            time_limit=0.1,
            repair=False,
            warmstart=warmstart,
            constraints=NetworkConstraints(topology='radial', feeders=2, balanced=True),
        )


def test_hgs_raises_on_warmstart_hgs_cvrp_rejects():
    A = as_normalized(get_bundle('example_location').A)
    T = A.graph['T']
    # a single route holding every terminal exceeds the capacity
    warmstart = _radial_S(1, T, {-1: [list(range(T))]})

    with pytest.raises(OWNWarmupFailed, match='exceeds the vehicle capacity'):
        hgs_mod.hgs_cvrp(A, capacity=3, time_limit=0.1, seed=0, warmstart=warmstart)


@pytest.mark.parametrize(
    ('site', 'constraints'),
    [
        ('example_location', NetworkConstraints()),
        ('albatros', NetworkConstraints(topology='ringed')),
        ('albatros', NetworkConstraints(feeders='min', balanced=True)),
        ('neart', NetworkConstraints()),
    ],
)
def test_hgs_warmstart_never_ends_worse_than_its_seed(site, constraints):
    A = as_normalized(get_bundle(site).A)
    S0 = hgs_mod.hgs_cvrp(
        A, capacity=4, time_limit=0.2, seed=0, repair=False, constraints=constraints
    )

    S1 = hgs_mod.hgs_cvrp(
        A,
        capacity=4,
        time_limit=0.1,
        seed=1,
        repair=False,
        warmstart=S0,
        constraints=constraints,
    )

    assert S1.graph['objective'] <= S0.graph['objective'] + 1e-9
    # the warm start's root assignment is kept
    assert [set(nx.descendants(S1, r) - {r}) for r in range(-A.graph['R'], 0)] == [
        set(nx.descendants(S0, r) - {r}) for r in range(-A.graph['R'], 0)
    ]


# --------------------------------------------------------------------------
# bounds
# --------------------------------------------------------------------------
def _neart_A_and_constraints(**kwargs):
    A = as_normalized(get_bundle('neart', read_powers=False).A)
    return A, NetworkConstraints(**kwargs)


def _per_root(S):
    return [(S.degree[r], S.nodes[r]['load']) for r in range(-S.graph['R'], 0)]


@pytest.mark.parametrize(
    ('constraints', 'expected'),
    [
        # unlimited, neart's root -1 takes 6 feeders and 34 turbines
        (
            {'max_substation_feeders': (None, 4), 'max_substation_loads': (None, 26)},
            lambda feeders, load: feeders <= 4 and load <= 26,
        ),
        ({'substation_feeders': (None, 4)}, lambda feeders, load: feeders == 4),
    ],
)
def test_hgs_keeps_the_substation_limits(constraints, expected):
    from optiwindnet.validating import validate_topology

    A, constraints = _neart_A_and_constraints(**constraints)

    S = hgs_mod.hgs_cvrp(A, capacity=7, time_limit=0.5, seed=1, constraints=constraints)

    assert expected(*_per_root(S)[1])
    assert validate_topology(S, 7) == []
    assert S.graph['solver_details']['root_feeders_ub'][0] is None


def test_hgs_balances_the_subtrees_of_every_substation_alike():
    A, constraints = _neart_A_and_constraints(feeders='min', balanced=True)
    bounds = constraints.resolve(A, capacity=7)
    load_lb, load_ub = bounds.group_load_lb, bounds.group_load_ub

    S = hgs_mod.hgs_cvrp(A, capacity=7, time_limit=0.5, seed=1, constraints=constraints)

    loads = [S.nodes[n]['load'] for r in (-2, -1) for n in S.neighbors(r)]
    assert len(loads) == bounds.groups_lb
    assert all(load_lb <= load <= load_ub for load in loads)


def test_hgs_cvrp_does_not_keep_feeders_straight():
    constraints = NetworkConstraints(feeder_route='straight')
    with pytest.raises(NotImplementedError, match='straight'):
        hgs_mod.hgs_cvrp(
            _make_A(T=4),
            capacity=2,
            time_limit=0.1,
            constraints=constraints,
            repair=False,
        )
