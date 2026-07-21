"""Validation tests for the RINGED topology.

The RINGED topology is now available across every solver family:

* the shared ring builders :func:`add_ring_to_S` and
  :func:`split_rings_and_calc_loads`, plus :func:`rings_from_S` for recovering
  rings from their canonical two-arm-with-split shape;
* the constructive heuristic ``method='ringed'`` (and its public
  :class:`~optiwindnet.api.EWRouter` wrapper);
* the MILP RINGED path (``ModelOptions(topology='ringed')``), both at the
  low level and through :class:`~optiwindnet.api.MILPRouter`;
* the HGS-CVRP and LKH-3 closed-CVRP paths (``ringed=True``).

Because every backend canonicalises its output to the same shape, the same
structural invariants hold for all of them. They
are collected in :func:`_assert_canonical_ringed` and reused throughout so the
suite pins the *shared contract* of a ringed solution rather than each
backend's incidental output.

As in ``test_MILP.py``, ``ortools.math_opt`` bundles copies of HiGHS/SCIP that
collide with the standalone packages if loaded into the same process, so every
ortools solve is dispatched to the shared ``ortools_worker`` subprocess. This
module must therefore not import ortools (directly or transitively) at module
scope -- keep such imports inside the individual test/worker functions.
"""

import math

import networkx as nx
import pytest

from optiwindnet.heuristics import constructor
from optiwindnet.interarraylib import (
    G_from_S,
    calcload,
    S_from_terse_links,
    add_ring_to_S,
    assign_cables,
    rings_from_S,
    split_rings_and_calc_loads,
    terse_links_from_S,
    validate_routeset,
    validate_topology,
)
from optiwindnet.mesh import make_planar_embedding
from optiwindnet.pathfinding import PathFinder

from .helpers import has_cycle, solver_unavailable, terminal_terminal_crossings


# --------------------------------------------------------------------------- #
# Shared invariants of a canonical RINGED topology (backend-agnostic)
# --------------------------------------------------------------------------- #
def _rings(S):
    """Recover the ``(root, [t1, ..., tn])`` rings of a ringed topology ``S``."""
    return rings_from_S(S)


def _assert_canonical_ringed(S, capacity):
    """Assert the invariants shared by every canonical RINGED solution.

    A canonical ringed ``S`` is a set of rings sharing the ``R`` roots. Each
    ring of ``n >= 2`` terminals is two radial arms joined at their tails, with
    two real (load-bearing) feeders and exactly one load-0 open point; a lone
    terminal (``n == 1``) collapses to a single radial stub (one feeder, no
    split). Returns the recovered rings for further per-test checks.

    The invariants live in :func:`~optiwindnet.interarraylib.validate_topology`;
    this only pins that a *solved* ``S`` carries the loads they read, so they
    cannot pass vacuously.
    """
    assert S.graph['topology'] == 'ringed', 'S must declare its topology'
    assert S.graph.get('has_loads'), 'a solved S must carry its loads'
    assert validate_topology(S, capacity) == []
    return _rings(S)


# --------------------------------------------------------------------------- #
# Unit invariants of the shared ring builders (no solver involved)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('n', range(1, 13))
def test_add_ring_to_S_canonical_shape(n):
    """A ring built from an ordered terminal list is in canonical form."""
    S = nx.Graph(R=1, T=n)
    S.add_node(-1)
    add_ring_to_S(S, (-1, -1), list(range(n)), subtree=0, A=None)

    kinds = {d.get('kind') for _, _, d in S.edges(data=True)}
    assert kinds <= {None}, 'topology-graph ring edges must not carry a kind'

    feeders = [d['load'] for u, v, d in S.edges(data=True) if u < 0 or v < 0]
    splits = [(u, v) for u, v, d in S.edges(data=True) if d.get('load') == 0]
    max_load = max(d['load'] for _, _, d in S.edges(data=True))

    # every arm holds at most half of the doubled ring capacity
    m = math.ceil(n / 2)
    assert max_load == m

    # the two feeders (both real cables) carry the two arm loads; a lone
    # terminal collapses to a single radial-stub feeder
    if n == 1:
        assert feeders == [1] and not splits
    else:
        assert sorted(feeders) == [n - m, m]
        assert len(splits) == 1, 'exactly one open point per ring'
        (su, sv) = splits[0]
        assert S[su][sv]['load'] == 0, 'no current flows through the open point'

    # node loads of the two feeder terminals sum to the ring size
    assert sum(S.nodes[t]['load'] for t in S.neighbors(-1)) == n
    # all nodes of the ring share one subtree id
    assert {S.nodes[t]['subtree'] for t in range(n)} == {0}


@pytest.mark.parametrize('n', range(1, 13))
def test_rings_from_S_roundtrip(n):
    """rings_from_S recovers the terminal set of a built ring."""
    S = nx.Graph(R=1, T=n)
    S.add_node(-1)
    add_ring_to_S(S, (-1, -1), list(range(n)), subtree=0, A=None)
    rings = rings_from_S(S)
    assert len(rings) == 1
    roots, ordered = rings[0]
    assert roots == (-1, -1)
    assert set(ordered) == set(range(n))


def test_rings_from_S_multiple_rings_and_roots():
    """Two roots, several rings: each ring is recovered with its own root."""
    S = nx.Graph(R=2, T=9)
    S.add_nodes_from([-2, -1])
    add_ring_to_S(S, (-1, -1), [0, 1, 2, 3], subtree=0, A=None)
    add_ring_to_S(S, (-1, -1), [4, 5], subtree=1, A=None)
    add_ring_to_S(S, (-2, -2), [6, 7, 8], subtree=2, A=None)
    rings = rings_from_S(S)
    recovered = {(roots, frozenset(ordered)) for roots, ordered in rings}
    assert recovered == {
        ((-1, -1), frozenset({0, 1, 2, 3})),
        ((-1, -1), frozenset({4, 5})),
        ((-2, -2), frozenset({6, 7, 8})),
    }


# --------------------------------------------------------------------------- #
# Bridging rings (a ring whose two feeders land on different roots)
# --------------------------------------------------------------------------- #
def test_has_cycle_sees_through_the_substations():
    """A bridging ring closes a cycle; a radial forest does not.

    Interconnecting the substations is what makes the cycle visible: a ring
    bridging two of them is a path between two roots in ``S`` alone.
    """
    bridged = nx.Graph(R=2, T=4)
    bridged.add_nodes_from([-2, -1])
    add_ring_to_S(bridged, (-1, -2), [0, 1, 2, 3], subtree=0, A=None)
    assert nx.is_forest(bridged), 'S alone carries no cycle'
    assert has_cycle(bridged)

    # the two bridged roots need not be adjacent in the path linking them
    far = nx.Graph(R=3, T=1)
    far.add_nodes_from(range(-3, 0))
    add_ring_to_S(far, (-1, -3), [0], subtree=0, A=None)
    assert has_cycle(far)

    # a radial forest must not be mistaken for a ring
    radial = nx.Graph(R=3, T=6, topology='radial')
    radial.add_nodes_from(range(-3, 0))
    radial.add_edges_from([(-1, 0), (0, 1), (-2, 2), (2, 3), (-3, 4), (4, 5)])
    assert not has_cycle(radial)


@pytest.mark.parametrize('n', range(1, 13))
def test_rings_from_S_roundtrip_bridging(n):
    """A ring bridging two roots is recovered with both of its roots."""
    S = nx.Graph(R=2, T=n)
    S.add_nodes_from([-2, -1])
    add_ring_to_S(S, (-1, -2), list(range(n)), subtree=0, A=None)
    rings = rings_from_S(S)
    assert len(rings) == 1
    roots, ordered = rings[0]
    assert set(roots) == {-1, -2}
    assert set(ordered) == set(range(n))


@pytest.mark.parametrize('n', range(1, 13))
def test_terse_roundtrip_preserves_bridging_ring(n):
    """A bridging ring survives the terse encoding intact.

    The encoding is what the public ``terse_links`` API and the database share,
    so a ring closing on a root other than the one it opened on must round-trip
    with both feeders and its open point.
    """
    S = nx.Graph(R=2, T=n)
    S.add_nodes_from([-2, -1])
    add_ring_to_S(S, (-1, -2), list(range(n)), subtree=0, A=None)
    calcload(S)

    terse = terse_links_from_S(S)
    S2 = S_from_terse_links(terse, R=2, T=n)

    assert set(map(frozenset, S2.edges())) == set(map(frozenset, S.edges()))
    assert not validate_topology(S2, capacity=math.ceil(n / 2))
    # re-encoding is stable: the decoder lands on the encoder's walk orientation
    assert terse_links_from_S(S2).tolist() == terse.tolist()


@pytest.mark.parametrize('n', range(2, 12, 2))
def test_add_ring_to_S_even_nodes_default_split_is_balanced(n):
    """Without ``A``, an even-node ring (odd n) splits at the exact midpoint.

    ``m = ceil(n / 2)`` puts arm 1 at ceil and arm 2 at floor, so the two arm
    loads differ by exactly one for the balanced (no-``A``) placement.
    """
    S = nx.Graph(R=1, T=n)
    S.add_node(-1)
    add_ring_to_S(S, (-1, -1), list(range(n)), subtree=0, A=None)
    feeder_loads = sorted(d['load'] for u, v, d in S.edges(data=True) if u < 0 or v < 0)
    assert feeder_loads == [n // 2, math.ceil(n / 2)]


def test_add_ring_to_S_odd_n_uses_longer_split_edge_when_A_given():
    """On an even-node ring the open point is the longer of the two mid edges.

    n = 3 has a unique middle terminal (index 1) with two candidate split
    edges (0-1 and 1-2); with ``A`` supplied the *longer* one is opened.
    """
    for longer, expected in ((0, 1), (1, 2)):
        S = nx.Graph(R=1, T=3)
        S.add_node(-1)
        A = nx.Graph()
        A.add_edge(0, 1, length=10.0 if longer == 0 else 1.0)
        A.add_edge(1, 2, length=10.0 if longer == 1 else 1.0)
        add_ring_to_S(S, (-1, -1), [0, 1, 2], subtree=0, A=A)
        (su, sv) = [(u, v) for u, v, d in S.edges(data=True) if d.get('load') == 0][0]
        assert {su, sv} == {longer, expected}


# --------------------------------------------------------------------------- #
# split_rings_and_calc_loads: path-form -> canonical ring conversion
# (used by HGS/LKH/constructor)
#
# It closes each path-form arm into a ring, adding the load-0 open point. These
# path-form tests don't care which open point wins, so an empty ``A`` (no
# lengths) is enough to take the default open point.
# --------------------------------------------------------------------------- #
_CONSTRUCT = nx.Graph()  # empty available-links graph: take the default open point


def _path_form_S(R, paths):
    """Build a path-form topology: one non-branching path per (root, [terminals])."""
    T = sum(len(p) for _, p in paths)
    S = nx.Graph(R=R, T=T)
    S.add_nodes_from(range(-R, 0))
    for root, ordered in paths:
        S.add_edge(root, ordered[0])
        for a, b in zip(ordered, ordered[1:]):
            S.add_edge(a, b)
    return S


def test_split_rings_single_root_matches_canonical_form():
    """Non-branching paths become canonical rings within capacity."""
    S = _path_form_S(1, [(-1, [0, 1, 2]), (-1, [3, 4])])
    split_rings_and_calc_loads(S, _CONSTRUCT)

    rings = _assert_canonical_ringed(S, capacity=2)
    assert {frozenset(ordered) for _, ordered in rings} == {
        frozenset({0, 1, 2}),
        frozenset({3, 4}),
    }
    # graph-level bookkeeping is recomputed
    assert S.graph['has_loads'] is True
    assert S.graph['max_load'] == 2
    assert S.nodes[-1]['load'] == 5


def test_split_rings_multi_root_keeps_rings_with_their_root():
    """Each root's paths become rings still attached to that root."""
    S = _path_form_S(2, [(-2, [0, 1]), (-1, [2, 3, 4, 5]), (-1, [6])])
    split_rings_and_calc_loads(S, _CONSTRUCT)
    _assert_canonical_ringed(S, capacity=3)
    rings = _rings(S)
    by_root = {r: {frozenset(o) for rr, o in rings if rr == (r, r)} for r in (-2, -1)}
    assert by_root[-2] == {frozenset({0, 1})}
    assert by_root[-1] == {frozenset({2, 3, 4, 5}), frozenset({6})}


def test_split_rings_renumbers_subtrees_uniquely_per_ring():
    """After ring construction each ring is a single, distinct subtree id."""
    S = _path_form_S(1, [(-1, [0, 1]), (-1, [2, 3]), (-1, [4, 5])])
    split_rings_and_calc_loads(S, _CONSTRUCT)
    ring_subtrees = [
        {S.nodes[t]['subtree'] for t in ordered} for _, ordered in _rings(S)
    ]
    # each ring shares one subtree id, and the ids are distinct across rings
    assert all(len(ids) == 1 for ids in ring_subtrees)
    flat = [next(iter(ids)) for ids in ring_subtrees]
    assert len(set(flat)) == len(flat)


def test_split_rings_rejects_branching_subtree():
    """A branching (non-path) subtree cannot be ringified."""
    S = nx.Graph(R=1, T=3)
    S.add_node(-1)
    S.add_edge(-1, 0)
    S.add_edge(0, 1)
    S.add_edge(0, 2)  # branch at terminal 0
    with pytest.raises(ValueError):
        split_rings_and_calc_loads(S, _CONSTRUCT)


# --------------------------------------------------------------------------- #
# Constructive heuristic RINGED (method='ringed') -- runs in-process
# --------------------------------------------------------------------------- #
# id -> (repository location name, number of roots R)
_RINGED_MESHES = {
    'albatros_1ss': ('albatros', 1),  # T=16
    'neart_2ss': ('neart', 2),  # T=54 (exercises per-root ring cycles)
}
# multi-root subset, for tests whose assertion only makes sense with R >= 2
_MULTIROOT_RINGED_MESHES = {
    mesh_id: name for mesh_id, (name, R) in _RINGED_MESHES.items() if R >= 2
}


def _load_ringed_mesh(name, locations):
    """(P, A) for a repository location, built via its planar embedding."""
    return make_planar_embedding(getattr(locations, name))


@pytest.fixture(
    scope='session',
    params=[name for name, _ in _RINGED_MESHES.values()],
    ids=list(_RINGED_MESHES),
)
def ringed_mesh(request, locations):
    """(P, A) for a repository location, built once per location per session."""
    return _load_ringed_mesh(request.param, locations)


@pytest.fixture(
    scope='session',
    params=list(_MULTIROOT_RINGED_MESHES.values()),
    ids=list(_MULTIROOT_RINGED_MESHES),
)
def ringed_mesh_multiroot(request, locations):
    """(P, A) restricted to multi-substation sites (R >= 2)."""
    return _load_ringed_mesh(request.param, locations)


@pytest.mark.parametrize('capacity', (3, 5, 8))
def test_constructor_ringed_is_canonical(ringed_mesh, capacity):
    """The 'ringed' constructor output satisfies the shared ringed contract."""
    _, A = ringed_mesh
    S = constructor(A, capacity=capacity, method='ringed')
    rings = _assert_canonical_ringed(S, capacity)

    # a genuine ring, not a radial forest: at least one ring closes a cycle
    assert has_cycle(S)
    assert any(len(ordered) > 1 for _, ordered in rings)


@pytest.mark.parametrize('capacity', (3, 5, 8))
def test_constructor_ringed_no_terminal_terminal_crossings(ringed_mesh, capacity):
    _, A = ringed_mesh
    S = constructor(A, capacity=capacity, method='ringed')
    crossings = terminal_terminal_crossings(S, A.graph['VertexC'])
    assert crossings == [], f'ringed produced terminal-terminal crossings: {crossings}'


def test_constructor_ringed_graph_metadata(ringed_mesh):
    _, A = ringed_mesh
    S = constructor(A, capacity=5, method='ringed')
    assert S.graph['creator'] == 'constructor'
    assert S.graph['capacity'] == 5
    assert S.graph['method_options']['method'] == 'ringed'
    # 'ringed' grows simple-path subtrees, so it tracks insertions like radial_EW
    assert 'num_insertions' in S.graph


def test_constructor_ringed_multi_root_covers_every_root(ringed_mesh_multiroot):
    """On a multi-substation site every root carries at least one ring."""
    _, A = ringed_mesh_multiroot
    R = A.graph['R']
    S = constructor(A, capacity=5, method='ringed')
    _assert_canonical_ringed(S, capacity=5)
    roots_used = {
        r
        for root_spec, _ in _rings(S)
        for r in ((root_spec,) if isinstance(root_spec, int) else root_spec)
    }
    assert roots_used == set(range(-R, 0))


@pytest.mark.parametrize('bias_margin', (0.0, 0.1, 0.5))
def test_constructor_ringed_bias_margin_accepted(ringed_mesh, bias_margin):
    """The ringed-specific ``bias_margin`` window yields a valid ringed S."""
    _, A = ringed_mesh
    S = constructor(A, capacity=5, method='ringed', bias_margin=bias_margin)
    _assert_canonical_ringed(S, capacity=5)


@pytest.mark.parametrize('feeder_route', ('segmented', 'straight'))
def test_constructor_ringed_end_to_end_is_valid(ringed_mesh, feeder_route):
    """constructor(ringed) -> PathFinder -> validate_routeset == []."""
    P, A = ringed_mesh
    if feeder_route == 'straight':
        kwargs = dict(weigh_detours=False, straight_feeder_route=True)
    else:
        kwargs = dict(weigh_detours=True, straight_feeder_route=False)
    S = constructor(A, capacity=5, method='ringed', **kwargs)
    G = PathFinder(G_from_S(S, A), planar=P, A=A).create_detours()
    assign_cables(G, [(5, 1.0)])
    assert validate_routeset(G) == []
    # the routed graph keeps the ring open points and stays within capacity
    assert sum(d.get('load') == 0 for _, _, d in G.edges(data=True)) >= 1
    assert G.graph['max_load'] <= 5
    assert G.size(weight='length') > 0.0


def test_no_crossing_pathfinder_compacts_stunt_contour_clone(locations):
    """A stunt contour survives G_from_S and the no-crossing PathFinder path.

    Stunt vertices are dropped from the compacted ``VertexC`` that routesets
    keep, so a contour clone that runs through a stunt cannot address it.
    :func:`G_from_S` must therefore emit an ``fnT`` that already maps such a
    clone to the stunt's original prime -- otherwise every ``fnT`` consumer
    (``gateXing_iter``, ``svgplot``, ...) would index past ``VertexC``. On top
    of that, PathFinder must, even on its no-crossing early return, close the
    stunt-id gap left in the clone *node* numbering.

    borkum2's ringed constructor output always has genuine feeder crossings
    where a stunt contour is present, so we drive the no-crossing branch with a
    minimal single-subtree routeset over an A-edge whose contour runs through a
    stunt.
    """
    P, A = _load_ringed_mesh('borkum2', locations)
    R, T, B_A = (A.graph[k] for k in 'RTB')
    stunts_primes = A.graph['stunts_primes']
    stunt_nodes = set(range(T + B_A - len(stunts_primes), T + B_A))
    stunt_prime_set = set(stunts_primes)

    def minimal_routeset(gate, leaf):
        """A single two-terminal subtree: root -> gate -> leaf."""
        # a single path subtree is radial: the ringed mesh is what this test
        # needs (stunt contours), not a ringed routeset
        S = nx.Graph(
            R=R, T=T, capacity=T, creator='synthetic', has_loads=True, topology='radial'
        )
        S.add_node(-1, load=2)
        S.add_node(gate, load=2, subtree=0)
        S.add_node(leaf, load=1, subtree=0)
        S.add_edge(-1, gate, load=2, reverse=False)
        S.add_edge(gate, leaf, load=1, reverse=False)
        return S

    # Terminal-to-terminal A-edges whose contour passes through a stunt vertex.
    # Route each as a lone subtree (either endpoint as the gate) and keep the
    # first whose single feeder crosses nothing, so PathFinder takes its
    # no-crossing early return while a stunt contour clone is present.
    stunt_edges = [
        (u, v)
        for u, v, mp in A.edges(data='midpath')
        if 0 <= u < T and 0 <= v < T and mp and any(m in stunt_nodes for m in mp)
    ]
    for u, v in stunt_edges:
        for gate, leaf in ((u, v), (v, u)):
            G_tentative = G_from_S(minimal_routeset(gate, leaf), A)
            tentative_fnT = G_tentative.graph.get('fnT')
            if tentative_fnT is None:  # contour got fully shortened -> no clone
                continue
            # a stunt-derived clone resolves either to the stunt node (before the
            # G_from_S remap) or to the stunt's prime (after it) -- detect both so
            # the check does not presuppose the fix.
            stunt_contours = {
                n
                for n, data in G_tentative.nodes(data=True)
                if data.get('kind') == 'contour'
                and (
                    tentative_fnT[n] in stunt_nodes
                    or tentative_fnT[n] in stunt_prime_set
                )
            }
            if not stunt_contours:
                continue
            # G_from_S must map the clone within the compacted VertexC bounds
            nrows = G_tentative.graph['VertexC'].shape[0]
            assert all(tentative_fnT[n] < nrows for n in stunt_contours)
            pathfinder = PathFinder(G_tentative, planar=P, A=A)
            if not pathfinder.Xings:
                break
        else:
            continue
        break
    else:
        pytest.skip('no crossing-free stunt-contour routeset found for borkum2')

    assert pathfinder.Xings == []  # the no-crossing early return is taken
    G = pathfinder.create_detours()

    R, T, B = (G.graph[k] for k in 'RTB')
    C, D = G.graph.get('C', 0), G.graph.get('D', 0)
    fnT = G.graph['fnT']
    contour_nodes = {
        n for n, data in G.nodes(data=True) if data.get('kind') == 'contour'
    }

    assert C > 0  # the stunt contour clone survived
    assert len(fnT) == T + B + C + D + R
    # the stunt-id gap in the clone node numbering was closed
    assert contour_nodes == set(range(T + B, T + B + C))
    # every clone maps to an original constraint vertex, never a stunt node
    assert all(fnT[n] < T + B for n in contour_nodes)
    assert not any(fnT[n] in stunt_nodes for n in contour_nodes)
    assert not any(fnT[n] in stunt_nodes for n in contour_nodes)


# --------------------------------------------------------------------------- #
# EWRouter(method='ringed') through the public WindFarmNetwork API
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    'name', ('albatros', 'neart'), ids=('albatros_1ss', 'neart_2ss')
)
def test_ewrouter_ringed_end_to_end(name, locations):
    """WindFarmNetwork.optimize(EWRouter(method='ringed')) yields a ringed route."""
    from optiwindnet.api import EWRouter, WindFarmNetwork

    capacity = 3
    wfn = WindFarmNetwork(cables=capacity, L=getattr(locations, name))
    wfn.optimize(router=EWRouter(method='ringed'))
    S, G = wfn.S, wfn.G

    _assert_canonical_ringed(S, capacity)
    assert sum(d.get('load') == 0 for _, _, d in G.edges(data=True)) >= 1
    assert G.graph['max_load'] <= capacity
    assert G.size(weight='length') > 0.0


# --------------------------------------------------------------------------- #
# RINGED warmstart: a ringed solution maps onto the model's single-chain flow
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('bridging', [False, True], ids=['single_root', 'bridging'])
@pytest.mark.parametrize('n', range(2, 11))
def test_ringed_warmstart_links_flow_conservation(n, bridging):
    """warmstart_links yields a flow-feasible single-chain assignment.

    A ring whose two feeders land on different roots radializes the same way: it
    drains through the root feeding the head of the walk and closes on the other.
    Which of the two drains is arbitrary -- it moves no cable.
    """
    from types import SimpleNamespace

    from optiwindnet.MILP._core import Topology, warmstart_links

    terminals = list(range(n))
    R = 2 if bridging else 1
    roots = list(range(-R, 0))
    E = [(u, v) for u in terminals for v in terminals if u < v]
    Ep = [(v, u) for u, v in E]
    stars = [(t, r) for t in terminals for r in roots]
    starsp = [(r, t) for t in terminals for r in roots]
    # map each variable key to itself, so the "variable" warmstart_links yields
    # is that link's own key and the assignment can be reconstructed below
    link_ = {k: k for k in E + Ep + stars + starsp}
    flow_ = {k: k for k in E + Ep + stars}
    metadata = SimpleNamespace(
        R=R,
        link_=link_,
        flow_=flow_,
        model_options={'topology': Topology.RINGED},
    )

    S = nx.Graph(R=R, T=n)
    S.add_nodes_from(roots)
    add_ring_to_S(S, (-1, -2) if bridging else (-1, -1), terminals, subtree=0, A=None)

    # rebuild the complete assignment from the active-link stream
    link_vals = dict.fromkeys(link_, 0)
    flow_vals = dict.fromkeys(flow_, 0)
    for link_var, flow_var, flow in warmstart_links(metadata, S):
        link_vals[link_var] = 1
        if flow_var is not None:
            flow_vals[flow_var] = flow

    # exactly one flow feeder (t → r) carries the whole ring (n)
    active_feeders = [(t, r) for t in terminals for r in roots if link_vals[(t, r)]]
    assert len(active_feeders) == 1
    assert flow_vals[active_feeders[0]] == n
    # exactly one flowless closing feeder (r → t)
    assert sum(link_vals[(r, t)] for t in terminals for r in roots) == 1

    # in-/out-degree of every terminal is exactly 1, and flow is conserved
    for t in terminals:
        out_links = [k for k in link_vals if k[0] == t and link_vals[k]]
        in_links = [k for k in link_vals if k[1] == t and link_vals[k]]
        assert len(out_links) == 1 and len(in_links) == 1
        out_flow = sum(flow_vals.get(k, 0) for k in out_links)
        in_flow = sum(flow_vals.get(k, 0) for k in in_links)
        assert out_flow - in_flow == 1  # each terminal sinks one unit


@pytest.mark.parametrize('bridging', [False, True], ids=['single_root', 'bridging'])
@pytest.mark.parametrize('n', range(1, 11))
def test_ringed_mip_decoder_reuses_flow_tree_decoding(n, bridging):
    """The shared decoder closes a MILP flow path into a canonical ring."""
    from types import SimpleNamespace

    from optiwindnet.MILP._core import Solver, Topology

    head_root = -1
    close_root = -2 if bridging else head_root
    R = 2 if bridging else 1
    flow_vals = {(0, head_root): n}
    flow_vals.update({(t, t - 1): n - t for t in range(1, n)})
    closing_link = (close_root, n - 1)
    link_vals = dict.fromkeys((*flow_vals, closing_link), True)
    metadata = SimpleNamespace(
        R=R,
        T=n,
        capacity=math.ceil(n / 2),
        model_options={'topology': Topology.RINGED},
        link_=link_vals,
        flow_=flow_vals,
    )
    A = nx.path_graph(n)
    nx.set_edge_attributes(A, 1.0, 'length')
    if n == 3:
        A[0][1]['length'] = 2.0

    class FakeSolver:
        name = 'fake'

        @staticmethod
        def _link_val(value):
            return value

        @staticmethod
        def _flow_val(value):
            return value

    fake = FakeSolver()
    fake.A = A
    fake.metadata = metadata
    S = Solver._topology_from_mip_sol(fake)

    _assert_canonical_ringed(S, metadata.capacity)
    assert {S.nodes[t]['subtree'] for t in range(n)} == {0}
    assert S.graph['max_load'] == math.ceil(n / 2)
    assert (
        S.nodes[head_root]['load'] + (S.nodes[close_root]['load'] if bridging else 0)
        == n
    )
    if n == 3:
        open_point = next({u, v} for u, v, d in S.edges(data=True) if d['load'] == 0)
        assert open_point == {0, 1}


def test_ringed_warmstart_roundtrip_scip(locations):
    """A ringed solution warm-starts a fresh ringed scip model (accepted)."""
    from optiwindnet.api import WindFarmNetwork
    from optiwindnet.MILP import ModelOptions, solver_factory

    opts = ModelOptions(topology='ringed')
    wfn = WindFarmNetwork(cables=3, L=locations.albatros)
    try:
        seed = solver_factory('scip')
        seed.set_problem(P=wfn.P, A=wfn.A, capacity=3, model_options=opts)
        seed.solve(time_limit=10, mip_gap=0.05)
        S_warm, _ = seed.get_solution()
    except (FileNotFoundError, ModuleNotFoundError):
        pytest.skip('scip solver unavailable')

    solver = solver_factory('scip')
    # set_problem calls warmup_model; a rejected hint raises OWNWarmupFailed
    solver.set_problem(
        P=wfn.P, A=wfn.A, capacity=3, model_options=opts, warmstart=S_warm
    )
    assert solver.metadata.warmed_by == S_warm.graph['creator']


# --------------------------------------------------------------------------- #
# MILP RINGED optimum bracket on repository instances (via the ortools worker)
# --------------------------------------------------------------------------- #
def _solve_milp_ringed(L, capacity, time_limit, mip_gap):
    """Worker job: solve one repository instance with ortools.cp_sat RINGED.

    Runs in the ortools subprocess and returns only picklable primitives.
    """
    from optiwindnet.api import WindFarmNetwork
    from optiwindnet.interarraylib import assign_cables
    from optiwindnet.MILP import ModelOptions, solver_factory

    wfn = WindFarmNetwork(cables=capacity, L=L)
    solver = solver_factory('ortools.cp_sat')
    solver.set_problem(
        P=wfn.P,
        A=wfn.A,
        capacity=capacity,
        model_options=ModelOptions(topology='ringed'),
    )
    info = solver.solve(time_limit=time_limit, mip_gap=mip_gap)
    S, G = solver.get_solution()
    assign_cables(G, [(capacity, 1.0)])  # exercises the load-0 split-edge path

    R, T = S.graph['R'], S.graph['T']
    return dict(
        termination=info.termination,
        # the MILP bounds the *undetoured* cable length: info.objective is the
        # incumbent, info.bound the dual bound. G.length adds PathFinder detours
        # and is only used for the structural (length > 0) sanity check.
        bound=info.bound,
        objective=info.objective,
        length=G.size(weight='length'),
        kinds=sorted({str(d.get('kind')) for _, _, d in S.edges(data=True)}),
        max_edge_load=max(d['load'] for _, _, d in S.edges(data=True)),
        sum_root_load=sum(S.nodes[r]['load'] for r in range(-R, 0)),
        num_splits=sum(d.get('load') == 0 for _, _, d in S.edges(data=True)),
        T=T,
        R=R,
        g_edges=G.number_of_edges(),
    )


# (name, capacity, time_limit, mip_gap, optimum)
#
# ``optimum`` is the proven-optimal RINGED MILP objective (undetoured cable
# length, raw coordinate units) computed offline with gurobi. A short 3s solve
# need not prove optimality; instead we assert that this known optimum lies
# within the incumbent-to-dual-bound bracket the solver reports, i.e.
# ``bound <= optimum <= objective``. That validates the solver's bounds without
# depending on it reaching the optimum inside the tiny budget.
_MILP_CASES = [
    ('albatros', 3, 3, 1e-3, 23819.887438),
    ('riffgat', 7, 3, 1e-3, 19196.193672),
    ('galloper', 5, 3, 1e-3, 66214.835372),
]


@pytest.mark.parametrize('name, capacity, time_limit, mip_gap, optimum', _MILP_CASES)
def test_milp_ringed_optimum_bracket(
    name, capacity, time_limit, mip_gap, optimum, ortools_worker, locations
):
    res = ortools_worker.run(
        _solve_milp_ringed,
        (getattr(locations, name), capacity, time_limit, mip_gap),
        time_limit + 60,
    )
    if isinstance(res, BaseException):
        raise res

    assert res['termination'] in ('OPTIMAL', 'FEASIBLE')
    # canonical representation: topology-graph ring edges carry no kind (open
    # points are marked by load == 0, not by a kind)
    assert set(res['kinds']) <= {'None'}
    # every cable segment (arm) stays within the per-cable capacity
    assert res['max_edge_load'] <= capacity
    # all terminals are connected
    assert res['sum_root_load'] == res['T']
    # a ringed solution has at least one open point
    assert res['num_splits'] >= 1
    # routing produced a non-trivial network
    assert res['g_edges'] > res['T']
    assert res['length'] > 0.0
    # the known optimum must fall within the solver's [dual bound, incumbent]
    # bracket (with a small tolerance for floating-point / near-optimal edges).
    assert res['bound'] * (1 - 1e-3) <= optimum <= res['objective'] * (1 + 1e-3)


# --------------------------------------------------------------------------- #
# MILP RINGED on the in-process solvers -- no ortools involved
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('solver_name', ['highs', 'scip', 'gurobi'])
def test_milp_ringed_inprocess_solvers_are_canonical(solver_name, locations):
    """Each in-process MILP backend produces a canonical ringed S.

    These solvers never touch OR-Tools' native libraries, so they run directly
    in-process. The solve is loose (short budget, wide gap) because we assert
    the *structure* of a ringed solution, not its optimal length. Skips when the
    backend is missing or unlicensed.
    """
    from optiwindnet.api import WindFarmNetwork
    from optiwindnet.MILP import ModelOptions, solver_factory

    capacity = 3
    wfn = WindFarmNetwork(cables=capacity, L=locations.albatros)
    try:
        solver = solver_factory(solver_name)
        solver.set_problem(
            P=wfn.P,
            A=wfn.A,
            capacity=capacity,
            model_options=ModelOptions(topology='ringed'),
        )
        info = solver.solve(time_limit=30, mip_gap=0.2)
    except BaseException as exc:
        if solver_unavailable(exc):
            pytest.skip(f'{solver_name} solver unavailable: {exc}')
        raise

    # any termination that leaves a feasible incumbent is fine here: with a
    # wide mip_gap a solver legitimately stops at 'gaplimit' (or 'timelimit'),
    # not only 'optimal'/'feasible'. The real check is get_solution() below --
    # it raises if there is no incumbent, and _assert_canonical_ringed pins the
    # structure of whatever solution was returned.
    assert info.termination.lower() in (
        'optimal',
        'feasible',
        'gaplimit',
        'timelimit',
    )
    S, G = solver.get_solution()
    _assert_canonical_ringed(S, capacity)
    assign_cables(G, [(capacity, 1.0)])
    assert G.size(weight='length') > 0.0


def test_milprouter_ringed_end_to_end(locations):
    """MILPRouter(topology='ringed') via WindFarmNetwork yields a ringed route.

    Exercises the high-level warmstart+solve path (api.py: MILPRouter.route),
    including the ringed-solution warmstart, not just the low-level solver.
    """
    from optiwindnet.api import MILPRouter, WindFarmNetwork
    from optiwindnet.MILP import ModelOptions

    capacity = 3
    wfn = WindFarmNetwork(cables=capacity, L=locations.albatros)
    try:
        wfn.optimize(
            router=MILPRouter(
                solver_name='scip',
                model_options=ModelOptions(topology='ringed'),
                time_limit=30,
                mip_gap=0.2,
            )
        )
    except (FileNotFoundError, ModuleNotFoundError):
        pytest.skip('scip solver unavailable')

    _assert_canonical_ringed(wfn.S, capacity)
    G = wfn.G
    assert sum(d.get('load') == 0 for _, _, d in G.edges(data=True)) >= 1
    assert G.graph['max_load'] <= capacity
    assert G.size(weight='length') > 0.0


# --------------------------------------------------------------------------- #
# HGS-CVRP RINGED (closed CVRP) -- runs in-process (no ortools involved)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('capacity', (2, 3, 5))
def test_hgs_ringed_albatros(capacity, locations):
    pytest.importorskip('hybgensea')
    from optiwindnet.api import WindFarmNetwork, as_normalized
    from optiwindnet.baselines.hgs import hgs_cvrp

    L = locations.albatros
    wfn = WindFarmNetwork(cables=capacity, L=L)
    # structure-only test: a short budget still yields a feasible ringed
    # solution (HGS spends its whole time_limit, so keep it small)
    S = hgs_cvrp(
        as_normalized(wfn.A),
        capacity=capacity,
        time_limit=0.5,
        ringed=True,
        seed=0,
    )

    _assert_canonical_ringed(S, capacity)

    G = PathFinder(G_from_S(S, wfn.A), planar=wfn.P, A=wfn.A).create_detours()
    assign_cables(G, [(capacity, 1.0)])
    assert validate_routeset(G) == []
    assert G.size(weight='length') > 0.0


def test_lkh_ringed_albatros(locations):
    """LKH-3 ringed (closed CVRP) produces a valid ring set covering all terminals."""
    import shutil

    if shutil.which('LKH') is None:
        pytest.skip('LKH executable not on PATH')
    from optiwindnet.api import WindFarmNetwork, as_normalized
    from optiwindnet.baselines.lkh import lkh3

    capacity = 3
    L = locations.albatros
    wfn = WindFarmNetwork(cables=capacity, L=L)
    S = lkh3(
        as_normalized(wfn.A),
        capacity=capacity,
        time_limit=5,
        ringed=True,
        seed=0,
    )

    _assert_canonical_ringed(S, capacity)

    G = PathFinder(G_from_S(S, wfn.A), planar=wfn.P, A=wfn.A).create_detours()
    assign_cables(G, [(capacity, 1.0)])
    assert validate_routeset(G) == []
    assert G.size(weight='length') > 0.0


def test_hgs_router_ringed_end_to_end(locations):
    """HGSRouter(ringed=True) produces a valid ringed routeset via WindFarmNetwork."""
    pytest.importorskip('hybgensea')
    from optiwindnet.api import HGSRouter, WindFarmNetwork

    capacity = 3
    L = locations.albatros
    wfn = WindFarmNetwork(cables=capacity, L=L)
    wfn.optimize(router=HGSRouter(time_limit=0.5, ringed=True, seed=0))
    S, G = wfn.S, wfn.G

    _assert_canonical_ringed(S, capacity)
    # the routed graph carries the ring open points and stays within capacity
    assert sum(d.get('load') == 0 for _, _, d in G.edges(data=True)) >= 1
    assert G.graph['max_load'] <= capacity
    assert G.size(weight='length') > 0.0
