"""Conversions among the location, topology, routeset and encoded forms."""

import copy
import itertools
import pickle

import networkx as nx
import numpy as np
import pytest
from bitarray import bitarray, frozenbitarray

from optiwindnet.converting import (
    G_from_S,
    L_from_G,
    L_from_site,
    S_from_G,
    S_from_linkbits,
    S_from_terse_links,
    _rings_from_S,
    linkbits_from_S,
    terse_links_from_S,
)
from optiwindnet.loads import (
    _add_ring_to_S,
    calcload,
)
from optiwindnet.MILP import Topology
from optiwindnet.transforming import (
    as_normalized,
)

from .helpers import ring_sets, ringed_S, tiny_wfn
from .sitecache import get_bundle


@pytest.mark.parametrize('n', range(1, 13))
@pytest.mark.parametrize('bridging', (False, True), ids=('one-root', 'bridging'))
def test_private_rings_from_S_roundtrip(n, bridging):
    R = 2 if bridging else 1
    S = nx.Graph(R=R, T=n)
    S.add_nodes_from(range(-R, 0))
    roots = (-1, -2) if bridging else (-1, -1)
    _add_ring_to_S(S, roots, list(range(n)), subtree=0, A=None)

    recovered_roots, ordered = _rings_from_S(S)[0]
    assert set(recovered_roots) == set(roots)
    assert set(ordered) == set(range(n))


def test_linkbits_from_S_uses_canonical_edge_and_feeder_order():
    A = nx.Graph(T=3, R=2)
    A.add_edges_from(((2, 1), (2, 0), (1, 0)))
    A.graph['_canonical_terminal_links'] = np.array(
        ((0, 1), (0, 2), (1, 2)), dtype=np.uint32
    )
    S = nx.Graph(((2, 0), (-1, 1), (-2, 2)))

    linkbits = linkbits_from_S(A, S)

    assert linkbits == frozenbitarray('010000110')
    assert A.graph['_canonical_terminal_links'].tolist() == [[0, 1], [0, 2], [1, 2]]


def test_linkbits_from_S_complete_positions_every_terminal_pair():
    """``complete=True`` reads only R and T: the position is arithmetic."""
    A = nx.Graph(T=3, R=2)  # no links of its own, canonical or otherwise
    S = nx.Graph(((2, 0), (-1, 1), (-2, 2)))

    linkbits = linkbits_from_S(A, S, complete=True)

    # (0,1) (0,2) (1,2), then the R*T feeders in terminal-major order
    assert linkbits == frozenbitarray('010000110')
    assert '_canonical_terminal_links' not in A.graph


@pytest.mark.parametrize('T', (2, 3, 7, 40))
def test_linkbits_from_S_complete_matches_the_materialized_linkset(T):
    """The arithmetic positions are the ones triu_indices order would give."""
    A_complete = nx.Graph(T=T, R=1)
    A_complete.graph['_canonical_terminal_links'] = np.stack(
        np.triu_indices(T, k=1), axis=1
    ).astype(np.uint32)
    A = nx.Graph(T=T, R=1)
    S = nx.Graph([(t, t + 1) for t in range(T - 1)] + [(-1, 0)])

    assert linkbits_from_S(A, S, complete=True) == linkbits_from_S(A_complete, S)


def test_linkbits_from_S_complete_accepts_a_link_A_lacks():
    A = nx.Graph(T=3, R=1, _canonical_terminal_links=np.empty((0, 2), np.uint32))
    S = nx.Graph(((0, 1), (-1, 0), (-1, 2)))

    with pytest.raises(ValueError, match='terminal link absent from A'):
        linkbits_from_S(A, S)
    assert linkbits_from_S(A, S, complete=True) == frozenbitarray('100101')


def test_linkbits_from_S_requires_canonical_terminal_links():
    A = nx.Graph(T=3, R=1)
    S = nx.Graph(((0, 2), (-1, 1)))

    with pytest.raises(KeyError, match='_canonical_terminal_links'):
        linkbits_from_S(A, S)


def test_linkbits_from_S_empty_terminal_linkset_has_only_feeders():
    A = nx.Graph(T=3, R=1, _canonical_terminal_links=np.empty((0, 2), np.uint32))
    S = nx.Graph(((-1, 0), (-1, 1), (-1, 2)))

    assert linkbits_from_S(A, S) == frozenbitarray('111')
    S.add_edge(0, 1)
    with pytest.raises(ValueError, match='terminal link absent from A'):
        linkbits_from_S(A, S)


@pytest.mark.parametrize('bits_type', (bitarray, frozenbitarray))
@pytest.mark.parametrize('endian', ('big', 'little'))
def test_S_from_linkbits_uses_canonical_edge_and_feeder_order(bits_type, endian):
    A = nx.Graph(T=3, R=2)
    A.add_edges_from(((2, 1), (2, 0), (1, 0), (-1, 0), (0, 3)))
    A.graph['_canonical_terminal_links'] = np.array(
        ((0, 1), (0, 2), (1, 2)), dtype=np.uint32
    )
    bits = bits_type('010000110', endian=endian)

    S = S_from_linkbits(bits, A)

    assert {frozenset(edge) for edge in S.edges} == {
        frozenset(edge) for edge in ((0, 2), (-1, 1), (-2, 2))
    }
    assert set(S) == set(range(-2, 3))
    assert S.graph == {'T': 3, 'R': 2}
    assert all(not attrs for _, attrs in S.nodes(data=True))
    assert linkbits_from_S(A, S) == bits


@pytest.mark.parametrize('terminal_edges', ((), ((0, 2),), ((0, 1), (0, 2), (1, 2))))
def test_S_from_linkbits_roundtrip_every_link(terminal_edges):
    A = nx.Graph(T=3, R=2)
    A.add_edges_from(terminal_edges)
    A.graph['_canonical_terminal_links'] = np.array(
        terminal_edges, dtype=np.uint32
    ).reshape(-1, 2)
    nbits = len(terminal_edges) + 6
    for position in range(nbits):
        bits = bitarray(nbits)
        bits.setall(0)
        bits[position] = 1
        S = S_from_linkbits(bits, A)
        assert S.number_of_edges() == 1
        assert linkbits_from_S(A, S) == bits


@pytest.mark.parametrize('R, T', ((2, 3), (1, 1), (1, 0), (0, 0), (0, 3)))
def test_S_from_linkbits_preserves_isolated_nodes(R, T):
    A = nx.Graph(R=R, T=T)
    A.graph['_canonical_terminal_links'] = np.array(
        list(itertools.combinations(range(T), 2)), dtype=np.uint32
    ).reshape(-1, 2)
    bits = frozenbitarray('0' * (T * (T - 1) // 2 + R * T))
    S = S_from_linkbits(bits, A)
    assert set(S) == set(range(-R, T))
    assert S.number_of_edges() == 0
    assert linkbits_from_S(A, S) == bits


@pytest.mark.parametrize('nbits', (0, 5, 7))
def test_S_from_linkbits_rejects_wrong_bit_count(nbits):
    A = nx.Graph(T=3, R=1)
    A.graph['_canonical_terminal_links'] = np.array(
        ((0, 1), (0, 2), (1, 2)), dtype=np.uint32
    )
    with pytest.raises(ValueError, match=f'Expected 6 link bits for A, got {nbits}'):
        S_from_linkbits(frozenbitarray('0' * nbits), A)


def test_L_from_site():
    T = 3
    R = 2
    V = T + R
    VertexC = np.zeros((V, 2))  # coordinates don't matter for this test

    # 1) Call without explicit 'handle', 'name', or 'B'
    L = L_from_site(VertexC=VertexC, T=T, R=R)

    assert np.array_equal(L.graph['VertexC'], VertexC)
    assert L.graph['T'] == T
    assert L.graph['R'] == R
    assert L.graph['handle'] == 'L_from_site'
    assert L.graph['name'] == ''
    assert L.graph['B'] == 0
    assert len(L.nodes) == T + R

    # Node kinds and counts
    for n in range(T):
        assert L.nodes[n]['kind'] == 'wtg'
    for n in range(-R, 0):
        assert L.nodes[n]['kind'] == 'oss'

    # 2) Call with explicit handle, name and B
    border = np.ones((4, 2))
    obstacles = [np.zeros((4, 2)), np.ones((4, 2))]
    B = 12
    V = T + R + B
    VertexC = np.zeros((V, 2))  # coordinates don't matter for this test
    L2 = L_from_site(
        VertexC=VertexC,
        T=T,
        R=R,
        handle='test',
        name='TestSite',
        B=B,
        border=border,
        obstacles=obstacles,
    )

    assert L2.graph['handle'] == 'test'
    assert L2.graph['name'] == 'TestSite'
    assert L2.graph['B'] == 12
    assert np.array_equal(L2.graph['border'], border)
    assert len(L2.graph['obstacles']) == len(obstacles)
    assert all(np.array_equal(a, b) for a, b in zip(L2.graph['obstacles'], obstacles))
    assert len(L.nodes) == T + R

    # Node kinds and counts
    for n in range(T):
        assert L2.nodes[n]['kind'] == 'wtg'
    for n in range(-R, 0):
        assert L2.nodes[n]['kind'] == 'oss'
    assert len(L2.nodes) == T + R


def test_S_from_G():
    wfn = tiny_wfn()
    G = wfn.G

    def check_nodes(G, expected):
        actual = {node: G.nodes.get(node) for node, _ in expected}
        expected_dict = dict(expected)
        assert actual == expected_dict
        return True

    def check_edges(G, expected):
        actual = {}
        for u, v, _ in expected:
            actual[(u, v)] = G[u][v] if G.has_edge(u, v) else None
        expected_dict = {(u, v): edgeD for u, v, edgeD in expected}
        assert actual == expected_dict
        return True

    expected_nodes = [
        (-1, {'kind': 'oss', 'load': 4}),
        (0, {'kind': 'wtg', 'load': 4, 'subtree': 0}),
        (1, {'kind': 'wtg', 'load': 3, 'subtree': 0}),
        (2, {'kind': 'wtg', 'load': 2, 'subtree': 0}),
        (3, {'kind': 'wtg', 'load': 1, 'subtree': 0}),
    ]

    expected_edges = [
        (-1, 0, {'load': 4, 'reverse': False}),
        (0, 1, {'load': 3, 'reverse': False}),
        (1, 2, {'load': 2, 'reverse': False}),
        (2, 3, {'load': 1, 'reverse': False}),
    ]

    S = S_from_G(G)

    assert check_nodes(S, expected_nodes)
    assert check_edges(S, expected_edges)
    assert S.graph['max_load'] == 4

    # test other branches
    G.graph['has_loads'] = False
    G.graph.pop('creator')
    G.graph.pop('method_options')

    S2 = S_from_G(G)
    assert check_nodes(S2, expected_nodes)
    assert check_edges(S2, expected_edges)
    assert S2.graph['has_loads']
    assert 'creator' not in S2.graph
    assert 'method_options' not in S2.graph


def test_S_from_G_carries_terminal_power():
    """Preserving terminal power allows the topology to reproduce routed loads."""
    wfn = tiny_wfn()
    G = wfn.G
    powers = {0: 2, 2: 3}
    nx.set_node_attributes(G, powers, 'power')
    calcload(G)

    S = S_from_G(G)

    assert {t: S.nodes[t].get('power') for t in range(G.graph['T'])} == {
        0: 2, 1: None, 2: 3, 3: None,
    }  # fmt: skip
    # Recalculating loads preserves the values copied from G.
    reference = S.copy()
    calcload(reference)
    assert {n: d['load'] for n, d in reference.nodes(data=True)} == {
        n: d['load'] for n, d in S.nodes(data=True)
    }
    assert S.nodes[-1]['load'] == sum(powers.values()) + 2


def test_S_from_G_rejects_a_route_node_that_is_not_part_of_a_chain():
    G = nx.Graph(R=1, T=2, capacity=2, topology=Topology.BRANCHED)
    G.add_edges_from([(0, 2), (2, 1), (2, -1)])

    with pytest.raises(ValueError, match='route from -1 is not a chain at 2'):
        S_from_G(G)


def test_G_from_S():
    wfn = tiny_wfn()
    A = wfn.A
    S = wfn.S
    S.graph['_linkbits'] = frozenbitarray('1')

    # 1) basic test
    G = G_from_S(S, A)
    expected = [(0, 12), (-1, 0), (1, 13), (1, 2), (2, 3), (12, 13)]
    assert all(uv in G.edges for uv in expected)

    # No tentative/rogue
    assert 'tentative' not in G.graph or G.graph.get('tentative') == []
    assert 'rogue' not in G.graph
    assert G.graph['_linkbits'] is S.graph['_linkbits']

    # num_diagonals present
    assert 'num_diagonals' in G.graph
    assert G.graph['num_diagonals'] == 0

    # 2) normalized A
    A2 = as_normalized(A)
    G2 = G_from_S(S, A2)
    assert G2.graph['is_normalized']

    # shortcuts in A — work on copies to avoid polluting later sub-tests
    A_sc = copy.deepcopy(A)
    S_sc = copy.deepcopy(S)
    A_sc[0][2]['shortcuts'] = [9]
    A_sc[2][-1]['shortcuts'] = [9]
    S_sc.add_edge(0, 2, load=1, reverse=False)
    S_sc.add_edge(2, -1, load=1, reverse=False)
    G = G_from_S(S_sc, A_sc)

    assert (0, 2) in G.edges
    assert 'kind' not in G[0][2]
    assert (0, 2) in G.graph['shortened_contours']

    edges_to_test = [(0, 1), (0, 2), (0, 3), (-1, 2)]

    for s, t in edges_to_test:
        # Deep copy from the shortcut-mutated state for each iteration
        A_copy = copy.deepcopy(A_sc)
        S_copy = copy.deepcopy(S_sc)

        # Add only the current edge
        S_copy.add_edge(s, t, load=1, reverse=False)
        S_copy.nodes[s]['subtree'] = 0
        S_copy.nodes[t]['subtree'] = 0

        if (s, t) == (0, 2):
            A_copy[s][t]['shortcuts'] = [999]
        else:
            A_copy[s][t]['shortcuts'] = A_copy[s][t]['midpath'].copy()

        # Run G_from_S
        G = G_from_S(S_copy, A_copy)

        # Check edge exists
        assert (s, t) in G.edges

        # Fully shortened non-gates are realized as ordinary direct edges.
        expected_kind = None if s >= 0 else 'tentative'
        actual_kind = G[s][t].get('kind')
        assert actual_kind == expected_kind

    edges_to_test = [(1, 3), (-1, 1)]

    for s, t in edges_to_test:
        # Deep copy from the shortcut-mutated state for each iteration
        A_copy = copy.deepcopy(A_sc)
        S_copy = copy.deepcopy(S_sc)

        # Add only the current edge
        S_copy.add_edge(s, t, load=1, reverse=False)
        S_copy.nodes[s]['subtree'] = 0
        S_copy.nodes[t]['subtree'] = 0

        # Run G_from_S
        G = G_from_S(S_copy, A_copy)

        # Check edge exists
        assert (s, t) in G.edges
        expected_kind = 'rogue' if s >= 0 else 'tentative'
        actual_kind = G[s][t]['kind']
        assert actual_kind == expected_kind


def test_G_from_S_expands_a_contoured_ring_zero_load_link():
    A = get_bundle('borkum2').A
    T, R = (A.graph[key] for key in 'TR')
    u, v = next(
        (u, v)
        for u, v, midpath in A.edges(data='midpath')
        if 0 <= u < T and 0 <= v < T and midpath
    )
    root = -1
    S = nx.Graph(
        T=T,
        R=R,
        topology=Topology.RINGED,
        capacity=1,
        has_loads=True,
        creator='synthetic',
    )
    S.add_nodes_from(range(-R, 0), load=0)
    S.add_node(u, load=1, subtree=0)
    S.add_node(v, load=1, subtree=0)
    S.add_edge(root, u, load=1, reverse=False)
    S.add_edge(u, v, load=0, reverse=False)
    S.add_edge(v, root, load=1, reverse=False)

    G = G_from_S(S, A)

    assert any(
        data.get('kind') == 'contour' and data['load'] == 0
        for *_, data in G.edges(data=True)
    )


def test_L_from_G():
    G = tiny_wfn().G
    R = G.graph['R']
    T = G.graph['T']

    # 1) test basics
    L = L_from_G(G)
    # Check number of nodes
    assert all(n in L.nodes() for n in range(T)), 'WTG nodes missing'
    assert all(r in L.nodes() for r in range(-R, 0)), 'OSS nodes missing'

    # Check node attributes
    for n in range(T):
        assert L.nodes[n]['label'] == G.nodes[n].get('label')
        assert L.nodes[n]['kind'] == 'wtg'
    for r in range(-R, 0):
        assert L.nodes[r]['label'] == G.nodes[r].get('label')
        assert L.nodes[r]['kind'] == 'oss'

    # Check edges are not carried
    assert L.number_of_edges() == 0
    assert L.graph['VertexC'].shape[0] == len(G.graph['VertexC'])

    # 2) test num_stunts
    original_B = G.graph['B']
    original_len = len(G.graph['VertexC'])
    G.graph['num_stunts'] = 2
    G.graph['B'] = original_B + 2
    G.graph['VertexC'] = np.vstack(
        (
            G.graph['VertexC'][:-R],
            np.array([[10.0, 10.0], [20.0, 20.0]]),
            G.graph['VertexC'][-R:],
        )
    )
    L_no_stunts = L_from_G(G)
    assert L_no_stunts.number_of_edges() == 0
    assert L_no_stunts.graph['B'] == original_B
    assert L_no_stunts.graph['VertexC'].shape[0] == original_len

    # 3) test stunts_primes
    G_stunts = tiny_wfn().G
    G_stunts.graph['stunts_primes'] = [100, 101]
    L_stunts = L_from_G(G_stunts)
    assert L_stunts.number_of_edges() == 0
    # Check VertexC adjusted for stunts_primes
    assert L_stunts.graph['VertexC'].shape[0] == len(G_stunts.graph['VertexC']) - len(
        G_stunts.graph['stunts_primes']
    )


def test_S_from_terse_links():
    terse_links = np.array([-1, 0, 1, 2])

    def check_S(S, expected_capacity=None):
        # Check number of nodes
        assert len(S.nodes()) == len(terse_links) + 1  # +1 for root node
        assert S.graph['T'] == 4
        assert S.graph['R'] == 1

        # Check edges
        expected_edges = [(0, -1), (1, 0), (2, 1), (3, 2)]
        actual_edges = [(u, v) for u, v in S.edges()]
        for e in expected_edges:
            assert e in actual_edges or e[::-1] in actual_edges

        # Check capacity
        assert 'capacity' in S.graph
        if expected_capacity is None:
            assert S.graph['capacity'] == S.graph.get('max_load')
        else:
            assert S.graph['capacity'] == expected_capacity

    # Test without explicit capacity
    S1 = S_from_terse_links(terse_links)
    check_S(S1)

    # Test with explicit capacity
    S2 = S_from_terse_links(terse_links, capacity=5)
    check_S(S2, expected_capacity=5)


def test_terse_links_from_S():
    S = tiny_wfn().S
    expected_terse = np.array([-1, 0, 1, 2])
    actual_terse = terse_links_from_S(S)

    assert np.array_equal(actual_terse, expected_terse), (
        f'terse_links {actual_terse} != expected_terse_links {expected_terse}'
    )


# --------------------------------------------------------------------------- #
# terse_links for RINGED topologies (sequence-of-routes encoding)
# --------------------------------------------------------------------------- #
def test_terse_links_ringed_wire_format():
    """Each route is entered as its root number followed by its nodes.

    Every route -- including the first -- carries its own leading root number,
    which both ends the previous route and names this route's root.
    """
    S = ringed_S(1, [(-1, [0, 1, 2, 3]), (-1, [4, 5])])
    terse = terse_links_from_S(S)
    assert terse.tolist() == [-1, 0, 1, 2, 3, -1, 4, 5]


@pytest.mark.parametrize(
    'R, ringspec',
    [
        (1, [(-1, [0, 1, 2, 3])]),  # a single ring (no special-casing needed)
        (1, [(-1, [0, 1]), (-1, [2, 3]), (-1, [4, 5])]),  # many rings, one root
        (2, [(-1, [0, 1, 2, 3]), (-1, [4, 5]), (-2, [6, 7, 8])]),  # multi-root
        (2, [(-2, [0, 1, 2])]),  # only the second root is used
        (1, [(-1, [0, 1, 2]), (-1, [3])]),  # a real ring plus a lone-terminal stub
        (3, [(-1, [0, 1]), (-3, [2, 3, 4, 5]), (-3, [6])]),  # gap in roots used
    ],
)
def test_terse_links_ringed_roundtrip(R, ringspec):
    S = ringed_S(R, ringspec)
    T = S.graph['T']
    terse = terse_links_from_S(S)
    # the ringed encoding always outgrows the T-entry forest one, which is how
    # the two are told apart
    assert len(terse) > T
    S2 = S_from_terse_links(terse, R=R, T=T)
    assert ring_sets(S2) == ring_sets(S)
    # the encoding is a fixed point: re-encoding the decoded S is identical
    assert terse_links_from_S(S2).tolist() == terse.tolist()


def test_tagged_ringed_roundtrip_infers_dimensions():
    S = ringed_S(2, [(-1, [0, 1, 2]), (-2, [3, 4])])

    S2 = S_from_terse_links(terse_links_from_S(S))

    assert S2.graph['topology'] == Topology.RINGED
    assert S2.graph['R'] == S.graph['R']
    assert S2.graph['T'] == S.graph['T']
    assert ring_sets(S2) == ring_sets(S)


@pytest.mark.parametrize('longer, expected_zero_load_link', [(0, (0, 1)), (1, (1, 2))])
def test_terse_links_ringed_preserves_zero_load_link(longer, expected_zero_load_link):
    """An odd-terminal ring's zero-load link survives the round-trip losslessly.

    The two balanced split edges of an odd ring map onto the two walk
    directions, so the encoder orients the walk to reproduce the exact zero-load
    link that ``_add_ring_to_S`` chose from ``A`` -- without storing it.
    """
    A = nx.Graph()
    A.add_edge(0, 1, length=10.0 if longer == 0 else 1.0)
    A.add_edge(1, 2, length=10.0 if longer == 1 else 1.0)
    S = nx.Graph(R=1, T=3)
    S.add_node(-1)
    _add_ring_to_S(S, (-1, -1), [0, 1, 2], subtree=0, A=A)
    S.nodes[-1]['load'] = sum(S.nodes[n]['load'] for n in S[-1])

    terse = terse_links_from_S(S)
    S2 = S_from_terse_links(terse, R=1, T=3)
    zero_load_link = next(
        {u, v} for u, v, d in S2.edges(data=True) if d.get('load') == 0
    )
    assert zero_load_link == set(expected_zero_load_link)


def test_terse_links_forest_still_positional():
    """A radial/branched (forest) S keeps the positional one-entry-per-node form."""
    S = tiny_wfn().S
    terse = terse_links_from_S(S)
    assert len(terse) == S.graph['T']
    S2 = S_from_terse_links(terse)
    assert set(map(frozenset, S2.edges())) == set(map(frozenset, S.edges()))


@pytest.mark.parametrize(
    'topology, links',
    [
        (Topology.RADIAL, [(-1, 0), (0, 1), (1, 2)]),
        (Topology.BRANCHED, [(-1, 0), (0, 1), (0, 2)]),
    ],
)
def test_terse_links_carries_forest_architecture(topology, links):
    """Identical-length forest encodings retain their architecture metadata."""
    S = nx.Graph(R=1, T=3, topology=topology)
    S.add_edges_from(links)
    calcload(S)

    terse = terse_links_from_S(S)
    S2 = S_from_terse_links(terse, R=1, T=3)
    S3 = S_from_terse_links(np.asarray(terse), R=1, T=3, topology=topology)

    assert terse.topology is Topology(topology)
    assert S2.graph['topology'] is Topology(topology)
    assert S3.graph['topology'] is Topology(topology)
    assert set(map(frozenset, S2.edges())) == set(map(frozenset, S.edges()))


def test_terse_links_preserves_architecture_when_pickled():
    terse = terse_links_from_S(tiny_wfn().S)

    restored = pickle.loads(pickle.dumps(terse))

    assert restored.topology == terse.topology
    assert np.array_equal(restored, terse)


def test_terse_links_repr_shows_self_describing_value():
    terse = terse_links_from_S(tiny_wfn().S)

    assert terse.links == (-1, 0, 1, 2)
    assert terse.topology is Topology.BRANCHED
    assert terse.T == 4
    assert terse.R == 1
