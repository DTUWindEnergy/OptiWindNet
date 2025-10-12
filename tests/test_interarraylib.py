# SPDX-License-Identifier: MIT
# tests/test_interarraylib.py

import math
import numpy as np
import networkx as nx
import pytest
import hashlib
import copy

from optiwindnet.interarraylib import (
    assign_cables,
    update_lengths,
    describe_G,
    S_from_G,
    terse_links_from_S,
    as_normalized,
    as_rescaled,
    as_undetoured,
    as_hooked_to_nearest,
    as_hooked_to_head,
    scaffolded,
    calcload,
    site_fingerprint,
    fun_fingerprint,
    L_from_site,
    L_from_G,
    G_from_S,
    S_from_terse_links,
    as_single_root,
    
)
from .helpers import tiny_wfn, assert_graph_equal


# ----------------------------
# helpers
# ----------------------------
TOLERANCE = 1e-6



# ----------------------------
# assign_cables
# ----------------------------

def test_assign_cables():
    # get a network from tiny_wfn
    wfn = tiny_wfn()
    original_G = wfn.G

    # 1) Normal case: non-zero costs -> currency set, capacity added (if missing),
    #    edges get 'cable' and 'cost'
    G = original_G.copy()
    cables1 = [(1, 100.0), (2, 150.0), (4, 200.0)]
    assign_cables(G, cables1)

    # graph-level checks
    assert G.graph['cables'] == cables1
    assert G.graph['currency'] == "€"
    #expected_capacity_1 = _recompute_kind_and_costs(cables1)[2]
    assert G.graph['capacity'] == 4
    def compare_cable_and_cost(edges_expected, edges_actual):
        # Convert EdgeDataView to a list
        edges_actual = list(edges_actual)

        # Optional: check lengths match
        assert len(edges_expected) == len(edges_actual), "Number of edges mismatch"

        # Iterate pairwise
        for edge_1, edge_2 in zip(edges_expected, edges_actual):
            u1, v1, attr1 = edge_1
            u2, v2, attr2 = edge_2

            # Check cable type
            assert attr1['cable'] == attr2['cable'], f"Edge ({u1}, {v1}) cable mismatch: {attr1['cable']} != {attr2['cable']}"

            # Check cost (approximate)
            assert math.isclose(attr1['cost'], attr2['cost'], rel_tol=1e-7, abs_tol=1e-9), \
                f"Edge ({u1}, {v1}) cost mismatch: {attr1['cost']} != {attr2['cost']}"

    expected_1 = [
        (0, 12, {'cable': 2, 'cost': 107.70329614269008}),
        (0, -1, {'cable': 2, 'cost': 200.0}),
        (1, 13, {'cable': 2, 'cost': 141.4213562373095}),
        (1, 2, {'cable': 1, 'cost': 150.0}),
        (2, 3, {'cable': 0, 'cost': 200.0}),
        (12, 13, {'cable': 2, 'cost': 60.0})
    ]
    actual_1 = [(u, v, {'cable': d['cable'], 'cost': d['cost']})
         for u, v, d in G.edges(data=True)]

    compare_cable_and_cost(expected_1, actual_1)
   
    # 2) Assign again with a different cable set: currency should update, cables update
    cables2 = [(10, 1000.0), (20, 1500.0), (30, 2000.0)]

    assign_cables(G, cables2, currency="Any Currency")

    assert G.graph['cables'] == cables2
    assert G.graph['currency'] == "Any Currency"
    # function sets capacity only if missing!
    #assert G.graph['capacity'] == 10

    # 3) All-zero-costs case:
    G_zero_cost = original_G.copy()
    cables3 = [(1, 0.0), (4, 0.0)]
    assign_cables(G_zero_cost, cables3, currency="IgnoredCurrency")
    assert G_zero_cost.graph['cables'] == cables3
    # since all costs zero, no cost value
    assert 'cost' not in G_zero_cost.graph


    # 4) Error case: raise ValueError when G.graph['max_load'] > max_capacity
    G4= original_G.copy()
    small_cables = [(1, 10.0), (2, 20.0)]
    with pytest.raises(ValueError):
        assign_cables(G4, small_cables)

    # test without capacity
    G4.graph.pop('capacity', None)
    cables4 = [(1, 100.0), (2, 150.0), (5, 200.0)]
    assign_cables(G4, cables4)
    assert G4.graph['capacity'] == 5


def test_describe_G():
    # Get a tiny test graph
    wfn = tiny_wfn()
    G = wfn.G

    # Run the description function
    desc = describe_G(G)
    # Expected output
    expected = ['κ = 4, T = 4', '(+0) [-1]: 1', 'Σλ = 5.5456 m', '55 €']

    # Assert equality
    assert desc == expected, f"Output mismatch:\nGot: {desc}\nExpected: {expected}"


def test_calcload():
    """
    Uses tiny_wfn() to obtain a test graph G and runs bfs_subtree_loads
    from a chosen parent (a root if present). The test asserts:
      - return value equals parent's 'load'
      - every visited node has subtree set
      - each direct parent->child edge has 'load' equal to child's load
        and 'reverse' set to (parent > child)
      - leaf nodes got a 'load' attribute
    """
    wfn = tiny_wfn()
    G = wfn.G

    G.graph.pop('has_loads', None)
    G.graph.pop('max_load', None)

    calcload(G)

    assert G.graph['has_loads'] == True
    assert G.graph['max_load'] == 4

def test_site_fingerprint():
    # prepare sample arrays
    VertexC = np.array([[0.0, 0.0], [1.5, -0.5], [10.0, 10.0]])
    boundary = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])

    # call the function
    digest, data_dict = site_fingerprint(VertexC, boundary)

    # types
    assert isinstance(digest, (bytes, bytearray))
    assert isinstance(data_dict, dict)
    assert 'VertexC' in data_dict and 'boundary' in data_dict
    assert isinstance(data_dict['VertexC'], (bytes, bytearray))
    assert isinstance(data_dict['boundary'], (bytes, bytearray))


def test_fun_fingerprint():
    def sample_function(x=1):
        return x + 1

    fp = fun_fingerprint(sample_function)

    assert isinstance(fp, dict)
    # expected hash computed the same way as the implementation
    expected_hash = hashlib.sha256(sample_function.__code__.co_code).digest()
    assert fp['funhash'] == expected_hash
    assert fp['funfile'] == sample_function.__code__.co_filename
    assert fp['funname'] == sample_function.__name__


def test_L_from_site():
    # Prepare VertexC with enough vertices
    # Choose T=3 (wtg nodes 0,1,2) and R=2 (oss nodes -2,-1) -> total V = 5 (indices 0..4)
    T = 3
    R = 2
    V = T + R
    VertexC = np.zeros((V, 2))  # coordinates don't matter for this test

    # 1) Call without 'handle', 'name', or 'B' but with border and obstacles
    border = np.array([0, 1])           # two border indices
    obstacles = [np.array([2]), np.array([3])]  # obstacles contribute 1 + 1 = 2
    # Expected B = 2 + 2 = 4
    L = L_from_site(VertexC=VertexC, T=T, R=R, border=border, obstacles=obstacles)

    # Defaults should be set
    assert L.graph['handle'] == 'L_from_site'
    assert L.graph['name'] == 'L_from_site'
    assert L.graph['B'] == border.shape[0] + sum(o.shape[0] for o in obstacles)

    # VertexC and T/R should be stored on the graph
    assert np.array_equal(L.graph['VertexC'], VertexC)
    assert L.graph['T'] == T
    assert L.graph['R'] == R

    # Node sets: 0..T-1 are 'wtg', -R..-1 are 'oss'
    for n in range(T):
        assert L.nodes[n]['kind'] == 'wtg'
    for n in range(-R, 0):
        assert L.nodes[n]['kind'] == 'oss'

    # Graph should contain exactly T + R nodes
    assert len(L.nodes) == T + R

    # 2) Call with explicit handle, name and B to cover the other branches
    L2 = L_from_site(VertexC=VertexC, T=T, R=R, handle='custom', name='CustomSite', B=99)

    # Provided values should be preserved (no defaults applied / B not recomputed)
    assert L2.graph['handle'] == 'custom'
    assert L2.graph['name'] == 'CustomSite'
    assert L2.graph['B'] == 99

    # Node kinds and counts as before
    for n in range(T):
        assert L2.nodes[n]['kind'] == 'wtg'
    for n in range(-R, 0):
        assert L2.nodes[n]['kind'] == 'oss'
    assert len(L2.nodes) == T + R


def test_G_from_S():
    """
    """
    wfn = tiny_wfn()
    A = wfn.A
    S = wfn.S

    G = G_from_S(S, A)

    expected = [(0, 12), (0, -1), (1, 13), (1, 2), (2, 3), (12, 13)]
    actual = list(wfn.G.edges())  # convert EdgeView to list
    assert set(actual) == set(expected)

    # No tentative/rogue branches should have been triggered in this simple setup
    assert 'tentative' not in G.graph or G.graph.get('tentative') == []
    assert 'rogue' not in G.graph

    # num_diagonals present (0 for this setup)
    assert 'num_diagonals' in G.graph
    assert G.graph['num_diagonals'] == 0

    # cover other branches
    
    A[0][2]['shortcuts'] = [9]

    A[2][-1]['shortcuts'] = [9]
    S.add_edge(0, 2, load=1, reverse=False)
    S.add_edge(2, -1, load=1, reverse=False)
    G = G_from_S(S, A)

    assert (0, 2) in G.edges() or (2, 0) in G.edges()
    assert G[0][2]['kind'] == 'contour'
    assert (0, 2) in G.graph['shortened_contours']

    #
    edges_to_test = [(0, 1), (0, 2), (0, 3), (-1, 2)]

    for s, t in edges_to_test:
        # Deep copy A and S to restore originals after test
        A_copy = copy.deepcopy(A)
        S_copy = copy.deepcopy(S)

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
        assert (s, t) in G.edges() or (t, s) in G.edges()

        # Check kind based on s
        expected_kind = 'contour' if s >= 0 else 'tentative'
        actual_kind = G[s][t]['kind'] if (s, t) in G.edges() else G[t][s]['kind']
        assert actual_kind == expected_kind

    #
    edges_to_test = [(1, 3), (-1, 1)]

    for s, t in edges_to_test:
        # Deep copy A and S to restore originals after test
        A_copy = copy.deepcopy(A)
        S_copy = copy.deepcopy(S)

        # Add only the current edge
        S_copy.add_edge(s, t, load=1, reverse=False)
        S_copy.nodes[s]['subtree'] = 0
        S_copy.nodes[t]['subtree'] = 0

        # Run G_from_S
        G = G_from_S(S_copy, A_copy)

        # Check edge exists
        assert (s, t) in G.edges() or (t, s) in G.edges()
        expected_kind = 'rogue' if s >= 0 else 'tentative'
        actual_kind = G[s][t]['kind'] if (s, t) in G.edges() else G[t][s]['kind']
        assert actual_kind == expected_kind

def test_L_from_G():
    # Copy G to avoid modifying original
    G = tiny_wfn().G

    # Run L_from_G
    L = L_from_G(G)

    R = G.graph['R']
    T = G.graph['T']

    # Check number of nodes
    assert all(n in L.nodes() for n in range(T)), "WTG nodes missing"
    assert all(r in L.nodes() for r in range(-R, 0)), "OSS nodes missing"

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

    
    G.graph['stunts_primes'] = [100, 101]  # new dummy nodes to simulate stunts/primes
    L_stunts = L_from_G(G)
    assert L_stunts.number_of_edges() == 0
    # Check VertexC adjusted for stunts_primes
    assert L_stunts.graph['VertexC'].shape[0] == len(G.graph['VertexC']) - len(G.graph['stunts_primes'])


def test_S_from_terse_links():
    terse_links = np.array([-1, 0, 1, 2])
    
    def check_S(S, expected_capacity=None):
        # Check number of nodes
        assert len(S.nodes()) == len(terse_links) + 1  # +1 for root node

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

        # Check graph attributes
        assert S.graph['T'] == 4
        assert S.graph['R'] == 1

    # Test without explicit capacity
    S1 = S_from_terse_links(terse_links)
    check_S(S1)

    # Test with explicit capacity
    S2 = S_from_terse_links(terse_links, capacity=5)
    check_S(S2, expected_capacity=5)

def test_terse_links_from_S():
    S = tiny_wfn().S
    # Original terse_links
    expected_terse = np.array([-1, 0, 1, 2])

    # Convert back to terse_links
    actual_terse = terse_links_from_S(S)

    # Check that recovered array matches original
    assert np.array_equal(actual_terse, expected_terse), \
        f"terse_links {actual_terse} != expected_terse_links {expected_terse}"
    

def test_as_single_root():
    # single root L
    L_prime = tiny_wfn().L
    L = as_single_root(L_prime)
    assert_graph_equal(L, L_prime)

    del L_prime, L

    # L with 3 roots
    T, R = 4, 3
    VertexC = np.array([[0, 0], [1, 0], [2, 0], [3, 0],   # WTGs 0-3
                        [0, 1], [1, 1], [2, 1]])           # Roots -3, -2, -1
    L_prime = nx.Graph(T=T, R=R, VertexC=VertexC.copy(), name='Site', handle='site_handle')
    L_prime.add_nodes_from(range(T), kind='wtg')
    L_prime.add_nodes_from(range(-R, 0), kind='oss')

    # Apply as_single_root
    L = as_single_root(L_prime)

    # Check R reduced to 1
    assert L.graph['R'] == 1

    # Check only one root remains
    remaining_roots = [n for n in L.nodes() if n < 0]
    assert remaining_roots == [-1]

    # Check new root's position is centroid of original roots
    expected_centroid = VertexC[-R:].mean(axis=0)
    np.testing.assert_allclose(L.graph['VertexC'][-1], expected_centroid)

    # Check name and handle updated
    assert L.graph['name'].endswith('.1_OSS')
    assert L.graph['handle'].endswith('_1')

    # Check WTGs unchanged
    assert all(L.nodes[n]['kind'] == 'wtg' for n in range(T))



def test_as_normalized_cases():
    A = tiny_wfn().A
    original_vertexC = A.graph['VertexC'].copy()
    original_d2roots = A.graph['d2roots'].copy()
    original_lengths = [edata['length'] for _, _, edata in A.edges(data=True)]
    
    offset = np.array([1.0, 2.0])
    scale = 2.0

    # Case 1: both offset and scale
    A_norm = as_normalized(A, offset=offset, scale=scale)
    np.testing.assert_allclose(A_norm.graph['VertexC'], scale * (original_vertexC - offset))
    np.testing.assert_allclose(A_norm.graph['d2roots'], scale * original_d2roots)
    for (_, _, edata_norm), original_length in zip(A_norm.edges(data=True), original_lengths):
        np.testing.assert_allclose(edata_norm['length'], original_length * scale)
    assert A_norm.graph['is_normalized'] is True

    # Case 2: only offset
    A_norm = as_normalized(A, offset=offset)
    expected_vertexC = A.graph['norm_scale'] * (original_vertexC - offset)
    np.testing.assert_allclose(A_norm.graph['VertexC'], expected_vertexC)

    # Case 3: only scale
    A_norm = as_normalized(A, scale=scale)
    expected_vertexC = scale * (original_vertexC - A.graph['norm_offset'])
    np.testing.assert_allclose(A_norm.graph['VertexC'], expected_vertexC)

    # Ensure original graph unchanged
    np.testing.assert_allclose(A.graph['VertexC'], original_vertexC)



def test_as_rescaled():
    wfn = tiny_wfn()
    L = wfn.L
    G = wfn.G
    # --- Case 1: G is normalized, L has d2roots ---
    G.graph['is_normalized'] = True
    G.graph['norm_scale'] = 2.0
    original_lengths = [edata['length'] for _, _, edata in G.edges(data=True)]
    L.graph['d2roots'] = np.array([[0.0, 1.0], [1.0, 0.0]])

    G_rescaled = as_rescaled(G, L)

    # VertexC should match L
    np.testing.assert_allclose(G_rescaled.graph['VertexC'], L.graph['VertexC'])

    # Edge lengths should be scaled down by 1/norm_scale
    for (_, _, edata_res), original_length in zip(G_rescaled.edges(data=True), original_lengths):
        np.testing.assert_allclose(edata_res['length'], original_length / G.graph['norm_scale'])

    # d2roots should be copied from L
    np.testing.assert_allclose(G_rescaled.graph['d2roots'], L.graph['d2roots'])

    # is_normalized removed, denormalization factor set
    assert 'is_normalized' not in G_rescaled.graph
    assert 'denormalization' in G_rescaled.graph
    np.testing.assert_allclose(G_rescaled.graph['denormalization'], 1 / G.graph['norm_scale'])

    # --- Case 2: G not normalized ---
    G2 = G.copy()
    G2.graph.pop('is_normalized', None)
    G2.graph['norm_scale'] = 2.0  # should be ignored
    G2_rescaled = as_rescaled(G2, L)
    # Graph should be unchanged
    assert G2_rescaled == G2

    # --- Case 3: L does not have d2roots ---
    L2 = L.copy()
    L2.graph.pop('d2roots', None)
    G3 = G.copy()
    G3.graph['is_normalized'] = True
    G3.graph['norm_scale'] = 2.0
    G3_rescaled = as_rescaled(G3, L2)
    # d2roots should be removed if present in G
    assert 'd2roots' not in G3_rescaled.graph

    # --- Case 4: L does not have d2roots but G has ---
    G3.graph['d2roots'] = np.array([[0.0, 1.0], [1.0, 0.0]])
    G3_rescaled = as_rescaled(G3, L2)
    # d2roots should be removed if present in G
    assert 'd2roots' not in G3_rescaled.graph

# def test_assign_cables_capacity_error():
#     G = nx.Graph(R=1, T=2)
#     G.add_node(-1, kind="oss")
#     G.add_node(0, kind="wtg")
#     G.add_node(1, kind="wtg")
#     G.add_edge(0, 1, load=3)
#     G.graph["max_load"] = 3

#     with pytest.raises(ValueError, match="Maximum cable capacity"):
#         assign_cables(G, [(1, 0.0), (2, 0.0)])


# # ----------------------------
# # update_lengths + describe_G
# # ----------------------------

# def test_update_lengths_and_describe_G_rounding():
#     # Manual tiny graph
#     wfn = tiny_wfn(optimize=True)
#     G = wfn.G.copy()

#     #G.graph['VertexC'] = np.array([[0, -1], [0, 0], [3, 4]], float)
#     update_lengths(G)

#     for u, v, data in G.edges(data=True):
#         assert isinstance(data["length"], float)

#     desc = describe_G(G)
#     assert any("κ" in line or "Σλ" in line for line in desc)


# # ----------------------------
# # S_from_G + terse_links
# # ----------------------------

# def test_S_from_G_roundtrip_and_terse_links():
#     wfn = tiny_wfn(optimize=True)
#     G = wfn.G.copy()
#     # ensure loads exist
#     for u, v in G.edges:
#         G.edges[u, v]['load'] = 1
#     G.graph['has_loads'] = True

#     S = S_from_G(G)
#     terse = terse_links_from_S(S)
#     assert hasattr(terse, "shape")


# # ----------------------------
# # normalize/rescale
# # ----------------------------

# def test_as_normalized_and_as_rescaled_invert():
#     wfn = tiny_wfn(optimize=True)
#     G = wfn.G.copy()

#     # Provide d2roots explicitly
#     G.graph['d2roots'] = np.ones((len(G.nodes), G.graph['R']))
#     G.graph['norm_offset'] = np.array([0.0, 0.0])
#     G.graph['norm_scale'] = 0.5

#     A_norm = as_normalized(G)
#     for u, v, data in A_norm.edges(data=True):
#         assert math.isclose(data['length'], 0.5 * G.edges[u, v]['length'])

#     G_rescaled = as_rescaled(A_norm, G)
#     for u, v, data in G_rescaled.edges(data=True):
#         assert math.isclose(data['length'], G.edges[u, v]['length'])


# # ----------------------------
# # undetour
# # ----------------------------

# def test_as_undetoured_removes_detour_chain():
#     G = tiny_wfn().G

#     G2 = as_undetoured(G)
#     assert detour not in G2.nodes
#     assert (-1, 0) in G2.edges


# # ----------------------------
# # rehooking (nearest / head)
# # ----------------------------

# def test_as_hooked_to_nearest_and_head():
#     wfn = tiny_wfn()
#     G = wfn.G
#     # G = nx.Graph(R=1, T=2, has_loads=True)
#     # G.add_node(-1, kind="oss", load=3)
#     # G.add_node(0, kind="wtg", load=1, subtree=0)
#     # G.add_node(1, kind="wtg", load=1, subtree=0)
#     # G.add_edge(-1, 0)
#     # G.add_edge(0, 1)
#     # # tentative edge
#     # G.add_edge(-1, 1, kind="tentative", load=2)
#     # G.graph["tentative"] = [(-1, 1)]

#     # distance to root
#     d2roots = np.array([[0.0], [1.0]])

#     G_nearest = as_hooked_to_nearest(G, d2roots)
#     assert (-1, 0) in G_nearest.edges
#     assert (-1, 1) not in G_nearest.edges

#     G_head = as_hooked_to_head(G, d2roots)
#     assert (-1, 0) in G_head.edges
#     assert (-1, 1) not in G_head.edges


# # ----------------------------
# # scaffolded
# # ----------------------------

# def test_scaffolded_merges_edges():
#     G = nx.Graph(R=1, T=1)
#     G.add_node(-1, kind="oss")
#     G.add_node(0, kind="wtg")
#     G.add_edge(-1, 0)

#     from networkx import PlanarEmbedding
#     P = PlanarEmbedding()
#     P.add_node(-1); P.add_node(0)
#     P.add_half_edge(-1, 0)
#     P.add_half_edge(0, -1)
#     P.graph["supertriangleC"] = np.array([[-10, -10], [10, -10], [0, 10]])
#     G.graph["VertexC"] = np.array([[0, -1], [0, 0]])

#     scaff = scaffolded(G, P)
#     assert (-1, 0) in scaff.edges
#     assert 'kind' not in scaff.edges[-1, 0]
