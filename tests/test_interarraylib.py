import copy
import math
import networkx as nx
import numpy as np
import pytest

from optiwindnet.interarraylib import (
    G_from_S,
    L_from_G,
    L_from_site,
    S_from_G,
    S_from_terse_links,
    as_normalized,
    as_rescaled,
    as_single_root,
    as_stratified_vertices,
    as_undetoured,
    assign_cables,
    calcload,
    describe_G,
    fun_fingerprint,
    scaffolded,
    site_fingerprint,
    terse_links_from_S,
    update_lengths,
    count_diagonals,
    as_hooked_to_head,
    as_hooked_to_nearest,
)

from .helpers import assert_graph_equal, tiny_wfn


# ----------
# tests
# ----------


def test_assign_cables():
    # use tiny_wfn
    wfn = tiny_wfn()
    original_G = wfn.G

    # 1) check defaults
    G = original_G.copy()
    cables1 = [(1, 100.0), (2, 150.0), (4, 200.0)]
    assign_cables(G, cables1)

    # graph-level checks
    assert G.graph['cables'] == cables1
    assert G.graph['currency'] == '€'
    assert G.graph['capacity'] == 4

    wfn2 = tiny_wfn(cables=1)
    G2 = wfn2.G

    # 1) check defaults
    cables2 = [(1, 0.0)]
    assign_cables(G2, cables2)

    # graph-level checks
    assert G2.graph['cables'] == cables2
    # assert G2.graph['currency'] == '€'
    assert G2.graph['capacity'] == 1

    def compare_cable_and_cost(G, edges_expected):
        # Optional: check lengths match
        assert len(edges_expected) == G.number_of_edges(), 'Number of edges mismatch'

        # Iterate pairwise
        for u, v, expectedD in edges_expected:
            actualD = G[u][v]

            # Check cable type
            assert expectedD['cable'] == actualD['cable'], (
                f'Edge {u, v} cable mismatch: {expectedD["cable"]} != {actualD["cable"]}'
            )

            # Check cost (approximate)
            assert math.isclose(
                expectedD['cost'], actualD['cost'], rel_tol=1e-7, abs_tol=1e-9
            ), f'Edge {u, v} cost mismatch: {expectedD["cost"]} != {actualD["cost"]}'

    expected_1 = [
        (0, 12, {'cable': 2, 'cost': 107.70329614269008}),
        (-1, 0, {'cable': 2, 'cost': 200.0}),
        (1, 13, {'cable': 2, 'cost': 141.4213562373095}),
        (1, 2, {'cable': 1, 'cost': 150.0}),
        (2, 3, {'cable': 0, 'cost': 200.0}),
        (12, 13, {'cable': 2, 'cost': 60.0}),
    ]

    compare_cable_and_cost(G, expected_1)

    # 2) Assign again with a different cable set: currency should update, cables update
    cables2 = [(10, 1000.0), (20, 1500.0), (30, 2000.0)]

    assign_cables(G, cables2, currency='Any Currency')

    assert G.graph['cables'] == cables2
    assert G.graph['currency'] == 'Any Currency'

    # 3) All-zero-costs case:
    G_zero_cost = original_G.copy()
    cables3 = [(1, 0.0), (4, 0.0)]
    assign_cables(G_zero_cost, cables3, currency='IgnoredCurrency')
    assert G_zero_cost.graph['cables'] == cables3
    # since all costs zero, no cost value
    assert 'cost' not in G_zero_cost.graph

    # 4) Error case: raise ValueError when G.graph['max_load'] > max_capacity
    G4 = original_G.copy()
    small_cables = [(1, 10.0), (2, 20.0)]
    with pytest.raises(ValueError):
        assign_cables(G4, small_cables)

    # 5) test without capacity
    G4.graph.pop('capacity', None)
    cables4 = [(1, 100.0), (2, 150.0), (5, 200.0)]
    assign_cables(G4, cables4)
    assert G4.graph['capacity'] == 5


def test_describe_G():
    wfn = tiny_wfn()
    G = wfn.G

    desc = describe_G(G)
    expected = ['κ = 4, T = 4', '(+0) [-1]: 1', 'Σλ = 5.5456\u00a0m', '55\u00a0€']

    assert desc == expected, f'Output mismatch:\nGot: {desc}\nExpected: {expected}'


def test_calcload():
    wfn = tiny_wfn()
    G = wfn.G

    G.graph.pop('has_loads', None)
    G.graph.pop('max_load', None)

    calcload(G)

    assert G.graph['has_loads']
    assert G.graph['max_load'] == 4


def test_site_fingerprint():
    VertexC = np.array([[0.0, 0.0], [1.5, -0.5], [10.0, 10.0]])
    boundary = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])

    digest, data_dict = site_fingerprint(VertexC, boundary)

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
    assert all(k in fp for k in ('funhash', 'funfile', 'funname'))


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
        return all(G.nodes[node] == nodeD for node, nodeD in expected)

    def check_edges(G, expected):
        return all(G[u][v] == edgeD for u, v, edgeD in expected)

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


def test_G_from_S():
    wfn = tiny_wfn()
    A = wfn.A
    S = wfn.S

    # 1) basic test
    G = G_from_S(S, A)
    expected = [(0, 12), (-1, 0), (1, 13), (1, 2), (2, 3), (12, 13)]
    assert all(uv in G.edges for uv in expected)

    # No tentative/rogue
    assert 'tentative' not in G.graph or G.graph.get('tentative') == []
    assert 'rogue' not in G.graph

    # num_diagonals present
    assert 'num_diagonals' in G.graph
    assert G.graph['num_diagonals'] == 0

    # 2) normalized A
    A2 = as_normalized(A)
    G2 = G_from_S(S, A2)
    assert G2.graph['is_normalized']

    # shortcuts in A
    A[0][2]['shortcuts'] = [9]
    A[2][-1]['shortcuts'] = [9]
    S.add_edge(0, 2, load=1, reverse=False)
    S.add_edge(2, -1, load=1, reverse=False)
    G = G_from_S(S, A)

    assert (0, 2) in G.edges
    assert G[0][2]['kind'] == 'contour'
    assert (0, 2) in G.graph['shortened_contours']

    #
    edges_to_test = [(0, 1), (0, 2), (0, 3), (-1, 2)]

    for s, t in edges_to_test:
        # Deep copy A and S to restore originals at each iteration
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
        assert (s, t) in G.edges

        # Check kind based on s
        expected_kind = 'contour' if s >= 0 else 'tentative'
        actual_kind = G[s][t]['kind'] if (s, t) in G.edges() else G[t][s]['kind']
        assert actual_kind == expected_kind

    #
    edges_to_test = [(1, 3), (-1, 1)]

    for s, t in edges_to_test:
        # Deep copy A and S to restore originals at each iteration
        A_copy = copy.deepcopy(A)
        S_copy = copy.deepcopy(S)

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

    # 2) test stunts_primes
    G.graph['stunts_primes'] = [100, 101]  # new dummy nodes to simulate stunts/primes
    L_stunts = L_from_G(G)
    assert L_stunts.number_of_edges() == 0
    # Check VertexC adjusted for stunts_primes
    assert L_stunts.graph['VertexC'].shape[0] == len(G.graph['VertexC']) - len(
        G.graph['stunts_primes']
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


def test_as_single_root():
    # 1) single root L
    L_prime = tiny_wfn().L
    L = as_single_root(L_prime)
    assert_graph_equal(L, L_prime)

    del L_prime, L

    # 2) L with 3 roots
    T, R = 4, 3
    VertexC = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [0, 1],
            [1, 1],
            [2, 1],
        ]
    )  # Roots -3, -2, -1
    L_prime = nx.Graph(
        T=T, R=R, B=0, VertexC=VertexC, name='Site', handle='site_handle'
    )
    L_prime.add_nodes_from(range(T), kind='wtg')
    L_prime.add_nodes_from(range(-R, 0), kind='oss')

    # Apply as_single_root
    L = as_single_root(L_prime)

    # Check R reduced to 1
    assert L.graph['R'] == 1
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
    np.testing.assert_allclose(
        A_norm.graph['VertexC'], scale * (original_vertexC - offset)
    )
    np.testing.assert_allclose(A_norm.graph['d2roots'], scale * original_d2roots)
    for (_, _, edata_norm), original_length in zip(
        A_norm.edges(data=True), original_lengths
    ):
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
    for (_, _, edata_res), original_length in zip(
        G_rescaled.edges(data=True), original_lengths
    ):
        np.testing.assert_allclose(
            edata_res['length'], original_length / G.graph['norm_scale']
        )

    # d2roots should be copied from L
    np.testing.assert_allclose(G_rescaled.graph['d2roots'], L.graph['d2roots'])

    # is_normalized removed, denormalization factor set
    assert 'is_normalized' not in G_rescaled.graph
    assert 'denormalization' in G_rescaled.graph
    np.testing.assert_allclose(
        G_rescaled.graph['denormalization'], 1 / G.graph['norm_scale']
    )

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

    # G with d2roots
    G3.graph['d2roots'] = np.array([[0.0, 1.0], [1.0, 0.0]])
    G3_rescaled = as_rescaled(G3, L2)
    # d2roots should be removed if present in G
    assert 'd2roots' not in G3_rescaled.graph

    # G without d2roots
    G3.graph.pop('d2roots', None)
    G3_rescaled = as_rescaled(G3, L2)
    assert 'd2roots' not in G3_rescaled.graph


def test_as_undetoured():
    wfn = tiny_wfn()
    G = wfn.G

    # --- Case A: no detour in G
    G1 = G.copy()
    out1 = as_undetoured(G1)
    assert_graph_equal(out1, G1)

    # --- Case B: D == 0
    G2 = G.copy()
    G2.graph['D'] = 0  # explicitly mark no detours
    out2 = as_undetoured(G2)
    assert_graph_equal(out2, G2)

    # --- Case C: D > 0 and C == 0
    G3 = G.copy()
    detour_node = 100
    target_wtg = 3
    # add detour node and connect root -> detour -> target_wtg
    G3.add_node(detour_node)
    G3.add_edge(-1, detour_node)
    G3.add_edge(detour_node, target_wtg)
    G3.graph['D'] = 1
    out3 = as_undetoured(G3)

    assert detour_node not in out3.nodes(), (
        'detour node should be removed by as_undetoured()'
    )

    # --- Case D: D > 0,  C == 0 and fnT
    G4 = G.copy()
    G4.graph['D'] = 1
    G4.graph['C'] = 0
    G4.graph['fnT'] = 'dummy fnT'

    out4 = as_undetoured(G4)
    assert 'fnT' not in out4.graph, 'fnT should be removed if D > 0 and no contour'


def test_as_stratified_vertices():
    wfn = tiny_wfn()
    L0 = wfn.L.copy()

    # --- Case A: border-vertices are all in the B-range of VertexC
    L1 = as_stratified_vertices(L0)
    assert_graph_equal(L0, L1)

    # --- Case B: border-vertices are NOT in the B-range of VertexC
    L0.graph['border'] = np.array([0, 5, 6, 7])
    expected_VertexC = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [2.0, 3.0],
            [1.0, 0.0],
            [2.0, -2.0],
            [2.0, 4.0],
            [-2.0, 4.0],
            [1.2, -0.5],
            [1.2, 1.0],
            [1.8, 0.5],
            [1.5, -0.5],
            [0.0, 0.0],
        ]
    )
    L2 = as_stratified_vertices(L0)
    assert np.array_equal(L2.graph['VertexC'], expected_VertexC)


def test_scaffolded():
    wfn = tiny_wfn()
    G = wfn.G.copy()
    P = wfn.P.copy()
    scaff = scaffolded(G, P)

    # Check that returned graph is undirected
    assert not scaff.is_directed()

    # All nodes from G should be in scaff
    for n in G.nodes():
        assert n in scaff.nodes()
        for k, v in G.nodes[n].items():
            assert scaff.nodes[n][k] == v

    # fnT should exist and match expected length
    assert 'fnT' in scaff.graph
    assert len(scaff.graph['fnT']) == 15

    # VertexC should contain G's VertexC plus P's supertriangle
    R = G.graph.get('R', 0)
    supertriangleC = P.graph['supertriangleC']
    VertexC_expected = np.vstack(
        (G.graph['VertexC'][:-R], supertriangleC, G.graph['VertexC'][-R:])
    )
    assert np.allclose(scaff.graph['VertexC'], VertexC_expected)


def test_update_lengths():
    wfn = tiny_wfn()
    G = wfn.G.copy()

    expected_lengths = [
        (0, 12, 0.5385164807134504),
        (0, -1, 1.0),
        (1, 13, 0.7071067811865476),
        (1, 2, 1.0),
        (2, 3, 2.0),
        (12, 13, 0.30000000000000004),
    ]

    # remove some of lengths from G
    del G.edges[0, -1]['length']
    del G.edges[2, 3]['length']

    update_lengths(G)

    # check all lengths are available in G
    for u, v, expected in expected_lengths:
        actual_length = G.edges[u, v].get('length')
        assert actual_length == pytest.approx(expected), (
            f'Edge {(u, v)} length {actual_length} != expected {expected}'
        )


def test_count_diagonals():
    wfn = tiny_wfn()
    diagonals = count_diagonals(wfn.S, wfn.A)
    assert diagonals == 0


def test_as_hooked_to_head():
    wfn1 = tiny_wfn()
    G1 = as_hooked_to_head(wfn1.S, wfn1.A.graph['d2roots'])
    expected = [(-1, 0)]
    assert G1.graph['tentative'] == expected

    wfn2 = tiny_wfn(cables=1)
    G2 = as_hooked_to_head(wfn2.S, wfn2.A.graph['d2roots'])
    expected = [(-1, 0), (-1, 1), (-1, 2), (-1, 3)]
    assert G2.graph['tentative'] == expected


def test_as_hooked_to_nearest():
    wfn1 = tiny_wfn()
    G1 = as_hooked_to_nearest(wfn1.S, wfn1.A.graph['d2roots'])
    expected = [(-1, 0)]
    assert G1.graph['tentative'] == expected

    wfn2 = tiny_wfn(cables=1)
    G2 = as_hooked_to_nearest(wfn2.S, wfn2.A.graph['d2roots'])
    expected = [(-1, 0), (-1, 1), (-1, 2), (-1, 3)]
    assert G2.graph['tentative'] == expected
