# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Derived graph variants: normalized, single-root, undetoured, rehooked."""

import networkx as nx
import numpy as np

from optiwindnet.transforming import (
    as_hooked_to_head,
    as_hooked_to_nearest,
    as_normalized,
    as_obstacle_free,
    as_rescaled,
    as_single_root,
    as_stratified_vertices,
    as_undetoured,
)

from .helpers import assert_graph_equal, tiny_wfn


def test_as_single_root():
    # 1) single root L
    L_prime = tiny_wfn().L
    L = as_single_root(L_prime)
    assert_graph_equal(L, L_prime)

    del L_prime, L

    # 2) L with 3 roots
    T, R = 4, 3
    # the last R=3 rows are the roots -3, -2, -1
    VertexC = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1],
    ])  # fmt: skip
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
    # strata: 4 terminals, 4 border, 4 obstacle, 1 root
    expected_VertexC = np.array([
        [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 3.0],
        [1.0, 0.0], [2.0, -2.0], [2.0, 4.0], [-2.0, 4.0],
        [1.2, -0.5], [1.2, 1.0], [1.8, 0.5], [1.5, -0.5],
        [0.0, 0.0],
    ])  # fmt: skip
    L2 = as_stratified_vertices(L0)
    assert np.array_equal(L2.graph['VertexC'], expected_VertexC)


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


def test_as_hooked_to_nearest_clears_clone_loads():
    """A clone inside the rehooked subtree must not keep its previous load.

    ``as_hooked_to_nearest()`` accepts a routeset, whose subtrees may contain
    clones. Their loads are stale once the subtree is rehooked, and
    ``bfs_subtree_loads()`` reads a surviving one back as the node's base load.
    """
    # -1 — 3 — 2 — 4(clone) — 1 — 0, hooked at 3 but 0 is the nearest to the root
    T, R, clone = 4, 1, 4
    G = nx.Graph(R=R, T=T, has_loads=True, max_load=4)
    G.add_node(-1, kind='oss', load=4)
    G.add_nodes_from(range(T), kind='wtg')
    G.add_node(clone, kind='contour')
    nx.add_path(G, (-1, 3, 2, clone, 1, 0))
    for node, load in ((3, 4), (2, 3), (clone, 2), (1, 2), (0, 1)):
        G.nodes[node].update(load=load, subtree=0)
    for u, v in G.edges:
        G[u][v]['load'] = min(G.nodes[u].get('load', 4), G.nodes[v].get('load', 4))
    # terminal 0 is the closest to the root, so the feeder moves from 3 to 0
    d2roots = np.array([[1.0], [2.0], [3.0], [4.0]])

    H = as_hooked_to_nearest(G, d2roots)

    assert H.graph['tentative'] == [(-1, 0)]
    assert H.nodes[-1]['load'] == T
    # loads now run 0 → root, and the clone carries the load of its two sides
    assert [H.nodes[n]['load'] for n in (0, 1, clone, 2, 3)] == [4, 3, 2, 2, 1]


# --- as_obstacle_free ---


def test_as_obstacle_free_no_obstacles():
    """When L has no obstacles the function returns early with a copy."""
    wfn = tiny_wfn(optimize=False)
    L = wfn.L.copy()
    L.graph['obstacles'] = None
    L_out = as_obstacle_free(L)
    assert L_out.graph.get('obstacles') is None
    assert np.array_equal(L_out.graph['VertexC'], L.graph['VertexC'])


def test_as_obstacle_free_removes_obstacles():
    """With obstacles present, vertex count and obstacle key are cleaned up."""
    wfn = tiny_wfn(optimize=False)
    L = wfn.L
    assert L.graph.get('obstacles') is not None
    L_out = as_obstacle_free(L)
    assert 'obstacles' not in L_out.graph
    # VertexC shrinks: obstacle-only border vertices are removed
    assert L_out.graph['VertexC'].shape[0] <= L.graph['VertexC'].shape[0]
    assert L_out.graph['B'] <= L.graph['B']


# --- as_single_root: no roots referenced by border ---


def test_as_single_root_no_root_in_border():
    """Border has no negative-index entries → B stays the same, no vertex transfer."""
    wfn = tiny_wfn(optimize=False)
    L = wfn.L.copy()
    T = L.graph['T']
    # replace border with indices all in [T, T+B) range (no root refs)
    B = L.graph['B']
    if B > 0:
        L.graph['border'] = np.arange(T, T + min(B, 3), dtype=int)
    else:
        L.graph['border'] = np.array([0, 1], dtype=int)  # turbine indices, also fine
    L_out = as_single_root(L)
    assert L_out.graph['R'] == 1


def test_as_single_root_transfers_repeated_border_roots():
    L = nx.Graph(
        T=1,
        R=3,
        B=1,
        VertexC=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]),
        border=np.array([-3, 1, -3, -2]),
        name='roots-in-border',
        handle='roots_in_border',
    )
    L.add_node(0, kind='wtg')
    L.add_nodes_from(range(-3, 0), kind='oss')

    result = as_single_root(L)

    assert result.graph['R'] == 1
    assert result.graph['B'] == 3
    border = result.graph['border']
    assert border[0] == border[2]
    assert set(border) == {1, 2, 3}
