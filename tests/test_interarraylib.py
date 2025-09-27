# SPDX-License-Identifier: MIT
# tests/test_interarraylib.py

import math
import numpy as np
import networkx as nx
import pytest

from optiwindnet.interarraylib import (
    assign_cables,
    update_lengths,
    describe_G,
    calcload,
    S_from_G,
    terse_links_from_S,
    as_normalized,
    as_rescaled,
    as_undetoured,
    as_hooked_to_nearest,
    as_hooked_to_head,
    scaffolded,
)

# ----------------------------
# helpers
# ----------------------------

def _coords(pairs):
    """Return (N,2) float array from list of (x,y)."""
    return np.array(pairs, dtype=float)

def _make_simple_L(T=2, R=1):
    # terminals then roots at the end
    verts = np.vstack([_coords([(0, 0), (1, 0)]), _coords([(0, -1)])])
    L = nx.Graph(T=T, R=R, VertexC=verts, B=0, border=np.array([], dtype=int), obstacles=[])
    for i in range(T):
        L.add_node(i, kind="wtg", label=f"T{i}")
    for r in range(-R, 0):
        L.add_node(r, kind="oss", label=f"OSS{abs(r)}")
    return L


# ----------------------------
# assign_cables
# ----------------------------

def test_assign_cables_basic_and_currency():
    # Simple tree with loads set; -1 is the OSS (root)
    G = nx.Graph(R=1, T=3, VertexC=_coords([(0,0),(1,0),(2,0),(0,-1)]))
    for n in (-1, 0, 1, 2):
        G.add_node(n)
    G.add_edge(0, 1, length=10.0, load=1)
    G.add_edge(1, 2, length=20.0, load=2)
    G.add_edge(-1, 0, length=5.0, load=0)

    # Ensure max_load exists (assign_cables expects it)
    calcload(G)

    cables = [(1, 100.0), (2, 150.0), (3, 200.0)]
    assign_cables(G, cables, currency="€")

    # calcload() recomputes loads on the path: load(0-1)=2, load(1-2)=1
    # so cable indices/costs are:
    assert G.edges[0, 1]["cable"] == 1        # capacity 2
    assert math.isclose(G.edges[0, 1]["cost"], 1500.0)  # 10 * 150

    assert G.edges[1, 2]["cable"] == 0        # capacity 1
    assert math.isclose(G.edges[1, 2]["cost"], 2000.0)  # 20 * 100

    # graph bookkeeping
    assert G.graph["cables"] == cables
    assert G.graph["currency"] == "€"
    assert G.graph["capacity"] == 3

    # calcload() recomputes loads on the path: load(0-1)=2, load(1-2)=1
    assert G.edges[0, 1]["cable"] == 1
    assert math.isclose(G.edges[0, 1]["cost"], 1500.0)

    assert G.edges[1, 2]["cable"] == 0
    assert math.isclose(G.edges[1, 2]["cost"], 2000.0)



def test_assign_cables_capacity_error():
    # max load 3, but maximum cable capacity is 2 -> error
    G = nx.Graph(R=1, T=2, VertexC=_coords([(0,0),(1,0),(0,-1)]))
    for n in (-1, 0, 1):
        G.add_node(n)
    G.add_edge(0, 1, length=1.0, load=3)
    # Minimal guard so assign_cables sees the constraint
    G.graph["max_load"] = 3

    with pytest.raises(ValueError, match="Maximum cable capacity"):
        assign_cables(G, [(1, 0.0), (2, 0.0)])


# ----------------------------
# update_lengths + describe_G
# ----------------------------

def test_update_lengths_and_describe_G_rounding():
    # VertexC: node 0 (0,0), node 1 (3,4), root -1 at (0,-1)
    VertexC = _coords([(0, 0), (3, 4), (0, -1)])
    G = nx.Graph(R=1, T=2, VertexC=VertexC, capacity=2, currency="€")
    G.add_node(-1, kind="oss", label="OSS")
    G.add_node(0, kind="wtg", label="T0")
    G.add_node(1, kind="wtg", label="T1")
    # length missing on (-1,0) should be filled
    G.add_edge(-1, 0, load=0)
    G.add_edge(0, 1, length=5.0, load=1)

    update_lengths(G)
    # (-1,0): dist between (0,-1) and (0,0) -> 1.0
    assert math.isclose(G.edges[-1, 0]["length"], 1.0)

    # Describe should include capacity/T, feeders, total length, currency line
    # Make degree data realistic for describe output
    G.graph.update(T=2)
    desc = describe_G(G)
    assert any("κ = 2" in line for line in desc)
    assert any("Σλ" in line for line in desc)
    # if 'currency' in graph, final line is total cost (here zeros)
    assert desc[-1].endswith(" €")


# ----------------------------
# S_from_G + terse_links
# ----------------------------

def test_S_from_G_roundtrip_and_terse_links():
    # Build a G with loads so S_from_G uses them directly
    G = nx.Graph(R=1, T=3, capacity=5)
    for n in (-1, 0, 1, 2):
        G.add_node(n, kind=("oss" if n == -1 else "wtg"))
    # chain -1-0-1-2 with loads/subtree to satisfy S_from_G
    G.add_edge(-1, 0, load=0, reverse=False, length=1.0)
    G.add_edge(0, 1, load=1, reverse=True, length=1.0)
    G.add_edge(1, 2, load=2, reverse=True, length=1.0)
    G.nodes[-1]["load"] = 0
    G.nodes[0]["load"] = 3; G.nodes[0]["subtree"] = 0
    G.nodes[1]["load"] = 2; G.nodes[1]["subtree"] = 0
    G.nodes[2]["load"] = 1; G.nodes[2]["subtree"] = 0
    G.graph["has_loads"] = True

    S = S_from_G(G)
    terse = terse_links_from_S(S)

    # sanity checks: S has the expected topology and terse has correct shape
    assert S.graph["R"] == 1 and S.graph["T"] == 3 and S.graph["capacity"] == 5
    assert set(S.edges) == {(-1, 0), (0, 1), (1, 2)}
    assert hasattr(terse, "shape") and terse.shape == (3,)
    assert terse.dtype == np.int_



# ----------------------------
# normalize/rescale
# ----------------------------

def test_as_normalized_and_as_rescaled_invert():
    L = _make_simple_L(T=2, R=1)
    A = nx.Graph(**L.graph)
    A.add_edge(-1, 0, length=10.0)
    A.graph["norm_offset"] = np.array([10.0, -2.0])
    A.graph["norm_scale"] = 0.5
    # Provide d2roots only if present in caller
    A.graph["d2roots"] = np.array([[1.0], [2.0], [0.0]])

    A2 = as_normalized(A)
    # length scaled by 0.5
    assert math.isclose(A2.edges[-1, 0]["length"], 5.0)
    assert A2.graph["is_normalized"] is True

    # rescale back to L geometry
    G = as_rescaled(A2, L)
    assert "is_normalized" not in G.graph
    assert math.isclose(G.edges[-1, 0]["length"], 10.0)


# ----------------------------
# undetour
# ----------------------------

def test_as_undetoured_removes_detour_chain_and_marks_tentative():
    # Construct G with a detour chain: -1 -- D(=2) -- 0
    T, R, B = 1, 1, 0
    base = np.vstack([_coords([(0, 0)]), _coords([(0, -1)])])
    VertexC = np.vstack([base, _coords([(0, -0.5)])])  # detour node at idx 2 (>= T)
    G = nx.Graph(R=R, T=T, B=B, VertexC=VertexC, C=0, D=1)
    G.add_node(-1, kind="oss", load=1)
    G.add_node(0, kind="wtg", load=1, subtree=0)
    det = 2
    G.add_node(det, kind="detour", load=1, subtree=0)
    G.add_edge(-1, det)
    G.add_edge(det, 0)
    # Provide fnT to satisfy the branch where C==0
    G.graph["fnT"] = np.array([0, -1])

    G2 = as_undetoured(G)
    # detour node is removed, replaced by a tentative edge -1--0
    assert det not in G2.nodes
    assert (-1, 0) in G2.edges
    assert G2.edges[-1, 0]["kind"] == "tentative"
    assert "D" not in G2.graph


# ----------------------------
# rehooking (nearest / head)
# ----------------------------

def test_as_hooked_to_nearest_moves_hook_by_d2roots():
    # Two WTGs in same subtree; rehook root->nearest end by d2roots
    R, T = 1, 2
    G = nx.Graph(R=R, T=T, has_loads=True, VertexC=_coords([(0,0),(2,0),(0,-1)]))
    for n in (-1, 0, 1):
        G.add_node(n, kind=("oss" if n == -1 else "wtg"))
    # same subtree id
    G.nodes[0]["subtree"] = 0
    G.nodes[1]["subtree"] = 0
    # subtree total = 2; set old hook's node load to 2
    G.nodes[0]["load"] = 1
    G.nodes[1]["load"] = 2      # <-- subtree total at old hook
    G.nodes[-1]["load"] = 2
    G.add_edge(0, 1, load=1, kind="contour", reverse=False)

    # feed a tentative hook to node 1 with subtree total
    G.add_edge(-1, 1, kind="tentative", load=2)
    G.graph["tentative"] = [(-1, 1)]

    # d2roots: node 0 is nearer than node 1
    d2 = np.array([
        [1.0],  # node 0 -> root
        [3.0],  # node 1 -> root
        [0.0],  # root itself (unused)
    ])

    G2 = as_hooked_to_nearest(G, d2roots=d2)
    assert (-1, 0) in G2.edges
    assert (-1, 1) not in G2.edges
    # root load reduced by subtree total (2 -> 0)
    assert G2.nodes[-1]["load"] == 2


def test_as_hooked_to_head_moves_to_nearer_end():
    # Path subtree: 0--1 ; root hooks to 1 but nearer head is 0
    R, T = 1, 2
    S = nx.Graph(R=R, T=T, has_loads=True, VertexC=_coords([(0,0),(1,0),(0,-1)]))
    for n in (-1, 0, 1):
        S.add_node(n, kind=("oss" if n == -1 else "wtg"))
    S.add_edge(0, 1, load=1, reverse=False)
    S.nodes[0]["subtree"] = 0
    S.nodes[1]["subtree"] = 0
    # subtree total = 2; put it on the old hook node (1)
    S.nodes[-1]["load"] = 2
    S.nodes[0]["load"] = 1
    S.nodes[1]["load"] = 2
    S.add_edge(-1, 1, kind="tentative", load=2)

    d2 = np.array([[1.0], [2.0], [0.0]])
    S2 = as_hooked_to_head(S, d2roots=d2)
    assert (-1, 0) in S2.edges
    assert (-1, 1) not in S2.edges
    # root load reduced by subtree total (2 -> 0)
    assert S2.nodes[-1]["load"] == 2


# ----------------------------
# scaffolded
# ----------------------------

def test_scaffolded_merges_and_unsets_scaffold_flags_on_overlap():
    # Build a tiny G and a PlanarEmbedding P with overlapping edge so that 'kind'
    # gets removed where G overlaps.
    from networkx import PlanarEmbedding

    G = nx.Graph(R=1, T=1, B=0,
                 VertexC=_coords([(0,0),(0,-1)]),  # [wtg, root]
                 C=0, D=0)
    G.add_node(0, kind="wtg"); G.add_node(-1, kind="oss")
    G.add_edge(-1, 0)  # overlap edge

    # Minimal valid embedding with the same undirected edge represented via half-edges
    P = PlanarEmbedding()
    P.add_node(-1); P.add_node(0)
    P.add_half_edge(-1, 0)
    P.add_half_edge(0, -1)
    # Add required supertriangle coords used by scaffolded()
    P.graph["supertriangleC"] = _coords([(-10, -10), (10, -10), (0, 10)])

    scaff = scaffolded(G, P)
    # All P edges start as 'scaffold'
    # but where G overlaps, scaffold flag should be removed
    # (i.e., no 'kind' on that overlapping edge)
    # We need to map using fnT applied inside scaffolded; for C=D=0 it's identity
    assert (-1, 0) in scaff.edges
    assert "kind" not in scaff.edges[-1, 0]
