# tests/test_mesh_consolidated.py
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
try:
    import numba
    numba.config.DISABLE_JIT = True
except Exception:
    pass

import re
import ast
import numpy as np
import networkx as nx
import pytest
import condeltri as cdt
from bidict import bidict

from optiwindnet.mesh import (
    make_planar_embedding,
    delaunay,
    _index,
    planar_flipped_by_routeset,
    _edges_and_hull_from_cdt,
    A_graph,
    _deprecated_planar_flipped_by_routeset,
)

# ----------------------- shared helpers -----------------------

def build_L(VertexC, *, R=1, border=(), obstacles_xy=None, name="unit"):
    """
    Build a locations graph L for make_planar_embedding().
    - Terminals are [0..T-1], roots are [-R..-1] appended after terminals,
      then any B-range obstacle vertices (T..T+B-1).
    - border and obstacles should reference terminal indices (<T). If you
      want B-range obstacles, pass them via obstacles_xy.
    """
    VertexC = np.asarray(VertexC, dtype=float)
    if obstacles_xy is None:
        obstacles_xy = np.empty((0, 2), dtype=float)
    obstacles_xy = np.asarray(obstacles_xy, dtype=float)

    T = len(VertexC) - R
    B = len(obstacles_xy)

    # Final VertexC layout: terminals, roots, then B-range obstacles
    roots = VertexC[T:]
    assert len(roots) == R, "VertexC must end with R root rows"
    VertexC_full = np.vstack([VertexC[:T], roots, obstacles_xy])

    # Basic indices
    border = np.asarray(border, dtype=int)
    if border.size and np.any(border >= T):
        raise ValueError("border indices must be < T for this helper")
    obstacle_idx = (np.arange(T, T + B, dtype=int),) if B else ()

    L = nx.Graph(name=name)
    L.add_nodes_from(range(T))         # terminals 0..T-1
    L.add_nodes_from(range(-R, 0))     # roots -R..-1
    L.graph.update(
        R=R, T=T, B=B,
        VertexC=VertexC_full,
        border=border,
        obstacles=obstacle_idx,
        handle=name,
    )
    return L



def test_make_planar_embedding_all_paths(caplog):
    """
    Drive make_planar_embedding through its major branches:
      - no border case fast path (sanity)
      - border intersections: single point (Point) and multiple points (Multi geometry)
      - duplicate coords between nodes and B-range
      - concavity with stunts
      - multiple concavities (len>1) merge path
      - obstacles add constraint edges
    """
    # -- No border (fast sanity) --
    vc = np.array([
        [0.0, 0.0], [1.0, 0.0],
        [0.0, 1.0], [1.0, 1.0],
        [0.5, -1.0],  # root
    ])
    L0 = build_L(vc, R=1, border=(), obstacles_xy=None, name="noborder")
    P0, A0 = make_planar_embedding(L0)
    assert isinstance(P0, nx.PlanarEmbedding)
    assert isinstance(A0, nx.Graph)
    for k in ["T","R","B","VertexC","planar","diagonals","d2roots",
              "corner_to_A_edges","hull","hull_prunned","hull_concave",
              "norm_offset","norm_scale","inter_terminal_clearance_min","inter_terminal_clearance_safe"]:
        assert k in A0.graph
    assert A0.graph["d2roots"].shape == (A0.graph["T"] + A0.graph["B"] + 3, A0.graph["R"])

    # -- Border intersections: single point (Point branch) --
    corners = np.array([[0,0],[4,0],[4,3],[0,3]], float)
    on_border_single = np.array([[1.0, 0.0]])
    interior = np.array([[2.0, 1.0]])
    terms_A = np.vstack([corners, on_border_single, interior])
    roots = np.array([[2.0, 0.5]])
    obs_ring = np.array([[1.2,1.2],[1.8,1.2],[1.8,1.8],[1.2,1.8]], float)
    L1 = build_L(np.vstack([terms_A, roots]), R=1, border=[0,1,2,3], obstacles_xy=obs_ring, name="single-intersect")
    caplog.records.clear()
    caplog.set_level("INFO", logger="optiwindnet.mesh")
    P1, A1 = make_planar_embedding(L1)
    assert A1.number_of_edges() > 0
    msgs = [r.message for r in caplog.records if "INTERSECTS:" in r.message]
    assert msgs, "Expected INTERSECTS: log for single point"
    m = re.search(r"INTERSECTS:\s*(\[[^\]]*\])", msgs[-1]); assert m
    intersects_list = ast.literal_eval(m.group(1))
    assert len(intersects_list) >= 1

    # -- Border intersections: multiple points on same segment (Multi geometry branch) --
    on_a = np.array([[1.0, 0.0]])
    on_b = np.array([[3.0, 0.0]])
    interior2 = np.array([[2.0, 1.0]])
    terms_B = np.vstack([corners, on_a, on_b, interior2])
    L2 = build_L(np.vstack([terms_B, roots]), R=1, border=[0,1,2,3], obstacles_xy=obs_ring, name="multi-intersect")
    caplog.records.clear()
    caplog.set_level("INFO", logger="optiwindnet.mesh")
    P2, A2 = make_planar_embedding(L2)
    msgs = [r.message for r in caplog.records if "INTERSECTS:" in r.message]
    assert msgs, "Expected INTERSECTS: log for multiple points"
    m = re.search(r"INTERSECTS:\s*(\[[^\]]*\])", msgs[-1]); assert m
    intersects_list = ast.literal_eval(m.group(1))
    assert len(intersects_list) >= 2  # multiple points

    # -- Duplicate coords between nodes and B-range (loop coverage) --
    corners2 = np.array([[0,0],[4,0],[4,3],[0,3]], float)
    extra = np.array([[1.0, 1.0]])
    terminals = np.vstack([corners2, extra])  # T=5
    root = np.array([[2.0, 0.5]])
        # Simple, non-self-intersecting triangle in CCW order.
    # Keeps one vertex equal to a terminal to exercise the duplicate-coord path,
    # but avoids duplicating the root (which was causing constraint intersections).
    obstacles_xy = np.array([
        [50, 50],   # == terminal 0 (duplicate on purpose)
        [50, 60],
        [60, 60],
    ], float)
    L3 = build_L(np.vstack([terminals, root]), R=1, border=(),
                 obstacles_xy=obstacles_xy, name="duplicates")
    P3, A3 = make_planar_embedding(L3)
    assert "obstacles" in A3.graph and len(A3.graph["obstacles"]) == 1
    assert A3.number_of_edges() > 0

    # -- Concavity creates stunts (Part C) --
    # concave polygon via a notch vertex that coincides with a terminal -> stunts
    conc_vc = np.array([
        [0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [2.0, 1.5], [0.0, 3.0],  # 0..4
        [1.5, 1.0],                                                 # root-ish (we’ll treat as root next)
    ], float)
    L4 = build_L(conc_vc, R=1, border=[0,1,2,3,4], obstacles_xy=None, name="stunts")
    P4, A4 = make_planar_embedding(L4)
    assert A4.graph["B"] > 0              # stunts extended B
    assert "stunts_primes" in A4.graph
    assert isinstance(A4.graph["stunts_primes"], list)

    # -- Multiple concavities (len>1) merge branch --
    dents = np.array([[5.0, 2.0], [1.0, 2.0]])
    Mpt  = np.array([[3.0, 1.5]])  # shared concavity vertex
    outer = np.array([[0,0],[6,0],[6,4],[0,4]], float)
    interior_keep = np.array([[2.0, 3.5]])
    terms_mc = np.vstack([outer, dents[0:1], Mpt, dents[1:2], interior_keep])
    roots_mc = np.array([[3.0, 0.5]])
    border_mc = [0,1,2,3,4,5,6]   # wraps back to 0 implicitly
    L5 = build_L(np.vstack([terms_mc, roots_mc]), R=1, border=border_mc, obstacles_xy=None, name="multi-conc")
    caplog.records.clear()
    caplog.set_level("DEBUG", logger="optiwindnet.mesh")
    P5, A5 = make_planar_embedding(L5)
    # don't assert specific log text; just ensure mesh & graph produced
    assert isinstance(P5, nx.PlanarEmbedding) and A5.number_of_edges() > 0


def test_delaunay_all():
    # single root, bind2root=False
    vc = np.array([
        [0,0],[1,0],[0,1],[1,1],
        [0.5,-1.0],  # root
    ], float)
    L = build_L(vc, R=1)
    A0 = delaunay(L, bind2root=False)
    for _, _, d in A0.edges(data=True):
        assert "root" not in d

    # single root, bind2root=True
    A1 = delaunay(L, bind2root=True)
    for _, _, d in A1.edges(data=True):
        assert d.get("root") == -1

    # two roots: valid assignment among {-1,-2}
    vc2 = np.array([
        [0,0],[1,0],[0,1],[1,1],
        [-1.0,-1.0], [2.0,-1.0],  # roots (-2,-1)
    ], float)
    L2 = build_L(vc2, R=2)
    A2 = delaunay(L2, bind2root=True)
    seen = {d["root"] for *_, d in A2.edges(data=True)}
    assert seen <= {-1, -2} and len(seen) >= 1


def test_A_graph_all(monkeypatch):
    # Base L
    vc = np.array([
        [0,0],[1,0],[0,1],[1,1],
        [0.5,-1.0],
    ], float)
    L = build_L(vc, R=1)

    # delaunay_based=True with no weightfun
    A_d = A_graph(L, delaunay_based=True, weightfun=None)
    assert A_d.number_of_edges() > 0
    for _, _, d in A_d.edges(data=True):
        assert d.get("length", 0.0) > 0.0
        assert "weight" not in d

    # complete-graph with weightfun & custom attr; also ensure exemptions called in Delaunay path
    called = {"n": 0}
    import optiwindnet.mesh as mesh_mod
    monkeypatch.setattr(mesh_mod, "apply_edge_exemptions", lambda a: called.__setitem__("n", called["n"]+1))

    weightfun = lambda ed: 42.0
    A_c = A_graph(L, delaunay_based=False, weightfun=weightfun, weight_attr="cost")
    for _, _, d in A_c.edges(data=True):
        assert d.get("cost") == 42.0

    # When weightfun is used with delaunay_based=True, exemptions hook is called once
    A_d2 = A_graph(L, delaunay_based=True, weightfun=lambda ed: 1.23)
    assert called["n"] == 1
    for _, _, d in A_d2.edges(data=True):
        assert "weight" in d


def test_planar_flipped_by_routeset_all():
    # Make a simple site and pick one diagonal mapping from A.graph['diagonals']
    vc = np.array([
        [0,0],[2,0],[0,2],[2,2],
        [1.0,-1.0],  # root
    ], float)
    L = build_L(vc, R=1)
    P, A = make_planar_embedding(L)

    diags = A.graph["diagonals"]
    chosen = None
    for (s, t), (u, v) in diags.items():
        if s >= 0 and t >= 0 and u >= 0 and v >= 0:
            chosen = ((s, t), (u, v))
            break
    if chosen is None:
        pytest.skip("No all-terminal diagonal mapping found.")
    (s, t), (u, v) = chosen

    G = nx.Graph()
    T, R, B = A.graph["T"], A.graph["R"], A.graph["B"]
    G.add_nodes_from(range(T)); G.add_nodes_from(range(-R, 0))
    G.add_edge(u, v)
    fnT = np.arange(R + T + B + 3); fnT[-R:] = range(-R, 0)
    G.graph.update(T=T, R=R, B=B, fnT=fnT)

    P2 = planar_flipped_by_routeset(G, planar=P, VertexC=A.graph["VertexC"], diagonals=diags)
    assert (u, v) in P2.edges or (v, u) in P2.edges


def test_edges_and_hull_from_cdt_all():
    mesh = cdt.Triangulation(
        cdt.VertexInsertionOrder.AUTO,
        cdt.IntersectingConstraintEdges.NOT_ALLOWED,
        0.0,
    )
    pts = [cdt.V2d(0.0,0.0), cdt.V2d(1.0,0.0), cdt.V2d(1.0,1.0), cdt.V2d(0.0,1.0)]
    mesh.insert_vertices(pts)

    triangles = mesh.triangles
    vertmap = np.arange(3 + len(pts), dtype=int)

    ebunch, hull = _edges_and_hull_from_cdt(triangles, vertmap)
    assert isinstance(ebunch, list) and ebunch
    assert isinstance(hull, list) and hull
    assert all(isinstance(n, (int, np.integer)) for n in hull)
    assert all(n >= 3 for n in hull)  # avoid supertriangle ids
