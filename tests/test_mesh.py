# imports
import os

import numpy as np
import networkx as nx
import condeltri as cdt

from optiwindnet.mesh import (
    _edges_and_hull_from_cdt,
    A_graph,
)
from .helpers import tiny_wfn

# disable Numba JIT for CI/test determinism if available
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
try:
    import numba

    numba.config.DISABLE_JIT = True
except Exception:
    pass

# ----------------------- tests -----------------------
def test_make_planar_embedding_basic():
    wfn = tiny_wfn()

    P = wfn.P
    A = wfn.A
    assert isinstance(P, nx.PlanarEmbedding)
    assert isinstance(A, nx.Graph)
    assert A.number_of_nodes() > 0
    assert A.number_of_edges() > 0

    # check basic keys in A
    for key in ("T", "R", "B", "VertexC", "hull"):
        assert key in A.graph


def test_A_graph_all(monkeypatch):
    # use the tiny default wfn (no optimize)
    wfn = tiny_wfn(optimize=False)
    L = wfn.L

    # Delaunay-based, no weightfun
    A_d = A_graph(L, delaunay_based=True, weightfun=None)
    assert A_d.number_of_edges() > 0
    assert all(d.get("length", 0.0) > 0.0 for _, _, d in A_d.edges(data=True))
    assert all("weight" not in d for _, _, d in A_d.edges(data=True))

    # Complete graph path with custom weight function that sets 'cost'
    def cost_fn(_edge):
        return 42.0

    A_c = A_graph(L, delaunay_based=False, weightfun=cost_fn, weight_attr="cost")
    assert A_c.number_of_edges() > 0
    assert all(d.get("cost") == 42.0 for _, _, d in A_c.edges(data=True))

    # When weightfun is provided on the Delaunay path, apply_edge_exemptions should be called.
    calls = {"n": 0}
    import optiwindnet.mesh as mesh_mod

    def fake_apply_edge_exemptions(a):
        calls["n"] += 1
        return a

    monkeypatch.setattr(mesh_mod, "apply_edge_exemptions", fake_apply_edge_exemptions)

    A_dw = A_graph(L, delaunay_based=True, weightfun=lambda e: 1.23)
    assert calls["n"] == 1
    assert all("weight" in d for _, _, d in A_dw.edges(data=True))


def test_edges_and_hull_from_cdt_all():
    mesh = cdt.Triangulation(
        cdt.VertexInsertionOrder.AUTO,
        cdt.IntersectingConstraintEdges.NOT_ALLOWED,
        0.0,
    )
    pts = [cdt.V2d(0.0, 0.0), cdt.V2d(1.0, 0.0), cdt.V2d(1.0, 1.0), cdt.V2d(0.0, 1.0)]
    mesh.insert_vertices(pts)

    triangles = mesh.triangles
    vertmap = np.arange(3 + len(pts), dtype=int)

    ebunch, hull = _edges_and_hull_from_cdt(triangles, vertmap)
    assert isinstance(ebunch, list) and len(ebunch) > 0
    assert isinstance(hull, list) and len(hull) > 0
    assert all(isinstance(n, (int, np.integer)) for n in hull)
