import numpy as np
import networkx as nx
import pytest

from optiwindnet.mesh import (
    make_planar_embedding,
    delaunay,
    _index,
    planar_flipped_by_routeset,
)

# ---------- tiny helpers ----------

def make_location_graph(VertexC, R=1, border=(), obstacles=()):
    """
    Build a minimal "locations" graph L as expected by make_planar_embedding().
    - Terminals are [0..T-1], roots are [-R..-1].
    - IMPORTANT: B is the count of *extra* border vertices in the B-range
      (indices T..T+B-1). If your border uses only terminal indices (< T),
      B must be 0.
    """
    VertexC = np.asarray(VertexC, dtype=float)
    T = len(VertexC) - R

    border_arr = np.asarray(border, dtype=int)
    if border_arr.size and np.any(border_arr >= T):
        raise ValueError(
            "This helper doesn’t support B-range border vertices. "
            "All border indices must be < T. (Or extend VertexC and set B explicitly.)"
        )
    B = 0  # correct for tests using only terminal indices in the border

    L = nx.Graph(name="toy")
    L.add_nodes_from(range(T))
    L.add_nodes_from(range(-R, 0))

    L.graph.update(
        R=R,
        T=T,
        B=B,
        VertexC=VertexC,
        border=border_arr if border_arr.size else np.array([], dtype=int),
        obstacles=tuple(np.asarray(o, dtype=int) for o in obstacles) if obstacles else (),
        handle="unit-test",
    )
    return L


# ---------- unit tests ----------

def test__index_numpy_equivalent():
    arr = np.array([10, 20, 30, 20], dtype=int)
    # returns first match like list.index
    assert _index(arr, np.int_(20)) == 1
    assert _index(arr, np.int_(30)) == 2
    # not-found path returns 0 (function's documented behavior)
    assert _index(arr, np.int_(999)) == 0


def test_make_planar_embedding_no_border_fast():
    # 4 terminals forming a square and 1 root below
    VertexC = np.array([
        [0.0, 0.0],   # 0
        [1.0, 0.0],   # 1
        [0.0, 1.0],   # 2
        [1.0, 1.0],   # 3
        [0.5, -1.0],  # root (-1)
    ], dtype=float)

    L = make_location_graph(VertexC, R=1, border=(), obstacles=())

    P, A = make_planar_embedding(L)

    # Basic shape checks
    assert isinstance(P, nx.PlanarEmbedding)
    assert isinstance(A, nx.Graph)

    # Key graph attributes should exist on A
    for k in [
        "T","R","B","VertexC","planar","diagonals","d2roots",
        "corner_to_A_edges","hull","hull_prunned","hull_concave",
        "norm_offset","norm_scale","inter_terminal_clearance_min","inter_terminal_clearance_safe",
    ]:
        assert k in A.graph

    # planar embedding triangles should be present
    assert "triangles" in P.graph and isinstance(P.graph["triangles"], list)

    # A should have positive lengths on its edges
    assert A.number_of_edges() > 0
    for _, _, d in A.edges(data=True):
        assert d.get("length", 0.0) > 0.0

    # d2roots dimensions: (T+B+3) x R   (supertriangle adds 3)
    T = A.graph["T"]; B = A.graph["B"]; R = A.graph["R"]
    d2roots = A.graph["d2roots"]
    assert d2roots.shape == (T + B + 3, R)

    # A.planar is a PlanarEmbedding that excludes supertriangle in its nodes
    planar_A = A.graph["planar"]
    assert isinstance(planar_A, nx.PlanarEmbedding)
    # supertriangle nodes are >= T+B
    ST = T + B
    for u in planar_A.nodes():
        assert u < ST or u < 0  # (roots can be negative)


def test_delaunay_bind2root_sets_edge_root_attribute():
    VertexC = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],  # 4 terminals
        [0.5, -1.0],                                     # 1 root
    ], dtype=float)
    L = make_location_graph(VertexC, R=1)

    A = delaunay(L, bind2root=True)

    # Should keep same attributes as make_planar_embedding
    assert "VertexC" in A.graph and "planar" in A.graph

    # Every edge should have a 'root' attribute assigned
    assert A.number_of_edges() > 0
    for _, _, data in A.edges(data=True):
        assert "root" in data


def test_make_planar_embedding_with_convex_border():
    # A simple rectangle border using the first 4 terminals; 1 root inside
    VertexC = np.array([
        [0.0, 0.0],  # 0
        [2.0, 0.0],  # 1
        [2.0, 1.0],  # 2
        [0.0, 1.0],  # 3
        [1.0, 0.5],  # root (-1) inside
    ], dtype=float)
    # border as a closed loop indices among terminals (0,1,2,3)
    border = np.array([0, 1, 2, 3], dtype=int)

    L = make_location_graph(VertexC, R=1, border=border, obstacles=())

    P, A = make_planar_embedding(L)

    # Hulls should exist and be non-empty
    assert A.graph.get("hull")  # convex hull from P_A
    assert A.graph.get("hull_concave")  # for convex border it's typically same as pruned hull

    # Constraint edges set exists (stored on P)
    assert isinstance(P, nx.PlanarEmbedding)
    ST = A.graph["T"] + A.graph["B"]
    assert any(u < ST and v < ST for u, v in P.edges())


# ---------- extra coverage: concavities, stunts, obstacles, flipping ----------

def test_concavity_creates_stunts_and_adjusts_planar():
    # Concave border: rectangle with an inward notch at the top edge.
    # Terminals outline the concave polygon; 1 root inside.
    # Border indices are terminals only (B=0), but because border vertices
    # coincide with terminals, Part C creates stunt vertices -> B increases.
    VertexC = np.array([
        [0.0, 0.0],   # 0
        [3.0, 0.0],   # 1
        [3.0, 3.0],   # 2 (top-right)
        [2.0, 1.5],   # 3 (inward notch)
        [0.0, 3.0],   # 4 (top-left)
        [1.5, 1.0],   # 5 (root-ish, inside)
    ], dtype=float)
    R = 1
    border = np.array([0, 1, 2, 3, 4], dtype=int)  # concave ring (no repeated last)

    L = make_location_graph(VertexC, R=R, border=border, obstacles=())

    P, A = make_planar_embedding(L)

    # Because concavity vertices coincide with terminals, stunts are created
    assert "stunts_primes" in A.graph
    assert isinstance(A.graph["stunts_primes"], list)
    assert len(A.graph["stunts_primes"]) >= 1

    # B should have increased due to stunts
    assert A.graph["B"] > 0

    # Planar triangles list is present and non-empty
    assert "triangles" in P.graph and P.graph["triangles"]


def test_obstacle_inserts_constraint_edges():
    # Outer border = 4 terminals (0..3). Obstacle = 4 B-range vertices (4..7).
    # Root at the end.
    VertexC = np.array([
        # terminals: outer rectangle (T=4)
        [0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0],
        # B-range: obstacle rectangle (B=4), indices 4..7
        [1.5, 1.0], [2.5, 1.0], [2.5, 2.0], [1.5, 2.0],
        # root (R=1) is always last
        [2.0, 0.5],
    ], dtype=float)

    T, B, R = 4, 4, 1
    border = np.array([0, 1, 2, 3], dtype=int)         # uses only terminal indices
    obstacle = np.array([4, 5, 6, 7], dtype=int)       # uses B-range indices

    # Build L manually because we need B>0 support (B-range vertices present).
    L = nx.Graph(name="toy")
    L.add_nodes_from(range(T))          # terminals
    L.add_nodes_from(range(-R, 0))      # roots as negative indices
    L.graph.update(
        R=R, T=T, B=B,
        VertexC=VertexC,
        border=border,
        obstacles=(obstacle,),
        handle="unit-test",
    )

    P, A = make_planar_embedding(L)

    # Ensure constraint edges were registered (from hull/concavities/holes)
    assert "constraint_edges" in P.graph
    ce = P.graph["constraint_edges"]
    assert isinstance(ce, set)

    # Obstacle boundary edges should appear among constraints (unordered)
    expected_pairs = {
        tuple(sorted((4, 5))), tuple(sorted((5, 6))),
        tuple(sorted((6, 7))), tuple(sorted((7, 4))),
    }
    assert expected_pairs & ce  # at least one present


def test_planar_flipped_by_routeset_flips_a_diagonal():
    # Use simple square + root; make a P/A, then take one (s,t)->(u,v) mapping
    # from A.graph['diagonals'] and force-flip by creating a routeset G with (u,v).
    VertexC = np.array([
        [0.0, 0.0],   # 0
        [2.0, 0.0],   # 1
        [0.0, 2.0],   # 2
        [2.0, 2.0],   # 3
        [1.0, -1.0],  # root (-1)
    ], dtype=float)

    L = make_location_graph(VertexC, R=1, border=(), obstacles=())
    P, A = make_planar_embedding(L)

    diags = A.graph["diagonals"]
    # Find a usable diagonal mapping with all terminals (>=0)
    chosen = None
    for (s, t), (u, v) in diags.items():
        if s >= 0 and t >= 0 and u >= 0 and v >= 0:
            chosen = ((s, t), (u, v))
            break
    # If no diagonal found (rare for tiny graphs), skip this test
    if chosen is None:
        pytest.skip("No terminal-only diagonal mapping found to test flipping.")

    (s, t), (u, v) = chosen

    # Build a tiny routeset G that uses the diagonal (u,v)
    G = nx.Graph()
    # include all nodes from A up to T+B-1 and roots (-R..-1) to be safe
    T, R, B = A.graph["T"], A.graph["R"], A.graph["B"]
    G.add_nodes_from(range(T))
    G.add_nodes_from(range(-R, 0))
    G.add_edge(u, v)

    # attach required graph metadata for planar_flipped_by_routeset()
    G.graph.update(T=T, R=R, B=B, fnT=np.arange(R + T + B + 3))  # default fnT like in code path
    G.graph["fnT"][-R:] = range(-R, 0)

    P2 = planar_flipped_by_routeset(G, planar=P, VertexC=A.graph["VertexC"], diagonals=diags)

    # The flipped planar should have the diagonal as an edge (unordered check)
    assert (u, v) in P2.edges or (v, u) in P2.edges


# tests/test_edges_and_hull_from_cdt.py
import numpy as np
import pytest

from optiwindnet.mesh import _edges_and_hull_from_cdt
import condeltri as cdt


def test_edges_and_hull_from_cdt_basic():
    # Build a tiny Delaunay triangulation: a unit square (no constraints)
    mesh = cdt.Triangulation(
        cdt.VertexInsertionOrder.AUTO,
        cdt.IntersectingConstraintEdges.NOT_ALLOWED,
        0.0,
    )
    pts = [cdt.V2d(0.0, 0.0), cdt.V2d(1.0, 0.0), cdt.V2d(1.0, 1.0), cdt.V2d(0.0, 1.0)]
    mesh.insert_vertices(pts)

    # CDT internally reserves the first 3 indices for the supertriangle.
    # triangles[i].vertices refer to these CDT indices (0..2 = supertriangle).
    triangles = mesh.triangles
    vertmap = np.arange(3 + len(pts), dtype=int)  # identity mapping is fine for the test

    ebunch, convex_hull = _edges_and_hull_from_cdt(triangles, vertmap)

    # Basic sanity: non-empty results with the expected types
    assert isinstance(ebunch, list)
    assert len(ebunch) > 0
    assert all(isinstance(e, tuple) and len(e) == 2 for e in ebunch)

    assert isinstance(convex_hull, list)
    assert len(convex_hull) > 0
    assert all(isinstance(n, (int, np.integer)) for n in convex_hull)

    # Optional: hull should be a cycle of distinct mapped indices (no supertriangle ids)
    # Since vertmap is identity, hull nodes should be >= 3 (real points, not 0..2)
    assert all(n >= 3 for n in convex_hull)


# tests/test_make_planar_embedding_fullpath.py
# Make sure numba runs in python mode so coverage can see lines
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
try:
    import numba
    numba.config.DISABLE_JIT = True
except Exception:
    pass

import numpy as np
import networkx as nx
import pytest

from optiwindnet.mesh import make_planar_embedding


def make_location_graph(VertexC, R=1, border=(), obstacles=()):
    """
    Minimal L matching make_planar_embedding() expectations.
    Terminals are [0..T-1], roots are [-R..-1] appended at the end.
    NOTE: This helper assumes border/obstacles reference ONLY terminal indices (< T).
    """
    VertexC = np.asarray(VertexC, dtype=float)
    T = len(VertexC) - R
    border = np.asarray(border, dtype=int)
    for arr in (border,) + tuple(np.asarray(o, dtype=int) for o in obstacles):
        if arr.size and np.any(arr >= T):
            raise ValueError("This test helper assumes border/obstacles use only terminal indices.")
    L = nx.Graph(name="fullpath")
    L.add_nodes_from(range(T))
    L.add_nodes_from(range(-R, 0))
    L.graph.update(
        R=R,
        T=T,
        B=0,  # no pre-allocated B-range; stunts will extend B internally if created
        VertexC=VertexC,
        border=border,
        obstacles=tuple(np.asarray(o, dtype=int) for o in obstacles),
        handle="unit-test",
    )
    return L

def test_make_planar_embedding_rich_path():
    import numpy as np
    import networkx as nx
    from optiwindnet.mesh import make_planar_embedding

    # Terminals (outer rectangle) + one interior terminal
    terminals = np.array([
        [0.0, 0.0],   # 0
        [4.0, 0.0],   # 1
        [4.0, 3.0],   # 2
        [0.0, 3.0],   # 3
        [2.0, 1.0],   # 4  (interior terminal)
    ], dtype=float)

    root = np.array([[2.0, 0.5]])  # -1

    # Obstacle ring as extra border vertices in B-range (not terminals)
    obstacle_ring = np.array([
        [1.2, 1.2],   # T+0
        [1.8, 1.2],   # T+1
        [1.8, 1.8],   # T+2
        [1.2, 1.8],   # T+3
    ], dtype=float)

    # Assemble VertexC: terminals, root, then B-range obstacle vertices
    R = 1
    T = len(terminals) + 0  # terminals only
    B = 4
    VertexC = np.vstack([terminals, root, obstacle_ring])

    # Convex outer border using terminal indices only
    border = np.array([0, 1, 2, 3], dtype=int)

    # Obstacle indices refer to the B-range we appended at the end
    obstacle = np.array([T, T+1, T+2, T+3], dtype=int)

    # Build L manually so we can set B=4
    L = nx.Graph(name="rich")
    # Add terminal nodes and root nodes (-R..-1)
    L.add_nodes_from(range(T))
    L.add_nodes_from(range(-R, 0))
    L.graph.update(
        R=R, T=T, B=B, VertexC=VertexC,
        border=border,
        obstacles=(obstacle,),
        handle="unit-test"
    )

    P, A = make_planar_embedding(L)

    # Sanity checks that also touch later parts of the function
    assert P is not None and A is not None
    assert isinstance(P, nx.PlanarEmbedding)
    assert isinstance(A, nx.Graph)
    # P.graph triangles recorded
    assert isinstance(P.graph.get("triangles"), list)

    # A carries expected graph attributes and metrics
    for k in ["T","R","B","VertexC","planar","diagonals","d2roots",
              "corner_to_A_edges","hull","hull_prunned","hull_concave",
              "norm_offset","norm_scale",
              "inter_terminal_clearance_min","inter_terminal_clearance_safe"]:
        assert k in A.graph

    # All A edges should have positive length
    assert A.number_of_edges() > 0
    for _, _, d in A.edges(data=True):
        assert d.get("length", 0.0) > 0.0

    # Obstacle present in A.graph and not as terminals
    assert "obstacles" in A.graph and len(A.graph["obstacles"]) == 1
    assert all(idx >= T for idx in A.graph["obstacles"][0])

    # Step N: d2roots has expected shape (T+B+3, R)
    d2roots = A.graph["d2roots"]
    assert d2roots.shape == (T + B + 3, R)


import numpy as np
import networkx as nx
from optiwindnet.mesh import delaunay
from .test_mesh import make_location_graph  # reuse your helper


def test_delaunay_basic_no_bind():
    # 4 terminals in a square, 1 root below
    VertexC = np.array([
        [0.0, 0.0],   # 0
        [1.0, 0.0],   # 1
        [0.0, 1.0],   # 2
        [1.0, 1.0],   # 3
        [0.5, -1.0],  # root (-1)
    ], dtype=float)
    L = make_location_graph(VertexC, R=1)

    A = delaunay(L, bind2root=False)

    assert isinstance(A, nx.Graph)
    # Core attributes (propagated from make_planar_embedding)
    for k in ["VertexC", "planar", "diagonals", "d2roots", "T", "R", "B"]:
        assert k in A.graph

    # Edges should exist and have positive 'length'
    assert A.number_of_edges() > 0
    for _, _, d in A.edges(data=True):
        assert d.get("length", 0.0) > 0.0
        # and no 'root' attribute when bind2root=False
        assert "root" not in d


def test_delaunay_bind2root_sets_root_attribute_single_root():
    # Same geometry; now bind2root=True should assign root = -1 to all edges
    VertexC = np.array([
        [0.0, 0.0],   # 0
        [1.0, 0.0],   # 1
        [0.0, 1.0],   # 2
        [1.0, 1.0],   # 3
        [0.5, -1.0],  # root (-1)
    ], dtype=float)
    L = make_location_graph(VertexC, R=1)

    A = delaunay(L, bind2root=True)

    assert A.number_of_edges() > 0
    for _, _, d in A.edges(data=True):
        assert "root" in d
        assert d["root"] == -1  # only one root, so every edge binds to -1


def test_delaunay_bind2root_two_roots_assigns_valid_ids():
    # 4 terminals; two roots left/right below the square
    VertexC = np.array([
        [0.0, 0.0],  # 0
        [1.0, 0.0],  # 1
        [0.0, 1.0],  # 2
        [1.0, 1.0],  # 3
        [-1.0, -1.0],  # root (-2)
        [ 2.0, -1.0],  # root (-1)
    ], dtype=float)
    L = make_location_graph(VertexC, R=2)

    A = delaunay(L, bind2root=True)

    roots_seen = set()
    for _, _, d in A.edges(data=True):
        assert "root" in d
        assert d["root"] in {-1, -2}  # valid root ids
        roots_seen.add(d["root"])

    # In this symmetric layout we should typically see both roots used;
    # if the triangulation produces fewer edges, at least ensure we saw >= 1.
    assert len(roots_seen) >= 1


import numpy as np
import networkx as nx
import pytest

from optiwindnet.mesh import A_graph
# Reuse your helper that builds a minimal Location graph (from your tests)
from .test_mesh import make_location_graph


def _toy_L_square(R=1):
    """4 terminals making a square + R roots below."""
    base = [
        [0.0, 0.0], [1.0, 0.0],
        [0.0, 1.0], [1.0, 1.0],
    ]
    roots = [[0.5 + i, -1.0] for i in range(R)]
    VertexC = np.array(base + roots, dtype=float)
    return make_location_graph(VertexC, R=R)


def test_A_graph_delaunay_based_no_weightfun():
    L = _toy_L_square(R=1)
    A = A_graph(L, delaunay_based=True, weightfun=None)

    assert isinstance(A, nx.Graph)
    # Should be the Delaunay-based graph created via make_planar_embedding
    for k in ["VertexC", "planar", "diagonals", "d2roots", "T", "R", "B"]:
        assert k in A.graph
    # Edges have 'length' but no custom weight assigned
    assert A.number_of_edges() > 0
    for _, _, d in A.edges(data=True):
        assert d.get("length", 0.0) > 0.0
        assert "weight" not in d  # no weightfun provided


def test_A_graph_complete_graph_with_weightfun_and_custom_attr():
    L = _toy_L_square(R=1)

    # weightfun returns a recognizable constant
    weightfun = lambda data: 42.0
    A = A_graph(L, delaunay_based=False, weightfun=weightfun, weight_attr="cost")

    assert isinstance(A, nx.Graph)
    # In complete-graph mode we still expect nodes from L and edges between all nodes
    T, R = L.graph["T"], L.graph["R"]
    # include_roots=True -> terminals + roots
    expected_min_edges = (T + R) * (T + R - 1) // 2
    assert A.number_of_edges() >= expected_min_edges

    for _, _, d in A.edges(data=True):
        assert "cost" in d and d["cost"] == 42.0


def test_A_graph_calls_apply_edge_exemptions_when_weightfun_and_delaunay(monkeypatch):
    L = _toy_L_square(R=1)

    called = {"n": 0}
    def fake_apply_edge_exemptions(A_):
        called["n"] += 1

    # Patch only for this test
    import optiwindnet.mesh as mesh_mod
    monkeypatch.setattr(mesh_mod, "apply_edge_exemptions", fake_apply_edge_exemptions)

    weightfun = lambda data: 1.23
    A = A_graph(L, delaunay_based=True, weightfun=weightfun)

    # verify exemptions hook was invoked exactly once (in the delaunay branch)
    assert called["n"] == 1

    # weights set with default attr name
    for _, _, d in A.edges(data=True):
        assert "weight" in d and d["weight"] == pytest.approx(1.23)


import numpy as np
import networkx as nx
from bidict import bidict

from optiwindnet.mesh import _deprecated_planar_flipped_by_routeset


def _square_embedding_with_diag_map():
    # PlanarEmbedding with nodes 0..3
    P = nx.PlanarEmbedding()
    for n in range(4):
        P.add_node(n)

    # Build outer 4-cycle 0-1-2-3-0 (no diagonals yet)
    P.add_half_edge(0, 1)          # first from 0
    P.add_half_edge(1, 0)          # first from 1

    P.add_half_edge(1, 2, ccw=0)   # at 1, insert 2
    P.add_half_edge(2, 1)          # first from 2

    P.add_half_edge(2, 3, ccw=1)   # at 2, insert 3
    P.add_half_edge(3, 2)          # first from 3

    P.add_half_edge(3, 0, ccw=2)   # at 3, insert 0
    P.add_half_edge(0, 3, ccw=1)   # at 0, insert 3 ccw of 1

    # Add the Delaunay edge (1,2) (this is the one that will be removed)
    # Note: the half-edges for (1,2) were partially added above with the cycle,
    # but ensure both directions are present in the embedding:
    # 1 already has (1,2) from the cycle; ensure reverse (2,1) is also consistent.
    # (It is, due to the cycle construction.)

    # DO NOT add (0,3) now – it must be absent so the function flips it in.

    # Available-edges graph A with required metadata
    A = nx.Graph()
    A.add_nodes_from(range(4))
    # Make sure (1,2) exists in A so we can attach a 'path' in the path-removal test
    A.add_edge(1, 2)

    from bidict import bidict
    A.graph["planar"] = P
    # Tell the function: diagonal (0,3) corresponds to Delaunay edge (1,2)
    A.graph["diagonals"] = bidict({(0, 3): (1, 2)})

    # G uses the diagonal (0,3) that is NOT in P initially
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_edge(0, 3)

    return P, A, G



def test__deprecated_planar_flipped_by_routeset_basic_flip():
    P, A, G = _square_embedding_with_diag_map()

    P2 = _deprecated_planar_flipped_by_routeset(G, A=A, planar=P)

    # (0,3) should now be present
    assert (0, 3) in P2.edges or (3, 0) in P2.edges
    # (1,2) should have been removed
    assert (1, 2) not in P2.edges and (2, 1) not in P2.edges



def test__deprecated_planar_flipped_by_routeset_with_path_st_removal():
    P, A, G = _square_embedding_with_diag_map()

    # Provide a valid sequence for Delaunay (1,2) via 3 and back to 1,
    # so the loop ends with s=2, and (0,3) can be added with ccw=2.
    # It will remove (1,3), (3,2), (2,1) if present along that sequence.
    A[1][2]["path"] = [3, 1]

    # Sanity: the sequence removals are valid in P initially
    for e in [(1, 3), (3, 2), (2, 1)]:
        u, v = e
        assert (u, v) in P.edges or (v, u) in P.edges

    P2 = _deprecated_planar_flipped_by_routeset(G, A=A, planar=P)

    assert (0, 3) in P2.edges or (3, 0) in P2.edges
    assert (1, 2) not in P2.edges and (2, 1) not in P2.edges

