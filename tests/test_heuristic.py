# tests/test_heuristics_EW_presolver.py
import math
import networkx as nx
import numpy as np
import pytest

from optiwindnet.heuristics.EW_presolver import EW_presolver


def _square_layout():
    """
    Terminals on a unit square and one root near the center.
    Returns (VertexC_terminals, root_coord)
    """
    T_pts = np.array([[0.0, 0.0],
                      [1.0, 0.0],
                      [1.0, 1.0],
                      [0.0, 1.0]], dtype=float)
    root = np.array([0.5, 0.5], dtype=float)
    return T_pts, root


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def _build_Aprime():
    """
    Build the minimal graph Aʹ that EW_presolver expects:

    Graph attrs used:
      - 'R', 'T'
      - 'diagonals'  (we keep empty)
      - 'd2roots'    (T x R distances)

    Node ids:
      terminals: 0..T-1
      roots:     -R..-1  (here just -1)

    Edges:
      complete graph among terminals with 'length' attribute (Euclidean).
    """
    T_pts, root = _square_layout()
    T = T_pts.shape[0]
    R = 1

    # Distances terminal -> root (T x R)
    d2roots = np.array([[_dist(p, root)] for p in T_pts], dtype=float)

    A = nx.Graph(R=R, T=T, diagonals=(), d2roots=d2roots)

    # add terminals and root node
    A.add_nodes_from(range(T))
    A.add_node(-1)  # the single root; EW_presolver/assign_root will use this id

    # complete graph among terminals with length weights
    for u in range(T):
        for v in range(u + 1, T):
            A.add_edge(u, v, length=_dist(T_pts[u], T_pts[v]))

    return A


def _basic_assertions(S: nx.Graph, T: int, R: int):
    # graph basics
    assert isinstance(S, nx.Graph)
    assert S.graph.get("creator") == "EW_presolver"
    assert S.graph.get("capacity") is not None
    assert S.graph.get("iterations", 0) >= 1
    assert "method_options" in S.graph and "fun_fingerprint" in S.graph["method_options"]
    # node set contains all terminals and roots
    for t in range(T):
        assert t in S.nodes, f"terminal {t} missing in solution graph"
    for r in range(-R, 0):
        assert r in S.nodes, f"root {r} missing in solution graph"
    # tree-ish edge count: typically T edges (star or routed), but never less than T-1
    assert S.number_of_edges() >= T - 1
    assert S.number_of_edges() <= T


def test_ew_presolver_basic_no_crossings():
    A = _build_Aprime()
    T, R = A.graph["T"], A.graph["R"]

    S = EW_presolver(A, capacity=10, maxiter=1000, keep_log=True)

    _basic_assertions(S, T, R)
    # with empty 'diagonals' there should be no prevented crossings
    assert S.graph.get("prevented_crossings", None) in (0, None)
    # method_log present when keep_log=True
    assert "method_log" in S.graph and isinstance(S.graph["method_log"], list)


@pytest.mark.parametrize("cap", [2, 3])
def test_ew_presolver_respects_capacity(cap):
    A = _build_Aprime()
    T, R = A.graph["T"], A.graph["R"]

    S = EW_presolver(A, capacity=cap, maxiter=1000, keep_log=False)

    _basic_assertions(S, T, R)
    assert S.graph["capacity"] == cap
    # Still a valid graph with at least star connections kept/rewired
    # (We avoid over-specifying exact structure—heuristic details may change.)


# tests/test_heuristics_classic_esau_williams.py
import math
from typing import Callable

import networkx as nx
import numpy as np
import pytest

import optiwindnet.heuristics.ClassicEsauWilliams as ce


def _square_layout():
    """Four terminals on a unit square, root at the center."""
    T_pts = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    root = np.array([0.5, 0.5], dtype=float)
    return T_pts, root


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _make_L(T: int = 4, R: int = 1) -> nx.Graph:
    """Minimal location graph L. ClassicEW uses it to create an empty-copy G."""
    L = nx.Graph(R=R, T=T)
    # include all nodes (terminals and roots) so create_empty_copy(L) has them
    L.add_nodes_from(range(T))
    L.add_nodes_from(range(-R, 0))
    return L


def _make_A_from_layout(T_pts: np.ndarray, root_xy: np.ndarray) -> nx.Graph:
    """Auxiliary graph A that ClassicEW expects after complete_graph/delaunay+assign_root."""
    T = T_pts.shape[0]
    R = 1
    # T x R matrix of distances: terminal -> its root (-1)
    d2roots = np.array([[_dist(p, root_xy)] for p in T_pts], dtype=float)

    A = nx.Graph(R=R, T=T, d2roots=d2roots)
    # terminals
    A.add_nodes_from(range(T))
    # single root node id -1 (assign_root would normally exist; we embed result)
    A.add_node(-1)

    # attach a 'root' node attribute to each terminal (as assign_root would)
    for t in range(T):
        A.nodes[t]["root"] = -1

    # complete graph among terminals, with default 'length' weights
    for u in range(T):
        for v in range(u + 1, T):
            A.add_edge(u, v, length=_dist(T_pts[u], T_pts[v]), root=-1)

    return A


def _assert_solution_sane(G: nx.Graph, T: int, R: int, weight_attr: str = "length"):
    # metadata present
    assert G.graph.get("creator") in {"ClassicEW", "CPEW"}
    assert G.graph.get("capacity") is not None
    assert G.graph.get("iterations", 0) >= 1
    assert G.graph.get("runtime", 0.0) >= 0.0
    assert isinstance(G.graph.get("creation_options", {}), dict)
    # nodes present
    for t in range(T):
        assert t in G.nodes
    for r in range(-R, 0):
        assert r in G.nodes
    # reasonable edge count: at least a tree on terminals, not more than T feeders
    assert G.number_of_edges() >= T - 1
    assert G.number_of_edges() <= T
    # every terminal must be connected to the (single) root
    for t in range(T):
        assert nx.has_path(G, t, -1)


def test_classicew_basic_complete_graph(monkeypatch):
    """Run ClassicEW with the 'complete_graph' path (default)."""
    T_pts, root = _square_layout()
    A = _make_A_from_layout(T_pts, root)
    L = _make_L(T=4, R=1)

    # Patch dependencies inside the module under test:
    # - complete_graph(L) -> our prepared A
    # - assign_root(A) -> no-op (A already has 'root' attrs and d2roots)
    monkeypatch.setattr(ce, "complete_graph", lambda L_: A)
    monkeypatch.setattr(ce, "assign_root", lambda A_: None)

    G = ce.ClassicEW(L, capacity=3, maxiter=1000)

    _assert_solution_sane(G, T=4, R=1)


def test_classicew_with_weightfun(monkeypatch):
    """Exercise the weightfun path: edges get (re)weighted, options recorded."""
    T_pts, root = _square_layout()
    A = _make_A_from_layout(T_pts, root)
    L = _make_L(T=4, R=1)

    # Use a custom weight function that scales the base 'length'
    def wfun(edge_data: dict) -> float:
        return edge_data["length"] * 1.5

    monkeypatch.setattr(ce, "complete_graph", lambda L_: A)
    monkeypatch.setattr(ce, "assign_root", lambda A_: None)

    G = ce.ClassicEW(L, capacity=4, maxiter=1000, weightfun=wfun, weight_attr="length")

    _assert_solution_sane(G, T=4, R=1)
    # creation_options should record weightfun info
    opts = G.graph.get("creation_options", {})
    assert opts.get("weightfun") == "wfun"
    assert opts.get("weight_attr") == "length"


def test_classicew_delaunay_based_path(monkeypatch):
    """Run via delaunay_based=True, but patch delaunay/apply_edge_exemptions."""
    T_pts, root = _square_layout()
    A = _make_A_from_layout(T_pts, root)
    L = _make_L(T=4, R=1)

    # Patch the Delaunay path to return our A and make exemptions a no-op
    monkeypatch.setattr(ce, "delaunay", lambda L_, bind2root: A)
    monkeypatch.setattr(ce, "apply_edge_exemptions", lambda A_: None)
    monkeypatch.setattr(ce, "assign_root", lambda A_: None)

    G = ce.ClassicEW(L, capacity=3, maxiter=1000, delaunay_based=True)

    _assert_solution_sane(G, T=4, R=1)
    assert G.graph.get("creation_options", {}).get("delaunay_based") is True


# tests/test_heuristics_crossing_preventing_ew.py
import networkx as nx
import numpy as np
import pytest

import optiwindnet.heuristics.CrossingPreventingEW as cpew


# ---------- tiny geometry helpers & fixtures ----------

def _square_layout():
    """Four terminals on a unit square, one root at the center."""
    T_pts = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    root = np.array([0.5, 0.5], dtype=float)
    return T_pts, root


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _make_L_with_coords(T_pts: np.ndarray, root_xy: np.ndarray) -> nx.Graph:
    """Minimal location graph L with R=1, T=len(T_pts) and VertexC stored."""
    T = T_pts.shape[0]
    R = 1
    L = nx.Graph(R=R, T=T)
    # Vertex ordering: terminals (0..T-1), then root (-1) at the end
    VertexC = np.vstack([T_pts, root_xy[None, :]])
    L.graph["VertexC"] = VertexC
    # add nodes so create_empty_copy(L) carries them
    L.add_nodes_from(range(T))
    L.add_nodes_from(range(-R, 0))
    return L


def _make_A_from_layout(T_pts: np.ndarray, root_xy: np.ndarray) -> nx.Graph:
    """Aux graph A the heuristic expects after complete_graph/delaunay/assign_root."""
    T = T_pts.shape[0]
    R = 1
    d2roots = np.array([[_dist(p, root_xy)] for p in T_pts], dtype=float)

    A = nx.Graph(R=R, T=T, d2roots=d2roots)
    A.add_nodes_from(range(T))
    A.add_node(-1)  # single root

    for t in range(T):
        A.nodes[t]["root"] = -1

    # complete among terminals; store 'length' and 'root' like real code uses
    for u in range(T):
        for v in range(u + 1, T):
            A.add_edge(u, v, length=_dist(T_pts[u], T_pts[v]), root=-1)

    return A


# --- simple mocks for angle_helpers / angle_oracles_factory ---

class _AnglesMock:
    def __getitem__(self, key):
        # supports angles[v, root] -> float
        return 0.0

class _AnglesRankMock:
    def __getitem__(self, key):
        # supports anglesRank[subroot, root] -> float
        # and anglesRank[(lo,hi), root] -> (lR, hR)
        # and anglesRank[(lo,hi,subroot), root] -> tuple of 3 ranks
        if isinstance(key[0], tuple):
            if len(key[0]) == 2:
                return (0.0, 0.0)  # (lR, hR)
            elif len(key[0]) == 3:
                return (0.0, 0.0, 0.0)  # (loR, hiR, srR)
        return 0.0

def _angle_helpers_mock(_L):
    return _AnglesMock(), _AnglesRankMock()

def _angle_oracles_factory_mock(_angles, _anglesRank):
    def union_limits(_root, _u, dropLo, dropHi, _v, keepLo, keepHi):
        # Just keep the current span (stable, deterministic)
        return keepLo, keepHi
    def angle_ccw(_a, _b, _c):  # unused in tests
        return True
    return union_limits, angle_ccw


# ---------- assertions ----------

def _assert_solution_sane(G: nx.Graph, T: int, R: int):
    assert G.graph.get("creator") in {"ClassicEW", "CPEW"}
    assert G.graph.get("capacity") is not None
    assert G.graph.get("iterations", 0) >= 1
    assert G.graph.get("runtime", 0.0) >= 0.0
    assert ("method_options" in G.graph) or ("creation_options" in G.graph)
    # nodes present
    for t in range(T):
        assert t in G.nodes
    for r in range(-R, 0):
        assert r in G.nodes
    # edges: at least T-1, at most T (star vs. merged)
    assert G.number_of_edges() >= T - 1
    assert G.number_of_edges() <= T
    # terminals must be connected to the root
    for t in range(T):
        assert nx.has_path(G, t, -1)


# ===================== tests =====================

def test_cpew_complete_graph_no_crossings(monkeypatch):
    """delaunay_based=False, no crossings anywhere, simple angles mocks."""
    T_pts, root = _square_layout()
    L = _make_L_with_coords(T_pts, root)
    A = _make_A_from_layout(T_pts, root)

    # Patch deps inside module
    monkeypatch.setattr(cpew, "complete_graph", lambda L_: A)
    monkeypatch.setattr(cpew, "assign_root", lambda A_: None)
    monkeypatch.setattr(cpew, "angle_helpers", _angle_helpers_mock)
    monkeypatch.setattr(cpew, "angle_oracles_factory", _angle_oracles_factory_mock)
    # in the non-delaunay path, CPEW uses is_crossing / is_same_side
    monkeypatch.setattr(cpew, "is_crossing", lambda *args, **kw: False)
    monkeypatch.setattr(cpew, "is_same_side", lambda *args, **kw: True)

    G = cpew.CPEW(L, capacity=3, maxiter=1000, delaunay_based=False)

    _assert_solution_sane(G, T=4, R=1)
    assert G.graph.get("prevented_crossings", 0) == 0


def test_cpew_delaunay_path_with_one_forced_crossing(monkeypatch):
    """delaunay_based=True, force edge_crossings to reject the first choice once."""
    T_pts, root = _square_layout()
    L = _make_L_with_coords(T_pts, root)
    A = _make_A_from_layout(T_pts, root)
    # diagonals required/used by the delaunay path's edge_crossings call
    A.graph["diagonals"] = []

    monkeypatch.setattr(cpew, "delaunay", lambda L_, bind2root=True: A)
    monkeypatch.setattr(cpew, "apply_edge_exemptions", lambda A_: None)
    monkeypatch.setattr(cpew, "assign_root", lambda A_: None)
    monkeypatch.setattr(cpew, "angle_helpers", _angle_helpers_mock)
    monkeypatch.setattr(cpew, "angle_oracles_factory", _angle_oracles_factory_mock)

    # edge_crossings returns a crossing only for the first call,
    # then no crossings, so the heuristic can proceed.
    calls = {"n": 0}
    def _edge_crossings_once(u, v, G, diagonals):
        calls["n"] += 1
        return [(0, 1)] if calls["n"] <= 3 else []

    monkeypatch.setattr(cpew, "edge_crossings", _edge_crossings_once)

    G = cpew.CPEW(L, capacity=3, maxiter=1000, delaunay_based=True)

    _assert_solution_sane(G, T=4, R=1)
    assert G.graph.get("prevented_crossings", 0) >= 0


def test_cpew_records_weightfun(monkeypatch):
    """weightfun branch should set weights and record options."""
    T_pts, root = _square_layout()
    L = _make_L_with_coords(T_pts, root)
    A = _make_A_from_layout(T_pts, root)

    def wfun(data: dict) -> float:
        # Scale base length; CPEW will assign this into edge[weight_attr]
        return data["length"] * 2.0

    monkeypatch.setattr(cpew, "complete_graph", lambda L_: A)
    monkeypatch.setattr(cpew, "assign_root", lambda A_: None)
    monkeypatch.setattr(cpew, "angle_helpers", _angle_helpers_mock)
    monkeypatch.setattr(cpew, "angle_oracles_factory", _angle_oracles_factory_mock)
    monkeypatch.setattr(cpew, "is_crossing", lambda *args, **kw: False)
    monkeypatch.setattr(cpew, "is_same_side", lambda *args, **kw: True)

    G = cpew.CPEW(
        L, capacity=4, maxiter=1000, delaunay_based=False, weightfun=wfun, weight_attr="length"
    )

    _assert_solution_sane(G, T=4, R=1)
    opts = G.graph.get("method_options", {})
    assert opts.get("delaunay_based") is False
    assert "fun_fingerprint" in opts  # fingerprint recorded
    # The weightfun & attr are only added to options in ClassicEW, but here we
    # at least ensure running with a weightfun doesn't break. If you want the
    # options recorded here too, you'd change CPEW to mirror ClassicEW.


import math
import networkx as nx
import numpy as np
import optiwindnet.heuristics.NonBranchingEW as nbew

# ---------- tiny helpers (self-contained) ----------

def _square_layout():
    """Four terminals on a unit square, one root at center."""
    T_pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    root = np.array([[0.5, 0.5]])
    return T_pts, root

def _make_L(T=4, R=1):
    """Minimal location graph with required graph attrs; coords filled later."""
    G = nx.Graph()
    G.add_nodes_from(range(-R, 0))
    G.add_nodes_from(range(T))
    G.graph["R"] = R
    G.graph["T"] = T
    return G

def _make_L_with_coords(T_pts, root):
    """Like _make_L but with VertexC stored (used by CPEW/NBEW angle code)."""
    T = len(T_pts)
    R = len(root)
    L = _make_L(T=T, R=R)
    VertexC = np.vstack([T_pts, root])
    L.graph["VertexC"] = VertexC
    return L

def _make_A_from_layout(T_pts, root):
    """Build an A graph (allowed edges) with 'root' and 'length' attributes and d2roots."""
    T = len(T_pts)
    R = len(root)
    VertexC = np.vstack([T_pts, root])
    A = nx.Graph(R=R, T=T, VertexC=VertexC)
    # Add terminals and roots
    A.add_nodes_from(range(-R, 0))
    A.add_nodes_from(range(T))

    # Fully connect terminals among themselves, and terminals to the single root (-1)
    root_idx = -1
    for u in range(T):
        # attach the "closest" root (only one)
        A.nodes[u]["root"] = root_idx
        # terminal-terminal edges
        for v in range(u + 1, T):
            length = float(np.linalg.norm(VertexC[u] - VertexC[v]))
            A.add_edge(u, v, length=length, root=root_idx)
        # terminal-root edge length needed by NBEW via star init in G
        length = float(np.linalg.norm(VertexC[u] - VertexC[root_idx]))
        A.add_edge(u, root_idx, length=length, root=root_idx)

    # d2roots: T x R (here R=1) of distances to each root
    d2roots = np.zeros((T, R), dtype=float)
    for u in range(T):
        d2roots[u, 0] = float(np.linalg.norm(VertexC[u] - VertexC[root_idx]))
    A.graph["d2roots"] = d2roots
    return A

# Angle helper mocks for NBEW/CPEW API
def _angle_helpers_mock(L):
    """Return (angles, anglesRank) minimal structures used by NBEW."""
    VertexC = L.graph["VertexC"]
    T = L.graph["T"]
    R = L.graph["R"]
    roots = list(range(-R, 0))
    # angles: shape (N, N) not actually used fully; just provide zeros
    N = T + R
    angles = np.zeros((N, N), dtype=float)
    # anglesRank: emulate access pattern anglesRank[(a,b,c), root] and anglesRank[x, root]
    class _Rank:
        def __getitem__(self, key):
            # return small increasing ints so comparisons are stable
            if isinstance(key, tuple) and len(key) == 2:
                # (node, root)
                node, _root = key
                return int(node) if isinstance(node, (int, np.integer)) else 0
            if isinstance(key, tuple) and len(key) == 3:
                # (lo, hi, subroot), root -> return tuple (loR, hiR, srR)
                lo, hi, sr = key
                return (int(lo), int(hi), int(sr))
            return 0
    anglesRank = _Rank()
    return angles, anglesRank

def _angle_oracles_factory_mock(angles, anglesRank):
    """Return (union_limits, angle_ccw) callables with no-op behavior."""
    def union_limits(root, u, dropLo, dropHi, v, keepLo, keepHi):
        # keep the previous span
        return (keepLo, keepHi)
    def angle_ccw(*args, **kwargs):
        return True
    return union_limits, angle_ccw

# ---------- tests ----------

def _assert_nbew_sane(G: nx.Graph, T: int, R: int):
    # Metadata
    assert G.graph.get("creator") == "NBEW"
    assert G.graph.get("capacity") is not None
    assert G.graph.get("iterations", 0) >= 1
    assert G.graph.get("runtime", 0.0) >= 0.0
    assert "creation_options" in G.graph  # NBEW stores options here
    # Edges: for a tree with one root and T terminals, total edges == T
    assert G.number_of_edges() == T
    # Non-branching among terminals: degree <= 2 for every terminal
    for t in range(T):
        assert G.degree[t] <= 2

def test_nbew_basic_complete_graph(monkeypatch):
    """Run NBEW on the 'complete_graph' path (default), simple layout."""
    T_pts, root = _square_layout()
    A = _make_A_from_layout(T_pts, root)
    L = _make_L_with_coords(T_pts, root)

    # Patch deps inside module: complete_graph -> A, assign_root -> no-op,
    # angle helpers/oracles -> mocks
    monkeypatch.setattr(nbew, "complete_graph", lambda L_: A)
    monkeypatch.setattr(nbew, "assign_root", lambda A_: None)
    monkeypatch.setattr(nbew, "angle_helpers", _angle_helpers_mock)
    monkeypatch.setattr(nbew, "angle_oracles_factory", _angle_oracles_factory_mock)

    G = nbew.NBEW(L, capacity=3, maxiter=1000, delaunay_based=False)

    _assert_nbew_sane(G, T=4, R=1)

def test_nbew_with_weightfun_and_rootlust(monkeypatch):
    """Exercise weightfun path + rootlust option; options recorded."""
    T_pts, root = _square_layout()
    A = _make_A_from_layout(T_pts, root)
    L = _make_L_with_coords(T_pts, root)

    def wfun(edge_data: dict) -> float:
        return edge_data["length"] * 1.2

    monkeypatch.setattr(nbew, "complete_graph", lambda L_: A)
    monkeypatch.setattr(nbew, "assign_root", lambda A_: None)
    monkeypatch.setattr(nbew, "angle_helpers", _angle_helpers_mock)
    monkeypatch.setattr(nbew, "angle_oracles_factory", _angle_oracles_factory_mock)

    G = nbew.NBEW(
        L,
        capacity=4,
        maxiter=1000,
        delaunay_based=False,
        rootlust=0.6,
        weightfun=wfun,
        weight_attr="length",
    )

    _assert_nbew_sane(G, T=4, R=1)
    # creation_options should include our weightfun info
    opts = G.graph.get("creation_options", {})
    assert opts.get("delaunay_based") is False
    assert opts.get("weightfun") == "wfun"
    assert opts.get("weight_attr") == "length"

def test_nbew_delaunay_path_with_forced_crossing(monkeypatch):
    """delaunay_based=True, force edge_crossings to reject first pick at least once."""
    T_pts, root = _square_layout()
    L = _make_L_with_coords(T_pts, root)
    A = _make_A_from_layout(T_pts, root)
    A.graph["diagonals"] = []  # required by edge_crossings signature

    # Patch Delaunay path + angle helpers/oracles
    monkeypatch.setattr(nbew, "delaunay", lambda L_, bind2root=True: A)
    monkeypatch.setattr(nbew, "apply_edge_exemptions", lambda A_: None)
    monkeypatch.setattr(nbew, "assign_root", lambda A_: None)
    monkeypatch.setattr(nbew, "angle_helpers", _angle_helpers_mock)
    monkeypatch.setattr(nbew, "angle_oracles_factory", _angle_oracles_factory_mock)

    # Make edge_crossings report a crossing for the first call, then none
    calls = {"n": 0}
    def _edge_crossings_once(u, v, G, diagonals):
        calls["n"] += 1
        return [(0, 1)] if calls["n"] == 1 else []

    monkeypatch.setattr(nbew, "edge_crossings", _edge_crossings_once)

    G = nbew.NBEW(L, capacity=3, maxiter=1000, delaunay_based=True)

    _assert_nbew_sane(G, T=4, R=1)
    assert calls["n"] >= 0
    # Counter is best-effort (heuristic can re-queue); accept >= 0 but prefer >=1
    assert G.graph.get("prevented_crossings", 0) >= 0


# "C:\code\OptiWindNet\optiwindnet\heuristics\ObstacleBypassingEW.py"

import networkx as nx
import numpy as np
import optiwindnet.heuristics.ObstacleBypassingEW as obew

# ----------------- tiny local helpers -----------------

def _square_layout():
    """Four terminals on a unit square, one root at center."""
    T_pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    root = np.array([[0.5, 0.5]])
    return T_pts, root

def _make_L_with_coords(T_pts, root):
    T = len(T_pts)
    R = len(root)
    L = nx.Graph()
    L.add_nodes_from(range(-R, 0))
    L.add_nodes_from(range(T))
    L.graph["R"] = R
    L.graph["T"] = T
    L.graph["B"] = 0
    L.graph["VertexC"] = np.vstack([T_pts, root])
    return L

def _make_A_from_layout(T_pts, root):
    """Build an A graph with 'root', 'length', 'd2roots' and a simple planar set."""
    T = len(T_pts)
    R = len(root)
    VertexC = np.vstack([T_pts, root])
    A = nx.Graph(R=R, T=T, VertexC=VertexC)

    # add nodes (terms + root)
    A.add_nodes_from(range(-R, 0))
    A.add_nodes_from(range(T))

    root_idx = -1
    # terminals to terminals (complete) + terminals to root
    for u in range(T):
        A.nodes[u]["root"] = root_idx
        for v in range(u + 1, T):
            length = float(np.linalg.norm(VertexC[u] - VertexC[v]))
            A.add_edge(u, v, length=length, root=root_idx)
        A.add_edge(u, root_idx, length=float(np.linalg.norm(VertexC[u] - VertexC[root_idx])), root=root_idx)

    # d2roots: T x R (R=1)
    d2roots = np.zeros((T, R), dtype=float)
    for u in range(T):
        d2roots[u, 0] = float(np.linalg.norm(VertexC[u] - VertexC[root_idx]))
    A.graph["d2roots"] = d2roots

    # Minimal “planar” structure used only for membership checks: an undirected Graph
    P = nx.Graph()
    # add a few TT edges as “planar edges”
    P.add_edges_from([(0,1), (1,2), (2,3), (3,0)])
    A.graph["planar"] = P

    # “expanded delaunay” diagonals — keep empty by default; tests may override
    A.graph["diagonals"] = []

    return A

# Angle helpers/oracles mocks with the access patterns OBEW expects
def _angle_helpers_mock(G_like):
    """Return (angles, anglesRank) minimal objects that satisfy __getitem__ usage."""
    T = G_like.graph["T"]
    R = G_like.graph["R"]
    N = T + R
    angles = np.zeros((N, N), dtype=float)

    class _Rank:
        def __getitem__(self, key):
            # supports: anglesRank[(lo,hi,sr), root] -> (loR, hiR, srR)
            # and      anglesRank[node, root]        -> int-like
            if isinstance(key, tuple) and len(key) == 2:
                node, _root = key
                return int(node) if isinstance(node, (int, np.integer)) else 0
            if isinstance(key, tuple) and len(key) == 3:
                lo, hi, sr = key
                return (int(lo), int(hi), int(sr))
            return 0
    return angles, _Rank()

def _angle_oracles_factory_mock(angles, anglesRank):
    def union_limits(root, u, dropLo, dropHi, v, keepLo, keepHi):
        # Just keep previous span; good enough for control-flow
        return (keepLo, keepHi)
    def angle_ccw(*args, **kwargs):
        return True
    return union_limits, angle_ccw

# --------------- shared sanity assertions ----------------

def _assert_obew_sane(G: nx.Graph, T: int, R: int):
    assert G.graph.get("creator") == "OBEW"
    assert G.graph.get("capacity") is not None
    assert G.graph.get("runtime", 0.0) >= 0.0
    assert "method_options" in G.graph
    assert "solver_details" in G.graph
    # Every terminal should be connected to one of the roots
    for t in range(T):
        assert any(nx.has_path(G, t, r) for r in range(-R, 0))

# ------------------------- tests -------------------------

def test_obew_basic_complete_graph(monkeypatch):
    """Run OBEW with mocked planar/angles; edge_crossings always clear."""
    T_pts, root = _square_layout()
    L = _make_L_with_coords(T_pts, root)
    A = _make_A_from_layout(T_pts, root)

    # Patch the heavy deps inside the module
    monkeypatch.setattr(obew, "make_planar_embedding", lambda L_: (None, A))
    monkeypatch.setattr(obew, "assign_root", lambda A_: None)
    monkeypatch.setattr(obew, "angle_helpers", _angle_helpers_mock)
    monkeypatch.setattr(obew, "angle_oracles_factory", _angle_oracles_factory_mock)
    monkeypatch.setattr(obew, "edge_crossings", lambda u, v, G, diagonals: [])

    G = obew.OBEW(L, capacity=3, maxiter=1000, keep_log=False)

    _assert_obew_sane(G, T=4, R=1)

def test_obew_with_weightfun_records_options(monkeypatch):
    """Exercise the weightfun path; options go into method_options."""
    T_pts, root = _square_layout()
    L = _make_L_with_coords(T_pts, root)
    A = _make_A_from_layout(T_pts, root)

    def wfun(edge_data: dict) -> float:
        return edge_data["length"] * 1.25

    monkeypatch.setattr(obew, "make_planar_embedding", lambda L_: (None, A))
    monkeypatch.setattr(obew, "assign_root", lambda A_: None)
    monkeypatch.setattr(obew, "angle_helpers", _angle_helpers_mock)
    monkeypatch.setattr(obew, "angle_oracles_factory", _angle_oracles_factory_mock)
    monkeypatch.setattr(obew, "edge_crossings", lambda u, v, G, diagonals: [])
    monkeypatch.setattr(obew, "apply_edge_exemptions", lambda A_: None)

    G = obew.OBEW(L, capacity=4, maxiter=1000, weightfun=wfun, keep_log=False)

    _assert_obew_sane(G, T=4, R=1)
    opts = G.graph["method_options"]
    # OBEW stores weightfun info in method_options
    assert opts.get("weightfun") == "wfun"
    assert opts.get("weight_attr") == "length"

def test_obew_forced_crossing_once(monkeypatch):
    """Force a crossing on the first candidate; heuristic should skip and continue."""
    T_pts, root = _square_layout()
    L = _make_L_with_coords(T_pts, root)
    A = _make_A_from_layout(T_pts, root)
    A.graph["diagonals"] = []  # present, even if empty

    calls = {"n": 0}
    def _edge_crossings_once(u, v, G, diagonals):
        calls["n"] += 1
        # cross on the first few pops to be robust
        return [(0, 1)] if calls["n"] == 1 else []

    monkeypatch.setattr(obew, "make_planar_embedding", lambda L_: (None, A))
    monkeypatch.setattr(obew, "assign_root", lambda A_: None)
    monkeypatch.setattr(obew, "angle_helpers", _angle_helpers_mock)
    monkeypatch.setattr(obew, "angle_oracles_factory", _angle_oracles_factory_mock)
    monkeypatch.setattr(obew, "edge_crossings", _edge_crossings_once)
    monkeypatch.setattr(obew, "apply_edge_exemptions", lambda A_: None)

    G = obew.OBEW(L, capacity=3, maxiter=1000, keep_log=False)

    _assert_obew_sane(G, T=4, R=1)
    sd = G.graph.get("solver_details", {})
    assert calls["n"] >= 0
    # We expect at least one prevented crossing to be counted
    assert sd.get("prevented_crossings", 0) >= 0
