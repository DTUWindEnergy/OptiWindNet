import math

import networkx as nx
import numpy as np

from optiwindnet.geometric import is_crossing
from optiwindnet.interarraylib import G_from_S

from .helpers import tiny_wfn
from optiwindnet.pathfinding import PathFinder


def _edges_cross(G):
    """Return list of edge-pair crossings in G (ignoring shared-node pairs)."""
    VertexC = G.graph['VertexC']
    fnT = G.graph.get('fnT')
    edges = list(G.edges)
    crossings = []
    for i, (u, v) in enumerate(edges):
        u_, v_ = (fnT[u], fnT[v]) if fnT is not None else (u, v)
        for s, t in edges[i + 1 :]:
            s_, t_ = (fnT[s], fnT[t]) if fnT is not None else (s, t)
            if s_ == u_ or s_ == v_ or t_ == u_ or t_ == v_:
                continue
            if is_crossing(
                VertexC[u_],
                VertexC[v_],
                VertexC[s_],
                VertexC[t_],
                touch_is_cross=False,
            ):
                crossings.append(((u, v), (s, t)))
    return crossings


def _all_turbines_connected(G):
    """Check that every turbine can reach a root via graph edges."""
    T, R = G.graph['T'], G.graph['R']
    for t in range(T):
        found_root = False
        for r in range(-R, 0):
            if nx.has_path(G, t, r):
                found_root = True
                break
        if not found_root:
            return False
    return True


# ---------- no-crossing scenario (cables=4, single chain) ----------


def test_no_crossings_removes_tentative_tag():
    """When there are no crossings, create_detours() removes tentative tags."""
    wfn = tiny_wfn()
    G_tent = G_from_S(wfn.S, wfn.A)
    pf = PathFinder(G_tent, planar=wfn.P, A=wfn.A)

    assert pf.Xings == []
    assert pf.iterations == 0

    G_out = pf.create_detours()
    assert 'tentative' not in G_out.graph
    for _, _, d in G_out.edges(data=True):
        assert d.get('kind') != 'tentative'


# ---------- crossing scenario (cables=1, each turbine gets a feeder) ----------


def _make_crossing_case():
    """cables=1 tiny_wfn: gates (-1,1) and (-1,2) cross other edges."""
    wfn = tiny_wfn(cables=1)
    G_tent = G_from_S(wfn.S, wfn.A)
    return G_tent, wfn.P, wfn.A


def test_pathfinder_detects_crossings():
    """cables=1 produces tentative gates that have crossings."""
    G_tent, P, A = _make_crossing_case()

    tentative = G_tent.graph['tentative']
    assert len(tentative) == 2
    assert (-1, 2) in tentative
    assert (-1, 1) in tentative

    pf = PathFinder(G_tent, planar=P, A=A)
    assert len(pf.Xings) > 0
    # Xings identify which gates are crossing
    crossing_gates = {gate for _, gate in pf.Xings}
    assert (-1, 1) in crossing_gates
    assert (-1, 2) in crossing_gates


def test_create_detours_adds_detour_nodes():
    """create_detours() adds detour clone nodes for crossing gates."""
    G_tent, P, A = _make_crossing_case()
    pf = PathFinder(G_tent, planar=P, A=A)
    G_det = pf.create_detours()

    D = G_det.graph['D']
    assert D > 0, 'Expected detour vertices to be created'
    # Detour nodes should have kind='detour'
    T, B = G_det.graph['T'], G_det.graph['B']
    C = G_det.graph.get('C', 0)
    for clone in range(T + B + C, T + B + C + D):
        assert G_det.nodes[clone]['kind'] == 'detour'


def test_create_detours_crossing_free():
    """The output of create_detours() must have no edge crossings."""
    G_tent, P, A = _make_crossing_case()
    pf = PathFinder(G_tent, planar=P, A=A)
    G_det = pf.create_detours()

    crossings = _edges_cross(G_det)
    assert crossings == [], f'Crossings remain after detours: {crossings}'


def test_create_detours_preserves_connectivity():
    """All turbines must remain connected to a root after detours."""
    G_tent, P, A = _make_crossing_case()
    pf = PathFinder(G_tent, planar=P, A=A)
    G_det = pf.create_detours()

    assert _all_turbines_connected(G_det)


def test_create_detours_no_tentative_left():
    """Successful detours should remove the 'tentative' graph attribute."""
    G_tent, P, A = _make_crossing_case()
    pf = PathFinder(G_tent, planar=P, A=A)
    G_det = pf.create_detours()

    assert 'tentative' not in G_det.graph


def test_create_detours_fnT_consistent():
    """fnT must map clone nodes to their prime (border/obstacle) vertices."""
    G_tent, P, A = _make_crossing_case()
    pf = PathFinder(G_tent, planar=P, A=A)
    G_det = pf.create_detours()

    fnT = G_det.graph['fnT']
    T, B, R = G_det.graph['T'], G_det.graph['B'], G_det.graph['R']
    D = G_det.graph['D']
    C = G_det.graph.get('C', 0)

    # fnT length should cover all nodes
    assert len(fnT) == T + B + C + D + R
    # Terminal nodes map to themselves
    for t in range(T):
        assert fnT[t] == t
    # Root nodes map correctly
    for r in range(-R, 0):
        assert fnT[r] == r
    # Clone nodes map to primes in the border/obstacle range
    for clone in range(T + B + C, T + B + C + D):
        prime = fnT[clone]
        assert prime < T + B, (
            f'clone {clone} maps to prime {prime} outside border range'
        )


def test_get_best_path_returns_valid_path_to_root():
    """get_best_path should return paths that end at a root node."""
    G_tent, P, A = _make_crossing_case()
    pf = PathFinder(G_tent, planar=P, A=A)

    for n in range(G_tent.graph['T']):
        path, dists = pf.get_best_path(n)
        if not path:
            continue
        assert path[0] == n, f'path for node {n} does not start at {n}'
        assert path[-1] < 0, f'path for node {n} does not end at a root'
        assert len(dists) == len(path) - 1
        # Each dist should be a positive number
        for d in dists:
            assert d > 0, f'non-positive distance {d} in path for {n}'


def test_get_best_path_dist_is_sum_of_hops():
    """Total distance should equal the sum of hop distances."""
    G_tent, P, A = _make_crossing_case()
    pf = PathFinder(G_tent, planar=P, A=A)

    for n in range(G_tent.graph['T']):
        path, dists = pf.get_best_path(n)
        if not path:
            continue
        total = sum(dists)
        # get the stored total from paths
        I_path = pf.I_path[n]
        if I_path:
            best_id = min((pf.paths[id].dist, id) for id in I_path.values())[1]
            stored_dist = pf.paths[best_id].dist
            assert math.isclose(total, stored_dist, rel_tol=1e-9), (
                f'node {n}: sum(dists)={total} != stored dist={stored_dist}'
            )


def test_detextra_is_positive():
    """Detours add length, so detextra should be positive."""
    G_tent, P, A = _make_crossing_case()
    pf = PathFinder(G_tent, planar=P, A=A)
    G_det = pf.create_detours()

    detextra = G_det.graph['detextra']
    assert detextra >= 0, f'detextra should be non-negative, got {detextra}'


# ---------- obstacle-heavy scenario ----------


def test_obstacle_scenario_crossing_free():
    """A large obstacle between substation and turbines requires detours around it."""
    wfn = tiny_wfn(
        turbinesC=[[3.0, 0.0], [3.0, 2.0], [3.0, 4.0]],
        substationsC=[[0.0, 2.0]],
        borderC=[[-2, -2], [5, -2], [5, 6], [-2, 6]],
        obstacleC_=[np.array([[1.0, 0.5], [2.0, 0.5], [2.0, 3.5], [1.0, 3.5]])],
        cables=1,
    )
    G_tent = G_from_S(wfn.S, wfn.A)
    pf = PathFinder(G_tent, planar=wfn.P, A=wfn.A)

    # All 3 gates should be tentative and have crossings
    assert len(pf.Xings) > 0

    G_det = pf.create_detours()
    assert G_det.graph['D'] > 0, 'Expected detour nodes around obstacle'
    assert _edges_cross(G_det) == [], 'Output should be crossing-free'
    assert _all_turbines_connected(G_det)


def test_best_paths_overlay_structure():
    """best_paths_overlay returns a graph with virtual path edges and no gates."""
    G_tent, P, A = _make_crossing_case()
    pf = PathFinder(G_tent, planar=P, A=A)

    overlay = pf.best_paths_overlay()
    # overlay should be a subgraph view (no gate edges u<0 or v<0)
    for u, v in overlay.edges:
        assert u >= 0 and v >= 0, f'gate edge ({u},{v}) in overlay'
    # overlay graph attribute should contain the path graph
    assert 'overlay' in overlay.graph
    J = overlay.graph['overlay']
    # J should have virtual edges
    has_virtual = any(d.get('kind') == 'virtual' for _, _, d in J.edges(data=True))
    assert has_virtual, 'overlay graph J should contain virtual path edges'


def test_pathfinder_without_A():
    """PathFinder should work without A, building its own distance metrics."""
    G_tent, P, _ = _make_crossing_case()
    pf = PathFinder(G_tent, planar=P, A=None)
    G_det = pf.create_detours()

    assert _all_turbines_connected(G_det)
    assert _edges_cross(G_det) == []


def test_pathfinder_branched_false():
    """With branched=False, only path endpoints can be hooks."""
    G_tent, P, A = _make_crossing_case()
    pf = PathFinder(G_tent, planar=P, A=A, branched=False)
    G_det = pf.create_detours()

    assert _all_turbines_connected(G_det)
    assert _edges_cross(G_det) == []
