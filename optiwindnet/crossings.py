# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import math
import warnings
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import combinations, pairwise
from typing import Any

import networkx as nx
import numpy as np
import shapely as shp
from numba import njit

from .geometric import (
    _is_crossing_clear,
    _orient2d_filtered,
    angle_helpers,
    is_bunch_split_by_corner,
    polyline_rays_at_point,
    rays_alternate,
)
from .utils import BiMap


@dataclass(frozen=True)
class _RoutePolyline:
    """A route section and its optional non-root endpoint."""

    nodes: tuple[int, ...]
    non_root_end: int | None = None


def get_interferences_list(
    Edge: np.ndarray, VertexC: np.ndarray, fnT: np.ndarray | None = None, EPSILON=1e-15
) -> list[tuple[tuple[int, int, int, int], int | None]]:
    """List all crossings between edges in the ``Edge`` (E×2) numpy array.

    Coordinates must be provided in the ``VertexC`` (V×2) array.

    ``Edge`` contains indices to VertexC. If ``Edge`` includes detour nodes
    (i.e. indices go beyond ``VertexC``'s length), ``fnT`` translation table
    must be provided.

    Should be used when edges are not limited to the expanded Delaunay set.

    Returns:
      List of interferences. Each interference is a tuple ``((4 vertices of the
      two edges involved), one of the vertices or None)``, whose last element
      indicates the index (0..3) of the vertex that lays exactly on the edge in
      cases of touching (not crossing).
    """
    crossings = []
    if fnT is None:
        V = VertexC[Edge[:, 1]] - VertexC[Edge[:, 0]]
    else:
        V = VertexC[fnT[Edge[:, 1]]] - VertexC[fnT[Edge[:, 0]]]
    for i, ((UVx, UVy), (u, v)) in enumerate(zip(V[:-1], Edge[:-1].tolist())):
        u_, v_ = (u, v) if fnT is None else fnT[[u, v]]
        (uCx, uCy), (vCx, vCy) = VertexC[[u_, v_]]
        for (STx, STy), (s, t) in zip(-V[i + 1 :], Edge[i + 1 :].tolist()):
            s_, t_ = (s, t) if fnT is None else fnT[[s, t]]
            if s_ == u_ or t_ == u_ or s_ == v_ or t_ == v_:
                # <edges have a common node>
                continue
            # bounding box check
            (sCx, sCy), (tCx, tCy) = VertexC[[s_, t_]]

            # X
            lo, hi = (vCx, uCx) if UVx < 0 else (uCx, vCx)
            if STx > 0:  # s - t > 0 -> hi: s, lo: t
                if hi < tCx or sCx < lo:
                    continue
            else:  # s - t < 0 -> hi: t, lo: s
                if hi < sCx or tCx < lo:
                    continue

            # Y
            lo, hi = (vCy, uCy) if UVy < 0 else (uCy, vCy)
            if STy > 0:
                if hi < tCy or sCy < lo:
                    continue
            else:
                if hi < sCy or tCy < lo:
                    continue

            # TODO: save the edges that have interfering bounding boxes
            #       to be checked in a vectorized implementation of
            #       the math below
            UV = UVx, UVy
            ST = STx, STy

            # denominator
            f = STx * UVy - STy * UVx
            # TODO: verify if this arbitrary tolerance is appropriate
            if math.isclose(f, 0.0, abs_tol=1e-5):
                # segments are parallel
                # TODO: there should be check for branch splitting in parallel
                #       cases with touching points
                continue

            C = uCx - sCx, uCy - sCy
            touch_found = []
            Xcount = 0
            for k, num in enumerate(
                (Px * Qy - Py * Qx) for (Px, Py), (Qx, Qy) in ((C, ST), (UV, C))
            ):
                if f > 0:
                    if -EPSILON <= num <= f + EPSILON:  # num < 0 or f < num:
                        Xcount += 1
                        if math.isclose(num, 0, abs_tol=EPSILON):
                            touch_found.append(2 * k)
                        if math.isclose(num, f, abs_tol=EPSILON):
                            touch_found.append(2 * k + 1)
                else:
                    if f - EPSILON <= num <= EPSILON:  # 0 < num or num < f:
                        Xcount += 1
                        if math.isclose(num, 0, abs_tol=EPSILON):
                            touch_found.append(2 * k)
                        if math.isclose(num, f, abs_tol=EPSILON):
                            touch_found.append(2 * k + 1)

            if Xcount == 2:
                # segments cross or touch
                uvst = (u, v, s, t)
                if touch_found:
                    assert len(touch_found) == 1, 'ERROR: too many touching points.'
                    #  p = uvst[touch_found[0]]
                    p = touch_found[0]
                else:
                    p = None
                crossings.append((uvst, p))
    return crossings


def edge_conflicts(u: int, v: int, diagonals: BiMap) -> Iterator[tuple[int, int]]:
    """Iterate over edges conflicting with ``(u, v)``.

    Args:
      u: node
      v: node
      diagonals: map of crossings Delaunay↔diagonals
    """
    u, v = (u, v) if u < v else (v, u)
    st = diagonals.get((u, v))
    if st is None:
        # ⟨u, v⟩ is a Delaunay edge
        st = diagonals.inv.get((u, v))
        if st is not None and st[0] >= 0:
            yield st
    else:
        # ⟨u, v⟩ is a diagonal of Delanay edge ⟨s, t⟩
        # crossing with Delaunay edge
        yield st

        s, t = st
        # two triangles may contain ⟨s, t⟩, each defined by their non-st vertex
        for hat in (u, v):
            for diag in (
                diagonals.inv.get((w, y) if w < y else (y, w))
                for w, y in ((s, hat), (hat, t))
            ):
                if diag is not None and diag[0] >= 0:
                    yield diag


def edge_crossings(
    u: int, v: int, G: nx.Graph, diagonals: BiMap
) -> list[tuple[int, int]]:
    u, v = (u, v) if u < v else (v, u)
    st = diagonals.get((u, v))
    conflicting = []
    if st is None:
        # ⟨u, v⟩ is a Delaunay edge
        st = diagonals.inv.get((u, v))
        if st is not None and st[0] >= 0:
            conflicting.append(st)
    else:
        # ⟨u, v⟩ is a diagonal of Delanay edge ⟨s, t⟩
        s, t = st
        # crossing with Delaunay edge
        conflicting.append(st)

        # two triangles may contain ⟨s, t⟩, each defined by their non-st vertex
        for hat in (u, v):
            for diag in (
                diagonals.inv.get((w, y) if w < y else (y, w))
                for w, y in ((s, hat), (hat, t))
            ):
                if diag is not None and diag[0] >= 0:
                    conflicting.append(diag)
    return [edge for edge in conflicting if edge in G.edges]


def edgeset_edgeXing_iter(diagonals: BiMap) -> Iterator[list[tuple[int, int]]]:
    """Iterator over all edge crossings in an expanded Delaunay edge set ``A``.

    Each crossing is a 2 or 3-tuple of (u, v) edges. Does not include gates.
    """
    checked = set()
    for (u, v), (s, t) in diagonals.items():
        # ⟨u, v⟩ is a diagonal of Delaunay ⟨s, t⟩
        if u < 0:
            # diagonal is a gate
            continue
        uv = (u, v)
        if s >= 0:
            # crossing with Delaunay edge
            yield [(s, t), uv]
        # two triangles may contain ⟨s, t⟩, each defined by their non-st vertex
        for hat in uv:
            triangle = tuple(sorted((s, t, hat)))
            if triangle in checked:
                continue
            checked.add(triangle)
            conflicting = [uv]
            for diag in (
                diagonals.inv.get((w, y) if w < y else (y, w))
                for w, y in ((s, hat), (hat, t))
            ):
                if diag is not None and diag[0] >= 0:
                    conflicting.append(diag)
            if len(conflicting) > 1:
                yield conflicting


# kinds of feeder-edge intersections found by _feeder_intersections_core()
_FEEDER_CROSS = 0
_FEEDER_TOUCH = 1
_FEEDER_ALONG = 2
_FEEDER_UNDECIDED = 3


@njit(cache=True)
def _feeder_intersections_core(
    edges: np.ndarray,
    VertexC: np.ndarray,
    angle__: np.ndarray,
    angle_rank__: np.ndarray,
    R: int,
    hooks: np.ndarray,
    hooks_start: np.ndarray,
) -> np.ndarray:
    """Find the intersections of ``edges`` with the feeders to ``hooks``.

    Kernel of :func:`_feeder_intersections`. Each row of ``edges`` is an edge
    ``(u, v)``. Root ``j - R`` is checked against nodes
    ``hooks[hooks_start[j]:hooks_start[j + 1]]``. Orientation signs are
    certified by :func:`~optiwindnet.geometric._orient2d_filtered`. Pairs with
    an uncertain sign, or crossing so close to an endpoint or so nearly parallel
    that GEOS might find a touch, are returned as undecided.

    Returns:
      (X, 6) array of intersections, each row as ``(a, b, root, n, kind, side)``:

      * ``_FEEDER_CROSS``: ⟨a, b⟩ properly crosses the feeder ⟨root, n⟩;
      * ``_FEEDER_TOUCH``: ``a`` lies inside the feeder and ``b`` off its line,
        on ``side`` (1: left of root→n, -1: right);
      * ``_FEEDER_ALONG``: ``a`` and ``b`` both lie inside the feeder;
      * ``_FEEDER_UNDECIDED``: to be classified by GEOS.
    """
    out = np.empty((max(16, edges.shape[0]), 6), dtype=np.int64)
    k = 0
    for i in range(edges.shape[0]):
        u = edges[i, 0]
        v = edges[i, 1]
        ux, uy = VertexC[u]
        vx, vy = VertexC[v]
        for j in range(hooks_start.shape[0] - 1):
            root = j - R
            rx, ry = VertexC[root]
            uvA = angle__[v, root] - angle__[u, root]
            if -np.pi < uvA < 0.0 or np.pi < uvA:
                loR, hiR = angle_rank__[v, root], angle_rank__[u, root]
            else:
                loR, hiR = angle_rank__[u, root], angle_rank__[v, root]
            wraps = loR > hiR
            dr, cr = _orient2d_filtered(ux, uy, vx, vy, rx, ry)
            for h in range(hooks_start[j], hooks_start[j + 1]):
                n = hooks[h]
                pR = angle_rank__[n, root]
                # closed window: a feeder going over u or v is a candidate
                supL = loR <= pR
                infH = pR <= hiR
                if not ((supL != infH) if wraps else (supL and infH)):
                    continue
                if n == u or n == v:
                    continue
                nx_, ny = VertexC[n]
                # most candidates end before reaching the edge
                dn, cn = _orient2d_filtered(ux, uy, vx, vy, nx_, ny)
                if cr and cn and ((dr > 0.0 and dn > 0.0) or (dr < 0.0 and dn < 0.0)):
                    continue
                du, cu = _orient2d_filtered(rx, ry, nx_, ny, ux, uy)
                dv, cv = _orient2d_filtered(rx, ry, nx_, ny, vx, vy)
                a, b, side = u, v, 0
                if not (cr and cn and cu and cv):
                    kind = _FEEDER_UNDECIDED
                elif (du > 0.0 and dv < 0.0) or (du < 0.0 and dv > 0.0):
                    if dr == 0.0 or dn == 0.0:
                        continue
                    kind = (
                        _FEEDER_CROSS
                        if _is_crossing_clear(
                            ux, uy, vx, vy, rx, ry, nx_, ny, dr, dn, du, dv
                        )
                        else _FEEDER_UNDECIDED
                    )
                elif du == 0.0 and dv == 0.0:
                    # collinear points are ordered as their coordinates along
                    # the dominant axis
                    c = 0 if abs(nx_ - rx) >= abs(ny - ry) else 1
                    rc, nc = (rx, nx_) if c == 0 else (ry, ny)
                    uc, vc = (ux, vx) if c == 0 else (uy, vy)
                    lo, hi = min(rc, nc), max(rc, nc)
                    if not (lo < uc < hi and lo < vc < hi):
                        continue
                    kind = _FEEDER_ALONG
                elif du != 0.0 and dv != 0.0 or dr == 0.0 or dn == 0.0:
                    # same side; or the end on the feeder's line (the only
                    # point where the edge's line meets it) is not inside it
                    continue
                else:
                    kind = _FEEDER_TOUCH
                    if du == 0.0:
                        side = 1 if dv > 0.0 else -1
                    else:
                        a, b, side = v, u, 1 if du > 0.0 else -1
                if k == out.shape[0]:
                    grown = np.empty((2 * k, 6), dtype=np.int64)
                    grown[:k] = out
                    out = grown
                out[k] = a, b, root, n, kind, side
                k += 1
    return out[:k]


def _feeder_intersections(G: nx.Graph, hooks: Iterable | None = None) -> np.ndarray:
    """Find the intersections of the non-feeder edges of ``G`` with feeders.

    Feeders are the straight lines from each root to each of its hooks. An edge
    intersects a feeder by crossing it, by touching it with one end (the feeder
    goes over that node) or by lying along it (both ends inside the feeder).
    Not reported: edges ending at the hook, and edges going over the hook.

    Args:
      G: Routeset or edgeset (A) to examine. If ``G`` has ``'fnT'``, edges are
        tested (and reported) by their prime nodes.
      hooks: Nodes to check, grouped by root in subsequences from root ``-R``
        to ``-1``. If ``None``, every terminal is checked against every root.

    Returns:
      (X, 6) int array of intersections, each row as ``(a, b, root, n, kind, side)``,
      with ``kind`` one of ``_FEEDER_CROSS``, ``_FEEDER_TOUCH`` or
      ``_FEEDER_ALONG`` (see :func:`_feeder_intersections_core`).

    Raises:
      IndexError: if an edge end or hook has no angle rank wrt the roots.
    """
    R, T, VertexC = (G.graph[k] for k in ('R', 'T', 'VertexC'))
    fnT = G.graph.get('fnT')
    angle_rank__ = G.graph.get('angle_rank__', None)
    if angle_rank__ is None:
        angle__, angle_rank__, _ = angle_helpers(G)
    else:
        angle__ = G.graph['angle__']
    # TODO: There is a corner case here: for multiple roots, the gates are not
    #       being checked between different roots. Unlikely but possible case.
    # non-gate edges:
    edges = np.array(
        [(u, v) for u, v in G.edges if u >= 0 and v >= 0], dtype=np.int64
    ).reshape(-1, 2)
    if fnT is not None:
        edges = fnT[edges]
    edges.sort(axis=1)
    if hooks is None:
        hooks_ = [np.arange(T)] * R
    else:
        hooks_ = [np.asarray(h, dtype=np.int64) for h in hooks][:R]
    hooks_start = np.zeros(len(hooks_) + 1, dtype=np.int64)
    np.cumsum([len(h) for h in hooks_], out=hooks_start[1:])
    hooks_flat = np.concatenate([np.empty(0, dtype=np.int64), *hooks_])
    # the kernel does not check bounds
    num_ranked = angle_rank__.shape[0]
    if (edges.size and edges.max() >= num_ranked) or (
        hooks_flat.size and hooks_flat.max() >= num_ranked
    ):
        raise IndexError('node without angle rank wrt the roots')
    found = _feeder_intersections_core(
        edges, VertexC, angle__, angle_rank__, R, hooks_flat, hooks_start
    )
    undecided = found[:, 4] == _FEEDER_UNDECIDED
    if undecided.any():
        found[undecided] = _feeder_intersections_by_geos(found[undecided], VertexC)
        found = found[found[:, 4] >= 0]
    return found


def _feeder_intersections_by_geos(rows: np.ndarray, VertexC: np.ndarray) -> np.ndarray:
    """Classify undecided rows of :func:`_feeder_intersections_core` with GEOS.

    The DE-9IM matrix of edge ⟨a, b⟩ against feeder ⟨root, n⟩ tells a crossing
    (interiors meet in a point), an edge along the feeder (within it) or a touch
    (an edge end in the feeder's interior), consistently with shapely's
    ``crosses`` and ``touches``.

    Returns:
      ``rows`` classified, with ``kind = -1`` where there is no intersection.
    """
    a, b, root, n = rows[:, :4].T
    feedersS = shp.linestrings(np.stack((VertexC[root], VertexC[n]), axis=1))
    edgesS = shp.linestrings(np.stack((VertexC[a], VertexC[b]), axis=1))
    # DE-9IM order: II IB IE BI BB BE EI EB EE (edge first)
    matrices = np.asarray(shp.relate(edgesS, feedersS), dtype='U9').view('U1')
    matrices = matrices.reshape(-1, 9)
    crosses = matrices[:, 0] == '0'
    along = (matrices[:, 0] == '1') & (matrices[:, 2] == 'F') & (matrices[:, 5] == 'F')
    touches = (matrices[:, 0] == 'F') & (matrices[:, 3] == '0')
    a_on = np.asarray(shp.intersects(shp.points(VertexC[a]), feedersS), dtype=bool)
    on, off = np.where(a_on, a, b), np.where(a_on, b, a)
    rings = shp.linearrings(np.stack((VertexC[root], VertexC[n], VertexC[off]), axis=1))
    side = np.where(np.asarray(shp.is_ccw(rings), dtype=bool), 1, -1)
    out = rows.copy()
    out[:, 4] = -1
    out[crosses, 4] = _FEEDER_CROSS
    out[along, 4] = _FEEDER_ALONG
    out[touches, 0] = on[touches]
    out[touches, 1] = off[touches]
    out[touches, 4] = _FEEDER_TOUCH
    out[:, 5] = np.where(touches, side, 0)
    return out


def _feeder_crossings(G: nx.Graph, hooks: Iterable | None = None) -> np.ndarray:
    """Find the crossings between feeders and the non-feeder edges of ``G``.

    Same arguments as :func:`_feeder_intersections`. Touching (a feeder going
    over a node) counts as crossing; an edge lying along a feeder does not.

    Returns:
      (X, 4) int array of crossings, each row as ``(u, v, root, n)``: edge ⟨u, v⟩
      (``u < v``) crosses the feeder ⟨root, n⟩.
    """
    found = _feeder_intersections(G, hooks)
    crossings = found[found[:, 4] != _FEEDER_ALONG, :4]
    # only touches may have the edge's ends swapped
    flip = np.flatnonzero(crossings[:, 0] > crossings[:, 1])
    crossings[flip, :2] = crossings[flip, 1::-1]
    return crossings


def gateXing_iter(
    G: nx.Graph, *, hooks: Iterable | None = None
) -> Iterator[tuple[tuple[int, int], tuple[int, int]]]:
    """Iterate over all crossings between gates and edges in G.

    Deprecated: internal function, to be removed in v0.4.0.

    Args:
      G: Routeset or edgeset (A) to examine.
      hooks: Nodes to check, grouped by root in subsequences from root ``-R``
        to ``-1``. If ``None``, every terminal is checked against every root.

    Returns:
      Iterator over pairs of (edge, gate) that cross (each a 2-tuple of nodes).
    """
    warnings.warn(
        'optiwindnet.crossings.gateXing_iter is deprecated and will be removed in '
        'v0.4.0; it is internal and has no public replacement',
        DeprecationWarning,
        stacklevel=2,
    )
    return (
        ((u, v), (root, n))
        for u, v, root, n in _feeder_crossings(G, hooks=hooks).tolist()
    )


def find_routeset_crossings(G: nx.Graph) -> list[tuple[int, int, int, int]]:
    """Find edge crossings and branch splits in a routeset.

    Each of ``G``'s edges is tested as a straight segment between the prime
    coordinates of its endpoints, every pair against every other. Edges that
    merely touch are reported only where the touch splits a branch apart, and
    each detour node is checked for splitting the branch it routes around.

    Straight segments are what makes this cheaper than
    :func:`find_geometric_crossings`, which assembles whole polylines and so
    also reports collinear overlaps and touches. Neither requires ``G`` to be
    built from ``A`` -- unlike :func:`list_edge_crossings`.

    Args:
      G: routeset graph. Needs graph attributes ``'R'``, ``'T'``, ``'B'`` and
        ``'VertexC'``; ``'fnT'`` is required iff ``C > 0`` or ``D > 0``.

    Returns:
      list of ``(u, v, s, t)``, empty if ``G`` has neither. ``u != v`` means
      edge ⟨u, v⟩ crosses edge ⟨s, t⟩; ``u == v`` means the detour at ``u``
      splits the branch between ``s`` and ``t``.
    """
    T, B = (G.graph[k] for k in 'TB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    VertexC = G.graph['VertexC']
    fnT = _routeset_fnT(G)

    # check edge×edge crossings
    #  Edge = np.array(tuple((fnT[u], fnT[v]) for u, v in G.edges))
    XTings = get_interferences_list(np.array(G.edges), VertexC, fnT)
    # parallel is considered no crossing
    # analyse cases of touch
    Xings = []
    for uvst, p in XTings:
        if p is None:
            Xings.append(uvst)
            continue
        if G.degree[p] == 1:
            # trivial case: no way to break a branch apart
            continue
        # make u be the touch-point within ⟨s, t⟩
        u = uvst[p]
        s, t = uvst[2:] if p < 2 else uvst[:2]

        u_, s_, t_ = fnT[(u, s, t),].tolist()
        bunch = [fnT[nb].item() for nb in G[u]]
        is_split, insideI, outsideI = is_bunch_split_by_corner(
            VertexC[bunch], *VertexC[[s_, u_, t_]]
        )
        if is_split:
            Xings.append((s_, t_, bunch[insideI[0]], bunch[outsideI[0]]))

    # check detour nodes for branch-splitting
    d_start = T + B + C
    for d, d_ in enumerate(fnT[d_start : d_start + D].tolist(), start=d_start):
        if d_ >= T or G.degree[d_] == 1:
            # either the detour node is over a border vertex or the node is a leaf:
            #   no branch splitting possible
            continue
        dA, dB = (fnT[nb] for nb in G[d])
        bunch = [fnT[nb].item() for nb in G[d_]]
        is_split, insideI, outsideI = is_bunch_split_by_corner(
            VertexC[bunch], *VertexC[[dA, d_, dB]]
        )
        if is_split:
            Xings.append((d_, d_, bunch[insideI[0]], bunch[outsideI[0]]))
    return Xings


def _routeset_fnT(G: nx.Graph) -> np.ndarray:
    """Identity translation table (clones → primes); synthesized when G has none."""
    R, T, B = (G.graph[k] for k in 'RTB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    if C > 0 or D > 0:
        return G.graph['fnT']
    fnT = np.arange(T + B + R)
    fnT[-R:] = range(-R, 0)
    return fnT


def _canonical_prime_path(
    G: nx.Graph, path: tuple[int, ...], fnT: np.ndarray
) -> tuple[int, ...]:
    """Translate a polyline to primes, drop bordering roots, canonicalize direction."""
    prime_path = tuple(int(fnT[n]) for n in path)
    R = G.graph['R']
    trimmed = prime_path
    while len(trimmed) > 1 and -R <= trimmed[0] < 0:
        trimmed = trimmed[1:]
    while len(trimmed) > 1 and -R <= trimmed[-1] < 0:
        trimmed = trimmed[:-1]
    if len(trimmed) > 1:
        prime_path = trimmed
    if len(prime_path) > 1 and prime_path[0] < prime_path[-1]:
        prime_path = prime_path[::-1]
    return prime_path


def _routeset_polylines(G: nx.Graph) -> list[_RoutePolyline]:
    """Trace G into one polyline per feeder plus one per inter-junction link.

    A feeder runs from a root through a degree-2 chain to the first leaf or
    branching node. A link runs between two non-degree-2 nodes that are not
    roots. Together these cover every edge exactly once.

    A RINGED route is a chain of degree-2 terminals between two feeders, so the
    trace crosses the ring's zero-load link and comes out at a root: the whole
    ring is one unit. Its path returns to its starting root when both feeders
    share one root and ends at a different root when it bridges two roots. Either
    way the second feeder's edge is already visited, so the root loop must skip
    it -- otherwise the ring's last segment would be emitted a second time as a
    two-node stub and counted twice.
    """
    R = G.graph['R']
    roots = set(range(-R, 0))
    visited = set()
    polylines: list[_RoutePolyline] = []

    def edge_key(u: int, v: int) -> tuple[int, int]:
        return (u, v) if u < v else (v, u)

    def walk(prev: int, node: int) -> tuple[int, ...]:
        path = [prev, node]
        visited.add(edge_key(prev, node))
        while node not in roots and G.degree[node] == 2:
            candidates = [nb for nb in G[node] if nb != prev]
            if len(candidates) != 1:
                break
            nxt = candidates[0]
            key = edge_key(node, nxt)
            if key in visited:
                break
            prev, node = node, nxt
            path.append(node)
            visited.add(key)
        return tuple(path)

    for root in sorted(roots):
        for nb in G[root]:
            if edge_key(root, nb) in visited:
                # the far feeder of a ring was already traced from its other end
                continue
            path = walk(root, nb)
            # a ring ends on a root, which is no terminal end to exempt splits at
            end = path[-1]
            polylines.append(
                _RoutePolyline(
                    path,
                    non_root_end=None if end in roots else end,
                )
            )

    starts = [n for n in G if n not in roots and G.degree[n] != 2]
    for start in starts:
        for nb in G[start]:
            if nb in roots:
                continue
            if edge_key(start, nb) in visited:
                continue
            path = walk(start, nb)
            polylines.append(_RoutePolyline(path))

    return polylines


def _polyline_primes(
    VertexC: np.ndarray, fnT: np.ndarray, path: tuple[int, ...]
) -> np.ndarray:
    """A path's primes, with consecutive coordinate duplicates collapsed."""
    primes = fnT[list(path)]
    if len(primes) <= 1:
        return primes
    raw = VertexC[primes]
    keep = np.empty(len(raw), dtype=bool)
    keep[0] = True
    keep[1:] = np.any(raw[1:] != raw[:-1], axis=1)
    return primes[keep]


def _polyline_coords(
    VertexC: np.ndarray, fnT: np.ndarray, path: tuple[int, ...]
) -> np.ndarray:
    """(N, 2) coords of a path's primes, with consecutive duplicates collapsed."""
    return VertexC[_polyline_primes(VertexC, fnT, path)]


def _run_end_side(
    run_ray: np.ndarray, ray_a: np.ndarray, ray_b: np.ndarray, *, angle_tol: float
) -> int:
    """Order two rays leaving a run's end, counterclockwise from ``run_ray``.

    ``run_ray`` points from the end into the run. Returns ``1`` if ``ray_a``
    comes first, ``-1`` if ``ray_b`` does, and ``0`` if a ray is null or along
    ``run_ray`` or the two rays coincide.
    """
    angles = []
    for ray in (run_ray, ray_a, ray_b):
        norm = math.hypot(*ray)
        if norm == 0.0:
            return 0
        angles.append(math.atan2(ray[1], ray[0]))
    run_angle, angle_a, angle_b = angles
    angle_a = (angle_a - run_angle) % (2 * math.pi)
    angle_b = (angle_b - run_angle) % (2 * math.pi)
    if min(angle_a, angle_b, 2 * math.pi - max(angle_a, angle_b)) <= angle_tol:
        return 0
    if abs(angle_a - angle_b) <= angle_tol:
        return 0
    return 1 if angle_a < angle_b else -1


def _shared_run_swaps_sides(
    coords_a: np.ndarray, coords_b: np.ndarray, *, angle_tol: float
) -> bool:
    """``True`` iff two polylines cross along their longest shared vertex run.

    The run may be traversed in either direction. The polylines cross iff they
    leave it on opposite sides at both ends, i.e. their exits come in the same
    counterclockwise order (see :func:`_run_end_side`) at both ends. Both need
    a segment of context beyond each end of the run.

    Operates on raw polyline coords (with consecutive duplicates collapsed) rather
    than canonical prime paths, so root-leg context preserved in the geometry —
    but trimmed from canonical prime paths — remains available here.
    """
    Na, Nb = len(coords_a), len(coords_b)
    if Na < 2 or Nb < 2:
        return False

    best = None
    best_len = 1
    for cb in (coords_b, coords_b[::-1]):
        for i in range(Na):
            for j in range(Nb):
                if not np.array_equal(coords_a[i], cb[j]):
                    continue
                length = 1
                while (
                    i + length < Na
                    and j + length < Nb
                    and np.array_equal(coords_a[i + length], cb[j + length])
                ):
                    length += 1
                if length > best_len:
                    best_len = length
                    best = (i, i + length, j, j + length, cb)
    if best is None:
        return False

    start_a, end_a, start_b, end_b, oriented_b = best
    # require at least one segment of context on each side of the shared run
    if not (0 < start_a and end_a < Na and 0 < start_b and end_b < Nb):
        return False

    ends = (
        (start_a, start_a + 1, start_a - 1, start_b - 1),
        (end_a - 1, end_a - 2, end_a, end_b),
    )
    sides = [
        _run_end_side(
            coords_a[inward] - coords_a[end],
            coords_a[out_a] - coords_a[end],
            oriented_b[out_b] - coords_a[end],
            angle_tol=angle_tol,
        )
        for end, inward, out_a, out_b in ends
    ]
    return sides[0] != 0 and sides[0] == sides[1]


def _split_branch_nodes(
    G: nx.Graph,
    prime: int,
    route_coords: np.ndarray,
    fnT: np.ndarray,
    VertexC: np.ndarray,
    *,
    tol: float,
    angle_tol: float,
) -> tuple[int, int] | None:
    """Return two neighbours of ``prime`` in ``G`` separated by ``route_coords``."""
    pC = VertexC[prime]
    route_rays = polyline_rays_at_point(route_coords, pC, tol=tol, angle_tol=angle_tol)
    if len(route_rays) != 2:
        return None

    branch_rays = []
    for nb in G[prime]:
        nb_ = int(fnT[nb])
        ray = VertexC[nb_] - pC
        norm = math.hypot(*ray)
        if norm <= tol:
            continue
        unit = ray / norm
        if any(
            abs(unit[0] * route_ray[1] - unit[1] * route_ray[0]) <= angle_tol
            and np.dot(unit, route_ray) > 0
            for route_ray in route_rays
        ):
            continue
        branch_rays.append((nb_, unit))
    for (a, ray_a), (b, ray_b) in combinations(branch_rays, 2):
        if rays_alternate(route_rays, [ray_a, ray_b]):
            return a, b
    return None


def _detour_splits(
    G: nx.Graph,
    fnT: np.ndarray,
    VertexC: np.ndarray,
    *,
    endpoint_tol: float,
    angle_tol: float,
) -> dict[int, tuple[int, tuple[int, int]]]:
    """Map detour clones in ``G`` that separate rays at their prime terminal.

    Returns ``{detour_node: (prime, (neighbor_a, neighbor_b))}``, where the two
    neighbours lie in different sectors defined by the detour route.
    """
    T, B = (G.graph[k] for k in 'TB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    if D == 0:
        return {}
    splits: dict[int, tuple[int, tuple[int, int]]] = {}
    for d in range(T + B + C, T + B + C + D):
        prime = int(fnT[d])
        if prime not in G or not 0 <= prime < T:
            continue
        if G.degree[prime] == 1 or G.degree[d] != 2:
            continue
        dA, dB = (int(fnT[nb]) for nb in G[d])
        route_coords = VertexC[[dA, prime, dB]]
        scale = np.linalg.norm(np.diff(route_coords, axis=0), axis=1).max(initial=1.0)
        split_nodes = _split_branch_nodes(
            G,
            prime,
            route_coords,
            fnT,
            VertexC,
            tol=endpoint_tol * scale,
            angle_tol=angle_tol,
        )
        if split_nodes is not None:
            splits[d] = (prime, split_nodes)
    return splits


def _branch_split_findings(
    splits: dict[int, tuple[int, tuple[int, int]]],
    polylines: list[_RoutePolyline],
    prime_paths: list[tuple[int, ...]],
    fnT: np.ndarray,
    VertexC: np.ndarray,
) -> list[dict[str, Any]]:
    """Return findings for polylines that traverse mapped detour splits."""
    if not splits:
        return []
    findings: list[dict[str, Any]] = []
    for polyline, prime_path in zip(polylines, prime_paths):
        non_root_end = (
            int(fnT[polyline.non_root_end])
            if polyline.non_root_end is not None
            else None
        )
        seen_primes: set[int] = set()
        for node in polyline.nodes[1:-1]:
            split = splits.get(node)
            if split is None:
                continue
            prime, split_nodes = split
            if prime in seen_primes or prime == non_root_end:
                continue
            seen_primes.add(prime)
            findings.append(
                {
                    'kind': 'branch_split',
                    'path_nodes_a': polyline.nodes,
                    'path_nodes_b': split_nodes,
                    'path_a': prime_path,
                    'path_b': (prime, prime, *split_nodes),
                    'split_node': prime,
                    'geometry': shp.Point(VertexC[prime].tolist()),
                }
            )
    return findings


def _exclusion_coords(
    path_a: tuple[int, ...],
    path_b: tuple[int, ...],
    fnT: np.ndarray,
    VertexC: np.ndarray,
    splits: dict[int, tuple[int, tuple[int, int]]],
) -> np.ndarray:
    """Coordinates where intersections are not crossings: shared nodes,
    endpoints of either polyline, and detour-split primes that both paths visit."""
    primes: set[int] = {int(fnT[n]) for n in path_a} & {int(fnT[n]) for n in path_b}
    primes |= {
        int(fnT[path_a[0]]),
        int(fnT[path_a[-1]]),
        int(fnT[path_b[0]]),
        int(fnT[path_b[-1]]),
    }
    for path in (path_a, path_b):
        for node in path[1:-1]:
            split = splits.get(node)
            if split is not None:
                primes.add(split[0])
    return VertexC[sorted(primes)]


def _iter_points(geometry) -> Iterator[tuple[float, float]]:
    """Yield (x, y) for each Point inside ``geometry``; line parts are skipped."""
    if geometry.geom_type == 'Point':
        yield geometry.x, geometry.y
    elif geometry.geom_type == 'MultiPoint':
        for point in geometry.geoms:
            yield point.x, point.y
    elif geometry.geom_type == 'GeometryCollection':
        for part in geometry.geoms:
            yield from _iter_points(part)


def _intersection_only_at_excluded(
    intersection,
    excluded: np.ndarray,
    *,
    endpoint_tol: float,
) -> bool:
    """``True`` if every Point of a length-0 intersection lies at an excluded coord."""
    if intersection.length > 0:
        return False
    points = list(_iter_points(intersection))
    if not points:
        return False
    P = np.asarray(points)
    # distance from each intersection point to each excluded coord
    diffs = P[:, None, :] - excluded[None, :, :]
    dists = np.hypot(diffs[..., 0], diffs[..., 1])
    return bool(np.all(np.any(dists <= endpoint_tol, axis=1)))


@dataclass(frozen=True)
class _RetracedRun:
    """A corridor that one open polyline traverses twice.

    ``run_segments``: segment indices of both traversals; ``members``: those
    plus the segments entering or leaving the corridor; ``vertices``: the
    corridor's coordinates.
    """

    run_segments: frozenset[int]
    members: frozenset[int]
    vertices: np.ndarray
    crosses: bool


def _retraced_runs(
    coords: np.ndarray, foreign_turbine_at: list[bool], *, angle_tol: float
) -> list[_RetracedRun]:
    """Find the corridors that an open polyline traverses more than once.

    Returns one run per maximal pair of traversals of a common vertex sequence.
    A pair crosses as decided by :func:`_shared_run_swaps_sides`, except a fold
    (the route turning back along the corridor), which never crosses. No run is
    returned for a fold at a vertex flagged in ``foreign_turbine_at``: a
    turbine with cables off the route, which the fold would wrap around.
    """
    n = len(coords)
    _, vid = np.unique(coords, axis=0, return_inverse=True)
    vid = vid.ravel().tolist()
    occurrences: dict[tuple[int, int], list[int]] = {}
    for k in range(n - 1):
        u, v = vid[k], vid[k + 1]
        occurrences.setdefault((u, v) if u < v else (v, u), []).append(k)
    # (i, j) -> step of j along the corridor: 1 if parallel, -1 if antiparallel
    pairs = {
        (i, j): 1 if vid[i] == vid[j] else -1
        for ks in occurrences.values()
        for i, j in combinations(ks, 2)
    }

    runs = []
    for (i, j), step in pairs.items():
        if pairs.get((i - 1, j - step)) == step:
            # not the start of a maximal run
            continue
        m = 1
        while pairs.get((i + m, j + step * m)) == step and (
            i + m < j if step > 0 else i + m < j - m
        ):
            m += 1
        # segments are i..i+m-1 for traversal a and b_segments for b; the
        # vertex spans lo:hi extend each by one segment at both ends
        b_segments = range(j, j + m) if step > 0 else range(j - m + 1, j + 1)
        fold = step < 0 and b_segments.start == i + m
        if fold and foreign_turbine_at[i + m]:
            continue
        lo_a, hi_a = max(i - 1, 0), min(i + m + 2, n)
        lo_b, hi_b = max(b_segments.start - 1, 0), min(b_segments.stop + 2, n)
        runs.append(
            _RetracedRun(
                run_segments=frozenset((*range(i, i + m), *b_segments)),
                members=frozenset((*range(lo_a, hi_a - 1), *range(lo_b, hi_b - 1))),
                vertices=coords[i : i + m + 1],
                crosses=not fold
                and _shared_run_swaps_sides(
                    coords[lo_a:hi_a], coords[lo_b:hi_b], angle_tol=angle_tol
                ),
            )
        )
    return runs


def _self_intersection_findings(
    polyline: _RoutePolyline,
    prime_path: tuple[int, ...],
    coords: np.ndarray,
    line,
    foreign_turbine_at: list[bool],
    *,
    include_touches: bool,
    length_tol: float,
    angle_tol: float,
    endpoint_tol: float,
) -> list[dict[str, Any]]:
    """Return improper intersections between segments of one route.

    A closed RINGED route has cyclic adjacency at its root. Its two arms may
    also share a corridor or meet at a routing vertex, just as two separate
    radial routes may; only a proper interior crossing is invalid there.

    The intersections of an open route along a corridor it traverses twice
    (see :func:`_retraced_runs`) are replaced by one ``self_overlap_cross`` if
    the traversals cross, else by a ``touch`` if ``include_touches``.
    """
    if line.is_simple:
        return []

    closed = polyline.nodes[0] == polyline.nodes[-1]
    segments = [shp.LineString(segment) for segment in pairwise(coords)]
    scale = max((segment.length for segment in segments), default=1.0) or 1.0
    runs = (
        []
        if closed
        else _retraced_runs(coords, foreign_turbine_at, angle_tol=angle_tol)
    )
    findings = [
        {
            'kind': 'self_overlap_cross' if run.crosses else 'touch',
            'path_nodes_a': polyline.nodes,
            'path_nodes_b': polyline.nodes,
            'path_a': prime_path,
            'path_b': prime_path,
            'geometry': shp.LineString(run.vertices),
        }
        for run in runs
        if run.crosses or include_touches
    ]
    tree = shp.STRtree(segments)
    seen = set()
    for i, segment_a in enumerate(segments):
        for j in tree.query(segment_a, predicate='intersects').tolist():
            if j <= i:
                continue
            intersection = segment_a.intersection(segments[j])
            if intersection.is_empty:
                continue
            if any(
                {i, j} <= run.run_segments
                if intersection.length > 0
                else {i, j} <= run.members
                and _intersection_only_at_excluded(
                    intersection, run.vertices, endpoint_tol=endpoint_tol * scale
                )
                for run in runs
            ):
                continue
            if closed:
                if intersection.geom_type != 'Point':
                    continue
                point = np.array(intersection.coords[0])
                corners = np.vstack((coords[i : i + 2], coords[j : j + 2]))
                if np.min(np.hypot(*(corners - point).T)) <= endpoint_tol * scale:
                    continue
            if (
                j == i + 1
                and intersection.geom_type == 'Point'
                and intersection.equals(shp.Point(coords[j]))
            ):
                continue
            key = intersection.wkb
            if key in seen:
                continue
            seen.add(key)
            findings.append(
                {
                    'kind': (
                        'self_overlap'
                        if intersection.length > length_tol
                        else 'self_cross'
                    ),
                    'path_nodes_a': polyline.nodes,
                    'path_nodes_b': polyline.nodes,
                    'path_a': prime_path,
                    'path_b': prime_path,
                    'geometry': intersection,
                }
            )
    return findings


def find_geometric_crossings(
    G: nx.Graph,
    *,
    include_touches: bool = False,
    length_tol: float = 1e-12,
    angle_tol: float = 1e-10,
    endpoint_tol: float = 1e-9,
) -> list[dict]:
    """Find invalid route intersections in routeset ``G`` using route polylines.

    The route decomposition contains one polyline per feeder and one per
    section between non-root nodes whose degree is not two. Clone nodes are
    translated through ``fnT`` to their prime coordinates. The checks cover
    intersections between polylines, self-intersections, coincident runs,
    routes that separate a terminal's incident rays, and degenerate geometry.

    A RINGED route returning to the root where it started is a closed polyline.
    Its first and last segments are cyclically adjacent, and its two arms may
    share routing vertices or corridors just as separate radial routes may. A
    ring bridging two roots remains an open polyline. All intersections between
    distinct routes are classified from their cable centerlines in the same way,
    regardless of topology.

    An open route may traverse a corridor (a vertex sequence) twice, e.g. when
    contouring into a pocket behind an obstacle corner and back out. As with two
    routes sharing a run, the traversals cross only if they leave the corridor
    on opposite sides at both ends; a fold (the route turning back along the
    corridor) or a traversal ending at the route's endpoint never does.
    Traversals that do not cross are reported only as a ``'touch'``.

    Args:
      G: routeset graph with ``T``, ``R``, ``B`` and ``VertexC`` graph
        attributes. ``fnT`` is required when ``C > 0`` or ``D > 0``.
      include_touches: also report point contacts that are not proper crossings
        (otherwise touches are silently dropped).
      length_tol: collinear overlaps shorter than this are not classified.
      angle_tol: minimum cross-product magnitude used to deduplicate co-directional
        rays in the local crossing test.
      endpoint_tol: distance below which an intersection point is treated as
        coincident with a path endpoint, shared node, or detour-split prime.

    Returns:
      One dict per finding, with the keys described below.

      - ``'kind'``: one of
          - ``'cross'``: two polylines cross at one or more isolated points;
          - ``'overlap_cross'``: two polylines share a sub-run and exit the
            overlap on opposite sides at both ends (a true cross expressed as
            a coincident segment);
          - ``'branch_split'``: a route through a real terminal's coordinate
            separates that terminal's incident topology rays;
          - ``'self_cross'``: non-adjacent segments of one route cross (for a
            closed ring, adjacency is cyclic and coincident runs are tolerated,
            as they are between two separate routes);
          - ``'self_overlap_cross'``: like ``'overlap_cross'``, for two
            traversals of one corridor by the same route;
          - ``'self_overlap'``: any other retrace of a route over itself, e.g.
            a partially coincident segment or a U-turn around a turbine with
            cables off the route;
          - ``'degenerate'``: a route lacks two finite distinct coordinates;
          - ``'touch'`` (only when ``include_touches=True``): point contact
            that is not classified as a cross (e.g. tangent kiss).
      - ``path_nodes_a``, ``path_nodes_b``: raw node sequences or the separated
        neighbour pair for a branch split.
      - ``path_a``, ``path_b``: prime-coordinate path descriptions. For a
        branch split, ``path_a`` is the passing route and ``path_b`` identifies
        the split terminal and two separated neighbours.
      - ``geometry``: WKT string of the offending Shapely geometry (Point,
        MultiPoint, LineString, MultiLineString, …).
    """
    VertexC = G.graph['VertexC']
    fnT = _routeset_fnT(G)
    polylines = _routeset_polylines(G)
    paths = [polyline.nodes for polyline in polylines]
    prime_paths = [_canonical_prime_path(G, path, fnT) for path in paths]
    path_vertices = [_polyline_primes(VertexC, fnT, path) for path in paths]
    path_coords = [VertexC[primes] for primes in path_vertices]
    path_primes = [{int(fnT[node]) for node in path} for path in paths]
    splits = _detour_splits(
        G,
        fnT,
        VertexC,
        endpoint_tol=endpoint_tol,
        angle_tol=angle_tol,
    )

    findings = _branch_split_findings(splits, polylines, prime_paths, fnT, VertexC)

    path_terminals = [
        {node for node in path if 0 <= node < G.graph['T']} for path in paths
    ]
    split_primes = [
        terminal
        for terminal in range(G.graph['T'])
        if terminal in G and G.degree[terminal] > 1
    ]
    split_points = [shp.Point(VertexC[terminal].tolist()) for terminal in split_primes]
    split_tree = shp.STRtree(split_points)

    lines = []
    line_paths = []
    for path_i, (polyline, prime_path, coords, vertices) in enumerate(
        zip(polylines, prime_paths, path_coords, path_vertices)
    ):
        if len(coords) < 2 or not np.isfinite(coords).all():
            geometry = (
                shp.Point(coords[0].tolist())
                if len(coords) and np.isfinite(coords[0]).all()
                else shp.GeometryCollection()
            )
            findings.append(
                {
                    'kind': 'degenerate',
                    'path_nodes_a': polyline.nodes,
                    'path_nodes_b': polyline.nodes,
                    'path_a': prime_path,
                    'path_b': prime_path,
                    'geometry': geometry,
                }
            )
            continue
        try:
            line = shp.LineString(coords)
        except (shp.errors.GEOSException, ValueError):
            findings.append(
                {
                    'kind': 'degenerate',
                    'path_nodes_a': polyline.nodes,
                    'path_nodes_b': polyline.nodes,
                    'path_a': prime_path,
                    'path_b': prime_path,
                    'geometry': shp.GeometryCollection(),
                }
            )
            continue
        findings += _self_intersection_findings(
            polyline,
            prime_path,
            coords,
            line,
            [
                0 <= prime < G.graph['T'] and prime not in polyline.nodes[1:-1]
                for prime in vertices.tolist()
            ],
            include_touches=include_touches,
            length_tol=length_tol,
            angle_tol=angle_tol,
            endpoint_tol=endpoint_tol,
        )
        lines.append(line)
        line_paths.append(path_i)

    tree = shp.STRtree(lines)
    seen_splits: set[tuple[int, int]] = set()

    for line_i, line_a in enumerate(lines):
        path_i = line_paths[line_i]
        for line_j in tree.query(line_a, predicate='intersects').tolist():
            if line_j <= line_i:
                continue
            path_j = line_paths[line_j]
            intersection = line_a.intersection(lines[line_j])
            if intersection.is_empty:
                continue

            classified_split_points = []
            for split_i in split_tree.query(
                intersection, predicate='intersects'
            ).tolist():
                prime = split_primes[split_i]
                in_i = prime in path_terminals[path_i]
                in_j = prime in path_terminals[path_j]
                if in_i == in_j:
                    continue
                route_path = path_j if in_i else path_i
                route_coords = path_coords[route_path]
                scale = np.linalg.norm(np.diff(route_coords, axis=0), axis=1).max(
                    initial=1.0
                )
                tol = endpoint_tol * scale
                key = route_path, prime
                if prime in path_primes[route_path]:
                    continue
                if key in seen_splits:
                    classified_split_points.append(VertexC[prime])
                    continue
                split_nodes = _split_branch_nodes(
                    G,
                    prime,
                    route_coords,
                    fnT,
                    VertexC,
                    tol=tol,
                    angle_tol=angle_tol,
                )
                if split_nodes is None:
                    continue
                classified_split_points.append(VertexC[prime])
                seen_splits.add(key)
                findings.append(
                    {
                        'kind': 'branch_split',
                        'path_nodes_a': paths[route_path],
                        'path_nodes_b': split_nodes,
                        'path_a': prime_paths[route_path],
                        'path_b': (prime, prime, *split_nodes),
                        'split_node': prime,
                        'geometry': split_points[split_i],
                    }
                )

            path_a, path_b = prime_paths[path_i], prime_paths[path_j]
            kind: str | None = None
            geometry = intersection

            excluded = _exclusion_coords(
                paths[path_i], paths[path_j], fnT, VertexC, splits
            )
            if classified_split_points:
                excluded = np.vstack((excluded, classified_split_points))
            if _intersection_only_at_excluded(
                intersection, excluded, endpoint_tol=endpoint_tol
            ):
                continue
            if intersection.length > length_tol and _shared_run_swaps_sides(
                path_coords[path_i], path_coords[path_j], angle_tol=angle_tol
            ):
                kind = 'overlap_cross'
            else:
                crossings = _filter_crossing_points(
                    intersection,
                    excluded,
                    path_coords[path_i],
                    path_coords[path_j],
                    angle_tol=angle_tol,
                    endpoint_tol=endpoint_tol,
                )
                if crossings:
                    kind = 'cross'
                    geometry = (
                        shp.Point(crossings[0])
                        if len(crossings) == 1
                        else shp.MultiPoint(crossings)
                    )
                elif include_touches:
                    kind = 'touch'
                else:
                    continue

            path_nodes_a, path_nodes_b = paths[path_i], paths[path_j]
            if path_b < path_a:
                path_nodes_a, path_nodes_b = path_nodes_b, path_nodes_a
                path_a, path_b = path_b, path_a
            findings.append(
                {
                    'kind': kind,
                    'path_nodes_a': path_nodes_a,
                    'path_nodes_b': path_nodes_b,
                    'path_a': path_a,
                    'path_b': path_b,
                    'geometry': geometry,
                }
            )

    return [{**finding, 'geometry': finding['geometry'].wkt} for finding in findings]


def _filter_crossing_points(
    intersection,
    excluded: np.ndarray,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    *,
    angle_tol: float,
    endpoint_tol: float,
) -> list[np.ndarray]:
    """Return point-intersections that are genuine X-crossings.

    Drops points near any excluded coord (shared nodes, polyline endpoints,
    detour-split primes) and points where the two routes' local rays don't
    alternate. Non-excluded route vertices are classified by their local rays.
    """
    P = np.asarray(list(_iter_points(intersection)))
    if len(P) == 0:
        return []
    if len(excluded):
        diffs = P[:, None, :] - excluded[None, :, :]
        near_excluded = np.any(
            np.hypot(diffs[..., 0], diffs[..., 1]) <= endpoint_tol, axis=1
        )
    else:
        near_excluded = np.zeros(len(P), dtype=bool)
    # tolerance scales with the largest segment among either polyline
    scale = max(
        np.linalg.norm(np.diff(coords_a, axis=0), axis=1).max(initial=1.0),
        np.linalg.norm(np.diff(coords_b, axis=0), axis=1).max(initial=1.0),
    )
    tol = endpoint_tol * scale
    crossings = []
    for k, point in enumerate(P):
        if near_excluded[k]:
            continue
        rays_a = polyline_rays_at_point(coords_a, point, tol=tol, angle_tol=angle_tol)
        rays_b = polyline_rays_at_point(coords_b, point, tol=tol, angle_tol=angle_tol)
        if rays_alternate(rays_a, rays_b):
            crossings.append(point)
    return crossings


def list_edge_crossings(
    S: nx.Graph, A: nx.Graph
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """List edge×edge crossings for the network topology in S.

    ``S`` must only use extended Delaunay edges. It will not detect crossings
    of non-extDelaunay gates or detours.

    Args:
      S: solution topology
      A: available links used in creating ``S``

    Returns:
      list of 2-tuple (crossing) of 2-tuple (edge, ordered)
    """
    eeXings = []
    checked = set()
    diagonals = A.graph['diagonals']
    for u, v in S.edges:
        u, v = (u, v) if u < v else (v, u)
        st = diagonals.get((u, v))
        if st is not None:
            # ⟨u, v⟩ is a diagonal of Delanay edge ⟨s, t⟩
            if st in S.edges:
                # crossing with Delaunay edge ⟨s, t⟩
                eeXings.append((st, (u, v)))
            s, t = st
            # ⟨s, t⟩ may be part of up to two triangles, check their 4 sides
            sides = (
                ((w, y) if w < y else (y, w))
                for w, y in ((u, s), (s, v), (v, t), (t, u))
            )
            for side in sides:
                diag = diagonals.inv.get(side, False)
                if diag and diag in S.edges and diag not in checked:
                    checked.add((u, v))
                    eeXings.append((diag, (u, v)))
    return eeXings
