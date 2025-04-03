# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import math
import operator
from collections import defaultdict
from itertools import combinations, pairwise, product
from math import isclose
from typing import Callable, Literal, NewType

import networkx as nx
import numba as nb
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_array
from scipy.sparse.csgraph import minimum_spanning_tree as scipy_mst
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from .utils import F, NodeStr

__all__ = (
    'triangle_AR',
    'point_d2line',
    'is_same_side',
    'any_pairs_opposite_edge',
    'rotate',
    'angle_numpy',
    'angle',
    'angle_oracles_factory',
    'find_edges_bbox_overlaps',
    'is_crossing_numpy',
    'is_crossing_no_bbox',
    'is_crossing',
    'is_bunch_split_by_corner',
    'is_triangle_pair_a_convex_quadrilateral',
    'perimeter',
    'angle_helpers',
    'assign_root',
    'get_crossings_map',
    'complete_graph',
    'minimum_spanning_forest',
    'rotation_checkers_factory',
    'rotating_calipers',
    'area_from_polygon_vertices',
)

NULL = np.iinfo(int).min

CoordPair = NewType('CoordPair', np.ndarray[tuple[Literal[2]], np.dtype[np.float64]])
CoordPairs = NewType(
    'CoordPairs', np.ndarray[tuple[int, Literal[2]], np.dtype[np.float64]]
)
IndexPairs = NewType(
    'IndexPairs', np.ndarray[tuple[int, Literal[2]], np.dtype[np.int_]]
)
Indices = NewType('Indices', np.ndarray[tuple[int], np.dtype[np.int_]])


@nb.njit(cache=True)
def triangle_AR(base1C: CoordPair, base2C: CoordPair, topC: CoordPair) -> float:
    """Calculate the ratio: dist(base1, base2)/dist(base, top).

    Numerator is the length of the base of the triagle (base1C, base2C).

    Denominator is the distance from point topC to the base line.

    Args:
      uC, vC, tC: triangle vertices coordinates as (2,) numpy arrays

    Returns:
      Aspect ratio of the triangle defined by the three 2D points.
    """
    x1, y1 = base1C
    x2, y2 = base2C
    xt, yt = topC

    dx = x2 - x1
    dy = y2 - y1
    den = abs(dy * xt - dx * yt + x2 * y1 - y2 * x1)
    if den < 1e-12:
        return np.inf
    base_sqr = dx**2 + dy**2
    return base_sqr / den


@nb.njit(cache=True)
def point_d2line(pC: CoordPair, uC: CoordPair, vC: CoordPair) -> np.float64:
    """Calculate the distance from point `pC` to the `uC`-`vC` line."""
    x0, y0 = pC
    x1, y1 = uC
    x2, y2 = vC
    return abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / np.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2
    )


@nb.njit(cache=True)
def is_same_side(
    uC: CoordPair,
    vC: CoordPair,
    sC: CoordPair,
    tC: CoordPair,
    touch_is_cross: bool = True,
) -> bool:
    """Check if points `sC` an `tC` are on the same side of the line defined
    by points `uC` and `vC`.

    Note: often used to check crossings with feeder links, where the feeder link
    `sC`-`tC` is already known to be on a line that crosses the edge `uC`–`vC`
    (using the angle rank).
    """

    denom = uC[0] - vC[0]
    # test to avoid division by zero
    if denom != 0:
        a = -(uC[1] - vC[1]) / denom
        c = -a * uC[0] - uC[1]
        num = a * sC[0] + sC[1] + c
        den = a * tC[0] + tC[1] + c
        #  discriminator = float(num*den)
        discriminator = num * den
    else:
        # this means the line is vertical (uC[0] = vC[0])
        # which makes the test simpler
        #  discriminator = float((sC[0] - uC[0])*(tC[0] - uC[0]))
        discriminator = (sC[0] - uC[0]) * (tC[0] - uC[0])
    return discriminator > 0.0 or (touch_is_cross and discriminator == 0.0)


@nb.njit(cache=True)
def any_pairs_opposite_edge(
    nodesC: CoordPairs, uC: CoordPair, vC: CoordPair, margin: float = 0.0
) -> bool:
    """Compare relative position of vertices wrt line segment.

    Args:
      nodesC: (N, 2) array of test coordinates
      uC, vC: (2,) array of coordinates of edge ends

    Returns:
      True if any two of `nodesC` are on opposite sides of the edge.
    """
    maxidx = len(nodesC) - 1
    if maxidx <= 0:
        return False
    refC = nodesC[0]
    i = 1
    while point_d2line(refC, uC, vC) <= margin:
        # ref node is approx. overlapping the edge: get the next one
        refC = nodesC[i]
        i += 1
        if i > maxidx:
            return False

    for cmpC in nodesC[i:]:
        if point_d2line(cmpC, uC, vC) <= margin:
            # cmp node is approx. overlapping the edge: skip
            continue
        if not is_same_side(uC, vC, refC, cmpC, touch_is_cross=False):
            return True
    return False


@nb.njit(cache=True)
def rotate(coords: CoordPairs, angle: float) -> CoordPairs:
    """Rotates `coords` (numpy array T×2) by `angle` (degrees)"""
    rotation = np.deg2rad(angle)
    c, s = np.cos(rotation), np.sin(rotation)
    return np.dot(coords, np.array([[c, s], [-s, c]]))


@nb.njit(cache=True)
def angle_numpy(
    aC: CoordPairs, pivotC: CoordPairs, bC: CoordPairs
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Calculate the angle a-pivot-b.

    * can operate on multiple point triplets
    * angle is within ±π (shortest arc from a to b around pivot)
    * positive direction is counter-clockwise

    Args:
      aC, pivotC, bC: (N, 2) numpy arrays of coordinate pairs

    Returns:
      Angles a-pivot-b (radians)
    """
    A = aC - pivotC
    B = bC - pivotC
    # dot_prod = np.dot(A, B) if len(A) >= len(B) else np.dot(B, A)
    dot_prod = A @ B.T  # if len(A) >= len(B) else np.dot(B, A)
    # return np.arctan2(np.cross(A, B), np.dot(A, B))
    return np.arctan2(np.cross(A, B), dot_prod)


def angle(aC, pivotC, bC):
    """Calculate the angle aC-pivotC-bC.

    * angle is within ±π (shortest arc from a to b around pivot)
    * positive direction is counter-clockwise

    Args:
      aC, pivotC, bC: (2,) numpy arrays of coordinate pairs

    Returns:
      Angle aC-pivotC-bC (radians)
    """
    Ax, Ay = aC - pivotC
    Bx, By = bC - pivotC
    # debug and print(VertexC[a], VertexC[b])
    ang = np.arctan2(Ax * By - Ay * Bx, Ax * Bx + Ay * By)
    # debug and print(f'{ang*180/np.pi:.1f}')
    return ang


def angle_oracles_factory(
    angles: NDArray[np.float64], anglesRank: NDArray[np.int_]
) -> tuple[
    Callable[[int, int, int, int, int, int, int], tuple[float, float]],
    Callable[[int, int, int], float],
]:
    """Make functions to answer queries about relative angles.

    Inputs are the outputs of `angle_helpers()`.

    Args:
      angles: (T, R)-array of angles wrt root (+-pi)
      anglesRank: (T, R)-array of the relative placement of angles

    Returns:
      union_limits() and angle_ccw()
    """

    def is_within(pR: int, lR: int, hR: int) -> bool:
        W = lR > hR  # wraps, i.e. angle(low) > angle(high)
        L = lR <= pR  # angle(low) <= angle(probe)
        H = pR <= hR  # angle(probe) <= angle(high)
        return (not W and L and H) or (W and not L and H) or (W and L and not H)

    def union_limits(
        root: int, u: int, LO: int, HI: int, v: int, lo: int, hi: int
    ) -> tuple[float, float]:
        LOR, HIR, loR, hiR = anglesRank[(LO, HI, lo, hi), root]
        lo_within = is_within(loR, LOR, HIR)
        hi_within = is_within(hiR, LOR, HIR)
        if lo_within and hi_within:
            # print('LOHI contains lohi')
            return LO, HI
        elif lo_within:
            # print('partial overlap, hi outside LOHI')
            return LO, hi
        elif hi_within:
            # print('partial overlap, lo outside LOHI')
            return lo, HI
        elif is_within(LOR, loR, hiR):
            # print('lohi contains LOHI')
            return lo, hi
        else:
            # print('LOHI and lohi are disjoint')
            uvA = angle_ccw(u, root, v)
            return (LO, hi) if uvA <= math.pi else (lo, HI)

    def angle_ccw(a: int, pivot: int, b: int) -> float:
        """Calculate the angle a-pivot-b.

        * angle is ccw from 0 to 2π

        Args:
          a, pivot, b: vertex indices

        Returns:
          Angle a-pivot-b (radians)
        """
        if a == b:
            return 0.0
        aR, bR = anglesRank[(a, b), pivot]
        aA, bA = angles[(a, b), pivot]
        a_to_bA = (bA - aA).item()
        return a_to_bA if aR <= bR else (2 * math.pi + a_to_bA)

    return union_limits, angle_ccw


def find_edges_bbox_overlaps(
    VertexC: CoordPairs, u: int, v: int, edges: IndexPairs
) -> NDArray[np.int_]:
    """Find which `edges` have a bounding box overlap with ⟨u, v⟩.

    This is a preliminary filter for crossing checks. Enables avoiding the more
    costly geometric crossing calculations for segments that are clearly
    disjoint.

    Args:
      VertexC: (N×2) point coordinates
      u, v: indices of probed edge
      edges: list of index pairs representing edges to check against

    Returns:
      numpy array with the indices of overlaps in `edges`
    """
    uC, vC = VertexC[u], VertexC[v]
    edgesC = VertexC[edges]
    return np.flatnonzero(
        ~np.logical_or(
            (edgesC > np.maximum(uC, vC)).all(axis=1),
            (edgesC < np.minimum(uC, vC)).all(axis=1),
        ).any(axis=1)
    )


def is_crossing_numpy(u, v, s, t):
    """Checks if (u, v) crosses (s, t).

    Returns:
      True in case of crossing.
    """
    # TODO: document output in corner cases (e.g. superposition, touch)

    # adapted from Franklin Antonio's insectc.c lines_intersect()
    # Faster Line Segment Intersection
    # Graphics Gems III (http://www.graphicsgems.org/)
    # license: https://github.com/erich666/GraphicsGems/blob/master/LICENSE.md

    A = v - u
    B = s - t

    # bounding box check
    for i in (0, 1):  # X and Y
        lo, hi = (v[i], u[i]) if A[i] < 0 else (u[i], v[i])
        if B[i] > 0:
            if hi < t[i] or s[i] < lo:
                return False
        else:
            if hi < s[i] or t[i] < lo:
                return False

    C = u - s

    # denominator
    f = np.cross(B, A)
    if f == 0:
        # segments are parallel
        return False

    # alpha and beta numerators
    for num in (np.cross(P, Q) for P, Q in ((C, B), (A, C))):
        if f > 0:
            if num < 0 or num > f:
                return False
        else:
            if num > 0 or num < f:
                return False

    # code to calculate intersection coordinates omitted
    # segments do cross
    return True


@nb.njit(cache=True)
def _cross_prod_2d(P: CoordPair, Q: CoordPair) -> np.float64:
    return P[0] * Q[1] - P[1] * Q[0]


@nb.njit(cache=True)
def is_crossing_no_bbox(
    uC: CoordPair, vC: CoordPair, sC: CoordPair, tC: CoordPair
) -> bool:
    """Checks if (uC, vC) crosses (sC, tC).

    Does not check for bounding-box overlap. Use `find_edges_bbox_overlap()`
    first to filter out edges with disjoint bounding boxes (cheaper than the
    calculations here).

    Returns:
      True in case of crossing.
    """
    # TODO: document output in corner cases (e.g. superposition, touch)

    # adapted from Franklin Antonio's insectc.c lines_intersect()
    # Faster Line Segment Intersection
    # Graphic Gems III
    A = vC - uC
    B = sC - tC
    C = uC - sC

    # denominator
    f = _cross_prod_2d(B, A)
    # TODO: arbitrary threshold
    if abs(f) < 1e-10:
        # segments are parallel
        return False

    # alpha and beta numerators
    #  for num in (Px*Qy - Py*Qx for (Px, Py), (Qx, Qy) in ((C, B), (A, C))):
    for P, Q in ((C, B), (A, C)):
        num = _cross_prod_2d(P, Q)
        if f > 0:
            if num < 0 or f < num:
                return False
        else:
            if 0 < num or num < f:
                return False

    # code to calculate intersection coordinates omitted
    # segments do cross
    return True


def is_crossing(
    uC: CoordPair,
    vC: CoordPair,
    sC: CoordPair,
    tC: CoordPair,
    touch_is_cross: bool = True,
) -> bool:
    """Checks if (uC, vC) crosses (sC, tC).

    Args:
      uC, vC, sC, tC: (2,) numpy array coordinates of edge ends
      touch_is_cross: whether to consider any common point as a crossing

    Returns:
      True in case of crossing.
    """
    # choices for `less`:
    # -> operator.lt counts touching as crossing
    # -> operator.le does not count touching as crossing
    less = operator.lt if touch_is_cross else operator.le

    # adapted from Franklin Antonio's insectc.c lines_intersect()
    # Faster Line Segment Intersection
    # Graphic Gems III

    A = vC - uC
    B = sC - tC

    # bounding box check
    for i in (0, 1):  # X and Y
        lo, hi = (vC[i], uC[i]) if A[i] < 0 else (uC[i], vC[i])
        if B[i] > 0:
            if hi < tC[i] or sC[i] < lo:
                return False
        else:
            if hi < sC[i] or tC[i] < lo:
                return False

    Ax, Ay = A
    Bx, By = B
    C = uC - sC

    # denominator
    # print(Ax, Ay, Bx, By)
    f = Bx * Ay - By * Ax
    # print('how close: ', f)
    # TODO: arbitrary threshold
    if isclose(f, 0.0, abs_tol=1e-10):
        # segments are parallel
        return False

    # alpha and beta numerators
    for num in (Px * Qy - Py * Qx for (Px, Py), (Qx, Qy) in ((C, B), (A, C))):
        if f > 0:
            if less(num, 0) or less(f, num):
                return False
        else:
            if less(0, num) or less(num, f):
                return False

    # code to calculate intersection coordinates omitted
    # segments do cross
    return True


def is_bunch_split_by_corner(bunch, a, o, b, margin=1e-3):
    """Check if a cone splits a bunch of points in two sets.

    Args:
      bunch: numpy array of points (T×2)
      a, o, b: points that define the cone's angle

    Returns:
      True if points in bunch are both inside and outside cone a-o-b
    """
    AngleA = angle_numpy(a, o, bunch)
    AngleB = angle_numpy(b, o, bunch)
    # print('AngleA', AngleA, 'AngleB', AngleB)
    # keep only those that don't fall over the angle-defining lines
    keep = ~np.logical_or(
        np.isclose(AngleA, 0, atol=margin), np.isclose(AngleB, 0, atol=margin)
    )
    angleAB = angle(a, o, b)
    angAB = angleAB > 0
    inA = AngleA > 0 if angAB else AngleA < 0
    inB = AngleB > 0 if ~angAB else AngleB < 0
    # print(angleAB, keep, inA, inB)
    inside = np.logical_and(keep, np.logical_and(inA, inB))
    outside = np.logical_and(keep, np.logical_or(~inA, ~inB))
    split = any(inside) and any(outside)
    return split, np.flatnonzero(inside), np.flatnonzero(outside)


@nb.njit(cache=True)
def is_triangle_pair_a_convex_quadrilateral(
    uC: CoordPair, vC: CoordPair, sC: CoordPair, tC: CoordPair
) -> bool:
    """Check convexity of quadrilateral.

    ⟨u, v⟩ is the common side; ⟨s, t⟩ are the opposing vertices;
    only works if ⟨s, t⟩ crosses the line defined by ⟨u, v⟩

    Returns:
      True if the quadrilateral is convex and is not a triangle
    """
    # this used to be called `is_quadrilateral_convex()`
    # us × ut
    usut = _cross_prod_2d(sC - uC, tC - uC)
    # vt × vs
    vtvs = _cross_prod_2d(tC - vC, sC - vC)
    if usut == 0.0 or vtvs == 0.0:
        # the four vertices form a triangle
        return False
    return (usut > 0.0) == (vtvs > 0.0)


def is_blocking(
    rootC: CoordPair, uC: CoordPair, vC: CoordPair, sC: CoordPair, tC: CoordPair
) -> bool:
    """DEPRECATED

    This is used only by apply_edge_exemptions()
    """
    # s and t are necessarily on opposite sides of uv
    # (because of Delaunay – see the triangles construction)
    # hence, if (root, t) are on the same side, (s, root) are not
    return (
        is_triangle_pair_a_convex_quadrilateral(uC, vC, sC, rootC)
        if is_same_side(uC, vC, rootC, tC)
        else is_triangle_pair_a_convex_quadrilateral(uC, vC, rootC, tC)
    )


def apply_edge_exemptions(G, allow_edge_deletion=True):
    """DEPRECATED (depends on 'E_hull' and 'N_hull' graph attributes)

    Exemption is used by weighting functions that take into account the angular
    sector blocked by each edge w.r.t. the closest root node.
    """
    E_hull = G.graph['E_hull']
    N_hull = G.graph['N_hull']
    N_inner = set(G.nodes) - N_hull
    R = G.graph['R']
    # T = G.number_of_nodes() - R
    VertexC = G.graph['VertexC']
    # roots = range(T, T + R)
    roots = range(-R, 0)
    triangles = G.graph['triangles']
    angles = G.graph['angles']

    # set hull edges as exempted
    for edge in E_hull:
        G.edges[edge]['exempted'] = True

    # expanded E_hull to contain edges exempted from blockage penalty
    # (edges that do not block line from nodes to root)
    E_hull_exp = E_hull.copy()

    # check if edges touching the hull should be exempted from blockage penalty
    for n_hull in N_hull:
        for n_inner in N_inner & set([v for _, v in G.edges(n_hull)]):
            uv = frozenset((n_hull, n_inner))
            u, v = uv
            opposites = triangles[uv]
            if len(opposites) == 2:
                s, t = triangles[uv]
                rootC = VertexC[G.edges[u, v]['root']]
                uvstC = tuple((VertexC[n] for n in (*uv, s, t)))
                if not is_blocking(rootC, *uvstC):
                    E_hull_exp.add(uv)
                    G.edges[uv]['exempted'] = True

    # calculate blockage arc for each edge
    zeros = np.full((R,), 0.0)
    for u, v, d in list(G.edges(data=True)):
        if (frozenset((u, v)) in E_hull_exp) or (u in roots) or (v in roots):
            angdiff = zeros
        else:
            # angdiff = (angles[:, u] - angles[:, v]) % (2*np.pi)
            # angdiff = abs(angles[:, u] - angles[:, v])
            angdiff = abs(angles[u] - angles[v])
        arc = np.empty((R,), dtype=float)
        for i in range(R):  # TODO: vectorize this loop
            arc[i] = angdiff[i] if angdiff[i] < np.pi else 2 * np.pi - angdiff[i]
        d['arc'] = arc
        # if arc is π/2 or more, remove the edge (it's shorter to go to root)
        if allow_edge_deletion and any(arc >= np.pi / 2):
            G.remove_edge(u, v)
            print('angles', arc, 'removing «', '–'.join([F[n] for n in (u, v)]), '»')


def perimeter(VertexC, vertices_ordered):
    """Calculate the perimeter of the polygon defined by `vertices_ordered`.

    Args:
      vertices_ordered: indices of `VertexC` in clockwise or counter-clockwise
        orientation.

    Return:
      The perimeter length.
    """
    vec = VertexC[vertices_ordered[:-1]] - VertexC[vertices_ordered[1:]]
    return np.hypot(*vec.T).sum() + np.hypot(
        *(VertexC[vertices_ordered[-1]] - VertexC[vertices_ordered[0]])
    )


def angle_helpers(L: nx.Graph) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Create auxiliary arrays of node attributes based on polar coordinates.

    Args:
        L: location (also works with A or G)

    Returns:
        Tuple of (angles, anglesRank)
    """

    T, R, VertexC = (L.graph[k] for k in ('T', 'R', 'VertexC'))
    B = L.graph.get('B', 0)
    NodeC = VertexC[: T + B]
    RootC = VertexC[-R:]

    angles = np.empty((T + B, R), dtype=float)
    for n, nodeC in enumerate(NodeC):
        x, y = (nodeC - RootC).T
        angles[n] = np.arctan2(y, x)

    anglesRank = rankdata(angles, method='dense', axis=0)
    return angles, anglesRank


def assign_root(A: nx.Graph) -> None:
    """Add node attribute 'root' with the root closest to each node.

    Changes A in-place.

    Args:
      A: available-edges graph
    """
    closest_root = -A.graph['R'] + np.argmin(A.graph['d2roots'], axis=1)
    nx.set_node_attributes(A, {n: r.item() for n, r in enumerate(closest_root)}, 'root')


# TODO: get new implementation from Xings.ipynb
# xingsmat, edge_from_Eidx, Eidx__
def get_crossings_map(Edge, VertexC, prune=True):
    crossings = defaultdict(list)
    for i, A in enumerate(Edge[:-1]):
        u, v = A
        uC, vC = VertexC[A]
        for B in Edge[i + 1 :]:
            s, t = B
            if s == u or t == u or s == v or t == v:
                # <edges have a common node>
                continue
            sC, tC = VertexC[B]
            if is_crossing(uC, vC, sC, tC):
                crossings[frozenset((*A,))].append((*B,))
                crossings[frozenset((*B,))].append((*A,))
    return crossings


# TODO: test this implementation
def complete_graph(
    G_base: nx.Graph,
    *,
    include_roots: bool = False,
    prune: bool = True,
    map_crossings: bool = False,
) -> nx.Graph:
    """Create a complete graph based on G_base.

    Produces a networkx Graph connecting all non-root nodes to every
    other non-root node. Edges with an arc > pi/2 around root are discarded
    The length of each edge is the euclidean distance between its vertices.
    """
    R, T = (G_base.graph[k] for k in 'RT')
    VertexC = G_base.graph['VertexC']
    TerminalC = VertexC[:T]
    RootC = VertexC[-R:]
    NodeC = np.vstack((TerminalC, RootC))
    Root = range(-R, 0)
    V = T + (R if include_roots else 0)
    G = nx.complete_graph(V)
    EdgeComplete = np.column_stack(np.triu_indices(V, k=1))
    #  mask = np.zeros((V,), dtype=bool)
    mask = np.zeros_like(EdgeComplete[:, 0], dtype=bool)
    if include_roots:
        # mask root-root edges
        offset = 0
        for i in range(0, R - 1):
            for j in range(0, R - i - 1):
                mask[offset + j] = True
            offset += V - i - 1

        # make node indices span -R:(T - 1)
        EdgeComplete -= R
        nx.relabel_nodes(G, dict(zip(range(T, T + R), Root)), copy=False)
        C = cdist(NodeC, NodeC)
    else:
        C = cdist(TerminalC, TerminalC)
    if prune:
        # prune edges that cover more than 90° angle from any root
        SrcC = VertexC[EdgeComplete[:, 0]]
        DstC = VertexC[EdgeComplete[:, 1]]
        for root in Root:
            rootC = VertexC[root]
            # calculates the dot product of vectors representing the
            # nodes of each edge wrt root; then mark the negative ones
            # (angle > pi/2) on `mask`
            mask |= ((SrcC - rootC) * (DstC - rootC)).sum(axis=1) < 0
    Edge = EdgeComplete[~mask]
    # discard masked edges
    G.remove_edges_from(EdgeComplete[mask])
    if map_crossings:
        # get_crossings_map() takes time and space
        G.graph['crossings_map'] = get_crossings_map(Edge, VertexC)
    # assign nodes to roots?
    # remove edges between nodes belonging to distinct roots whose length is
    # greater than both d2root
    G.graph.update(G_base.graph)
    G.graph['d2roots'] = cdist(TerminalC, RootC)
    nx.set_node_attributes(G, G_base.nodes)
    for u, v, edgeD in G.edges(data=True):
        edgeD['length'] = C[u, v]
        # assign the edge to the root closest to the edge's middle point
        edgeD['root'] = -R + np.argmin(
            cdist(((VertexC[u] + VertexC[v]) / 2)[np.newaxis, :], RootC)
        )
    return G


def minimum_spanning_forest(A: nx.Graph) -> nx.Graph:
    """Create the minimum spanning forest from the Delaunay edges of `A`.

    There is one tree for each root and exactly one root per tree.
    If the graph has more than one root, the minimum spanning tree of the
    entire graph is split on its longest links between each root pair.

    Returns:
      Topology S containing the forest.
    """
    R, T = (A.graph[k] for k in 'RT')
    N = R + T
    P_A = A.graph['planar']
    num_edges = P_A.number_of_edges()
    edges_ = np.empty((num_edges // 2, 2), dtype=np.int32)
    length_ = np.empty(edges_.shape[0], dtype=np.float64)
    for i, (u, v) in enumerate((u, v) for u, v in P_A.edges if u < v):
        edges_[i] = u, v
        length_[i] = A[u][v]['length']
    edges_[edges_ < 0] += N
    P_ = coo_array((length_, (*edges_.T,)), shape=(N, N))
    Q_ = scipy_mst(P_)
    U, V = Q_.nonzero()
    U[U >= T] -= N
    V[V >= T] -= N
    S = nx.Graph(
        T=T,
        R=R,
        capacity=T,
        handle=A.graph.get('handle'),
        creator='minimum_spanning_forest',
    )
    for u, v in zip(U, V):
        S.add_edge(u.item(), v.item(), length=Q_[u, v].item())
    if R > 1:
        # if multiple roots, split the MST in multiple trees
        removals = R - 1
        pair_checks = combinations(range(-R, 0), 2)
        paths = []
        while removals:
            if not paths:
                r1, r2 = next(pair_checks)
                try:
                    path = nx.bidirectional_shortest_path(S, r1, r2)
                except nx.NetworkXNoPath:
                    continue
                i = 0
                for j, p in enumerate(path[1:-1], 1):
                    if p < 0:
                        # split path
                        paths.append(path[i : j + 1])
                        i = j
                paths.append(path[i:])
            path = paths.pop()
            λ_incumbent = 0.0
            uv_incumbent = None
            for u, v, λ_hop in ((u, v, A[u][v]['length']) for u, v in pairwise(path)):
                if λ_hop > λ_incumbent:
                    λ_incumbent = λ_hop
                    uv_incumbent = u, v
            S.remove_edge(*uv_incumbent)
            removals -= 1
    return S


# TODO: MARGIN is ARBITRARY - depends on the scale
def check_crossings(G, debug=False, MARGIN=0.1):
    """DEPRECATED. Use functions from module `crossings` instead.

    Checks for crossings (touch/overlap is not considered crossing).
    This is an independent check on the tree resulting from the heuristic.
    It is not supposed to be used within the heuristic.
    MARGIN is how far an edge can advance across another one and still not be
    considered a crossing.
    """
    VertexC = G.graph['VertexC']
    R, T, B = (G.graph[k] for k in 'RTB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    raise NotImplementedError('CDT requires changes in this function')
    if C > 0 or D > 0:
        # detournodes = range(T, T + D)
        # G.add_nodes_from(((s, {'kind': 'detour'})
        #                   for s in detournodes))
        # clone2prime = G.graph['clone2prime']
        # assert len(clone2prime) == D, \
        #     'len(clone2prime) != D'
        # fnT = np.arange(T + D + R)
        # fnT[T: T + D] = clone2prime
        # DetourC = VertexC[clone2prime].copy()
        fnT = G.graph['fnT']
        AllnodesC = np.vstack((VertexC[:T], VertexC[fnT[T : T + D]], VertexC[-R:]))
    else:
        fnT = np.arange(T + R)
        AllnodesC = VertexC
    roots = range(-R, 0)
    fnT[-R:] = roots
    n2s = NodeStr(fnT, T)

    crossings = []
    pivot_plus_edge = []

    def check_neighbors(neighbors, w, x, pivots):
        """Neighbors is a bunch of nodes, `pivots` is used only for reporting.
        (`w`, `x`) is the edge to be checked if it splits neighbors apart.
        """
        maxidx = len(neighbors) - 1
        if maxidx <= 0:
            return
        ref = neighbors[0]
        i = 1
        while point_d2line(*AllnodesC[[ref, w, x]]) < MARGIN:
            # ref node is approx. overlapping the edge: get the next one
            ref = neighbors[i]
            i += 1
            if i > maxidx:
                return

        for n2test in neighbors[i:]:
            if point_d2line(*AllnodesC[[n2test, w, x]]) < MARGIN:
                # cmp node is approx. overlapping the edge: skip
                continue
            # print(F[fnT[w]], F[fnT[x]], F[fnT[ref]], F[fnT[cmp]])
            if not is_same_side(*AllnodesC[[w, x, ref, n2test]], touch_is_cross=False):
                print(
                    f'ERROR <splitting>: edge {n2s(w, x)} crosses '
                    f'{n2s(ref, *pivots, n2test)}'
                )
                # crossings.append(((w,  x), (ref, pivot, cmp)))
                crossings.append(((w, x), (ref, n2test)))
                return True

    # TODO: check crossings among edges connected to different roots
    for root in roots:
        # edges = list(nx.edge_dfs(G, source=root))
        edges = list(nx.edge_bfs(G, source=root))
        # outstr = ', '.join([f'«{F[fnT[u]]}–{F[fnT[v]]}»' for u, v in edges])
        # print(outstr)
        potential = []
        for i, (u, v) in enumerate(edges):
            u_, v_ = fnT[u], fnT[v]
            for s, t in edges[(i + 1) :]:
                s_, t_ = fnT[s], fnT[t]
                if s_ == u_ or s_ == v_ or t_ == u_ or t_ == v_:
                    # no crossing if the two edges share a vertex
                    continue
                uvst = np.array((u, v, s, t), dtype=int)
                if is_crossing(*AllnodesC[uvst], touch_is_cross=True):
                    potential.append(uvst)
                    distances = np.fromiter(
                        (
                            point_d2line(*AllnodesC[[p, w, x]])
                            for p, w, x in ((u, s, t), (v, s, t), (s, u, v), (t, u, v))
                        ),
                        dtype=float,
                        count=4,
                    )
                    # print('distances[' +
                    #       ', '.join((F[fnT[n]] for n in (u, v, s, t))) +
                    #       ']: ', distances)
                    nearmask = distances < MARGIN
                    close_count = sum(nearmask)
                    # print('close_count =', close_count)
                    if close_count == 0:
                        # (u, v) crosses (s, t) away from nodes
                        crossings.append(((u, v), (s, t)))
                        # print(distances)
                        print(
                            f'ERROR <edge-edge>: '
                            f'edge «{F[fnT[u]]}–{F[fnT[v]]}» '
                            f'crosses «{F[fnT[s]]}–{F[fnT[t]]}»'
                        )
                    elif close_count == 1:
                        # (u, v) and (s, t) touch node-to-edge
                        (pivotI,) = np.flatnonzero(nearmask)
                        w, x = (u, v) if pivotI > 1 else (s, t)
                        pivot = uvst[pivotI]
                        neighbors = list(G[pivot])
                        entry = (pivot, w, x)
                        if entry not in pivot_plus_edge and check_neighbors(
                            neighbors, w, x, (pivot,)
                        ):
                            pivot_plus_edge.append(entry)
                    elif close_count == 2:
                        # TODO: This case probably never happens, remove it.
                        #       This would only happen for coincident vertices,
                        #       which might have been possible in the past.
                        print('&&&&& close_count = 2 &&&&&')
                        # (u, v) and (s, t) touch node-to-node
                        touch_uv, touch_st = uvst[np.flatnonzero(nearmask)]
                        free_uv, free_st = uvst[np.flatnonzero(~nearmask)]
                        # print(
                        #    f'touch/free u, v :«{F[fnT[touch_uv]]}–'
                        #    f'{F[fnT[free_uv]]}»; s, t:«{F[fnT[touch_st]]}–'
                        #    f'{F[fnT[free_st]]}»')
                        nb_uv, nb_st = list(G[touch_uv]), list(G[touch_st])
                        # print([F[fnT[n]] for n in nb_uv])
                        # print([F[fnT[n]] for n in nb_st])
                        nbNuv, nbNst = len(nb_uv), len(nb_st)
                        if nbNuv == 1 or nbNst == 1:
                            # <a leaf node with a clone – not a crossing>
                            continue
                        elif nbNuv == 2:
                            crossing = is_bunch_split_by_corner(
                                AllnodesC[nb_st],
                                *AllnodesC[[nb_uv[0], touch_uv, nb_uv[1]]],
                                margin=MARGIN,
                            )[0]
                        elif nbNst == 2:
                            crossing = is_bunch_split_by_corner(
                                AllnodesC[nb_uv],
                                *AllnodesC[[nb_st[0], touch_st, nb_st[1]]],
                                margin=MARGIN,
                            )[0]
                        else:
                            print('UNEXPECTED case!!! Look into it!')
                            # mark as crossing just to make sure it is noticed
                            crossing = True
                        if crossing:
                            print(
                                f'ERROR <split>: edges '
                                f'«{F[fnT[u]]}–{F[fnT[v]]}» '
                                f'and «{F[fnT[s]]}–{F[fnT[t]]}» '
                                f'break a bunch apart at '
                                f'{F[fnT[touch_uv]]}, {F[fnT[touch_st]]}'
                            )
                            crossings.append(((u, v), (s, t)))
                    else:  # close_count > 2:
                        # segments (u, v) and (s, t) are almost parallel
                        # find the two nodes furthest apart
                        pairs = np.array(
                            ((u, v), (u, s), (u, t), (s, t), (v, t), (v, s))
                        )
                        furthest = np.argmax(
                            np.hypot(
                                *(AllnodesC[pairs[:, 0]] - AllnodesC[pairs[:, 1]]).T
                            )
                        )
                        # print('furthest =', furthest)
                        w, x = pairs[furthest]
                        q, r = pairs[furthest - 3]
                        if furthest % 3 == 0:
                            # (q, r) is contained within (w, x)
                            neighbors = list(G[q]) + list(G[r])
                            neighbors.remove(q)
                            neighbors.remove(r)
                            check_neighbors(neighbors, w, x, (q, r))
                        else:
                            # (u, v) partially overlaps (s, t)
                            neighbors_q = list(G[q])
                            neighbors_q.remove(w)
                            check_neighbors(neighbors_q, s, t, (q,))
                            # print(crossings)
                            neighbors_r = list(G[r])
                            neighbors_r.remove(x)
                            check_neighbors(neighbors_r, u, v, (r,))
                            # print(crossings)
                            if neighbors_q and neighbors_r:
                                for a, b in product(neighbors_q, neighbors_r):
                                    if is_same_side(*AllnodesC[[q, r, a, b]]):
                                        print(
                                            f'ERROR <partial overlap>: edge '
                                            f'«{F[fnT[u]]}–{F[fnT[v]]}» '
                                            f'crosses '
                                            f'«{F[fnT[s]]}–{F[fnT[t]]}»'
                                        )
                                        crossings.append(((u, v), (s, t)))
    debug and potential and print(
        'potential crossings: '
        + ', '.join(
            [
                f'«{F[fnT[u]]}–{F[fnT[v]]}» × «{F[fnT[s]]}–{F[fnT[t]]}»'
                for u, v, s, t in potential
            ]
        )
    )
    return crossings


def rotation_checkers_factory(
    VertexC: CoordPairs,
) -> tuple[Callable[[int, int, int], bool], Callable[[int, int, int], bool]]:
    def cw(A: int, B: int, C: int) -> bool:
        """Check cw orientation.

        Returns:
          True: if A->B->C traverses the triangle ABC clockwise
          False: otherwise
        """
        Ax, Ay = VertexC[A]
        Bx, By = VertexC[B]
        Cx, Cy = VertexC[C]
        return (Bx - Ax) * (Cy - Ay) < (By - Ay) * (Cx - Ax)

    def ccw(A: int, B: int, C: int) -> bool:
        """Check ccw orientation.

        Returns:
          True: if A->B->C traverses the triangle ABC counter-clockwise
          False: otherwise
        """
        Ax, Ay = VertexC[B]
        Bx, By = VertexC[A]
        Cx, Cy = VertexC[C]
        return (Bx - Ax) * (Cy - Ay) < (By - Ay) * (Cx - Ax)

    return cw, ccw


def rotating_calipers(
    convex_hull: NDArray[np.int_],
) -> tuple[NDArray[np.int_], float, float, CoordPairs]:
    # inspired by:
    # jhultman/rotating-calipers:
    #   CUDA and Numba implementations of computational geometry algorithms.
    # (https://github.com/jhultman/rotating-calipers)
    """Find the shortest width of a polygon.

    Reference: Toussaint, Godfried T. "Solving geometric problems with the
    rotating calipers." Proc. IEEE Melecon. Vol. 83. 1983.

    Args:
      convex_hull: (H, 2) array of coordinates of the convex hull
        in counter-clockwise order

    Returns:
      best_calipers, best_caliper_angle, area_min, bbox
    """
    caliper_angles = np.array([0.5 * np.pi, 0, -0.5 * np.pi, np.pi], dtype=float)
    area_min = np.inf
    H = convex_hull.shape[0]
    left, bottom = convex_hull.argmin(axis=0)
    right, top = convex_hull.argmax(axis=0)

    calipers = np.array([left, top, right, bottom], dtype=np.int_)

    for _ in range(H):
        # Roll vertices counter-clockwise
        calipers_advanced = (calipers - 1) % H
        # Vectors from previous calipers to candidates
        vec = convex_hull[calipers_advanced] - convex_hull[calipers]
        # Find angles of candidate edgelines
        angles = np.arctan2(vec[:, 1], vec[:, 0])
        # Find candidate angle deltas
        angle_deltas = caliper_angles - angles
        # Select pivot with smallest rotation
        pivot = np.abs(angle_deltas).argmin()
        # Advance selected pivot caliper
        calipers[pivot] = calipers_advanced[pivot]
        # Rotate all supporting lines by angle delta
        caliper_angles -= angle_deltas[pivot]

        # calculate area for current calipers
        angle = caliper_angles[np.abs(caliper_angles).argmin()]
        c, s = np.cos(angle), np.sin(angle)
        calipers_rot = convex_hull[calipers] @ np.array(((c, -s), (s, c)))
        bbox_rot_min = calipers_rot.min(axis=0)
        bbox_rot_max = calipers_rot.max(axis=0)
        area = (bbox_rot_max - bbox_rot_min).prod()
        # check if area is a new minimum
        if area < area_min:
            area_min = area
            best_calipers = calipers.copy()
            best_caliper_angle = angle
            best_bbox_rot_min = bbox_rot_min
            best_bbox_rot_max = bbox_rot_max

    c, s = np.cos(-best_caliper_angle), np.sin(-best_caliper_angle)
    t = best_bbox_rot_max
    b = best_bbox_rot_min
    # calculate bbox coordinates in original reference frame, ccw vertices
    bbox = np.array(
        ((b[0], b[1]), (b[0], t[1]), (t[0], t[1]), (t[0], b[1])), dtype=float
    ) @ np.array(((c, -s), (s, c)))

    return best_calipers, best_caliper_angle, area_min, bbox


def area_from_polygon_vertices(X: np.ndarray, Y: np.ndarray) -> float:
    """Calculate the area enclosed by the polygon with the vertices (x, y).

    Vertices must be in sequence around the perimeter (either clockwise or
    counter-clockwise).

    Args:
      X: array of X coordinates
      Y: array of Y coordinates

    Returns:
      area
    """
    # Shoelace formula for area (https://stackoverflow.com/a/30408825/287217).
    return 0.5 * abs(
        X[-1] * Y[0] - Y[-1] * X[0] + np.dot(X[:-1], Y[1:]) - np.dot(Y[:-1], X[1:])
    )
