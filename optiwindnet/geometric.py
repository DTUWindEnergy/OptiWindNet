# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import math
import operator
from collections import defaultdict
from collections.abc import Callable, Sequence
from itertools import combinations, pairwise
from math import isclose
from typing import Any, Literal

import networkx as nx
import numba as nb
import numpy as np
import shapely as shp
from numpy.typing import NDArray
from scipy.sparse import coo_array
from scipy.sparse.csgraph import minimum_spanning_tree as scipy_mst
from scipy.spatial.distance import cdist

__all__ = (
    'CoordPair', 'CoordPairs', 'IndexPairs',
    'angle', 'angle_helpers', 'angle_numpy', 'angle_oracles_factory',
    'any_pairs_opposite_edge', 'area_from_polygon_vertices', 'complete_graph',
    'find_edges_bbox_overlaps', 'find_segments_crossing_any', 'get_crossings_map',
    'is_bunch_split_by_corner', 'is_crossing', 'is_crossing_no_bbox',
    'is_crossing_numpy', 'is_same_side',
    'is_triangle_pair_a_convex_quadrilateral', 'minimum_spanning_forest',
    'perimeter', 'point_d2line', 'point_to_segment_distance',
    'polyline_rays_at_point', 'polylines_cross_at_point', 'rays_alternate',
    'rotate', 'rotating_calipers', 'rotation_checkers_factory',
    'triangle_AR', 'unique_rays',
)  # fmt: skip

NULL = np.iinfo(int).min

CoordPair = np.ndarray[tuple[Literal[2]], np.dtype[np.float64]]
"""Coordinates of a single point, as a float array of shape ``(2,)``."""

CoordPairs = np.ndarray[tuple[int, Literal[2]], np.dtype[np.float64]]
"""Coordinates of multiple points, as a float array of shape ``(N, 2)``."""

IndexPairs = np.ndarray[tuple[int, Literal[2]], np.dtype[np.int_]]
"""Vertex index pairs, such as edges, as an int array of shape ``(N, 2)``."""

Indices = np.ndarray[tuple[int], np.dtype[np.int_]]


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
    """Calculate the distance from point ``pC`` to the ``uC``-``vC`` line."""
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
    """Check if points ``sC`` and ``tC`` are on the same side of the line defined
    by points ``uC`` and ``vC``.

    Note: often used to check crossings with feeder links, where the feeder link
    ``sC``-``tC`` is already known to be on a line that crosses the edge ``uC``–``vC``
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
      ``True`` if any two of ``nodesC`` are on opposite sides of the edge.
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
    """Rotate ``coords`` (numpy array T×2) by ``angle`` (degrees)."""
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
    dot_prod = A @ B.T
    cross_prod = A[..., 0] * B[..., 1] - A[..., 1] * B[..., 0]
    return np.arctan2(cross_prod, dot_prod)


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


def angle_helpers(
    L: nx.Graph, include_borders: bool = True
) -> tuple[NDArray[np.float64], NDArray[np.int_], list[dict[int, int]]]:
    """Create auxiliary arrays of node attributes based on polar coordinates.

    The ranks of the angles and calculated per root and start from 0. The duplicates
    mapping is a list of dicts and is indexed first by the root.

    Args:
        L: location (also works with A or G)

    Returns:
        Tuple of (``angle__``, ``angle_rank__``, ``dups_from_root_rank__``)
    """

    T, R, VertexC = (L.graph[k] for k in ('T', 'R', 'VertexC'))
    B = L.graph.get('B', 0)
    NodeC = VertexC[: (T + B) if include_borders else T]
    Vec = NodeC[:, None, :] - VertexC[-R:]
    angle__ = np.arctan2(Vec[..., 1], Vec[..., 0])
    angle_rank__ = np.empty_like(angle__, dtype=np.int_)
    dups_from_root_rank__ = [{} for _ in range(-R, 0)]
    for r in range(-R, 0):
        _, angle_rank__[:, r], counts = np.unique(
            angle__[:, r], return_inverse=True, return_counts=True
        )
        for i in np.flatnonzero(counts > 1):
            dups_from_root_rank__[r][i.item()] = np.flatnonzero(
                angle_rank__[:, r] == i
            ).tolist()
    return angle__, angle_rank__, dups_from_root_rank__


def angle_oracles_factory(
    angle__: NDArray[np.float64], angle_rank__: NDArray[np.int_]
) -> tuple[
    Callable[[int, int, int, int, int, int, int], tuple[int, int]],
    Callable[[int, int, int], float],
]:
    """Make functions to answer queries about relative angles.

    Inputs are the outputs of :func:`angle_helpers`.

    Args:
      ``angle__``: (T, R)-array of angles wrt root (+-pi)
      ``angle_rank__``: (T, R)-array of the relative placement of angles

    Returns:
      ``union_limits()`` and ``angle_ccw()``
    """

    def is_within(pR: int, lR: int, hR: int) -> bool:
        W = lR > hR  # wraps, i.e. angle(low) > angle(high)
        L = lR <= pR  # angle(low) <= angle(probe)
        H = pR <= hR  # angle(probe) <= angle(high)
        return (not W and L and H) or (W and not L and H) or (W and L and not H)

    def union_limits(
        root: int, u: int, LO: int, HI: int, v: int, lo: int, hi: int
    ) -> tuple[int, int]:
        LOR, HIR, loR, hiR = angle_rank__[(LO, HI, lo, hi), root]
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
        aR, bR = angle_rank__[(a, b), pivot]
        aA, bA = angle__[(a, b), pivot]
        a_to_bA = (bA - aA).item()
        return a_to_bA if aR <= bR else (2 * math.pi + a_to_bA)

    return union_limits, angle_ccw


def find_edges_bbox_overlaps(
    VertexC: CoordPairs, u: int, v: int, edges: IndexPairs
) -> NDArray[np.int_]:
    """Find which ``edges`` have a bounding box overlap with ⟨u, v⟩.

    This is a preliminary filter for crossing checks. Enables avoiding the more
    costly geometric crossing calculations for segments that are clearly
    disjoint.

    Args:
      VertexC: (N×2) point coordinates
      u, v: indices of probed edge
      edges: list of index pairs representing edges to check against

    Returns:
      numpy array with the indices of overlaps in ``edges``
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

    Parallel or collinear segments (including superposition) are not considered
    crossings. Touching segments (including common endpoints) are considered
    crossings.

    Returns:
      ``True`` in case of crossing.
    """

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
    f = B[0] * A[1] - B[1] * A[0]
    if f == 0:
        # segments are parallel
        return False

    # alpha and beta numerators
    for num in (P[0] * Q[1] - P[1] * Q[0] for P, Q in ((C, B), (A, C))):
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
    """Checks if (``uC``, ``vC``) crosses (``sC``, ``tC``).

    Does not check for bounding-box overlap. Use :func:`find_edges_bbox_overlaps`
    first to filter out edges with disjoint bounding boxes (cheaper than the
    calculations here).

    Parallel or collinear segments (including superposition) are not considered
    crossings. Touching segments (including common endpoints) are considered
    crossings.

    Returns:
      ``True`` in case of crossing.
    """

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


# Shewchuk's first-stage error bound for orient2d (float64, epsilon = 2**-53):
# a determinant larger in magnitude than this times the sum of the magnitudes
# of its two products has the sign of the exact determinant.
_ORIENT2D_ERRBOUND = (3.0 + 16.0 * 2.0**-53) * 2.0**-53
# A certified crossing is left to GEOS if an endpoint is within this many units
# in the last place (of the largest coordinate) from the other segment's line,
# or if the sine of the angle between the segments is below the second bound.
_CROSSING_MARGIN_ULPS = 64.0
_CROSSING_MIN_SINE = 1e-6


@nb.njit(cache=True)
def _orient2d_filtered(
    ax: float, ay: float, bx: float, by: float, cx: float, cy: float
) -> tuple[float, bool]:
    """Orientation determinant of ⟨a, b, c⟩ and whether its sign is certain.

    Implements the floating-point filter of Shewchuk's adaptive ``orient2d``
    (Robust Adaptive Floating-Point Geometric Predicates, 1996, public domain).
    Coincident points yield an exact 0.

    Returns:
      The float determinant (twice the signed area) and whether its sign is
      the sign of the exact determinant.
    """
    if (ax == cx and ay == cy) or (bx == cx and by == cy) or (ax == bx and ay == by):
        return 0.0, True
    detleft = (ax - cx) * (by - cy)
    detright = (ay - cy) * (bx - cx)
    det = detleft - detright
    # the sign of a rounded difference or product is exact, so the determinant's
    # sign is certain when the two products do not have the same sign
    if detleft > 0.0:
        if detright <= 0.0:
            return det, True
        detsum = detleft + detright
    elif detleft < 0.0:
        if detright >= 0.0:
            return det, True
        detsum = -detleft - detright
    else:
        return det, True
    return det, abs(det) >= _ORIENT2D_ERRBOUND * detsum


@nb.njit(cache=True)
def _is_crossing_clear(
    ux: float,
    uy: float,
    vx: float,
    vy: float,
    sx: float,
    sy: float,
    tx: float,
    ty: float,
    d1: float,
    d2: float,
    d3: float,
    d4: float,
) -> bool:
    """Whether a certified proper crossing of ⟨u, v⟩ and ⟨s, t⟩ is one for GEOS.

    GEOS reports a crossing as a touch if the intersection point it computes
    equals an endpoint. That cannot happen if every endpoint is far from the
    other segment's line and the segments are not nearly parallel.

    Args:
      ux, uy, vx, vy, sx, sy, tx, ty: coordinates of the segments' ends.
      d1, d2: orientation determinants of ``s`` and ``t`` wrt ⟨u, v⟩.
      d3, d4: orientation determinants of ``u`` and ``v`` wrt ⟨s, t⟩.
    """
    uv_len = math.hypot(vx - ux, vy - uy)
    st_len = math.hypot(tx - sx, ty - sy)
    scale = max(abs(ux), abs(uy), abs(vx), abs(vy), abs(sx), abs(sy), abs(tx), abs(ty))
    margin = _CROSSING_MARGIN_ULPS * 2.0**-52 * scale
    dmin_uv = min(abs(d1), abs(d2)) / uv_len
    dmin_st = min(abs(d3), abs(d4)) / st_len
    sine = abs((vx - ux) * (ty - sy) - (vy - uy) * (tx - sx)) / (uv_len * st_len)
    return dmin_uv > margin and dmin_st > margin and sine > _CROSSING_MIN_SINE


@nb.njit(cache=True)
def _find_segments_crossing_any_filtered(
    probesC: np.ndarray, segmentsC: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Decide :func:`find_segments_crossing_any` where float arithmetic can.

    A pair is decided here if its orientation signs are certain and either
    show no proper crossing, or show one whose intersection point cannot
    round onto an endpoint (every endpoint far from the other segment's line,
    segments not nearly parallel). Otherwise it is left undecided.

    Returns:
      Two boolean arrays of shape ``(N,)``: probes crossing some segment, and
      probes not found crossing any segment but with some undecided pair.
    """
    N = probesC.shape[0]
    M = segmentsC.shape[0]
    crossed = np.zeros(N, dtype=np.bool_)
    undecided = np.zeros(N, dtype=np.bool_)
    lo = np.empty((M, 2))
    hi = np.empty((M, 2))
    for j in range(M):
        for k in range(2):
            lo[j, k] = min(segmentsC[j, 0, k], segmentsC[j, 1, k])
            hi[j, k] = max(segmentsC[j, 0, k], segmentsC[j, 1, k])
    for i in range(N):
        ux, uy = probesC[i, 0, 0], probesC[i, 0, 1]
        vx, vy = probesC[i, 1, 0], probesC[i, 1, 1]
        ulo0, uhi0 = min(ux, vx), max(ux, vx)
        ulo1, uhi1 = min(uy, vy), max(uy, vy)
        for j in range(M):
            if uhi0 < lo[j, 0] or hi[j, 0] < ulo0 or uhi1 < lo[j, 1] or hi[j, 1] < ulo1:
                continue
            sx, sy = segmentsC[j, 0, 0], segmentsC[j, 0, 1]
            tx, ty = segmentsC[j, 1, 0], segmentsC[j, 1, 1]
            # s and t on the same side of (or touching) the line through u, v
            d1, c1 = _orient2d_filtered(ux, uy, vx, vy, sx, sy)
            d2, c2 = _orient2d_filtered(ux, uy, vx, vy, tx, ty)
            if c1 and c2 and not ((d1 > 0.0 and d2 < 0.0) or (d1 < 0.0 and d2 > 0.0)):
                continue
            # u and v on the same side of (or touching) the line through s, t
            d3, c3 = _orient2d_filtered(sx, sy, tx, ty, ux, uy)
            d4, c4 = _orient2d_filtered(sx, sy, tx, ty, vx, vy)
            if c3 and c4 and not ((d3 > 0.0 and d4 < 0.0) or (d3 < 0.0 and d4 > 0.0)):
                continue
            if (
                c1
                and c2
                and c3
                and c4
                and _is_crossing_clear(ux, uy, vx, vy, sx, sy, tx, ty, d1, d2, d3, d4)
            ):
                crossed[i] = True
                break
            undecided[i] = True
        if crossed[i]:
            undecided[i] = False
    return crossed, undecided


def find_segments_crossing_any(
    probesC: np.ndarray, segmentsC: np.ndarray
) -> np.ndarray:
    """Flag which probe segments cross at least one of ``segmentsC``.

    The result is shapely's (GEOS) ``crosses`` predicate for two-point line
    strings: the segments meet in a single point interior to both. Touching
    (including common endpoints or an endpoint lying on the other segment) and
    parallel or collinear segments (including superposition) are not
    crossings. As in GEOS, a crossing whose intersection point rounds onto an
    endpoint is a touch, which keeps the answer consistent with the
    intersection point floating-point coordinates can represent.

    Pairs are decided in float arithmetic with certified orientation signs;
    the rare ones near that rounding limit are passed to shapely.

    Args:
      probesC: (N×2×2) coordinates of the probe segments' ends.
      segmentsC: (M×2×2) coordinates of the segments to check against.

    Returns:
      Boolean array of shape ``(N,)``.
    """
    crossed, undecided = _find_segments_crossing_any_filtered(probesC, segmentsC)
    if undecided.any():
        segmentsS = shp.linestrings(segmentsC)
        for i in np.flatnonzero(undecided).tolist():
            crossed[i] = shp.crosses(shp.linestrings(probesC[i]), segmentsS).any()
    return crossed


def is_crossing(
    uC: CoordPair,
    vC: CoordPair,
    sC: CoordPair,
    tC: CoordPair,
    touch_is_cross: bool = True,
) -> bool:
    """Checks if (uC, vC) crosses (sC, tC).

    Parallel or collinear segments (including superposition) are not considered
    crossings. For non-parallel segments, behavior for touching points
    (including common endpoints) is determined by ``touch_is_cross``.

    Args:
      uC, vC, sC, tC: (2,) numpy array coordinates of edge ends
      touch_is_cross: whether to consider any common point as a crossing

    Returns:
      ``True`` in case of crossing.
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
      ``True`` if points in ``bunch`` are both inside and outside cone ``a-o-b``
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


def point_to_segment_distance(pC: np.ndarray, aC: np.ndarray, bC: np.ndarray) -> float:
    """Calculate the distance from point ``pC`` to the closed segment ``aC``-``bC``.

    The projection of ``pC`` onto the line through ``aC`` and ``bC`` is clamped to
    the segment, so the result is the distance to the nearest endpoint when
    the foot of the perpendicular lies outside the segment. For an unclamped
    line distance, see :func:`point_d2line`.

    Args:
      pC: query point coordinates as a (2,) numpy array
      aC, bC: segment endpoint coordinates as (2,) numpy arrays

    Returns:
      Euclidean distance from ``pC`` to the closest point on the segment.
    """
    ab = bC - aC
    denom = np.dot(ab, ab)
    if denom == 0.0:
        return math.dist(pC, aC)
    t = np.clip(np.dot(pC - aC, ab) / denom, 0.0, 1.0)
    closest = aC + t * ab
    return math.dist(pC, closest)


def unique_rays(rays: list[np.ndarray], angle_tol: float) -> list[np.ndarray]:
    """Deduplicate 2D rays by direction; anti-parallel rays are kept distinct.

    Two rays are considered duplicates when their unit vectors are parallel
    *and* point the same way: cross-product magnitude ≤ ``angle_tol`` and dot
    product ≥ 0. Rays with zero norm are dropped.

    Args:
      rays: rays from a shared origin, each a (2,) numpy array
      angle_tol: maximum unit cross-product magnitude treated as parallel

    Returns:
      List of unit-length (2,) numpy arrays, one per distinct direction.
    """
    unique: list[np.ndarray] = []
    for ray in rays:
        norm = math.hypot(*ray)
        if norm == 0.0:
            continue
        unit = ray / norm
        if all(
            abs(unit[0] * other[1] - unit[1] * other[0]) > angle_tol
            or np.dot(unit, other) < 0
            for other in unique
        ):
            unique.append(unit)
    return unique


def polyline_rays_at_point(
    coords: np.ndarray,
    pC: np.ndarray,
    *,
    tol: float,
    angle_tol: float,
) -> list[np.ndarray]:
    """Get the local rays of a polyline at a point lying on (or near) it.

    For each segment whose perpendicular distance to ``pC`` is at most ``tol``,
    rays from ``pC`` toward each segment endpoint that sits farther than ``tol``
    from ``pC`` are collected, then deduplicated by direction via
    :func:`unique_rays`. The result is empty if ``pC`` is far from every
    segment, has at most one ray when ``pC`` is at an endpoint of an extreme
    segment, or two anti-parallel rays when ``pC`` is interior to a segment.

    Args:
      coords: polyline vertices as an (N, 2) numpy array
      pC: query point coordinates as a (2,) numpy array
      tol: distance threshold for treating ``pC`` as on a segment or at an endpoint
      angle_tol: passed through to :func:`unique_rays` for direction dedup

    Returns:
      List of unit-length (2,) numpy arrays, the directions from ``pC`` along
      the polyline at ``pC``.
    """
    rays = []
    for aC, bC in pairwise(coords):
        if point_to_segment_distance(pC, aC, bC) > tol:
            continue
        if math.dist(aC, pC) > tol:
            rays.append(aC - pC)
        if math.dist(bC, pC) > tol:
            rays.append(bC - pC)
    return unique_rays(rays, angle_tol)


def rays_alternate(rays_a: list[np.ndarray], rays_b: list[np.ndarray]) -> bool:
    """Check whether two ray sets interleave cyclically around the origin.

    Picks every 2-ray pair from each set, orders the four rays by polar angle,
    and checks for the cyclic label pattern ABAB (or BABA). When found, set
    ``a`` and set ``b`` separate one another — the geometric definition of a
    crossing at the shared origin. Returns ``False`` when either set has fewer
    than 2 rays.

    Args:
      rays_a, rays_b: 2D rays from a shared origin, each a (2,) numpy array

    Returns:
      ``True`` if some 2-ray pairs from ``rays_a`` and ``rays_b`` alternate around the
      origin in cyclic order.
    """
    if len(rays_a) < 2 or len(rays_b) < 2:
        return False
    for pair_a in combinations(rays_a, 2):
        for pair_b in combinations(rays_b, 2):
            ordered = sorted(
                (math.atan2(ray[1], ray[0]) % (2 * math.pi), label)
                for label, pair in (('a', pair_a), ('b', pair_b))
                for ray in pair
            )
            labels = [label for _, label in ordered]
            if labels in (['a', 'b', 'a', 'b'], ['b', 'a', 'b', 'a']):
                return True
    return False


def polylines_cross_at_point(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    pC: np.ndarray,
    *,
    tol: float,
    angle_tol: float,
) -> bool:
    """Check whether two polylines cross each other at a given point.

    Returns ``False`` when ``pC`` is within ``tol`` of any vertex of either polyline
    — those cases are ambiguous and the caller is expected to classify them
    separately (e.g. via shared-node or endpoint filters). Otherwise extracts
    the local rays at ``pC`` for each polyline (via :func:`polyline_rays_at_point`)
    and tests whether they alternate (via :func:`rays_alternate`).

    Args:
      coords_a, coords_b: polyline vertices as (N, 2) and (M, 2) numpy arrays
      pC: query point coordinates as a (2,) numpy array, expected to lie on
        the geometric intersection of the two polylines
      tol: distance threshold for vertex-coincidence and segment-membership
      angle_tol: minimum unit cross-product magnitude used during ray dedup

    Returns:
      ``True`` when the polylines cross transversely at ``pC``.
    """
    if np.any(np.hypot(*(pC - coords_a).T) <= tol) or np.any(
        np.hypot(*(pC - coords_b).T) <= tol
    ):
        return False
    rays_a = polyline_rays_at_point(coords_a, pC, tol=tol, angle_tol=angle_tol)
    rays_b = polyline_rays_at_point(coords_b, pC, tol=tol, angle_tol=angle_tol)
    return rays_alternate(rays_a, rays_b)


# A quadrilateral whose angle at u or v has a sine below this is degenerate
# (a triangle): decimal coordinates of collinear points leave float residuals of
# at most ~1e-14 in the sine, while angles of actual layouts are above ~1e-7.
_CONVEX_QUAD_MIN_SINE_SQ = 1e-10**2


@nb.njit(cache=True)
def is_triangle_pair_a_convex_quadrilateral(
    uC: CoordPair, vC: CoordPair, sC: CoordPair, tC: CoordPair
) -> bool:
    """Check convexity of quadrilateral.

    ⟨u, v⟩ is the common side; ⟨s, t⟩ are the opposing vertices;
    only works if ⟨s, t⟩ crosses the line defined by ⟨u, v⟩

    A quadrilateral with a (nearly) straight angle at u or v is a triangle,
    i.e. u or v lies on ⟨s, t⟩. The test for it is relative to the side
    lengths, so it gives the same answer in any translation or scale of the
    coordinates.

    Returns:
      ``True`` if the quadrilateral is convex and is not a triangle
    """
    # this used to be called `is_quadrilateral_convex()`
    us = sC - uC
    ut = tC - uC
    vs = sC - vC
    vt = tC - vC
    # us × ut
    usut = _cross_prod_2d(us, ut)
    # vt × vs
    vtvs = _cross_prod_2d(vt, vs)
    if not ((usut > 0.0 and vtvs > 0.0) or (usut < 0.0 and vtvs < 0.0)):
        return False
    # the four vertices must not form a triangle
    us2 = us[0] * us[0] + us[1] * us[1]
    ut2 = ut[0] * ut[0] + ut[1] * ut[1]
    vs2 = vs[0] * vs[0] + vs[1] * vs[1]
    vt2 = vt[0] * vt[0] + vt[1] * vt[1]
    return (
        usut * usut > _CONVEX_QUAD_MIN_SINE_SQ * us2 * ut2
        and vtvs * vtvs > _CONVEX_QUAD_MIN_SINE_SQ * vs2 * vt2
    )


def is_triangle_pair_a_convex_quadrilateral_XY(
    XY: Sequence[Sequence[float]], u: int, v: int, s: int, t: int
) -> bool:
    """Check convexity of quadrilateral given by vertex indices.

    Same test as :func:`is_triangle_pair_a_convex_quadrilateral`, but reads the
    coordinates from a list of pairs (e.g. ``VertexC.tolist()``). In a Python
    loop, this is ~10x faster than indexing a numpy array and calling the
    numba-compiled version.

    Args:
      XY: coordinate pairs indexed by vertex
      u: vertex of the common side ⟨u, v⟩
      v: vertex of the common side ⟨u, v⟩
      s: opposing vertex
      t: opposing vertex

    Returns:
      ``True`` if the quadrilateral is convex and is not a triangle
    """
    ux, uy = XY[u]
    vx, vy = XY[v]
    sx, sy = XY[s]
    tx, ty = XY[t]
    usx, usy = sx - ux, sy - uy
    utx, uty = tx - ux, ty - uy
    vsx, vsy = sx - vx, sy - vy
    vtx, vty = tx - vx, ty - vy
    # us × ut
    usut = usx * uty - usy * utx
    # vt × vs
    vtvs = vtx * vsy - vty * vsx
    if not ((usut > 0.0 and vtvs > 0.0) or (usut < 0.0 and vtvs < 0.0)):
        return False
    # the four vertices must not form a triangle
    us2 = usx * usx + usy * usy
    ut2 = utx * utx + uty * uty
    vs2 = vsx * vsx + vsy * vsy
    vt2 = vtx * vtx + vty * vty
    return (
        usut * usut > _CONVEX_QUAD_MIN_SINE_SQ * us2 * ut2
        and vtvs * vtvs > _CONVEX_QUAD_MIN_SINE_SQ * vs2 * vt2
    )


def perimeter(VertexC, vertices_ordered):
    """Calculate the perimeter of the polygon defined by ``vertices_ordered``.

    Args:
      vertices_ordered: indices of ``VertexC`` in clockwise or counter-clockwise
        orientation.

    Return:
      The perimeter length.
    """
    vec = VertexC[vertices_ordered[:-1]] - VertexC[vertices_ordered[1:]]
    return np.hypot(*vec.T).sum() + math.dist(
        VertexC[vertices_ordered[-1]], VertexC[vertices_ordered[0]]
    )


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
    """Create a complete graph based on ``G_base``.

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
        for i in range(R - 1):
            for j in range(R - i - 1):
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
    u_, v_ = Edge.T
    lengths = C[u_, v_].tolist()
    # assign each edge to the root closest to the edge's middle point
    roots = (-R + cdist((VertexC[u_] + VertexC[v_]) / 2, RootC).argmin(axis=1)).tolist()
    for u, v, length, root in zip(u_.tolist(), v_.tolist(), lengths, roots):
        edgeD = G[u][v]
        edgeD['length'] = length
        edgeD['root'] = root
    return G


def minimum_spanning_forest(A: nx.Graph) -> nx.Graph:
    """Create the minimum spanning forest from the Delaunay edges of ``A``.

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
        creator='minimum_spanning_forest',
    )
    for u, v in zip(U, V):
        u, v = int(u), int(v)
        S.add_edge(u, v, length=Q_[u, v].item())
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
            # remove the longest hop of the path connecting two roots
            u, v = max(pairwise(path), key=lambda uv: A[uv[0]][uv[1]]['length'])
            S.remove_edge(u, v)
            removals -= 1
    return S


def rotation_checkers_factory(
    VertexC: CoordPairs | Sequence[Sequence[float]],
) -> tuple[
    Callable[[int, int, int], bool],
    Callable[[int, int, int], bool],
    Callable[[int, int, int], float],
]:
    def cross(A: int, B: int, C: int) -> float:
        """Signed twice-area of triangle ABC.

        > 0: A→B→C is counter-clockwise
        < 0: A→B→C is clockwise
        == 0: A, B, C are collinear (use a tolerance for float input)
        """
        Ax, Ay = VertexC[A]
        Bx, By = VertexC[B]
        Cx, Cy = VertexC[C]
        return (Bx - Ax) * (Cy - Ay) - (By - Ay) * (Cx - Ax)

    def cw(A: int, B: int, C: int) -> bool:
        """``True`` if ``A→B→C`` traverses the triangle ``ABC`` clockwise."""
        return cross(A, B, C) < 0

    def ccw(A: int, B: int, C: int) -> bool:
        """``True`` if ``A→B→C`` traverses the triangle ``ABC`` counter-clockwise."""
        return cross(A, B, C) > 0

    return cw, ccw, cross


def rotating_calipers(
    convex_hull: NDArray,
    metric: str = 'height',
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
      metric: what should be minimized, one of {``'height'``, ``'area'``}

    Returns:
      ``best_calipers``, ``best_caliper_angle``, ``best_metric``, ``bbox``
    """
    best_metric = np.inf
    best: tuple[NDArray[np.int_], Any, NDArray, NDArray] | None = None
    H = convex_hull.shape[0]
    min_x, min_y = convex_hull.argmin(axis=0)
    max_x, max_y = convex_hull.argmax(axis=0)

    calipers = np.array([min_y, max_x, max_y, min_x], dtype=np.int_)
    caliper_angles = np.array([np.pi, -0.5 * np.pi, 0, 0.5 * np.pi], dtype=float)

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

        angle = caliper_angles[np.abs(caliper_angles).argmin()]
        c, s = np.cos(angle), np.sin(angle)
        calipers_rot = convex_hull[calipers] @ np.array(((c, -s), (s, c)))
        bbox_rot_min = calipers_rot.min(axis=0)
        bbox_rot_max = calipers_rot.max(axis=0)
        width, height = (bbox_rot_max - bbox_rot_min).tolist()
        angle_offset = 0
        if metric == 'height':
            if width < height:
                metric_value = width
            else:
                metric_value = height
                angle_offset = 0.5 * math.pi
        elif metric == 'area':
            metric_value = width * height
        else:
            raise ValueError(f'Unknown metric: {metric}')
        # check if area is a new minimum
        if metric_value < best_metric:
            best_metric = metric_value
            best = calipers.copy(), angle + angle_offset, bbox_rot_min, bbox_rot_max

    if best is None:
        raise ValueError('`convex_hull` must have at least one vertex.')
    best_calipers, best_caliper_angle, best_bbox_rot_min, best_bbox_rot_max = best

    c, s = np.cos(-best_caliper_angle), np.sin(-best_caliper_angle)
    t = best_bbox_rot_max
    b = best_bbox_rot_min
    # calculate bbox coordinates in original reference frame, ccw vertices
    bbox = np.array(
        ((b[0], b[1]), (b[0], t[1]), (t[0], t[1]), (t[0], b[1])), dtype=float
    ) @ np.array(((c, -s), (s, c)))

    return best_calipers, best_caliper_angle.item(), best_metric, bbox


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
