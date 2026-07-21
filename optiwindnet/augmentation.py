# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import math
import warnings
from itertools import pairwise
from typing import Callable

import matplotlib.pyplot as plt
import networkx as nx
import numba as nb
import numpy as np
from numba.typed import List
from scipy.spatial import ConvexHull

from .geometric import (
    CoordPair,
    CoordPairs,
    IndexPairs,
    area_from_polygon_vertices,
    rotate,
    rotating_calipers,
)
from .interarraylib import L_from_site

_lggr = logging.getLogger(__name__)
info, warn = _lggr.info, _lggr.warning

__all__ = ('get_shape_to_fill', 'poisson_disc_filler', 'turbinate')


def _get_border_scale_offset(
    BorderC: CoordPairs,
) -> tuple[CoordPair, float, float, float]:
    offsetC = BorderC.min(axis=0)
    width, height = BorderC.max(axis=0) - offsetC
    # Take the sqrt() of the area and invert for the linear factor such that
    # area=1.
    norm_scale = 1.0 / math.sqrt(area_from_polygon_vertices(*(BorderC - offsetC).T))
    return offsetC, norm_scale, width, height


@nb.njit(cache=True, inline='always')
def _clears(RepellerC: CoordPairs, repel_radius_sq: float, point: CoordPair) -> bool:
    """Check if there is a minimum distance between a point and all repellers.

    The point must be at least ``sqrt(repel_radius_sq)`` apart from repellers.

    Args:
      RepellerC: coordinates (R, 2) of repellers
      repel_radius_sq: the square of the minimum radius required
      point: coordinate (2,) of point to test

    Returns:
      ``True`` if ``point`` clears all discs centered on ``RepellerC``.
    """
    return (
        ((point[np.newaxis, :] - RepellerC) ** 2).sum(axis=1) >= repel_radius_sq
    ).all()


@nb.njit(cache=True, inline='always')
def _fully_covered_neighbor(x: float, y: float) -> tuple[int, int]:
    """Identify a unit-cell edge-neighbor made unreachable by a dart at (x, y).

    A dart thrown at fractional position ``(x, y)`` within its unit cell
    (cell side = 1, min_dist = sqrt(2), matching the scaled coordinates used
    by :func:`_poisson_disc_filler_core`) excludes a disc of radius
    ``sqrt(2)`` around itself. When the dart lands close enough to one side
    of the cell, that whole disc covers the entirety of the single
    edge-adjacent neighbor cell across that side, meaning no future dart in
    that neighbor cell could ever be valid.

    The nearer of the two axis-wise distances to an edge (``ex``, ``ey``)
    identifies the candidate side; the farther one parametrizes the critical
    threshold below which the disc reaches the neighbor's farthest corner.

    Args:
      x: fractional x position within the unit cell, in [0, 1).
      y: fractional y position within the unit cell, in [0, 1).

    Returns:
      ``(di, dj)`` offset of the fully-covered neighbor cell, or ``(0, 0)``
      if none of the four edge-adjacent neighbors is fully covered.
    """
    ex = x if x < 0.5 else 1.0 - x
    ey = y if y < 0.5 else 1.0 - y
    if ex < ey:
        h_ref = math.sqrt(1.0 + 2.0 * ey - ey * ey) - 1.0
        if ex < h_ref:
            return (1, 0) if x > 0.5 else (-1, 0)
    else:
        h_ref = math.sqrt(1.0 + 2.0 * ex - ex * ex) - 1.0
        if ey < h_ref:
            return (0, 1) if y > 0.5 else (0, -1)
    return (0, 0)


@nb.njit(cache=True, inline='always')
def _walk_along_perimeter(
    polygonC: CoordPairs, max_loops: int
) -> tuple[list[tuple[int, int]], list[tuple[float, float]]]:
    """Find crossings of the polygons' perimeter with the unit grid.

    Auxiliar function to :func:`poisson_disc_filler`. Used for identifying unit cells
    that cover the polygon's shell. The polygon's vertices are also included in the
    lists.

    Args:
      polygonC: N×2 coordinates that define the polygon (unique vertices)
      max_loops: a failsafe maximum for the inner loop (e.g. i_len + j_len + 1)

    Returns:
      list of cell index pairs and list of coordinate pairs
    """
    # list cells along the boundary (essential when cells are few)
    points = []  # only used for the debugging plot
    cells = []
    xy = polygonC[-1]
    ij = np.floor(xy)
    vec = np.empty((2,))
    t = np.empty((2,))
    for k, xy_fwd in enumerate(polygonC):
        ij_fwd = np.floor(xy_fwd)
        point = xy[0], xy[1]
        vec[:] = xy_fwd - xy
        is_vec_gt0 = vec > 0
        vec_sign = np.sign(vec)
        frac = xy - ij
        for _ in range(max_loops):
            points.append(point)
            cells.append((int(ij[0]), int(ij[1])))
            if (ij == ij_fwd).all():
                break
            t[:] = (
                ((is_vec_gt0[0] - frac[0]) / vec[0]) if vec[0] != 0 else np.inf,
                ((is_vec_gt0[1] - frac[1]) / vec[1]) if vec[1] != 0 else np.inf,
            )
            y_not_x = np.argmin(t)
            if y_not_x:
                frac[:] = frac[0] + t[1] * vec[0], 1.0 * ~is_vec_gt0[1]
                point = ij[0] + frac[0], ij[1] + is_vec_gt0[1]
                ij[1] += vec_sign[1]
            else:
                frac[:] = 1.0 * ~is_vec_gt0[0], frac[1] + t[0] * vec[1]
                point = ij[0] + is_vec_gt0[0], ij[1] + frac[1]
                ij[0] += vec_sign[0]
        xy, ij = xy_fwd, ij_fwd
    return cells, points


def _contains_np(
    polyC: CoordPairs, pts: CoordPairs
) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
    """Evaluate if ``polygon`` (N, 2) covers points in ``pts`` (M, 2).

    Args:
      polyC: coordinates of polygon vertices (N, 2).
      pts: coordinates of points to test (M, 2).

    Returns:
      boolean array shaped (M,) (``True`` if ``pts[i]`` inside ``polygon``).
    """
    polyC_rolled = np.roll(polyC, -1, axis=0)
    vectors = polyC_rolled - polyC
    mask1 = (pts[:, None] == polyC).all(-1).any(-1)
    m1 = (polyC[:, 1] > pts[:, None, 1]) != (polyC_rolled[:, 1] > pts[:, None, 1])
    slope = vectors[:, 1] * (pts[:, None, 0] - polyC[:, 0]) - vectors[:, 0] * (
        pts[:, None, 1] - polyC[:, 1]
    )
    m2 = slope == 0
    mask2 = (m1 & m2).any(-1)
    m3 = (slope < 0) != (polyC_rolled[:, 1] < polyC[:, 1])
    m4 = m1 & m3
    count = np.count_nonzero(m4, axis=-1)
    mask3 = ~(count % 2 == 0)
    return mask1 | mask2 | mask3


@nb.njit(cache=True)
def _contains(polyC: CoordPairs, point: CoordPair) -> bool:
    """Evaluate if polygon (N, 2) covers ``point`` (2,).

    Args:
      polyC: coordinates of polygon vertices (N, 2).
      point: coordinates of point to test (2,).

    Returns:
      ``True`` if ``point`` inside polygon, ``False`` otherwise
    """
    intersections = 0
    dx2, dy2 = point - polyC[-1]

    for p in polyC:
        dx, dy = dx2, dy2
        dx2, dy2 = point - p

        F = (dx - dx2) * dy - dx * (dy - dy2)
        if np.isclose(F, 0.0, rtol=0.0) and dx * dx2 <= 0 and dy * dy2 <= 0:
            return True

        if (dy >= 0 and dy2 < 0) or (dy2 >= 0 and dy < 0):
            if F > 0:
                intersections += 1
            elif F < 0:
                intersections -= 1
    return intersections != 0


@nb.njit(cache=True)
def _poisson_disc_filler_core(
    T: int,
    max_iter: int,
    single_run: bool,
    i_len: int,
    j_len: int,
    cell_idc: IndexPairs,
    BorderS: CoordPairs,
    cell_strictly_inside_border: np.ndarray[tuple[int, int], np.dtype[np.bool_]],
    obstacleS__: list[CoordPairs],
    cell_clear_of_obstacles: np.ndarray[tuple[int, int], np.dtype[np.bool_]],
    cell_fully_clear: np.ndarray[tuple[int, int], np.dtype[np.bool_]],
    repel_radius_sq: float,
    RepellerS: CoordPairs | None,
    rng: np.random.Generator,
) -> CoordPairs:
    """This is the numba-compilable core called by :func:`poisson_disc_filler`."""
    # [Poisson-Disc Sampling](https://www.jasondavies.com/poisson-disc/)

    # mask for the 20 neighbors
    # (5x5 grid excluding corners and center)
    neighbormask = np.array(
        (
            (False, True, True, True, False),
            (True, True, True, True, True),
            (True, True, False, True, True),
            (True, True, True, True, True),
            (False, True, True, True, False),
        )
    )

    # points to be returned by this function
    points = np.empty((T, 2), dtype=np.float64)
    # grid for mapping of cell to position in array `points` (T means not set)
    cells = np.full((i_len, j_len), T, dtype=np.int64)

    def no_conflict(p: int, q: int, point: CoordPair) -> bool:
        """Check for conflict with points from the 20 cells neighboring the
        current cell.

        Args:
          p: x cell index.
          q: y cell index.
          point: numpy array shaped (2,) with the point's coordinates

        Returns:
          ``True`` if ``point`` does not conflict, ``False`` otherwise.
        """
        p_min, p_max = max(0, p - 2), min(i_len, p + 3)
        q_min, q_max = max(0, q - 2), min(j_len, q + 3)
        cells_window = cells[p_min:p_max, q_min:q_max].copy()
        mask = neighbormask[
            2 + p_min - p : 2 + p_max - p, 2 + q_min - q : 2 + q_max - q
        ] & (cells_window < T)
        ii = cells_window.reshape(mask.size)[np.flatnonzero(mask.flat)]
        return not (((point[None, :] - points[ii]) ** 2).sum(axis=-1) < 2).any()

    # `idc_arr[:avail_count]` holds indices into `cell_idc` for the cells
    # still available for dart-throwing. `pos_in_list[i, j]` maps a cell
    # back to its position within `idc_arr` (-1 if unavailable), so both the
    # just-occupied cell and any neighbor cell that becomes fully covered by
    # the new point's exclusion disc can be dropped from the pool in O(1)
    # via swap-with-last, instead of the O(n) `del idc_list[k]` this
    # replaces.
    idc_arr = np.arange(len(cell_idc), dtype=np.int64)
    pos_in_list = np.full((i_len, j_len), -1, dtype=np.int64)

    iters_per_attempt = []

    def reset_attempt(prev_attempt_iter: int) -> tuple[int, int, float, int]:
        """Reset all per-attempt state for a fresh attempt.

        Logs how many iterations the just-abandoned attempt took, unless
        ``prev_attempt_iter`` is negative (the very first attempt, with
        nothing to log yet). Returns the reset ``(out_count, avail_count,
        ema_hit_rate, attempt_iter)``.
        """
        if prev_attempt_iter >= 0:
            iters_per_attempt.append(prev_attempt_iter)
        cells[:, :] = T
        idc_arr[:] = np.arange(len(cell_idc), dtype=np.int64)
        for k in range(len(cell_idc)):
            ci, cj = cell_idc[k]
            pos_in_list[ci, cj] = k
        return 0, len(cell_idc), 1.0, 0

    def discard_cell(i: int, j: int, k: int, avail_count: int) -> int:
        """Remove cell (i, j), found at position k in idc_arr, from the pool."""
        last = avail_count - 1
        last_flat = idc_arr[last]
        idc_arr[k] = last_flat
        li, lj = cell_idc[last_flat]
        pos_in_list[li, lj] = k
        pos_in_list[i, j] = -1
        return last

    # The EMA tracks the local dart-throw success rate and adapts as the
    # field fills. It is combined with the number of available cells to
    # determine whether the current attempt can finish within the budget.
    # Bound the decay scale to avoid extreme values for small or large T.
    T_decay = min(max(T, 10), 240)
    hit_rate_decay = 6.0 / T_decay**2
    out_count, avail_count, ema_hit_rate, attempt_iter = reset_attempt(-1)

    # Once an attempt is unlikely to complete within the remaining budget,
    # continue it. ``single_run`` disables restarts.
    restarts_exhausted = single_run

    best_out_count = 0
    best_points = np.empty((0, 2), dtype=np.float64)
    restart_count = 0

    def save_best(out_count: int) -> None:
        """Snapshot the current attempt's points if it beats the best so far."""
        nonlocal best_out_count, best_points
        if out_count > best_out_count:
            best_out_count = out_count
            best_points = points[:out_count].copy()

    for remaining_iters in range(max_iter, 1, -1):
        attempt_iter += 1
        if not restarts_exhausted:
            remaining_points = T - out_count
            pace_limit = min(avail_count, ema_hit_rate * remaining_iters)
            if remaining_points > pace_limit:
                # A restart is useful only if enough budget remains to
                # plausibly complete a new attempt. The threshold is the
                # longest prior or current attempt, with T as a lower bound.
                restart_floor = T
                for a in iters_per_attempt:
                    if a > restart_floor:
                        restart_floor = a
                if attempt_iter > restart_floor:
                    restart_floor = attempt_iter
                if remaining_iters > restart_floor:
                    # Restart with an empty field.
                    save_best(out_count)
                    out_count, avail_count, ema_hit_rate, attempt_iter = reset_attempt(
                        attempt_iter
                    )
                    restart_count += 1
                    continue
                else:
                    # Budget too low for a restart; keep improving the current attempt.
                    restarts_exhausted = True

        # pick random empty cell
        empty_idx = rng.integers(low=0, high=avail_count)
        ij = cell_idc[idc_arr[empty_idx]]
        i, j = ij

        # dart throw inside cell
        fracC = rng.random(2)
        dartC = ij + fracC

        miss = False
        # check border and obstacles; a cell flagged in `cell_fully_clear`
        # is both entirely inside the border and clear of every obstacle,
        # so neither containment test is needed for a dart thrown in it.
        # Otherwise, `cell_strictly_inside_border` and `cell_clear_of_obstacles`
        # each skip just the one test that does not apply to this cell.
        if not cell_fully_clear[i, j]:
            if not cell_strictly_inside_border[i, j] and not _contains(BorderS, dartC):
                miss = True
            elif not cell_clear_of_obstacles[i, j]:
                for obstacleS_ in obstacleS__:
                    if _contains(obstacleS_, dartC):
                        miss = True
                        break

        # check overlap and repel_radius
        if not miss:
            if not no_conflict(i, j, dartC):
                miss = True
            elif RepellerS is not None and not _clears(
                RepellerS, repel_radius_sq, dartC
            ):
                miss = True

        ema_hit_rate *= 1 - hit_rate_decay
        if miss:
            continue
        ema_hit_rate += hit_rate_decay

        # add new point and remove its cell from the available pool
        points[out_count] = dartC
        cells[i, j] = out_count
        avail_count = discard_cell(i, j, empty_idx, avail_count)

        # a dart landing close enough to a cell's side fully covers the
        # single neighbor cell across that side with its exclusion
        # disc; such a neighbor can never host a valid point, so drop
        # it from the pool too (if it is still in it)
        di, dj = _fully_covered_neighbor(fracC[0], fracC[1])
        if di != 0 or dj != 0:
            ni, nj = i + di, j + dj
            if 0 <= ni < i_len and 0 <= nj < j_len:
                nk = pos_in_list[ni, nj]
                if nk >= 0:
                    avail_count = discard_cell(ni, nj, nk, avail_count)

        out_count += 1
        if out_count == T:
            break
        if avail_count == 0:
            # This layout does not have enough available cells. A fresh layout
            # may still complete the placement, so restart when allowed.
            if not restarts_exhausted:
                save_best(out_count)
                out_count, avail_count, ema_hit_rate, attempt_iter = reset_attempt(
                    attempt_iter
                )
                restart_count += 1
                continue
            break

    if out_count > best_out_count:
        best_out_count = out_count
        best_points = points[:out_count]

    return best_points, iters_per_attempt, restart_count


def get_shape_to_fill(L: nx.Graph) -> tuple[CoordPairs, CoordPairs]:
    """Calculate the area and scale the border so that it has area 1.

    The border and OSS are translated to the 1st quadrant, near the origin.

    IF SITE HAS MULTIPLE OSSs, ONLY 1 IS RETURNED (mean of the OSSs' coords).
    """
    R = L.graph['R']
    VertexC = L.graph['VertexC']
    BorderC = VertexC[L.graph['border']].copy()
    offsetC, norm_scale, _, _ = _get_border_scale_offset(BorderC)
    # deal with multiple roots
    if R > 1:
        RootC = ((VertexC[-R:].mean(axis=0) - offsetC) * norm_scale)[np.newaxis, :]
    else:
        RootC = (VertexC[-1:] - offsetC) * norm_scale
    BorderC -= offsetC
    BorderC *= norm_scale
    return BorderC, RootC


def poisson_disc_filler(
    T: int,
    min_dist: float,
    BorderC: CoordPairs,
    RepellerC: CoordPairs | None = None,
    repel_radius: float = 0.0,
    obstacleC__: list[CoordPairs] = [],
    seed: int | None = None,
    max_iter: int = 30000,
    plot: bool = False,
    partial_fulfilment: bool = True,
    single_run: bool = False,
) -> CoordPairs:
    """Randomly place points inside an area respecting a minimum separation.

    Fills the area delimited by ``BorderC`` with ``T`` randomly
    placed points that are at least ``min_dist`` apart and that
    don't fall inside any of the ``RepellerC`` discs or ``obstacles`` areas.

    ``max_iter`` is shared by all attempts. The sampler restarts when the
    current placement rate cannot meet the remaining target and sufficient
    budget remains for a new attempt.

    Args:
      T: number of points to place.
      min_dist: minimum distance between place points.
      BorderC: coordinates (B × 2) of border polygon.
      RepellerC: coordinates (R × 2) of the centers of forbidden discs.
      repel_radius: the radius of the forbidden discs.
      obstacleC__: sequence of coordinate arrays (X × 2).
      max_iter: total dart-throw iteration budget.
      partial_fulfilment: whether to return or reject a partial result.
      single_run: whether to disable adaptive restarts.

    Returns:
      coordinates (T, 2) of placed points
    """
    # quick check for outrageous densities
    # circle packing efficiency limit: η = π srqt(3)/6 = 0.9069
    # A Simple Proof of Thue's Theorem on Circle Packing
    # https://arxiv.org/abs/1009.4322
    area_avail = area_from_polygon_vertices(*BorderC.T)
    area_demand = T * np.pi * min_dist**2 / 4
    efficiency = area_demand / area_avail
    efficiency_optimal = math.pi * math.sqrt(3) / 6
    if efficiency > efficiency_optimal:
        msg = (
            f'(T = {T}, min_dist = {min_dist}) imply a packing '
            f'efficiency of {efficiency:.3f} which is higher than '
            f'the optimal possible ({efficiency_optimal:.3f}).'
        )
        if partial_fulfilment:
            info('Attempting partial fullfillment %s. Reduce T and/or min_dist.', msg)
        else:
            raise ValueError(msg)

    offsetC = BorderC.min(axis=0)
    width, height = BorderC.max(axis=0) - offsetC

    # create auxiliary grid covering the defined BorderC
    cell_size = min_dist / math.sqrt(2)
    i_len = math.ceil(width / cell_size)
    j_len = math.ceil(height / cell_size)
    BorderS = (BorderC - offsetC) / cell_size
    if RepellerC is None or repel_radius == 0.0:
        RepellerS = None
        repel_radius_sq = 0.0
    else:
        # check if Repellers are inside borders
        is_inside_rep = _contains_np(BorderC, RepellerC)
        if is_inside_rep.any():
            RepellerS = (RepellerC[is_inside_rep] - offsetC) / cell_size
            repel_radius_sq = (repel_radius / cell_size) ** 2
        else:
            RepellerS = None
            repel_radius_sq = 0.0
    obstacleS__ = List.empty_list(nb.float64[:, :])
    for obsC_ in obstacleC__:
        obstacleS__.append((obsC_ - offsetC) / cell_size)
        #  obstacleS__.append((obsC_[::-1] - offsetC) / cell_size)

    # Alternate implementation using np.mgrid
    #  pts = np.reshape(
    #      np.moveaxis(np.mgrid[0: i_len + 1, 0: j_len + 1], 0, -1),
    #      ((i_len + 1)*(j_len + 1), 2)
    #  )
    cornerS_ = np.empty(((i_len + 1) * (j_len + 1), 2), dtype=int)
    cornerS__ = cornerS_.reshape((i_len + 1, j_len + 1, 2))
    cornerS__[..., 0] = np.arange(i_len + 1)[:, np.newaxis]
    cornerS__[..., 1] = np.arange(j_len + 1)[np.newaxis, :]

    def _cell_corners_view(corner_flag_: np.ndarray) -> np.ndarray:
        corner_flag = corner_flag_.reshape((i_len + 1, j_len + 1), copy=False)
        return np.lib.stride_tricks.as_strided(
            corner_flag,
            shape=(2, 2, i_len, j_len),
            strides=corner_flag.strides * 2,
            writeable=False,
        )

    # process the area's border
    is_corner_within_border_ = _contains_np(BorderS, cornerS_)
    # corners satisfying the border alone (obstacles ignored), used below to
    # find cells lying entirely within the border so that the per-dart
    # border containment check can be skipped for darts thrown in them
    is_corner_within_border_only_ = is_corner_within_border_.copy()

    # a cell is clear of an obstacle if none of its corners lie inside it
    # and the obstacle's perimeter does not cross it; cells clear of every
    # obstacle can skip the per-dart obstacle containment loop entirely
    cell_clear_of_obstacles__ = np.ones((i_len, j_len), dtype=bool)
    for obstacleS_ in obstacleS__:
        is_corner_in_obstacle_ = _contains_np(obstacleS_, cornerS_)
        is_corner_within_border_ &= ~is_corner_in_obstacle_
        cell_clear_of_obstacles__ &= ~_cell_corners_view(is_corner_in_obstacle_).any(
            axis=(0, 1)
        )
        obstacle_cells, _ = _walk_along_perimeter(obstacleS_, i_len + j_len + 1)
        for oi, oj in obstacle_cells:
            cell_clear_of_obstacles__[oi, oj] = False

    is_corner_within_border = is_corner_within_border_.reshape(
        (i_len + 1, j_len + 1), copy=False
    )

    cell_corners = _cell_corners_view(is_corner_within_border_)
    cell_covers_polygon__ = cell_corners.any(axis=(0, 1))
    cell_strictly_inside_polygon__ = cell_corners.all(axis=(0, 1))

    cell_strictly_inside_border__ = _cell_corners_view(
        is_corner_within_border_only_
    ).all(axis=(0, 1))

    cells, points = _walk_along_perimeter(BorderS, i_len + j_len + 1)
    for i, j in cells:
        cell_covers_polygon__[i, j] = True
        cell_strictly_inside_polygon__[i, j] = False
        cell_strictly_inside_border__[i, j] = False

    cell_intercepts_polygon__ = np.logical_and(
        cell_covers_polygon__, ~cell_strictly_inside_polygon__
    )

    # cells needing neither the border nor any obstacle containment test
    cell_fully_clear__ = cell_strictly_inside_border__ & cell_clear_of_obstacles__

    if RepellerS is not None and repel_radius >= min_dist:
        # the cells that contain the repellers can be discarded
        for r_i, r_j in RepellerS.astype(int):
            cell_covers_polygon__[r_i, r_j] = False

    # useful plot for debugging purposes only
    if plot:
        fig, ax = plt.subplots(layout='constrained')
        ax.pcolormesh(cell_covers_polygon__.T + 2 * cell_intercepts_polygon__.T)
        ax.plot(*np.vstack((BorderS, BorderS[:1])).T, 'k', lw=1)
        for obstacleS_ in obstacleS__:
            ax.plot(*np.vstack((obstacleS_, obstacleS_[:1])).T, 'navy', lw=1)
        ax.plot(*np.vstack((BorderS, BorderS[:1])).T, 'k', lw=1)
        ax.scatter(*np.nonzero(is_corner_within_border), marker='+', s=15, c='k', lw=1)
        ax.scatter(*BorderS.T, marker='o', s=15, lw=1, c='navy')
        ax.scatter(*np.array(points).T, marker='x', s=12, lw=0.8, c='red')
        ax.axis('off')
        ax.set_aspect('equal')

    # Sequence of (i, j) of cells that overlap with the polygon
    cell_idc = np.argwhere(cell_covers_polygon__)

    rng = np.random.default_rng(seed)
    points, iters_per_attempt, restart_count = _poisson_disc_filler_core(
        T,
        max_iter,
        single_run,
        i_len,
        j_len,
        cell_idc,
        BorderS,
        cell_strictly_inside_border__,
        obstacleS__,
        cell_clear_of_obstacles__,
        cell_fully_clear__,
        repel_radius_sq,
        RepellerS,
        rng,
    )
    best_T = points.shape[0]

    # check if request was fulfilled
    if best_T < T:
        if partial_fulfilment:
            warn(
                'Only %d points generated (requested: %d, efficiency '
                'requested: %.3f, max_iter: %d, iters_per_attempt: %s, restarts: %d).',
                best_T,
                T,
                efficiency,
                max_iter,
                iters_per_attempt,
                restart_count,
            )
        else:
            raise ValueError(
                f'Only {best_T} points generated (requested: {T}), '
                f'using max_iter={max_iter} ({iters_per_attempt=} '
                f'consumed across {restart_count} restarts; requested '
                f'packing efficiency: {efficiency:.3f}). Increase '
                f'max_iter, reduce T or min_dist, or set '
                f'partial_fulfilment=True to accept a partial result '
                f'instead of raising.'
            )

    return points * cell_size + offsetC


def turbinate(
    L: nx.Graph,
    T: int,
    d: float,
    *,
    root_clearance: float | None = None,
    plot: bool = False,
    max_iter: int = 100_000,
    rounds: int | None = None,
    single_run: bool = False,
) -> nx.Graph:
    """Fills the location ``L`` with ``T`` turbines spaced at least ``d`` apart.

    Only the border and root locations from ``L`` are used.

    The placement of turbines is random and some combinations of ``T`` and ``d``
    will result in fewer placements than requested. Increase ``max_iter`` to
    apply more effort before aborting.

    Args:
      L: reference location (only borders, obstacles and substations are used)
      T: desired number of turbines to place
      d: minimum spacing between turbines
      root_clearance: minimum spacing from turbine to substation (if not given,
        ``d`` is used)
      max_iter: total placement iteration budget.
      rounds: deprecated multiplier for ``max_iter``.
      single_run: whether to disable adaptive restarts.

    Returns:
      A location with randomly placed turbines (the number may be lower than T)
    """
    if rounds is not None:
        warnings.warn(
            'turbinate(rounds=...) is deprecated, pass a higher max_iter '
            'instead. Increasing max_iter to max_iter*rounds.',
            DeprecationWarning,
            stacklevel=2,
        )
        max_iter *= rounds

    VertexC = L.graph['VertexC']
    border = L.graph['border']
    R = L.graph['R']
    B = L.graph['B']
    BorderC = VertexC[border]
    _, best_caliper_angle, _, _ = rotating_calipers(
        BorderC[ConvexHull(BorderC).vertices],
        metric='area',
    )
    # angle that minimizes the number of cells created by poisson_disc_filler
    rotation = best_caliper_angle * 180.0 / np.pi

    RootC = VertexC[-R:]
    obstacles = L.graph.get('obstacles', [])
    obstacleC__ = [VertexC[obs] for obs in obstacles]

    TerminalC = rotate(
        poisson_disc_filler(
            T,
            d,
            BorderC=rotate(BorderC, -rotation),
            RepellerC=rotate(RootC, -rotation),
            obstacleC__=[rotate(obsC_, -rotation) for obsC_ in obstacleC__],
            repel_radius=(d if root_clearance is None else root_clearance),
            max_iter=max_iter,
            plot=plot,
            single_run=single_run,
            #  partial_fulfilment=False,
        ),
        rotation,
    )
    T = TerminalC.shape[0]
    border_sizes = np.array(
        [border.shape[0]] + [obsC_.shape[0] for obsC_ in obstacleC__]
    )
    B = border_sizes.sum()
    obstacle_idxs = np.cumsum(border_sizes) + T
    return L_from_site(
        T=T,
        B=B,
        border=np.arange(T, T + border.shape[0]),
        obstacles=[np.arange(a, b) for a, b in pairwise(obstacle_idxs)],
        VertexC=np.vstack((TerminalC, BorderC, *obstacleC__, RootC)),
        **{
            k: v
            for k, v in L.graph.items()
            if k in ('R', 'handle', 'name', 'landscape_angle')
        },
    )


# iCDF_factory(T_min = 70,  T_max = 200, η = 0.6, d_lb = 0.045):
def iCDF_factory(
    T_min: int, T_max: int, η: float, d_lb: float
) -> Callable[[float], int]:
    """Helper for producing inverted cumulative distribution function (CDF).

    Goal: randomly sample the number of turbines ``T`` and the minimum clearance
    distance ``d`` between any two turbines.

    ``iCDF = iCDF_factory(...)``

    Sample the number of turbines: ``T ~ iCDF(uniform(0, 1))``

    Calculate the feasible range for ``d``: ``d_ub(T) = 2*sqrt(η/π/T)``

    Sample the minimum distance: ``d ~ uniform(d_lb, d_ub)``

    This exists because increasing both ``T`` and ``d`` may result in unfeasible
    combinations. One way to randomize both parameters is to first pick one
    and then limit the range for picking the other. This approach picks ``T``
    first, but from a non-uniform distribution. The non-uniformity is such that
    the parameter space ``T``×``d`` is uniformly sampled within the feasible area.

    The parameter ``η`` defines the curve for the upper bound of ``d_min``:
    ``d_ub(T)``.
    The theoretical optimum packing efficiency for circles is 0.9069, but when
    they are randomly placed, a more realistic feasible value is close to 0.6.

    Example::

      rng = np.random.default_rng()
      T_bounds = (50, 200)
      d_low_bound = 0.045
      η = 0.6  # 0.55..0.64, depending on the shape
      iCDF = iCDF_factory(*T_bounds, η, d_low_bound)
      T = iCDF(rng.uniform())
      d_high_bound = 2*np.sqrt(η/np.pi/T)
      d = rng.uniform(d_low_bound, d_high_bound)
      poisson_disc_filler(T, d, BorderC, ...)

    Args:
      T: number of terminals
      η: maximum feasible packing efficiency (for randomly placed circles)
      d_lb: lower bound for the minimum distance between WT

    Returns:
      Inverted CDF function.
    """

    def integral(x: float) -> float:  # integral of y(x) wrt x
        return 4 * math.sqrt(x * η / math.pi) - d_lb * x

    def integral_inv(y: float) -> float:  # integral_inv(integral(x)) = x
        return (
            -4 * math.sqrt(4 * η**2 - math.pi * η * d_lb * y)
            + 8 * η
            - math.pi * d_lb * y
        ) / (math.pi * d_lb**2)

    offset = integral(T_min - 0.4999999)
    area_under_curve = integral(T_max + 0.5) - offset

    def iCDF(u: float) -> int:
        """Inverted CDF.

        Maps from ``u ~ uniform(0, 1)`` to random variable ``T ~ custom_PDF()``.
        """
        return int(round(integral_inv(u * area_under_curve + offset)))

    return iCDF
