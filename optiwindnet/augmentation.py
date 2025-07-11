# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import math
from typing import Callable

import matplotlib.pyplot as plt
import networkx as nx
import numba as nb
import numpy as np

from .geometric import CoordPair, CoordPairs, IndexPairs, area_from_polygon_vertices
from .interarraylib import L_from_site

_lggr = logging.getLogger(__name__)
warn = _lggr.warning

__all__ = ('normalize_site_single_oss', 'poisson_disc_filler', 'turbinate')


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

    The point must be at least sqrt(repel_radius_sq) apart from repellers.

    Args:
      RepellerC: coordinates (R, 2) of repellers
      repel_radius_sq: the square of the minimum radius required
      point: coordinate (2,) of point to test

    Returns:
      True if `point` clears all discs centered on `RepellerC`.
    """
    return (
        ((point[np.newaxis, :] - RepellerC) ** 2).sum(axis=1) >= repel_radius_sq
    ).all()


def _contains_np(
    polyC: CoordPairs, pts: CoordPairs
) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
    """Evaluate if `polygon` (N, 2) covers points in `pts` (M, 2).

    Args:
      polyC: coordinates of polygon vertices (N, 2).
      pts: coordinates of points to test (M, 2).

    Returns:
      boolean array shaped (M,) (True if pts[i] inside `polygon`).
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
    """Evaluate if polygon (N, 2) covers `point` (2,).

    Args:
      polyC: coordinates of polygon vertices (N, 2).
      point: coordinates of point to test (2,).

    Returns:
      True if `point` inside polygon, False otherwise
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
    i_len: int,
    j_len: int,
    cell_idc: IndexPairs,
    BorderS: CoordPairs,
    repel_radius_sq: float,
    RepellerS: CoordPairs | None,
    rng: np.random.Generator,
) -> CoordPairs:
    """This is the numba-compilable core called by `poisson_disc_filler()`."""
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
          True if point does not conflict, False otherwise.
        """
        p_min, p_max = max(0, p - 2), min(i_len, p + 3)
        q_min, q_max = max(0, q - 2), min(j_len, q + 3)
        cells_window = cells[p_min:p_max, q_min:q_max].copy()
        mask = neighbormask[
            2 + p_min - p : 2 + p_max - p, 2 + q_min - q : 2 + q_max - q
        ] & (cells_window < T)
        ii = cells_window.reshape(mask.size)[np.flatnonzero(mask.flat)]
        return not (((point[None, :] - points[ii]) ** 2).sum(axis=-1) < 2).any()

    out_count = 0
    idc_list = list(range(len(cell_idc)))

    # dart-throwing loop
    miss_streak = 0
    for iter_count in range(1, max_iter + 1):
        if miss_streak > (max_iter - iter_count) / (T - out_count):
            # give up if the rate of misses is too high for the amount still
            # remaining to be placed
            break
        # pick random empty cell
        empty_idx = rng.integers(low=0, high=len(idc_list))
        ij = cell_idc[idc_list[empty_idx]]
        i, j = ij

        # dart throw inside cell
        dartC = ij + rng.random(2)

        # check border, overlap and repel_radius
        if not _contains(BorderS, dartC):
            miss_streak += 1
            continue
        elif not no_conflict(i, j, dartC):
            miss_streak += 1
            continue
        elif RepellerS is not None:
            if not _clears(RepellerS, repel_radius_sq, dartC):
                miss_streak += 1
                continue
        miss_streak = 0
        # add new point and remove cell from empty list
        points[out_count] = dartC
        cells[i, j] = out_count
        del idc_list[empty_idx]
        out_count += 1
        if out_count == T or not idc_list:
            break

    return points[:out_count], iter_count


def normalize_site_single_oss(L: nx.Graph) -> tuple[CoordPairs, CoordPair]:
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
        L.graph['R'] = 1
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
    obstacles: list[CoordPairs] | None = None,
    seed: int | None = None,
    max_iter: int = 10000,
    plot: bool = False,
    partial_fulfilment: bool = True,
    rounds: int = 1,
) -> CoordPairs:
    """Randomly place points inside an area respecting a minimum separation.

    Fills the area delimited by `BorderC` with `T` randomly
    placed points that are at least `min_dist` apart and that
    don't fall inside any of the `RepellerC` discs or `obstacles` areas.

    >>> Handling of `obstacles` is not yet implemented. <<<<

    Args:
      T: number of points to place.
      min_dist: minimum distance between place points.
      BorderC: coordinates (B × 2) of border polygon.
      RepellerC: coordinates (R × 2) of the centers of forbidden discs.
      repel_radius: the radius of the forbidden discs.
      obstacles: iterable (O × X × 2).
      iter_max_factor: factor to multiply by `T` to limit the number of
        iterations.
      rounds: number of times to start from empty while `T` is not reached.
      partial_fulfilment: whether to return less than `T` points (True) or
        to raise exception (False) if unable to fulfill request.

    Returns:
      coordinates (T, 2) of placed points
    """
    # TODO: implement obstacles zones
    if obstacles is not None:
        raise NotImplementedError('obstacles not implemented')

    offsetC, norm_factor, width, height = _get_border_scale_offset(BorderC)
    area_avail = 1.0 / norm_factor**2

    # quick check for outrageous densities
    # circle packing efficiency limit: η = π srqt(3)/6 = 0.9069
    # A Simple Proof of Thue's Theorem on Circle Packing
    # https://arxiv.org/abs/1009.4322
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
            print(
                'Info: Attempting partial fullfillment.',
                msg,
                'Try with lower T and/or min_dist.',
            )
        else:
            raise ValueError(msg)

    # create auxiliary grid covering the defined BorderC
    cell_size = min_dist / math.sqrt(2)
    i_len = math.ceil(width / cell_size)
    j_len = math.ceil(height / cell_size)
    BorderS = (BorderC - offsetC) / cell_size
    if RepellerC is None or repel_radius == 0.0:
        RepellerS = None
        repel_radius_sq = 0.0
    else:
        RepellerS = (RepellerC - offsetC) / cell_size
        repel_radius_sq = (repel_radius / cell_size) ** 2

    # Alternate implementation using np.mgrid
    #  pts = np.reshape(
    #      np.moveaxis(np.mgrid[0: i_len + 1, 0: j_len + 1], 0, -1),
    #      ((i_len + 1)*(j_len + 1), 2)
    #  )
    pts = np.empty(((i_len + 1) * (j_len + 1), 2), dtype=int)
    pts_temp = pts.reshape((i_len + 1, j_len + 1, 2))
    pts_temp[..., 0] = np.arange(i_len + 1)[:, np.newaxis]
    pts_temp[..., 1] = np.arange(j_len + 1)[np.newaxis, :]
    inside = _contains_np(BorderS, pts).reshape((i_len + 1, j_len + 1))

    # reduce 2×2 sub-matrices of `inside` with logical_or (i.e. .any())
    cell_covers_polygon = np.lib.stride_tricks.as_strided(
        inside,
        shape=(2, 2, inside.shape[0] - 1, inside.shape[1] - 1),
        strides=inside.strides * 2,
        writeable=False,
    ).any(axis=(0, 1))

    # check boundary's vertices
    for k, (i, j) in enumerate(BorderS.astype(int)):
        if not cell_covers_polygon[i, j]:
            ij = BorderS[k].copy()
            direction = BorderS[k - 1] - ij
            direction /= np.linalg.norm(direction)
            to_mark = [(i, j)]
            while True:
                nbr = (
                    cell_covers_polygon[max(0, i - 1), j]
                    or cell_covers_polygon[min(i_len - 1, i + 1), j]
                    or cell_covers_polygon[i, max(0, j - 1)]
                    or cell_covers_polygon[i, min(j_len - 1, j + 1)]
                )
                if nbr:
                    break
                ij += direction * 0.999
                i, j = ij.astype(int)
                to_mark.append((i, j))
            for i, j in to_mark:
                cell_covers_polygon[i, j] = True

    if RepellerS is not None and repel_radius >= min_dist:
        # the cells that contain the repellers can be discarded
        for r_i, r_j in RepellerS.astype(int):
            cell_covers_polygon[r_i, r_j] = False
    # Sequence of (i, j) of cells that overlap with the polygon
    cell_idc = np.argwhere(cell_covers_polygon)

    rng = np.random.default_rng(seed)

    # useful plot for debugging purposes only
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(
            cell_covers_polygon.T,
            origin='lower',
            extent=[0, cell_covers_polygon.shape[0], 0, cell_covers_polygon.shape[1]],
        )
        ax.scatter(*np.nonzero(inside), marker='+', facecolor='tab:green', s=7, lw=0.2)
        ax.scatter(
            *BorderS.T,
            marker='o',
            facecolor='none',
            s=10,
            lw=0.4,
            edgecolor='tab:orange',
            alpha=0.6,
        )
        ax.plot(*np.vstack((BorderS, BorderS[:1])).T)
        ax.axis('off')
        ax.grid()

    best_T = 0
    iter_counts = []
    for _ in range(rounds):
        # point-placing function
        points, iter_count = _poisson_disc_filler_core(
            T,
            max_iter,
            i_len,
            j_len,
            cell_idc,
            BorderS,
            repel_radius_sq,
            RepellerS,
            rng,
        )
        iter_counts.append(iter_count)
        if points.shape[0] > best_T:
            best_T = points.shape[0]
            best_points = points
            if best_T == T:
                break
    points = best_points

    # check if request was fulfilled
    if best_T < T:
        warn(
            'Only %d points generated (requested: %d, efficiency requested: '
            '%.3f, max_iter: %d). Iterations per round: %s',
            best_T,
            T,
            efficiency,
            max_iter,
            iter_counts,
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
    rounds: int = 5,
) -> nx.Graph:
    """Fills the location `L` with `T` turbines spaced at least `d` apart.

    Only the border and root locations from `L` are used.

    The placement of turbines is random and some combinations of `T` and `d`
    will result in fewer placements than requested. Increase `max_iter` and
    `rounds` to apply more effort before aborting.

    Args:
      L: reference location (only borders, obstacles and substations are used)
      T: desired number of turbines to place
      d: minimum spacing between turbines
      root_clearance: minimum spacing from turbine to substation (if not given,
        `d` is used)
      max_iter: maximum number of turbine placement attempts per empty field.
      rounds: how many times to start from an empty field before aborting.

    Returns:
      A location with randomly placed turbines (the number may be lower than T)
    """
    VertexC = L.graph['VertexC']
    border = L.graph['border']
    R = L.graph['R']
    BorderC = VertexC[border]
    offsetC, norm_scale, _, _ = _get_border_scale_offset(BorderC)
    BorderS = (BorderC - offsetC) * norm_scale
    RepellerS = (VertexC[-R:] - offsetC) * norm_scale
    TerminalS = poisson_disc_filler(
        T,
        d,
        BorderC=BorderS,
        RepellerC=RepellerS,
        repel_radius=(d if root_clearance is None else root_clearance),
        max_iter=max_iter,
        rounds=rounds,
        plot=plot,
    )
    T = TerminalS.shape[0]
    B = BorderS.shape[0]
    return L_from_site(
        T=T,
        B=B,
        border=np.arange(T, T + B),
        VertexC=np.vstack((TerminalS, BorderS, RepellerS)),
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
    """Helper for producing inverted cummulative ditribution function (CDF).

    Goal: randomly sample the number of turbines `T` and the minimum clearance
    distance `d` between any two turbines.

    iCDF = iCDF_factory(...)

    Sample the number of turbines: T~iCDF(uniform(0, 1))

    Calculate the feasible range for `d`: d_ub(T) = 2*sqrt(η/π/T)

    Sample the minimum distance: d~uniform(d_lb, d_ub)

    This exists because increasing both `T` and `d` may result in unfeasible
    combinations. One way to randomize both parameters is to first pick one
    and then limit the range for picking the other. This approach picks `T`
    first, but from a non-uniform distribution. The non-uniformity is such that
    the parameter space `T`×`d` is uniformly sampled within the feasible area.

    The parameter `η` defines the curve for the upper bound of d_min: d_ub(T).
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

        Maps from u ~ uniform(0, 1) to random variable T ~ custom_PDF().
        """
        return int(round(integral_inv(u * area_under_curve + offset)))

    return iCDF
