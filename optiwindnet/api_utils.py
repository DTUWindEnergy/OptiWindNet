import logging
import math
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from .importer import L_from_site

logger = logging.getLogger(__name__)
warning, info = logger.warning, logger.info


def expand_polygon_safely(polygon, buffer_dist):
    if not polygon.equals(polygon.convex_hull):
        max_buffer_dist = polygon.exterior.minimum_clearance / 2
        if buffer_dist >= max_buffer_dist:
            warning(
                'The defined border is non-convex and buffering may introduce unexpected changes. For visual comparison use plot_original_vs_buffered().'
            )
    return polygon.buffer(buffer_dist, resolution=2)


def shrink_polygon_safely(polygon, shrink_dist, indx):
    """Shrink a polygon and warn if it splits or disappears."""
    shrunk_polygon = polygon.buffer(-shrink_dist)

    if shrunk_polygon.is_empty:
        warning(
            'Buffering by %.2f completely removed the obstacle at index %d. For visual comparison use plot_original_vs_buffered().',
            shrink_dist,
            indx,
        )
        return None

    elif shrunk_polygon.geom_type == 'MultiPolygon':
        warning(
            'Shrinking by %.2f split the obstacle at index %d into %d pieces. For visual comparison use plot_original_vs_buffered().',
            shrink_dist,
            indx,
            len(shrunk_polygon.geoms),
        )
        return [np.array(part.exterior.coords) for part in shrunk_polygon.geoms]

    elif shrunk_polygon.geom_type == 'Polygon':
        return np.array(shrunk_polygon.exterior.coords)

    else:
        warning(
            'Unexpected geometry type %s after shrinking obstacle at index %d. The obstacle is totally removed. For visual comparison use plot_original_vs_buffered().',
            shrunk_polygon.geom_type,
            indx,
        )
        return None


def plot_org_buff(borderC, border_bufferedC, obstaclesC, obstacles_bufferedC, **kwargs):
    fig = plt.figure(**({'layout': 'constrained'} | kwargs))
    ax = fig.add_subplot()
    ax.set_title('Original and Buffered Shapes')

    # Plot original
    ax.add_patch(
        MplPolygon(
            borderC,
            closed=True,
            edgecolor='none',
            facecolor='lightblue',
            label='Original Border',
        )
    )
    for i, obs in enumerate(obstaclesC):
        ax.add_patch(
            MplPolygon(
                obs,
                closed=True,
                edgecolor='none',
                facecolor='white',
                label='Original Obstacle' if i == 0 else None,
            )
        )

    # Plot buffered
    ax.add_patch(
        MplPolygon(
            border_bufferedC,
            closed=True,
            edgecolor='red',
            linestyle='--',
            facecolor='none',
            label='Buffered Border',
        )
    )
    for i, obs in enumerate(obstacles_bufferedC):
        ax.add_patch(
            MplPolygon(
                obs,
                closed=True,
                edgecolor='black',
                linestyle='--',
                facecolor='none',
                label='Buffered Obstacle' if i == 0 else None,
            )
        )

    # Collect all coordinates for axis scaling
    all_x = np.concatenate(
        [borderC[:, 0], border_bufferedC[:, 0]]
        + [obs[:, 0] for obs in obstaclesC]
        + [obs[:, 0] for obs in obstacles_bufferedC]
    )
    all_y = np.concatenate(
        [borderC[:, 1], border_bufferedC[:, 1]]
        + [obs[:, 1] for obs in obstaclesC]
        + [obs[:, 1] for obs in obstacles_bufferedC]
    )

    # Add padding
    x_pad = 0.05 * (all_x.max() - all_x.min())
    y_pad = 0.05 * (all_y.max() - all_y.min())
    ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
    ax.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)

    ax.set_aspect('equal')
    ax.legend()
    ax.set_axis_off()
    return ax


def is_warmstart_eligible(
    S_warm,
    cables_capacity,
    model_options,
    S_warm_has_detour,
    solver_name,
    logger,
    verbose=False,
):
    verbose_warmstart = verbose or logger.isEnabledFor(logging.INFO)

    if S_warm is None:
        if verbose_warmstart:
            print('>>> No solution is available for warmstarting! <<<')
            print()
        return False

    R = S_warm.graph['R']
    T = S_warm.graph['T']
    capacity = cables_capacity

    reasons = []

    # Feeder constraints
    feeder_counts = [S_warm.degree[r] for r in range(-R, 0)]
    feeder_limit_mode = model_options.get('feeder_limit', 'unlimited')
    feeder_minimum = math.ceil(T / capacity)

    if feeder_limit_mode == 'unlimited':
        feeder_limit = float('inf')
    elif feeder_limit_mode == 'specified':
        feeder_limit = model_options.get('max_feeders')
    elif feeder_limit_mode == 'minimum':
        feeder_limit = feeder_minimum
    elif feeder_limit_mode == 'min_plus1':
        feeder_limit = feeder_minimum + 1
    elif feeder_limit_mode == 'min_plus2':
        feeder_limit = feeder_minimum + 2
    elif feeder_limit_mode == 'min_plus3':
        feeder_limit = feeder_minimum + 3
    else:
        feeder_limit = float('inf')

    if feeder_counts[0] > feeder_limit:
        reasons.append(
            f'number of feeders ({feeder_counts[0]}) exceeds feeder limit ({feeder_limit})'
        )

    # Detour constraint
    if S_warm_has_detour and model_options.get('feeder_route') == 'straight':
        reasons.append('detours present but feeder_route is set to "straight"')

    # Topology constraint
    branched_nodes = [n for n in S_warm.nodes if n >= 0 and S_warm.degree[n] > 2]
    if branched_nodes and model_options.get('topology') == 'radial':
        reasons.append('branched structure not allowed under "radial" topology')

    # Output
    if reasons and verbose_warmstart:
        print()
        print(
            'Warning: No warmstarting (even though a solution is available) due to the following reason(s):'
        )
        for reason in reasons:
            print(f'    - {reason}')
        print()
        return False
    elif solver_name != 'scip':
        msg = 'Using warm start: the model is initialized with the provided solution S.'
        if verbose_warmstart:
            print(msg)
            print()
        return True
    else:
        return False


def parse_cables_input(
    cables: int | list[int] | list[tuple[int, float]] | np.ndarray,
) -> list[tuple[int, float]]:
    # If input is numpy array, convert to list for uniform processing
    if isinstance(cables, np.ndarray):
        cables = cables.tolist()

    if isinstance(cables, int):
        # single number means the maximum capacity, set cost to 0
        return [(cables, 0.0)]
    elif isinstance(cables, Sequence):
        cables_out = []
        for entry in cables:
            if isinstance(entry, int):
                # any entry that is a single number is the capacity, set cost to 0
                cables_out.append((entry, 0.0))
            elif isinstance(entry, Sequence) and len(entry) == 2:
                cables_out.append(tuple(entry))
            else:
                raise ValueError(f'Invalid cable values: {cables}')
        return cables_out


def enable_ortools_logging_if_jupyter(solver):
    try:
        shell = get_ipython().__class__.__name__
    except NameError:
        pass
    else:
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or lab
            solver.solver.log_callback = print


def compute_edge_gradients(G, gradient_type='length'):
    if gradient_type.lower() not in ['cost', 'length']:
        raise ValueError("gradient_type should be either 'cost' or 'length'")

    VertexC = G.graph['VertexC']
    R = G.graph['R']
    T = G.graph['T']
    gradients = np.zeros_like(VertexC)

    fnT = G.graph.get('fnT')
    if fnT is not None:
        _u, _v = fnT[np.array(G.edges)].T
    else:
        _u, _v = np.array(G.edges).T

    vec = VertexC[_u] - VertexC[_v]
    norm = np.hypot(*vec.T)
    norm[norm < 1e-12] = 1.0
    vec /= norm[:, None]

    if gradient_type.lower() == 'cost':
        cable_costs = np.fromiter(
            (G.graph['cables'][cable]['cost'] for *_, cable in G.edges(data='cable')),
            dtype=np.float64,
            count=G.number_of_edges(),
        )
        vec *= cable_costs[:, None]

    np.add.at(gradients, _u, vec)
    np.subtract.at(gradients, _v, vec)

    return gradients[:T], gradients[-R:]


def extract_network_as_array(G):
    network_array_type = np.dtype(
        [
            ('src', int),
            ('tgt', int),
            ('length', float),
            ('load', float),
            ('reverse', bool),
            ('cable', int),
            ('cost', float),
        ]
    )

    def iter_edges(G, keys):
        for s, t, edgeD in G.edges(data=True):
            yield s, t, *(edgeD[key] for key in keys)

    network = np.fromiter(
        iter_edges(G, network_array_type.names[2:]),
        dtype=network_array_type,
        count=G.number_of_edges(),
    )
    return network


import numpy as np
from itertools import pairwise
from matplotlib.path import Path
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import explain_validity

# --- small utilities ---------------------------------------------------------


def _stack_vertices_no_border(turbinesC, substationsC, obstaclesC):
    """Return (vertexC, B, obstacle_ranges) when there is NO exterior border."""
    if not obstaclesC:
        vertexC = np.vstack((turbinesC, substationsC))
        return vertexC, 0, []
    border_sizes = np.array([0] + [obs.shape[0] for obs in obstaclesC])
    B = int(border_sizes.sum())
    T = turbinesC.shape[0]
    obstacle_start_idxs = np.cumsum(border_sizes) + T
    obstacle_ranges = [np.arange(s, e) for s, e in pairwise(obstacle_start_idxs)]
    vertexC = np.vstack((turbinesC, *obstaclesC, substationsC))
    return vertexC, B, obstacle_ranges


def _merge_obstacles_into_border(
    borderC, obstaclesC, *, border_subtraction_verbose=True
):
    """
    Ensure obstacles fully outside are dropped, fully inside are kept,
    and obstacles intersecting/touching the exterior border are subtracted
    from the border (with MultiPolygon check).
    Returns (new_borderC, remaining_obstaclesC).
    """
    if not obstaclesC:
        return borderC, []

    border_polygon = Polygon(borderC)
    remaining_obstaclesC = []

    for i, obs in enumerate(obstaclesC):
        obs_poly = Polygon(obs)
        if not obs_poly.is_valid:
            warning('Obstacle %d invalid: %s', i, explain_validity(obs_poly))
            obs_poly = obs_poly.buffer(0)

        intersection = border_polygon.boundary.intersection(obs_poly)

        # keep if fully inside and not touching boundary
        if border_polygon.contains(obs_poly) and intersection.length == 0:
            remaining_obstaclesC.append(obs)
            continue

        # drop if completely outside
        if not border_polygon.contains(obs_poly) and not border_polygon.intersects(
            obs_poly
        ):
            warning(
                'Obstacle at index %d is completely outside the border and is neglected.',
                i,
            )
            continue

        # merge with exterior border: subtract from border
        warning(
            'Obstacle at index %d intersects with the exteriour border and is merged into the exterior border.',
            i,
        )
        new_border_polygon = border_polygon.difference(obs_poly)

        if new_border_polygon.is_empty:
            raise ValueError(
                'Obstacle subtraction resulted in an empty border — check your geometry.'
            )

        if border_subtraction_verbose:
            info(
                'At least one obstacle intersects/touches the border. The border is redefined to exclude those obstacles.'
            )
            border_subtraction_verbose = False

        if isinstance(new_border_polygon, MultiPolygon):
            raise ValueError(
                'Obstacle subtraction resulted in multiple pieces (MultiPolygon) — check your geometry.'
            )

        border_polygon = new_border_polygon

    new_borderC = np.array(border_polygon.exterior.coords[:-1])
    return new_borderC, remaining_obstaclesC


def _buffer_geometries(
    borderC,
    obstaclesC,
    buffer_dist,
    self_obj,
):
    """
    Apply buffering (expand border, shrink obstacles) if buffer_dist > 0.
    Updates `self_obj` cache attributes similarly to the original method.
    Returns (borderC, obstaclesC).
    """
    if buffer_dist < 0:
        raise ValueError('Buffer value must be equal or greater than 0!')

    # Always set originals
    self_obj._borderC_original = borderC
    self_obj._obstaclesC_original = obstaclesC

    if buffer_dist == 0:
        self_obj._border_bufferedC = borderC
        self_obj._obstacles_bufferedC = obstaclesC
        return borderC, obstaclesC

    # border
    border_polygon = Polygon(borderC)
    border_polygon = expand_polygon_safely(border_polygon, buffer_dist)
    borderC = np.array(border_polygon.exterior.coords[:-1])

    # obstacles
    shrunk_obstaclesC = []
    shrunk_obstaclesC_including_removed = []
    for i, obs in enumerate(obstaclesC):
        obs_poly = Polygon(obs)
        obs_bufferedC = shrink_polygon_safely(obs_poly, buffer_dist, i)

        if isinstance(obs_bufferedC, list):  # MultiPolygon
            shrunk_obstaclesC.extend(obs_bufferedC)
            shrunk_obstaclesC_including_removed.extend(obs_bufferedC)
        elif obs_bufferedC is not None:  # Polygon
            shrunk_obstaclesC.append(obs_bufferedC)
            shrunk_obstaclesC_including_removed.append(obs_bufferedC)
        else:  # None (fully removed)
            shrunk_obstaclesC_including_removed.append([])

    obstaclesC = shrunk_obstaclesC

    # cache
    self_obj._border_bufferedC = borderC
    self_obj._obstacles_bufferedC = obstaclesC
    self_obj._obstacles_bufferedC_incl_removed = shrunk_obstaclesC_including_removed
    return borderC, obstaclesC


def _validate_turbines_within_area(turbinesC, borderC, obstaclesC):
    """
    Ensure all turbines are inside the (possibly buffered) border and outside all obstacles.
    Raises ValueError on failure.
    """
    border_path = Path(borderC)
    # include a tiny tolerance around edges
    in_border_neg = border_path.contains_points(turbinesC, radius=-1e-10)
    in_border_pos = border_path.contains_points(turbinesC, radius=1e-10)
    in_border = in_border_neg | in_border_pos

    if not np.all(in_border):
        outside_idx = np.where(~in_border)[0]
        raise ValueError('Turbines at indices %s are outside the border!' % outside_idx)

    for i, obs in enumerate(obstaclesC):
        obs_path = Path(obs)
        in_obstacle = obs_path.contains_points(turbinesC, radius=-1e-10)
        if np.any(in_obstacle):
            inside_idx = np.where(in_obstacle)[0]
            raise ValueError(
                f'Turbines at indices {inside_idx} are inside the obstacle at index {i}!'
            )


def _assemble_vertices_and_ranges(turbinesC, borderC, obstaclesC, substationsC):
    """
    Compute B, border indices, obstacle ranges, and stacked vertex array for L_from_site.
    """
    T = turbinesC.shape[0]
    border_sizes = np.array([borderC.shape[0]] + [obs.shape[0] for obs in obstaclesC])
    B = int(border_sizes.sum())
    obstacle_start_idxs = np.cumsum(border_sizes) + T

    border_range = np.arange(T, T + borderC.shape[0])
    obstacle_ranges = [np.arange(s, e) for s, e in pairwise(obstacle_start_idxs)]
    vertexC = np.vstack((turbinesC, borderC, *obstaclesC, substationsC))
    return B, border_range, obstacle_ranges, vertexC

