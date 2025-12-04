import logging
import math
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import explain_validity

logger = logging.getLogger(__name__)
warning, info = logger.warning, logger.info

__all__ = ()


def expand_polygon_safely(polygon, buffer_dist):
    if not polygon.equals(polygon.convex_hull):
        max_buffer_dist = polygon.exterior.minimum_clearance / 2
        if buffer_dist >= max_buffer_dist:
            warning(
                'The defined border is non-convex and buffering may introduce unexpected changes. For visual comparison use plot_original_vs_buffered().'
            )
    return polygon.buffer(buffer_dist, quad_segs=2)


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
        reasons.append(
            'segmented feeders are incompatible with model option: feeder_route="straight"'
        )

    # Topology constraint
    branched_nodes = [n for n in S_warm.nodes if n >= 0 and S_warm.degree[n] > 2]
    if branched_nodes and model_options.get('topology') == 'radial':
        reasons.append(
            'branched network incompatible with model option: topology="radial"'
        )

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


def extract_network_as_array(G):
    keys = ['src', 'tgt', 'length', 'load', 'cable']
    types = [int, int, float, float, int]
    if 'has_cost' in G.graph:
        keys.append('cost')
        types.append(float)

    def iter_edges():
        for s, t, edgeD in G.edges(data=True):
            s, t = (s, t) if ((s < t) == edgeD['reverse']) else (t, s)
            yield s, t, *(edgeD[key] for key in keys[2:])

    network = np.fromiter(
        iter_edges(),
        dtype=list(zip(keys, types)),
        count=G.number_of_edges(),
    )

    return network


def merge_obs_into_border(L):
    V = L.graph['VertexC']
    T, R = L.graph['T'], L.graph['R']

    turbinesC = V[:T]
    substationsC = V[-R:]

    border_idx = L.graph.get('border')
    obstacles_idx = L.graph.get('obstacles', [])

    # Do nothing if there's no border or no obstacles
    if border_idx is None or len(obstacles_idx) == 0:
        return L

    borderC = V[border_idx]
    obstaclesC = [V[idx] for idx in obstacles_idx]

    # To print only once even if multiple obstacles are intersecting with border
    border_subtraction_verbose = True

    #
    border_polygon = Polygon(borderC)

    remaining_obstaclesC = []
    for i, obs in enumerate(obstaclesC):
        if obs.size == 0:
            continue

        obs_poly = Polygon(obs)

        if not obs_poly.is_valid:
            warning('Obstacle %d invalid: %s', i, explain_validity(obs_poly))

        if obs_poly.is_empty:
            warning('Obstacle %d became an empty polygon; skipping.', i)
            continue

        intersection = border_polygon.boundary.intersection(obs_poly)

        # If the obstacle is completely within the border, keep it
        if (
            border_polygon.contains(obs_poly)
            and getattr(intersection, 'length', 0) == 0
        ):
            remaining_obstaclesC.append(obs)

        # If completely outside -> drop with a warning
        elif (not border_polygon.contains(obs_poly)) and (
            not border_polygon.intersects(obs_poly)
        ):
            warning(
                'Obstacle at index %d is completely outside the border and is neglected.',
                i,
            )
        else:
            # Subtract this obstacle from the border
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

            # If the subtraction results in multiple pieces (MultiPolygon), raise error
            if isinstance(new_border_polygon, MultiPolygon):
                raise ValueError(
                    'Obstacle subtraction resulted in multiple pieces (MultiPolygon) — check your geometry.'
                )
            else:
                border_polygon = new_border_polygon

    # Update the border as a NumPy array of exterior coordinates
    new_borderC = np.array(border_polygon.exterior.coords[:-1])
    # Update obstacles
    new_obstaclesC = remaining_obstaclesC

    # --- Rebuild VertexC and L
    coordinatesC = [turbinesC]
    border_idx_new = None
    cursor = T

    if new_borderC.size > 0:
        coordinatesC.append(new_borderC)
        border_len = new_borderC.shape[0]
        border_idx_new = np.arange(cursor, cursor + border_len, dtype=int)
        cursor += border_len

    obstacle_ranges_new = []
    for obs in new_obstaclesC:
        n = obs.shape[0]
        if n == 0:
            continue
        coordinatesC.append(obs)
        idx = np.arange(cursor, cursor + n, dtype=int)
        obstacle_ranges_new.append(idx)
        cursor += n

    # substations
    coordinatesC.append(substationsC)

    new_V = np.vstack(coordinatesC)

    # Update graph attributes
    L.graph['VertexC'] = new_V
    L.graph['border'] = border_idx_new
    L.graph['obstacles'] = obstacle_ranges_new
    L.graph['B'] = (new_borderC.shape[0] if new_borderC.size else 0) + sum(
        o.shape[0] for o in new_obstaclesC
    )

    return L


def buffer_border_obs(L, buffer_dist):
    V = L.graph['VertexC']
    T, R = L.graph['T'], L.graph['R']

    # Extract coordinates
    turbinesC = V[:T]
    substationsC = V[-R:]
    border_idx = L.graph.get('border')
    obstacles_idx = L.graph.get('obstacles', [])

    borderC = V[border_idx] if border_idx is not None else None
    obstaclesC = [V[idx] for idx in obstacles_idx]

    pre_buffer = {
        'borderC': borderC.copy(),
        'obstaclesC': [obs.copy() for obs in obstaclesC],
    }

    if buffer_dist == 0:
        return L, pre_buffer

    elif buffer_dist > 0:
        # Border
        if borderC is not None:
            border_polygon = Polygon(borderC)
            border_polygon = expand_polygon_safely(border_polygon, buffer_dist)
            borderC = np.array(border_polygon.exterior.coords[:-1])

        # Obstacles
        shrunk_obstaclesC = []
        for i, obs in enumerate(obstaclesC):
            if getattr(obs, 'size', 0) == 0:
                continue
            obs_poly = Polygon(obs)
            obs_bufferedC = shrink_polygon_safely(obs_poly, buffer_dist, i)

            if isinstance(obs_bufferedC, list):  # MultiPolygon -> list of arrays
                shrunk_obstaclesC.extend(obs_bufferedC)
            elif obs_bufferedC is not None:  # Single polygon
                shrunk_obstaclesC.append(obs_bufferedC)

        obstaclesC = shrunk_obstaclesC

        # --- Update L
        coordinatesC = [turbinesC]
        cursor = T
        border_idx_new = None

        if borderC is not None:
            coordinatesC.append(borderC)
            blen = borderC.shape[0]
            border_idx_new = np.arange(cursor, cursor + blen, dtype=int)
            cursor += blen

        obstacle_ranges_new = []
        for obs in obstaclesC:
            if getattr(obs, 'size', 0) == 0:
                continue
            n = obs.shape[0]
            coordinatesC.append(obs)
            obstacle_ranges_new.append(np.arange(cursor, cursor + n, dtype=int))
            cursor += n

        coordinatesC.append(substationsC)

        new_V = np.vstack(coordinatesC)

        L.graph['VertexC'] = new_V
        L.graph['border'] = border_idx_new
        L.graph['obstacles'] = obstacle_ranges_new
        L.graph['B'] = (0 if border_idx_new is None else len(border_idx_new)) + sum(
            len(idx) for idx in obstacle_ranges_new
        )

        return L, pre_buffer

    else:  # buffer_dist < 0
        raise ValueError('Buffer value must be equal or greater than 0!')
