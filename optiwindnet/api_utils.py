import logging
import math
from itertools import pairwise
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import explain_validity

from .importer import L_from_site

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


def merge_obs_into_border(L):

    V = L.graph['VertexC']
    T, R = L.graph['T'], L.graph['R']

    turbinesC = V[:T]
    substationsC = V[-R:] if R > 0 else np.empty((0, 2), dtype=float)

    border_idx = L.graph.get('border')
    obstacles_idx = L.graph.get('obstacles', [])

    # Nothing to do if there's no border or no obstacles
    if border_idx is None or len(obstacles_idx) == 0:
        return  # leave L as-is

    borderC = V[border_idx]
    obstaclesC = [V[idx] for idx in obstacles_idx]

    # --- merge/tidy obstacles vs border (your original logic, with small guards) ---
    border_subtraction_verbose = True

    # Start with the original border polygon
    border_polygon = Polygon(borderC)
    if border_polygon.is_empty:
        raise ValueError("Border polygon is empty; cannot merge obstacles.")

    remaining_obstaclesC = []

    for i, obs in enumerate(obstaclesC):
        if obs.size == 0:  # skip empties defensively
            continue

        obs_poly = Polygon(obs)

        if not obs_poly.is_valid:
            warning('Obstacle %d invalid: %s', i, explain_validity(obs_poly))
            obs_poly = obs_poly.buffer(0)  # Try fixing it

        # Degenerate after fix?
        if obs_poly.is_empty:
            warning('Obstacle %d became empty after fix; skipping.', i)
            continue

        intersection = border_polygon.boundary.intersection(obs_poly)

        # If the obstacle is completely within the border, keep it
        if border_polygon.contains(obs_poly) and getattr(intersection, "length", 0) == 0:
            remaining_obstaclesC.append(obs)

        # If completely outside -> drop with a warning
        elif (not border_polygon.contains(obs_poly)) and (not border_polygon.intersects(obs_poly)):
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
    # Update obstacles (only those fully contained are kept)
    new_obstaclesC = remaining_obstaclesC

    # --- Rebuild VertexC and indices IN PLACE on L ---------------------------
    pieces = [turbinesC]
    border_idx_new = None
    cursor = T

    if new_borderC.size > 0:
        pieces.append(new_borderC)
        border_len = new_borderC.shape[0]
        border_idx_new = np.arange(cursor, cursor + border_len, dtype=int)
        cursor += border_len

    obstacle_ranges_new = []
    for obs in new_obstaclesC:
        n = obs.shape[0]
        if n == 0:
            continue
        pieces.append(obs)
        idx = np.arange(cursor, cursor + n, dtype=int)
        obstacle_ranges_new.append(idx)
        cursor += n

    if R > 0:
        pieces.append(substationsC)

    new_V = np.vstack(pieces) if pieces else np.empty((0, 2), dtype=float)

    # Update graph attributes
    L.graph['VertexC']   = new_V
    L.graph['border']    = border_idx_new
    L.graph['obstacles'] = obstacle_ranges_new
    L.graph['B']         = (new_borderC.shape[0] if new_borderC.size else 0) + sum(o.shape[0] for o in new_obstaclesC)

    return L

def buffer_border_obs(L, buffer_dist):
    V = L.graph["VertexC"]
    T, R = L.graph["T"], L.graph["R"]

    # Extract current pieces
    turbinesC    = V[:T]
    substationsC = V[-R:] if R > 0 else np.empty((0, 2), dtype=float)
    border_idx   = L.graph.get("border")
    obstacles_idx = L.graph.get("obstacles", [])

    borderC    = V[border_idx] if border_idx is not None else None
    obstaclesC = [V[idx] for idx in obstacles_idx]

    pre_buffer = {
        "borderC": None if borderC is None else borderC.copy(),
        "obstaclesC": [o.copy() for o in obstaclesC],
    }

    if buffer_dist > 0:

        # Border (guard for None/empty)
        if borderC is not None and getattr(borderC, "size", 0) > 0:
            border_polygon = Polygon(borderC)
            border_polygon = expand_polygon_safely(border_polygon, buffer_dist)
            borderC = np.array(border_polygon.exterior.coords[:-1])
        else:
            borderC = None  # keep as "no border"

        # Obstacles
        shrunk_obstaclesC = []
        shrunk_obstaclesC_including_removed = []
        for i, obs in enumerate(obstaclesC):
            if getattr(obs, "size", 0) == 0:
                shrunk_obstaclesC_including_removed.append([])
                continue
            obs_poly = Polygon(obs)
            obs_bufferedC = shrink_polygon_safely(obs_poly, buffer_dist, i)

            if isinstance(obs_bufferedC, list):     # MultiPolygon -> list of arrays
                shrunk_obstaclesC.extend(obs_bufferedC)
                shrunk_obstaclesC_including_removed.extend(obs_bufferedC)
            elif obs_bufferedC is not None:         # Single polygon
                shrunk_obstaclesC.append(obs_bufferedC)
                shrunk_obstaclesC_including_removed.append(obs_bufferedC)
            else:                                   # Removed
                shrunk_obstaclesC_including_removed.append([])

        obstaclesC = shrunk_obstaclesC

    elif buffer_dist < 0:
        raise ValueError('Buffer value must be equal or greater than 0!')

    # ----------------- Update L (rebuild VertexC and indices) -----------------
    pieces = [turbinesC]
    cursor = T
    border_idx_new = None

    if borderC is not None and getattr(borderC, "size", 0) > 0:
        pieces.append(borderC)
        blen = borderC.shape[0]
        border_idx_new = np.arange(cursor, cursor + blen, dtype=int)
        cursor += blen

    obstacle_ranges_new = []
    for obs in obstaclesC:
        if getattr(obs, "size", 0) == 0:
            continue
        n = obs.shape[0]
        pieces.append(obs)
        obstacle_ranges_new.append(np.arange(cursor, cursor + n, dtype=int))
        cursor += n

    if R > 0:
        pieces.append(substationsC)

    new_V = np.vstack(pieces) if pieces else np.empty((0, 2), dtype=float)

    L.graph["VertexC"]   = new_V
    L.graph["border"]    = border_idx_new
    L.graph["obstacles"] = obstacle_ranges_new
    L.graph["B"]         = (0 if border_idx_new is None else len(border_idx_new)) + sum(len(idx) for idx in obstacle_ranges_new)

    return L, pre_buffer


def _is_nonempty_xy(arr):
    return isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] > 0

def assert_inside_border(points, borderC, label):
    import numpy as np
    from matplotlib.path import Path
    if not _is_nonempty_xy(points) or not _is_nonempty_xy(borderC):
        return
    border_path = Path(borderC)
    in_neg = border_path.contains_points(points, radius=-1e-10)
    in_pos = border_path.contains_points(points, radius= 1e-10)
    inside = in_neg | in_pos
    if not np.all(inside):
        bad = np.where(~inside)[0]
        raise ValueError(f'{label} at indices {bad.tolist()} are outside the border!')

def assert_outside_obstacles(points, obstaclesC, label):
    if not _is_nonempty_xy(points):
        return
    for i, obs in enumerate(obstaclesC):
        if not _is_nonempty_xy(obs):
            continue
        obs_path = Path(obs)
        in_obs = obs_path.contains_points(points, radius=-1e-10)
        if np.any(in_obs):
            bad = np.where(in_obs)[0]
            raise ValueError(f'{label} at indices {bad.tolist()} are inside the obstacle at index {i}!')
