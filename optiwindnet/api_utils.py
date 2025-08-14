import logging
import math
from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import explain_validity

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


def plot_org_buff(borderC, border_bufferedC, obstaclesC, obstacles_bufferedC):
    plt.figure(figsize=(10, 10))
    plt.title('Original and Buffered Shapes')
    ax = plt.gca()

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
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def from_coordinates(
    self,
    turbinesC,
    substationsC,
    borderC,
    obstaclesC,
    name,
    handle,
    buffer_dist,
    **kwargs,
):
    """Constructs a site graph from coordinate-based inputs."""

    border_subtraction_verbose = True

    R = substationsC.shape[0]
    T = turbinesC.shape[0]

    if borderC is None:
        if obstaclesC is None:
            vertexC = np.vstack((turbinesC, substationsC))
            return L_from_site(
                R=R, T=T, B=0, name=name, handle=handle, VertexC=vertexC, **kwargs
            )
        else:
            border_sizes = np.array([0] + [obs.shape[0] for obs in obstaclesC])
            B = border_sizes.sum()
            obstacle_start_idxs = np.cumsum(border_sizes) + T
            obstacle_ranges = [
                np.arange(start, end) for start, end in pairwise(obstacle_start_idxs)
            ]

            vertexC = np.vstack((turbinesC, *obstaclesC, substationsC))

            return L_from_site(
                R=R,
                T=T,
                B=B,
                obstacles=obstacle_ranges,
                name=name,
                handle=handle,
                VertexC=vertexC,
                **kwargs,
            )

    if obstaclesC is None:
        obstaclesC = []
    else:
        # Start with the original border polygon
        border_polygon = Polygon(borderC)
        remaining_obstaclesC = []

        for i, obs in enumerate(obstaclesC):
            obs_poly = Polygon(obs)

            if not obs_poly.is_valid:
                warning('Obstacle %d invalid: %s', i, explain_validity(obs_poly))
                obs_poly = obs_poly.buffer(0)  # Try fixing it

            intersection = border_polygon.boundary.intersection(obs_poly)

            # If the obstacle is completely within the border, keep it
            if border_polygon.contains(obs_poly) and intersection.length == 0:
                remaining_obstaclesC.append(obs)

            elif not border_polygon.contains(
                obs_poly
            ) and not border_polygon.intersects(obs_poly):
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
        borderC = np.array(border_polygon.exterior.coords[:-1])

        # Update obstacles (only those fully contained are kept)
        obstaclesC = remaining_obstaclesC

    if buffer_dist > 0:
        self._borderC_original = borderC
        self._obstaclesC_original = obstaclesC
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

            else:  # None
                shrunk_obstaclesC_including_removed.append([])

        # Update obstacles
        obstaclesC = shrunk_obstaclesC

        self._border_bufferedC = borderC
        self._obstacles_bufferedC = obstaclesC
        self._obstacles_bufferedC_incl_removed = shrunk_obstaclesC_including_removed

    elif buffer_dist < 0:
        raise ValueError('Buffer value must be equal or greater than 0!')
    else:
        self._borderC_original = borderC
        self._obstaclesC_original = obstaclesC
        self._border_bufferedC = borderC
        self._obstacles_bufferedC = obstaclesC

    # check_turbine_locations(border, obstacles, turbines):
    border_path = Path(borderC)
    # Border path, with tolerance for edge inclusion
    in_border_neg = border_path.contains_points(turbinesC, radius=-1e-10)
    in_border_pos = border_path.contains_points(turbinesC, radius=1e-10)
    in_border = in_border_neg | in_border_pos

    # Check if any turbine is outside the border
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

    border_sizes = np.array([borderC.shape[0]] + [obs.shape[0] for obs in obstaclesC])
    B = border_sizes.sum().item()
    obstacle_start_idxs = np.cumsum(border_sizes) + T

    border_range = np.arange(T, T + borderC.shape[0])
    obstacle_ranges = [
        np.arange(start, end) for start, end in pairwise(obstacle_start_idxs)
    ]

    vertexC = np.vstack((turbinesC, borderC, *obstaclesC, substationsC))

    return L_from_site(
        R=R,
        T=T,
        B=B,
        border=border_range,
        obstacles=obstacle_ranges,
        name=name,
        handle=handle,
        VertexC=vertexC,
        **kwargs,
    )


def check_warmstart_feasibility(
    S_warm,
    cables_capacity,
    model_options,
    S_warm_has_detour,
    solver_name,
    verbose=False,
    logger=None,
):
    if logger is None:
        logger = logging.getLogger()

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
        feeder_limit = float('inf')  # fallback

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


def parse_cables_input(cables):
    cables_array = np.array(cables)
    if isinstance(cables, int):
        return [(cables, 1)]
    elif (
        cables_array.ndim == 1
        and cables_array.shape[0] == 1
        and isinstance(cables_array.item(), int)
    ):
        return [(int(cables_array[0]), 1)]
    elif cables_array.ndim == 1 and cables_array.shape[0] == 2:
        return [(int(cables_array[0]), float(cables_array[1]))]
    elif cables_array.ndim == 2 and cables_array.shape[1] == 2:
        return [(int(cap), float(cost)) for cap, cost in cables_array]
    else:
        raise ValueError(f'Invalid cable values: {cables}')


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
