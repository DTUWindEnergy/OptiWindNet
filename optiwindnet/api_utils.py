import numpy as np
from shapely.geometry import Polygon, MultiPoint, MultiPolygon
from shapely.validation import explain_validity
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path
from optiwindnet.importer import L_from_site
import logging


from itertools import pairwise
from pathlib import Path
from shapely.geometry import Polygon, MultiPoint, MultiPolygon
from matplotlib.patches import Polygon as MplPolygon
import copy
from matplotlib.path import Path
import numpy as np
import yaml
import yaml_include
import logging
import numpy as np
import matplotlib.pyplot as plt
from shapely.ops import unary_union

# Local utilities
from optiwindnet.utils import NodeTagger
F = NodeTagger()


logger = logging.getLogger(__name__)
warning, info = logger.warning, logger.info

def expand_polygon_safely(polygon, buffer_dist):
    if not polygon.equals(polygon.convex_hull):
        max_buffer_dist = polygon.exterior.minimum_clearance / 2
        if buffer_dist >= max_buffer_dist:
            warning("⚠️ The defined border is non-convex and buffering may introduce unexpected changes. For visual comparison use plot_original_vs_buffered().")
    return polygon.buffer(buffer_dist, resolution=2)

def shrink_polygon_safely(polygon, shrink_dist, indx):
    """Shrink a polygon and warn if it splits or disappears."""
    shrunk_polygon = polygon.buffer(-shrink_dist)

    if shrunk_polygon.is_empty:
        warning("⚠️ Buffering by %.2f completely removed the obstacle at index %d. For visual comparison use plot_original_vs_buffered()." % (shrink_dist, indx))
        return None

    elif shrunk_polygon.geom_type == 'MultiPolygon':
        warning("⚠️ Shrinking by %.2f split the obstacle at index %d into %d pieces. For visual comparison use plot_original_vs_buffered()." % (shrink_dist, indx, len(shrunk_polygon.geoms)))
        return [np.array(part.exterior.coords) for part in shrunk_polygon.geoms]

    elif shrunk_polygon.geom_type == 'Polygon':
        return np.array(shrunk_polygon.exterior.coords)

    else:
        warning("⚠️ Unexpected geometry type %s after shrinking obstacle at index %d. The obstacle is totally removed. For visual comparison use plot_original_vs_buffered()." %
                (shrunk_polygon.geom_type, indx))
        return None

def plot_org_buff(borderC, border_bufferedC, obstaclesC, obstacles_bufferedC):
    plt.figure(figsize=(10, 10))
    plt.title("Original and Buffered Shapes")
    ax = plt.gca()

    # Plot original
    ax.add_patch(MplPolygon(borderC, closed=True, edgecolor='none', facecolor='lightblue', label='Original Border'))
    for i, obs in enumerate(obstaclesC):
        ax.add_patch(MplPolygon(obs, closed=True, edgecolor='none', facecolor='white', label='Original Obstacle' if i == 0 else None))

    # Plot buffered
    ax.add_patch(MplPolygon(border_bufferedC, closed=True, edgecolor='red', linestyle='--', facecolor='none', label='Buffered Border'))
    for i, obs in enumerate(obstacles_bufferedC):
        ax.add_patch(MplPolygon(obs, closed=True, edgecolor='black', linestyle='--', facecolor='none', label='Buffered Obstacle' if i == 0 else None))

    # Collect all coordinates for axis scaling
    all_x = np.concatenate([borderC[:, 0], border_bufferedC[:, 0]] + [obs[:, 0] for obs in obstaclesC] + [obs[:, 0] for obs in obstacles_bufferedC])
    all_y = np.concatenate([borderC[:, 1], border_bufferedC[:, 1]] + [obs[:, 1] for obs in obstaclesC] + [obs[:, 1] for obs in obstacles_bufferedC])

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


def from_coordinates(self, turbinesC, substationsC, borderC, obstaclesC, name, handle, buffer_dist):
    """Constructs a site graph from coordinate-based inputs."""
    
    border_subtraction_verbose = True

    R = substationsC.shape[0]
    T = turbinesC.shape[0]

    if borderC is None:
        if obstaclesC is None:
            vertexC = np.vstack((turbinesC, substationsC))
            return L_from_site(
                R=R,
                T=T,
                B=0,
                name=name,
                handle=handle,
                VertexC=vertexC
            )
        else:    
            border_sizes = np.array([0] + [obs.shape[0] for obs in obstaclesC])
            B = border_sizes.sum()
            obstacle_start_idxs = np.cumsum(border_sizes) + T
            obstacle_ranges = [np.arange(start, end) for start, end in pairwise(obstacle_start_idxs)]

            vertexC = np.vstack((turbinesC, *obstaclesC, substationsC))

            return L_from_site(
                R=R,
                T=T,
                B=B,
                obstacles=obstacle_ranges,
                name=name,
                handle=handle,
                VertexC=vertexC
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
                warning("⚠️ Obstacle %d invalid: %s" %(i, explain_validity(obs_poly)))
                obs_poly = obs_poly.buffer(0)  # Try fixing it

            intersection = border_polygon.boundary.intersection(obs_poly)

            # If the obstacle is completely within the border, keep it
            if border_polygon.contains(obs_poly) and intersection.length == 0:
                remaining_obstaclesC.append(obs)

            elif not border_polygon.contains(obs_poly) and not border_polygon.intersects(obs_poly):
                warning("⚠️ Obstacle at index %d is completely outside the border and is neglegcted." %i)
            else:
                # Subtract this obstacle from the border
                warning("⚠️ Obstacle at index %d intersects with the exteriour border and is merged into the exterior border." %i)
                new_border_polygon = border_polygon.difference(obs_poly)

                if new_border_polygon.is_empty:
                    raise ValueError("Obstacle subtraction resulted in an empty border — check your geometry.")

                if border_subtraction_verbose:
                    info("At least one obstacle intersects/touches the border. The border is redefined to exclude those obstacles.")
                    border_subtraction_verbose = False

                # If the subtraction results in multiple pieces (MultiPolygon), raise error
                if isinstance(new_border_polygon, MultiPolygon):
                    raise ValueError("Obstacle subtraction resulted in multiple pieces (MultiPolygon) — check your geometry.")
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

            elif obs_bufferedC is not None:      # Polygon
                shrunk_obstaclesC.append(obs_bufferedC)
                shrunk_obstaclesC_including_removed.append(obs_bufferedC)  

            else: # None
                shrunk_obstaclesC_including_removed.append([])

        # Update obstacles
        obstaclesC = shrunk_obstaclesC
        
        self._border_bufferedC = borderC
        self._obstacles_bufferedC =  obstaclesC
        self._obstacles_bufferedC_incl_removed = shrunk_obstaclesC_including_removed

    elif buffer_dist < 0:
        raise ValueError("Buffer value must be equal or greater than 0!")
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
        raise ValueError("Turbines at indices %s are outside the border!" % outside_idx)

    for i, obs in enumerate(obstaclesC):
        obs_path = Path(obs)
        in_obstacle = obs_path.contains_points(turbinesC, radius=-1e-10)
        if np.any(in_obstacle):
            inside_idx = np.where(in_obstacle)[0]
            raise ValueError(f"Turbines at indices {inside_idx} are inside the obstacle at index {i}!")
        
    border_sizes = np.array([borderC.shape[0]] + [obs.shape[0] for obs in obstaclesC])
    B = border_sizes.sum()
    obstacle_start_idxs = np.cumsum(border_sizes) + T

    border_range = np.arange(T, T + borderC.shape[0])
    obstacle_ranges = [np.arange(start, end) for start, end in pairwise(obstacle_start_idxs)]

    vertexC = np.vstack((turbinesC, borderC, *obstaclesC, substationsC))

    return L_from_site(
        R=R,
        T=T,
        B=B,
        border=border_range,
        obstacles=obstacle_ranges,
        name=name,
        handle=handle,
        VertexC=vertexC
    )