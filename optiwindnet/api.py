from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import explain_validity
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
###################
# OptiWindNet API #
###################

from abc import ABC, abstractmethod
from typing import Any, Mapping
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

PackType = Mapping[str, Any]

# optiwindnet modules
from optiwindnet.importer import L_from_yaml, L_from_pbf, L_from_site
from optiwindnet.plotting import gplot, pplot
from optiwindnet.svg import svgplot
from optiwindnet.mesh import make_planar_embedding
from optiwindnet.pathfinding import PathFinder
from optiwindnet.interface import assign_cables
from optiwindnet.interarraylib import G_from_S, calcload

# Heuristics
from optiwindnet.heuristics import EW_presolver

# Metaheuristics
from optiwindnet.baselines.hgs import iterative_hgs_cvrp

# MILP
from optiwindnet.MILP import solver_factory, ModelOptions

#################################################
#                                               #
#################################################

logger = logging.getLogger(__name__)
error, warning, info = logger.error, logger.warning, logger.info

class WindFarmNetwork:
    """
    Represents a wind farm electrical network, capable of processing 
    layout data from different formats and computing network properties.
    """

    def __init__(self, cables, turbinesC=None, substationsC=None,
                 borderC=None, obstaclesC=None,
                 name='', handle='', L=None, router=None, buffer_dist = 0, **kwargs):

        # Default router if none provided
        if router is None:
            router = Heuristic(solver='Esau_Williams')
        self.router = router
        cables_array = np.array(cables)

        if isinstance(cables, int):
            cables = [(cables, 1)]

        elif cables_array.ndim == 1 and cables_array.shape[0] == 1 and isinstance(cables_array.item(), int):
            cables = [(int(cables_array[0]), 1)]
        elif (cables_array.ndim == 1 and cables_array.shape[0] == 2):
            cables = [(int(cables_array[0]), float(cables_array[1]))]
        elif cables_array.ndim == 2:
            if cables_array.shape[1] == 2:
                cables = [(int(cap), float(cost)) for cap, cost in cables_array]
        else:
            error(f"Invalid cable values: {cables}")


        self.cables = cables
        self.cables_capacity = max(c[0] for c in cables)

        # Load layout from coordinates if turbinesC provided
        if turbinesC is not None and substationsC is not None:
            L = self._from_coordinates(turbinesC, substationsC, borderC, obstaclesC, name, handle, buffer_dist)
        elif L is None:
            raise ValueError('Both turbinesC and substationsC must be provided!')
        self.L = L

        # Planar embedding
        self.P, self.A = make_planar_embedding(L)

        # Geometry inputs
        self.borderC = borderC
        self.obstaclesC = obstaclesC

        # Graph/network placeholders and status flags
        self.S = None
        self.G = None

    def cost(self):
        """Returns the total cost of the network."""
        return self.G.size(weight="cost")

    def length(self):
        """Returns the total cable length of the network."""
        return self.G.size(weight="length")

    def expand_polygon_safely(self, polygon, buffer_dist):
        """Expand a polygon and warn if buffer might fill narrow gaps."""
        
        if not polygon.equals(polygon.convex_hull):
            max_buffer_dist = polygon.exterior.minimum_clearance / 2

            if buffer_dist >= max_buffer_dist:
                warning("⚠️ The defined border is non-convex and buffering may introduce unexpexted changes in the exterior border. For visual comparison use plot_original_vs_buffered().")
        
        expanded_polygon = polygon.buffer(buffer_dist, resolution=2)

        return expanded_polygon

    def shrink_polygon_safely(self, polygon, shrink_dist, indx):
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
            print(shrunk_polygon.geom_type)
            warning("⚠️ Unexpected geometry type %s after shrinking obstacle at index %d. The obstacle is totally removed. For visual comparison use plot_original_vs_buffered()." %
                    (shrunk_polygon.geom_type, indx))
            return None


    def plot_original_vs_buffered(self):
        """
        Plot original and buffered borders and obstacles on a single plot.
        """
        borderC = self._borderC_original
        border_bufferedC = self._border_bufferedC
        obstaclesC = self._obstaclesC_original
        obstacles_bufferedC = self._obstacles_bufferedC

        plt.figure(figsize=(10, 10))
        plt.title("Original and Buffered Shapes")
        ax = plt.gca()

        # Original border and obstacles
        ax.add_patch(MplPolygon(borderC, closed=True, edgecolor='none', facecolor='lightblue', label='Original Border'))
        for i, obs in enumerate(obstaclesC):
            ax.add_patch(MplPolygon(obs, closed=True, edgecolor='none', facecolor='white',
                                    label='Original Obstacle' if i == 0 else None))

        # Buffered border and obstacles
        ax.add_patch(MplPolygon(border_bufferedC, closed=True, edgecolor='red', linestyle='--',
                                facecolor='none', label='Buffered Border'))
        for i, obs in enumerate(obstacles_bufferedC):
            ax.add_patch(MplPolygon(obs, closed=True, edgecolor='black', linestyle='--',
                                    facecolor='none', label='Buffered Obstacle' if i == 0 else None))

        ax.set_aspect('equal')

        # Axis limits
        all_x = np.concatenate([
            borderC[:, 0], border_bufferedC[:, 0],
            *[obs[:, 0] for obs in obstaclesC],
            *[obs[:, 0] for obs in obstacles_bufferedC]
        ])
        all_y = np.concatenate([
            borderC[:, 1], border_bufferedC[:, 1],
            *[obs[:, 1] for obs in obstaclesC],
            *[obs[:, 1] for obs in obstacles_bufferedC]
        ])
        x_pad = 0.05 * (all_x.max() - all_x.min())
        y_pad = 0.05 * (all_y.max() - all_y.min())
        ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
        ax.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)

        ax.set_axis_off()

        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def _from_coordinates(self, turbinesC, substationsC, borderC, obstaclesC, name, handle, buffer_dist):
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

            warning('⚠️ Obstacles are given while no border coordinate is defined, optiwindnet is creating borders based on turbine and obstacle coordinates.')
            
            all_points = [turbinesC, substationsC]
            all_points_flat = np.vstack(all_points)
            hull = MultiPoint([tuple(p) for p in all_points_flat]).convex_hull
            borderC = np.array(hull.exterior.coords[:-1])  # drop closing point
            border_subtraction_verbose = False

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
            border_polygon = self.expand_polygon_safely(border_polygon, buffer_dist)
            borderC = np.array(border_polygon.exterior.coords[:-1])

            # obstacles
            shrunk_obstaclesC = []
            shrunk_obstaclesC_including_removed = []
            for i, obs in enumerate(obstaclesC):

                obs_poly = Polygon(obs)
                obs_bufferedC = self.shrink_polygon_safely(obs_poly, buffer_dist, i)

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
    
        
    @classmethod
    def from_yaml(cls, filepath: str, **kwargs):
        """Creates a WindFarmNetwork instance from a YAML file."""
        if not isinstance(filepath, str):
            raise TypeError("Filepath must be a string")

        L = L_from_yaml(filepath)
        return cls(L=L, **kwargs)
    
    @classmethod
    def from_pbf(cls, filepath: str, **kwargs):
        """Creates a WindFarmNetwork instance from a pbf file."""
        if not isinstance(filepath, str):
            error("Filepath must be a string")

        L = L_from_pbf(filepath)
        return cls(L=L, **kwargs)
        
    @classmethod
    def from_windIO(cls, filepath: str, **kwargs):
        """Creates a WindFarmNetwork instance from WindIO yaml file."""
        if not isinstance(filepath, str):
            raise TypeError("Filepath must be a string")
        
        fpath = Path(filepath)

        yaml.add_constructor(
            "!include", yaml_include.Constructor(base_dir='data'))
        
        with open(fpath, 'r') as f:
            system = yaml.full_load(f)

        coords = system['wind_farm']['layouts']['initial_layout']['coordinates']
        terminalC = np.c_[coords['x'], coords['y']]
        coords = system['wind_farm']['electrical_substations']['coordinates']
        rootC = np.c_[coords['x'], coords['y']]
        coords = system['site']['boundaries']['polygons'][0]
        borderC = np.c_[coords['x'], coords['y']]

        T = terminalC.shape[0]
        R = rootC.shape[0]
        name_tokens = fpath.stem.split('_')

        L = L_from_site(R=R, T=T, VertexC=np.vstack((terminalC, borderC, rootC)),
                        border=np.arange(T, T + borderC.shape[0]),
                        name=' '.join(name_tokens),
                        handle=f"{name_tokens[0].lower()}_{name_tokens[1][:4].lower()}_{name_tokens[2][:3].lower()}")

        return cls(L=L, **kwargs)

    def _repr_svg_(self):
        # Give the WindFarmNetwork object a SVG representation (it is picked up automatically by the notebook)
        return svgplot(self.G)._repr_svg_()
    
    def plot(self, *args, **kwargs):
        return gplot(self.G, *args, **kwargs)
    
    def plot_location(self, **kwargs):
        """Plots the wind farm network graph."""
        return gplot(self.L, **kwargs)
    
    def plot_available_links(self, **kwargs):
        """Plots the wind farm network graph."""
        return gplot(self.A, **kwargs)
    
    def plot_navigation_mesh(self, **kwargs):
        """Plots the wind farm network graph."""
        return pplot(self.P, self.A, **kwargs)
    
    def plot_selected_links(self, **kwargs):
        """Plots the wind farm network graph."""
        G_tentative = G_from_S(self.S, self.A)
        assign_cables(G_tentative, self.cables)
        return gplot(G_tentative, **kwargs)

    def get_network(self):
        """Returns the network edges with cable data."""
        net_graph = self.G.edges(data=True)
        net = list(net_graph)  # Keep it as a list of tuples
        return net
    
    def terse_links(self):
        '''Returns S links'''
        R, T = (self.S.graph[k] for k in 'RT')
        terse = np.empty(T, dtype=int)

        for u, v, reverse in self.S.edges(data='reverse'):
            if reverse is None:
                error('reverse must not be None')
            u, v = (u, v) if u < v else (v, u)
            i, target = (u, v) if reverse else (v, u)
            terse[i] = target

        return terse

    def update_from_terse_links(self, terse_links: np.ndarray, turbinesC=None, substationsC=None) -> None:
        '''Undate class from terse links'''

        # --- Added block: check input format ---
        terse_links = np.asarray(terse_links)

        if not np.issubdtype(terse_links.dtype, np.integer):
            raise ValueError(
                f"terse_links must be an array of integers. Got {terse_links.dtype} instead.\n"
                f"Hint: You can fix it by doing terse_links = [int(x) for x in terse_links]."
            )

        if terse_links.ndim != 1:
            raise ValueError(
                f"terse_links must be a 1D array. Got shape {terse_links.shape} instead."
            )
        # If new coordinates are provided, update them
        if turbinesC is not None:
            self.L.graph['VertexC'][:turbinesC.shape[0], :] = turbinesC
    
        if substationsC is not None:
            self.L.graph['VertexC'][-substationsC.shape[0]:, :] = substationsC
            
        if turbinesC is not None or substationsC is not None:
            self.P, self.A = make_planar_embedding(self.L)

        self.S.remove_edges_from(list(self.S.edges()))
        for i, j in enumerate(terse_links):
            self.S.add_edge(i, j)

        calcload(self.S)

        G_tentative = G_from_S(self.S, self.A)

        self.G = PathFinder(G_tentative, planar=self.P, A=self.A).create_detours()

        assign_cables(self.G, self.cables)

        return self.G

    def get_network_array(self):
        """Returns the network edges with cable data."""
        network_array_type = np.dtype([
            ('src', int),
            ('tgt', int),
            ('length', float),
            ('load', float),
            ('reverse', bool),
            ('cable', int),
            ('cost', float),
        ])

        def iter_edges(G, keys):
            for s, t, edgeD in G.edges(data=True):
                yield s, t, *(edgeD[key] for key in keys)


        network_array = np.fromiter(iter_edges(self.G, network_array_type.names[2:]),
                                dtype=network_array_type, count=self.G.number_of_edges())
        return network_array

    def _set_coordinates(self, turbinesC, substationsC):
        """Updates the coordinates of turbines and substationsC."""

        info('wfn._set_coordinates is not checking for feasiblity')

        if not hasattr(self.L, 'graph') or 'VertexC' not in self.L.graph:
            error("Graph L does not contain 'VertexC' attribute.")
        
        # Update coordinates
        if turbinesC is not None:
            self.L.graph['VertexC'][:turbinesC.shape[0], :] = turbinesC
            self.G.graph['VertexC'][:turbinesC.shape[0], :] = turbinesC
        
        if substationsC is not None:
            self.L.graph['VertexC'][-substationsC.shape[0]:, :] = substationsC
            self.G.graph['VertexC'][-substationsC.shape[0]:, :] = substationsC

        # Update length
        VertexC = self.G.graph['VertexC']
        for u, v, data in self.G.edges(data=True):
            coord_u = VertexC[u, :]
            coord_v = VertexC[v, :]
            data['length'] = np.linalg.norm(np.array(coord_u) - np.array(coord_v))
        
        # Update cost
        assign_cables(self.G, self.cables)


    def gradient(self, turbinesC=None, substationsC=None, gradient_type='length'):
        """
        Calculate the gradient of the length and cost of cable with respect to the positions of the nodes.
        """
        if gradient_type.lower() not in ['cost', 'length']:
            raise ValueError("gradient_type should be either 'cost' or 'length'")         

        if turbinesC is None and substationsC is None:
            G = self.G
        else:
            info('wfn.gradient is not checking for the feasibility of the layout with new coordinates!')
            G = copy.deepcopy(self.G)
            if turbinesC is not None:
                G.graph['VertexC'][:turbinesC.shape[0], :] = turbinesC

            if substationsC is not None:
                G.graph['VertexC'][-substationsC.shape[0]:, :] = substationsC

        vertexC = G.graph['VertexC']
        R = G.graph['R']
        T = G.graph['T']
        N = len(vertexC)
        gradients = np.zeros((N, 2))

        fnT = G.graph.get('fnT')

        for u, v in G.edges():
            if fnT is not None:
                u = fnT[u]
                v = fnT[v]

            vec = vertexC[u] - vertexC[v]
            norm = np.hypot(*vec)

            if norm < 1e-12:
                continue  # Skip zero-length edges

            gradinc = vec / norm

            if gradient_type.lower() == 'cost':
                gradinc *= G.graph['cables'][G[u][v]['cable']][1]

            gradients[u] += gradinc
            gradients[v] -= gradinc

        # wind turbines
        gradients_wt = gradients[:T]
        # substations
        gradients_ss = gradients[N - R:]

        return gradients_wt, gradients_ss

    def optimize(self, turbinesC=None, substationsC=None, router=None, verbose=None):

        if router is None:
            router = self.router
        else:
            self.router = router     
            
        # If new coordinates are provided, update them
        if turbinesC is not None:
            self.L.graph['VertexC'][:turbinesC.shape[0], :] = turbinesC
    
        if substationsC is not None:
            self.L.graph['VertexC'][-substationsC.shape[0]:, :] = substationsC
            
        if turbinesC is not None or substationsC is not None:
            self.P, self.A = make_planar_embedding(self.L)
        
        if turbinesC is None:
            turbinesC = self.L.graph['VertexC'][:self.L.graph['T'], :]

        if substationsC is None:
            substationsC = substationsC or self.L.graph['VertexC'][-self.L.graph['R']:, :]

        S, G = router(A=self.A, P=self.P, S=self.S, turbinesC=turbinesC, substationsC=substationsC, cables=self.cables, cables_capacity=self.cables_capacity, verbose=verbose)
        self.S = S
        self.G = G

        terse_links = self.terse_links()
        return terse_links

class OptiWindNetSolver(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def optimize(self, turbinesC=None, substationsC=None, verbose=None, **kwargs):
        """
        Perform cable layout optimization. Must be implemented by subclasses.
        """
        pass

    def __call__(self, turbinesC=None, substationsC=None, verbose=False, **kwargs):
        """Make the instance callable, calling optimize() internally."""
        return self.optimize(turbinesC=turbinesC, substationsC=substationsC, verbose=verbose, **kwargs)  


class Heuristic(OptiWindNetSolver):
    def __init__(self, solver='Esau_Williams', maxiter=10000, verbose=False, **kwargs):
        if solver not in ['Esau_Williams']:
            raise ValueError(
                f"{solver} is not among the supported Heuristic solvers. Choose among: ['Esau_Williams']."
            )

        # Call the base class initialization
        self.verbose = verbose
        self.solver = solver
        self.maxiter = maxiter

    def optimize(self, A, P, cables, cables_capacity, verbose=None, **kwargs):

        if verbose is None:
            verbose = self.verbose

        # optimizing
        if self.solver in ['Esau_Williams', 'EW']:
            S = EW_presolver(A, capacity=cables_capacity, maxiter=self.maxiter)
        else:
            pass
            
        G_tentative = G_from_S(S, A)

        G = PathFinder(G_tentative, planar=P, A=A).create_detours()

        assign_cables(G, cables)

        return S, G
    
class MetaHeuristic(OptiWindNetSolver):
    def __init__(self, time_limit, solver='HGS', gates_limit: int | None = None, max_iter=10, seed: int = 0, verbose=False, **kwargs):
        # Call the base class initialization
        self.time_limit = time_limit
        self.gates_limit = gates_limit
        self.solver = solver
        self.verbose = verbose
        self.max_iter = max_iter
        self.gates_limit = gates_limit
        self.seed = seed

    def optimize(self, A, P, cables, cables_capacity, verbose=None, **kwargs):
        # If verbose argument is None, use the value of self.verbose
        if verbose is None:
            verbose = self.verbose

        # optimizing
        if self.solver.lower() in ['hgs', 'hybrid genetic search', 'hybrid_genetic_search']:
            S = iterative_hgs_cvrp(A, capacity=cables_capacity, time_limit=self.time_limit, max_iter=self.max_iter, vehicles=self.gates_limit, seed=self.seed)
        else:
            raise ValueError(
                f"{self.solver} is not among the supported Meta-Heuristic solvers. Choose among: HGS.")
        
        G_tentative = G_from_S(S, A)

        G = PathFinder(G_tentative, planar=P, A=A).create_detours()

        assign_cables(G, cables)
        
        return S, G

class MILP(OptiWindNetSolver):
    def __init__(self, solver_name, time_limit, mip_gap, solver_options=None, model_options=None, verbose=False, **kwargs):
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.solver_name = solver_name
        self.solver_options = solver_options or {}
        self.model_options = model_options or ModelOptions()
        self.verbose = verbose

    def optimize(self, P, A, cables, cables_capacity, S_warm=None, verbose=None, **kwargs):

        if verbose is None:
            verbose = self.verbose
    
        # warm start
        if S_warm is not None:
            info('S is not None and the model is warmed up with the available S.')

        solver = solver_factory(self.solver_name)

        solver.set_problem(P, A, cables_capacity, warmstart=S_warm, model_options=self.model_options)
        
        solver.solve(time_limit=self.time_limit, mip_gap=self.mip_gap, options=self.solver_options, verbose=verbose)

        S, G = solver.get_solution()

        assign_cables(G, cables)
        
        return S, G



