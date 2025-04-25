###################
# OptiWindNet API #
###################

from abc import ABC, abstractmethod
from typing import Any, Mapping
from itertools import pairwise
from pathlib import Path
from shapely.geometry import Polygon, MultiPoint, MultiPolygon
import copy
from matplotlib.path import Path
import numpy as np
import yaml
import yaml_include

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
ModelOptions.help()

#################################################
# To Do: if face error do a simple optimization #
#################################################


class WindFarmNetwork:
    """
    Represents a wind farm electrical network, capable of processing 
    layout data from different formats and computing network properties.
    """

    def __init__(self, cables, turbinesC=None, substationsC=None,
                 borderC=None, obstaclesC=None,
                 name='', handle='', L=None, router=None,
                 verbose=False, **kwargs):

        self.verbose = verbose

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
            raise ValueError(f"Invalid cable values: {cables}")


        self.cables = cables
        self.cables_capacity = max(c[0] for c in cables)

        # Load layout from coordinates if turbinesC provided
        if turbinesC is not None and substationsC is not None:
            L = self._from_coordinates(turbinesC, substationsC, borderC, obstaclesC, name, handle)
        elif L is None:
            raise ValueError('Both turbinesC and substationsC must be provided')
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

    
    def _from_coordinates(self, turbinesC, substationsC, borderC, obstaclesC, name, handle):
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

            if self.verbose:
                print('WARNING: Obstacles are given while no border coordinate is defined. The tool is creating borders based on turbine and obstacle coordinates')
            
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

            for obs in obstaclesC:
                obs_poly = Polygon(obs)
                intersection = border_polygon.boundary.intersection(obs_poly)

                # If the obstacle is completely within the border, keep it
                if border_polygon.contains(obs_poly) and intersection.length == 0:
                    remaining_obstaclesC.append(obs)
                else:
                    # Subtract this obstacle from the border
                    new_borderC = border_polygon.difference(obs_poly)

                    if new_borderC.is_empty:
                        raise ValueError("Obstacle subtraction resulted in an empty border — check your geometry.")

                    if self.verbose and border_subtraction_verbose:
                        print("WARNING: At least one obstacle intersects/touches the border. The border is redefined to exclude those obstacles.")
                        border_subtraction_verbose = False

                    # If the subtraction results in multiple pieces (MultiPolygon), raise error
                    if isinstance(new_borderC, MultiPolygon):
                        raise ValueError("Obstacle subtraction resulted in multiple pieces (MultiPolygon) — check your geometry.")
                    else:
                        border_polygon = new_borderC

            # Update the border as a NumPy array of exterior coordinates
            borderC = np.array(border_polygon.exterior.coords[:-1])

            # Update obstacles (only those fully contained are kept)
            obstaclesC = remaining_obstaclesC

        # check_turbine_locations(border, obstacles, turbines):
        border_path = Path(borderC)
        # Border path, with tolerance for edge inclusion
        in_border_neg = border_path.contains_points(turbinesC, radius=-1e1)
        in_border_pos = border_path.contains_points(turbinesC, radius=1e1)
        in_border = in_border_neg | in_border_pos

        # Check if any turbine is outside the border
        if not np.all(in_border):
            outside_idx = np.where(~in_border)[0]
            raise ValueError(f"Turbines at indices {outside_idx} are outside the border!")

        for i, obs in enumerate(obstaclesC):
            obs_path = Path(obs)
            in_obstacle = obs_path.contains_points(turbinesC, radius=-1e-10)
            if np.any(in_obstacle):
                inside_idx = np.where(in_obstacle)[0]
                raise ValueError(f"Turbines at indices {inside_idx} are inside obstacle {i}!")

                        
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
            raise TypeError("Filepath must be a string")

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
                raise ValueError('reverse must not be None')
            u, v = (u, v) if u < v else (v, u)
            i, target = (u, v) if reverse else (v, u)
            terse[i] = target

        return terse

    def update_from_terse_links(self, terse_links: np.ndarray, turbinesC=None, substationsC=None) -> None:
        '''Rebuilds G from terse links'''
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

    def _set_coordinates(self, turbinesC, substationsC, verbose=None):
        """Updates the coordinates of turbines and substationsC."""
        if verbose is None:
            verbose = self.verbose

        if verbose:
            print('WARNING: wfn._set_coordinates is not checking for feasiblity')

        if not hasattr(self.L, 'graph') or 'VertexC' not in self.L.graph:
            raise ValueError("Graph L does not contain 'VertexC' attribute.")
        
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


    def gradient(self, turbinesC=None, substationsC=None, verbose=None, gradient_type='cost'):
        """
        Calculate the gradient of the length and cost of cable with respect to the positions of the nodes.
        """
        if gradient_type.lower() not in ['cost', 'length']:
            raise ValueError("gradient_type should be either 'cost' or 'length'")
        
        # If verbose argument is None, use the value of self.verbose
        if verbose is None:
            verbose = self.verbose           

        if turbinesC is None and substationsC is None:
            G = self.G
        else:
            if verbose:
                print('WARNIMG: gradient is not checking for the feasibility of the layout with new coordinates!')
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

        for u, v in G.edges():
            vec = vertexC[u] - vertexC[v]
            norm = np.hypot(*vec)

            if norm < 1e-12:
                continue  # Skip zero-length edges

            gradinc = vec / norm

            if gradient_type.lower() == 'cost':
                gradinc *= G.graph['cables'][G[u][v]['cable']][1]

            gradients[u] += gradinc
            gradients[v] -= gradinc

            ####################################
            # To Do: check fnt for detour nodes
            ####################################

        # wind turbines
        gradients_wt = gradients[:T]
        # substations
        gradients_ss = gradients[N - R:]

        return gradients_wt, gradients_ss

    def optimize(self, turbinesC=None, substationsC=None, verbose=None, router=None):
        if router is not None:
            self.router = router
        
        router = self.router  # Use provided router or the existing one in the class
        # if router is None:
        #     raise ValueError(
        #                 "To run the optimization, a router must be initialized. "
        #                 "This can be done either during the creation of the WFN object "
        #                 "or via the `optimize` method of the WFN object.")
        
        if verbose is None:
            verbose = self.verbose

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

        terse_links = 0 # self.terse_links()
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

    def __call__(self, turbinesC=None, substationsC=None, verbose=None, **kwargs):
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
    def __init__(self, solver_name='ortools', solver_options=None, model_options=None, verbose=False, **kwargs):
        self.verbose = verbose
        self.solver_name = solver_name
        self.solver_options = solver_options or {}
        self.model_options = model_options or {}

    def optimize(self, A, P, cables, cables_capacity, S_warm=None, verbose=None, **kwargs):

        if verbose is None:
            verbose = self.verbose
      
        # warm start
        if S_warm is not None:
            if verbose:
                print('S is not None and the model is warmed up with the available S.')

        solver = solver_factory(self.solver_name)

        solver.set_problem(P, A, cables_capacity, warmstart=S_warm, model_options=self.model_options)
        
        solver.solve(time_limit=15, mip_gap=0.01, options=self.solver_options)

        S, G = solver.get_solution()
        
        return S, G



