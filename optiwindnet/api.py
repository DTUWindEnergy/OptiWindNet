###################
# OptiWindNet API #
###################

from abc import ABC, abstractmethod
from typing import Any, Mapping
from itertools import pairwise
from pathlib import Path

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
from optiwindnet.mesh import make_planar_embedding
from optiwindnet.pathfinding import PathFinder
from optiwindnet.interface import assign_cables
from optiwindnet.interarraylib import G_from_S, calcload

# Heuristics
from optiwindnet.heuristics import EW_presolver

# Metaheuristics
from optiwindnet.baselines.hgs import iterative_hgs_cvrp

# MILP
from optiwindnet.MILP import ortools as ort
import optiwindnet.MILP.pyomo as omo
from pyomo.contrib.appsi.solvers import Highs
from pyomo import environ as pyo


#################################################
# To Do: if face error do a simple optimization #
#################################################

class WindFarmNetwork:
    """
    Represents a wind farm electrical network, capable of processing 
    layout data from different formats and computing network properties.
    """

    def __init__(self, turbines=None, substations=None,
                 cables=None, border=None, obstacles=None,
                 name='', handle='', L=None, router=None,
                 verbose=False, **kwargs):

        self.verbose = verbose

        # Default router if none provided
        if router is None:
            router = Heuristic(solver='Esau_Williams')
        self.router = router

        # Handle cable formats
        if cables is None:
            if verbose:
                print("WARNING: No cable data provided. Defaulting to cables = [(10, 1)]")
            cables = [(10, 1)]
        elif isinstance(cables, int):
            cables = [(cables, 1)]
        elif isinstance(cables, (list, tuple, np.ndarray)):
            if all(isinstance(c, int) for c in cables):
                cables = [(c, 1) for c in cables]
            elif all(isinstance(c, (list, tuple, np.ndarray)) and len(c) == 2 for c in cables):
                cables = [tuple(c) for c in cables]
            else:
                raise ValueError(f"Invalid cable format: {cables}")
        else:
            raise ValueError(f"Invalid cable format: {cables}")

        self.cables = cables
        self.cables_capacity = max(c[0] for c in cables)

        # Load layout from coordinates if turbines provided
        if turbines is not None:
            L = self._from_coordinates(turbines, substations, border, obstacles, name, handle)
        self.L = L

        # Planar embedding
        self._P, self._A = make_planar_embedding(L)

        # Geometry inputs
        self.border = border
        self.obstacles = obstacles

        # Graph/network placeholders and status flags
        self._S = None
        self._G_tentative = None
        self.G = None
        self._A_updated = True
        self._P_updated = True
        self._S_updated = False
        self._G_updated = False


    def _update_planar_embedding(self):
        """Update the planar embedding (P and A) if marked stale."""
        if not (self._P_updated and self._A_updated):
            self._P, self._A = make_planar_embedding(self.L)
            self._P_updated = self._A_updated = True


    def terse_links(self):
        '''Returns S links'''
        _, T = (self.S.graph[k] for k in 'RT')
        terse = np.empty(T, dtype=int)


        for u, v, in self.S.edges:
            u, v = (u, v) if u < v else (v, u)
            i, target = (v, u)
            terse[i] = target

        return terse

    def G_from_terse_links(self, terse_links: np.ndarray, turbines=None, substations=None) -> None:
        '''Rebuilds G from terse links'''
        if turbines is not None or substations is not None:
            self._set_coordinates(turbines=turbines, substations=substations, verbose=False)
            self._P, self._A = make_planar_embedding(self.L)

        self.S.remove_edges_from(list(self.S.edges()))
        for i, j in enumerate(terse_links):
            self.S.add_edge(i, j, load=0)
        
        calcload(self.S)

        self.G_tentative = G_from_S(self.S, self.A)

        self.G = PathFinder(self._G_tentative, planar=self.P, A=self.A).create_detours()

        assign_cables(self.G, self.cables)

    @property
    def A(self):
        """Lazy update of A when accessed."""
        self._update_planar_embedding()
        return self._A
    
    @A.setter
    def A(self, value):
        self._A = value

    @property
    def P(self):
        """Lazy update of P when accessed."""
        self._update_planar_embedding()
        return self._P
    
    @P.setter
    def P(self, value):
        self._P = value

    @property
    def S(self):
        if not self._S_updated and self.verbose:
            print('S is not updated')
        return self._S
    
    @S.setter
    def S(self, value):
        self._S = value

    @property
    def G(self):
        if not self._G_updated and self.verbose:
            print('G is not updated')
        return self._G

    @G.setter
    def G(self, value):
        self._G = value
    
    def _from_coordinates(self, turbines, substations, border, obstacles, name, handle):
        """Constructs a site graph from coordinate-based inputs."""

        from shapely.geometry import Polygon, MultiPoint, MultiPolygon
        
        border_subtraction_verbose = True

        R = substations.shape[0]
        T = turbines.shape[0]

        if border is None:
            if obstacles is None:
                vertex_coords = np.vstack((turbines, substations))
                return L_from_site(
                    R=R,
                    T=T,
                    B=0,
                    name=name,
                    handle=handle,
                    VertexC=vertex_coords
                )

            if self.verbose:
                print('WARNING: Obstacles are given while no border is defined. The tool is creating borders based on turbine and obstacle coordinates')
            
            all_points = [turbines, substations]
            all_points_flat = np.vstack(all_points)
            hull = MultiPoint([tuple(p) for p in all_points_flat]).convex_hull
            border = np.array(hull.exterior.coords[:-1])  # drop closing point
            border_subtraction_verbose = False

        if obstacles is None:
            obstacles = []
        else:

            # Start with the original border polygon
            border_polygon = Polygon(border)
            remaining_obstacles = []

            for obs in obstacles:
                obs_poly = Polygon(obs)

                # If the obstacle is completely within the border (and not touching exterior), keep it
                if border_polygon.contains(obs_poly):
                    remaining_obstacles.append(obs)
                else:
                    # Subtract this obstacle from the border
                    new_border = border_polygon.difference(obs_poly)

                    if new_border.is_empty:
                        raise ValueError("Obstacle subtraction resulted in an empty border — check your geometry.")

                    if self.verbose and border_subtraction_verbose:
                        print("WARNING: At least one obstacle intersects/touches the border. The border is redefined to exclude those obstacles.")
                        border_subtraction_verbose = False

                    # If the subtraction results in multiple pieces (MultiPolygon), keep the largest one
                    if isinstance(new_border, MultiPolygon):
                        border_polygon = max(new_border.geoms, key=lambda g: g.area)
                    else:
                        border_polygon = new_border

            # Update the border as a NumPy array of exterior coordinates
            border = np.array(border_polygon.exterior.coords[:-1])

            # Update obstacles (only those fully contained are kept)
            obstacles = remaining_obstacles
                        
        border_sizes = np.array([border.shape[0]] + [obs.shape[0] for obs in obstacles])
        B = border_sizes.sum()
        obstacle_start_idxs = np.cumsum(border_sizes) + T

        border_range = np.arange(T, T + border.shape[0])
        obstacle_ranges = [np.arange(start, end) for start, end in pairwise(obstacle_start_idxs)]

        vertex_coords = np.vstack((turbines, border, *obstacles, substations))

        return L_from_site(
            R=R,
            T=T,
            B=B,
            border=border_range,
            obstacles=obstacle_ranges,
            name=name,
            handle=handle,
            VertexC=vertex_coords
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
        return gplot(self._G_tentative, **kwargs)

    def get_network(self):
        """Returns the network edges with cable data."""
        net_graph = self.G.edges(data=True)
        net = list(net_graph)  # Keep it as a list of tuples
        return net
    
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
                            
    def cost(self):
        """Returns the total cost of the network."""
        return self.G.size(weight="cost")

    def length(self):
        """Returns the total cable length of the network."""
        return self.G.size(weight="length")

    def _set_coordinates(self, turbines, substations, verbose=None):
        """Updates the coordinates of turbines and substations."""
        if verbose is None:
            verbose = self.verbose
        self._G_updated = True

        if verbose:
            print('WARNING: wfn._set_coordinates is not checking for feasiblity')

        if not hasattr(self.L, 'graph') or 'VertexC' not in self.L.graph:
            raise ValueError("Graph L does not contain 'VertexC' attribute.")
        
        # Update coordinates
        if turbines is not None:
            self.L.graph['VertexC'][:turbines.shape[0], :] = turbines
            self._G_tentative.graph['VertexC'][:turbines.shape[0], :] = turbines
            self.G.graph['VertexC'][:turbines.shape[0], :] = turbines
        
        if substations is not None:
            self.L.graph['VertexC'][-substations.shape[0]:, :] = substations
            self._G_tentative.graph['VertexC'][-substations.shape[0]:, :] = substations
            self.G.graph['VertexC'][-substations.shape[0]:, :] = substations

        # Update length
        VertexC = self.G.graph['VertexC']
        for u, v, data in self.G.edges(data=True):
            coord_u = VertexC[u, :]
            coord_v = VertexC[v, :]
            data['length'] = np.linalg.norm(np.array(coord_u) - np.array(coord_v))
        
        # Update cost
        assign_cables(self.G, self.cables)

        self._A_updated = False
        self._P_updated = False
        self._S_updated = False


    def gradient(self, turbines=None, substations=None, network_tree=None, verbose=None, gradient_type='cost'):
        """
        Calculate the gradient of the length and cost of cable with respect to the positions of the nodes.
        """
        if gradient_type.lower() not in ['cost', 'length']:
            raise ValueError("gradient_type should be either 'cost' or 'length'")
        # If verbose argument is None, use the value of self.verbose
        if verbose is None:
            verbose = self.verbose

        if turbines is not None or substations is not None:
            self.wfn._set_coordinates(turbines=turbines, substations=substations)

        if network_tree is not None:
            self.wfn.set_network(network_tree=network_tree)

        G = self.G
        vertices = G.graph['VertexC']
        R = G.graph['R']
        T = G.graph['T']
        N = len(vertices)
        gradients = np.zeros((N, 2))

        for u, v in G.edges():
            vec = vertices[u] - vertices[v]
            norm = np.hypot(*vec)

            if norm < 1e-12:
                continue  # Skip zero-length edges

            gradinc = vec / norm

            if gradient_type.lower() == 'cost':
                gradinc *= G.graph['cables'][G[u][v]['cable']][2]

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


    def optimize(self, turbines=None, substations=None, verbose=None, router=None):
        router = router or self.router  # Use provided router or the existing one in the class
        if router is None:
            raise ValueError(
                        "To run the optimization, a router must be initialized. "
                        "This can be done either during the creation of the WFN object "
                        "or via the `optimize` method of the WFN object.")
        
        if verbose is None:
            verbose = self.verbose

        # If new coordinates are provided, update them
        if turbines is not None or substations is not None:
            self._set_coordinates(turbines=turbines, substations=substations, verbose=False)
            self._P, self._A = make_planar_embedding(self.L)
        
        if turbines is None:
            turbines = self.L.graph['VertexC'][:self.L.graph['T'], :]

        if substations is None:
            substations = substations or self.L.graph['VertexC'][-self.L.graph['R']:, :]

        self._A_updated = True
        self._P_updated = True
        self._S_updated = True
        self._G_updated = True
        S, G_tentative, G = router(A=self.A, P=self.P, S=self.S, turbines=turbines, substations=substations, cables=self.cables, cables_capacity=self.cables_capacity, verbose=verbose)
        self.S = S
        self._G_tentative = G_tentative
        self.G = G

        network_array = self.get_network_array()
        return network_array


class OptiWindNetSolver(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def optimize(self, turbines=None, substations=None, verbose=None, **kwargs):
        """
        Perform cable layout optimization. Must be implemented by subclasses.
        """
        pass

    def __call__(self, turbines=None, substations=None, verbose=None, **kwargs):
        """Make the instance callable, calling optimize() internally."""
        return self.optimize(turbines=turbines, substations=substations, verbose=verbose, **kwargs)  


class Heuristic(OptiWindNetSolver):
    def __init__(self, solver='Esau_Williams', maxiter=10000, debug=False, verbose=None, **kwargs):
        if solver not in ['Esau_Williams']:
            raise ValueError(
                f"{solver} is not among the supported Heuristic solvers. Choose among: ['Esau_Williams']."
            )

        # Call the base class initialization
        self.verbose = verbose
        self.solver = solver
        self.maxiter = maxiter
        self.debug = debug

    def optimize(self, A, P, cables=None, cables_capacity=None, verbose=None, **kwargs):

        if verbose is None:
            verbose = self.verbose

        # optimizing
        if self.solver in ['Esau_Williams', 'EW']:
            S = EW_presolver(A, capacity=cables_capacity, maxiter=self.maxiter, debug=self.debug)
        else:
            pass
            
        G_tentative = G_from_S(S, A)

        G = PathFinder(G_tentative, planar=P, A=A).create_detours()

        assign_cables(G, cables)

        return S, G_tentative, G
    
class MetaHeuristic(OptiWindNetSolver):
    def __init__(self, time_limit, solver='HGS', gates_limit: int | None = None, max_iter=10, seed: int = 0, verbose=None, **kwargs):
        # Call the base class initialization
        self.time_limit = time_limit
        self.gates_limit = gates_limit
        self.solver = solver
        self.verbose = verbose
        self.max_iter = max_iter
        self.gates_limit = gates_limit
        self.seed = seed

    def optimize(self, A, P, cables=None, cables_capacity=None, verbose=None, **kwargs):
        # If verbose argument is None, use the value of self.verbose
        if verbose is None:
            verbose = self.verbose

        # optimizing
        if self.solver== 'HGS': # Hybrid Genetic Search
            S = iterative_hgs_cvrp(A, capacity=cables_capacity, time_limit=self.time_limit, max_iter=self.max_iter, vehicles=self.gates_limit, seed=self.seed)
        else:
            raise ValueError(
                f"{self.solver} is not among the supported Meta-Heuristic solvers. Choose among: HGS.")
        
        G_tentative = G_from_S(S, A)

        G = PathFinder(G_tentative, planar=P, A=A).create_detours()

        assign_cables(G, cables)
        
        return S, G_tentative, G

class MILP(OptiWindNetSolver):
    def __init__(self, solver='ortools', solver_options=None, model_options=None, verbose=None, **kwargs):
        self.verbose = verbose
        self.solver = solver
        self.solver_options = solver_options or {}
        self.model_options = model_options or {}

    def optimize(self, A, P, S=None, cables=None, cables_capacity=None, verbose=None, **kwargs):
        # If verbose argument is None, use the value of self.verbose
        if verbose is None:
            verbose = self.verbose

        # optimizing
        if self.solver == 'ortools':
            # initialize
            #orter = ort.CpSat()
            # set the model
            model = ort.make_min_length_model(
                        A,
                        cables_capacity,
                        gateXings_constraint=self.model_options.get("gateXing_constraint", False),
                        branching=self.model_options.get("branching", True),
                        gates_limit=self.model_options.get("gates_limit", False)
                    )

            orter = ort.cp_model.CpSolver()
            # settings
            orter.parameters.max_time_in_seconds = self.solver_options.get("max_time_in_seconds", 40)
            orter.parameters.relative_gap_limit = self.solver_options.get("relative_gap_limi", 0.005)
            orter.parameters.num_workers = self.solver_options.get("num_workers", 8)
            orter.parameters.log_search_progress = verbose
            orter.log_callback = print # required to get the log inside the notebook (goes only to console otherwise)

            ##########
            # result #
            ##########
            result = orter.solve(model)

            if verbose: # print info about the final solution
                gap = 1 - orter.BestObjectiveBound()/orter.ObjectiveValue()
                print('=================================================================',
                    #orter.ResponseStats(),  # uncomment if orter.parameters.log_search_progress == False
                    f"\nbest solution's strategy: {orter.SolutionInfo()}",
                    f'\ngap: {100*gap:.1f}%')

            S = ort.S_from_solution(model, orter, result)
  

        elif self.solver in ['cplex', 'cbc', 'gurobi', 'highs', 'scip']:
            ##############
            # initialize #
            ##############
            # Define solver and solver_io mapping for special cases
            solver_mapping = {
                'highs': 'appsi_highs',
                }
            solver_io_mapping ={
                'gurobi': 'python',
                'cplex': 'python',
                # add more if solver_io is not None
            }

            # Get the solver name, falling back to self.solver if not in mapping
            solver_name = solver_mapping.get(self.solver, self.solver)
            solver_io = solver_io_mapping.get(self.solver, None)

            # Initialize solver with or without solver_io
            pyo_solver = pyo.SolverFactory(solver_name, solver_io=solver_io) if solver_io else pyo.SolverFactory(solver_name)
            
            if verbose:
                pyo_solver.available(), type(pyo_solver)

            model = omo.make_min_length_model(
                A, cables_capacity,      
                gateXings_constraint=self.model_options.get("gateXing_constraint", False),
                branching=self.model_options.get("branching", True), # if branching is false 
                gates_limit=self.model_options.get("gates_limit", False) # if ew_pre_solver does not fit is True (warm start it with metaheauristic)
                )
            
            # warm start
            S_updated = True
            if S is not None and S_updated:
                if verbose:
                    print('S is not None and the model is warmed up with the available S.')
            else:
                if verbose:
                    print('S is None or not updated. Esau-Williams Heuristic is used for warmstarting the MILP solver.')
                S = EW_presolver(A, capacity=cables_capacity)

            omo.warmup_model(model, S)


            pyo_solver.options.update(self.solver_options)

            if verbose:
                print(f'Solving "{model.handle}": {{R={len(model.R)}, T={len(model.T)}, κ={model.k.value}}}\n')


            # Define solver-specific arguments for solving
            #######################################################################
            solver_args = {'tee': self.solver_options.get("tee", True)}

            if self.solver in ['gurobi', 'cbc']:
                solver_args['warmstart'] = model.warmed_by
            if self.solver in ['cplex']:
                solver_args['warmstart'] = True

            ##########
            # result #
            ##########
            result = pyo_solver.solve(model, **solver_args)

            S = omo.S_from_solution(model, pyo_solver, result)
           
        else:
            raise ValueError(
                f"{self.solver} is not among the supported MILP solvers. Choose among: ortools, gurobi, cplex, highs, scip, cbc.")

        G_tentative = G_from_S(S, A)

        if self.model_options.get("gateXing_constraint", False):
            G = G_tentative
        else:
            G = PathFinder(G_tentative, planar=P, A=A).create_detours()

        assign_cables(G, cables)
        
        return S, G_tentative, G



