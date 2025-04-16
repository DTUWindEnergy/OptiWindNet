###################
# OptiWindNet API #
###################

from abc import ABC, abstractmethod
from typing import Any, Mapping
from itertools import pairwise
from pathlib import Path
from shapely.geometry import Polygon, MultiPoint, MultiPolygon
import copy

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
from optiwindnet.interarraylib import G_from_S, calcload, as_normalized

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

    def __init__(self, turbinesC=None, substationsC=None,
                 cables=None, borderC=None, obstaclesC=None,
                 name='', handle='', L=None, router=None,
                 verbose=False, **kwargs):

        self.verbose = verbose

        # Default router if none provided
        if router is None:
            router = Heuristic(solver='Esau_Williams')
        self.router = router
        cables_array = np.array(cables)
        if cables is None:
            if verbose:
                print("WARNING: No cable data provided. Defaulting to cables = [(10, 1)]")
            cables = [(10, 1)]

        elif isinstance(cables, int):
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
        if turbinesC is not None:
            L = self._from_coordinates(turbinesC, substationsC, borderC, obstaclesC, name, handle)
        self.L = L

        # Planar embedding
        self.P, self.A = make_planar_embedding(L)

        # Geometry inputs
        self.borderC = borderC
        self.obstaclesC = obstaclesC

        # Graph/network placeholders and status flags
        self.S = None
        self._G_tentative = None
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

                # If the obstacle is completely within the border (and not touching exterior), keep it
                if border_polygon.contains(obs_poly):
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
        return gplot(self._G_tentative, **kwargs)

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

    def G_from_terse_links(self, terse_links: np.ndarray, turbinesC=None, substationsC=None) -> None:
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

        self.G_tentative = G_from_S(self.S, self.A)

        self.G = PathFinder(self._G_tentative, planar=self.P, A=self.A).create_detours()

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
            self._G_tentative.graph['VertexC'][:turbinesC.shape[0], :] = turbinesC
            self.G.graph['VertexC'][:turbinesC.shape[0], :] = turbinesC
        
        if substationsC is not None:
            self.L.graph['VertexC'][-substationsC.shape[0]:, :] = substationsC
            self._G_tentative.graph['VertexC'][-substationsC.shape[0]:, :] = substationsC
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
        if router is None:
            raise ValueError(
                        "To run the optimization, a router must be initialized. "
                        "This can be done either during the creation of the WFN object "
                        "or via the `optimize` method of the WFN object.")
        
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

        S, G_tentative, G = router(A=self.A, P=self.P, S=self.S, turbinesC=turbinesC, substationsC=substationsC, cables=self.cables, cables_capacity=self.cables_capacity, verbose=verbose)
        self.S = S
        self._G_tentative = G_tentative
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

    def __call__(self, turbinesC=None, substationsC=None, verbose=None, **kwargs):
        """Make the instance callable, calling optimize() internally."""
        return self.optimize(turbinesC=turbinesC, substationsC=substationsC, verbose=verbose, **kwargs)  


class Heuristic(OptiWindNetSolver):
    def __init__(self, solver='Esau_Williams', maxiter=10000, debug=False, verbose=False, **kwargs):
        if solver not in ['Esau_Williams']:
            raise ValueError(
                f"{solver} is not among the supported Heuristic solvers. Choose among: ['Esau_Williams']."
            )

        # Call the base class initialization
        self.verbose = verbose
        self.solver = solver
        self.maxiter = maxiter
        self.debug = debug

    def optimize(self, A, P, cables, cables_capacity, verbose=None, **kwargs):

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
        
        return S, G_tentative, G

class MILP(OptiWindNetSolver):
    def __init__(self, solver='ortools', solver_options=None, model_options=None, verbose=False, **kwargs):
        self.verbose = verbose
        self.solver = solver
        self.solver_options = solver_options or {}
        self.model_options = model_options or {}

    def optimize(self, A, P, cables, cables_capacity, S=None, verbose=None, **kwargs):
       
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
            
            # warm start
            if S is not None:
                if verbose:
                    print('S is not None and the model is warmed up with the available S.')

                ort.warmup_model(model, S)

            orter = ort.cp_model.CpSolver()
            # settings
            orter.parameters.max_time_in_seconds = self.solver_options.get("max_time_in_seconds", 40)
            orter.parameters.relative_gap_limit = self.solver_options.get("relative_gap_limi", 0.005)
            orter.parameters.num_workers = self.solver_options.get("num_workers", 8)
            orter.parameters.log_search_progress = verbose or self.solver_options.get("log_search_progress", False)
            if verbose or self.solver_options.get('log_callback', None)==print:
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
            if S is not None:
                if verbose:
                    print('S is not None and the model is warmed up with the available S.')

                omo.warmup_model(model, S)


            pyo_solver.options.update(self.solver_options)

            if verbose:
                print(f'Solving "{model.handle}": {{R={len(model.R)}, T={len(model.T)}, κ={model.k.value}}}\n')


            # Define solver-specific arguments for solving
            #######################################################################
            solver_args = {'tee': verbose or self.solver_options.get("tee", False)}


            if S is not None:
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



