# OptiWindNet API
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import yaml
#import yaml_include
from itertools import pairwise


# interarray
from interarray.svg import svgplot
from interarray.importer import L_from_yaml, L_from_pbf, L_from_site
from interarray.plotting import gplot
from interarray.mesh import make_planar_embedding
from interarray.pathfinding import PathFinder
from interarray.interface import assign_cables
from interarray.importer import load_repository
from interarray.interarraylib import G_from_S

# Heuristic
from interarray.heuristics import EW_presolver
# Metha-Heuristic
from interarray.baselines.hgs import iterative_hgs_cvrp


# MILP
from interarray.MILP import ortools as ort
from interarray.pathfinding import PathFinder
import interarray.MILP.pyomo as omo
from pyomo.contrib.appsi.solvers import Highs
from pyomo import environ as pyo

def process_coordinates(wt_x, wt_y, substations):
    try:
        # Convert wind turbine inputs to NumPy arrays with dtype=float.
        x = np.asarray(wt_x, dtype=float) if wt_x is not None else np.array([])
        y = np.asarray(wt_y, dtype=float) if wt_y is not None else np.array([])
    except Exception as e:
        raise ValueError("wt_x and wt_y must be iterables of numbers convertible to floats.") from e

    # Ensure the arrays are one-dimensional.
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("wt_x and wt_y must be one-dimensional sequences (lists, tuples, or 1D arrays).")

    # Check that both arrays have the same length.
    if x.shape[0] != y.shape[0]:
        raise ValueError("wt_x and wt_y must have the same number of elements.")

    # Combine the two arrays into a single (T,2) array.
    turbines = np.column_stack((x, y)) if x.size > 0 else np.empty((0, 2))

    # Substation coordinates processing
    if ss_x is not None and ss_y is not None:
        try:
            ss_x = np.asarray(ss_x, dtype=float)
            ss_y = np.asarray(ss_y, dtype=float)
        except Exception as e:
            raise ValueError("ss_x and ss_y must be iterables of numbers convertible to floats.") from e

        # Ensure the arrays are one-dimensional.
        if ss_x.ndim != 1 or ss_y.ndim != 1:
            raise ValueError("ss_x and ss_y must be one-dimensional sequences (lists, tuples, or 1D arrays).")

        # Check that both arrays have the same length.
        if ss_x.shape[0] != ss_y.shape[0]:
            raise ValueError("ss_x and ss_y must have the same number of elements.")

        # Combine the substation coordinates into a single (m,2) array.
        substations = np.column_stack((substations))
    else:
        substations = np.empty((0, 2))

    return turbines, substations


def update_wfn(wfn, turbines=None, substations=None):
    
    # update wfn
    wfn.L['VertexC'][:wfn.turbines.shape[0], :] = turbines or wfn.turbines
    wfn.L['VertexC'][-wfn.substations.shape[0]:, :] = substations or wfn.substations


def turbines_per_cable(cable_cross_section, turbine_power, voltage, efficiency=0.95):
    """
    Calculate how many wind turbines a given cable can handle.

    Parameters:
    - cable_cross_section (float): Cable cross-section in mm².
    - turbine_power (float): Rated power of a single wind turbine in MW.
    - voltage (float): Operating voltage in Volts.
    - efficiency (float): Efficiency/power factor (default: 0.95).

    Returns:
    - int: Maximum number of turbines the cable can handle.
    """

    # Lookup table for current capacity (Ampacity) based on cross-section
    cable_ampacity_lookup = {
        50: 140,  # 50 mm² -> 140A
        95: 220,  # 95 mm² -> 220A
        150: 300, # 150 mm² -> 300A
        240: 400, # 240 mm² -> 400A
        400: 550, # 400 mm² -> 550A
    }

    # Get max cable current capacity (A)
    if cable_cross_section not in cable_ampacity_lookup:
        raise ValueError("Cable cross-section not found in lookup table!")

    I_cable_max = cable_ampacity_lookup[cable_cross_section]  # Amps

    # Convert turbine power from MW to Watts
    P_turbine = turbine_power * 1e6  # MW to W

    # Calculate max number of turbines the cable can handle
    N_turbines = (I_cable_max * voltage * efficiency) / P_turbine

    return int(N_turbines)  # Return integer number of turbines

class WindFarmNetwork():
    """
    Represents a wind farm electrical network, capable of processing 
    layout data from different formats and computing network properties.
    """

    def __init__(self, turbines=None, substations=None,
                 cables=None, border=None, obstacles=None, L=None, S=None, 
                 G_tentative=None, G=None, **kwargs):

        #
        if turbines is not None:
            L = self._from_coordinates(turbines, substations, border, obstacles)
        self.L = L

        # Compute the planar embedding
        self.P, self.A = make_planar_embedding(L)

        # Assign additional attributes
        self.S = S
        self.G_tentative = G_tentative
        self.G = G
        self.border = border
        self.obstacles = obstacles
        self.cables = cables

        if self.cables is not None:
            self.cables_capacity = max(cable[1] for cable in cables) # np.max(self.cables[:, 1])

        self.add_aux_methods()
    
    def _from_coordinates(self, turbines, substations, border, obstacles):
        """Handles input format from coordinates."""
        R = substations.shape[0]
        T = turbines.shape[0]
        border_sizes = np.array([border.shape[0]] + [obstacle.shape[0] for obstacle in obstacles]) if obstacles else np.array([])
        B = border_sizes.sum() if border_sizes.size > 0 else 0
        obstacle_idxs = np.cumsum(border_sizes) + T

        return L_from_site(
            R=R, T=T, B=B,
            border=np.arange(T, T + border.shape[0]) if border is not None and border.shape[0] > 0 else np.array([]),
            obstacles=[np.arange(a, b) for a, b in pairwise(obstacle_idxs)] if obstacles else [],
            name='Example Location',
            handle='example',
            VertexC=np.vstack((turbines, border, *obstacles, substations)),
        )
    
    @classmethod
    def from_yaml(cls, filepath: str, cables=None, **kwargs):
        """Creates a WindFarmNetwork instance from a YAML file."""
        if not isinstance(filepath, str):
            raise TypeError("Filepath must be a string")

        L = L_from_yaml(filepath)
        return cls(L=L, cables= cables, **kwargs)
        
    @classmethod
    def from_windIO(cls, filepath: str, **kwargs):
        """Creates a WindFarmNetwork instance from WindIO format."""
        if not isinstance(filepath, str):
            raise TypeError("Filepath must be a string")
        
        fpath = Path(filepath)
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
    
    def add_aux_methods(self):
        """Attach auxiliary methods to the instance."""

        def plot():
            """Plots the wind farm network graph."""
            return gplot(self.G)

        def get_network():
            """Returns the network edges with cable data."""
            return self.G.edges(data='cable')

        def cost():
            """Returns the total cost of the network."""
            return self.G.size(weight="cost")

        def length():
            """Returns the total cable length of the network."""
            return self.G.size(weight="length")

        def set_coordinates(turbines, substations):
            """Updates the coordinates of turbines and substations."""
            self.L['VertexC'][:turbines.shape[0], :] = turbines
            self.L['VertexC'][-substations.shape[0]:, :] = substations

        def set_network(new_edges):
            """Updates the graph with a new set of edges."""
            if self.G is None:
                raise ValueError("Graph (G) is not initialized.")

            if not isinstance(new_edges, list):
                raise TypeError("new_edges must be a list of tuples (node1, node2, attributes_dict).")

            for edge in new_edges:
                if not (isinstance(edge, tuple) and len(edge) == 3 and isinstance(edge[2], dict)):
                    raise ValueError(f"Invalid edge format: {edge}. Must be (node1, node2, attributes_dict).")

            self.G.remove_edges_from(list(self.G.edges()))
            self.G.add_edges_from(new_edges)

        # Bind these methods to `self`
        self.plot = plot
        self.get_network = get_network
        self.cost = cost
        self.length = length
        self.set_coordinates = set_coordinates
        self.set_network = set_network

    
    # def add_aux_methods(self):
    #     setattr(self, "plot", lambda: gplot(self.G))  # Now attached to self
    #     setattr(self, "get_network", lambda: self.G.edges(data='cable'))  # Now attached to self
    #     setattr(self, "cost", lambda: self.G.size(weight="cost"))
    #     setattr(self, "length", lambda: self.G.size(weight="length"))

    #     def set_coordinates(turbines, substations):
            
    #         # update wfn
    #         self.L['VertexC'][:self.turbines.shape[0], :] = turbines
    #         self.L['VertexC'][-self.substations.shape[0]:, :] = substations

        
    #     setattr(self, "set_coordinates", set_coordinates)

    #     def set_network(new_edges):
    #         """
    #         Updates the graph with a new set of edges.
            
    #         Args:
    #             new_edges (list of tuples): Each tuple should be (node1, node2, attributes_dict).
            
    #         Raises:
    #             ValueError: If new_edges is not properly formatted.
    #         """

    #         #
    #         if not isinstance(new_edges, list):
    #             raise TypeError("new_edges must be a list of tuples (node1, node2, attributes_dict).")

    #         #
    #         for edge in new_edges:
    #             if (
    #                 not isinstance(edge, tuple) or
    #                 len(edge) != 3 or
    #                 not isinstance(edge[0], (int, str)) or  # Node 1 should be int or str
    #                 not isinstance(edge[1], (int, str)) or  # Node 2 should be int or str
    #                 not isinstance(edge[2], dict)  # Edge attributes should be a dictionary
    #             ):
    #                 raise ValueError(
    #                     f"Invalid edge format: {edge}. Each edge must be (node1, node2, attributes_dict).")

    #         self.G.remove_edges_from(list(self.G.edges()))
    #         self.G.add_edges_from(new_edges)

    #     setattr(self, "set_network", set_network)

    def wt_per_cable(cable_cross_section, turbine_power, voltage, efficiency):
        return turbines_per_cable(cable_cross_section, turbine_power, voltage, efficiency)


class OptiWindNetSolver(ABC):
    def __init__(self, wfn, verbose=True, **kwargs):
        self.wfn = wfn
        self.verbose = verbose


    @abstractmethod
    def optimize(self, wfn=None, verbose=True, **kwargs):
        """
        Perform cable layout optimization. Must be implemented by subclasses.
        """
        pass

    def __call__(self, wfn=None, verbose=True, **kwargs):
        """Make the instance callable, calling optimize() internally."""
        return self.optimize(wfn=wfn, verbose=True, **kwargs)  


    def gradient(self, wfn=None, verbose=None, gradient_type='cost'):
        """
        Calculates the gradient of the length and cost of cable with respect to the positions of the nodes.
        """
        if wfn is not None:
            self.wfn = wfn
        else:
            wfn = self.wfn

        # If verbose argument is None, use the value of self.verbose
        if verbose is None:
            verbose = self.verbose

        G = self.wfn.G # self.evaluate(turbines=turbines, substations=substations, verbose=verbose, node_type='all')
        vertexes = G.graph['VertexC']
        R = G.graph['R']
        T = G.graph['T']
        N = len(vertexes)
        gradients_length = np.zeros((N, 2))
        gradients_cost = np.zeros((N, 2))

        # Calculate gradient for each node
        for i, v in enumerate(vertexes):
            # Calculate gradient for x and y dimensions
            for j, u in enumerate(vertexes):
                ii = i if i < N - R else i - N
                jj = j if j < N - R else j - N
                if j != i and G.has_edge(ii, jj):
                    gradients_length[i, 0] += (v[0] - u[0]) / np.linalg.norm(v - u)
                    gradients_length[i, 1] += (v[1] - u[1]) / np.linalg.norm(v - u)
                    gradients_cost[i, 0] += ((v[0] - u[0]) / np.linalg.norm(v - u)) * G.graph['cables'][G.edges[(ii, jj)]['cable']][2]
                    gradients_cost[i, 1] += ((v[1] - u[1]) / np.linalg.norm(v - u)) * G.graph['cables'][G.edges[(ii, jj)]['cable']][2]

        # Filter the results based on the requested node type
        gradients_length_wt = gradients_length[:T]
        gradients_cost_wt = gradients_cost[:T]

        gradients_length_ss = gradients_length[N - R:]
        gradients_cost_ss = gradients_cost[N - R:]

        if gradient_type=='cost':
            gradients_wt = gradients_cost_wt
            gradients_ss = gradients_cost_ss
        elif gradient_type=='length':
            gradients_wt = gradients_length_wt
            gradients_ss = gradients_length_ss
        else:
            raise ValueError("gradient_type should be either 'cost' or 'length'")

        return gradients_wt, gradients_ss


class Heuristic(OptiWindNetSolver):
    def __init__(self, wfn, solver='EW', verbose=True, **kwargs):
        # Call the base class initialization
        self.solver = solver
        super().__init__(wfn=wfn, **kwargs)

    def optimize(self, wfn=None, **kwargs):
        if wfn is not None:
            self.wfn = wfn
        else:
            wfn = self.wfn

        # optimizing
        if self.solver=='EW':
            S = EW_presolver(wfn.A, capacity=wfn.cables_capacity)
        else:
            raise ValueError(
                f"{self.solver} is not among the supported Heuristic solvers. Choose among: EW.")

        G_tentative = G_from_S(S, wfn.A)
        G = G_tentative
        assign_cables(G, wfn.cables)

        # update wfn attributes
        wfn.S = S
        wfn.G_tentative = G_tentative
        wfn.G = G

        self.wfn = wfn
        
        return wfn
    
class MetaHeuristic(OptiWindNetSolver):
    def __init__(self, solver='HGS', time_limit=3, **kwargs):
        # Call the base class initialization
        super().__init__()
        self.solver = solver
        self.time_limit = time_limit

    def optimize(self, wfn=None, verbose=True, **kwargs):
        if wfn is not None:
            self.wfn = wfn
        else:
            wfn = self.wfn

        # optimizing
        if self.solver== 'HGS': # Hybrid Genetic Search
            S = iterative_hgs_cvrp(wfn.A, capacity=wfn.cables_capacity, time_limit=self.time_limit)
        else:
            raise ValueError(
                f"{self.solver} is not among the supported Meta-Heuristic solvers. Choose among: HGS.")
        
        G_tentative = G_from_S(S, wfn.A)
        G = G_tentative
        assign_cables(G, wfn.cables)

        self.wfn = wfn
        
        return wfn

class MILP(OptiWindNetSolver):
    def __init__(self, solver='ortools', solver_options=None, model_options=None, **kwargs):
        # Call the base class initialization
        super().__init__()
        #
        if solver_options is None:
            solver_options = {}

        if model_options is None:
            model_options = {}

        self.solver = solver
        self.solver_options = solver_options
        self.model_options = model_options  

        
    def optimize(self, wfn=None, verbose=True, **kwargs):
        if wfn is not None:
            self.wfn = wfn
        else:
            wfn = self.wfn

        # optimizing
        if self.solver == 'ortools':
            # initialize
            orter = ort.CpSat()
            # set the model
            model = ort.make_min_length_model(
                        wfn.A, wfn.cables_capacity,
                        gateXings_constraint=self.model_options.get("gateXring_constraint", False),
                        branching=self.model_options.get("branching", True),
                        gates_limit=self.model_options.get("gates_limit", False)
                    )
            
            # warm start
            if wfn.S is not None:
                ort.warmup_model(model, wfn.S)

            # settings
            orter.parameters.max_time_in_seconds = self.solver_options.get("max_time_in_seconds", 40)
            orter.parameters.relative_gap_limit = self.solver_options.get("relative_gap_limi", 0.005)
            orter.parameters.num_workers = self.solver_options.get("num_workers", 8)
            orter.parameters.log_search_progress = verbose
            orter.log_callback = print # required to get the log inside the notebook (goes only to console otherwise)

            result = orter.solve(model)

            if verbose: # print info about the final solution
                gap = 1 - orter.BestObjectiveBound()/orter.ObjectiveValue()
                print('=================================================================',
                    #orter.ResponseStats(),  # uncomment if orter.parameters.log_search_progress == False
                    f"\nbest solution's strategy: {orter.SolutionInfo()}",
                    f'\ngap: {100*gap:.1f}%')

            G = ort.investigate_pool(wfn.P, wfn.A, model, orter, result)

        elif self.solver == 'cplex' or 'cbc' or 'gurobi' or 'highs' or 'scip':
            ##############
            # initialize #
            ##############
            # Define solver and solver_io mapping for special cases
            solver_mapping = {
                'highs': 'appsi_highs', 
                # Add cases where solver name needs to be adapted to call from pyomo solver factory
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
                wfn.A, wfn.cables_capacity,      
                gateXings_constraint=self.model_options.get("gateXring_constraint", False),
                branching=self.model_options.get("branching", True),
                gates_limit=self.model_options.get("gates_limit", False)
                )
            
            # warm start
            if wfn.S is not None:
                omo.warmup_model(model, wfn.S)

            # options
            solver_options_mapping = {
                'gurobi': {
                    'mipgap': self.solver_options.get("mingap", 0.005),   # Relative lower-bound to objective-value gap
                    'timelimit': self.solver_options.get("timelimit", 60),   # Time limit in seconds
                    'mipfocus': self.solver_options.get("mipfocus", 1),     # defualt=1: Focus on producing solutions
                },
                'cplex': {
                    'mipgap': self.solver_options.get("mingap", 0.005),
                    'timelimit': self.solver_options.get("timelimit", 60),
                    'parallel': self.solver_options.get("parallel", -1),    # default=-1 : Opportunistic parallelism (non-deterministic)
                    'emphasis_mip': self.solver_options.get("emphasis_mip", 4),    # default=4 : , # Focus on producing solutions
                },
                'cbc': {
                    'ratioGap': self.solver_options.get("mingap", 0.005),
                    'seconds': self.solver_options.get("timelimit", 90),
                    'timeMode': 'elapsed',
                    'threads': self.solver_options.get("parallel", 8),
                    'RandomCbcSeed': self.solver_options.get("seed", 4321),  # Seed for repeatable results
                    'Dins': 'on',
                    'VndVariableNeighborhoodSearch': 'on',
                },
                'highs': {
                    'time_limit': self.solver_options.get("timelimit", 60),     # Time limit in seconds
                    'mip_rel_gap': self.solver_options.get("mingap", 0.005), # MIP gap
                },
                'scip': {
                    'limits/gap': self.solver_options.get("mingap", 0.005),
                    'limits/time': self.solver_options.get("timelimit", 180),
                    'display/freq': 0.5,
                    'parallel/maxnthreads': self.solver_options.get("parallel", 16), # Currently not used in Pyomo
                }
            }

            pyo_solver.options.update(solver_options_mapping[self.solver])

            if verbose:
                print(f'Solving "{model.handle}": {{R={len(model.R)}, T={len(model.T)}, κ={model.k.value}}}\n')

            ##########
            # result #
            ##########
            # Define solver-specific arguments for solving
            solver_args = {
                'gurobi': {'warmstart': model.warmed_by, 'tee': True},
                'cbc': {'warmstart': model.warmed_by, 'tee': True},
                'cplex': {'warmstart': True, 'tee': True},
                'highs': {'tee': True},
                'scip': {'tee': True}
            }
            result = pyo_solver.solve(model, **solver_args[self.solver])

            S = omo.S_from_solution(model, pyo_solver, result)
            G_tentative = G_from_S(S, self.A)
            assign_cables(G_tentative, self.cables)
            G = PathFinder(G_tentative, planar=wfn.P, A=wfn.A).create_detours()
        
        else:
            raise ValueError(
                f"{self.solver} is not among the supported MILP solvers. Choose among: ortools, gurobi, cplex, highs, scip, cbc.")

        wfn.S = S
        wfn.G_tentative = G_tentative
        wfn.G = G

        # update self.wfn
        self.wfn = wfn

        return wfn

#######
# example usage
# if __name__ == "__main__":
        

#     wfn = WindFarmNetwork.from_yaml(filepath='example_location.yaml') # input format: coordinates by default
#     # wfn = WindFarmNetwork(input_format='yaml', filepath='path to the yaml file')
#     # wfn = WindFarmNetwork.from_yaml(filepath='path to the yaml file')
#     # wfn = WindFarmNetwork(input_format='windIo', filepath='path to the yaml file')
#     print('wfn initialized properly')
#     print(wfn.L)
    #optimzer = Heuristic(wfn=wfn, solver='EW') # default is EW
    # wfn = optimzer.optimize(wt_x=0, wt_y=0, ss_x=0, ss_y=0) # am not sure if we should return wfn or just update it internally
    # gradients_wt, gradients_ss = optimzer.gradient(wt_x=0, wt_y=0, ss_x=0, ss_y=0) # gradient_type = cost by default
    # gradients_wt, gradients_ss = optimzer.gradient(wt_x=0, wt_y=0, ss_x=0, ss_y=0, gradient_type='length')

    # optimzer = MetaHeuristic(wfn=wfn, solver='GHS') # default is GHS
    # optimizer = MILP(wfn=wfn, solver='ortools', solver_options={}, model_options={})

###########################################################
# gitlab repo
# shall we build wfn in optimizer.optimize() from the coordinates if wfn is None?
#       in my view, no since it brings complexity for users.
# S, A, ... are shallow copied of L. when we attach them as atributes of wfn (wfn.L, wfn.A, ...)
# does wfn.A be still a shallow copy of wfn.L?


###### to do:
# installation of the dependancies, to include git link in the installation




