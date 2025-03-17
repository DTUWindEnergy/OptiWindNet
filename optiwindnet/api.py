# OptiWindNet API
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import yaml
import yaml_include
from itertools import pairwise


# optiwindnet
from optiwindnet.svg import svgplot
from optiwindnet.importer import L_from_yaml, L_from_pbf, L_from_site
from optiwindnet.plotting import gplot
from optiwindnet.mesh import make_planar_embedding
from optiwindnet.pathfinding import PathFinder
from optiwindnet.interface import assign_cables
from optiwindnet.importer import load_repository
from optiwindnet.interarraylib import G_from_S

# Heuristic
from optiwindnet.heuristics import EW_presolver
# Metha-Heuristic
from optiwindnet.baselines.hgs import iterative_hgs_cvrp


# MILP
from optiwindnet.MILP import ortools as ort
from optiwindnet.pathfinding import PathFinder
import optiwindnet.MILP.pyomo as omo
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
                 cables=None, border=None, obstacles=None, name='', handle='', L=None, verbose=True, **kwargs):

        #
        if turbines is not None:
            L = self._from_coordinates(turbines, substations, border, obstacles, name, handle)
        self.L = L

        # Compute the planar embedding
        self._P, self._A = make_planar_embedding(L)
        self.border = border
        self.obstacles = obstacles
        self.cables = cables

        # Flags to track update status (initialized as updated)
        self._S = None
        self._G_tentative = None
        self.G = None
        self._A_updated = True
        self._P_updated = True
        self._S_updated = False
        self._G_tentative_updated = False
        self._G_updated = False

        if self.cables is not None:
            self.cables_capacity = max(cable[1] for cable in cables) # np.max(self.cables[:, 1])

        self.verbose = verbose

    def _update_planar_embedding(self):
        """Updates P and A if they are stale."""
        if not self._P_updated or not self._A_updated:
            self._P, self._A = make_planar_embedding(self.L)
            self._P_updated = True
            self._A_updated = True

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
    def G_tentative(self):
        if not self._G_tentative_updated and self.verbose:
            print('G_tentative is not updated')
        return self._G_tentative
    
    @G_tentative.setter
    def G_tentative(self, value):
        self._G_tentative = value


    @property
    def G(self):
        if not self._G_updated and self.verbose:
            print('G is not updated')
        return self._G

    @G.setter
    def G(self, value):
        self._G = value
    
    def _from_coordinates(self, turbines, substations, border, obstacles, name, handle):
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
            name=name,
            handle=handle,
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

    
    def plot(self):
        """Plots the wind farm network graph."""
        return gplot(self.G)
    
    def plot_L(self):
        """Plots the wind farm network graph."""
        return gplot(self.L)
    
    def plot_A(self):
        """Plots the wind farm network graph."""
        return gplot(self.A)
    
    def plot_G_tentative(self):
        """Plots the wind farm network graph."""
        return gplot(self.G_tentative)

    def get_network(self):
        """Returns the network edges with cable data."""
        net_graph = self.G.edges(data=True)
        net = list(net_graph)  # Keep it as a list of tuples
        return net
    
    def get_network_array(self):
        """Returns the network edges with cable data."""
        network = self.G.edges(data=True)
        network_array =  np.array([
                            [entry[0], entry[1], entry[2]['length'], entry[2]['load'], int(entry[2]['reverse']), entry[2]['cable'], entry[2]['cost']]
                            for entry in network
                            ])
        return network_array

    def cost(self):
        """Returns the total cost of the network."""
        return self.G.size(weight="cost")

    def length(self):
        """Returns the total cable length of the network."""
        return self.G.size(weight="length")

    def set_coordinates(self, turbines, substations):
        """Updates the coordinates of turbines and substations."""
        self._G_updated = True
        
        if self.verbose:
            print('WARNING: wfn.set_coordinates is not checking for feasiblity')

        if not hasattr(self.L, 'graph') or 'VertexC' not in self.L.graph:
            raise ValueError("Graph L does not contain 'VertexC' attribute.")

        # Update coordinates
        self.L.graph['VertexC'][:turbines.shape[0], :] = turbines
        self.L.graph['VertexC'][-substations.shape[0]:, :] = substations
        self.G.graph['VertexC'][:turbines.shape[0], :] = turbines
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
        self._G_tentative_updated = False

    def set_network(self, network_tree):
        """Updates the graph with a new network tree.""" 
        self._G_updated = True
        
        if self.verbose:
            print('WARNING: wfn.set_network is not checking for feasibility')

        if self.G is None:
            raise ValueError("Graph (G) is not initialized.")

        if not isinstance(network_tree, list):
            raise TypeError("network_tree must be a list of tuples (node1, node2, attributes_dict).")

        for edge in network_tree:
            if not (isinstance(edge, tuple) and len(edge) == 3 and isinstance(edge[2], dict)):
                raise ValueError(f"Invalid edge format: {edge}. Must be (node1, node2, attributes_dict).")

        self.G.remove_edges_from(list(self.G.edges()))
        self.G.add_edges_from(network_tree)

        # Update length
        for u, v, data in self.G.edges(data=True):
            coord_u = self.G.graph['VertexC'][u, :]
            coord_v = self.G.graph['VertexC'][v, :]
            data['length'] = np.linalg.norm(np.array(coord_u) - np.array(coord_v))
        
        # Update cost
        assign_cables(self.G, self.cables)

        self._A_updated = False
        self._P_updated = False
        self._S_updated = False
        self._G_tentative_updated = False

    
    def set_network_array(self, network_array):
        network = [
                (int(row[0]), int(row[1]), {
                    'length': np.float64(row[2]),
                    'load': int(row[3]),
                    'reverse': bool(row[4]),
                    'cable': int(row[5]),
                    'cost': np.float64(row[6])
                })
                for row in network_array
            ]
        self.set_network(network_tree=network)


    def wt_per_cable(cable_cross_section, turbine_power, voltage, efficiency):
        return turbines_per_cable(cable_cross_section, turbine_power, voltage, efficiency)


class OptiWindNetSolver(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def optimize(self, turbines=None, substations=None, network_array=None, verbose=True, **kwargs):
        """
        Perform cable layout optimization. Must be implemented by subclasses.
        """
        pass

    def __call__(self, turbines=None, substations=None, network_array=None, verbose=True, **kwargs):
        """Make the instance callable, calling optimize() internally."""
        return self.optimize( turbines=turbines, substations=substations, network_array=network_array, verbose=verbose, **kwargs)  

    def gradient(self, turbines=None, substations=None, network_tree=None, verbose=True, gradient_type='cost'):
        """
        Calculates the gradient of the length and cost of cable with respect to the positions of the nodes.
        """
        # If verbose argument is None, use the value of self.verbose
        if verbose is None:
            verbose = self.verbose

        if turbines is not None or substations is not None:
            self.wfn.set_coordinates(turbines=turbines, substations=substations)

        if network_tree is not None:
            self.wfn.set_network(network_tree=network_tree)

        G = self.wfn.G
        print(G.edges(data=True))
        vertexes = G.graph['VertexC']
        R = G.graph['R']
        T = G.graph['T']
        N = len(vertexes)
        gradients_length = np.zeros((N, 2))
        gradients_cost = np.zeros((N, 2))

        # Iterate over edges directly to avoid duplicate calculations
        for ii, jj in G.edges():
            if ii == jj:  # Skip self-loops
                continue
            
            v = vertexes[ii]
            u = vertexes[jj]
            
            if gradient_type == 'cost':
                gradients_cost[ii, 0] += ((v[0] - u[0]) / np.linalg.norm(v - u)) * G.graph['cables'][G.edges[(ii, jj)]['cable']][2]
                gradients_cost[ii, 1] += ((v[1] - u[1]) / np.linalg.norm(v - u)) * G.graph['cables'][G.edges[(ii, jj)]['cable']][2]
            else:
                gradients_length[ii, 0] += (v[0] - u[0]) / np.linalg.norm(v - u)
                gradients_length[ii, 1] += (v[1] - u[1]) / np.linalg.norm(v - u)


        # wind turbines
        gradients_length_wt = gradients_length[:T]
        gradients_cost_wt = gradients_cost[:T]

        # substations
        gradients_length_ss = gradients_length[N - R:]
        gradients_cost_ss = gradients_cost[N - R:]

        # Filter the results based on the requested gradient type
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
    def __init__(self, wfn, solver='EW', detour=True, verbose=True, **kwargs):
        # Call the base class initialization
        self.wfn = wfn
        self.verbose = verbose
        self.solver = solver
        self.detour = detour
        #super().__init__(wfn=wfn, **kwargs)

    def optimize(self, turbines=None, substations=None, network_array=None, verbose=None, **kwargs):
        wfn = self.wfn
        # set to True at the begining to avoid unwanted verbose info
        wfn._S_updated = True
        wfn._G_tentative_updated = True
        wfn._G_updated = True
        
        if turbines is not None or substations is not None:
            wfn.set_coordinates(turbines=turbines, substations=substations)

        if network_array is not None:
            print('The optimizer is not run since a network_array is given')
            wfn.set_network_array(network_array=network_array)
        else:
            # If verbose argument is None, use the value of self.verbose
            if verbose is None:
                verbose = self.verbose

            # optimizing
            if self.solver=='EW':
                S = EW_presolver(wfn.A, capacity=wfn.cables_capacity)
            else:
                raise ValueError(
                    f"{self.solver} is not among the supported Heuristic solvers. Choose among: EW.")

            G_tentative = G_from_S(S, wfn.A)
            
            if self.detour:
                G = PathFinder(G_tentative, planar=wfn.P, A=wfn.A).create_detours()
            else:
                G = G_tentative

            assign_cables(G, wfn.cables)

            # update wfn attributes
            wfn.S = S
            wfn.G_tentative = G_tentative
            wfn.G = G

            self.wfn = wfn

            if verbose:
                print('S, G_tentative, and G got updated!')
        return wfn
    
class MetaHeuristic(OptiWindNetSolver):
    def __init__(self, wfn, solver='HGS', time_limit=3, verbose=True, **kwargs):
        # Call the base class initialization
        self.wfn = wfn
        self.verbose = verbose
        self.solver = solver
        self.time_limit = time_limit

    def optimize(self, turbines=None, substations=None, network_tree=None, **kwargs):
        wfn = self.wfn
        # set to True at the begining to avoid unwanted verbose info
        wfn._S_updated = True
        wfn._G_tentative_updated = True
        wfn._G_updated = True

        if turbines is not None or substations is not None:
            self.wfn.set_coordinates(turbines=turbines, substations=substations)

        if network_tree is not None:
            self.wfn.set_network(network_tree=network_tree)

        # If verbose argument is None, use the value of self.verbose
        if verbose is None:
            verbose = self.verbose

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
    def __init__(self, wfn, solver='ortools', solver_options=None, model_options=None, verbose=True, **kwargs):  
        self.wfn = wfn
        self.verbose = verbose
        #
        if solver_options is None:
            solver_options = {}

        if model_options is None:
            model_options = {}

        self.solver = solver
        self.solver_options = solver_options
        self.model_options = model_options  

        
    def optimize(self, turbines=None, substations=None, network_tree=None, verbose=None, **kwargs):
        wfn = self.wfn
        # set to True at the begining to avoid unwanted verbose info
        wfn._S_updated = True
        wfn._G_tentative_updated = True
        wfn._G_updated = True

        if turbines is not None or substations is not None:
            self.wfn.set_coordinates(turbines=turbines, substations=substations)

        if network_tree is not None:
            self.wfn.set_network(network_tree=network_tree)

        # If verbose argument is None, use the value of self.verbose
        if verbose is None:
            verbose = self.verbose

        # optimizing
        if self.solver == 'ortools':
            # initialize
            #orter = ort.CpSat()
            # set the model
            model = ort.make_min_length_model(
                        wfn.A,
                        wfn.cables_capacity,
                        gateXings_constraint=self.model_options.get("gateXring_constraint", False),
                        branching=self.model_options.get("branching", True),
                        gates_limit=self.model_options.get("gates_limit", False)
                    )
            
            # warm start
            if wfn.S is not None:
                ort.warmup_model(model, wfn.S)

            orter = ort.cp_model.CpSolver()
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

            S = ort.S_from_solution(model, orter, result)
            G_tentative = G_from_S(S, wfn.A)
            G = PathFinder(G_tentative, planar=wfn.P, A=wfn.A).create_detours()

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

        
        assign_cables(G, wfn.cables)
        
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




