import logging
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
import yaml_include

# OptiWindNet modules
from .baselines.hgs import hgs_multiroot, iterative_hgs_cvrp
from .heuristics import CPEW, EW_presolver
from .importer import L_from_pbf, L_from_site, L_from_yaml
from .importer import load_repository as load_repository
from .interarraylib import G_from_S, S_from_G, as_normalized, calcload
from .interface import assign_cables
from .mesh import make_planar_embedding
from .MILP import ModelOptions, solver_factory
from .pathfinding import PathFinder
from .plotting import gplot, pplot
from .svg import svgplot

from .api_utils import (
    check_warmstart_feasibility,
    enable_ortools_logging_if_jupyter,
    extract_network_as_array,
    from_coordinates,
    parse_cables_input,
    plot_org_buff,
)

###################
# OptiWindNet API #
###################

# Keep text editable (not converted to paths) in SVG output
plt.rcParams['svg.fonttype'] = 'none'

# Set up a logger and create shortcuts for error, warning, and info logging methods
logger = logging.getLogger(__name__)
error, warning, info = logger.error, logger.warning, logger.info


class WindFarmNetwork:
    """
    Represents a wind farm electrical network, capable of processing
    layout data from different formats and computing network properties.
    """

    def __init__(
        self,
        cables,
        turbinesC=None,
        substationsC=None,
        borderC=None,
        obstaclesC=None,
        name='',
        handle='',
        L=None,
        router=None,
        buffer_dist=0,
        **kwargs,
    ):
        # Use a default router if none is provided
        if router is None:
            router = EWRouter()
        self.router = router

        # Parse and validate cables input; convert to list of (capacity, cost) tuples
        self.cables = parse_cables_input(cables)
        self.cables_capacity = max(c[0] for c in self.cables)

        # Construct layout from coordinates if not directly provided
        if turbinesC is not None and substationsC is not None:
            if L is not None:
                warning(
                    'Both coordinates and L are given, OptiWindNet prioritizes coordinates over L and neglects the provided L.'
                )
            L = from_coordinates(
                self,
                turbinesC,
                substationsC,
                borderC,
                obstaclesC,
                name,
                handle,
                buffer_dist,
                **kwargs,
            )
        elif L is None:
            raise ValueError(
                'Both turbinesC and substationsC must be provided! Or alternatively L should be given.'
            )

        self.L = L  # Location graph

        # Create planar embedding from L
        self.P, self.A = make_planar_embedding(L)

        # Initialize graph objects for S and G (to be filled later)
        self.S = None
        self.G = None

    def cost(self):
        """Returns the total cost of the network."""
        return self.G.size(weight='cost')

    def length(self):
        """Returns the total cable length of the network."""
        return self.G.size(weight='length')

    def plot_original_vs_buffered(self):
        """Plot original and buffered borders and obstacles on a single plot."""
        # get coordinates
        borderC = self._borderC_original
        border_bufferedC = self._border_bufferedC
        obstaclesC = self._obstaclesC_original
        obstacles_bufferedC = self._obstacles_bufferedC

        plot_org_buff(borderC, border_bufferedC, obstaclesC, obstacles_bufferedC)

    @classmethod
    def from_yaml(cls, filepath: str, **kwargs):
        """Creates a WindFarmNetwork instance from a YAML file."""
        if not isinstance(filepath, str):
            raise TypeError('Filepath must be a string')

        L = L_from_yaml(filepath)
        return cls(L=L, **kwargs)

    @classmethod
    def from_pbf(cls, filepath: str, **kwargs):
        """Creates a WindFarmNetwork instance from a PBF file."""
        if not isinstance(filepath, str):
            error('Filepath must be a string')

        L = L_from_pbf(filepath)
        return cls(L=L, **kwargs)

    @classmethod
    def from_windIO(cls, filepath: str, **kwargs):
        """Creates a WindFarmNetwork instance from WindIO yaml file."""
        if not isinstance(filepath, str):
            raise TypeError('Filepath must be a string')

        fpath = Path(filepath)

        yaml.add_constructor('!include', yaml_include.Constructor(base_dir='data'))

        with open(fpath, 'r') as f:
            system = yaml.full_load(f)

        # Parse coordinate data
        coords = system['wind_farm']['layouts']['initial_layout']['coordinates']
        terminalC = np.c_[coords['x'], coords['y']]
        coords = system['wind_farm']['electrical_substations']['coordinates']
        rootC = np.c_[coords['x'], coords['y']]
        coords = system['site']['boundaries']['polygons'][0]
        borderC = np.c_[coords['x'], coords['y']]

        T = terminalC.shape[0]
        R = rootC.shape[0]
        name_tokens = fpath.stem.split('_')

        # Construct L
        L = L_from_site(
            R=R,
            T=T,
            VertexC=np.vstack((terminalC, borderC, rootC)),
            border=np.arange(T, T + borderC.shape[0]),
            name=' '.join(name_tokens),
            handle=f'{name_tokens[0].lower()}_{name_tokens[1][:4].lower()}_{name_tokens[2][:3].lower()}',
            **kwargs,
        )

        return cls(L=L, **kwargs)

    def _repr_svg_(self):
        """IPython hook for rendering the graph as SVG in notebooks."""
        return svgplot(self.G)._repr_svg_()

    def plot(self, *args, **kwargs):
        """Plots the final optimized network."""
        return gplot(self.G, *args, **kwargs)

    def plot_location(self, **kwargs):
        """Plots the location (vertices and borders)."""
        return gplot(self.L, **kwargs)

    def plot_available_links(self, **kwargs):
        """Plots the available links from planar embedding."""
        return gplot(self.A, **kwargs)

    def plot_navigation_mesh(self, **kwargs):
        """Plots the navigation mesh (planar graph and adjacency)."""
        return pplot(self.P, self.A, **kwargs)

    def plot_selected_links(self, **kwargs):
        """Plots the currently selected links in cable layout (chosen from available links)."""
        G_tentative = G_from_S(self.S, self.A)
        assign_cables(G_tentative, self.cables)
        return gplot(G_tentative, **kwargs)

    def terse_links(self):
        """Returns a compact representation of the selected links as an array of link targets."""
        R, T = (self.S.graph[k] for k in 'RT')
        terse = np.empty(T, dtype=int)

        for u, v, reverse in self.S.edges(data='reverse'):
            if reverse is None:
                error('reverse must not be None')
            u, v = (u, v) if u < v else (v, u)
            i, target = (u, v) if reverse else (v, u)
            terse[i] = target

        return terse

    def update_from_terse_links(
        self, terse_links: np.ndarray, turbinesC=None, substationsC=None
    ) -> None:
        """
        Updates the network from terse link representation.
        Optionally updates node coordinates.
        """
        terse_links = [int(x) for x in terse_links]
        # --- Added block: check input format ---
        terse_links = np.asarray(terse_links)

        # Validate input shape and type
        if not np.issubdtype(terse_links.dtype, np.integer):
            raise ValueError(
                f'terse_links must be an array of integers. Got {terse_links.dtype} instead.\n'
                f'Hint: You can fix it by doing terse_links = [int(x) for x in terse_links].'
            )

        if terse_links.ndim != 1:
            raise ValueError(
                f'terse_links must be a 1D array. Got shape {terse_links.shape} instead.'
            )

        # Update coordinates if provided
        if turbinesC is not None:
            self.L.graph['VertexC'][: turbinesC.shape[0], :] = turbinesC

        if substationsC is not None:
            self.L.graph['VertexC'][-substationsC.shape[0] :, :] = substationsC

        if turbinesC is not None or substationsC is not None:
            self.P, self.A = make_planar_embedding(self.L)

        # Rebuild the selected edge set (links)
        self.S.remove_edges_from(list(self.S.edges()))
        for i, j in enumerate(terse_links):
            self.S.add_edge(i, j)

        calcload(self.S)

        G_tentative = G_from_S(self.S, self.A)

        self.G = PathFinder(G_tentative, planar=self.P, A=self.A).create_detours()

        assign_cables(self.G, self.cables)

        return self.G

    def get_network(self):
        """Returns the network as a structured array of edge data."""
        return extract_network_as_array(self.G)

    def map_detour_vertex(self):
        if self.G.graph.get('C') or self.G.graph.get('D'):
            R, T, B = (self.G.graph[k] for k in 'RTB')
            map = dict(
                enumerate(
                    (n.item() for n in self.G.graph['fnT'][T + B : -R]), start=T + B
                )
            )
        else:
            map = {}
        return map

    def gradient(self, turbinesC=None, substationsC=None, gradient_type='length'):
        """
        Computes gradients of total cable length or cost with respect to the node positions.
        """
        if gradient_type.lower() not in ['cost', 'length']:
            raise ValueError("gradient_type should be either 'cost' or 'length'")

        G = self.G
        VertexC = G.graph['VertexC']
        if turbinesC is not None or substationsC is not None:
            info(
                'wfn.gradient is not checking for the feasibility of the layout with new coordinates!'
            )
            VertexC = VertexC.copy()
            if turbinesC is not None:
                VertexC[: turbinesC.shape[0], :] = turbinesC
            if substationsC is not None:
                VertexC[-substationsC.shape[0] :, :] = substationsC

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
        # suppress the contributions of zero-length edges
        norm[norm < 1e-12] = 1.0
        vec /= norm[:, None]

        if gradient_type.lower() == 'cost':
            cable_costs = np.fromiter(
                (
                    G.graph['cables'][cable]['cost']
                    for *_, cable in G.edges(data='cable')
                ),
                dtype=np.float64,
                count=G.number_of_edges(),
            )
            vec *= cable_costs[:, None]

        np.add.at(gradients, _u, vec)
        np.subtract.at(gradients, _v, vec)

        # wind turbines
        gradients_wt = gradients[:T]
        # substations
        gradients_ss = gradients[-R:]

        return gradients_wt, gradients_ss

    def optimize(self, turbinesC=None, substationsC=None, router=None, verbose=None):
        if router is None:
            router = self.router
        else:
            self.router = router

        # If new coordinates are provided, update them
        if turbinesC is not None:
            self.L.graph['VertexC'][: turbinesC.shape[0], :] = turbinesC

        if substationsC is not None:
            self.L.graph['VertexC'][-substationsC.shape[0] :, :] = substationsC

        if turbinesC is not None or substationsC is not None:
            self.P, self.A = make_planar_embedding(self.L)

        D = (
            self.G.graph['D']
            if hasattr(self, 'G') and self.G is not None and 'D' in self.G.graph
            else 0
        )
        S_warm_has_detour = D > 0

        S, G = router(
            L=self.L,
            A=self.A,
            P=self.P,
            S_warm=self.S,
            S_warm_has_detour=S_warm_has_detour,
            cables=self.cables,
            cables_capacity=self.cables_capacity,
            verbose=verbose,
        )
        self.S = S
        self.G = G

        terse_links = self.terse_links()
        return terse_links


class Router(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def optimize(
        self,
        L=None,
        A=None,
        P=None,
        cables=None,
        cables_capacity=None,
        S_warm=None,
        S_warm_has_detour=False,
        verbose=False,
        **kwargs,
    ):
        pass

    def __call__(
        self,
        L=None,
        A=None,
        P=None,
        cables=None,
        cables_capacity=None,
        S_warm=None,
        S_warm_has_detour=False,
        verbose=False,
    ):
        """Make the instance callable, calling optimize() internally."""
        return self.optimize(
            L=L,
            A=A,
            P=P,
            cables=cables,
            cables_capacity=cables_capacity,
            S_warm=S_warm,
            S_warm_has_detour=S_warm_has_detour,
            verbose=verbose,
        )


class EWRouter(Router):
    def __init__(
        self,
        maxiter=10000,
        feeder_route='segmented',
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Call the base class initialization
        self.verbose = verbose
        self.maxiter = maxiter
        self.feeder_route = feeder_route

    def optimize(self, L, A, P, cables, cables_capacity, verbose=None, **kwargs):
        if verbose is None:
            verbose = self.verbose

        # optimizing
        if self.feeder_route == 'segmented':
            S = EW_presolver(A, capacity=cables_capacity, maxiter=self.maxiter)
        elif self.feeder_route == 'straight':
            G_cpew = CPEW(L, capacity=cables_capacity, maxiter=self.maxiter)
            S = S_from_G(G_cpew)
        else:
            raise ValueError(
                f'{self.feeder_route} is not among the valid feeder_route values. Choose among: ("segmented", "straight").'
            )

        G_tentative = G_from_S(S, A)

        G = PathFinder(G_tentative, planar=P, A=A).create_detours()

        assign_cables(G, cables)

        return S, G


class HGSRouter(Router):
    def __init__(
        self,
        time_limit,
        feeder_limit: int | None = None,
        max_retries=10,
        balanced=False,
        seed: int = 0,
        verbose=False,
        **kwargs,
    ):
        # Call the base class initialization
        super().__init__(**kwargs)
        self.time_limit = time_limit
        self.verbose = verbose
        self.max_retries = max_retries
        self.feeder_limit = feeder_limit
        self.balanced = balanced
        self.seed = seed

    def optimize(
        self, A, P, cables, cables_capacity, S_warm=None, verbose=None, **kwargs
    ):
        # If verbose argument is None, use the value of self.verbose
        if verbose is None:
            verbose = self.verbose

        # optimizing
        R = A.graph['R']
        if R == 1:
            S = iterative_hgs_cvrp(
                as_normalized(A),
                capacity=cables_capacity,
                time_limit=self.time_limit,
                max_retries=self.max_retries,
                vehicles=self.feeder_limit,
                seed=self.seed,
            )
        else:
            S = hgs_multiroot(
                as_normalized(A),
                capacity=cables_capacity,
                time_limit=self.time_limit,
                balanced=self.balanced,
                seed=self.seed,
            )
            if verbose and self.feeder_limit:
                print(
                    'WARNING: HGSRouter is used for a plant with more than one substation and feeder-limit is neglected (The current implementation of HGSRouter does not support limiting the number of feeders in multi-substation plants.)'
                )

        G_tentative = G_from_S(S, A)

        G = PathFinder(G_tentative, planar=P, A=A).create_detours()

        assign_cables(G, cables)

        return S, G


class MILPRouter(Router):
    def __init__(
        self,
        solver_name,
        time_limit,
        mip_gap,
        solver_options=None,
        model_options=None,
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.solver_name = solver_name
        self.solver_options = solver_options or {}
        self.model_options = model_options or ModelOptions()
        self.verbose = verbose
        self.solver = solver_factory(solver_name)
        try:
            self.optiwindnet_default_options = self.solver.options
        except AttributeError:
            self.optiwindnet_default_options = 'Not available'

        if verbose and solver_name == 'ortools':
            enable_ortools_logging_if_jupyter(self.solver)

    def optimize(
        self,
        P,
        A,
        cables,
        cables_capacity,
        S_warm=None,
        S_warm_has_detour=False,
        verbose=None,
        **kwargs,
    ):
        if verbose is None:
            verbose = self.verbose

        warmstart_state = check_warmstart_feasibility(  # noqa
            S_warm=S_warm,
            cables_capacity=cables_capacity,
            model_options=self.model_options,
            S_warm_has_detour=S_warm_has_detour,
            solver_name=self.solver_name,
            verbose=verbose,
            logger=None,
        )

        # To Do: maybe if warmstart_state is False deactivate the warmstarting procedure in MILPRouter?

        solver = self.solver

        solver.set_problem(
            P,
            A,
            capacity=cables_capacity,
            model_options=self.model_options,
            warmstart=S_warm,
        )

        solution_info = solver.solve(
            time_limit=self.time_limit,
            mip_gap=self.mip_gap,
            options=self.solver_options,
            verbose=verbose,
        )

        S, G = solver.get_solution()

        G.SolutionInfo = solution_info

        assign_cables(G, cables)

        return S, G
