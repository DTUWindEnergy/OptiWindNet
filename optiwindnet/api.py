import logging
from abc import ABC, abstractmethod
from itertools import pairwise
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import networkx as nx
import numpy as np
import shapely as shp

from .api_utils import (
    buffer_border_obs,
    enable_ortools_logging_if_jupyter,
    extract_network_as_array,
    is_warmstart_eligible,
    merge_obs_into_border,
    parse_cables_input,
    plot_org_buff,
)
from .baselines.hgs import hgs_multiroot, iterative_hgs_cvrp
from .heuristics import CPEW, EW_presolver
from .importer import L_from_pbf, L_from_site, L_from_yaml, L_from_windIO
from .importer import load_repository as load_repository
from .interarraylib import (
    G_from_S,
    S_from_G,
    as_normalized,
    as_stratified_vertices,
    assign_cables,
    calcload,
)
from .mesh import make_planar_embedding
from .MILP import ModelOptions, solver_factory, OWNSolutionNotFound, OWNWarmupFailed
from .pathfinding import PathFinder
from .plotting import gplot, pplot
from .svg import svgplot

##################################
# OptiWindNet Network/Router API #
##################################

# Keep text editable (not converted to paths) in SVG output
plt.rcParams['svg.fonttype'] = 'none'

# Set up a logger and create shortcuts for error, warning, and info logging methods
logger = logging.getLogger(__name__)
error, warning, info = logger.error, logger.warning, logger.info


class Router(ABC):
    """Abstract base class for routing algorithms in OptiWindNet.

    Each Router implementation must define a `route` method.
    """

    _summary_attrs: tuple[str, ...]

    @abstractmethod
    def route(
        self,
        P: nx.PlanarEmbedding,
        A: nx.Graph,
        cables: list[tuple[int, float]],
        cables_capacity: int,
        verbose: bool,
        **kwargs,
    ) -> tuple[nx.Graph, nx.Graph]:
        """Run the routing optimization.

        Args:
          P : Navigation mesh for the location.
          A : Graph of available links.
          cables: set of cable specifications as [(capacity, linear_cost), ...].
          cables_capacity: highest cable capacity in cables.
          verbose : Whether to print progress/logging info.
          **kwargs : Additional router-specific parameters.

        Returns:
          S : Solution topology (selected links).
          G : Optimized network graph with routes and cable types.
        """
        pass


class WindFarmNetwork:
    """Wind farm electrical network.

    Wrapper of most of OptiWindNet's functionality (optimization, visualization,
    cost/length evaluation, and gradient calculation).

    An instance represents a wind farm location, which initially contains the number
    and positions of wind turbines and substations, the delimited area and eventual
    obstacles. A cable network may be provided or a ``Router`` instance may be used
    to create an optimized network.
    """

    _is_stale_PA: bool = True
    _is_stale_SG: bool = True
    _is_stale_polygon: bool = True
    _buffer_dist: float = 0.0

    def __init__(
        self,
        cables: int | list[int] | list[tuple[int, float]] | np.ndarray,
        turbinesC: np.ndarray | None = None,
        substationsC: np.ndarray | None = None,
        borderC: np.ndarray = np.empty((0, 2), dtype=np.float64),
        obstacleC_: Sequence[np.ndarray] = [],
        name: str = '',
        handle: str = '',
        L: nx.Graph | None = None,
        router: Router | None = None,
        verbose: bool = False,
    ):
        """Initialize a wind farm electrical network.

        Args:
          cables: Multiple formats are accepted (capacity is in number of turbines):
            * Set of cable specifications as: [(capacity, linear_cost), ...].
            * Sequence of maximum capacity per cable type: [capacity_0, capacity_1, ...]
            * Maximum capacity of all available cables: capacity
          turbinesC: Turbine coordinates (T, 2): [(x, y), ...].
          substationsC: Substation coordinates (R, 2): [(x, y), ...].
          borderC: Polygonal border coordinates (_, 2): [(x, y), ...].
          obstacleC_: One or more polygons for exclusion zones list of (_, 2): [[(x, y), ...], ...].
          name: Human-readable instance name. Defaults to "".
          handle: Short instance identifier. Defaults to "".
          L: Location geometry (takes precedence over coordinate inputs).
          router: Routing algorithm instance. Defaults to `EWRouter`.
          buffer_dist: Buffer distance to dilate borders / erode obstacles. Defaults to 0.

        Notes:
          * If both `L` and coordinates are provided, `L` takes precedence.
          * Changing coordinate data after creation (`turbinesC` and/or `substationsC`)
              rebuilds `L` and refreshes the navigation mesh and available links.

        Example::

          wfn = WindFarmNetwork(
            cables=[(3, 100.0), (5, 150.0)],
            turbinesC=np.array([[0, 0], [1, 0], [0, 1]]),
            substationsC=np.array([[10, 0]]),
          )
          wfn.optimize()
          print(wfn.cost(), wfn.length())
        """
        # simple fields via setters (for validation/normalization)
        self.name = name
        'Instance name.'
        self.handle = handle
        'Short instance identifier.'
        self.router = router if router is not None else EWRouter()
        self.cables = cables

        self.verbose = verbose
        'Enable verbose logging.'

        # decide source of L
        if L is not None:
            if turbinesC is not None or substationsC is not None:
                warning(
                    'Both coordinates and L are given, OptiWindNet prioritizes L over coordinates.'
                )
            L = as_stratified_vertices(L)
            T = L.graph['T']
        elif turbinesC is not None and substationsC is not None:
            T = turbinesC.shape[0]
            border_sizes = np.array(
                [borderC.shape[0]] + [obs.shape[0] for obs in obstacleC_], dtype=np.int_
            )
            obstacle_slicelims = tuple(pairwise(T + np.cumsum(border_sizes)))

            L = L_from_site(
                R=substationsC.shape[0],
                T=T,
                B=border_sizes.sum().item(),
                **(
                    {'border': np.arange(T, T + borderC.shape[0])}
                    if (borderC is not None and borderC.shape[0] >= 3)
                    else {}
                ),
                obstacles=[np.arange(a, b) for a, b in obstacle_slicelims],
                name=name,
                handle=handle,
                VertexC=np.vstack((turbinesC, borderC, *obstacleC_, substationsC)),
            )
        else:
            raise TypeError(
                'Both turbinesC and substationsC must be provided! Alternatively, L should be given.'
            )
        self._L = L
        self._VertexC = L.graph['VertexC']
        self._R, self._T = L.graph['R'], T

    # -------- helpers --------
    def _refresh_planar(self):
        polygon = self.polygon
        if polygon is not None:
            # check if any of the new turbine coordinates lie outside the polygon
            if isinstance(polygon, shp.Polygon):
                out_of_bounds = shp.MultiPoint(self._VertexC[: self._T]) - self.polygon
            else:
                # polygon is a Multipolygon of the obstacles
                out_of_bounds = polygon & shp.MultiPoint(self._VertexC[: self._T])
                for obstacle in polygon.geoms:
                    if out_of_bounds.is_empty:
                        break
                    # remove from out_of_bounds the points lying on the border
                    out_of_bounds -= obstacle.exterior
            if not out_of_bounds.is_empty:
                # TODO: if relevant, get coordinates of turbines from out_of_bounds
                #  print(list(out_of_bounds.geoms))
                raise ValueError('Turbine out of bounds!')
        self._P, self._A = make_planar_embedding(self._L)
        self._is_stale_PA = False

    # -------- properties --------
    @property
    def L(self) -> nx.Graph:
        "Location geometry (turbines, substations, borders, obstacles)."
        return self._L

    @property
    def polygon(self) -> shp.Polygon | shp.MultiPolygon | None:
        "Shapely (Multi)Polygon that bounds the cable-laying area."
        if self._is_stale_polygon:
            L = self._L
            T = L.graph['T']
            border_sizes = np.array(
                #  [len(L.graph['border'])] + [len(obs) for obs in L.graph['obstacles']],
                [len(L.graph.get('border', []))]
                + [len(obs) for obs in L.graph.get('obstacles', [])],
                dtype=np.int_,
            )
            obstacle_slicelims = tuple(pairwise(T + np.cumsum(border_sizes)))
            if border_sizes[0] > 0:
                self._polygon = shp.Polygon(
                    shell=self._VertexC[T : T + border_sizes[0]],
                    holes=[self._VertexC[a:b] for a, b in obstacle_slicelims],
                )
            elif border_sizes.shape[0] > 1:
                self._polygon = shp.MultiPolygon(
                    [self._VertexC[a:b] for a, b in obstacle_slicelims]
                )
            else:
                return None
            shp.prepare(self._polygon)
            self._is_stale_polygon = False
        return self._polygon

    @property
    def P(self) -> nx.PlanarEmbedding:
        "Triangular mesh over `L` (navigation mesh)."
        if self._is_stale_PA:
            self._refresh_planar()
        return self._P

    @property
    def A(self) -> nx.Graph:
        "Available links graph (search space)."
        if self._is_stale_PA:
            self._refresh_planar()
        return self._A

    @property
    def S(self) -> nx.Graph:
        "Solution topology (selected links)."
        if self._is_stale_SG:
            raise RuntimeError('Call the `optimize()` method to update G.')
        return self._S

    @property
    def G(self) -> nx.Graph:
        "Optimized network with cable routes and types."
        if self._is_stale_SG:
            raise RuntimeError('Call the `optimize()` method to update G.')
        return self._G

    @property
    def cables(self) -> list[tuple[int, float]]:
        "Set of cable specifications as [(capacity, linear_cost), ...]."
        return self._cables

    @cables.setter
    def cables(self, cables):
        parsed = parse_cables_input(cables)
        self._cables = parsed
        self.cables_capacity = max(parsed)[0]
        'highest cable capacity in cables.'
        if not self._is_stale_SG:
            assign_cables(self._G, parsed)

    @property
    def router(self) -> Router:
        "Router instance used for optimization."
        return self._router

    @router.setter
    def router(self, router: Router):
        self._router = router if router is not None else EWRouter()

    @property
    def buffer_dist(self) -> float:
        "Buffer distance applied to dilate borders / erode obstacles."
        return self._buffer_dist

    def cost(self) -> float:
        """Get the total cost of the optimized network."""
        return self.G.size(weight='cost')

    def length(self) -> float:
        """Get the total cable length of the optimized network."""
        return self.G.size(weight='length')

    def plot_original_vs_buffered(self, **kwargs) -> Axes | None:
        """Plot original and buffered borders and obstacles on a single plot.

        Args:
          **kwargs: passed to matplotlib's pyplot.figure()

        Returns:
          matplotlib Axes instance.
        """
        L = self._L
        VertexC = self._VertexC
        landscape_angle = L.graph.get('landscape_angle', False)
        if landscape_angle:
            pass  # TODO: to be added

        borderC = VertexC[L.graph.get('border', [])]
        obstacleC_ = [VertexC[obs] for obs in L.graph.get('obstacles', [])]

        try:
            return plot_org_buff(
                self._pre_buffer_border_obs['borderC'],
                borderC,
                self._pre_buffer_border_obs['obstaclesC'],
                obstacleC_,
                **kwargs,
            )
        except AttributeError:
            print('No buffering is performed')

    @classmethod
    def from_yaml(cls, filepath: str, **kwargs):
        """Create a WindFarmNetwork instance from a YAML file."""
        return cls(L=L_from_yaml(filepath), **kwargs)

    @classmethod
    def from_pbf(cls, filepath: Path | str, **kwargs):
        """Create a WindFarmNetwork instance from a .OSM.PBF file."""
        return cls(L=L_from_pbf(filepath), **kwargs)

    @classmethod
    def from_windIO(cls, filepath: Path | str, **kwargs):
        """Create a WindFarmNetwork instance from WindIO yaml file."""
        return cls(L=L_from_windIO(filepath), **kwargs)

    def _repr_svg_(self):
        """IPython hook for rendering the graph as SVG in notebooks."""
        return svgplot(self.L if self._is_stale_SG else self.G)._repr_svg_()

    def plot(self, *args, **kwargs):
        """Plot the optimized network."""
        return gplot(self.G, *args, **kwargs)

    def plot_location(self, **kwargs):
        """Plot the original location geometry."""
        return gplot(self.L, **kwargs)

    def plot_available_links(self, **kwargs):
        """Plot available links from planar embedding."""
        return gplot(self.A, **kwargs)

    def plot_navigation_mesh(self, **kwargs):
        """Plot navigation mesh (planar graph and adjacency)."""
        return pplot(self.P, self.A, **kwargs)

    def plot_selected_links(self, **kwargs):
        """Plot tentative link selection."""
        G_tentative = G_from_S(self.S, self.A)
        assign_cables(G_tentative, self.cables)
        return gplot(G_tentative, **kwargs)

    def terse_links(self):
        """Get a compact representation of the solution topology."""
        T = self.S.graph['T']
        terse = np.empty(T, dtype=int)

        for u, v, reverse in self.S.edges(data='reverse'):
            if reverse is None:
                error('reverse must not be None')
            u, v = (u, v) if u < v else (v, u)
            i, target = (u, v) if reverse else (v, u)
            terse[i] = target

        return terse

    def update_from_terse_links(
        self,
        terse_links: np.ndarray,
        turbinesC: np.ndarray | None = None,
        substationsC: np.ndarray | None = None,
    ):
        """Update the network from terse link representation.

        Accepts integers or integer-like floats (e.g., 3.0).
        """
        T = self._T
        R = self._R

        terse_links_ints = np.asarray(terse_links, dtype=np.int64)

        # Update coordinates if provided
        if turbinesC is not None:
            self._VertexC[:T] = turbinesC
            self._is_stale_PA = True

        if substationsC is not None:
            self._VertexC[-R:] = substationsC
            self._is_stale_PA = True

        S = nx.Graph(R=R, T=T, creator='from_terse_links')
        for i, j in enumerate(terse_links_ints):
            S.add_edge(i, j)

        calcload(S)

        G_tentative = G_from_S(S, self.A)

        self._S = S
        self._G = PathFinder(G_tentative, planar=self.P, A=self.A).create_detours()

        assign_cables(self._G, self.cables)
        self._is_stale_SG = False

        return

    def get_network(self):
        """Export the optimized network as a structured array."""
        return extract_network_as_array(self.G)

    def map_detour_vertex(self):
        """Map detour vertices back to their original coordinate indices."""
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

    def merge_obstacles_into_border(self):
        L = merge_obs_into_border(self._L)
        self._L = L
        self._VertexC = L.graph['VertexC']
        self._is_stale_polygon = True
        self._is_stale_PA = True

    def add_buffer(self, buffer_dist):
        """Dilate the cable-laying area by `buffer_dist`.

        Useful if boundaries are not strictly enforced during optimization. This may
        happen if boundary compliance is achieved through the application of penalties
        for violations. OptiWindNet will fail if turbines are outside the border, so
        choose a `buffer_dist` that is greater than the maximum single step in position.

        Args:
          buffer_dist: Buffer distance to dilate borders / erode obstacles.
        """
        L, self._pre_buffer_border_obs = buffer_border_obs(
            self._L, buffer_dist=buffer_dist
        )
        self._L = L
        self._VertexC = L.graph['VertexC']
        self._is_stale_polygon = True
        self._is_stale_PA = True

    def gradient(self, turbinesC=None, substationsC=None, gradient_type='length'):
        """Compute length/cost gradients with respect to node positions."""
        if gradient_type.lower() not in ['cost', 'length']:
            raise ValueError("gradient_type should be either 'cost' or 'length'")

        G = self.G
        VertexC = G.graph['VertexC'].copy()
        T = self._T
        R = self._R

        # Update coordinates if provided
        if turbinesC is not None:
            VertexC[:T] = turbinesC

        if substationsC is not None:
            VertexC[-R:] = substationsC

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
            cost_ = [cost for _, cost in G.graph['cables']]
            cable_costs = np.fromiter(
                (cost_[cable] for *_, cable in G.edges(data='cable')),
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

    def optimize(self, turbinesC=None, substationsC=None, router=None, verbose=False):
        """Optimize electrical network."""
        R, T = self._R, self._T
        if router is None:
            router = self.router
        else:
            self.router = router

        verbose = verbose or self.verbose

        # If new coordinates are provided, update them
        if turbinesC is not None:
            self._VertexC[:T] = turbinesC
            self._is_stale_PA = True

        if substationsC is not None:
            self._VertexC[-R:] = substationsC
            self._is_stale_PA = True

        if not self._is_stale_SG:
            warmstart = dict(
                S_warm=self._S,
                S_warm_has_detour=self._G.graph.get('D', 0) > 0,
            )
        else:
            warmstart = {}

        self._S, self._G = router.route(
            P=self.P,
            A=self.A,
            cables=self.cables,
            cables_capacity=self.cables_capacity,
            verbose=verbose,
            **warmstart,
        )
        self._is_stale_SG = False

        terse_links = self.terse_links()
        return terse_links

    def solution_info(self):
        """Get model and solver information of the latest solution (runtime, objective, gap, etc.)."""
        info = {
            'router': self.router.__class__.__name__,
            'capacity': self.cables_capacity,
        }
        info.update(
            {
                k: v
                for k, v in self.G.graph['method_options'].items()
                if not k.startswith('fun')
            }
        )
        info.update({k: self.G.graph[k] for k in self.router._summary_attrs})
        return info


class EWRouter(Router):
    """A lightweight, ultra-fast router for electrical network optimization.

    * Uses a modified Esau-Williams heuristic (segmented or straight feeders).
    * Produces solutions in milliseconds, suitable for quick solutions or warm starts.
    """

    _summary_attrs = ('iterations',)

    def __init__(
        self,
        maxiter: int = 10_000,
        feeder_route: str = 'segmented',
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Create a Esau-Williams-based router.
        Args:
          maxiter: Maximum iterations.
          feeder_route: Feeder routing mode ("segmented" or "straight").
          verbose: Enable verbose logging.
        """

        super().__init__(**kwargs)

        # Call the base class initialization
        self.verbose = verbose
        self.maxiter = maxiter
        self.feeder_route = feeder_route

    def route(self, P, A, cables, cables_capacity, verbose=False, **kwargs):
        verbose = verbose or self.verbose

        # optimizing
        if self.feeder_route == 'segmented':
            S = EW_presolver(A, capacity=cables_capacity, maxiter=self.maxiter)
        elif self.feeder_route == 'straight':
            G_cpew = CPEW(A, capacity=cables_capacity, maxiter=self.maxiter)
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
    """A fast router based on Hybrid Genetic Search (HGS-CVRP).

    Uses the method and implementation by Vidal, 2022:
      Vidal, T. (2022). Hybrid genetic search for the CVRP: Open-source implementation
      and SWAP* neighborhood. Computers & Operations Research, 140, 105643.
      https://doi.org/10.1016/j.cor.2021.105643

    * Balances solution quality and runtime.
    * Produces only radial solutions.
    """

    _summary_attrs = ('runtime',)

    def __init__(
        self,
        time_limit: float,
        feeder_limit: int | None = None,
        max_retries: int = 10,
        balanced: bool = False,
        seed: int | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Create an HGS-based router.

        Args:
            time_limit: Maximum runtime for a single HGS run (in seconds).
            feeder_limit: Maximum number of feeders allowed (ignored if multiple substations).
            max_retries: Maximum number of retries if a feasible solution is not found.
            balanced: Whether to balance turbines/loads across feeders.
            seed: Set the seed of the pseudo-random number generator (reproducibility).
            verbose: Enable verbose logging.

        Notes:
            * The total runtime may reach up to `max_retries * time_limit` in the worst case.
        """
        # Call the base class initialization
        super().__init__(**kwargs)
        self.time_limit = time_limit
        self.verbose = verbose
        self.max_retries = max_retries
        self.feeder_limit = feeder_limit
        self.balanced = balanced
        self.seed = seed

    def route(self, P, A, cables, cables_capacity, verbose=False, **kwargs):
        verbose = verbose or self.verbose

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
                    'WARNING: HGSRouter is used for a plant with more than one '
                    'substation and feeder-limit is neglected (The current '
                    'implementation of HGSRouter does not support limiting the number '
                    'of feeders in multi-substation plants.)'
                )

        G_tentative = G_from_S(S, A)

        G = PathFinder(G_tentative, planar=P, A=A, branched=False).create_detours()

        assign_cables(G, cables)

        return S, G


class MILPRouter(Router):
    """An exact router using mathematical programming.

    * Uses a Mixed-Integer Linear Programming (MILP) model of the problem.
    * Produces provably optimal or near-optimal networks (with quality metrics).
    * Requires a longer runtime than heuristics- and meta-heuristics-based routers.
    """

    _summary_attrs = ('runtime', 'bound', 'objective', 'relgap', 'termination')

    def __init__(
        self,
        solver_name: str,
        time_limit: float,
        mip_gap: float,
        solver_options: dict | None = None,
        model_options: ModelOptions | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Create a MILP-based router.
        Args:
            solver_name: Name of solver (e.g., "gurobi", "cbc", "ortools", "cplex", "highs", "scip").
            time_limit: Maximum runtime (seconds).
            mip_gap: Relative MIP optimality gap tolerance.
            solver_options: Extra solver-specific options.
            model_options: Options for the MILP model.
            verbose: Enable verbose logging.
        """
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

    def route(
        self,
        P,
        A,
        cables,
        cables_capacity,
        verbose=False,
        S_warm=None,
        S_warm_has_detour=False,
        num_retries: int = 2,
        **kwargs,
    ):
        verbose = verbose or self.verbose

        if self.solver_name == 'ortools':
            # pyomo-based solvers already do a thorough feasibility check on warmstarts
            is_warmstart_eligible(
                S_warm=S_warm,
                cables_capacity=cables_capacity,
                model_options=self.model_options,
                S_warm_has_detour=S_warm_has_detour,
                solver_name=self.solver_name,
                logger=logging.getLogger(__name__),
                verbose=verbose,
            )

        solver = self.solver

        for _ in range(2):
            try:
                solver.set_problem(
                    P,
                    A,
                    capacity=cables_capacity,
                    model_options=self.model_options,
                    warmstart=S_warm,
                )
                break
            except OWNWarmupFailed:
                if self.model_options['topology'] == 'branched':
                    feeder_route = self.model_options['feeder_route']
                    if feeder_route == 'segmented':
                        S_warm = EW_presolver(A, capacity=cables_capacity)
                    elif feeder_route == 'straight':
                        S_warm = S_from_G(CPEW(A, capacity=cables_capacity))
                else:
                    if A.graph['R'] == 1:
                        S_warm = iterative_hgs_cvrp(
                            as_normalized(A),
                            capacity=cables_capacity,
                            time_limit=min(self.time_limit, 0.2),
                        )
                    else:
                        S = hgs_multiroot(
                            as_normalized(A),
                            capacity=cables_capacity,
                            time_limit=min(self.time_limit, 0.2),
                        )

        else:
            raise OWNWarmupFailed('Unable to warm-start model.')

        for _ in range(num_retries + 1):
            try:
                solver.solve(
                    time_limit=self.time_limit,
                    mip_gap=self.mip_gap,
                    options=self.solver_options,
                    verbose=verbose,
                )
                break
            except OWNSolutionNotFound:
                continue
        else:
            raise OWNSolutionNotFound(
                f'Unable to find a solution to the MILP model after {num_retries} retries'
            )

        S, G = solver.get_solution()

        assign_cables(G, cables)

        return S, G
