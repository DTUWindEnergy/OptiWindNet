import logging
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml
import yaml_include

from .api_utils import (
    enable_ortools_logging_if_jupyter,
    extract_network_as_array,
    from_coordinates,
    is_warmstart_eligible,
    parse_cables_input,
    plot_org_buff,
)
from .baselines.hgs import hgs_multiroot, iterative_hgs_cvrp
from .heuristics import CPEW, EW_presolver
from .importer import L_from_pbf, L_from_site, L_from_yaml
from .importer import load_repository as load_repository
from .interarraylib import G_from_S, S_from_G, as_normalized, assign_cables, calcload
from .mesh import make_planar_embedding
from .MILP import ModelOptions, solver_factory
from .pathfinding import PathFinder
from .plotting import gplot, pplot
from .svg import svgplot

###################
# OptiWindNet API #
###################

# Keep text editable (not converted to paths) in SVG output
plt.rcParams['svg.fonttype'] = 'none'

# Set up a logger and create shortcuts for error, warning, and info logging methods
logger = logging.getLogger(__name__)
error, warning, info = logger.error, logger.warning, logger.info


class Router(ABC):
    """Abstract base class for routing algorithms in OptiWindNet.

    Each Router implementation must define a `route` method.
    """

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

    Attributes:
      cables: set of cable specifications as [(capacity, linear_cost), ...].
      cables_capacity: highest cable capacity in cables.
      L: Location geometry (turbines, substations, borders, obstacles).
      P: Triangular mesh over `L` (navigation mesh).
      A: Available links graph (search space).
      S: Solution topology (selected links).
      G: Optimized network with cable routes and types.
      name: Instance name.
      handle: Short instance identifier.
      buffer_dist: Border/obstacle buffer distance.
      router: Router instance used for optimization.
    """

    _is_stale_PA: bool = True
    _is_stale_SG: bool = True

    def __init__(
        self,
        cables: int | list[int] | list[tuple[int, float]] | np.ndarray,
        turbinesC: np.ndarray | None = None,
        substationsC: np.ndarray | None = None,
        borderC: np.ndarray | None = None,
        obstaclesC: np.ndarray | None = None,
        name: str = '',
        handle: str = '',
        L: nx.Graph | None = None,
        router: Router | None = None,
        buffer_dist: float = 0.0,
        verbose: bool = False,
        **kwargs,
    ):
        """Initialize a wind farm electrical network.

        Args:
          cables: Multiple formats are accepted (capacity is in number of turbines):
            * Set of cable specifications as: [(capacity, linear_cost), ...].
            * Sequence of maximum capacity per cable type: [capacity_0, capacity_1, ...]
            * Maximum capacity of all available cables: capacity
          turbinesC: Turbine coordinates: [(x, y), ...].
          substationsC: Substation coordinates: [(x, y), ...].
          borderC: Polygonal border coordinates: [(x, y), ...].
          obstaclesC: One or more polygons for exclusion zones: [[(x, y), ...], ...].
          name: Human-readable instance name. Defaults to "".
          handle: Short instance identifier. Defaults to "".
          L: Location geometry (takes precedence over coordinate inputs).
          router: Routing algorithm instance. Defaults to `EWRouter`.
          buffer_dist: Buffer distance to inflate borders / shrink obstacles. Defaults to 0.
          **kwargs: Additional keyword arguments forwarded to network-construction helpers.

        Notes:
          * If both `L` and coordinates are provided, `L` takes precedence.
          * Changing coordinate data after creation (`turbinesC`, `substationsC`,
              `borderC`, `obstaclesC`) rebuilds `L` and refreshes the navigation mesh
              and available links.

        Example::

          cables = [(3, 100.0), (5, 150.0)]
          turbines = np.array([[0, 0], [1, 0], [0, 1]])
          substations = np.array([[10, 0]])
          wfn = WindFarmNetwork(cables=cables, turbinesC=turbines, substationsC=substations)
          wfn.optimize()
          print(wfn.cost(), wfn.length())
        """
        # keep coord-related kwargs so rebuilds are consistent
        self._coord_kwargs = dict(kwargs)

        # simple fields via setters (for validation/normalization)
        self.name = name
        self.handle = handle
        self.buffer_dist = buffer_dist
        self.router = router if router is not None else EWRouter()
        self.cables = cables  # computes cables_capacity

        self.verbose = verbose

        # decide source of L
        if L is not None:
            if turbinesC is not None or substationsC is not None:
                warning(
                    'Both coordinates and L are given, OptiWindNet prioritizes L over coordinates.'
                )
        elif turbinesC is not None and substationsC is not None:
            L = from_coordinates(
                self,
                turbinesC,
                substationsC,
                borderC,
                obstaclesC,
                name,
                handle,
                buffer_dist,
            )
        else:
            raise TypeError(
                'Both turbinesC and substationsC must be provided! Alternatively, L should be given.'
            )
        self._L = L
        self._VertexC = L.graph['VertexC']
        self._R, self._T = L.graph['R'], L.graph['T']
        self._refresh_planar()

    # -------- helpers --------
    def _refresh_planar(self):
        self._P, self._A = make_planar_embedding(self._L)
        self._is_stale_PA = False

    # -------- properties --------
    @property
    def L(self):
        return self._L

    @property
    def P(self):
        if self._is_stale_PA:
            self._refresh_planar()
        return self._P

    @property
    def A(self):
        if self._is_stale_PA:
            self._refresh_planar()
        return self._A

    @property
    def S(self):
        if self._is_stale_SG:
            return None
        return self._S

    @property
    def G(self):
        if self._is_stale_SG:
            return None
        return self._G

    @property
    def cables(self):
        return self._cables

    @cables.setter
    def cables(self, cables):
        parsed = parse_cables_input(cables)
        self._cables = parsed
        self.cables_capacity = max(parsed)[0]
        if not self._is_stale_SG:
            assign_cables(self._G, cables)

    @property
    def router(self):
        return self._router

    @router.setter
    def router(self, router):
        self._router = router if router is not None else EWRouter()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = str(name)

    @property
    def handle(self):
        return self._handle

    @handle.setter
    def handle(self, handle):
        self._handle = str(handle)

    @property
    def buffer_dist(self):
        return self._buffer_dist

    @buffer_dist.setter
    def buffer_dist(self, dist):
        if not isinstance(dist, (int, float)):
            raise TypeError('buffer_dist must be numeric')
        self._buffer_dist = dist

    def cost(self):
        """Get the total cost of the optimized network."""
        return self.G.size(weight='cost')

    def length(self):
        """Get the total cable length of the optimized network."""
        return self.G.size(weight='length')

    def plot_original_vs_buffered(self, **kwargs):
        """Plot original and buffered borders and obstacles on a single plot.

        Args:
          **kwargs: passed to matplotlib's pyplot.figure()

        Returns:
          matplotlib Axes instance.
        """
        return plot_org_buff(
            self._borderC_original,
            self._border_bufferedC,
            self._obstaclesC_original,
            self._obstacles_bufferedC,
            **kwargs,
        )

    @classmethod
    def from_yaml(cls, filepath: str, **kwargs):
        """Create a WindFarmNetwork instance from a YAML file."""
        if not isinstance(filepath, str):
            raise TypeError('Filepath must be a string')

        L = L_from_yaml(filepath)
        return cls(L=L, **kwargs)

    @classmethod
    def from_pbf(cls, filepath: str, **kwargs):
        """Create a WindFarmNetwork instance from a PBF file."""
        if not isinstance(filepath, str):
            error('Filepath must be a string')

        L = L_from_pbf(filepath)
        return cls(L=L, **kwargs)

    @classmethod
    def from_windIO(cls, filepath: str, **kwargs):
        """Create a WindFarmNetwork instance from WindIO yaml file."""
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
        """Plot the optimized network."""
        return gplot(self.G, *args, **kwargs)

    def plot_location(self, **kwargs):
        """Plot the original location graph."""
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

        S = nx.Graph(R=R, T=T)
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
        """Get solver summary (runtime, objective, gap, etc.)."""
        return {
            k: self.G.graph[k]
            for k in ('runtime', 'bound', 'objective', 'relgap', 'termination')
        }


class EWRouter(Router):
    """A lightweight, ultra-fast router for electrical network optimization.

    * Uses a modified Esau-Williams heuristic (segmented or straight feeders).
    * Produces solutions in milliseconds, suitable for quick solutions or warm starts.
    """

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

    def route(self, P, A, cables, cables_capacity, verbose=None, **kwargs):
        if verbose is None:
            verbose = self.verbose

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

    def route(self, P, A, cables, cables_capacity, verbose=None, **kwargs):
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
                    'WARNING: HGSRouter is used for a plant with more than one '
                    'substation and feeder-limit is neglected (The current '
                    'implementation of HGSRouter does not support limiting the number '
                    'of feeders in multi-substation plants.)'
                )

        G_tentative = G_from_S(S, A)

        G = PathFinder(G_tentative, planar=P, A=A).create_detours()

        assign_cables(G, cables)

        return S, G


class MILPRouter(Router):
    """An exact router using mathematical programming.

    * Uses a Mixed-Integer Linear Programming (MILP) model of the problem.
    * Produces provably optimal or near-optimal networks (with quality metrics).
    * Requires a longer runtime than heuristics- and meta-heuristics-based routers.
    """

    def __init__(
        self,
        solver_name: str,
        time_limit: int,
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
        verbose=None,
        S_warm=None,
        S_warm_has_detour=False,
        **kwargs,
    ):
        if verbose is None:
            verbose = self.verbose

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

        solver.set_problem(
            P,
            A,
            capacity=cables_capacity,
            model_options=self.model_options,
            warmstart=S_warm,
        )

        solver.solve(
            time_limit=self.time_limit,
            mip_gap=self.mip_gap,
            options=self.solver_options,
            verbose=verbose,
        )

        S, G = solver.get_solution()

        assign_cables(G, cables)

        return S, G
