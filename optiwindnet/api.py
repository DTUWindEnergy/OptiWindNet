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
    validate_terse_links,
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


class WindFarmNetwork:
    """
    Representation of an offshore wind farm electrical network.

    The `WindFarmNetwork` class encapsulates the data structures and methods
    required to build, analyze, and optimize a wind farm's electrical cable layout.
    It supports initialization from raw coordinate data or from serialized formats
    (YAML, PBF), and provides methods for visualization, cost/length
    evaluation, and gradient-derivation.

    The network is modeled as a set of turbines, substations, site boundaries,
    and obstacles. From these, a location graph `L` is created, which can be
    embedded planarly (`P`, `A`), optimized (`S`, `G`).

    Parameters
    ----------
    cables : list or array-like
        Cable specifications, given as a list of `(capacity, cost)` tuples.
    turbinesC : array-like of shape (n_turbines, 2), optional
        Coordinates of wind turbines in Cartesian (x, y).
    substationsC : array-like of shape (n_substations, 2), optional
        Coordinates of substations in Cartesian (x, y).
    borderC : array-like of shape (n_border_points, 2), optional
        Polygonal coordinates defining the wind farm border.
    obstaclesC : array-like, optional
        One or more polygons defining exclusion zones (obstacles).
    name : str, default=""
        Human-readable name for the project.
    handle : str, default=""
        Short identifier for the project, typically used in filenames.
    L : networkx.Graph, optional (only if turbineC and SubstaionC are provided)
        Pre-constructed location graph. If provided, it takes precedence over coordinates.
    router : object, optional
        Routing algorithm instance. Defaults to `EWRouter` if not provided.
    buffer_dist : float, default=0
        Buffer distance used when inflating borders/obstacles.
    **kwargs : dict
        Additional keyword arguments forwarded to layout-construction helpers.

    Attributes
    ----------
    cables : list of tuple
        Parsed cable specifications, each entry is `(capacity, cost)`.
    cables_capacity : float
        Maximum capacity among all defined cables.
    L : networkx.Graph
        Location graph representing turbines, substations, borders, and obstacles.
    P : object
        Planar embedding of `L` (navigation mesh).
    A : networkx.Graph
        Adjacency graph of the planar embedding.
    S : networkx.Graph or None
        Current solution graph of selected links (edges between turbines and substations).
    G : networkx.Graph or None
        Current optimized network with assigned cables.
    name : str
        Project name.
    handle : str
        Project identifier.
    buffer_dist : float
        Border/obstacle buffer distance.
    router : object
        Router instance used for optimization.

    Methods
    -------
    cost()
        Return the total cost of the optimized network.
    length()
        Return the total cable length of the optimized network.
    plot(...)
        Plot the optimized network.
    plot_location(...)
        Plot the original location graph.
    plot_available_links(...)
        Plot available links from the planar embedding.
    plot_navigation_mesh(...)
        Plot navigation mesh with adjacency.
    plot_selected_links(...)
        Plot tentative link selection from available links.
    terse_links()
        Return compact representation of selected links.
    update_from_terse_links(terse_links, ...)
        Update the network from terse link encoding.
    get_network()
        Export the optimized network as a structured array.
    map_detour_vertex()
        Map detour vertices back to their original indices.
    gradient(...)
        Compute cable length/cost gradients wrt node positions.
    optimize(...)
        Run routing algorithm to find optimized cable layout using the selected router.
    solution_info()
        Return solver summary: runtime, objective, gap, etc.

    Notes
    -----
    - Changing coordinate inputs (`turbinesC`, `substationsC`, `borderC`, `obstaclesC`)
      automatically rebuilds the location graph `L` and refreshes its planar embedding.
    - If both `L` and coordinates are provided, `L` takes precedence unless overridden.
    - Cables are stored internally as list of tuples (capacities, costs).

    Examples
    --------
    >>> cables = [(100, 10), (200, 20)]
    >>> turbines = np.array([[0, 0], [1, 0], [0, 1]])
    >>> substations = np.array([[10, 0]])
    >>> wfn = WindFarmNetwork(cables, turbinesC=turbines, substationsC=substations)
    >>> wfn.optimize()
    >>> print(wfn.cost(), wfn.length())
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
        # keep coord-related kwargs so rebuilds are consistent
        self._coord_kwargs = dict(kwargs)

        # simple fields via setters (for validation/normalization)
        self.name = name
        self.handle = handle
        self.buffer_dist = buffer_dist
        self.router = router if router is not None else EWRouter()
        self.cables = cables  # computes cables_capacity

        # coordinates
        self._turbinesC = turbinesC
        self._substationsC = substationsC
        self._borderC = borderC
        self._obstaclesC = obstaclesC

        # decide source of L
        if L is not None:
            self._L = L
            self._refresh_planar()
            if turbinesC is not None and substationsC is not None:
                warning(
                    'Both coordinates and L are given, OptiWindNet prioritizes L over coordinates and neglects the provided L.'
                )
        elif turbinesC is not None and substationsC is not None:
            self._rebuild_L_from_coordinates()
        else:
            raise TypeError(
                'Both turbinesC and substationsC must be provided! Alternatively, L should be given.'
            )

        # placeholders
        self.S = None
        self.G = None

    # -------- helpers --------
    def _refresh_planar(self):
        self.P, self.A = make_planar_embedding(self._L)

    def _rebuild_L_from_coordinates(self):
        if self._turbinesC is None or self._substationsC is None:
            warning(
                'Coordinate changed but cannot rebuild L until both turbinesC and substationsC are set.'
            )
            return
        self._L = from_coordinates(
            self,
            self._turbinesC,
            self._substationsC,
            self._borderC,
            self._obstaclesC,
            self.name,
            self.handle,
            self.buffer_dist,
            **self._coord_kwargs,
        )
        self._refresh_planar()
        # downstream graphs depend on L/A → invalidate
        self.S = None
        self.G = None

    # -------- properties --------
    # L is read/write; writing L refreshes planar, and overrides coord-driven L
    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, value):
        self._L = value
        self._refresh_planar()
        self.S = None
        self.G = None

    @property
    def cables(self):
        return self._cables

    @cables.setter
    def cables(self, value):
        parsed = parse_cables_input(value)
        self._cables = parsed
        self.cables_capacity = max(parsed)[0]

    @property
    def router(self):
        return self._router

    @router.setter
    def router(self, value):
        self._router = value if value is not None else EWRouter()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def handle(self):
        return self._handle

    @handle.setter
    def handle(self, value):
        self._handle = str(value)

    @property
    def buffer_dist(self):
        return self._buffer_dist

    @buffer_dist.setter
    def buffer_dist(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError('buffer_dist must be numeric')
        self._buffer_dist = value

    # ---- coordinates:
    # changing any of these rebuilds L, then refreshes P/A
    @property
    def turbinesC(self):
        return self._turbinesC

    @turbinesC.setter
    def turbinesC(self, value):
        self._turbinesC = value
        self._rebuild_L_from_coordinates()

    @property
    def substationsC(self):
        return self._substationsC

    @substationsC.setter
    def substationsC(self, value):
        self._substationsC = value
        self._rebuild_L_from_coordinates()

    @property
    def borderC(self):
        return self._borderC

    @borderC.setter
    def borderC(self, value):
        self._borderC = value
        self._rebuild_L_from_coordinates()

    @property
    def obstaclesC(self):
        return self._obstaclesC

    @obstaclesC.setter
    def obstaclesC(self, value):
        self._obstaclesC = value
        self._rebuild_L_from_coordinates()

    def cost(self):
        """Returns the total cost of the network."""
        return self.G.size(weight='cost')

    def length(self):
        """Returns the total cable length of the network."""
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
        """Updates the network from terse link representation.

        Accepts integers or integer-like floats (e.g., 3.0). Rejects non-integers.
        """
        validated_terse_links = validate_terse_links(terse_links=terse_links, L=self.L)

        # Update coordinates if provided
        if turbinesC is not None:
            self.L.graph['VertexC'][: turbinesC.shape[0], :] = turbinesC

        if substationsC is not None:
            self.L.graph['VertexC'][-substationsC.shape[0] :, :] = substationsC

        if turbinesC is not None or substationsC is not None:
            self.P, self.A = make_planar_embedding(self.L)

        # Rebuild the selected edge set (links)
        if self.S is None:
            self.S = nx.Graph(R=self.L.graph['R'], T=self.L.graph['T'])
        else:
            self.S.clear_edges()

        for i, j in enumerate(validated_terse_links):
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

    def optimize(self, turbinesC=None, substationsC=None, router=None, verbose=False):
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

    def solution_info(self):
        return {
            k: self.G.graph[k]
            for k in ('runtime', 'bound', 'objective', 'relgap', 'termination')
        }


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
