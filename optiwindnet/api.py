import logging
from abc import ABC, abstractmethod
from itertools import pairwise
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import shapely as shp
from matplotlib.axes import Axes

from .api_utils import (
    buffer_border_obs,
    enable_ortools_logging_if_jupyter,
    extract_network_as_array,
    is_warmstart_eligible,
    merge_obs_into_border,
    parse_cables_input,
    plot_org_buff,
)
from .baselines.hgs import hgs_cvrp
from .heuristics import constructor
from .importer import L_from_pbf, L_from_site, L_from_windIO, L_from_yaml
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
from .MILP import ModelOptions, OWNSolutionNotFound, OWNWarmupFailed, solver_factory
from .pathfinding import PathFinder
from .plotting import gplot, pplot
from .svg import svgplot, svgpplot

##################################
# OptiWindNet Network/Router API #
##################################

# Keep text editable (not converted to paths) in SVG output
plt.rcParams['svg.fonttype'] = 'none'

# Set up a logger and create shortcuts for error, warning, and info logging methods
_logger = logging.getLogger(__name__)
_error, _warning, _info = _logger.error, _logger.warning, _logger.info


class Router(ABC):
    """Abstract base class for routing algorithms in OptiWindNet.

    Each Router implementation must define a :meth:`route` method.
    """

    _summary_attrs: tuple[str, ...]
    _repr_attrs: tuple[str, ...] = ()

    def __repr__(self) -> str:
        # Defensive by design: getattr-guard every field and skip None values so
        # the repr never raises, even on a partially-initialized instance.
        parts = [type(self).__name__]
        for attr in self._repr_attrs:
            val = getattr(self, attr, None)
            if val is not None:
                parts.append(f'{attr}={val!r}')
        return '<' + ' '.join(parts) + '>'

    @abstractmethod
    def route(
        self,
        P: nx.PlanarEmbedding,
        A: nx.Graph,
        cables: list[tuple[int, float | int]],
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
          Tuple of (solution topology (selected links), optimized route set).
        """
        pass


class WindFarmNetwork:
    """Wind farm electrical network.

    Wrapper of most of OptiWindNet's functionality (optimization, visualization,
    cost/length evaluation, and gradient calculation).

    An instance represents a wind farm location, which initially contains the number
    and positions of wind turbines and substations, the delimited area and eventual
    obstacles. A cable network may be provided or a :class:`Router` instance may be used
    to create an optimized network.
    """

    _is_stale_PA: bool = True
    _is_stale_SG: bool = True
    _is_stale_polygon: bool = True
    _buffer_dist: float = 0.0

    def __init__(
        self,
        cables: int | list[int] | list[tuple[int, float | int]] | np.ndarray,
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
          cables: properties of the available cable types (see Cable Specs below)
          turbinesC: Turbine coordinates (T, 2): ``[(x, y), ...]``.
          substationsC: Substation coordinates (R, 2): ``[(x, y), ...]``.
          borderC: Polygonal border coordinates (_, 2): ``[(x, y), ...]``.
          obstacleC_: One or more polygons for exclusion zones list of
            (_, 2): ``[[(x, y), ...], ...]``.
          name: Human-readable instance name. Defaults to "".
          handle: Short instance identifier. Defaults to "".
          L: Location geometry (takes precedence over coordinate inputs).
          router: Routing algorithm instance. Defaults to :class:`EWRouter`.
          buffer_dist: Buffer distance to dilate borders / erode obstacles.
            Defaults to 0.

        **Cable Specs** (``capacity`` is in number of turbines):
            * List of 2-tuple cable specifications:
              ``[(capacity, linear_cost), ...]``.

            * List of capacity per cable type:
              ``[capacity, ...]``.

            * Maximum capacity of all available cables (single value): ``capacity``.

        Note:
          * If both ``L`` and coordinates are provided, ``L`` takes precedence.
          * Changing coordinate data after creation (``turbinesC`` and/or
            ``substationsC``) rebuilds ``L`` and refreshes the navigation mesh
            and available links.

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
                _warning(
                    'Both coordinates and L are given, OptiWindNet prioritizes'
                    ' L over coordinates.'
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
                'Both turbinesC and substationsC must be provided!'
                ' Alternatively, L should be given.'
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
        "Triangular mesh over ``L`` (navigation mesh)."
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
            raise RuntimeError('Call the `optimize()` method to update S.')
        return self._S

    @property
    def G(self) -> nx.Graph:
        "Optimized network with cable routes and types."
        if self._is_stale_SG:
            raise RuntimeError('Call the `optimize()` method to update G.')
        return self._G

    @property
    def cables(self) -> list[tuple[int, float | int]]:
        "Set of cable specifications as ``[(capacity, linear_cost), ...]``."
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
            _logger.info('No buffering is performed')

    @classmethod
    def from_own_yaml(cls, filepath: str, **kwargs):
        """Create a WindFarmNetwork instance from an OptiWindNet (OWN) YAML file."""
        return cls(L=L_from_yaml(filepath), **kwargs)

    @classmethod
    def from_yaml(cls, filepath: str, **kwargs):
        """Deprecated: use :meth:`from_own_yaml` instead."""
        import warnings

        warnings.warn(
            'from_yaml() is deprecated, use from_own_yaml() instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.from_own_yaml(filepath, **kwargs)

    @classmethod
    def from_pbf(cls, filepath: Path | str, **kwargs):
        """Create a WindFarmNetwork instance from a .OSM.PBF file."""
        return cls(L=L_from_pbf(filepath), **kwargs)

    @classmethod
    def from_windIO(cls, filepath: Path | str, **kwargs):
        """Create a WindFarmNetwork instance from WindIO yaml file."""
        return cls(L=L_from_windIO(filepath), **kwargs)

    def __repr__(self) -> str:
        """Concise one-line summary for console/debugging.

        Defensive by design: instance attributes are getattr-guarded so the repr
        never raises, even on a partially-initialized instance (e.g. if :meth:`__init__`
        aborted before ``_T``/``_R`` were set). The solved-network branch is reached
        only when ``_is_stale_SG`` is ``False``, which guarantees ``_G`` exists.
        """
        handle = getattr(self, 'handle', '') or ''
        parts = [f'WindFarmNetwork {handle!r}'] if handle else ['WindFarmNetwork']
        name = getattr(self, 'name', '') or ''
        if name and name != handle:
            parts.append(f'name={name!r}')
        T = getattr(self, '_T', None)
        if T is not None:
            parts.append(f'T={T}')
        R = getattr(self, '_R', None)
        if R is not None:
            parts.append(f'R={R}')
        capacity = getattr(self, 'cables_capacity', None)
        if capacity is not None:
            parts.append(f'capacity={capacity}')
        router = getattr(self, '_router', None)
        if router is not None:
            parts.append(f'router={type(router).__name__}')
        if getattr(self, '_is_stale_SG', True):
            parts.append('unsolved')
        else:
            parts.append('length={:_.0f}'.format(self._G.size(weight='length')))
        return '<' + ' '.join(parts) + '>'

    def _repr_svg_(self):
        """IPython hook for rendering the graph as SVG in notebooks."""
        return svgplot(self.L if self._is_stale_SG else self.G)._repr_svg_()

    def plot(self, *args, **kwargs):
        """Plot the optimized network.

        By default, this method utilizes the modern vector SVG-based plotting
        backend (:func:`svgplot`) which returns an :class:`SvgRepr` suitable for clean
        interactive inline displays in Jupyter notebooks.

        To switch to the Matplotlib-based plotting backend (:func:`gplot`), specify the
        ``ax`` parameter as a keyword argument.

        Note:
          Passing ``ax=None`` explicitly routes to the Matplotlib backend and
          automatically instantiates a new figure and axes on the fly, allowing
          Matplotlib figures to be created without importing ``matplotlib`` or
          ``pyplot`` directly in the user code.
        """
        if 'ax' in kwargs:
            return gplot(self.G, *args, **kwargs)
        return svgplot(self.G, *args, **kwargs)

    def plot_location(self, **kwargs):
        """Plot the original location geometry."""
        if 'ax' in kwargs:
            return gplot(self.L, **kwargs)
        return svgplot(self.L, **kwargs)

    def plot_available_links(self, **kwargs):
        """Plot available links from planar embedding."""
        if 'ax' in kwargs:
            return gplot(self.A, **kwargs)
        return svgplot(self.A, **kwargs)

    def plot_navigation_mesh(self, **kwargs):
        """Plot navigation mesh (planar graph and adjacency)."""
        if 'ax' in kwargs:
            return pplot(self.P, self.A, **kwargs)
        return svgpplot(self.P, self.A, **kwargs)

    def plot_selected_links(self, **kwargs):
        """Plot link selection (tentative feeder routes)."""
        G_tentative = G_from_S(self.S, self.A)
        assign_cables(G_tentative, self.cables)
        if 'ax' in kwargs:
            return gplot(G_tentative, **kwargs)
        return svgplot(G_tentative, **kwargs)

    def terse_links(self):
        """Get a compact representation of the solution topology."""
        T = self.S.graph['T']
        terse = np.empty(T, dtype=int)

        for u, v, reverse in self.S.edges(data='reverse'):
            if reverse is None:
                _error('reverse must not be None')
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
        """Dilate the cable-laying area by ``buffer_dist``.

        Useful if boundaries are not strictly enforced during optimization. This may
        happen if boundary compliance is achieved through the application of penalties
        for violations. OptiWindNet will fail if turbines are outside the border, so
        choose a ``buffer_dist`` that is greater than the maximum single step in
        position.

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
        """Get model and solver information of the latest solution
        (runtime, objective, gap, etc.)."""
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

    * Uses a modified Esau-Williams (EW) heuristic (segmented or straight feeders).
    * Produces solutions in milliseconds, suitable for quick solutions or warm starts.
    """

    _summary_attrs = ('iterations',)
    _repr_attrs = ('method', 'maxiter', 'feeder_route', 'bias_margin')

    def __init__(
        self,
        maxiter: int = 10_000,
        feeder_route: str = 'segmented',
        method: str = 'biased_EW',
        bias_margin: float | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Create a Esau-Williams-based router.

        Available Methods:
          ``'esau_williams'``
            Esau-Williams C-MST heuristic modified to avoid crossings (EW).
          ``'biased_EW'``
            EW with a bias towards moving radially (root-ward) on quasi-ties.
          ``'rootlust'``
            EW with a tunable root-ward bias that increases as capacity decreases.
          ``'radial_EW'``
            EW variant that produces radial subtrees (simple paths from root).
          ``'ringed'``
            Closes each subtree into a ring: both endpoints connect to the same
            root (two feeders) joined at an open point. ``cables_capacity`` is
            the per-arm limit, so a ring holds up to twice as many terminals.
            Unions are ranked by their total saving — the feeders shed at the
            two joined endpoints minus the connecting edge's length
            (Clarke-Wright style) — with a ``bias_margin`` window favoring the
            more root-ward union on quasi-ties.

        Args:
          maxiter: Maximum iterations.
          feeder_route: Feeder routing mode (``'segmented'`` or ``'straight'``).
          method: one of the **Available Methods**, defaults to ``'biased_EW'``).
          bias_margin: Fractional margin within which candidates are considered
            equivalent, resolving the quasi-tie root-ward (used by
            ``'biased_EW'``, ``'radial_EW'`` and ``'ringed'``; for ``'ringed'``
            the margin is a fraction of the best union ``saving`` rather than the
            edge ``extent``).
            Defaults to the constructor's built-in default (0.02) when ``None``.
          verbose: Enable verbose logging.
        """

        super().__init__(**kwargs)

        self.verbose = verbose
        self.maxiter = maxiter
        self.feeder_route = feeder_route
        self.method = method
        self.bias_margin = bias_margin

    def route(self, P, A, cables, cables_capacity, verbose=False, **kwargs):
        constructor_args = dict(method=self.method, maxiter=self.maxiter)
        if self.bias_margin is not None:
            constructor_args['bias_margin'] = self.bias_margin
        if self.feeder_route == 'segmented':
            constructor_args.update(weigh_detours=True, straight_feeder_route=False)
        elif self.feeder_route == 'straight':
            constructor_args.update(weigh_detours=False, straight_feeder_route=True)
        else:
            raise ValueError(
                f'{self.feeder_route} is not among the valid feeder_route values.'
                ' Choose among: ("segmented", "straight").'
            )

        S = constructor(A, capacity=cables_capacity, **constructor_args)
        G_tentative = G_from_S(S, A)

        # RINGED subtrees are closed loops: PathFinder must resolve crossings in
        # ringed mode (matching the other ringed solver paths)
        ringed = self.method == 'ringed'
        G = PathFinder(G_tentative, planar=P, A=A, ringed=ringed).create_detours()

        assign_cables(G, cables)

        return S, G


class HGSRouter(Router):
    """A fast router based on Hybrid Genetic Search (HGS-CVRP).

    Uses the method and implementation by Vidal, 2022:
      Vidal, T. (2022). Hybrid genetic search for the CVRP: Open-source implementation
      and SWAP* neighborhood. Computers & Operations Research, 140, 105643.
      https://doi.org/10.1016/j.cor.2021.105643

    * Balances solution quality and runtime.
    * Produces radial solutions, or RINGED (closed-loop) solutions with ``ringed=True``.
    """

    _summary_attrs = ('runtime',)
    _repr_attrs = (
        'time_limit',
        'feeder_limit',
        'feeder_exact',
        'max_retries',
        'balanced',
        'ringed',
        'seed',
    )

    def __init__(
        self,
        time_limit: float,
        feeder_limit: int | None = None,
        feeder_exact: bool = False,
        max_retries: int = 10,
        balanced: bool = False,
        ringed: bool = False,
        seed: int | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Create an HGS-based router.

        Args:
          time_limit: Maximum runtime for a single HGS run (in seconds).
          feeder_limit: Maximum number of feeders allowed
              (ignored if multiple substations); the exact number if ``feeder_exact``.
          feeder_exact: Whether ``feeder_limit`` is the exact number of feeders,
              instead of an upper bound (requires ``balanced=True`` and a single
              substation).
          max_retries: Maximum number of retries if a feasible solution is not found.
          balanced: Whether to balance turbines/loads across feeders.
          ringed: Whether to produce a RINGED topology (closed loops) instead of a
              radial one. HGS then solves the closed CVRP, so each ring holds up to
              ``2 * cables_capacity`` turbines (two arms of ``cables_capacity`` each).
          seed: Set the seed of the pseudo-random number generator (reproducibility).
          verbose: Enable verbose logging.

        Note:
          The total runtime may reach up to ``(max_retries + 1) * time_limit`` in the
          worst case.
        """
        # Call the base class initialization
        super().__init__(**kwargs)
        self.time_limit = time_limit
        self.verbose = verbose
        self.max_retries = max_retries
        self.feeder_limit = feeder_limit
        self.feeder_exact = feeder_exact
        self.balanced = balanced
        self.ringed = ringed
        self.seed = seed

    def route(self, P, A, cables, cables_capacity, verbose=False, **kwargs):
        # optimizing
        S = hgs_cvrp(
            as_normalized(A),
            capacity=cables_capacity,
            time_limit=self.time_limit,
            max_retries=self.max_retries,
            vehicles=self.feeder_limit,
            vehicles_exact=self.feeder_exact,
            balanced=self.balanced,
            ringed=self.ringed,
            seed=self.seed,
        )

        G_tentative = G_from_S(S, A)

        G = PathFinder(
            G_tentative, planar=P, A=A, branched=False, ringed=self.ringed
        ).create_detours()

        assign_cables(G, cables)

        return S, G


class MILPRouter(Router):
    """An exact router using mathematical programming.

    * Uses a Mixed-Integer Linear Programming (MILP) model of the problem.
    * Produces provably optimal or near-optimal networks (with quality metrics).
    * Requires a longer runtime than heuristics- and meta-heuristics-based routers.
    """

    default_heuristic = 'rootlust'
    _summary_attrs = ('runtime', 'bound', 'objective', 'relgap', 'termination')
    _repr_attrs = ('solver_name', 'time_limit', 'mip_gap')

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
          solver_name: Name of solver (e.g., ``'gurobi'``, ``'cbc'``, ``'ortools'``,
            ``'cplex'``, ``'highs'``, ``'scip'``).
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
            self.optiwindnet_default_options = {}

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
                logger=_logger,
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
                        constructor_args = dict(
                            method=self.default_heuristic,
                            weigh_detours=True,
                            straight_feeder_route=False,
                        )
                    elif feeder_route == 'straight':
                        constructor_args = dict(
                            method=self.default_heuristic,
                            weigh_detours=False,
                            straight_feeder_route=True,
                        )
                    S_warm = S_from_G(
                        constructor(A, capacity=cables_capacity, **constructor_args)
                    )
                else:
                    # a RINGED model warmstarts from a ringed solution, a radial
                    # model from a radial one
                    S_warm = hgs_cvrp(
                        as_normalized(A),
                        capacity=cables_capacity,
                        time_limit=min(self.time_limit, 0.2),
                        repair=True,
                        ringed=self.model_options['topology'] == 'ringed',
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
                f'Unable to find a solution to the MILP model'
                f' after {num_retries} retries'
            )

        S, G = solver.get_solution()

        assign_cables(G, cables)

        return S, G
