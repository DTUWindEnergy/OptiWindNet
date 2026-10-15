# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import shapely as shp

from .api_utils import (
    buffer_border_obs,
    enable_ortools_logging_if_jupyter,
    extract_network_as_array,
    merge_obs_into_border,
    parse_cables_input,
)
from .baselines.hgs import hgs_cvrp
from .constraints import NetworkConstraints, _resolved
from .converting import G_from_S, S_from_G, terse_links_from_S
from .heuristics import constructor
from .heuristics.constructor import _METHOD_TOPOLOGY
from .importer import (
    L_from_pbf,
    L_from_site,
    L_from_windIO,
    L_from_yaml,
    load_repository,
)
from .interarraylib import assign_cables, links_over_length
from .loads import (
    DEFAULT_POWER_RTOL,
    _clear_terminal_power,
    _is_positive_integer,
    _validated_capacity_nominal,
    _validated_rtol,
    nonunit_inflow,
    set_terminal_power,
    validate_terminal_power,
)
from .mesh import make_planar_embedding
from .MILP import OWNSolutionNotFound, OWNWarmupFailed, solver_factory
from .pathfinding import PathFinder
from .presenting import _nominal_loads
from .svg import SvgRepr, svgplot, svgpplot
from .terse import LinkScope, TerseLinks
from .transforming import as_normalized, as_stratified_vertices
from .types import FeederRoute, NodeHighlight, Topology

if TYPE_CHECKING:
    from matplotlib.axes import Axes

##################################
# OptiWindNet Network/Router API #
##################################

# Set up a logger and create shortcuts for error, warning, and info logging methods
_logger = logging.getLogger(__name__)
_error, _warning, _info = _logger.error, _logger.warning, _logger.info

# `NetworkConstraints` and `load_repository` are defined elsewhere but
# belong to this module's surface: the notebooks reach for them as
# `from optiwindnet.api import ...`. Naming them here puts them in this page's outline,
# pointing at where they are documented.
__all__ = (
    'EWRouter', 'HGSRouter', 'MILPRouter', 'NetworkConstraints',
    'Router', 'WindFarmNetwork', 'load_repository',
)  # fmt: skip

_EMPTY_BORDER = np.empty((0, 2), dtype=np.float64)


class Router(ABC):
    """Abstract base class for routing algorithms in OptiWindNet.

    Each Router implementation must define a :meth:`route` method.
    """

    _summary_attrs: tuple[str, ...]
    _repr_attrs: tuple[str, ...] = ()
    power_rtol: float = DEFAULT_POWER_RTOL

    def __init__(self, power_rtol: float = DEFAULT_POWER_RTOL, **kwargs) -> None:
        """Set the options common to all routers.

        Args:
          power_rtol: quantization tolerance of unequal turbine power (see
            :func:`~optiwindnet.loads.quantize_for_capacity`).
        """
        super().__init__(**kwargs)
        _validated_rtol(power_rtol)
        self.power_rtol = power_rtol

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
        cables: list[tuple[int, float | int]] | list[tuple[Fraction, float | int]],
        constraints: NetworkConstraints,
        verbose: bool = False,
        **kwargs,
    ) -> tuple[nx.Graph, nx.Graph]:
        """Run the routing optimization.

        Args:
          P : Navigation mesh for the location.
          A : Graph of available links.
          cables: set of cable specifications as [(capacity, linear_cost), ...].
          constraints: Network constraints, cable capacity included (see
            :class:`~optiwindnet.constraints.NetworkConstraints`); they are
            resolved for ``A`` at the router's ``power_rtol``.
          verbose : Whether to print progress/logging info.
          **kwargs : Additional router-specific parameters.

        Returns:
          Tuple of (solution topology (selected links), optimized routeset).
        """


def _validated_inflow(turbine_powers: Sequence | np.ndarray, T: int) -> list[int]:
    """Per-turbine integer inflow, as ``turbine_powers`` gives it without a unit."""
    if len(turbine_powers) != T:
        raise ValueError(
            f'turbine_powers has {len(turbine_powers)} entries but T={T} turbines.'
        )
    for value in turbine_powers:
        if not _is_positive_integer(value):
            raise ValueError(
                f'turbine_powers must be positive integers (inflow) if power_unit '
                f'is None: got {value!r}.'
            )
    return [int(value) for value in turbine_powers]


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
        cables: float | Sequence[float] | Sequence[tuple[float, float]] | np.ndarray,
        turbinesC: np.ndarray | None = None,
        substationsC: np.ndarray | None = None,
        borderC: np.ndarray = _EMPTY_BORDER,
        obstacleC_: Sequence[np.ndarray] = [],
        name: str = '',
        handle: str = '',
        L: nx.Graph | None = None,
        router: Router | None = None,
        verbose: bool = False,
        turbine_powers: Sequence[float | Fraction] | np.ndarray | None = None,
        power_unit: str | None = None,
        constraints: NetworkConstraints | None = None,
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
          verbose: Enable verbose logging during network construction.
          turbine_powers: Enables mixed power ratings (not required if the location
            uses uniform generators). Passed as a sequence of power values in the
            same order as the turbine coordinates: positive integers (inflow) if
            ``power_unit`` is None, nominal power in ``power_unit`` otherwise.
          power_unit: Physical unit (e.g. ``'MW'``) of ``turbine_powers`` and of the
            cable capacities. If None (default), both are integer inflow, which
            count turbines wherever each injects unitary inflow. If a string, each
            solve quantizes unequal powers for the largest capacity at the router's
            ``power_rtol`` (see :func:`~optiwindnet.loads.quantized`); the powers
            come from ``L`` if ``turbine_powers`` is omitted. A location ``L`` that
            declares a ``'power_unit'`` requires the same string, or None, in which
            case its power declarations are dropped.
          constraints: Constraints on the network: cable capacity, topology,
            feeder route, feeder counts, substation loads, load balance and link
            length (see :class:`~optiwindnet.constraints.NetworkConstraints`).
            Per-substation values follow the order of the substations
            (``substationsC``, or roots ``-R`` to ``-1`` of ``L``). Without a
            capacity, the largest cable capacity applies (as ``capacity_nominal``
            in ``power_unit``, or as ``capacity`` in inflow if ``power_unit`` is
            None); a capacity above it raises ``ValueError`` when optimizing. If
            None (default), only the cable capacity constrains the network. With
            a ``max_link_length``, :meth:`optimize` warns if detours stretch a
            link past it.

        **Cable Specs** (``capacity`` in inflow if ``power_unit`` is None, in
        ``power_unit`` otherwise):

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
        self.constraints = constraints

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
                border=(
                    np.arange(T, T + borderC.shape[0])
                    if (borderC is not None and borderC.shape[0] >= 3)
                    else None
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

        L_power_unit = L.graph.get('power_unit')
        if power_unit is None:
            declared = {'power_unit', 'powers_set', 'power_per_inflow'} & L.graph.keys()
            if 'powers_set' in declared:
                _warning(
                    'L declares unequal turbine power in %s, which is ignored '
                    'because power_unit is None: every turbine injects unitary '
                    'inflow. Pass power_unit=%r to route by the declared power.',
                    L_power_unit,
                    L_power_unit,
                )
            elif declared:
                _logger.info(
                    'L declares uniform turbine power in %s, which is dropped '
                    'because power_unit is None: capacities count turbines.',
                    L_power_unit,
                )
            if declared or turbine_powers is not None:
                _clear_terminal_power(L)
            if turbine_powers is not None:
                inflow = _validated_inflow(turbine_powers, T)
                nx.set_node_attributes(
                    L, {t: k for t, k in enumerate(inflow) if k != 1}, 'inflow'
                )
            validate_terminal_power(L)
        else:
            if L_power_unit not in (None, power_unit):
                raise ValueError(
                    f'power_unit={power_unit!r} differs from the power_unit '
                    f'{L_power_unit!r} that L declares.'
                )
            if turbine_powers is None:
                if L_power_unit is None:
                    raise ValueError(
                        f'power_unit={power_unit!r} requires turbine_powers, or a '
                        'location L that declares its power_unit.'
                    )
                validate_terminal_power(L)
            else:
                set_terminal_power(L, turbine_powers, power_unit)
        self._assign_cables(cables)

    # -------- helpers --------
    def _assign_cables(self, cables) -> None:
        """Parse the cable specifications and reassign the cables of the solution.

        Capacities are nominal power if ``L`` has a ``'power_unit'``, and inflow
        otherwise. A solution whose largest load exceeds the largest capacity is
        invalidated; one that fits keeps its links and gets its cables reassigned.
        """
        L, T = self._L, self._T
        raw = parse_cables_input(cables)
        nominal = 'power_unit' in L.graph
        if nominal:
            parsed = [
                (_validated_capacity_nominal(capacity), cost) for capacity, cost in raw
            ]
            largest_turbine = L.graph.get(
                'powers_set', (L.graph.get('power_per_inflow', 1),)
            )[-1]
        else:
            for capacity, _ in raw:
                if not _is_positive_integer(capacity):
                    raise ValueError(
                        f'Cable capacity {capacity!r} is not a positive integer: '
                        'capacities are inflow if power_unit is None.'
                    )
            parsed = [(int(capacity), cost) for capacity, cost in raw]
            largest_turbine = max(
                (L.nodes[t].get('inflow', 1) for t in range(T)), default=1
            )
        cables_capacity = max(parsed)[0]
        if largest_turbine > cables_capacity:
            raise ValueError(
                f'Maximum turbine power {float(largest_turbine):g} exceeds maximum '
                f'cable capacity {float(cables_capacity):g}.'
            )

        if not self._is_stale_SG:
            G = self._G
            max_load = (
                max(load for n, load in _nominal_loads(G).items() if n >= 0)
                if nominal
                else G.graph['max_load']
            )
            if max_load > cables_capacity:
                self._is_stale_SG = True
            else:
                assign_cables(G, parsed)
        self._cables = parsed
        self.cables_capacity = cables_capacity
        'highest cable capacity in cables.'

    def _refresh_planar(self):
        polygon = self.polygon
        if polygon is not None:
            # check if any of the new turbine coordinates lie outside the polygon
            if isinstance(polygon, shp.Polygon):
                out_of_bounds = shp.MultiPoint(self._VertexC[: self._T]) - polygon
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
    def cables(
        self,
    ) -> list[tuple[int, float | int]] | list[tuple[Fraction, float | int]]:
        """Cable specifications as ``[(capacity, linear_cost), ...]``.

        Capacities are exact fractions in :attr:`power_unit` where it is set, and
        integer inflow otherwise.
        """
        return self._cables

    @cables.setter
    def cables(self, cables):
        self._assign_cables(cables)

    @property
    def turbine_powers(self) -> list[Fraction] | list[int] | None:
        """Per-turbine power, or None if every turbine injects unitary inflow.

        With a :attr:`power_unit`, the nominal power each turbine declares, as
        exact fractions. Without, the integer inflow of each turbine.
        """
        L, T = self._L, self._T
        if self.power_unit is not None:
            if 'powers_set' in L.graph:
                return [L.nodes[t]['power'] for t in range(T)]
            return [Fraction(L.graph.get('power_per_inflow', 1))] * T
        if not nonunit_inflow(L):
            return None
        return [L.nodes[t].get('inflow', 1) for t in range(T)]

    @property
    def power_unit(self) -> str | None:
        "Physical unit of the turbine power and cable capacities, or None for inflow."
        return self._L.graph.get('power_unit')

    @property
    def constraints(self) -> NetworkConstraints | None:
        "Constraints on the network (None for the cable capacity only)."
        return self._constraints

    @constraints.setter
    def constraints(self, constraints: NetworkConstraints | None):
        if constraints is not None and not isinstance(constraints, NetworkConstraints):
            raise TypeError(
                'constraints must be a NetworkConstraints instance, got '
                f'{type(constraints).__name__}.'
            )
        self._constraints = constraints

    def _solve_constraints(self) -> NetworkConstraints:
        """Return the constraints to solve with, the cable capacity included.

        Raises:
          ValueError: the constraints' capacity exceeds the largest cable's.
        """
        constraints = self._constraints or NetworkConstraints()
        cables_capacity = self.cables_capacity
        nominal = 'power_unit' in self._L.graph
        if constraints.capacity is None and constraints.capacity_nominal is None:
            if nominal:
                return replace(constraints, capacity_nominal=Fraction(cables_capacity))
            return replace(constraints, capacity=int(cables_capacity))
        if constraints.capacity_nominal is not None:
            capacity = constraints.capacity_nominal
        else:
            capacity = constraints.capacity
            assert capacity is not None
            if nominal:
                # a capacity in inflow of a farm whose cables are in power_unit
                capacity *= self._L.graph.get('power_per_inflow', 1)
        if capacity > cables_capacity:
            raise ValueError(
                f'The capacity of the constraints ({float(capacity):g}) exceeds that '
                f'of the largest cable ({float(cables_capacity):g}).'
            )
        return constraints

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

    def plot_original_vs_buffered(self, **kwargs) -> 'Axes | SvgRepr | None':
        """Plot the location with the borders and obstacles prior to buffering.

        The buffered boundaries are drawn as the location's border, the original
        ones as outlines on top of it. This is :meth:`plot_location` for a
        location that went through :meth:`add_buffer`.

        Args:
          **kwargs: passed to :meth:`plot_location`.

        Returns:
          SvgRepr or matplotlib Axes instance; ``None`` if no buffering was done.
        """
        if '_original_boundaries' not in self._L.graph:
            _logger.info('No buffering is performed')
            return None
        return self.plot_location(**kwargs)

    @classmethod
    def from_own_yaml(cls, filepath: Path | str, **kwargs):
        """Create a WindFarmNetwork instance from an OptiWindNet (OWN) YAML file."""
        return cls(L=L_from_yaml(filepath), **kwargs)

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
        never raises, even on a partially-initialized instance (e.g. if ``__init__``
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
        backend (:func:`~optiwindnet.svg.svgplot`) which returns an
        :class:`~optiwindnet.svg.SvgRepr` suitable for clean interactive inline
        displays in Jupyter notebooks.

        To switch to the Matplotlib-based plotting backend
        (:func:`~optiwindnet.plotting.gplot`), specify the ``ax`` parameter as a
        keyword argument.

        Note:
          Passing ``ax=None`` explicitly routes to the Matplotlib backend and
          automatically instantiates a new figure and axes on the fly, allowing
          Matplotlib figures to be created without importing ``matplotlib`` or
          ``pyplot`` directly in the user code.
        """
        if 'ax' in kwargs:
            from .plotting import gplot

            return gplot(self.G, *args, **kwargs)
        return svgplot(self.G, *args, **kwargs)

    def plot_location(self, **kwargs):
        """Plot the original location geometry."""
        if 'ax' in kwargs:
            from .plotting import gplot

            return gplot(self.L, **kwargs)
        return svgplot(self.L, **kwargs)

    def plot_available_links(self, **kwargs):
        """Plot available links from planar embedding.

        Links longer than the constraints' ``max_link_length``, which no router
        may use, are left out of the plot (``A`` itself keeps them). Turbines
        without an available direct feeder use the theme's highlight outline.
        """
        A = self.A
        max_link_length = (
            None if self._constraints is None else self._constraints.max_link_length
        )
        if max_link_length is not None:
            T = A.graph['T']
            no_feeder_nodes = frozenset(
                np.flatnonzero(
                    (A.graph['d2roots'][:T] > max_link_length).all(axis=1)
                ).tolist()
            )
            adj = A.adj
            A = nx.subgraph_view(
                A, filter_edge=lambda u, v: adj[u][v]['length'] <= max_link_length
            )
            A.graph = A.graph.copy()
            A.graph['_node_highlight'] = NodeHighlight('no feeder', no_feeder_nodes)
        if 'ax' in kwargs:
            from .plotting import gplot

            return gplot(A, **kwargs)
        return svgplot(A, **kwargs)

    def plot_navigation_mesh(self, **kwargs):
        """Plot navigation mesh (planar graph and adjacency)."""
        if 'ax' in kwargs:
            from .plotting import pplot

            return pplot(self.P, self.A, **kwargs)
        return svgpplot(self.P, self.A, **kwargs)

    def plot_selected_links(self, **kwargs):
        """Plot link selection (tentative feeder routes)."""
        G_tentative = G_from_S(self.S, self.A)
        assign_cables(G_tentative, self._cables)
        if 'ax' in kwargs:
            from .plotting import gplot

            return gplot(G_tentative, **kwargs)
        return svgplot(G_tentative, **kwargs)

    def terse_links(self, *, routed: bool = False) -> TerseLinks:
        """Get a compact representation of the solution.

        By default the representation contains topology ``S`` and can be routed
        again against updated coordinates. With ``routed=True`` it contains the
        exact routeset ``G``, including contour and detour clone mappings.
        """
        return (
            TerseLinks.from_routeset(self.G) if routed else terse_links_from_S(self.S)
        )

    def update_from_terse_links(
        self,
        terse_links: TerseLinks | np.ndarray,
        turbinesC: np.ndarray | None = None,
        substationsC: np.ndarray | None = None,
        topology: Topology | str | None = None,
    ):
        """Update the network from terse link representation.

        A topology-scoped value is routed against this network's location; a
        routeset-scoped value restores its recorded contour and detour paths.
        Plain arrays remain accepted for topology input and may contain integers
        or integer-like floats (e.g., 3.0). Pass ``topology`` with a plain array
        when radial/branched identity must be retained.
        """
        T = self._T
        R = self._R

        if isinstance(terse_links, TerseLinks):
            encoding = terse_links
            if topology is not None and Topology(topology) is not encoding.topology:
                raise ValueError('topology disagrees with the TerseLinks value')
        else:
            terse_links_ints = np.asarray(terse_links, dtype=np.int64)
            encoding = TerseLinks.from_array(
                terse_links_ints.tolist(), topology=topology, R=R, T=T
            )
        if encoding.T != T or encoding.R != R:
            raise ValueError(
                'TerseLinks dimensions do not match this WindFarmNetwork: '
                f'(T={encoding.T}, R={encoding.R}) != (T={T}, R={R})'
            )
        if encoding.scope is LinkScope.ROUTESET and (
            turbinesC is not None or substationsC is not None
        ):
            raise ValueError(
                'routed TerseLinks are site-specific; use a topology encoding '
                'when updating coordinates'
            )

        # Update coordinates if provided
        if turbinesC is not None:
            self._VertexC[:T] = turbinesC
            self._is_stale_PA = True

        if substationsC is not None:
            self._VertexC[-R:] = substationsC
            self._is_stale_PA = True

        constraints = self._solve_constraints()
        capacity_kwargs: dict[str, Any] = (
            {'capacity': constraints.capacity}
            if constraints.capacity is not None
            else {
                'capacity_nominal': constraints.capacity_nominal,
                'power_rtol': self.router.power_rtol,
            }
        )
        if encoding.scope is LinkScope.ROUTESET:
            self._G = encoding.to_routeset(
                self._L, **capacity_kwargs, creator='from_terse_links'
            )
            self._S = S_from_G(self._G)
        else:
            S = encoding.to_topology(
                self._L, **capacity_kwargs, creator='from_terse_links'
            )
            G_tentative = G_from_S(S, self.A)
            self._S = S
            self._G = PathFinder(G_tentative, planar=self.P, A=self.A).create_detours()

        assign_cables(self._G, self._cables)
        self._is_stale_SG = False

    def get_network(self):
        """Export the optimized network with loads in nominal power units."""
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
        L = buffer_border_obs(self._L, buffer_dist=buffer_dist)
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

        warmstart = {} if self._is_stale_SG else {'S_warm': self._S}

        self._S, self._G = router.route(
            P=self.P,
            A=self.A,
            cables=self._cables,
            constraints=self._solve_constraints(),
            verbose=verbose,
            **warmstart,
        )
        self._is_stale_SG = False

        max_link_length = (
            None if self._constraints is None else self._constraints.max_link_length
        )
        if max_link_length is not None:
            self._G.graph['max_link_length'] = max_link_length
            over_limit = links_over_length(self._G, max_link_length)
            if over_limit:
                u, v, longest = over_limit[0]
                _warning(
                    '%d link(s) of the routed network exceed max_link_length=%g, '
                    'the longest being ⟨%d, %d⟩ at %g (%.1f%% over): detours '
                    'stretched them past it.',
                    len(over_limit),
                    max_link_length,
                    u,
                    v,
                    longest,
                    100.0 * (longest / max_link_length - 1.0),
                )

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
    * Keeps the topology and feeder route of the network constraints, but not
      feeder counts, substation loads, balance or link length (see
      :func:`~optiwindnet.heuristics.constructor`).
    """

    _summary_attrs = ('iterations',)
    _repr_attrs = ('method', 'maxiter', 'bias_margin')

    def __init__(
        self,
        maxiter: int = 10_000,
        method: str | None = None,
        bias_margin: float | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Create a Esau-Williams-based router.

        Available Methods:
          ``'esau_williams'``
            Esau-Williams C-MST heuristic modified to avoid crossings (EW).
          ``'biased_EW'``
            EW that favors growth towards the substation when candidates are
            nearly tied (the ``bias_margin`` defines the tie tolerance).
          ``'rootlust'``
            EW with a tunable preference for growth towards the substation,
            increasing as remaining capacity decreases.
          ``'radial_EW'``
            EW variant that connects each group of turbines along a single path
            from a substation (also uses ``bias_margin``).
          ``'ringed'``
            A mix of EW and Clarke-Wright algorithms to connect groups of turbines
            using a cyclic path starting and ending on a substation (also uses
            ``bias_margin``). Since each turbine group has two feeders, a ring can
            generate up to twice the total power of the groups in other topologies.

        The method follows the topology of the network constraints:
        ``'rootlust'`` for branched, ``'radial_EW'`` for radial and ``'ringed'``
        for ringed networks. Choosing another one is an advanced option, which
        must build networks within the topology (a radial one within branched).

        Args:
          maxiter: Maximum iterations.
          method: one of the **Available Methods** (defaults to the one of the
            topology).
          bias_margin: Fractional margin within which candidates are considered
            equivalent, with a preference for growth towards the substation (used by
            ``'biased_EW'``, ``'radial_EW'`` and ``'ringed'``; for ``'ringed'``
            the margin is a fraction of the best union ``saving`` rather than the
            edge ``extent``).
            Defaults to the constructor's built-in default (0.02) when ``None``.
          verbose: Unused: the heuristic is Python code, whose messages go
            to the ``optiwindnet.heuristics`` logger.
          **kwargs: Options common to all routers (see :class:`Router`).
        """

        super().__init__(**kwargs)
        if method is not None and method not in _METHOD_TOPOLOGY:
            raise ValueError(
                f'{method!r} is not among the methods of EWRouter: '
                f'{tuple(_METHOD_TOPOLOGY)}.'
            )
        self.verbose = verbose
        self.maxiter = maxiter
        self.method = method
        self.bias_margin = bias_margin

    def route(self, P, A, cables, constraints, verbose=False, **kwargs):
        S = constructor(
            A,
            constraints,
            method=self.method,
            power_rtol=self.power_rtol,
            maxiter=self.maxiter,
            bias_margin=self.bias_margin,
        )
        G_tentative = G_from_S(S, A)

        G = PathFinder(G_tentative, planar=P, A=A).create_detours()

        assign_cables(G, cables)

        return S, G


def _print_log_line(line: str) -> None:
    # HGS-CVRP log lines already end with a newline
    print(line, end='')


class HGSRouter(Router):
    """A fast router based on Hybrid Genetic Search (HGS-CVRP).

    Uses the method and implementation by Vidal, 2022:
      Vidal, T. (2022). Hybrid genetic search for the CVRP: Open-source implementation
      and SWAP* neighborhood. Computers & Operations Research, 140, 105643.
      https://doi.org/10.1016/j.cor.2021.105643

    * Balances solution quality and runtime.
    * Produces radial solutions, or ringed ones for a ringed topology.
    * Keeps every network constraint (see
      :func:`~optiwindnet.baselines.hgs.hgs_cvrp`), except straight feeders.
    """

    _summary_attrs = ('runtime',)
    _repr_attrs = ('time_limit', 'max_retries', 'seed')

    def __init__(
        self,
        time_limit: float,
        max_retries: int = 10,
        seed: int | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Create an HGS-based router.

        Args:
          time_limit: Maximum runtime for a single HGS run (in seconds).
          max_retries: Maximum number of retries if a feasible solution is not found.
          seed: Set the seed of the pseudo-random number generator (reproducibility).
          verbose: Print the HGS-CVRP log while it runs (single substation only).
          **kwargs: Options common to all routers (see :class:`Router`).

        Note:
          The total runtime may reach up to ``(max_retries + 1) * time_limit`` in the
          worst case.
        """
        # Call the base class initialization
        super().__init__(**kwargs)
        self.time_limit = time_limit
        self.verbose = verbose
        self.max_retries = max_retries
        self.seed = seed

    def route(self, P, A, cables, constraints, verbose=False, S_warm=None, **kwargs):
        verbose = verbose or self.verbose
        bounds = _resolved(A, constraints, self.power_rtol)

        # seed HGS-CVRP with the carried solution if it has the topology sought
        ringed = bounds.topology is Topology.RINGED
        topology = Topology.RINGED if ringed else Topology.RADIAL
        if S_warm is not None and S_warm.graph.get('topology') is not topology:
            S_warm = None

        # a warm start is an optimization, not a requirement: if HGS-CVRP cannot
        # use it, solve cold (which never raises OWNWarmupFailed)
        for warmstart in (S_warm, None):
            try:
                S = hgs_cvrp(
                    as_normalized(A),
                    bounds,
                    time_limit=self.time_limit,
                    max_retries=self.max_retries,
                    seed=self.seed,
                    log_callback=_print_log_line if verbose else None,
                    warmstart=warmstart,
                )
                break
            except OWNWarmupFailed as exc:
                if warmstart is None:
                    raise
                _warning('Unable to warm-start HGS-CVRP (%s): solving without it.', exc)

        # the cold pass either breaks out of the loop with S or raises
        G_tentative = G_from_S(S, A)  # pyrefly: ignore[unbound-name]

        # under per-substation bounds, every subtree stays on its substation
        G = PathFinder(
            G_tentative, planar=P, A=A, allow_root_change=not bounds.has_root_bounds
        ).create_detours()

        assign_cables(G, cables)

        return S, G


class MILPRouter(Router):
    """An exact router using mathematical programming.

    * Uses a Mixed-Integer Linear Programming (MILP) model of the problem.
    * Produces provably optimal or near-optimal networks (with quality metrics).
    * Requires a longer runtime than heuristics- and meta-heuristics-based routers.
    * Keeps every network constraint, except substation load limits for ringed
      networks.
    """

    default_heuristic = 'rootlust'
    _summary_attrs = ('runtime', 'bound', 'objective', 'relgap', 'termination')
    _repr_attrs = ('solver_name', 'time_limit', 'mip_gap', 'warmup', 'warmup_time')

    def __init__(
        self,
        solver_name: str,
        time_limit: float,
        mip_gap: float,
        solver_options: dict | None = None,
        warmup: bool = True,
        warmup_time: float = 0.2,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Create a MILP-based router.

        Args:
          solver_name: Name of solver (e.g., ``'gurobi'``, ``'ortools.cbcbox'``,
            ``'cbc'``, ``'ortools'``, ``'cplex'``, ``'highs'``, ``'pyomo.highs'``,
            ``'scip'``); see :func:`~optiwindnet.MILP.solver_factory`.
          time_limit: Maximum runtime (seconds).
          mip_gap: Relative MIP optimality gap tolerance.
          solver_options: Extra solver-specific options.
          warmup: Whether to warm-start the model. When ``True`` (the default), the
              solver is primed with an available solution -- the one carried across
              successive :meth:`WindFarmNetwork.optimize` calls, or one built by a
              fast heuristic when none is available or it no longer fits the model.
              When ``False``, the model is solved cold, ignoring any stored solution.
          warmup_time: Time budget (seconds) for *building* a warm start with the
              heuristic (only used when ``warmup`` is ``True`` and a warm start must
              be built). Capped by ``time_limit``. Building uses HGS-CVRP with crossing
              repair, so the worst-case build time can reach a few times this value.
          verbose: Enable verbose logging.
          **kwargs: Options common to all routers (see :class:`Router`).
        """
        super().__init__(**kwargs)
        if warmup_time <= 0:
            raise ValueError('warmup_time must be positive')
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.solver_name = solver_name
        self.solver_options = solver_options or {}
        self.warmup = warmup
        self.warmup_time = warmup_time
        self.verbose = verbose
        self.solver = solver_factory(solver_name)
        try:
            self.optiwindnet_default_options = self.solver.options
        except AttributeError:
            self.optiwindnet_default_options = {}

        # 'ortools' is the legacy alias for 'ortools.cp_sat': match the whole family,
        # since every MathOpt backend carries the log_callback this sets
        if verbose and solver_name.startswith('ortools'):
            enable_ortools_logging_if_jupyter(self.solver)

    def route(
        self,
        P,
        A,
        cables,
        constraints,
        verbose=False,
        S_warm=None,
        num_retries: int = 2,
        **kwargs,
    ):
        verbose = verbose or self.verbose

        # No eligibility pre-check: every backend's warmup_model() rejects a warm start
        # the model cannot take (topology, missing links, violated constraints) and the
        # loop below then warm-starts the model some other way.
        solver = self.solver
        bounds = _resolved(A, constraints, self.power_rtol)
        has_nonunit_inflow = bounds.inflow is not None

        if not self.warmup:
            # master switch off: solve cold, ignoring any carried/provided solution
            S_warm = None
        elif S_warm is None:
            # warm-starting on, nothing to reuse: build a warm start
            S_warm = self._make_warmstart(A, bounds)

        for _ in range(2):
            try:
                solver.set_problem(P, A, bounds, warmstart=S_warm)
                break
            except OWNWarmupFailed:
                if has_nonunit_inflow:
                    if S_warm is None:
                        raise
                    S_warm = None
                    continue
                S_warm = self._make_warmstart(A, bounds)

        else:
            # No available solution can warm-start this model, so solve it cold
            # rather than give up: a warm start is an optimization, not a requirement.
            _warning('Unable to warm-start model: solving without a warm start.')
            solver.set_problem(P, A, bounds)

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

    def _make_warmstart(self, A, bounds):
        """Build a warm start matched to the model's topology and feeder limit.

        Called (only when ``warmup`` is enabled) to obtain a warm start the model
        can take: either proactively, for a fresh solve with no solution to reuse,
        or to rebuild one when the solution carried across
        :class:`WindFarmNetwork` optimizations no longer fits the new model. The
        latter matters because a small problem change usually leaves the previous
        optimum close to the new one, so a short solve suffices -- but if that
        change also breaks the warm start's feasibility, a cold solve might not
        reach a feasible solution within the same budget. The build itself is
        given ``warmup_time`` (capped by ``time_limit``).

        HGS-CVRP gives a radial solution (or a ringed one for a RINGED model),
        and, unlike the constructor heuristics, it keeps the count and load
        bounds the model enforces (see :func:`~optiwindnet.baselines.hgs.hgs_cvrp`).
        A branched model with no upper bound on its count and no bounded
        substation takes the constructor's branched layout instead (a closer warm
        start), unless terminal inflow is nonunitary or the link length is
        limited, neither of which the constructor honours.

        Bounds HGS-CVRP cannot keep (and, with nonunitary terminal inflow,
        anything but an unbalanced, single-root, radial solve) return ``None``,
        i.e. the model is solved cold.

        Args:
          A: available-links graph, as given to :meth:`route`.
          bounds: the bounds the model enforces, resolved for ``A``.

        Returns:
          A topology to warm-start from, or ``None`` if none can be built.
        """
        if (
            bounds.topology is Topology.BRANCHED
            and bounds.feeders_ub is None
            and not bounds.has_root_bounds
            and bounds.inflow is None
            and bounds.link_length_ub is None
        ):
            return S_from_G(constructor(A, bounds, method=self.default_heuristic))
        try:
            return hgs_cvrp(
                as_normalized(A),
                # HGS-CVRP does not keep feeders straight: the model checks them
                replace(bounds, feeder_route=FeederRoute.SEGMENTED),
                # a built warm start must not outlast the solve it primes
                time_limit=min(self.time_limit, self.warmup_time),
                repair=True,
            )
        except (NotImplementedError, ValueError) as exc:
            _info('No warm start from HGS-CVRP: %s', exc)
            return None
