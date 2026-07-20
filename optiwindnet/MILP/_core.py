# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import abc
import logging
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum, auto
from inspect import cleandoc
from itertools import chain
from pathlib import Path
from textwrap import indent
from typing import TYPE_CHECKING, Any

import networkx as nx
from makefun import with_signature

from ..interarraylib import (
    G_from_S,
    add_ring_to_S,
    directed_links,
    rings_from_links,
)
from ..pathfinding import PathFinder

_lggr = logging.getLogger(__name__)
error, info, warn = _lggr.error, _lggr.info, _lggr.warning


def physical_core_count() -> int:
    """Count physical cores available to this process (cross-platform).

    On Linux, reads sysfs topology to count physical cores within the
    process affinity set. On other platforms, falls back to psutil.
    """
    if sys.platform == 'linux':
        try:
            affinity = os.sched_getaffinity(0)
        except OSError:
            affinity = None
        if affinity is not None:
            physical_cores = set()
            for cpu_id in affinity:
                topo = Path(f'/sys/devices/system/cpu/cpu{cpu_id}/topology')
                try:
                    pkg = (topo / 'physical_package_id').read_text().strip()
                    core = (topo / 'core_id').read_text().strip()
                    physical_cores.add((pkg, core))
                except OSError:
                    # sysfs unavailable (e.g. container), fall through
                    physical_cores = None
                    break
            if physical_cores is not None:
                return len(physical_cores)
    # Windows, macOS, or Linux without sysfs
    import psutil

    return psutil.cpu_count(logical=False) or os.cpu_count() or 1


def _identifier_from_class_name(c: type) -> str:
    "Convert a camel-case class name to a snake-case identifier"
    s = c.__name__
    return s[0].lower() + ''.join('_' + c.lower() if c.isupper() else c for c in s[1:])


class OWNWarmupFailed(Exception):
    pass


class OWNSolutionNotFound(Exception):
    pass


class Topology(StrEnum):
    "Set the topology of subtrees in the solution."

    RADIAL = auto()
    BRANCHED = auto()
    RINGED = auto()
    DEFAULT = BRANCHED


class FeederRoute(StrEnum):
    "If feeder routes must be ``'straight'`` or can be detoured (``'segmented'``)."

    STRAIGHT = auto()
    SEGMENTED = auto()
    DEFAULT = SEGMENTED


class FeederLimit(StrEnum):
    """Whether to limit the number of feeders. Both ``'specified'`` (an upper
    bound) and ``'exactly'`` (an exact count) require the additional kwarg
    ``'max_feeders'``. Option ``'balanced'`` is only enforceable if the feeder
    count is pinned to a single value, i.e. ``'minimum'``, ``'exactly'``, or
    ``'specified'`` with ``'max_feeders'`` at the minimum.
    """

    UNLIMITED = auto()
    EXACTLY = auto()
    SPECIFIED = auto()
    MINIMUM = auto()
    MIN_PLUS1 = auto()
    MIN_PLUS2 = auto()
    MIN_PLUS3 = auto()
    DEFAULT = UNLIMITED


def feeder_and_load_bounds(
    T: int,
    capacity: int,
    feeder_limit: FeederLimit,
    max_feeders: int,
    balanced: bool,
    feeders_per_subtree: int = 1,
) -> tuple[int, int | None, int | None, int | None]:
    """Derive the feeder-count and feeder-load bounds a model must enforce.

    Bounds are returned in units of **subtrees** (the quantity the model's
    feeder-sum constraint counts): one feeder per subtree for radial/branched
    topologies, but a RINGED subtree is a cycle with two feeders. The
    caller signals this with ``feeders_per_subtree`` (2 for RINGED), so that a
    user-supplied ``max_feeders`` — always expressed as the number of physical
    **substation connections** — is converted to the subtree count the model
    constrains, and the returned bounds are multiplied back by
    ``feeders_per_subtree`` for reporting.

    The subtree count is bounded below by ``min_feeders = ceil(T/capacity)``
    regardless of ``feeder_limit`` (a valid inequality). ``feeders_ub`` is
    ``None`` when the count is unbounded above; when it equals ``feeders_lb``,
    the count is pinned and callers should emit an equality constraint.

    Balanced subtrees (loads differing at most by one unit) are only expressible
    with a pinned feeder count ``F``, in which case the loads must lie in
    ``{T // F, ceil(T / F)}``. A load bound of ``None`` means "do not emit":
    either ``balanced`` is off, or it is not enforceable (a warning is issued),
    or the bound is already implied by the flow variable's own bounds.

    Returns:
        ``(feeders_lb, feeders_ub, load_lb, load_ub)`` in subtree-count units.
    """
    if feeders_per_subtree != 1 and max_feeders:
        if max_feeders % feeders_per_subtree:
            raise ValueError(
                f'max_feeders ({max_feeders}) must be a multiple of '
                f'{feeders_per_subtree} for a RINGED topology (each ring uses '
                f'{feeders_per_subtree} substation connections)'
            )
        max_feeders //= feeders_per_subtree
    min_feeders = math.ceil(T / capacity)
    if feeder_limit is FeederLimit.UNLIMITED:
        feeders_lb, feeders_ub = min_feeders, None
    elif feeder_limit is FeederLimit.MINIMUM:
        feeders_lb = feeders_ub = min_feeders
    elif feeder_limit is FeederLimit.EXACTLY:
        if max_feeders < min_feeders:
            raise ValueError('max_feeders is below the minimum necessary')
        if max_feeders > T:
            raise ValueError('max_feeders is above the number of terminals')
        feeders_lb = feeders_ub = max_feeders
    elif feeder_limit is FeederLimit.SPECIFIED:
        if max_feeders < min_feeders:
            raise ValueError('max_feeders is below the minimum necessary')
        feeders_lb, feeders_ub = min_feeders, max_feeders
    elif feeder_limit in (
        FeederLimit.MIN_PLUS1,
        FeederLimit.MIN_PLUS2,
        FeederLimit.MIN_PLUS3,
    ):
        plus = int(feeder_limit.value[-1])
        feeders_lb, feeders_ub = min_feeders, min_feeders + plus
    else:
        raise NotImplementedError('Unknown value:', feeder_limit)

    if not balanced:
        return feeders_lb, feeders_ub, None, None
    if feeders_lb != feeders_ub:
        warn(
            'Model option <balanced = True> requires a single possible feeder'
            f' count, but <feeder_limit = {feeder_limit.value.upper()}> allows a'
            ' range: model will not enforce balanced subtrees.'
        )
        return feeders_lb, feeders_ub, None, None
    F = feeders_lb
    load_lb, load_ub = T // F, math.ceil(T / F)
    # bounds at the extremes are already implied by the flow variable's bounds
    return (
        feeders_lb,
        feeders_ub,
        load_lb if load_lb > 1 else None,
        load_ub if load_ub < capacity else None,
    )


class ModelOptions(dict):
    """Hold options for the modelling of the cable routing problem.

    Use ModelOptions.help() to get the options and their permitted and default
    values. Use ModelOptions() without any parameters to use the defaults.
    """

    hints = {
        _identifier_from_class_name(kind): kind
        for kind in (Topology, FeederRoute, FeederLimit)
    }
    # this has to be kept in sync with make_min_length_model()
    simple = dict(
        balanced=(
            bool,
            False,
            'Whether to enforce balanced subtrees (subtree loads differ at most '
            'by one unit).',
        ),
        max_feeders=(
            int,
            0,
            'Number of feeders: the maximum if <feeder_limit = "specified">, '
            'the exact count if <feeder_limit = "exactly">',
        ),
    )

    # `with_signature` rewrites `__init__` at import time so that `help()`,
    # `inspect.signature` and IDE introspection show the real options. Type
    # checkers cannot follow that, so the same signature is declared statically
    # under TYPE_CHECKING -- keep the two in sync (they are both derived from
    # `hints` and `simple`).
    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            topology: Topology | str = ...,
            feeder_route: FeederRoute | str = ...,
            feeder_limit: FeederLimit | str = ...,
            balanced: bool = ...,
            max_feeders: int = ...,
        ) -> None: ...
    else:

        @with_signature(
            '__init__(self, *, '
            + ', '.join(
                chain(
                    (
                        f'{k}: {v.__name__} = "{v.DEFAULT.value}"'
                        for k, v in hints.items()
                    ),
                    (
                        f'{name}: {kind.__name__} = {default}'
                        for name, (kind, default, _) in simple.items()
                    ),
                )
            )
            + ')'
        )
        def __init__(self, **kwargs):
            # dispatch on the key, not on the value's type: a str passed for an
            # int option is a type error, not an enum to look up
            for k, v in kwargs.items():
                kind = self.hints.get(k)
                if kind is not None:
                    kwargs[k] = kind(v)
                else:
                    expected = self.simple[k][0]
                    if not isinstance(v, expected):
                        raise TypeError(
                            f'{k} must be {expected.__name__}, got '
                            f'{type(v).__name__}: {v!r}'
                        )

            super().__init__(kwargs)

    # Options are a value object: every value is coerced and every key is
    # present once __init__ returns, and the whole library reads them on that
    # basis (`is Topology.RINGED` is false for a plain 'ringed' str). Mutating
    # the mapping afterwards would bypass the coercion, so it is refused --
    # build a new instance instead.
    def _immutable(self, *args, **kwargs):
        raise TypeError(
            f'{type(self).__name__} is immutable: construct a new one instead of '
            'modifying it in place'
        )

    __setitem__ = __delitem__ = _immutable
    clear = pop = popitem = setdefault = update = _immutable

    @classmethod
    def help(cls):
        for k, v in cls.hints.items():
            doc = indent(cleandoc(v.__doc__ or ''), '    ')
            print(
                f'{k} in {{'
                + ', '.join(
                    f'"{m}"' for n, m in v.__members__.items() if n != 'DEFAULT'
                )
                + f'}} default: {cls.hints[k].DEFAULT.value}\n'
                f'{doc}\n'
            )
        for name, (kind, default, desc) in cls.simple.items():
            print(f'{name} [{kind.__name__}] default: {default}\n    {desc}\n')


_Link = tuple[int, int]


@dataclass(slots=True)
class ModelMetadata:
    R: int
    T: int
    capacity: int
    linkset: tuple[_Link, ...]
    link_: Mapping[_Link, Any]
    flow_: Mapping[_Link, Any]
    model_options: dict[str, Any]
    fun_fingerprint: dict[str, str | bytes]
    weight_: tuple[float, ...] = ()
    solution_hint: dict[Any, float] = field(default_factory=dict)
    warmed_by: str = ''


@dataclass(slots=True)
class SolutionInfo:
    runtime: float
    bound: float
    objective: float
    relgap: float
    termination: str


#: topologies of ``S`` that can seed a model of each topology. A radial
#: solution is a valid warmstart for a branched model, but not the reverse;
#: ringed is incomparable with both (a ringed model requires two feeders per
#: subtree, which no radial or branched solution provides).
_WARMSTART_OK: Mapping[Topology, frozenset[Topology]] = {
    Topology.RADIAL: frozenset({Topology.RADIAL}),
    Topology.BRANCHED: frozenset({Topology.RADIAL, Topology.BRANCHED}),
    Topology.RINGED: frozenset({Topology.RINGED}),
}


def warmstart_topology_mismatch(model_topology: Topology, S: nx.Graph) -> str:
    """Report whether ``S``'s topology can seed a ``model_topology`` model.

    The topology of ``S`` is read from ``S.graph['topology']``, never inferred
    from its structure.

    Returns:
      Human-readable incompatibility, or ``''`` if ``S`` is a valid warmstart.
    """
    S_topology = Topology(S.graph['topology'])
    if S_topology in _WARMSTART_OK[model_topology]:
        return ''
    return (
        f'{S_topology} network incompatible with model option: '
        f'topology="{model_topology}"'
    )


def warmstart_topology(metadata: ModelMetadata, S: nx.Graph) -> Topology:
    """Topology of the model ``metadata`` describes, checked against ``S``'s.

    Every backend's ``warmup_model()`` opens with this: the topology decides how
    the warmstart is mapped onto the model's variables, and ``S`` must be a
    valid seed for it.

    Returns:
      The model's topology.

    Raises:
        OWNWarmupFailed: ``S`` is not a valid warmstart for the model.
    """
    model_topology = metadata.model_options['topology']
    mismatch = warmstart_topology_mismatch(model_topology, S)
    if mismatch:
        raise OWNWarmupFailed(f'warmup_model() failed: {mismatch}')
    return model_topology


def ringed_warmstart_values(
    metadata: ModelMetadata, S: nx.Graph
) -> tuple[dict[_Link, int], dict[_Link, int]]:
    """Model link/flow variable values to warm-start a RINGED model from ``S``.

    In the flow formulation a ring is a single directed chain of its ``n``
    terminals, which is not how the canonical solution graph ``S`` stores it (two
    arms split at an open point). :func:`.interarraylib.directed_links` does that
    conversion — radializing the rings; this maps its output onto the model's
    link/flow variables.

    Returns ``(link_vals, flow_vals)``: complete assignments over
    ``metadata.link_`` / ``metadata.flow_`` keys (inactive variables set to 0).

    Raises:
        OWNWarmupFailed: a ring uses a link absent from the model.
    """
    link_vals: dict[_Link, int] = dict.fromkeys(metadata.link_, 0)
    flow_vals: dict[_Link, int] = dict.fromkeys(metadata.flow_, 0)
    for source, sink, flow in directed_links(S):
        key = (source, sink)
        if key not in link_vals:
            # a ring's closing feeder (r→t) is only a model variable under
            # Topology.RINGED; t→r feeders are there by construction.
            raise OWNWarmupFailed(f'warmup_model() failed: model lacks ring link {key}')
        link_vals[key] = 1
        if key in flow_vals:
            # a ring's closing feeder is a starsʹ link: it has no flow variable
            flow_vals[key] = flow
    return link_vals, flow_vals


class Solver(abc.ABC):
    "Common interface to multiple MILP solvers"

    name: str
    metadata: ModelMetadata
    solver: Any
    options: dict[str, Any]
    stopping: dict[str, Any]
    solution_info: SolutionInfo
    applied_options: dict[str, Any]

    @abc.abstractmethod
    def _link_val(self, var: Any) -> int | bool:
        "Get the value of a link variable from the current solution."
        pass

    @abc.abstractmethod
    def _flow_val(self, var: Any) -> int:
        "Get the value of a flow variable from the current solution."
        pass

    @abc.abstractmethod
    def set_problem(
        self,
        P: nx.PlanarEmbedding,
        A: nx.Graph,
        capacity: int,
        model_options: Mapping[str, Any],
        warmstart: nx.Graph | None = None,
    ):
        """Define the problem geometry, available edges and tree properties

        Args:
          P: planar embedding of the location
          A: available edges for the location
          capacity: maximum number of terminals in a subtree
          model_options: tree properties - see ModelOptions.help()
          warmstart: initial feasible solution to pass to solver
        """
        pass

    @abc.abstractmethod
    def solve(
        self,
        time_limit: float,
        mip_gap: float,
        options: dict[str, Any] = {},
        verbose: bool = False,
    ) -> SolutionInfo:
        """Run the MILP solver search.

        Args:
          time_limit: maximum time (s) the solver is allowed to run.
          mip_gap: relative difference from incumbent solution to lower bound
            at which the search may be stopped before ``time_limit`` is reached.
          options: additional options to pass to solver (see solver manual).

        Returns:
          General information about the solution search (use ``get_solution()`` for
            the actual solution).
        """
        pass

    @abc.abstractmethod
    def get_solution(self, A: nx.Graph | None = None) -> tuple[nx.Graph, nx.Graph]:
        """Output solution topology A and routeset G.

        Args:
          A: optionally replace the A given via set_problem() (if normalized A)

        Returns:
          Topology graph S and routeset G.
        """
        pass

    def _make_graph_attributes(self) -> dict[str, Any]:
        metadata, solution_info = self.metadata, self.solution_info
        solver_details = self.applied_options.copy()
        # the method_options dict is extracted by db utility function packmethod()
        method_options = dict(
            solver_name=self.name,
            fun_fingerprint=metadata.fun_fingerprint,
            **self.stopping,
            **metadata.model_options,
        )
        # remaining graph attributes (key=value) are stored in db.RouteSet[].misc
        attr = dict(
            **asdict(solution_info),
            method_options=method_options,
            solver_details=solver_details,
        )
        if 'max_feeders' in method_options:
            solver_details['max_feeders'] = method_options.pop('max_feeders')
        if metadata.warmed_by:
            attr['warmstart'] = metadata.warmed_by
        return attr

    def _topology_from_mip_sol(self):
        """Create a topology graph from the solution to the MILP model.

        Returns:
          Graph topology ``S`` from the solution.
        """
        metadata = self.metadata
        topology = metadata.model_options['topology']
        R = metadata.R
        S = nx.Graph(R=R, T=metadata.T)
        # ensure roots are added, even if some are not connected
        S.add_nodes_from(range(-R, 0))
        # Get active links and if flow is reversed (i.e. from small to big)
        rev_from_link = {
            (u, v): u < v
            for (u, v), var in metadata.link_.items()
            if self._link_val(var)
        }
        max_load = 0
        if topology is Topology.RINGED:
            # A ring is a cycle: two feeders to a shared root joined at
            # their tail ends. Recover each ring's ordered terminal sequence from
            # the active links and build it in canonical form (both feeders real,
            # a single ``split`` open point at the load midpoint).
            for subtree, (r, ordered) in enumerate(rings_from_links(rev_from_link, R)):
                add_ring_to_S(S, r, ordered, subtree, getattr(self, 'A', None))
            for r in range(-R, 0):
                rootload = 0
                for nbr in S.neighbors(r):
                    subtree_load = S.nodes[nbr]['load']
                    max_load = max(max_load, subtree_load)
                    rootload += subtree_load
                S.nodes[r]['load'] = rootload
        else:
            S.add_weighted_edges_from(
                (
                    (u, v, self._flow_val(metadata.flow_[u, v]))
                    for (u, v) in rev_from_link
                ),
                weight='load',
            )
            nx.set_edge_attributes(S, rev_from_link, name='reverse')
            # propagate loads from edges to nodes
            subtree = -1
            for r in range(-R, 0):
                for u, v in nx.edge_dfs(S, r):
                    S.nodes[v]['load'] = S[u][v]['load']
                    if u == r:
                        subtree += 1
                    S.nodes[v]['subtree'] = subtree
                rootload = 0
                for nbr in S.neighbors(r):
                    subtree_load = S.nodes[nbr]['load']
                    max_load = max(max_load, subtree_load)
                    rootload += subtree_load
                S.nodes[r]['load'] = rootload
        S.graph.update(
            topology=topology.name.lower(),
            capacity=metadata.capacity,
            max_load=max_load,
            has_loads=True,
            creator='MILP.' + self.name,
            solver_details={},
        )
        return S


class PoolHandler(abc.ABC):
    name: str
    num_solutions: int
    model_options: ModelOptions

    @abc.abstractmethod
    def _objective_at(self, index: int) -> float:
        "Get objective value from solution pool at position ``index``"
        pass

    @abc.abstractmethod
    def _topology_from_mip_pool(self) -> nx.Graph:
        "Build topology from the pool solution at the last requested position"
        pass

    def _investigate_pool(
        self, P: nx.PlanarEmbedding, A: nx.Graph
    ) -> tuple[nx.Graph, nx.Graph]:
        """Go through the solver's solutions checking which has the shortest length
        after applying the detours with PathFinder."""
        Λ = float('inf')
        S = G = None
        num_solutions = self.num_solutions
        info(f'Solution pool has {num_solutions} solutions.')
        for i in range(num_solutions):
            λ = self._objective_at(i)
            if λ > Λ:
                info(
                    f"#{i} halted pool search: objective ({λ:.3f}) > incumbent's length"
                )
                break
            Sʹ = self._topology_from_mip_pool()
            Gʹ = PathFinder(G_from_S(Sʹ, A), planar=P, A=A).create_detours()
            Λʹ = Gʹ.size(weight='length')
            if Λʹ < Λ:
                S, G, Λ = Sʹ, Gʹ, Λʹ
                G.graph['pool_entry'] = i, λ
                info(f'#{i} -> incumbent (objective: {λ:.3f}, length: {Λ:.3f})')
            else:
                info(f'#{i} discarded (objective: {λ:.3f}, length: {Λ:.3f})')
        if S is None or G is None:
            raise ValueError('Solution pool has no usable solution.')
        G.graph['pool_count'] = num_solutions
        return S, G
