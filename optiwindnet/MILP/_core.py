# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import abc
import logging
import math
import os
import sys
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum, auto
from fractions import Fraction
from inspect import cleandoc
from itertools import chain
from pathlib import Path
from textwrap import indent
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

import networkx as nx
import numpy as np
from bitarray import bitarray, frozenbitarray
from makefun import with_signature

from ..constraints import NetworkBounds, NetworkConstraints
from ..converting import G_from_S, S_from_linkbits
from ..identity import _CANONICAL_TERMINAL_LINKS, topology_id
from ..interarraylib import directed_links
from ..loads import (
    DEFAULT_POWER_RTOL,
    calcload,
    nonunit_inflow,
    quantized,
    split_rings_and_calc_loads,
    total_inflow,
    validate_terminal_power,
)
from ..pathfinding import PathFinder
from ..types import FeederRoute, OWNWarmupFailed, Topology

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


class OWNSolutionNotFound(Exception):
    pass


class FeederLimit(StrEnum):
    """Whether to limit the number of feeders. Both ``'specified'`` (an upper
    bound) and ``'exactly'`` (an exact count) require the additional kwarg
    ``'max_feeders'``. Option ``'balanced'`` is only enforceable if the feeder
    count is pinned to a single value, i.e. ``'minimum'`` or ``'exactly'``.

    Nonunitary terminal inflow rule out ``'minimum'`` and the ``'min_plus*'`` modes,
    whose bounds come from a minimum feeder count that indivisible terminal
    inflow may not reach.
    """

    UNLIMITED = auto()
    EXACTLY = auto()
    SPECIFIED = auto()
    MINIMUM = auto()
    MIN_PLUS1 = auto()
    MIN_PLUS2 = auto()
    MIN_PLUS3 = auto()
    DEFAULT = UNLIMITED


_Link = tuple[int, int]

_FEEDER_LIMITS_FROM_MINIMUM = (
    FeederLimit.MINIMUM, FeederLimit.MIN_PLUS1,
    FeederLimit.MIN_PLUS2, FeederLimit.MIN_PLUS3,
)  # fmt: skip


def _constraints_from_options(
    topology: Topology | str = Topology.DEFAULT,
    feeder_route: FeederRoute | str = FeederRoute.DEFAULT,
    feeder_limit: FeederLimit | str = FeederLimit.DEFAULT,
    balanced: bool = False,
    max_feeders: int = 0,
) -> NetworkConstraints:
    """Translate the options of :class:`ModelOptions` into network constraints.

    ``'min_plus<k>'`` adds ``k`` subtrees or rings to the minimum (``2 * k``
    feeders if RINGED). The constraints drop an unenforceable ``balanced`` (the
    feeder count not fixed) with a warning, as these options always did.
    """
    topology, feeder_limit = Topology(topology), FeederLimit(feeder_limit)
    per_group = 2 if topology is Topology.RINGED else 1
    feeders: int | str | None = None
    most_feeders: int | str | None = None
    if feeder_limit is FeederLimit.EXACTLY:
        feeders = max_feeders
    elif feeder_limit is FeederLimit.SPECIFIED:
        most_feeders = max_feeders
    elif feeder_limit is FeederLimit.MINIMUM:
        feeders = 'min'
    elif feeder_limit in _FEEDER_LIMITS_FROM_MINIMUM:
        most_feeders = f'min+{per_group * int(feeder_limit.value[-1])}'
    return _OptionsConstraints(
        topology=topology,
        feeder_route=feeder_route,
        feeders=feeders,
        max_feeders=most_feeders,
        balanced=balanced,
    )


class _OptionsConstraints(NetworkConstraints):
    """Network constraints translated from :class:`ModelOptions`.

    They resolve an unenforceable balance (a feeder count that is not fixed)
    into no balance, with a warning, instead of raising.
    """

    def __repr__(self) -> str:
        # it stands for the NetworkConstraints that the options translate into
        return 'NetworkConstraints' + super().__repr__().removeprefix(
            type(self).__qualname__
        )

    def _resolve(self, **kwargs) -> NetworkBounds:
        if not self.balanced or self.feeders is not None:
            return super()._resolve(**kwargs)
        bounds = replace(self, balanced=False)._resolve(**kwargs)
        if bounds.feeders_lb == bounds.feeders_ub:
            return super()._resolve(**kwargs)
        warn(
            'Model option <balanced = True> requires a single possible feeder '
            'count, but the feeder limit allows a range: model will not enforce '
            'balanced subtrees.'
        )
        return bounds


def model_bounds(
    A: nx.Graph,
    capacity: int,
    bounds: NetworkBounds | None,
    legacy: Mapping[str, Any],
) -> NetworkBounds:
    """Return the bounds a model builder enforces.

    ``legacy`` holds the builder's deprecated keyword arguments (the options of
    :class:`ModelOptions`), of which only those not ``None`` count. Without
    ``bounds`` or ``legacy`` options, the defaults of
    :class:`~optiwindnet.constraints.NetworkConstraints` apply.

    Raises:
      TypeError: both ``bounds`` and ``legacy`` options are given.
      ValueError: the bounds limit the load of the roots of a RINGED model, whose
        rings may connect to two roots.
    """
    given = {key: value for key, value in legacy.items() if value is not None}
    if bounds is None:
        if given:
            warnings.warn(
                f'make_min_length_model() options {sorted(given)} are deprecated: '
                'pass bounds=NetworkConstraints(...).resolve(A, capacity=...) instead.',
                DeprecationWarning,
                stacklevel=3,
            )
        bounds = _constraints_from_options(**given)._resolve(
            T=A.graph['T'],
            R=A.graph['R'],
            capacity=capacity,
            inflow_total=total_inflow(A),
        )
    elif given:
        raise TypeError(f'Give either bounds or the options {sorted(given)}.')
    if bounds.topology is Topology.RINGED and bounds.root_load_ub is not None:
        raise ValueError(
            'Substation load limits are not available for RINGED MILP models: '
            'the two feeders of a ring may connect it to two substations.'
        )
    return bounds


@dataclass(frozen=True, slots=True)
class BoundRow:
    """A bound on a sum of link (or flow) variables of a model.

    Attributes:
      name: constraint name, to be suffixed with ``_eq``, ``_lb`` or ``_ub``.
      links: keys of the variables to sum.
      flow: whether to sum the flow variables instead of the link variables.
      lb: lower bound (``None`` if none).
      ub: upper bound (``None`` if none).
    """

    name: str
    links: tuple[_Link, ...]
    flow: bool
    lb: int | None
    ub: int | None


def bound_rows(bounds: NetworkBounds, R: int, T: int) -> Iterator[BoundRow]:
    """Yield the count and load bounds of a model, except for balance.

    The feeder count is bounded in subtrees or rings, i.e. on the feeders that
    carry flow. Its lower bound is always given, as a valid inequality. A root's
    feeder count includes the closing feeders of rings.
    """
    ringed = bounds.topology is Topology.RINGED
    yield BoundRow(
        'feeder_limit',
        tuple((t, r) for r in range(-R, 0) for t in range(T)),
        False,
        bounds.groups_lb,
        bounds.groups_ub,
    )
    for r, lb, ub in zip(
        range(-R, 0),
        bounds.root_feeders_lb or (0,) * R,
        bounds.root_feeders_ub or (None,) * R,
    ):
        if lb or ub is not None:
            yield BoundRow(
                f'root_feeder_limit_r{-r}',
                tuple((t, r) for t in range(T))
                + (tuple((r, t) for t in range(T)) if ringed else ()),
                False,
                lb or None,
                ub,
            )
    for r, ub in zip(range(-R, 0), bounds.root_load_ub or (None,) * R):
        if ub is not None:
            yield BoundRow(
                f'root_load_limit_r{-r}',
                tuple((t, r) for t in range(T)),
                True,
                None,
                ub,
            )


def balance_bounds(
    bounds: NetworkBounds, capacity: int
) -> tuple[int | None, int | None]:
    """Return the load bounds of a feeder under balance, if they bind.

    Bounds at the extremes (``1`` and ``capacity``, which is per ring if RINGED)
    are already implied by the flow variable's own bounds and are ``None``.
    """
    lb, ub = bounds.group_load_lb, bounds.group_load_ub
    return (
        lb if lb is not None and lb > 1 else None,
        ub if ub is not None and ub < capacity else None,
    )


class ModelOptions(dict):
    """Hold options for the modelling of the cable routing problem.

    Superseded by :class:`~optiwindnet.constraints.NetworkConstraints`, which
    :meth:`to_constraints` translates them into.

    Use ModelOptions.help() to get the options and their permitted and default
    values. Use ModelOptions() without any parameters to use the defaults.
    """

    hints: ClassVar[dict[str, type]] = {
        _identifier_from_class_name(kind): kind
        for kind in (Topology, FeederRoute, FeederLimit)
    }
    # this has to be kept in sync with make_min_length_model()
    simple: ClassVar[dict[str, tuple]] = {
        'balanced': (
            bool,
            False,
            (
                'Whether to enforce balanced subtrees (subtree loads differ at most '
                'by one unit).'
            ),
        ),
        'max_feeders': (
            int,
            0,
            (
                'Number of feeders: the maximum if <feeder_limit = "specified">, '
                'the exact count if <feeder_limit = "exactly">'
            ),
        ),
    }

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

    __setitem__ = __delitem__ = __ior__ = _immutable
    clear = pop = popitem = setdefault = update = _immutable

    @classmethod
    def _from_pickle(cls, values):
        return cls(**values)

    def __reduce__(self):
        return type(self)._from_pickle, (dict(self),)

    def to_constraints(self) -> NetworkConstraints:
        """Translate the options into network constraints.

        ``feeder_limit='min_plus<k>'`` becomes ``max_feeders='min+<k>'``, with
        ``k`` doubled for RINGED (a ring has two feeders). An unenforceable
        ``balanced`` is dropped with a warning.
        """
        return _constraints_from_options(**self)

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


def canonical_linksets(
    A_terminals: nx.Graph,
    R: int,
    T: int,
    topology: Topology,
) -> tuple[tuple[_Link, ...], tuple[_Link, ...], tuple[_Link, ...], tuple[_Link, ...]]:
    """Return the link families in the canonical MILP variable order.

    Terminal-terminal links are normalized to their increasing orientation and
    sorted lexicographically. Their reverse orientations follow in the same
    order. Feeders are terminal-major, with roots ordered from ``-R`` to ``-1``;
    RINGED root-to-terminal closing links follow their corresponding feeders.

    Keeping this construction shared prevents graph insertion order and backend
    implementation details from changing the meaning of a link-value sequence.
    """
    terminal_links = A_terminals.graph.get(_CANONICAL_TERMINAL_LINKS)
    E = (
        tuple(map(tuple, terminal_links.tolist()))
        if terminal_links is not None
        else tuple(sorted((u, v) if u < v else (v, u) for u, v in A_terminals.edges()))
    )
    Eʹ = tuple((v, u) for u, v in E)
    stars = tuple((t, r) for t in range(T) for r in range(-R, 0))
    starsʹ = tuple((r, t) for t, r in stars) if topology is Topology.RINGED else ()
    return E, Eʹ, stars, starsʹ


def nonclosest_feeders(A: nx.Graph) -> Iterator[_Link]:
    """Yield the feeders ``(t, r)`` whose root is not the closest to ``t``.

    Closeness is by ``A.graph['d2roots']`` (contour-adjusted, not line-of-sight),
    with ties resolved to the lowest root index. The models fix these feeders to
    zero, on the premise that an optimal solution never connects a terminal to a
    root other than its closest one. Nothing is yielded for single-root sites.
    """
    R = A.graph['R']
    if R == 1:
        return
    T = A.graph['T']
    closest_root_ = A.graph['d2roots'][:T].argmin(axis=1) - R
    for t, closest in enumerate(closest_root_.tolist()):
        for r in range(-R, 0):
            if r != closest:
                yield t, r


def feeders_to_fix(A: nx.Graph, bounds: NetworkBounds) -> Iterator[_Link]:
    """Yield the feeders a model fixes to zero.

    These are the feeders to a root other than the terminal's closest (see
    :func:`nonclosest_feeders`), unless per-root bounds are given: a bound may
    require a terminal to connect to another root.
    """
    if not bounds.has_root_bounds:
        yield from nonclosest_feeders(A)


def passover_rows(
    passovers: np.ndarray, ringed: bool
) -> Iterator[tuple[list[_Link], int, str]]:
    """Yield the constraints for the pass-over rows of ``_feeder_link_conflicts()``.

    Each constraint is ``(links, rhs, name)``: the sum of the directed links
    must not exceed ``rhs``. The feeder and the links ⟨w1, a⟩, ⟨w2, b⟩ (and
    ⟨w1, w2⟩ if ``w1 != w2``) cannot all be active.
    """
    for r, t, w1, a, w2, b in passovers.tolist():
        pairs = [(w1, a), (w2, b)] + ([] if w1 == w2 else [(w1, w2)])
        links = [(t, r), (r, t)] if ringed else [(t, r)]
        links += [link for x, y in pairs for link in ((x, y), (y, x))]
        yield links, len(pairs), f'feeder_passover_{t}~r{-r}_{w1}~{a}_{w2}~{b}'


def linkbits_from_directed(
    values: Iterable[int | bool],
    *,
    terminal_link_count: int,
    feeder_link_count: int,
    ringed: bool,
) -> frozenbitarray:
    """Build undirected linkbits from canonical directed link values.

    ``values`` must follow the canonical block order ``E + Eʹ + stars`` (plus
    ``starsʹ`` for RINGED). The result is ordered as ``E + stars`` and sets an
    undirected link's bit when either direction is active.
    """
    if isinstance(values, bitarray) and values.endian == 'big':
        binary_values = values
    else:
        binary_values = bitarray(endian='big')
        for index, value in enumerate(values):
            if value not in (0, 1):
                raise ValueError(
                    f'link value at index {index} is not binary: {value!r}'
                )
            binary_values.append(value)
    expected_count = 2 * terminal_link_count + feeder_link_count * (1 + ringed)
    if len(binary_values) != expected_count:
        raise ValueError(
            'canonical link blocks and values must have equal lengths: '
            f'{expected_count} != {len(binary_values)}'
        )

    # bitarray(...) also unfreezes: slices of a frozenbitarray are frozen too
    terminal_bits = bitarray(binary_values[:terminal_link_count])
    terminal_bits |= binary_values[terminal_link_count : 2 * terminal_link_count]
    feeder_start = 2 * terminal_link_count
    feeder_bits = bitarray(
        binary_values[feeder_start : feeder_start + feeder_link_count]
    )
    if ringed:
        feeder_bits |= binary_values[feeder_start + feeder_link_count :]
    terminal_bits.extend(feeder_bits)
    return frozenbitarray(terminal_bits)


@dataclass(slots=True)
class ModelMetadata:
    R: int
    T: int
    capacity: int
    linkset: tuple[_Link, ...]
    link_: Mapping[_Link, Any]
    flow_: Mapping[_Link, Any]
    topology: Topology
    fun_fingerprint: dict[str, str | bytes]
    weight_: tuple[float, ...] = ()
    solution_hint: dict[Any, float] = field(default_factory=dict)
    warmed_by: str = ''


@dataclass(slots=True)
class SolutionInfo:
    """Search outcome, plus the id of the topology the search settled on.

    ``topology_id`` is the :func:`~optiwindnet.identity.topology_id` of the
    model-objective incumbent, which ``solve()`` derives from the link bits it
    reads off the solver -- no topology is built to obtain it, and the bits
    themselves stay on the solver, as :attr:`Solver.incumbent_linkbits`. It is
    empty only in an instance made outside a solver. A solution pool may hand
    over a different entry, whose own id is the ``_topology_id`` of the topology
    ``get_solution()`` returns.
    """

    runtime: float
    bound: float
    objective: float
    relgap: float
    termination: str
    topology_id: bytes = b''

    def __repr__(self) -> str:
        """Fixed-shape summary: one field per line, keys column-aligned.

        The floats are rounded for reading, so this is not a reconstructible
        repr: use :func:`dataclasses.asdict` when exact values are needed.
        """
        fields = (
            ('termination', repr(self.termination)),
            ('objective', f'{self.objective:.8g}'),
            ('bound', f'{self.bound:.8g}'),
            ('relgap', f'{self.relgap:.2%}'),
            ('runtime', f'{self.runtime:.4g}s'),
            ('topology_id', repr(self.topology_id.hex())),
        )
        body = ''.join(f'    {key:<11} = {val}\n' for key, val in fields)
        return f'SolutionInfo(\n{body})'


def check_inflow_support(A: nx.Graph, topology: Topology) -> int:
    """Validate terminal power and return total inflow for the model.

    A RINGED model splits each ring into two arms of at most ``capacity`` each,
    which it expresses by doubling the capacity of a single path. That holds only
    while every terminal weighs the same: with nonunitary inflow, a path within twice
    the capacity need not divide into two arms that each fit within it, so the
    model would admit rings that no pair of cables can carry.

    Raises:
        NotImplementedError: ``topology`` is RINGED and the terminals declare an
          inflow other than one unit.
    """
    validate_terminal_power(A)
    total = total_inflow(A)
    if topology is Topology.RINGED and total != A.graph['T']:
        raise NotImplementedError(
            "a RINGED model cannot honour a terminal 'inflow' other than 1: a ring "
            'within twice the capacity is not guaranteed to split into two arms '
            'that each fit within it'
        )
    return total


def check_warmstart_topology(metadata: ModelMetadata, S: nx.Graph) -> None:
    """Reject a warm start whose topology the model cannot start from.

    A model accepts its own topology, plus RADIAL into BRANCHED (a path is a
    tree). The comparison is by identity, and ``S`` comes from the caller, so its
    topology is checked for being a :class:`~optiwindnet.types.Topology`: an equal
    ``str`` is a type error, not a mismatch.

    Raises:
        OWNWarmupFailed: ``S``'s topology is one this model cannot start from, or
          is not a ``Topology``.
    """
    mt = metadata.topology
    st = S.graph['topology']
    if not isinstance(st, Topology):
        raise OWNWarmupFailed(
            f"warmup_model() failed: S.graph['topology'] must be a Topology "
            f'member, got {type(st).__name__}: {st!r}'
        )
    if not (st is mt or (mt is Topology.BRANCHED and st is Topology.RADIAL)):
        raise OWNWarmupFailed(
            f'warmup_model() failed: {st} network cannot warm-start a {mt} model'
        )


def warmstart_links(
    metadata: ModelMetadata, S: nx.Graph
) -> Iterator[tuple[Any, Any | None, int]]:
    """Yield ``(link_var, flow_var, flow)`` for every link ``S`` activates.

    Uses ``metadata``'s own link/flow maps as the index, orienting ``S`` the way
    the flow formulation expects — radializing RINGED rings into directed chains
    — via :func:`.interarraylib.directed_links`. ``flow_var`` is ``None`` for a
    ring's closing feeder (a link variable with no flow one). Everything not
    yielded is inactive (0); each backend initializes that however its warm-start API
    requires, then applies these.

    Raises:
        OWNWarmupFailed: ``S`` uses a link the model lacks.
    """
    for source, sink, flow in directed_links(S):
        key = (source, sink)
        if key not in metadata.link_:
            raise OWNWarmupFailed(f'warmup_model() failed: model lacks S link {key}')
        yield (
            metadata.link_[key],
            # not `.get()`: `flow_` may be a Pyomo IndexedVar, which supports
            # `in` and `[]` but is not a Mapping
            metadata.flow_[key] if key in metadata.flow_ else None,  # noqa: SIM401
            flow,
        )


# the bounds recorded in a solution's `solver_details`
_COUNT_DETAILS = (
    'feeders_lb', 'feeders_ub', 'root_feeders_lb', 'root_feeders_ub', 'root_load_ub',
)  # fmt: skip


class Solver(abc.ABC):
    "Common interface to multiple MILP solvers"

    name: str
    metadata: ModelMetadata
    A: nx.Graph
    # backend-native objects: every concrete solver sets both in `set_problem()`
    model: Any
    solver: Any
    options: dict[str, Any]
    stopping: dict[str, Any]
    constraints: NetworkConstraints
    bounds: NetworkBounds
    solution_info: SolutionInfo
    applied_options: dict[str, Any]
    _incumbent_linkbits: frozenbitarray
    # graph attributes recording the power basis of the solve (see `set_problem()`)
    _power_attrs: Mapping[str, Any] = MappingProxyType({})

    @property
    def incumbent_linkbits(self) -> frozenbitarray:
        """Canonical link bits of the model-objective incumbent, read by ``solve()``.

        The solution itself, in the encoding
        :func:`~optiwindnet.converting.S_from_linkbits` decodes: the route to a
        topology that skips the detour routing :meth:`get_solution` performs.
        These are the incumbent's bits, not those of whichever solution-pool entry
        ``get_solution()`` settles on once the entries are ranked by routed length.
        """
        try:
            return self._incumbent_linkbits
        except AttributeError as exc:
            exc.args += ('.solve() must be called before solution retrieval',)
            raise

    @abc.abstractmethod
    def _link_val(self, var: Any) -> int | bool:
        "Get the value of a link variable from the current solution."

    @abc.abstractmethod
    def _flow_val(self, var: Any) -> int:
        "Get the value of a flow variable from the current solution."

    def set_problem(
        self,
        P: nx.PlanarEmbedding,
        A: nx.Graph,
        *,
        constraints: NetworkConstraints | None = None,
        model_options: Mapping[str, Any] | None = None,
        capacity: int | None = None,
        capacity_nominal: float | Fraction | None = None,
        power_rtol: float = DEFAULT_POWER_RTOL,
        warmstart: nx.Graph | None = None,
    ) -> None:
        """Define the problem geometry, available links and tree properties.

        Args:
          P: planar embedding of the location
          A: available links for the location
          constraints: network constraints (defaults to
            ``NetworkConstraints()``)
          model_options: deprecated, use ``constraints``: tree properties - see
            ModelOptions.help()
          capacity: maximum inflow of a subtree
          capacity_nominal: maximum power of a subtree, instead of ``capacity``
            (see :func:`~optiwindnet.loads.quantized`)
          power_rtol: quantization tolerance of unequal turbine power
          warmstart: initial feasible solution to pass to solver

        Raises:
          TypeError: both ``constraints`` and ``model_options`` are given.
          ValueError: the constraints do not suit the problem (see
            :meth:`~optiwindnet.constraints.NetworkConstraints.resolve`).
        """
        if model_options is not None:
            if constraints is not None:
                raise TypeError('Give either constraints or model_options, not both.')
            warnings.warn(
                'set_problem(model_options=...) is deprecated: pass '
                'constraints=NetworkConstraints(...) instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            constraints = _constraints_from_options(**model_options)
        elif constraints is None:
            constraints = NetworkConstraints()
        A, self.capacity, self._power_attrs = quantized(
            A,
            capacity=capacity,
            capacity_nominal=capacity_nominal,
            power_rtol=power_rtol,
        )
        self.P, self.A = P, A
        self.constraints = constraints
        self.bounds = constraints._resolve(
            T=A.graph['T'],
            R=A.graph['R'],
            capacity=self.capacity,
            inflow_total=total_inflow(A),
            power_per_inflow=self._power_attrs.get('power_per_inflow'),
        )
        inflow = nonunit_inflow(A)
        if warmstart is not None and nonunit_inflow(warmstart) != inflow:
            # a warm start quantized otherwise carries loads in other inflow
            warmstart = warmstart.copy()
            for t in range(A.graph['T']):
                warmstart.nodes[t].pop('inflow', None)
            nx.set_node_attributes(warmstart, inflow, 'inflow')
            calcload(warmstart)
        self._set_model(warmstart)

    @abc.abstractmethod
    def _set_model(self, warmstart: nx.Graph | None) -> None:
        "Build the model of ``A``, ``capacity`` and ``bounds``."

    def _create_detours(
        self, S: nx.Graph, P: nx.PlanarEmbedding, A: nx.Graph
    ) -> nx.Graph:
        """Create the detours of the solution topology ``S`` with PathFinder.

        Under per-root bounds, every subtree stays connected to its root in ``S``,
        which is what the bounds were enforced on.
        """
        return PathFinder(
            G_from_S(S, A),
            planar=P,
            A=A,
            allow_root_change=not self.bounds.has_root_bounds,
        ).create_detours()

    @abc.abstractmethod
    def solve(
        self,
        time_limit: float,
        mip_gap: float,
        options: Mapping[str, Any] = MappingProxyType({}),
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

    @abc.abstractmethod
    def _read_incumbent_linkbits(self) -> frozenbitarray:
        """Read the model-objective incumbent's canonical link bits off the solver.

        Called once per search, by :meth:`_record_incumbent`, which keeps the
        result as :attr:`incumbent_linkbits`. Only the binary link variables are
        read: no topology is built, and no solver resource is released, because
        :meth:`get_solution` has yet to walk the solution pool.
        """

    def _record_incumbent(
        self, solution_info: SolutionInfo, applied_options: dict[str, Any]
    ) -> SolutionInfo:
        """Close a search: record the incumbent's link bits and stamp their id.

        Every backend ends ``solve()`` here, once its best solution is readable,
        so that the bits and the id are available as soon as ``solve()`` returns.
        No topology is built: :meth:`get_solution` rebuilds one from these very
        bits. Both stay pinned to the model-objective incumbent, which is not
        necessarily the topology ``get_solution()`` hands over.
        """
        self.solution_info, self.applied_options = solution_info, applied_options
        self._incumbent_linkbits = linkbits = self._read_incumbent_linkbits()
        solution_info.topology_id = topology_id(linkbits)
        info('>>> Solution <<<\n%s\n', solution_info)
        return solution_info

    def _incumbent_topology(self) -> nx.Graph:
        """Rebuild the incumbent topology from the link bits ``solve()`` recorded."""
        return self._S_from_linkbits(self.incumbent_linkbits)

    @abc.abstractmethod
    def get_solution(self, A: nx.Graph | None = None) -> tuple[nx.Graph, nx.Graph]:
        """Output solution topology S and routeset G.

        Args:
          A: optionally replace a normalized A given via set_problem() with its
            unscaled counterpart for routing. It must have the same candidate
            links; the MILP topology is defined by the original A.

        Returns:
          Topology graph S and routeset G.
        """

    def _make_graph_attributes(self) -> dict[str, Any]:
        metadata, solution_info = self.metadata, self.solution_info
        solver_details = self.applied_options.copy()
        # the method_options dict is extracted by db utility function packmethod()
        bounds = self.bounds
        method_options = dict(
            solver_name=self.name,
            fun_fingerprint=metadata.fun_fingerprint,
            **self.stopping,
            topology=bounds.topology,
            feeder_route=bounds.feeder_route,
            balanced=bounds.balanced,
        )
        # like the capacity, the counts are parameters of the method
        solver_details.update(
            (name, value)
            for name in _COUNT_DETAILS
            if (value := getattr(bounds, name)) is not None
        )
        # the incumbent's id is not a routeset attribute: the delivered
        # topology carries its own `_topology_id`, inherited from S
        outcome = {
            key: value
            for key, value in asdict(solution_info).items()
            if key != 'topology_id'
        }
        # remaining graph attributes (key=value) are stored in db.RouteSet[].misc
        attr = dict(
            **outcome,
            method_options=method_options,
            solver_details=solver_details,
        )
        if metadata.warmed_by:
            attr['warmstart'] = metadata.warmed_by
        return attr

    def _linkbits_from_mip_sol(self) -> frozenbitarray:
        """Read the current solution's selected links as canonical link bits.

        Mapping iteration is canonical by construction in all model builders, so
        reading each binary variable once and collapsing the directed blocks with
        bit operations yields a solver-independent solution representation.
        """
        metadata = self.metadata
        feeder_link_count = metadata.R * metadata.T
        ringed = self.bounds.topology is Topology.RINGED
        terminal_link_count = (
            len(metadata.linkset) - feeder_link_count * (1 + ringed)
        ) // 2
        link_values = bitarray(
            (self._link_val(var) for var in metadata.link_.values()), endian='big'
        )
        return linkbits_from_directed(
            link_values,
            terminal_link_count=terminal_link_count,
            feeder_link_count=feeder_link_count,
            ringed=ringed,
        )

    def _S_from_linkbits(self, linkbits: frozenbitarray) -> nx.Graph:
        """Decode link bits and calculate loads from terminal inflow.

        A RINGED model only ever carries unitary inflow (:func:`check_inflow_support`
        refuses the rest), which is what its terminal-counting arm split needs.
        """
        metadata = self.metadata
        topology = self.bounds.topology
        A = self.A
        S = S_from_linkbits(linkbits, A)
        # Preserve terminal inflow so validation can reproduce the loads.
        nx.set_node_attributes(S, nonunit_inflow(A), 'inflow')
        if topology is Topology.RINGED:
            # the bits close every ring: this only splits the arms and
            # derives their loads, in the form every ringed producer uses
            split_rings_and_calc_loads(S, A)
        else:
            calcload(S)
        S.graph.update(
            topology=topology,
            capacity=metadata.capacity,
            _linkbits=linkbits,
            _topology_id=topology_id(linkbits),
            _linkset_id=A.graph['_linkset_id'],
            creator='MILP.' + self.name,
            solver_details={},
            **self._power_attrs,
        )
        return S


class PoolHandler(abc.ABC):
    """Mixed into a :class:`Solver` whose backend keeps a pool of solutions.

    Ranking the pool reads the link bits of the selected entry and decodes them.
    Both members come from the ``Solver``, and are declared below for the reader
    and for the type checker.
    """

    name: str
    num_solutions: int
    solution_info: SolutionInfo

    if TYPE_CHECKING:
        # supplied by the `Solver` every `PoolHandler` is mixed into
        _linkbits_from_mip_sol: Callable[[], frozenbitarray]
        _S_from_linkbits: Callable[[frozenbitarray], nx.Graph]
        _create_detours: Callable[[nx.Graph, nx.PlanarEmbedding, nx.Graph], nx.Graph]

    @abc.abstractmethod
    def _objective_at(self, index: int) -> float:
        """Select the pool solution at ``index`` and return its model objective.

        Selecting it must also make its variable values the ones
        ``_link_val()`` and ``_flow_val()`` read.
        """

    def _read_incumbent_linkbits_from_pool(self) -> frozenbitarray:
        """Read pool entry zero, which must hold the best model objective."""
        try:
            expected_objective = self.solution_info.objective
        except AttributeError as exc:
            exc.args += ('.solve() must be called before solution retrieval',)
            raise
        objective = self._objective_at(0)
        if not math.isclose(objective, expected_objective, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                'Best solution-pool objective does not agree with solution_info: '
                f'{objective} != {expected_objective}'
            )
        return self._linkbits_from_mip_sol()

    def _investigate_pool(
        self, P: nx.PlanarEmbedding, A: nx.Graph
    ) -> tuple[nx.Graph, nx.Graph]:
        """Go through the solver's solutions checking which has the shortest length
        after applying the detours with PathFinder.

        An entry with feeders PathFinder cannot detour is skipped, with a warning.
        """
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
            Sʹ = self._S_from_linkbits(self._linkbits_from_mip_sol())
            Gʹ = self._create_detours(Sʹ, P, A)
            Λʹ = Gʹ.size(weight='length')
            if Gʹ.graph.get('tentative'):
                warn(
                    '#%d skipped (objective: %.3f, length: %.3f): PathFinder found '
                    'no non-crossing path for feeders %s',
                    i,
                    λ,
                    Λʹ,
                    ', '.join(
                        f'{r}→{n} (load {Gʹ.nodes[n]["load"]})'
                        for r, n in Gʹ.graph['tentative']
                    ),
                )
                continue
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
