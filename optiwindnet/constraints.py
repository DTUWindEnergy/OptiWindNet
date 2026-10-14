# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Constraints on the electrical network of a wind farm.

:class:`NetworkConstraints` states the constraints in the terms of the wind farm:
substations, feeders (substation connections) and power. Its
:meth:`~NetworkConstraints.resolve` checks them against a problem instance (a
graph and a cable capacity) and returns a :class:`NetworkBounds`, the bounds the
routers enforce.
"""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Real
from typing import Any

import networkx as nx

from .loads import (
    DEFAULT_POWER_RTOL,
    _validated_capacity_nominal,
    quantized,
    total_inflow,
)
from .types import FeederRoute, Topology

__all__ = ('NetworkBounds', 'NetworkConstraints')

_MIN_COUNT = re.compile(r'min(?:\+(\d+))?')


def _is_count(value: Any, minimum: int) -> bool:
    return (
        isinstance(value, Integral)
        and not isinstance(value, bool)
        and int(value) >= minimum
    )


def _farm_count(name: str, value: Any, per_group: int) -> tuple[bool, int] | None:
    """Parse a whole-farm feeder count into ``(relative, value)``."""
    if value is None:
        return None
    if isinstance(value, str):
        match = _MIN_COUNT.fullmatch(value.replace(' ', ''))
        if match is None:
            raise ValueError(
                f"{name} must be a positive integer, 'min' or 'min+k': got {value!r}."
            )
        relative, count = True, int(match[1] or 0)
    elif _is_count(value, 1):
        relative, count = False, int(value)
    else:
        raise ValueError(
            f"{name} must be a positive integer, 'min' or 'min+k': got {value!r}."
        )
    if count % per_group:
        raise ValueError(
            f'{name} ({value!r}) must count an even number of feeders for a '
            'RINGED topology (each ring has two feeders).'
        )
    return relative, count


def _per_substation(name: str, value: Any, is_valid) -> Any:
    """Validate a per-substation value: a scalar or a sequence (``None`` entries)."""
    if value is None or (not isinstance(value, Sequence) and is_valid(value)):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str):
        entries = tuple(value)
        if all(entry is None or is_valid(entry) for entry in entries):
            return entries
    raise ValueError(f'{name} has an invalid value: {value!r}.')


def _is_load(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _broadcast(name: str, value: Any, R: int) -> tuple[Any, ...]:
    """One entry per substation, from a scalar or a sequence."""
    if value is None:
        return (None,) * R
    if not isinstance(value, tuple):
        return (value,) * R
    if len(value) != R:
        raise ValueError(
            f'{name} has {len(value)} entries, but there are {R} substations.'
        )
    return value


@dataclass(frozen=True)
class NetworkBounds:
    """Bounds of a problem instance, as resolved from :class:`NetworkConstraints`.

    Counts are in feeders (substation connections, two per ring), loads are in
    inflow. Per-root tuples are in root order (``-R`` to ``-1``) and are ``None``
    if no root is bounded.
    """

    topology: Topology
    """Topology of the network."""
    feeder_route: FeederRoute
    """Feeder routing mode."""
    feeders_per_group: int
    """Feeders of a subtree (1) or of a ring (2)."""
    feeders_lb: int
    """Minimum number of feeders."""
    feeders_ub: int | None
    """Maximum number of feeders (``None`` if unlimited)."""
    root_feeders_lb: tuple[int, ...] | None
    """Minimum number of feeders of each root."""
    root_feeders_ub: tuple[int | None, ...] | None
    """Maximum number of feeders of each root (``None`` entries are unlimited)."""
    root_load_ub: tuple[int | None, ...] | None
    """Maximum load of each root (``None`` entries are unlimited)."""
    group_load_lb: int | None
    """Minimum load of a subtree or ring (``None`` if not balanced)."""
    group_load_ub: int | None
    """Maximum load of a subtree or ring (``None`` if not balanced)."""
    link_length_ub: float | None = None
    """Maximum routed length of an available link or feeder in the scale of
    the graph the bounds are resolved against (``None`` if unlimited)."""

    @property
    def groups_lb(self) -> int:
        """Minimum number of subtrees or rings."""
        return self.feeders_lb // self.feeders_per_group

    @property
    def groups_ub(self) -> int | None:
        """Maximum number of subtrees or rings (``None`` if unlimited)."""
        if self.feeders_ub is None:
            return None
        return self.feeders_ub // self.feeders_per_group

    @property
    def balanced(self) -> bool:
        """Whether the subtree (or ring) loads are balanced."""
        return self.group_load_lb is not None

    @property
    def has_root_bounds(self) -> bool:
        """Whether any root's feeders or load are bounded."""
        return any(
            bounds is not None
            for bounds in (
                self.root_feeders_lb,
                self.root_feeders_ub,
                self.root_load_ub,
            )
        )


def _check_link_length(A: nx.Graph, bounds: NetworkBounds) -> None:
    """Refuse a link length limit that is out of scale for ``A``.

    The checks are necessary conditions only, meant to catch a limit in the
    wrong unit or scale: every terminal needs a link or a feeder within the
    limit, and every subtree or ring needs a feeder within the limit to a
    terminal of its own. Whether the terminals can reach the roots in strings
    is left to the routers.

    Raises:
      ValueError: the limit leaves some terminal without any link or feeder, or
        leaves fewer terminals with a feeder than the subtrees or rings needed.
    """
    ub = bounds.link_length_ub
    if ub is None:
        return
    T = A.graph['T']
    feedable_ = (A.graph['d2roots'][:T] <= ub).any(axis=1)
    linked_ = feedable_.copy()
    for u, v, length in A.edges(data='length'):
        if u >= 0 and v >= 0 and length <= ub:
            linked_[u] = linked_[v] = True
    isolated = (~linked_).nonzero()[0].tolist()
    if isolated:
        raise ValueError(
            f'max_link_length leaves {len(isolated)} turbine(s) without any link '
            f'or feeder within it: {isolated[:10]}{"..." if len(isolated) > 10 else ""}'
        )
    num_feedable = int(feedable_.sum())
    if num_feedable < bounds.groups_lb:
        raise ValueError(
            f'max_link_length leaves {num_feedable} turbine(s) with a feeder '
            f'within it, fewer than the {bounds.groups_lb} subtrees or rings needed.'
        )


@dataclass(frozen=True, init=False)
class NetworkConstraints:
    """Constraints on the electrical network of a wind farm.

    ``topology`` and ``feeder_route`` limit what a solution may be: a radial
    network satisfies ``'branched'``, and straight feeders satisfy
    ``'segmented'``.

    Feeder counts are in feeders, i.e. substation connections (a ring has two).
    The whole-farm count is either exact (``feeders``) or an upper bound
    (``max_feeders``), given as an integer or relative to the minimum:
    ``'min'`` or ``'min+k'`` (``k`` extra feeders). The minimum is the capacity
    bound ``ceil(total load / capacity)`` (in subtrees or rings), which
    indivisible turbine power or substation limits may make unattainable;
    relative counts are rejected if turbine power is unequal.

    Per-substation counts are also exact (``substation_feeders``) or upper
    bounds (``max_substation_feeders``). Each is one integer for every
    substation or a sequence with one entry per substation, from ``-R`` to ``-1``
    (``None`` entries leave a substation free). A substation may have one or the
    other, not both.

    ``max_substation_loads`` limits the power each substation collects, in the
    unit of the turbine power (or in turbines, if no power is given).

    ``balanced`` makes the loads of all subtrees (or rings) of the network
    differ at most by one unit of the turbine power quantization. It requires
    the feeder count to be fixed, e.g. ``feeders='min'``.

    ``max_link_length`` limits the maximum routed length of available links and
    feeders in the unit of the coordinates. The available-links graph records
    lengths that account for contours around obstacles. Detours are created
    after optimization and are not represented in that graph; they may stretch
    a link past the limit (see
    :func:`~optiwindnet.interarraylib.links_over_length`).

    Args:
      topology: ``'radial'``, ``'branched'`` or ``'ringed'``.
      feeder_route: ``'segmented'`` or ``'straight'``.
      feeders: exact number of feeders of the wind farm.
      max_feeders: maximum number of feeders of the wind farm.
      substation_feeders: exact number of feeders of each substation.
      max_substation_feeders: maximum number of feeders of each substation.
      max_substation_loads: maximum load of each substation.
      balanced: whether the subtree (or ring) loads must be balanced.
      max_link_length: maximum routed length of an available link or feeder.

    Raises:
      ValueError: a value has the wrong form, or two constraints exclude each
        other regardless of the problem instance.

    Example::

      NetworkConstraints(max_substation_feeders=6, max_feeders='min+1')
    """

    topology: Topology
    feeder_route: FeederRoute
    feeders: int | str | None
    max_feeders: int | str | None
    substation_feeders: int | tuple[int | None, ...] | None
    max_substation_feeders: int | tuple[int | None, ...] | None
    max_substation_loads: float | Fraction | tuple[float | Fraction | None, ...] | None
    balanced: bool
    max_link_length: float | None

    def __init__(
        self,
        topology: Topology | str = Topology.DEFAULT,
        feeder_route: FeederRoute | str = FeederRoute.DEFAULT,
        feeders: int | str | None = None,
        max_feeders: int | str | None = None,
        substation_feeders: int | Sequence[int | None] | None = None,
        max_substation_feeders: int | Sequence[int | None] | None = None,
        max_substation_loads: float
        | Fraction
        | Sequence[float | Fraction | None]
        | None = None,
        balanced: bool = False,
        max_link_length: float | None = None,
    ) -> None:
        topology = Topology(topology)
        per_group = 2 if topology is Topology.RINGED else 1
        if feeders is not None and max_feeders is not None:
            raise ValueError('Give either feeders or max_feeders, not both.')
        _farm_count('feeders', feeders, per_group)
        _farm_count('max_feeders', max_feeders, per_group)

        def is_count(value):
            return _is_count(value, 0)

        substation_feeders = _per_substation(
            'substation_feeders', substation_feeders, is_count
        )
        max_substation_feeders = _per_substation(
            'max_substation_feeders', max_substation_feeders, is_count
        )
        if (
            substation_feeders is not None
            and max_substation_feeders is not None
            and not (
                isinstance(substation_feeders, tuple)
                and isinstance(max_substation_feeders, tuple)
                and len(substation_feeders) == len(max_substation_feeders)
                and all(
                    exact is None or most is None
                    for exact, most in zip(substation_feeders, max_substation_feeders)
                )
            )
        ):
            raise ValueError(
                'A substation may have either substation_feeders or '
                'max_substation_feeders, not both.'
            )
        if not isinstance(balanced, bool):
            raise TypeError(f'balanced must be a bool: got {balanced!r}.')
        if max_link_length is not None and not _is_load(max_link_length):
            raise ValueError(
                'max_link_length must be a positive and finite number: got '
                f'{max_link_length!r}.'
            )
        for name, value in (
            ('topology', topology),
            ('feeder_route', FeederRoute(feeder_route)),
            ('feeders', feeders),
            ('max_feeders', max_feeders),
            ('substation_feeders', substation_feeders),
            ('max_substation_feeders', max_substation_feeders),
            (
                'max_substation_loads',
                _per_substation('max_substation_loads', max_substation_loads, _is_load),
            ),
            ('balanced', balanced),
            (
                'max_link_length',
                None if max_link_length is None else float(max_link_length),
            ),
        ):
            object.__setattr__(self, name, value)

    def resolve(
        self,
        A: nx.Graph,
        *,
        capacity: int | None = None,
        capacity_nominal: float | Fraction | None = None,
        power_rtol: float = DEFAULT_POWER_RTOL,
    ) -> NetworkBounds:
        """Check the constraints against a problem instance and bound it.

        The instance is ``A`` (an available-links graph) at a cable
        capacity, given as for the routers' producers: ``capacity`` in inflow,
        or ``capacity_nominal`` in the unit of the turbine power, which is then
        quantized (see :func:`~optiwindnet.loads.quantized`). The bounds are
        those that :func:`~optiwindnet.baselines.hgs.hgs_cvrp` and
        :meth:`Solver.set_problem() <optiwindnet.MILP.Solver.set_problem>` derive
        for the same arguments.

        A ``max_link_length`` must leave every turbine a way to a substation
        through links and feeders within it; it is scaled as ``A`` is, if ``A``
        is normalized (see :func:`~optiwindnet.transforming.as_normalized`).

        The per-substation counts must be compatible with the whole-farm count:
        their sums must leave a count within the whole-farm bounds, which are
        narrowed to it. The substations must also have room for the total load,
        given their feeders (each carrying up to the capacity) and their load
        limits.

        Args:
          A: available-links graph.
          capacity: cable capacity, in inflow (per arm, for rings).
          capacity_nominal: cable capacity, in the unit of the turbine power,
            instead of ``capacity``.
          power_rtol: quantization tolerance of unequal turbine power.

        Returns:
          The bounds the routers enforce.

        Raises:
          ValueError: the constraints do not suit the instance, or cannot all
            be met; or the capacity does not suit the power ``A`` declares.
        """
        A_solve, capacity_inflow, power_attrs = quantized(
            A,
            capacity=capacity,
            capacity_nominal=capacity_nominal,
            power_rtol=power_rtol,
        )
        bounds = self._resolve(
            T=A.graph['T'],
            R=A.graph['R'],
            capacity=capacity_inflow,
            inflow_total=total_inflow(A_solve),
            power_per_inflow=power_attrs.get('power_per_inflow'),
            link_length_ub=self._link_length_ub(A),
        )
        _check_link_length(A, bounds)
        return bounds

    def _resolve(
        self,
        *,
        T: int,
        R: int,
        capacity: int,
        inflow_total: int | None = None,
        power_per_inflow: Fraction | None = None,
        link_length_ub: float | None = None,
    ) -> NetworkBounds:
        """Check the constraints against a problem instance given by numbers.

        See :meth:`resolve`, which derives the numbers from a graph.

        Args:
          T: number of turbines.
          R: number of substations.
          capacity: cable capacity, in inflow (per arm, for rings).
          inflow_total: sum of the turbines' inflow (defaults to ``T``).
          power_per_inflow: power of one unit of inflow, if the turbine power
            has a unit (then ``max_substation_loads`` is in that unit, and is
            rounded down to whole inflow); otherwise ``max_substation_loads``
            is in inflow and must be integer.
          link_length_ub: ``max_link_length`` in the scale of the graph, as
            given by :meth:`_link_length_ub` (required if it is set).

        Returns:
          The bounds the routers enforce.

        Raises:
          ValueError: the constraints do not suit the instance, or cannot all
            be met.
        """
        if (self.max_link_length is None) != (link_length_ub is None):
            raise ValueError(
                'link_length_ub must be given if, and only if, max_link_length is.'
            )
        if inflow_total is None:
            inflow_total = T
        per_group = 2 if self.topology is Topology.RINGED else 1
        min_feeders = per_group * math.ceil(inflow_total / (per_group * capacity))
        # each subtree or ring holds at least one turbine
        most_feeders = per_group * T

        lb, ub = min_feeders, most_feeders
        for name in ('feeders', 'max_feeders'):
            count = _farm_count(name, getattr(self, name), per_group)
            if count is None:
                continue
            relative, value = count
            if relative:
                if inflow_total != T:
                    raise ValueError(
                        f'{name} cannot be relative to the minimum with unequal '
                        'turbine power: the minimum may not be attainable.'
                    )
                value += min_feeders
            elif value < min_feeders:
                raise ValueError(
                    f'{name} ({value}) is below the minimum necessary ({min_feeders}).'
                )
            if value > most_feeders:
                raise ValueError(
                    f'{name} ({value}) is above the maximum possible ({most_feeders}).'
                )
            ub = value
            if name == 'feeders':
                lb = value

        exact_ = _broadcast('substation_feeders', self.substation_feeders, R)
        most_ = _broadcast('max_substation_feeders', self.max_substation_feeders, R)
        loads = _broadcast('max_substation_loads', self.max_substation_loads, R)
        load_ub_ = tuple(
            None if load is None else self._load_in_inflow(load, power_per_inflow)
            for load in loads
        )
        lo_ = tuple(exact or 0 for exact in exact_)
        hi_ = tuple(
            math.inf
            if exact is None and most is None
            else exact
            if most is None
            else most
            for exact, most in zip(exact_, most_)
        )
        for r, lo, load_ub in zip(range(-R, 0), lo_, load_ub_):
            # every subtree or ring carries at least one unit of inflow
            if load_ub is not None and math.ceil(lo / per_group) > load_ub:
                raise ValueError(
                    f'Substation {r} needs {lo} feeders, more than its load limit '
                    f'can supply.'
                )
        room = sum(
            min(hi * capacity, math.inf if load_ub is None else load_ub)
            for hi, load_ub in zip(hi_, load_ub_)
        )
        if room < inflow_total:
            raise ValueError(
                f'The substation limits hold a load of {room}, less than the total '
                f'({inflow_total}).'
            )
        lo_sum, hi_sum = sum(lo_), sum(hi_)
        if lo_sum > ub or hi_sum < lb:
            raise ValueError(
                f'The substation feeder counts allow between {lo_sum} and {hi_sum} '
                f'feeders, outside the range for the wind farm ({lb} to {ub}).'
            )
        # narrow to whole groups within both ranges
        lb = max(lb, per_group * math.ceil(lo_sum / per_group))
        if hi_sum < ub:
            ub = per_group * (int(hi_sum) // per_group)
        if lb > ub:
            raise ValueError(
                'The substation feeder counts leave no whole number of rings for '
                'the wind farm.'
            )

        group_load_lb = group_load_ub = None
        if self.balanced:
            if lb != ub:
                raise ValueError(
                    'balanced requires a fixed feeder count: give feeders (e.g. '
                    "feeders='min'), or substation_feeders for every substation."
                )
            groups = lb // per_group
            group_load_lb = inflow_total // groups
            group_load_ub = math.ceil(inflow_total / groups)
            if per_group == 1:
                for r, lo, load_ub in zip(range(-R, 0), lo_, load_ub_):
                    if load_ub is not None and lo * group_load_lb > load_ub:
                        raise ValueError(
                            f'Substation {r} cannot hold {lo} balanced subtrees '
                            'within its load limit.'
                        )

        return NetworkBounds(
            topology=self.topology,
            feeder_route=self.feeder_route,
            feeders_per_group=per_group,
            feeders_lb=lb,
            feeders_ub=None if ub == most_feeders and self._unlimited() else ub,
            root_feeders_lb=lo_ if any(lo_) else None,
            root_feeders_ub=(
                tuple(None if hi == math.inf else int(hi) for hi in hi_)
                if any(hi < math.inf for hi in hi_)
                else None
            ),
            root_load_ub=load_ub_ if any(ub is not None for ub in load_ub_) else None,
            group_load_lb=group_load_lb,
            group_load_ub=group_load_ub,
            link_length_ub=link_length_ub,
        )

    def _link_length_ub(self, A: nx.Graph) -> float | None:
        """Return ``max_link_length`` in the scale of ``A``.

        The limit is in the unit of the coordinates, so it is scaled as ``A`` is
        if ``A`` is normalized.
        """
        if self.max_link_length is None:
            return None
        ub = self.max_link_length
        if A.graph.get('is_normalized', False):
            ub *= A.graph['norm_scale']
        return ub

    def _unlimited(self) -> bool:
        """Whether the whole-farm count is left free (only the implicit bound)."""
        return self.feeders is None and self.max_feeders is None

    @staticmethod
    def _load_in_inflow(load: Any, power_per_inflow: Fraction | None) -> int:
        if power_per_inflow is None:
            if not _is_count(load, 1):
                raise ValueError(
                    'max_substation_loads must be positive integers (turbines or '
                    f'inflow) when the turbine power has no unit: got {load!r}.'
                )
            return int(load)
        nominal = _validated_capacity_nominal(load)
        return math.floor(nominal / Fraction(power_per_inflow))
