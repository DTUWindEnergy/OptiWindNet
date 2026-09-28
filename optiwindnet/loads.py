# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Cable-load computation over solution topologies and routesets."""

import functools
import math
from collections.abc import Iterable, Sequence
from fractions import Fraction
from numbers import Integral, Real
from typing import Any

import networkx as nx
import numpy as np

from .types import Topology

__all__ = (
    'bfs_subtree_loads',
    'calcload',
    'nonunit_inflow',
    'quantize_for_capacity',
    'quantized',
    'set_terminal_power',
    'split_rings_and_calc_loads',
    'total_inflow',
    'validate_terminal_power',
)  # fmt: skip


# Below this power per inflow the inflow exceed what a MILP can carry, so a
# tolerance that reaches it raises instead of returning the resulting integers.
_MAX_INFLOW = 10_000
# Relative tolerance the producers quantize to by default.
DEFAULT_POWER_RTOL = 0.01
# Graph attributes that describe the power of the terminals.
_POWER_GRAPH_ATTRS = (
    'power_per_inflow', 'power_unit', 'powers_set', 'power_quantization_inexact',
)  # fmt: skip


def _simplest_between(
    lo: Fraction, hi: Fraction, *, lo_open: bool = False, hi_open: bool = False
) -> Fraction:
    """Return the rational of least denominator within the given interval.

    The interval must be non-empty and its endpoints non-negative. Whenever it
    holds a whole number, the least of those is the answer; otherwise both
    endpoints share a whole part, and inverting what remains of them maps the
    interval to a wider one whose answer inverts back. Endpoint flags exclude
    the corresponding bounds; a singleton interval must be closed.
    """
    whole = math.floor(lo)
    if whole == lo and not lo_open:
        return Fraction(whole)
    if whole + 1 < hi or (whole + 1 == hi and not hi_open):
        return Fraction(whole + 1)
    if whole == lo:
        reciprocal = 1 / (hi - whole)
        denominator = math.floor(reciprocal) + 1 if hi_open else math.ceil(reciprocal)
        return whole + Fraction(1, denominator)
    return whole + 1 / _simplest_between(
        1 / (hi - whole),
        1 / (lo - whole),
        lo_open=hi_open,
        hi_open=lo_open,
    )


def _plainest_near(
    lo: Fraction,
    hi: Fraction,
    target: Fraction,
    *,
    lo_open: bool = False,
    hi_open: bool = False,
) -> Fraction:
    """Return the least-denominator rational nearest the target in the interval.

    Plainness is the denominator: among the rationals the interval admits, those
    of least denominator read as the quantity they stand for. Several can share
    that denominator -- any interval spanning a whole number admits two whole
    numbers at least -- and ``target`` decides between them, so that plainness
    never costs accuracy it need not. Endpoint flags exclude the corresponding
    bounds; the interval must be non-empty.
    """
    denominator = _simplest_between(
        lo, hi, lo_open=lo_open, hi_open=hi_open
    ).denominator
    first = math.floor(lo * denominator) + 1 if lo_open else math.ceil(lo * denominator)
    last = math.ceil(hi * denominator) - 1 if hi_open else math.floor(hi * denominator)
    numerator = min(
        max(round(target * denominator), first),
        last,
    )
    return Fraction(numerator, denominator)


def _rationalize(value: float) -> Fraction:
    """Return the simplest number that ``value`` is the nearest float to.

    A declared power is the rounding of a number someone wrote, so ``6.6`` is
    taken as ``33/5`` rather than as the binary expansion the float holds. Exact
    arithmetic on the numbers meant keeps the quantization free of rounding
    artifacts: three turbines of ``6.6 / 3`` fill a cable of ``6.6`` exactly,
    which their floats do not.
    """
    exact = Fraction(value)
    lo = (exact + Fraction(math.nextafter(value, -math.inf))) / 2
    hi = (exact + Fraction(math.nextafter(value, math.inf))) / 2
    return _plainest_near(lo, hi, exact)


def _inflow_of(power: Fraction, power_per_inflow: Fraction) -> int:
    """Return the integer inflow that quantizes ``power``.

    The ratio of ``power`` to ``power_per_inflow`` is rounded half up, and never
    below one. Unlike the round-half-even of :func:`round`, rounding half up
    gives every inflow ``k`` the same half-open range of ``power_per_inflow``:
    ``power / (k + 1/2) < power_per_inflow <= power / (k - 1/2)``.
    """
    return max(1, math.floor(power / power_per_inflow + Fraction(1, 2)))


def _packing_interval(
    unique: list[Fraction],
    inflow: list[int],
    capacities: Sequence[Fraction],
    lo: Fraction,
    hi: Fraction,
    lo_open: bool,
) -> tuple[Fraction, Fraction, bool] | None:
    """Intersect a quantization interval with cable-packing constraints.

    A cable of nominal capacity ``C`` becomes one of
    ``floor(C / power_per_inflow)`` inflow. The two agree when, for every
    multiset of at most ``floor(C / min_power) + 1`` turbines, the nominal sum
    fits within ``C`` exactly when the inflow sum fits within its integer
    capacity. The extra turbine matters: such multisets always overflow ``C``,
    yet their inflow sum may undercut that of every smaller overflowing
    multiset. Larger multisets contain one of these, so they add no bound. The
    greatest fitting inflow bounds that
    integer capacity below; the least overflowing inflow bounds it above.
    These bounds define an interval open on the left and closed on the right
    for ``power_per_inflow``. Integer capacities must be positive.
    """

    def walk(
        i: int,
        left: int,
        nominal: Fraction,
        integral: int,
        nominal_cap: Fraction,
    ) -> bool:
        nonlocal must_fit, must_not_fit
        if nominal > nominal_cap:
            # Extensions only increase inflow, so cannot tighten this bound.
            must_not_fit = (
                integral if must_not_fit is None else min(must_not_fit, integral)
            )
            return must_fit < must_not_fit
        must_fit = max(must_fit, integral)
        if must_not_fit is not None and must_fit >= must_not_fit:
            return False
        if i == len(unique):
            return True
        return all(
            walk(
                i + 1,
                left - c,
                nominal + c * unique[i],
                integral + c * inflow[i],
                nominal_cap,
            )
            for c in range(left + 1)
        )

    for capacity in capacities:
        must_fit = 1
        must_not_fit: int | None = None
        limit = capacity // unique[0] + 1
        if not walk(0, limit, Fraction(0), 0, capacity):
            return None
        hi = min(hi, capacity / must_fit)
        if must_not_fit is not None:
            bound = capacity / must_not_fit
            if bound >= lo:
                lo, lo_open = bound, True
        if lo > hi or (lo == hi and lo_open):
            return None
    return lo, hi, lo_open


def _validated_rtol(rtol: float) -> Fraction:
    """The quantization tolerance as an exact rational in ``[0, 1)``."""
    numeric = isinstance(rtol, Real) and not isinstance(rtol, bool)
    if not numeric or not 0 <= float(rtol) < 1:
        raise ValueError(f'power_rtol must be a number in [0, 1): got {rtol!r}.')
    return _rationalize(float(rtol))


def _validated_powers(
    turbine_powers: Sequence[float | Fraction] | np.ndarray,
    T: int | None,
    name: str = 'turbine_powers',
) -> list[Fraction]:
    """Power values, as exact positive rationals, ``T`` of them unless None."""
    if T is not None and len(turbine_powers) != T:
        raise ValueError(
            f'{name} has {len(turbine_powers)} entries but T={T} turbines.'
        )
    powers = []
    for raw_power in turbine_powers:
        numeric = isinstance(raw_power, Real) and not isinstance(raw_power, bool)
        if isinstance(raw_power, (Fraction, Integral)) and numeric:
            power = Fraction(raw_power)
        else:
            power = float(raw_power) if numeric else float('nan')
        if not isinstance(power, Fraction) and not math.isfinite(power):
            raise ValueError(f'All {name} must be finite numbers: got {raw_power!r}.')
        if power <= 0:
            raise ValueError(f'All {name} must be positive: got {raw_power!r}.')
        powers.append(power if isinstance(power, Fraction) else _rationalize(power))
    return powers


def _is_positive_integer(value) -> bool:
    return (
        isinstance(value, Integral) and not isinstance(value, bool) and int(value) >= 1
    )


def _validated_capacity_nominal(capacity_nominal: float | Fraction) -> Fraction:
    """The nominal capacity as an exact positive rational."""
    return _validated_powers([capacity_nominal], 1, 'capacity_nominal')[0]


def _power_quantization_intervals(unique: list[Fraction], rtol: Fraction):
    """Yield tolerance intervals for successive mappings, from coarse to fine.

    A mapping's rounding interval is open below and closed above. At its lower
    boundary, at least one inflow increases, so visiting that boundary traverses
    every mapping without skipping half-inflow ties. Intersect with tolerance
    bounds and the minimum power per inflow permitted by the solver limit.
    """
    if len(unique) == 1:
        yield [1], unique[0], unique[0], False, unique[0]
        return
    floor = unique[-1] / _MAX_INFLOW
    candidate = unique[0] * (1 + rtol)
    while candidate >= floor:
        inflow = [_inflow_of(p, candidate) for p in unique]
        rounding_lo = max(2 * p / (2 * k + 1) for p, k in zip(unique, inflow))
        lo = max(
            floor,
            rounding_lo,
            max(p * (1 - rtol) / k for p, k in zip(unique, inflow)),
        )
        hi = min(
            min(p * (1 + rtol) / k, 2 * p / (2 * k - 1)) for p, k in zip(unique, inflow)
        )
        lo_open = lo == rounding_lo
        candidate = rounding_lo
        if lo > hi or (lo == hi and lo_open):
            continue
        # The least squared deviation, which the interval need not contain.
        fit = Fraction(
            sum(k * p for p, k in zip(unique, inflow)), sum(k * k for k in inflow)
        )
        yield inflow, lo, hi, lo_open, fit


@functools.lru_cache(maxsize=256)
def _quantize_powers_set(
    powers_set: tuple[Fraction, ...], capacity: Fraction, rtol: Fraction
) -> tuple[tuple[int, ...], Fraction, int]:
    """Quantize validated distinct powers for one nominal capacity (memoized)."""
    unique = list(powers_set)
    for inflow, lo, hi, lo_open, fit in _power_quantization_intervals(unique, rtol):
        interval = _packing_interval(unique, inflow, (capacity,), lo, hi, lo_open)
        if interval is None:
            continue
        lo, hi, lo_open = interval
        power_per_inflow = _plainest_near(lo, hi, fit, lo_open=lo_open)
        return tuple(inflow), power_per_inflow, capacity // power_per_inflow
    raise ValueError(
        f'No power_per_inflow packs turbines of powers {[str(p) for p in unique]} '
        f'into capacity_nominal={capacity} as their nominal power does, within '
        f'power_rtol={rtol} at an inflow of at most {_MAX_INFLOW}; raise the '
        'tolerance.'
    )


def quantize_for_capacity(
    powers_set: Iterable[float | Fraction],
    capacity_nominal: float | Fraction,
    power_rtol: float = DEFAULT_POWER_RTOL,
) -> tuple[dict[Fraction, int], Fraction, int]:
    """Quantize turbine power while preserving cable-packing feasibility.

    The capacity constrains ``power_per_inflow`` beyond accuracy: among the
    mappings that keep every power within ``power_rtol``, this takes the first
    in order from coarse to fine that preserves packing. It selects the
    least-denominator rational nearest the least-squares fit within that
    mapping's feasible interval. A cable then admits the same sets of turbines
    in inflow as in nominal power, and the routed network is the one the nominal
    powers define.

    The exact ``power_per_inflow`` preserves packing because every achievable
    load is a whole multiple of it. It guarantees a candidate when its inflows
    are within the solver limit and the integer capacity is positive.

    Only the distinct powers matter, not how many turbines share each. Results
    are memoized.

    Args:
        powers_set: distinct turbine powers, in nominal units.
        capacity_nominal: cable capacity, in the same nominal units.
        power_rtol: most per-turbine deviation accepted from a larger
            ``power_per_inflow``.

    Returns:
        Inflow of each distinct power, nominal power per inflow, and the
        capacity in whole inflow.

    Raises:
        ValueError: the inputs are not positive finite numbers, ``power_rtol``
            is not in ``[0, 1)``, or no ``power_per_inflow`` preserves the
            packing at an inflow a MILP can carry.
    """
    powers = list(powers_set)
    unique = tuple(sorted(set(_validated_powers(powers, len(powers)))))
    inflow, power_per_inflow, capacity = _quantize_powers_set(
        unique,
        _validated_capacity_nominal(capacity_nominal),
        _validated_rtol(power_rtol),
    )
    return dict(zip(unique, inflow)), power_per_inflow, capacity


def quantized(
    A: nx.Graph,
    *,
    capacity: int | None = None,
    capacity_nominal: float | Fraction | None = None,
    power_rtol: float = DEFAULT_POWER_RTOL,
) -> tuple[nx.Graph, int, dict[str, Any]]:
    """Validate the power of ``A`` and quantize it for a solve.

    Exactly one of ``capacity`` and ``capacity_nominal`` is given. ``capacity``
    is in inflow and requires ``A`` to declare no unequal power.
    ``capacity_nominal`` is in the unit of the power ``A`` declares; unequal
    powers are quantized for it (see :func:`quantize_for_capacity`) on a copy of
    ``A`` whose terminals get ``'inflow'``.

    Args:
      A: location or available-links graph.
      capacity: cable capacity in inflow.
      capacity_nominal: cable capacity in nominal power.
      power_rtol: quantization tolerance, used with ``capacity_nominal``.

    Returns:
      The graph to solve over, the capacity in inflow, and the graph attributes
      that record the quantization on the solution.

    Raises:
      ValueError: both or neither capacity is given, the capacity does not suit
        the power ``A`` declares, or the quantization fails.
    """
    validate_terminal_power(A)
    graph = A.graph
    powers_set = graph.get('powers_set')
    if (capacity is None) == (capacity_nominal is None):
        raise ValueError('Exactly one of capacity and capacity_nominal must be given.')
    if capacity_nominal is None:
        if powers_set is not None:
            raise ValueError('A declares unequal turbine power: pass capacity_nominal.')
        if capacity is None or not _is_positive_integer(capacity):
            raise ValueError(f'capacity must be a positive integer: got {capacity!r}.')
        return A, int(capacity), {}
    if powers_set is None and 'power_per_inflow' not in graph:
        raise ValueError('capacity_nominal requires A to declare turbine power.')
    powers = powers_set or (Fraction(graph['power_per_inflow']),)
    nominal = _validated_capacity_nominal(capacity_nominal)
    if nominal < powers[-1]:
        raise ValueError(
            f'capacity_nominal={capacity_nominal!r} is below the turbine power '
            f'{powers[-1]}.'
        )
    inflow, power_per_inflow, integer_capacity = _quantize_powers_set(
        powers, nominal, _validated_rtol(power_rtol)
    )
    attrs = {
        'power_per_inflow': power_per_inflow,
        'capacity_nominal': nominal,
        'power_rtol': power_rtol,
    }
    if powers_set is None:
        return A, integer_capacity, attrs
    solved = A.copy()
    inflow_by_power = dict(zip(powers, inflow))
    for t in range(graph['T']):
        solved.nodes[t]['inflow'] = inflow_by_power[solved.nodes[t]['power']]
    solved.graph['power_per_inflow'] = power_per_inflow
    validate_terminal_power(solved)
    return solved, integer_capacity, attrs


def total_inflow(G: nx.Graph) -> int:
    """Return total terminal power injection in inflow integer units."""
    return sum(G.nodes[t].get('inflow', 1) for t in range(G.graph['T']))


def nonunit_inflow(G: nx.Graph) -> dict[int, int]:
    """Return terminal injections differing from one inflow, keyed by node."""
    return {
        t: inflow
        for t in range(G.graph['T'])
        if (inflow := G.nodes[t].get('inflow', 1)) != 1
    }


def validate_terminal_power(G: nx.Graph) -> None:
    """Validate terminal power, normalizing declared values to fractions.

    ``'power_per_inflow'`` is a positive rational and terminal ``'inflow'`` a
    positive integer, one by default. With ``'powers_set'`` (at least two
    distinct powers), every terminal has a ``'power'`` listed there. A graph
    with ``'powers_set'`` but no ``'power_per_inflow'`` declares unequal power,
    and its terminals have no ``'inflow'``. Otherwise, a terminal's ``'power'``
    must quantize to its inflow (their ratio to ``'power_per_inflow'`` rounds
    half up to it), and the graph attribute ``'power_quantization_inexact'`` is
    refreshed.

    Raises:
        ValueError: any of these conventions is violated.
    """
    graph = G.graph
    T = graph['T']
    nodes = G.nodes
    raw_power_per_inflow = graph.get('power_per_inflow', 1)
    numeric = isinstance(raw_power_per_inflow, (Fraction, Integral)) and not isinstance(
        raw_power_per_inflow, bool
    )
    power_per_inflow = Fraction(raw_power_per_inflow) if numeric else Fraction(0)
    if power_per_inflow <= 0:
        raise ValueError('power_per_inflow must be a positive rational.')
    powers_set = None
    if 'powers_set' in graph:
        powers_set = tuple(
            sorted(
                set(_validated_powers(list(graph['powers_set']), None, 'powers_set'))
            )
        )
        if len(powers_set) < 2:
            raise ValueError("'powers_set' must list at least two distinct powers.")
        graph['powers_set'] = powers_set
    declaration = powers_set is not None and 'power_per_inflow' not in graph
    inexact = False
    for t in range(T):
        attrs = nodes[t]
        raw_inflow = attrs.get('inflow', 1)
        if declaration and 'inflow' in attrs:
            raise ValueError(
                f"terminal {t} declares 'inflow' on a graph declaring 'powers_set' "
                "without 'power_per_inflow'"
            )
        if not _is_positive_integer(raw_inflow):
            raise ValueError(
                f'terminal {t} declares inflow {raw_inflow!r}: '
                "'inflow' must be a positive integer"
            )
        if 'power' not in attrs:
            if powers_set is not None:
                raise ValueError(f"terminal {t} declares no 'power'")
            continue
        power = _validated_powers([attrs['power']], 1)[0]
        if powers_set is not None and power not in powers_set:
            raise ValueError(
                f"terminal {t} declares power {power}, which 'powers_set' lacks"
            )
        attrs['power'] = power
        if declaration:
            continue
        inflow = int(raw_inflow)
        if inflow != _inflow_of(power, power_per_inflow):
            raise ValueError(
                f'terminal {t} declares power {power}, which quantizes to a '
                f'different inflow than {inflow} at {power_per_inflow} per inflow'
            )
        inexact |= power != inflow * power_per_inflow
    if powers_set is not None and powers_set != tuple(
        sorted({nodes[t]['power'] for t in range(T)})
    ):
        raise ValueError("'powers_set' lists powers that no terminal declares")
    if declaration:
        graph.pop('power_quantization_inexact', None)
    else:
        graph['power_quantization_inexact'] = inexact


def _clear_terminal_power(G: nx.Graph) -> None:
    """Remove the terminals' power and inflow and the graph's power attributes."""
    nodes = G.nodes
    for t in range(G.graph['T']):
        nodes[t].pop('inflow', None)
        nodes[t].pop('power', None)
    for key in _POWER_GRAPH_ATTRS:
        G.graph.pop(key, None)


def set_terminal_power(
    L: nx.Graph,
    powers: Sequence[float | Fraction] | np.ndarray,
    power_unit: str | None = None,
) -> None:
    """Set the nominal power of each terminal, replacing any power attributes.

    Equal powers become the graph attribute ``'power_per_inflow'``; unequal ones
    become terminal ``'power'`` and the graph attribute ``'powers_set'``.

    Args:
      L: location or available-links graph, changed in place.
      powers: nominal power of each terminal; floats are rationalized.
      power_unit: physical unit of the powers, e.g. ``'MW'``.

    Raises:
      ValueError: ``powers`` are not ``T`` positive finite numbers.
    """
    fractions = _validated_powers(powers, L.graph['T'])
    _clear_terminal_power(L)
    powers_set = tuple(sorted(set(fractions)))
    if len(powers_set) == 1:
        L.graph['power_per_inflow'] = powers_set[0]
    else:
        for t, power in enumerate(fractions):
            L.nodes[t]['power'] = power
        L.graph['powers_set'] = powers_set
    if power_unit is not None:
        L.graph['power_unit'] = power_unit
    validate_terminal_power(L)


# A link's ``'reverse'`` flag orients it independently of the node order it
# happens to be stored in. Current flows from the terminal that sources it to the
# root that sinks it, and readers recover that direction with::
#
#     u, v = (u, v) if ((u < v) == edgeD['reverse']) else (v, u)
#
# which yields ``(source, sink)`` for either stored order. So every writer sets
# ``reverse = source < sink``. Beware of writing ``load[u] < load[v]`` instead: it
# only matches while ``u < v`` holds, and silently mis-orients links stored the
# other way round. Feeders are never reversed, the sink being a root and a root's
# id negative; links carrying ``load=0`` (a ring's zero-load link) have no current and
# so no direction to encode.
def _bfs_loads_walk(_adj, _node, T, visited, queue, power_per_inflow=None) -> int:
    """Descend the subtrees seeded in ``queue``, appending every node reached.

    Each ``queue`` entry is ``(node, parent, edgeD, parentD, subtree)``, with
    ``edgeD`` the ⟨parent, node⟩ link data and ``parentD`` the parent's node
    data. Queue (BFS) order places every node before its descendants, which is
    what :func:`_bfs_loads_unwind` relies on. Every node gets the base its
    descendants' loads are added to in the target attribute; the accumulation
    happens on the way back up. Subtree ids are assigned only for routing loads.

    A terminal contributes its inflow, defaulting to one. With
    ``power_per_inflow`` given, it contributes its declared ``'power'``,
    defaulting to its inflow times ``power_per_inflow``, and the target becomes
    ``'load_nominal'``.
    A clone contributes zero. Internal nodes retain any existing target value,
    as needed by callers that clear only part of the graph, such as
    :func:`as_hooked_to_nearest`. Leaves restart from their own contribution.

    Returns:
      Number of terminals reached. Callers use this count to check traversal
      coverage independently of the terminals' inflow.
    """
    target = 'load' if power_per_inflow is None else 'load_nominal'
    i = 0
    terminals = 0
    while i < len(queue):
        node, parent, _, _, subtree = queue[i]
        i += 1
        nodeD = _node[node]
        if power_per_inflow is None:
            nodeD['subtree'] = subtree
        stop = len(queue)
        for nbr, edgeD in _adj[node].items():
            # a load=0 link is a ring zero-load link: never traverse across it
            if nbr == parent or edgeD.get('load') == 0:
                continue
            if nbr in visited:
                raise ValueError(f'node {nbr} reached twice: not a tree below {node}')
            visited.add(nbr)
            queue.append((nbr, node, edgeD, nodeD, subtree))
        if node < T:
            default = (
                nodeD.get('inflow', 1)
                if power_per_inflow is None
                else nodeD.get('power', nodeD.get('inflow', 1) * power_per_inflow)
            )
            terminals += 1
        else:
            default = 0
        nodeD[target] = default if len(queue) == stop else nodeD.get(target, default)
    return terminals


def _bfs_loads_unwind(_node, queue, target='load') -> None:
    """Accumulate the loads of the traversal recorded in ``queue``.

    Reversed BFS order visits every node after all of its descendants, so each
    node's load is complete before it is added to its parent's.
    """
    for node, parent, edgeD, parentD, _ in reversed(queue):
        load = _node[node][target]
        # the child sources the current, the parent sinks it (towards the root)
        edgeD[target] = load
        if target == 'load':
            edgeD['reverse'] = node < parent
        parentD[target] += load


def bfs_subtree_loads(G, parent, children, subtree, visited=None):
    """Descend the subtree, updating edge and node attributes.

    Meant to be called by :func:`calcload`, but can be used independently (e.g.
    from PathFinder). Nodes must not have a ``'load'`` attribute.

    Args:
      G: graph to traverse.
      parent: node the traversal descends from.
      children: nodes of ``G`` to descend into.
      subtree: subtree id to assign to every node visited.
      visited: nodes already claimed by this traversal; pass one set across
        several calls to keep them from claiming a node twice. A fresh set is
        used when omitted.

    Returns:
      Load of ``parent`` after accumulating its descendants' loads, including
      its own inflow if it is a terminal.

    Raises:
      ValueError: a node is reached twice, so the traversal is not descending a
        tree -- ``G`` holds a cycle, or two roots reach the same node.
    """
    T = G.graph['T']
    if visited is None:
        visited = {parent}
    _adj, _node = G._adj, G._node
    nodeD = _node[parent]
    # Terminals contribute their declared inflow, defaulting to one.
    # Roots and clones contribute zero.
    default = nodeD.get('inflow', 1) if 0 <= parent < T else 0
    if not children:
        nodeD['load'] = default
        return default
    nodeD['load'] = nodeD.get('load', default)
    adjP = _adj[parent]
    queue = []
    for child in children:
        if child in visited:
            raise ValueError(f'node {child} reached twice: not a tree below {parent}')
        visited.add(child)
        queue.append((child, parent, adjP[child], nodeD, subtree))
    _bfs_loads_walk(_adj, _node, T, visited, queue)
    _bfs_loads_unwind(_node, queue)
    return nodeD['load']


def split_rings_and_calc_loads(S: nx.Graph, A: nx.Graph) -> None:
    """Close path-form ring arms into canonical rings and compute their loads.

    Only the ringed builders (HGS, LKH and the ``method='ringed'`` constructor)
    call this, on a solution ``S`` that is still a set of simple
    ``root → … → root`` paths missing their zero-load links. Each path is walked
    and closed into a canonical ring (see :func:`_add_ring_to_S`), using ``A`` to
    pick the longer zero-load link on odd-length rings; a tail already touching
    a root bridges two roots ``(r1, r2)``. Every ring receives exactly one
    zero-load link (``load=0``, no current flows through it), and each node's
    subtree id and load, the edges' loads, and the graph's ``max_load`` /
    ``has_loads`` / root loads are set.

    All ringed solvers must call this before returning a solution, so that every
    ringed ``S`` carries exactly one ``load=0`` link per ring.
    """
    # Ring construction: S is path-form (no zero-load links yet). Walk each root's
    # single-feeder path to its tail, then close it into a canonical ring; a tail
    # already touching a root bridges two roots (r1, r2).
    R = S.graph['R']
    paths: list[tuple[tuple[int, int], list[int]]] = []
    seen: set[int] = set()  # first terminal of each ring already walked
    for root in range(-R, 0):
        for gate in S[root]:
            if gate in seen:
                # bridging ring: already walked from its other subroot's root
                continue
            ordered = [gate]
            back, fwd = root, gate
            while True:
                nbrs = [n for n in S[fwd] if n != back]
                if not nbrs:
                    break
                (nxt,) = nbrs  # ValueError here means S has a branching subtree
                if nxt < 0:
                    break
                ordered.append(nxt)
                back, fwd = fwd, nxt
            seen.update(ordered)
            tn = ordered[-1]
            end_roots = [
                r
                for r in range(-R, 0)
                if r in S[tn] and (len(ordered) > 1 or r != root)
            ]
            end_root = end_roots[0] if end_roots else root
            paths.append(((root, end_root), ordered))
    S.remove_edges_from(list(S.edges))
    max_load = 0
    for subtree_id, (roots, ordered) in enumerate(paths):
        _add_ring_to_S(S, roots, ordered, subtree_id, A)
        max_load = max(max_load, math.ceil(len(ordered) / 2))
    for root in range(-R, 0):
        # a load=0 feeder carries no current, so it adds nothing to its root
        # (the zero-load link of a bridging stub is a feeder, not an interior link)
        S.nodes[root]['load'] = sum(
            S.nodes[n]['load'] for n in S[root] if S[root][n]['load'] != 0
        )
    S.graph['max_load'] = max_load
    S.graph['has_loads'] = True


def calcload(G: nx.Graph, *, nominal: bool = False) -> None:
    """Calculate link loads and update edge and node attributes of ``G``.

    ``G`` must already be in final form (a forest, or a ring-form graph whose
    ``load=0`` zero-load links are present). A breadth-first traversal of each root's
    subtree propagates the loads, treating ``load=0`` links (ring zero-load links) as
    breaks. Each node's subtree id and outgoing load land on its ``'subtree'`` /
    ``'load'`` attributes. Edge loads, root loads, and the graph's ``'max_load'``
    and ``'has_loads'`` attributes are set.

    With ``nominal``, only ``'load_nominal'`` is updated on nodes and edges;
    integer loads, subtree ids, edge directions, and graph metadata are retained.
    Ring breaks are always identified by ``'load' == 0``.

    The traversal must reach all ``T`` terminals, regardless of their contribution.
    Ring construction — closing path-form arms into rings — lives in
    :func:`split_rings_and_calc_loads`, which the ringed builders call instead.

    Args:
        G: Forest or ring-form graph to update.
        nominal: Accumulate declared terminal ``'power'`` into ``'load_nominal'``
            instead of integer inflow into ``'load'``. Terminals without a
            declaration contribute their inflow times ``'power_per_inflow'``.
            Only worth the traversal when quantization is inexact: exact
            quantization makes nominal loads ``'load'`` times ``'power_per_inflow'``.
    """
    R, T = (G.graph[k] for k in 'RT')
    power_per_inflow = Fraction(G.graph.get('power_per_inflow', 1)) if nominal else None
    target = 'load_nominal' if nominal else 'load'
    # the raw dicts: indexing G[u][v] and G.nodes[n] instead would rebuild a
    # view object on every access, which dominates the cost of this traversal
    # pyrefly: ignore[missing-attribute]
    _adj, _node = G._adj, G._node
    for data in _node.values():
        data.pop(target, None)

    # one set across every root: a node claimed by two roots is reported too
    visited = set(range(-R, 0))
    queue = []
    subroots = []
    subtree = 0
    for root in range(-R, 0):
        rootD = _node[root]
        rootD[target] = 0
        for subroot, edgeD in _adj[root].items():
            # A load=0 feeder (degenerate multi-root ring zero-load link) carries
            # no load.
            if edgeD.get('load') == 0:
                continue
            if subroot in visited:
                raise ValueError(
                    f'node {subroot} reached twice: not a tree below {root}'
                )
            visited.add(subroot)
            queue.append((subroot, root, edgeD, rootD, subtree))
            subroots.append(subroot)
            subtree += 1
    terminals = _bfs_loads_walk(_adj, _node, T, visited, queue, power_per_inflow)
    _bfs_loads_unwind(_node, queue, target)

    max_load = max((_node[subroot][target] for subroot in subroots), default=0)
    if len(_node) > T + R:
        # Clones inside a routed ring's open cable are separated from both arms by
        # load=0 segments. They intentionally carry no current and are therefore
        # not reached by the root traversals above.
        for node, nodeD in _node.items():
            if (
                node >= T
                and target not in nodeD
                and all(edgeD.get('load') == 0 for edgeD in _adj[node].values())
            ):
                nodeD[target] = 0
    if terminals != T:
        raise ValueError(f'root traversals reached {terminals} terminals, not T = {T}')
    if nominal:
        for _, _, edgeD in G.edges(data=True):
            if edgeD.get('load') == 0:
                edgeD[target] = 0
    else:
        G.graph['has_loads'] = True
        G.graph['max_load'] = max_load


def _ring_split_position(ordered: list[int], A: nx.Graph | None = None) -> int:
    """Choose the balanced position between a ring's two arms."""
    n = len(ordered)
    m, mod = divmod(n, 2)
    m += mod
    if mod and n > 1 and A is not None:
        rev, center, fwd = ordered[m - 2], ordered[m - 1], ordered[m]
        rev_len = A[rev][center]['length'] if A.has_edge(rev, center) else 0
        fwd_len = A[center][fwd]['length'] if A.has_edge(center, fwd) else 0
        if rev_len > fwd_len:
            m -= 1
    return m


def _add_ring_to_S(
    S: nx.Graph,
    roots: tuple[int, int],
    ordered: list[int],
    subtree: int,
    A: nx.Graph | None = None,
) -> None:
    """Add a single ring to topology graph ``S`` in canonical form.

    A ring is the union of two radial arms, fed by ``r1`` and ``r2`` and joined
    at their tail ends; it bridges two substations when ``r1 != r2``. ``ordered``
    is the terminal sequence ``[t1, ..., tn]`` walked along the ring, so that
    ``t1`` and ``tn`` are the feeder-connected terminals. Both feeders
    ``(r1, t1)`` and ``(r2, tn)`` are real, load-bearing cables; the ring's single
    zero-load link is the edge at the load midpoint, marked by ``load=0`` (a real
    cable, no current flows through it).

    Arm 1 (the ``t1`` side) gets ``m = ceil(n / 2)`` terminals, so each arm holds at
    most ``ceil(n / 2)`` — i.e. half of the doubled ring capacity. When the ring has
    an even number of nodes (odd ``n``), the middle terminal has two candidate split
    edges yielding balanced arms; if ``A`` is provided, the longer of the two is
    chosen as the zero-load link.

    Node ``'load'``/``'subtree'`` and edge ``'load'``/``'reverse'`` are all set
    here; the caller is responsible for the root node's aggregate load.

    Args:
      S: topology graph to add the ring to (modified in place).
      roots: the pair ``(r1, r2)`` of (negative) root node ids, equal when both
        feeders share one root.
      ordered: terminal sequence ``[t1, ..., tn]`` along the ring.
      subtree: subtree id to assign to every node of the ring (both arms).
      A: optional available-links graph, used to pick the longer split edge on
        odd-node rings.
    """
    # the builder declares the shape it establishes
    S.graph['topology'] = Topology.RINGED
    r1, r2 = roots
    n = len(ordered)
    if n == 1:
        # Degenerate ring: a single terminal has feeder(s) on it.
        S.add_node(ordered[0], load=1, subtree=subtree)
        S.add_edge(r1, ordered[0], load=1, reverse=False)
        if r1 != r2:
            S.add_edge(r2, ordered[0], load=0, reverse=False)
        return
    m = _ring_split_position(ordered, A)
    # Node loads: arm 1 nodes ordered[0..m-1] carry m..1; arm 2 nodes
    # ordered[m..n-1] carry 1..(n - m) toward their own feeder.
    for i, t in enumerate(ordered):
        S.add_node(t, load=(m - i if i < m else i - m + 1), subtree=subtree)
    # Two feeders (both real cables) and the interior edges. A feeder sinks into
    # its root, whose id is negative, so it is never reversed.
    S.add_edge(r1, ordered[0], load=m, reverse=False)
    S.add_edge(r2, ordered[-1], load=n - m, reverse=False)
    for i in range(n - 1):
        u, v = ordered[i], ordered[i + 1]
        if i == m - 1:
            # zero-load link of the ring: real cable, no current (marked by load=0),
            # so it has no flow direction to encode
            S.add_edge(u, v, load=0, reverse=False)
        else:
            load = m - 1 - i if i < m else i - m + 1
            # current flows towards the arm's feeder, i.e. towards the heavier end
            u_lighter = S.nodes[u]['load'] < S.nodes[v]['load']
            source, sink = (u, v) if u_lighter else (v, u)
            S.add_edge(u, v, load=load, reverse=source < sink)
