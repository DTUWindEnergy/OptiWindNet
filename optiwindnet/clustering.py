# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import combinations_with_replacement

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from .constraints import NetworkBounds

__all__ = ('clusterize',)

# Beyond this many candidate feeder budgets, give up on enumerating them and
# move the budget one feeder at a time. Only reachable for R > 4.
_MAX_BUDGETS = 64

_Assignment = tuple[list[set[int]], int, float]


@dataclass(frozen=True, slots=True)
class ClusterPlan:
    """Terminals of each root, and the routes they are solved with.

    Attributes:
      cluster_: one set of terminals per root, in root order (``-R`` to ``-1``).
      budget_: number of routes (subtrees or rings) each root may use.
      pinned_: whether a root must use exactly its budget of routes.
    """

    cluster_: list[set[int]]
    budget_: tuple[int, ...]
    pinned_: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class _Limits:
    """Per-root limits of a clustering, in routes and terminals."""

    capacity: int
    lo_: tuple[int, ...]
    hi_: tuple[int, ...]
    load_cap_: tuple[int, ...]
    pinned_: tuple[bool, ...]
    min_route: int
    balance: tuple[int, int] | None

    def window(self, c: int, routes: int) -> tuple[int, int]:
        """Range of the number of terminals root ``c`` may take with ``routes``."""
        hi = min(routes * self.capacity, self.load_cap_[c])
        if self.balance is not None:
            load_lb, load_ub = self.balance
            return routes * max(load_lb, self.min_route), min(hi, routes * load_ub)
        return (routes * self.min_route if self.pinned_[c] else 0), hi


def _assign(
    budget: tuple[int, ...], limits: _Limits, d2roots: np.ndarray, T: int
) -> _Assignment:
    """Deal the terminals among the roots, at minimum total distance.

    Root ``c`` takes between the bounds of ``limits.window(c, budget[c])``
    terminals. One column per terminal slot makes this a rectangular assignment
    problem, solved exactly. Slots below a root's minimum carry a bonus larger
    than any total distance, so that they are all filled whenever the slots
    allow it.

    With ``sum(budget) == ceil(T/capacity)`` and every terminal dealt, the result
    cannot waste a feeder::

        ceil(T/capacity) <= Σ_c ceil(T_c/capacity) <= Σ_c budget[c] == ceil(T/capacity)

    Returns:
        One set of terminals per root, the number of terminals left out plus the
        number of minimum slots left empty, and the total terminal-to-root
        distance.
    """
    window_ = [limits.window(c, routes) for c, routes in enumerate(budget)]
    # a root can never take more than every terminal, however big its allowance
    slot_ = [c for c, (_, hi) in enumerate(window_) for _ in range(min(hi, T))]
    cost = d2roots[np.ix_(range(T), slot_)]
    if any(lo for lo, _ in window_):
        bonus = 1.0 + T * float(cost.max(initial=0.0))
        mandatory = np.array([i < lo for lo, hi in window_ for i in range(min(hi, T))])
        rows, cols = linear_sum_assignment(cost - bonus * mandatory)
    else:
        rows, cols = linear_sum_assignment(cost)

    cluster_: list[set[int]] = [set() for _ in budget]
    for row, col in zip(rows.tolist(), cols.tolist()):
        cluster_[slot_[col]].add(row)
    unfilled = sum(
        max(0, lo - len(cluster)) for (lo, _), cluster in zip(window_, cluster_)
    )
    return cluster_, T - len(rows) + unfilled, cost[rows, cols].sum().item()


def _budgets(
    base: tuple[int, ...], delta: int, limits: _Limits
) -> Iterator[tuple[int, ...]]:
    """Every way of adding ``delta`` routes to ``base`` (removing, if negative),
    keeping each root's budget within its limits.
    """
    step = 1 if delta > 0 else -1
    for combo in combinations_with_replacement(range(len(base)), abs(delta)):
        budget = list(base)
        for c in combo:
            budget[c] += step
        if all(lo <= b <= hi for b, lo, hi in zip(budget, limits.lo_, limits.hi_)):
            yield tuple(budget)


def _best_budget(
    base: tuple[int, ...],
    delta: int,
    limits: _Limits,
    assign: Callable[[tuple[int, ...]], _Assignment],
) -> tuple[tuple[int, ...], _Assignment] | None:
    """Assign the terminals by the best budget ``delta`` routes away from ``base``.

    Budgets are ranked by the terminals they fail to place, then by total
    distance. They are enumerated if few, otherwise built one route at a time.

    Returns:
        The best budget and its assignment, or ``None`` if no budget is within
        the limits.
    """

    def rank(item):
        return item[1][1:]

    num_steps = abs(delta)
    if math.comb(len(base) + num_steps - 1, num_steps) <= _MAX_BUDGETS:
        return min(
            ((b, assign(b)) for b in _budgets(base, delta, limits)),
            key=rank,
            default=None,
        )
    best = None
    budget = base
    for _ in range(num_steps):
        best = min(
            ((b, assign(b)) for b in _budgets(budget, delta // num_steps, limits)),
            key=rank,
            default=None,
        )
        if best is None:
            return None
        budget = best[0]
    return best


def _limits(
    R: int, T: int, capacity: int, bounds: NetworkBounds | None
) -> tuple[_Limits, int, int]:
    """Translate ``bounds`` into per-root limits and the range of the route count.

    Routes are subtrees or rings, which stay on a single root here.

    Raises:
      NotImplementedError: a root's exact feeder count is odd for rings.
    """
    min_routes = math.ceil(T / capacity)
    if bounds is None:
        limits = _Limits(capacity, (0,) * R, (T,) * R, (T,) * R, (False,) * R, 1, None)
        return limits, min_routes, T
    per_group = bounds.feeders_per_group
    lo_ = bounds.root_feeders_lb or (0,) * R
    if any(lo % per_group for lo in lo_):
        raise NotImplementedError(
            'An odd number of feeders at a substation needs a ring connecting two '
            'substations, which is not available.'
        )
    lo_ = tuple(lo // per_group for lo in lo_)
    hi_ = tuple(
        T if hi is None else hi // per_group
        for hi in bounds.root_feeders_ub or (None,) * R
    )
    load_cap_ = tuple(
        T if load is None else min(load, T)
        for load in bounds.root_load_ub or (None,) * R
    )
    groups_lb, groups_ub = bounds.groups_lb, bounds.groups_ub
    # a count fixed above the minimum is only held by solving with fixed routes
    pin_all = bounds.balanced or (groups_lb == groups_ub and groups_lb > min_routes)
    pinned_ = tuple(pin_all or (lo > 0 and lo == hi) for lo, hi in zip(lo_, hi_))
    balance = (
        (bounds.group_load_lb, bounds.group_load_ub)
        if bounds.group_load_lb is not None and bounds.group_load_ub is not None
        else None
    )
    limits = _Limits(capacity, lo_, hi_, load_cap_, pinned_, per_group, balance)
    return limits, max(groups_lb, min_routes), T if groups_ub is None else groups_ub


def cluster_plan(
    A: nx.Graph, capacity: int, bounds: NetworkBounds | None = None
) -> ClusterPlan:
    """Partition the terminals of ``A`` among the roots, within ``bounds``.

    See :func:`clusterize`, which returns the clusters of this plan. Each root
    gets a budget of routes (subtrees or rings, which stay on a single root),
    within its per-root bounds, and the budgets add up to the fewest routes the
    bounds allow. A root is pinned (its routes must be exactly its budget) under
    balance, under a whole-farm count fixed above the minimum, or under an exact
    count of its own.

    Raises:
      NotImplementedError: a root's exact feeder count is odd for rings.
      ValueError: no clustering satisfies the bounds.
    """
    R, T = (A.graph[k] for k in 'RT')
    d2roots = A.graph['d2roots']
    limits, routes_lb, routes_ub = _limits(R, T, capacity, bounds)

    cluster_: list[set[int]] = [set() for _ in range(R)]
    for n, c in enumerate(d2roots[:T].argmin(axis=1).tolist()):
        cluster_[c].add(n)
    voronoi = tuple(math.ceil(len(cluster) / capacity) for cluster in cluster_)
    if (bounds is None or not (bounds.has_root_bounds or any(limits.pinned_))) and sum(
        voronoi
    ) == routes_lb:
        # the closest-root partition already wastes no route: it is optimal
        return ClusterPlan(cluster_, voronoi, limits.pinned_)

    base = tuple(
        min(max(v, lo), hi) for v, lo, hi in zip(voronoi, limits.lo_, limits.hi_)
    )

    def assign(budget):
        return _assign(budget, limits, d2roots, T)

    for num_routes in range(routes_lb, min(routes_ub, sum(limits.hi_)) + 1):
        best = _best_budget(base, num_routes - sum(base), limits, assign)
        if best is not None and not best[1][1]:
            budget, (cluster_, _, _) = best
            return ClusterPlan(cluster_, budget, limits.pinned_)
    raise ValueError('No clustering of the terminals satisfies the substation limits.')


def plan_from_clusters(
    A: nx.Graph,
    capacity: int,
    bounds: NetworkBounds | None,
    cluster_: list[set[int]],
    budget_: tuple[int, ...],
) -> ClusterPlan:
    """Check given clusters and route counts against ``bounds`` (see
    :func:`cluster_plan`).

    Raises:
      NotImplementedError: a root's exact feeder count is odd for rings.
      ValueError: the route counts or the clusters break the bounds.
    """
    R, T = (A.graph[k] for k in 'RT')
    limits, routes_lb, routes_ub = _limits(R, T, capacity, bounds)
    if not routes_lb <= sum(budget_) <= routes_ub:
        raise ValueError(
            f'{sum(budget_)} routes, outside the range of {routes_lb} to {routes_ub}.'
        )
    for c, (cluster, routes) in enumerate(zip(cluster_, budget_)):
        lo, hi = limits.window(c, routes)
        if not (limits.lo_[c] <= routes <= limits.hi_[c] and lo <= len(cluster) <= hi):
            raise ValueError(
                f'Root {c - R} has {routes} routes and {len(cluster)} terminals, '
                'outside its bounds.'
            )
    return ClusterPlan(list(cluster_), tuple(budget_), limits.pinned_)


def clusterize(
    A: nx.Graph, capacity: int, bounds: NetworkBounds | None = None
) -> list[set[int]]:
    """Partition the terminals of ``A`` into one cluster per root.

    Without ``bounds``, clustering never costs the location a feeder::

        Σ_c ceil(T_c / capacity) == ceil(T / capacity)

    Several clusters may hold a partly filled feeder, as long as their remainders
    together need no more feeders than they occupy.

    Terminals start at their closest root (distance measured in ``P_paths`` - see
    :func:`.mesh.make_planar_embedding`), which minimizes the total terminal-to-root
    distance but may waste up to ``R - 1`` feeders. The wasted feeders are then shed:
    each way of doing so is a feeder budget (few of them, the waste being at most
    ``R - 1``), and ``_assign()`` deals the terminals out exactly for each. The
    cheapest one wins.

    ``bounds`` limit the routes and the terminals of each root, and the routes of
    the whole farm. The budgets then start at the closest-root ones, clipped to
    each root's limits, and the total number of routes grows one at a time from
    the fewest the bounds allow until a budget holds every terminal. Under
    balance (see :class:`~optiwindnet.constraints.NetworkBounds`), each root
    takes between its routes times the minimum and times the maximum load of a
    route, so that the routes of every root can be balanced to loads within
    those of the whole farm.

    That distance is only a proxy for cable length -- it has each terminal reach its
    root alone, which is what the routers avoid -- so optimizing it harder does not
    pay off.

    A cluster is empty iff its budget is zero, which happens when no terminal is
    closest to that root, or when the feeders are too few to go around
    (``ceil(T / capacity) < R``), or when draining the root is simply cheaper.
    Callers must handle a root with no terminals.

    Args:
      A: available-links graph, needs graph attributes ``R``, ``T`` and ``d2roots``
      capacity: maximum number of terminals a route (subtree or ring) may serve
      bounds: count and load bounds; loads count terminals (unit inflow)

    Returns:
      One set of terminals per root, in root order (``-R`` to ``-1``).

    Raises:
      NotImplementedError: a root's exact feeder count is odd for rings.
      ValueError: no clustering satisfies the bounds.
    """
    return cluster_plan(A, capacity, bounds).cluster_
