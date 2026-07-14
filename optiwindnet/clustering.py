# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import math
from itertools import combinations_with_replacement
from operator import itemgetter

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ('clusterize',)

# Beyond this many candidate feeder budgets, give up on enumerating them and
# shed the excess feeders one at a time. Only reachable for R > 4.
_MAX_BUDGETS = 64


def _assign(
    budget: tuple[int, ...], d2roots: np.ndarray, T: int, capacity: int
) -> tuple[list[set[int]], float]:
    """Deal the terminals among the roots, at minimum total distance.

    Root ``c`` may use ``budget[c]`` feeders, so it takes at most
    ``budget[c]*capacity`` terminals. One column per terminal slot makes this a
    rectangular assignment problem (slots outnumber terminals), solved exactly.

    With ``sum(budget) == ceil(T/capacity)``, the result cannot waste a feeder::

        ceil(T/capacity) <= Σ_c ceil(T_c/capacity) <= Σ_c budget[c] == ceil(T/capacity)

    Returns:
        One set of terminals per root, and the total terminal-to-root distance.
    """
    # a root can never take more than every terminal, however big its allowance
    slot_ = [
        c for c, feeders in enumerate(budget) for _ in range(min(feeders * capacity, T))
    ]
    cost = d2roots[np.ix_(range(T), slot_)]
    rows, cols = linear_sum_assignment(cost)

    cluster_: list[set[int]] = [set() for _ in budget]
    for row, col in zip(rows.tolist(), cols.tolist()):
        cluster_[slot_[col]].add(row)
    return cluster_, cost[rows, cols].sum().item()


def _budgets(voronoi: list[int], excess: int):
    """Every way of taking ``excess`` feeders away from the Voronoi budget.

    Budgets above the Voronoi one are not considered: a wider allowance can only
    draw in terminals that are closer to another root.
    """
    R = len(voronoi)
    for combo in combinations_with_replacement(range(R), excess):
        budget = list(voronoi)
        for c in combo:
            budget[c] -= 1
        if all(feeders >= 0 for feeders in budget):
            yield tuple(budget)


def clusterize(A: nx.Graph, capacity: int) -> list[set[int]]:
    """Partition the terminals of ``A`` into one cluster per root.

    Clustering never costs the location a feeder::

        Σ_c ceil(T_c / capacity) == ceil(T / capacity)

    Several clusters may hold a partly filled feeder, as long as their remainders
    together need no more feeders than they occupy.

    Terminals start at their closest root (distance measured in ``P_paths`` - see
    :func:`.mesh.make_planar_embedding`), which minimizes the total terminal-to-root
    distance but may waste up to ``R - 1`` feeders. The wasted feeders are then shed:
    each way of doing so is a feeder budget (few of them, the waste being at most
    ``R - 1``), and :func:`_assign` deals the terminals out exactly for each. The
    cheapest one wins.

    That distance is only a proxy for cable length -- it has each terminal reach its
    root alone, which is what the routers avoid -- so optimizing it harder does not
    pay off.

    A cluster is empty iff its budget is zero, which happens when no terminal is
    closest to that root, or when the feeders are too few to go around
    (``ceil(T / capacity) < R``), or when draining the root is simply cheaper.
    Callers must handle a root with no terminals.

    Args:
      A: available-edges graph, needs graph attributes ``R``, ``T`` and ``d2roots``
      capacity: maximum number of terminals a feeder may serve

    Returns:
      One set of terminals per root, in root order (``-R`` to ``-1``).
    """
    R, T = (A.graph[k] for k in 'RT')
    d2roots = A.graph['d2roots']

    cluster_: list[set[int]] = [set() for _ in range(R)]
    for n, c in enumerate(d2roots[:T].argmin(axis=1).tolist()):
        cluster_[c].add(n)

    voronoi = [math.ceil(len(cluster) / capacity) for cluster in cluster_]
    excess = sum(voronoi) - math.ceil(T / capacity)
    if not excess:
        # the closest-root partition already wastes no feeder: it is optimal
        return cluster_

    if math.comb(R + excess - 1, excess) <= _MAX_BUDGETS:
        candidate_ = _budgets(voronoi, excess)
        return min(
            (_assign(budget, d2roots, T, capacity) for budget in candidate_),
            key=itemgetter(1),
        )[0]

    # too many roots to enumerate the budgets: shed the excess feeders one at a time
    budget = tuple(voronoi)
    for _ in range(excess):
        budget, (cluster_, _cost) = min(
            (
                (candidate, _assign(candidate, d2roots, T, capacity))
                for candidate in _budgets(list(budget), 1)
            ),
            key=lambda pair: pair[1][1],
        )
    return cluster_
