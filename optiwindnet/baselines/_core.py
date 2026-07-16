# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Shared internal helpers for baseline VRP solvers."""

import logging
from collections import defaultdict

import numpy as np

_lggr = logging.getLogger(__name__)
_warn = _lggr.warning


def clamp_vehicles_to_min(vehicles: int, vehicles_min: int, capacity: int) -> int:
    """Warn about and clamp a vehicles request below the feasible minimum."""
    if vehicles < vehicles_min:
        _warn(
            'Vehicles (feeders) number (%d) too low for feasibilty '
            'with given capacity (%d). Setting to %d.',
            vehicles,
            capacity,
            vehicles_min,
        )
        vehicles = vehicles_min
    return vehicles


def add_branches_to_S(S, branches, root, subtree_id_start):
    """Add open-path routes to solution graph S in place, one subtree each.

    Args:
        S: solution graph (modified in place)
        branches: iterable of branches (each a list or array of node ids;
            empty branches are skipped)
        root: root node id
        subtree_id_start: starting subtree_id for numbering

    Returns:
        ``(max_load, next_subtree_id)``
    """
    max_load = 0
    subtree_id = subtree_id_start
    for branch in branches:
        branch_load = len(branch)
        if branch_load == 0:
            continue
        max_load = max(max_load, branch_load)
        loads = range(branch_load, 0, -1)
        branch_list = (
            branch.tolist() if isinstance(branch, np.ndarray) else list(branch)
        )
        S.add_nodes_from(
            ((n, {'load': load}) for n, load in zip(branch_list, loads)),
            subtree=subtree_id,
        )
        prev = [root] + branch_list[:-1]
        reverses = tuple(u < v for u, v in zip(branch_list, prev))
        edgeD = (
            {'load': load, 'reverse': reverse} for load, reverse in zip(loads, reverses)
        )
        S.add_edges_from(zip(prev, branch_list, edgeD))
        subtree_id += 1
    return max_load, subtree_id


def remove_offending_crossings(A, diagonals, crossings):
    """Remove edges from ``A`` (and ``diagonals``) responsible for the given crossings.

    Each entry in ``crossings`` is a pair ``(uv, st)`` of crossing edges. Edges with
    more crossings are removed first. When an edge ``uv`` crosses a single
    longer edge ``st``, ``st`` is removed instead of ``uv`` (preferring to keep
    the shorter alternative). ``A`` and ``diagonals`` are mutated in place.
    """
    crossing_counterparts = defaultdict(list)
    for uv, st in crossings:
        crossing_counterparts[uv].append(st)
        crossing_counterparts[st].append(uv)
    # sort so the most-crossed edges are removed first
    for uv in sorted(
        crossing_counterparts,
        key=lambda k: len(crossing_counterparts[k]),
        reverse=True,
    ):
        counterparts = crossing_counterparts[uv]
        if not counterparts:
            continue
        # if uv crosses a single link st and st is the longest, remove st instead
        if (
            len(counterparts) == 1
            and A.edges[counterparts[0]]['length'] > A.edges[uv]['length']
        ):
            st = counterparts[0]
            counterparts = crossing_counterparts[st]
            counterparts.remove(uv)
            uv = st
        for st in counterparts:
            crossing_counterparts[st].remove(uv)
        if uv in diagonals:
            del diagonals[uv]
        A.remove_edge(*uv)
