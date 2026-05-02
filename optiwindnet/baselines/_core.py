# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Shared internal helpers for baseline VRP solvers."""

from collections import defaultdict


def remove_offending_crossings(A, diagonals, crossings):
    """Remove edges from `A` (and `diagonals`) responsible for the given crossings.

    Each entry in `crossings` is a pair ``(uv, st)`` of crossing edges. Edges with
    more crossings are removed first. When an edge ``uv`` crosses a single
    longer edge ``st``, ``st`` is removed instead of ``uv`` (preferring to keep
    the shorter alternative). `A` and `diagonals` are mutated in place.
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
