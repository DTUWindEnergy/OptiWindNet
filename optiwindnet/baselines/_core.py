# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Shared internal helpers for baseline VRP solvers."""

import logging
from collections import defaultdict
from collections.abc import Sequence

import networkx as nx
import numpy as np
from numpy.typing import DTypeLike
from scipy.spatial.distance import pdist, squareform

from ..converting import linkbits_from_S
from ..identity import (
    _LINKSET_ID,
    _invalidate_canonical_linkset,
    complete_linkset_id,
    topology_id,
)

_lggr = logging.getLogger(__name__)
_warn = _lggr.warning


def clamp_vehicles_to_min(vehicles: int, vehicles_min: int, capacity: int) -> int:
    """Warn about and clamp a vehicles request below the feasible minimum."""
    if vehicles < vehicles_min:
        _warn(
            'Vehicles (feeders) number (%d) too low for feasibility '
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


def scaled_length_block(
    A: nx.Graph,
    terminals: Sequence[int],
    *,
    scale: float,
    complete: bool,
    absent: float,
    dtype: DTypeLike,
) -> tuple[np.ndarray, float, float]:
    """Build a scaled length matrix in the order given by ``terminals``.

    The HGS-CVRP and LKH-3 wrappers share this block. Each wrapper handles its
    own depot placement, slack nodes, and missing-link penalties.

    With ``complete=False``, pairs absent from ``A`` retain ``absent``.
    With ``complete=True``, initialize all pairs with their Euclidean distances,
    then overwrite entries for links in ``A`` with their stored lengths, which
    may include routes around obstacles. Round scaled lengths for integer
    ``dtype`` values.

    Returns:
        ``(block, fill_max, edge_max)`` containing the matrix and maximum scaled
        lengths from the Euclidean distances and stored links, respectively.
        The maxima are unrounded and default to 0.0 when no lengths contribute.
        Callers use them to set penalties and check for overflow.
    """
    integral = np.issubdtype(np.dtype(dtype), np.integer)
    fill_max = 0.0
    if complete:
        condensed = pdist(A.graph['VertexC'][terminals]) * scale
        fill_max = float(condensed.max(initial=0.0))
        if integral:
            np.round(condensed, out=condensed)
        # Convert the condensed array before expanding it to save memory.
        block = squareform(condensed.astype(dtype, copy=False))
    else:
        block = np.full((len(terminals), len(terminals)), absent, dtype=dtype)
    i_from_n = {n: i for i, n in enumerate(terminals)}
    edge_max = 0.0
    for u, v, length in A.edges(data='length'):
        iu = i_from_n.get(u)
        iv = i_from_n.get(v)
        if iu is None or iv is None:
            continue
        scaled = length * scale
        edge_max = max(edge_max, scaled)
        block[iu, iv] = block[iv, iu] = round(scaled) if integral else scaled
    return block, fill_max, edge_max


def linkset_identity(S: nx.Graph, A: nx.Graph, *, complete: bool) -> dict[str, object]:
    """Link-bit identity of solution ``S`` over the link set it was solved on.

    That set is ``A``'s links, or the complete terminal graph over ``A``'s
    terminals when the solve was given ``complete`` -- the two are told apart by
    the ``'_linkset_id'`` returned. The complete set is never built: its bit
    positions are arithmetic and its id is hashed a block at a time.

    Neither solver can be told which links exist: both price a link that ``A``
    lacks at a big-M and let the objective discourage it (see the specs written
    in :func:`~optiwindnet.baselines.lkh._do_lkh`). A solve with no feasible
    solution inside ``A`` therefore spends the big-M, and the resulting link has
    no bit position. The identity is dropped in that case -- with a warning, and
    rather than recording bits over a link set that does not contain ``S``. A
    ``complete`` solve has a position for every terminal pair, so it never is.

    Returns:
        The ``'_linkbits'``, ``'_topology_id'`` and ``'_linkset_id'`` graph
        attributes, or nothing at all if ``S`` left ``A``'s link set.
    """
    try:
        linkbits = linkbits_from_S(A, S, complete=complete)
    except ValueError as exc:
        _warn('Solution left the available-links set, so it is not identified: %s', exc)
        return {}
    return {
        '_linkbits': linkbits,
        '_topology_id': topology_id(linkbits),
        '_linkset_id': complete_linkset_id(A.graph['R'], A.graph['T'])
        if complete
        else A.graph[_LINKSET_ID],
    }


def remove_offending_crossings(A, diagonals, crossings):
    """Remove edges from ``A`` (and ``diagonals``) responsible for the given crossings.

    Each entry in ``crossings`` is a pair ``(uv, st)`` of crossing edges. Edges with
    more crossings are removed first. When an edge ``uv`` crosses a single
    longer edge ``st``, ``st`` is removed instead of ``uv`` (preferring to keep
    the shorter alternative). ``A`` and ``diagonals`` are mutated in place.
    """
    _invalidate_canonical_linkset(A)
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
