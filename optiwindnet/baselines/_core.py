# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Shared internal helpers for baseline VRP solvers."""

import logging
import math
from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction

import networkx as nx
import numpy as np
from numpy.typing import DTypeLike
from scipy.spatial.distance import pdist, squareform

from ..clustering import ClusterPlan
from ..constraints import NetworkBounds, NetworkConstraints
from ..converting import linkbits_from_S
from ..identity import _LINKSET_ID, _invalidate_canonical_linkset, topology_id
from ..types import FeederRoute, OWNWarmupFailed, Topology

_lggr = logging.getLogger(__name__)
_warn = _lggr.warning


def resolve_bounds(
    solver: str,
    constraints: NetworkConstraints | None,
    *,
    T: int,
    R: int,
    capacity: int,
    inflow_total: int,
    power_per_inflow: Fraction | None,
    link_length_ub: float | None = None,
) -> NetworkBounds:
    """Resolve the bounds of a VRP solve from ``constraints``.

    Without ``constraints``, those of ``NetworkConstraints(topology='radial')``
    apply.

    Raises:
        ValueError: the bounds do not suit the instance.
        NotImplementedError: the bounds ask for straight feeders.
    """
    if constraints is None:
        constraints = NetworkConstraints(topology=Topology.RADIAL)
    bounds = constraints._resolve(
        T=T,
        R=R,
        capacity=capacity,
        inflow_total=inflow_total,
        power_per_inflow=power_per_inflow,
        link_length_ub=link_length_ub,
    )
    if bounds.feeder_route is FeederRoute.STRAIGHT:
        raise NotImplementedError(f'{solver}() does not keep feeder routes straight.')
    return bounds


def root_fleet_(bounds: NetworkBounds, plan: ClusterPlan) -> list[int | None]:
    """Fleet of each unpinned root: its budget, or its own cap if the farm has none."""
    if bounds.feeders_ub is not None:
        return list(plan.budget_)
    per_group = bounds.feeders_per_group
    return [
        None if hi is None else hi // per_group
        for hi in bounds.root_feeders_ub or (None,) * len(plan.budget_)
    ]


def root_details(bounds: NetworkBounds) -> dict:
    """Per-root bounds to record in a solution's ``solver_details``."""
    return {
        name: value
        for name in ('root_feeders_lb', 'root_feeders_ub', 'root_load_ub')
        if (value := getattr(bounds, name)) is not None
    }


def routes_from_warmstart(warmstart: nx.Graph, R: int, T: int) -> list[list[list[int]]]:
    """Walk the routes of a warm-start solution topology, grouped by root.

    Each route lists the terminals of one feeder's subtree (radial) or of one
    ring, in depth-first order from its gate: a path comes out from the gate to
    its tail and a ring from one gate around to the other.

    Raises:
        OWNWarmupFailed: ``warmstart`` is not a topology over the same ``R`` roots and
            ``T`` terminals, leaves terminals out of the routes, or has a route
            that reaches two roots.
    """
    if (warmstart.graph.get('R'), warmstart.graph.get('T')) != (R, T):
        raise OWNWarmupFailed(
            f'Warm start has R={warmstart.graph.get("R")}, '
            f'T={warmstart.graph.get("T")}, but the problem has R={R}, T={T}.'
        )
    alien = [n for n in warmstart.nodes if not -R <= n < T]
    if alien:
        raise OWNWarmupFailed(
            f'Warm start has nodes {alien[:5]} that are neither terminals nor roots '
            '(pass a solution topology S, not a routeset G).'
        )
    seen = set()
    routes_ = []
    for r in range(-R, 0):
        routes = []
        for gate in warmstart.neighbors(r):
            if gate < 0:
                raise OWNWarmupFailed(f'Warm start links root {r} to root {gate}.')
            if gate in seen:
                # the other gate of a ring already walked
                continue
            seen.add(gate)
            route = []
            stack = [gate]
            while stack:
                n = stack.pop()
                route.append(n)
                for nbr in warmstart.neighbors(n):
                    if nbr < 0:
                        if nbr != r:
                            raise OWNWarmupFailed(
                                f'Warm start route from root {r} reaches root '
                                f'{nbr} at terminal {n}.'
                            )
                    elif nbr not in seen:
                        seen.add(nbr)
                        stack.append(nbr)
            routes.append(route)
        routes_.append(routes)
    if len(seen) != T:
        missing = sorted(set(range(T)) - seen)
        raise OWNWarmupFailed(
            f'Warm start leaves {len(missing)} terminals out of its routes '
            f'(e.g. {missing[:5]}).'
        )
    return routes_


def balanced_capacity(
    T: int, capacity: int, vehicles: int | None = None
) -> tuple[int, int, int]:
    """Derive the slack-node parameters for a balanced solve of ``T`` terminals.

    Slack nodes are depot clones that consume one unit of capacity each, so that
    every route comes out exactly full (i.e. loads are balanced). They are only
    reachable from the depot, hence a route may hold at most one of them.

    Using the requested ``capacity`` directly would ask for more slack nodes than
    there are routes whenever ``T < vehicles * (capacity - 1)``. Shrinking the
    capacity to the smallest value that still fits ``T`` in ``vehicles`` routes
    both restores ``num_slack < vehicles`` and makes the loads as even as
    possible. The vehicle count is unaffected and the effective capacity never
    exceeds the requested one, so the solution remains valid for ``capacity``.

    Since the total demand ``T + num_slack`` equals ``vehicles * capacity_effective``,
    every route is exactly full and none can be left empty: the feeder count comes
    out exactly ``vehicles``. Passing ``vehicles`` explicitly (any value from
    ``ceil(T / capacity)`` up to ``T``) is what pins the feeder count above its
    minimum; if ``None``, the minimum is used.

    Returns:
        ``(capacity_effective, vehicles, num_slack)``
    """
    if T == 0:
        return capacity, 0, 0
    if vehicles is None:
        vehicles = math.ceil(T / capacity)
    capacity_effective = math.ceil(T / vehicles)
    num_slack = vehicles * capacity_effective - T
    return capacity_effective, vehicles, num_slack


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
    link_length_ub: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Build a scaled length matrix in the order given by ``terminals``.

    The HGS-CVRP and LKH-3 wrappers share this block. Each wrapper handles its
    own depot placement, slack nodes, and missing-link penalties.

    With ``complete=False``, pairs absent from ``A`` retain ``absent``.
    With ``complete=True``, initialize all pairs with their Euclidean distances,
    then overwrite entries for links in ``A`` with their stored lengths, which
    may include routes around obstacles. Round scaled lengths for integer
    ``dtype`` values. Pairs longer than ``link_length_ub`` (unscaled, compared
    before rounding) retain ``absent``.

    Returns:
        ``(block, fill_max, edge_max)`` containing the matrix and maximum scaled
        lengths from the Euclidean distances and stored links, respectively.
        The maxima are unrounded and default to 0.0 when no lengths contribute.
        Callers use them to set penalties and check for overflow.
    """
    integral = np.issubdtype(np.dtype(dtype), np.integer)
    fill_max = 0.0
    if complete:
        condensed = pdist(A.graph['VertexC'][terminals])
        over = None if link_length_ub is None else condensed > link_length_ub
        condensed *= scale
        fill_max = float(condensed.max(initial=0.0))
        if integral:
            np.round(condensed, out=condensed)
        if over is not None:
            condensed[over] = absent
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
        if link_length_ub is not None and length > link_length_ub:
            block[iu, iv] = block[iv, iu] = absent
            continue
        scaled = length * scale
        edge_max = max(edge_max, scaled)
        block[iu, iv] = block[iv, iu] = round(scaled) if integral else scaled
    return block, fill_max, edge_max


def linkset_identity(S: nx.Graph, A: nx.Graph) -> dict[str, object]:
    """Return the identity of solution ``S`` relative to ``A``'s available links.

    The wrappers assign large costs to missing links instead of forbidding
    them, so either solver may select a link outside ``A``. Complete solves
    can also use links outside ``A``. Such links have no position in ``A``'s
    bit vector; log a warning and return an empty mapping in that case.

    A complete solve that uses only links in ``A`` retains its identity,
    allowing topology comparisons over the same available-links set.

    Returns:
        A mapping with ``'_linkbits'``, ``'_topology_id'`` and ``'_linkset_id'``,
        or an empty mapping if the solution cannot be encoded over ``A``.
    """
    try:
        linkbits = linkbits_from_S(A, S)
    except (KeyError, ValueError) as exc:
        _warn('Solution left the available-links set, so it is not identified: %s', exc)
        return {}
    return {
        '_linkbits': linkbits,
        '_topology_id': topology_id(linkbits),
        '_linkset_id': A.graph[_LINKSET_ID],
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
