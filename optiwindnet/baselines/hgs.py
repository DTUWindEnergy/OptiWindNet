# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence

import hybgensea
import networkx as nx
import numpy as np

from ..clustering import clusterize
from ..interarraylib import add_ring_to_S, fun_fingerprint
from ..repair import repair_routeset_path
from ._core import remove_offending_crossings

_lggr = logging.getLogger(__name__)
_warn = _lggr.warning


def _balanced_capacity(
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


def _length_matrix(
    A: nx.Graph,
    r: int,
    num_slack: int,
    n_from_i: np.ndarray,
    *,
    closed: bool = False,
    clip_factor: float = 5.0,
) -> np.ndarray:
    """Build the HGS-CVRP distance matrix (depot at index 0).

    By default the return leg to the depot is free (``W[:, 0] = 0``), turning the
    closed CVRP that HGS-CVRP solves into the Open-CVRP used for radial layouts.
    With ``closed=True`` the return leg costs the feeder distance, so every route
    pays for both of its feeder legs: HGS then finds closed loops (rings). Depot
    clones (slack nodes) stay at the depot, so their return leg is always free.
    """
    terminal_slice = slice(1, -num_slack if num_slack else None)
    i_from_n = {n: i for i, n in enumerate(n_from_i[terminal_slice].tolist(), 1)}
    W = np.full((len(n_from_i), len(n_from_i)), np.inf)
    w_max = 0.0
    for u, v, length in A.edges(data='length'):
        if u >= 0 and v >= 0:
            idx = i_from_n[u], i_from_n[v]
            W[idx] = W[idx[::-1]] = length
            w_max = max(w_max, length)

    d2roots = A.graph['d2roots'][n_from_i[terminal_slice], r]
    W[0, terminal_slice] = d2roots
    if closed:
        # closed CVRP (RINGED): the return leg costs the feeder distance
        W[terminal_slice, 0] = d2roots
        W[0, 0] = 0.0
    else:
        W[:, 0] = 0.0

    if num_slack:
        W[-num_slack:, terminal_slice] = W[0, terminal_slice]
        W[0, -num_slack:] = 0.0
        if closed:
            # slack nodes are depot clones: their return leg to the depot is free
            W[-num_slack:, 0] = 0.0
    np.clip(W, a_min=None, a_max=clip_factor * w_max, out=W)
    return W


def _length_matrices(
    A: nx.Graph,
    cluster_: list[set[int]],
    num_slack_: Sequence[int],
    closed: bool = False,
) -> tuple[list, list]:
    R = A.graph['R']
    W_ = []
    indices_ = []
    for r, (cluster, num_slack) in enumerate(zip(cluster_, num_slack_), start=-R):
        n_from_i = np.array([r] + sorted(cluster) + [r] * num_slack, dtype=int)
        A_clu = nx.subgraph_view(A, filter_node=lambda n: n in cluster)
        W = _length_matrix(A_clu, r, num_slack, n_from_i, closed=closed)
        W_.append(W)
        indices_.append(n_from_i)
    return W_, indices_


def _solution_time(log, objective) -> float:
    sol_repr = f'{objective:.2f}'
    for line in log.splitlines():
        if not line or line[0] == '-':
            continue
        fields = line.split(' | ')
        if len(fields) < 3:
            continue
        if fields[2] != 'NO-FEASIBLE':
            try:
                incumbent = fields[2].split(' ')[2]
            except IndexError:
                incumbent = ''
        else:
            incumbent = ''
        if incumbent == sol_repr:
            _, time = fields[1].split(' ')
            return float(time)
    # if sol_repr was not found, return total runtime
    try:
        return float(line.split(' ')[-1])
    except (IndexError, ValueError, UnboundLocalError):
        return 0.0


def _add_branches(S, branches, root, subtree_id_start):
    """Add branches to solution graph S.

    Args:
        S: solution graph (modified in place)
        branches: iterable of branches (each a list or array of node ids)
        root: root node id
        subtree_id_start: starting subtree_id for numbering

    Returns:
        ``(max_load, next_subtree_id)``
    """
    max_load = 0
    subtree_id = subtree_id_start
    for branch in branches:
        branch_load = len(branch)
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


def _add_rings(S, rings, root, subtree_id_start, A):
    """Add closed loops (rings) to solution graph S, one per CVRP tour.

    Each ring is the closed CVRP route ``root → t1 → … → tn → root``, rendered in
    canonical form by :func:`add_ring_to_S` (two real feeders joined at a single
    ``split`` open point at the load midpoint, so each arm holds at most half of
    the doubled ring capacity).

    Returns ``(max_load, next_subtree_id)``.
    """
    max_load = 0
    subtree_id = subtree_id_start
    for ring in rings:
        ordered = ring.tolist() if isinstance(ring, np.ndarray) else list(ring)
        if not ordered:
            continue
        add_ring_to_S(S, root, ordered, subtree_id, A)
        max_load = max(max_load, math.ceil(len(ordered) / 2))
        subtree_id += 1
    return max_load, subtree_id


def _do_hgs(W, coordinates, vehicles, capacity, hgs_options, log_callback=None):
    """Multithreading worker function that calls the external library"""
    n = coordinates.shape[1]
    demands = np.ones(n, dtype=np.float64)
    demands[0] = 0.0  # depot demand = 0

    num_vehicles = vehicles if vehicles is not None else -1
    result = hybgensea.solve_cvrp_dist_mtx(
        W,
        demands,
        float(capacity),
        x_coords=coordinates[0],
        y_coords=coordinates[1],
        num_vehicles=num_vehicles,
        log_callback=log_callback,
        **hgs_options,
    )

    solution_time = _solution_time(result.log, result.cost)
    return (
        result.routes,
        result.time,
        solution_time,
        result.cost,
        result.log,
        {**hybgensea.DEFAULT_ALGO_PARAMS, **hgs_options},
    )


def _solve_single_root(
    A,
    capacity,
    hgs_options,
    vehicles,
    balanced,
    log_callback,
    closed=False,
):
    T, VertexC = A.graph['T'], A.graph['VertexC']
    if balanced:
        capacity, vehicles, num_slack = _balanced_capacity(T, capacity, vehicles)
    else:
        num_slack = 0
    n_from_i = np.array([-1] + list(range(T)) + [-1] * num_slack, dtype=int)
    distance_matrix = _length_matrix(A, -1, num_slack, n_from_i, closed=closed)
    rootC = VertexC[-1:].T
    coordinates = np.hstack((rootC, VertexC[:T].T, *((rootC,) * num_slack)))

    outputs = _do_hgs(
        distance_matrix, coordinates, vehicles, capacity, hgs_options, log_callback
    )

    inputs_ = (vehicles,), (T,), (n_from_i,), (num_slack,), (capacity,)
    return inputs_, (outputs,)


def _no_terminals_output(hgs_options):
    """Stand-in for the output of a root that got no terminals.

    Such a root is never dispatched (HGS-CVRP cannot handle the resulting 1x1
    distance matrix) but keeps its place in the per-root lists, so that root ids
    stay aligned with cluster indices.
    """
    return [], 0.0, 0.0, 0.0, '', {**hybgensea.DEFAULT_ALGO_PARAMS, **hgs_options}


def _solve_multi_root(
    A, capacity, hgs_options, vehicles, balanced, log_callback, closed=False
):
    R, VertexC = A.graph['R'], A.graph['VertexC']
    cluster_ = clusterize(A, capacity)
    len_cluster_ = tuple(len(cluster) for cluster in cluster_)
    if balanced:
        # each cluster is balanced independently; clusterize() already ensures
        # that the per-cluster minimum feeder counts sum to the global minimum
        capacity_, vehicles_, num_slack_ = (
            list(values)
            for values in zip(
                *(
                    _balanced_capacity(len_cluster, capacity)
                    for len_cluster in len_cluster_
                )
            )
        )
    else:
        capacity_ = [capacity] * R
        num_slack_ = [0] * R
        if vehicles is None:
            vehicles_ = [None] * R
        else:
            vehicles_ = [
                math.ceil(len_cluster / capacity) for len_cluster in len_cluster_
            ]
    W_, indices_ = _length_matrices(A, cluster_, num_slack_, closed=closed)
    populated_ = [c for c, len_cluster in enumerate(len_cluster_) if len_cluster]
    cluster_data = [
        (W_[c], VertexC[indices_[c]].T, vehicles_[c], capacity_[c], hgs_options)
        for c in populated_
    ]

    # Launch one parallel HGS-CVRP solver process per populated root.
    with ThreadPoolExecutor(max_workers=len(populated_) or 1) as executor:
        solved_ = list(executor.map(lambda x: _do_hgs(*x), cluster_data))

    outputs_ = [_no_terminals_output(hgs_options) for _ in range(R)]
    for c, output in zip(populated_, solved_):
        outputs_[c] = output

    inputs_ = vehicles_, len_cluster_, indices_, num_slack_, capacity_
    return inputs_, outputs_


def _process_results(A, keep_log, balanced, inputs_, outputs_, ringed=False):
    R = A.graph['R']
    routes_, runtime_, solution_time_, cost_, log_, algo_params = zip(*outputs_)
    vehicles_, len_cluster_, indices_, num_slack_, capacity_ = inputs_

    if balanced:
        for num_slack, routes, len_cluster in zip(num_slack_, routes_, len_cluster_):
            # remove slack nodes from the routes
            if num_slack != 0:
                num_nodes = len_cluster + 1
                routes[:] = [[n for n in route if n < num_nodes] for route in routes]

    S = nx.Graph(
        R=R,
        T=A.graph['T'],
        objective=sum(cost_),
        runtime=max(runtime_),
        solution_time=solution_time_ if R > 1 else solution_time_[0],
        method_options=algo_params[0],
        solver_details=dict(
            vehicles=vehicles_ if R > 1 else vehicles_[0],
            **(
                dict(capacity_effective=capacity_ if R > 1 else capacity_[0])
                if balanced
                else {}
            ),
        ),
    )
    if keep_log:
        S.graph['method_log'] = log_ if R > 1 else log_[0]

    S.add_nodes_from(range(-R, 0))
    subtree_id_start = 0
    max_load = 0
    for r, (routes, indices) in enumerate(zip(routes_, indices_), start=-R):
        subtrees = (indices[route] for route in routes)
        if ringed:
            sub_max_load, subtree_id_start = _add_rings(
                S, subtrees, root=r, subtree_id_start=subtree_id_start, A=A
            )
        else:
            sub_max_load, subtree_id_start = _add_branches(
                S, subtrees, root=r, subtree_id_start=subtree_id_start
            )
        max_load = max(max_load, sub_max_load)
        root_load = sum(S.nodes[n]['load'] for n in S.neighbors(r))
        S.nodes[r]['load'] = root_load

    S.graph['max_load'] = max_load
    return S


def hgs_cvrp(
    A: nx.Graph,
    *,
    capacity: float,
    time_limit: float,
    vehicles: int | None = None,
    vehicles_exact: bool = False,
    seed: int | None = None,
    keep_log: bool = False,
    repair: bool = True,
    max_retries: int = 10,
    balanced: bool = False,
    ringed: bool = False,
    log_callback: Callable | None = None,
) -> nx.Graph:
    """Solves the O/CVRP using HGS-CVRP with links from ``A``.

    Wraps HybGenSea, which provides bindings to the HGS-CVRP library (Hybrid
    Genetic Search solver for Capacitated Vehicle Routing Problems). By default
    this function solves an Open-CVRP (vehicles do not return to the depot),
    yielding radial layouts. With ``ringed=True`` it solves the closed CVRP
    instead: every route returns to the depot, forming a ring (closed loop). The
    ring capacity is doubled internally (``2 * capacity``) so each of the ring's
    two arms holds at most ``capacity`` terminals.

    Normalization of input graph is recommended before calling this function.

    For single-root problems, the solver runs on the full graph. For multi-root
    problems, the graph is clustered and each cluster is solved concurrently.

    By default, ``vehicles`` is an upper bound on the feeder count: HGS-CVRP is
    free to use fewer, which it normally does, since a shorter solution seldom
    needs more than the minimum ``ceil(T / capacity)`` feeders. Pass
    ``vehicles_exact=True`` to pin the count to ``vehicles`` instead. This is
    only implemented together with ``balanced=True``: the slack nodes that make
    the loads balanced also make every route come out full, so no route can be
    left empty and the feeder count is necessarily ``vehicles``.

    For multi-root instances, the vehicles (feeders) parameter can only be left
    undefined (meaning unlimited) or set to the minimum feasible value. Any other
    value results in a warning and the minimum being used, or, if
    ``vehicles_exact=True``, in a ``ValueError``.

    If ``repair=True`` (the default), the solution is iteratively repaired
    until no crossings remain (or ``max_retries`` is reached). This may cause the
    actual runtime to be up to ``(max_retries + 1)`` times the given ``time_limit``.

    Args:
        A: graph with allowed edges (if it has 0 edges, use complete graph)
        capacity: maximum vehicle capacity
        time_limit: [s] solver run time limit
        vehicles: maximum number of vehicles (if None, let HGS-CVRP decide;
            clamped to the minimum for multi-root problems); the exact number of
            vehicles if ``vehicles_exact=True``
        vehicles_exact: whether ``vehicles`` is the exact feeder count instead of
            an upper bound (requires ``balanced=True``, a single root, and
            ``ceil(T / capacity) <= vehicles <= T``)
        seed: random seed for reproducibility
        keep_log: attach solver log to the solution graph
        repair: iteratively fix crossings (default True)
        max_retries: maximum repair iterations
        balanced: balance loads across feeders (per root, if multiple roots)
        log_callback: callback to receive each log line produced by HGS-CVRP
            (only for single-root instances)

    Returns:
        Solution topology S
    """
    R = A.graph['R']
    T = A.graph['T']
    # A ring holds up to 2 * capacity terminals (two arms of `capacity` each);
    # HGS solves it as a closed CVRP with the doubled vehicle capacity, while the
    # reported cable capacity stays `capacity`.
    solve_capacity = 2 * capacity if ringed else capacity
    if ringed and vehicles_exact:
        raise NotImplementedError(
            'vehicles_exact is not supported together with ringed=True.'
        )
    vehicles_min = math.ceil(T / solve_capacity)
    if vehicles_exact:
        if vehicles is None:
            raise ValueError('`vehicles_exact`=True requires `vehicles` to be set.')
        if not balanced:
            raise NotImplementedError(
                'An exact vehicles (feeders) count is only available with '
                '`balanced`=True.'
            )
        if vehicles < vehicles_min:
            raise ValueError(
                f'Vehicles (feeders) number ({vehicles}) is below the minimum '
                f'necessary ({vehicles_min}) for the given capacity ({capacity}).'
            )
        if vehicles > T:
            raise ValueError(
                f'Vehicles (feeders) number ({vehicles}) is above the number of '
                f'terminals ({T}).'
            )
        if R > 1 and vehicles != vehicles_min:
            raise ValueError(
                'For multi-root instances, an exact vehicles (feeders) count is '
                'only available at the minimum feasible value.'
            )
    elif vehicles is not None:
        if vehicles != vehicles_min:
            if balanced:
                raise ValueError(
                    'If `balanced`=True, the solver can only use the minimum number of '
                    'vehicles (feeders) (you may just pass None), unless '
                    '`vehicles_exact`=True.'
                )
            elif R > 1:
                _warn(
                    'For multi-root instances, the parameter vehicles (feeders) can '
                    'only be None or the minimum feasible: setting to the minimum.'
                )
                vehicles = vehicles_min
        if vehicles < vehicles_min:
            _warn(
                'Vehicles (feeders) number (%d) too low for feasibilty '
                'with given capacity (%d). Setting to %d.',
                vehicles,
                capacity,
                vehicles_min,
            )
            vehicles = vehicles_min
    feeders_above_min = None if vehicles is None else vehicles - vehicles_min

    if seed is None:
        seed = random.randrange(0, 2**31)
    hgs_options = dict(
        timeLimit=time_limit,
        seed=seed,
    )

    def _solve():
        solve = _solve_single_root if R == 1 else _solve_multi_root
        results_ = solve(
            A, solve_capacity, hgs_options, vehicles, balanced, log_callback, ringed
        )
        S = _process_results(A, keep_log, balanced, *results_, ringed=ringed)
        assert sum(S.nodes[r]['load'] for r in range(-R, 0)) == T, (
            'ERROR: root node load does not match T.'
        )
        if vehicles_exact:
            feeder_count = sum(S.degree[r] for r in range(-R, 0))
            assert feeder_count == vehicles, (
                f'ERROR: feeder count ({feeder_count}) does not match the exact '
                f'number requested ({vehicles}).'
            )
        return S

    # iterative repair loop
    diagonals = A.graph['diagonals']
    if R > 1:
        # needed in clustering
        A.graph['closest_root'] = -R + A.graph['d2roots'][:T].argmin(axis=1)
    crossings = []
    i = 0
    if not repair or ringed:
        # repair_routeset_path()/remove_offending_crossings() target open radial
        # branches; ring crossings are resolved downstream by PathFinder(ringed=True).
        S = _solve()
    else:
        while True:
            S = _solve()
            S = repair_routeset_path(S, A)
            crossings = S.graph.get('outstanding_crossings', [])
            if not crossings or i == max_retries:
                break
            i += 1
            if i == 1:
                A = A.copy()
                diagonals = diagonals.copy()
                A.graph['diagonals'] = diagonals
            remove_offending_crossings(A, diagonals, crossings)
    if i > 0:
        S.graph['retries'] = i
        if crossings:
            _warn('Solution contains crossings (max_retries reached)')

    S.graph.update(
        T=T,
        R=R,
        capacity=capacity,
        has_loads=True,
        creator='baselines.hgs',
        method_options=dict(
            solver_name='HGS-CVRP',
            complete=False,
            feeders_above_min=feeders_above_min,
            feeders_exact=vehicles_exact,
            fun_fingerprint=_hgs_cvrp_fun_fingerprint,
            **S.graph['method_options'],
        ),
        solver_details=dict(
            seed=seed,
            **S.graph['solver_details'],
        ),
    )
    return S


_hgs_cvrp_fun_fingerprint = fun_fingerprint(hgs_cvrp)
