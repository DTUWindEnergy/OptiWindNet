# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import math
import random
import warnings
from typing import Callable
from typing import Sequence
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import hybgensea
import networkx as nx
import numpy as np

from ..clustering import clusterize
from ..interarraylib import fun_fingerprint
from ..repair import repair_routeset_path

_lggr = logging.getLogger(__name__)
_warn = _lggr.warning


def _length_matrix(
    A: nx.Graph,
    r: int,
    num_slack: int,
    n_from_i: np.ndarray,
    *,
    clip_factor: float = 5.0,
) -> np.ndarray:
    terminal_slice = slice(1, -num_slack if num_slack else None)
    i_from_n = {n: i for i, n in enumerate(n_from_i[terminal_slice].tolist(), 1)}
    W = np.full((len(n_from_i), len(n_from_i)), np.inf)
    w_max = 0.0
    for u, v, length in A.edges(data='length'):
        if u >= 0 and v >= 0:
            idx = i_from_n[u], i_from_n[v]
            W[idx] = W[idx[::-1]] = length
            w_max = max(w_max, length)

    W[0, terminal_slice] = A.graph['d2roots'][n_from_i[terminal_slice], r]
    W[:, 0] = 0.0

    if num_slack:
        W[-num_slack:, terminal_slice] = W[0, terminal_slice]
        W[0, -num_slack:] = 0.0
    np.clip(W, a_min=None, a_max=clip_factor * w_max, out=W)
    return W


def _length_matrices(
    A: nx.Graph,
    cluster_: list[set[int]],
    num_slack_: Sequence[int],
) -> tuple[list, list]:
    R = A.graph['R']
    W_ = []
    indices_ = []
    for r, (cluster, num_slack) in enumerate(zip(cluster_, num_slack_), start=-R):
        n_from_i = np.array([r] + sorted(cluster) + [r] * num_slack, dtype=int)
        A_clu = nx.subgraph_view(A, filter_node=lambda n: n in cluster)
        W = _length_matrix(A_clu, r, num_slack, n_from_i)
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
        (max_load, next_subtree_id)
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
):
    T, VertexC = A.graph['T'], A.graph['VertexC']
    if balanced and (T % capacity):
        vehicles_min = math.ceil(T / capacity)
        num_slack = vehicles_min * capacity - T
        vehicles = vehicles_min
    else:
        num_slack = 0
    n_from_i = np.array([-1] + list(range(T)) + [-1] * num_slack, dtype=int)
    distance_matrix = _length_matrix(A, -1, num_slack, n_from_i)
    rootC = VertexC[-1:].T
    coordinates = np.hstack((rootC, VertexC[:T].T, *((rootC,) * num_slack)))

    outputs = _do_hgs(
        distance_matrix, coordinates, vehicles, capacity, hgs_options, log_callback
    )

    inputs_ = (vehicles,), (T,), (n_from_i,), (num_slack,)
    return inputs_, (outputs,)


def _solve_multi_root(A, capacity, hgs_options, vehicles, balanced, log_callback):
    R, VertexC = A.graph['R'], A.graph['VertexC']
    cluster_, num_slack_ = clusterize(A, capacity)
    W_, indices_ = _length_matrices(A, cluster_, num_slack_ if balanced else [0] * R)
    len_cluster_ = tuple(len(cluster) for cluster in cluster_)
    if not balanced and vehicles is None:
        vehicles_ = [None] * R
    else:
        vehicles_ = [math.ceil(len_cluster / capacity) for len_cluster in len_cluster_]
    cluster_data = zip(
        W_,
        [VertexC[indices].T for indices in indices_],
        vehicles_,
        [capacity] * R,
        [hgs_options] * R,
    )

    # Launch one parallel HGS-CVRP solver process per root.
    with ThreadPoolExecutor(max_workers=R) as executor:
        outputs_ = list(executor.map(lambda x: _do_hgs(*x), cluster_data))

    inputs_ = vehicles_, len_cluster_, indices_, num_slack_
    return inputs_, outputs_


def _process_results(A, keep_log, balanced, inputs_, outputs_):
    R = A.graph['R']
    routes_, runtime_, solution_time_, cost_, log_, algo_params = zip(*outputs_)
    vehicles_, len_cluster_, indices_, num_slack_ = inputs_

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
        ),
    )
    if keep_log:
        S.graph['method_log'] = log_

    S.add_nodes_from(range(-R, 0))
    subtree_id_start = 0
    max_load = 0
    for r, (routes, indices) in enumerate(zip(routes_, indices_), start=-R):
        branches = (indices[route] for route in routes)
        branch_max_load, subtree_id_start = _add_branches(
            S, branches, root=r, subtree_id_start=subtree_id_start
        )
        max_load = max(max_load, branch_max_load)
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
    seed: int | None = None,
    keep_log: bool = False,
    repair: bool = True,
    max_retries: int = 10,
    balanced: bool = False,
    log_callback: Callable | None = None,
) -> nx.Graph:
    """Solves the OCVRP using HGS-CVRP with links from `A`.

    Wraps HybGenSea, which provides bindings to the HGS-CVRP library (Hybrid
    Genetic Search solver for Capacitated Vehicle Routing Problems). This
    function uses it to solve an Open-CVRP i.e., vehicles do not return to the
    depot.

    Normalization of input graph is recommended before calling this function.

    For single-root problems, the solver runs on the full graph. For multi-root
    problems, the graph is clustered and each cluster is solved concurrently.

    For multi-root instances, the vehicles (feeders) parameter can only be left
    undefined (meaning unlimited) or set to the minimum feasible value. Attempting
    to set other values will result in a warning and the minimum being used. 

    If ``repair=True`` (the default), the solution is iteratively repaired
    until no crossings remain (or ``max_retries`` is reached). This may cause the
    actual runtime to be up to (max_retries + 1) times the given time_limit.

    Args:
        A: graph with allowed edges (if it has 0 edges, use complete graph)
        capacity: maximum vehicle capacity
        time_limit: [s] solver run time limit
        vehicles: number of vehicles (if None, let HGS-CVRP decide;
            ignored for multi-root problems)
        seed: random seed for reproducibility
        keep_log: attach solver log to the solution graph
        repair: iteratively fix crossings (default True)
        max_retries: maximum repair iterations
        balanced: balance loads across feeders (multi-root only)
        log_callback: callback to receive each log line produced by HGS-CVRP
            (only for single-root instances)

    Returns:
        Solution topology S
    """
    R = A.graph['R']
    T = A.graph['T']
    if vehicles is not None:
        vehicles_min = math.ceil(T / capacity)
        if vehicles != vehicles_min:
            if balanced:
                raise ValueError(
                    'If `balanced`=True, the solver can only use the minimum number of '
                    'vehicles (feeders) (you may just pass None).'
                )
            elif R > 1:
                _warn(
                    'For multi-root instances, the parameter vehicles (feeders) can '
                    'only be None or the minimum feasible: setting to the minimum.'
                )
        if vehicles < vehicles_min:
            _warn(
                'Vehicles (feeders) number (%d) too low for feasibilty '
                'with given capacity (%d). Setting to %d.',
                vehicles,
                capacity,
                vehicles_min,
            )
            vehicles = vehicles_min
        feeders_above_min = vehicles - vehicles_min
    else:
        feeders_above_min = None  # unlimited

    if seed is None:
        seed = random.randrange(0, 2**31)
    hgs_options = dict(
        timeLimit=time_limit,
        seed=seed,
    )

    def _solve():
        solve = _solve_single_root if R == 1 else _solve_multi_root
        results_ = solve(A, capacity, hgs_options, vehicles, balanced, log_callback)
        S = _process_results(A, keep_log, balanced, *results_)
        assert sum(S.nodes[r]['load'] for r in range(-R, 0)) == T, (
            'ERROR: root node load does not match T.'
        )
        return S

    # iterative repair loop
    diagonals = A.graph['diagonals']
    i = 0
    while repair:
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
        crossing_counterparts = defaultdict(list)
        for uv, st in crossings:
            crossing_counterparts[uv].append(st)
            crossing_counterparts[st].append(uv)
        for uv in sorted(
            crossing_counterparts,
            key=lambda k: len(crossing_counterparts[k]),
            reverse=True,
        ):
            counterparts = crossing_counterparts[uv]
            if counterparts:
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
    else:
        # repair was false, while loop skipped
        S = _solve()
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


# TODO: remove deprecated function
def iterative_hgs_cvrp(
    A: nx.Graph,
    *,
    capacity: float,
    time_limit: float,
    vehicles: int | None = None,
    seed: int | None = None,
    max_retries: int = 10,
    keep_log: bool = False,
    complete: bool = False,
) -> nx.Graph:
    """DEPRECATED: Backward-compatible alias of `hgs_cvrp()`, use it instead."""
    warnings.warn(
        '`iterative_hgs_cvrp()` is deprecated and will be removed in a future release. '
        'Use `hgs_cvrp()` instead, as it now iterates and repairs solution by default.',
        DeprecationWarning,
        stacklevel=2,
    )
    if complete:
        warnings.warn(
            'The `complete` parameter is deprecated and ignored.',
            DeprecationWarning,
            stacklevel=2,
        )
    if A.graph['R'] > 1:
        raise ValueError('Use hgs_cvrp() for multiple-root problems')
    return hgs_cvrp(
        A,
        capacity=capacity,
        time_limit=time_limit,
        vehicles=vehicles,
        seed=seed,
        keep_log=keep_log,
        repair=True,
        max_retries=max_retries,
        balanced=False,
    )


# TODO: remove deprecated function
def hgs_multiroot(
    A: nx.Graph,
    *,
    capacity: int,
    time_limit: float,
    balanced: bool = False,
    seed: int | None = None,
    keep_log: bool = False,
) -> nx.Graph:
    """DEPRECATED: Backward-compatible alias of `hgs_cvrp()`, use it instead."""
    warnings.warn(
        '`hgs_multiroot()` is deprecated and will be removed in a future release. '
        'Use `hgs_cvrp()` instead, as it now also works for multi-root instances.',
        DeprecationWarning,
        stacklevel=2,
    )
    return hgs_cvrp(
        A,
        capacity=capacity,
        time_limit=time_limit,
        vehicles=None,
        seed=seed,
        keep_log=keep_log,
        repair=False,
        balanced=balanced,
    )
