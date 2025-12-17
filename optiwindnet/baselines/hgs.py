# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import math
from multiprocessing import Pool
import random
from typing import Sequence
from collections import defaultdict

import networkx as nx
import numpy as np

from ..clustering import clusterize
from ..interarraylib import fun_fingerprint
from ..repair import repair_routeset_path
from ._core_hgs import do_hgs
from .utils import length_matrix_single_depot_from_G

_lggr = logging.getLogger(__name__)
debug, info, warn = _lggr.debug, _lggr.info, _lggr.warning


def hgs_cvrp(
    A: nx.Graph,
    *,
    capacity: float,
    time_limit: float,
    vehicles: int | None = None,
    seed: int | None = None,
    keep_log: bool = False,
    complete: bool = False,
) -> nx.Graph:
    """Solves the OCVRP using HGS-CVRP with links from `A`

    Wraps HybGenSea, which provides bindings to the HGS-CVRP library (Hybrid
    Genetic Search solver for Capacitated Vehicle Routing Problems). This
    function uses it to solve an Open-CVRP i.e., vehicles do not return to the
    depot.

    Normalization of input graph is recommended before calling this function.

    https://github.com/vidalt/HGS-CVRP#running-the-algorithm

    Args:
        A: graph with allowed edges (if it has 0 edges, use complete graph)
        capacity: maximum vehicle capacity
        time_limit: [s] solver run time limit
        vehicles: number of vehicles (if None, let HGS-CVRP decide)

    Returns:
        Solution topology S
    """
    R, T, VertexC = (A.graph[k] for k in ('R', 'T', 'VertexC'))
    if R > 1:
        raise ValueError('Use hgs_multiroot() for multiple-depot problems')

    # Solver initialization
    # class AlgorithmParameters:
    #     nbGranular: int = 20  # Granular search parameter, limits the number
    #                           # of moves in the RI local search
    #     mu: int = 25  # Minimum population size
    #     lambda_: int = 40  # Number of solutions created before reaching the
    #                        # maximum population size (i.e., generation size).
    #     nbElite: int = 4  # Number of elite individuals
    #     nbClose: int = 5  # Number of closest solutions/individuals consider-
    #                       # ed when calculating diversity contribution
    #     targetFeasible: float = 0.2  # target ratio of feasible individuals
    #                                  # between penalty updates
    #     seed: int = 0  # fixed seed
    #     nbIter: int = 20000  # max iterations without improvement
    #     timeLimit: float = 0.0  # seconds
    #     useSwapStar: bool = True

    if seed is None:
        seed = random.getrandbits(63)
    solver_options = dict(
        timeLimit=time_limit,  # seconds
        # nbIter=2000,  # max iterations without improvement (20,000)
        seed=seed,
    )
    coordinates = np.c_[VertexC[-R:].T, VertexC[:T].T]
    # data preparation
    # Distance_matrix may be provided instead of coordinates, or in addition to
    # coordinates. Distance_matrix is used for cost calculation if provided.
    # The additional coordinates will be helpful in speeding up the algorithm.
    weights, w_max = length_matrix_single_depot_from_G(A, scale=1.0, complete=complete)

    vehicles_min = math.ceil(T / capacity)
    if vehicles is not None:
        if vehicles < vehicles_min:
            print(
                f'Vehicle number ({vehicles}) too low for feasibilty '
                f'with capacity ({capacity}). Setting to {vehicles_min}.'
            )
            # set to minimum feasible vehicle number
            vehicles = vehicles_min
        feeders_above_min = vehicles - vehicles_min
    else:
        feeders_above_min = None  # unlimited

    # HGS-CVRP crashes if distance_matrix has inf values, but there must
    # be a strong incentive to choose A edges only. (5× is arbitrary)
    distance_matrix = weights.clip(max=5 * w_max)

    routes, runtime, solution_time, cost, log, algo_params = do_hgs(
        distance_matrix, coordinates, vehicles, capacity, solver_options
    )

    # create a topology graph S from the results
    S = nx.Graph(
        T=T,
        R=R,
        capacity=capacity,
        has_loads=True,
        objective=cost,
        creator='baselines.hgs',
        runtime=runtime,
        solution_time=solution_time,
        method_options=dict(
            solver_name='HGS-CVRP',
            feeders_above_min=feeders_above_min,
            complete=complete,
            fun_fingerprint=_hgs_cvrp_fun_fingerprint,
        )
        | algo_params,
        solver_details=dict(
            vehicles=vehicles,
            seed=seed,
        ),
    )
    # ensure roots are added, even if some are not connected
    S.add_nodes_from(range(-R, 0))
    if keep_log:
        S.graph['method_log'] = log

    branches = ([n - 1 for n in branch] for branch in routes)
    max_load = 0
    for subtree_id, branch in enumerate(branches):
        branch_load = len(branch)
        max_load = max(max_load, branch_load)
        loads = range(branch_load, 0, -1)
        S.add_nodes_from(
            ((n, {'load': load}) for n, load in zip(branch, loads)), subtree=subtree_id
        )
        branch_roll = [-1] + branch[:-1]
        reverses = tuple(u < v for u, v in zip(branch, branch_roll))
        edgeD = (
            {'load': load, 'reverse': reverse} for load, reverse in zip(loads, reverses)
        )
        S.add_edges_from(zip(branch_roll, branch, edgeD))
    root_load = sum(S.nodes[n]['load'] for n in S.neighbors(-1))
    S.nodes[-1]['load'] = root_load
    assert root_load == T, 'ERROR: root node load does not match T.'
    S.graph['max_load'] = max_load
    return S


_hgs_cvrp_fun_fingerprint = fun_fingerprint(hgs_cvrp)


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
    """Iterate until crossing-free solution is found (`hgs_cvrp()` wrapper).

    Each time a solution with a crossing is produced, one of the offending
    edges is removed from `A` and the solver is called again. In the same
    way as `hgs_cvrp()`, it is recommended to pass a normalized `A`.

    Args:
      *: see `hgs_cvrp()`
      max_retries: maximum number of retries to fix unrepairable crossings

    Returns:
      Solution topology S
    """
    diagonals = A.graph['diagonals']
    i = 0
    while True:
        # solve
        S = hgs_cvrp(
            A,
            capacity=capacity,
            time_limit=time_limit,
            vehicles=vehicles,
            seed=seed,
            keep_log=keep_log,
            complete=complete,
        )
        # repair
        S = repair_routeset_path(S, A)
        # TODO: accumulate solution_time throughout the iterations
        #       (makes sense to add a new field)
        crossings = S.graph.get('outstanding_crossings', [])
        if not crossings or i == max_retries:
            break
        i += 1
        # prepare A for retry
        if i == 1:
            # just copy once, on the first retry (not to change the given A)
            A = A.copy()
            diagonals = diagonals.copy()
            A.graph['diagonals'] = diagonals
        crossing_counterparts = defaultdict(list)
        for uv, st in crossings:
            # enabling the identification of a link crossing multiple links
            crossing_counterparts[uv].append(st)
            crossing_counterparts[st].append(uv)
        # sorting allows for removing first the links that have the most crossings
        for uv in sorted(
            crossing_counterparts,
            key=lambda k: len(crossing_counterparts[k]),
            reverse=True,
        ):
            counterparts = crossing_counterparts[uv]
            if counterparts:
                # when uv crosses a single link st and st is the longest, uv becomes st
                if (
                    len(counterparts) == 1
                    and A.edges[counterparts[0]]['length'] > A.edges[uv]['length']
                ):
                    st = counterparts[0]
                    counterparts = crossing_counterparts[st]
                    # st is after uv in the sorted list -> remove uv from its counterparts
                    counterparts.remove(uv)
                    uv = st
                # remove uv from the counterparts list of uv's counterparts
                for st in counterparts:
                    crossing_counterparts[st].remove(uv)
                if uv in diagonals:
                    del diagonals[uv]
                A.remove_edge(*uv)
    if i > 0:
        S.graph['retries'] = i
        if crossings:
            warn('Solution contains crossings (max_retries reached)')
    return S


def _length_matrices(
    A: nx.Graph,
    cluster_: list[set[int]],
    num_slack_: Sequence[int],
    clip_factor: float = 5.0,
) -> tuple[list, list]:
    d2roots = A.graph['d2roots']
    R = A.graph['R']
    W_ = []
    indices_ = []
    w_max = 0.0
    for r, (cluster, num_slack) in enumerate(zip(cluster_, num_slack_), start=-R):
        n_from_i = np.array([r] + sorted(cluster) + [r] * num_slack, dtype=int)
        terminal_slice = slice(1, -num_slack if num_slack else None)
        i_from_n = {n: i for i, n in enumerate(n_from_i[terminal_slice], 1)}
        # non-available edges will have infinite length
        A_clu = nx.subgraph_view(A, filter_node=lambda n: n in cluster)
        W_dim = len(cluster) + num_slack + 1
        W_clu = np.full((W_dim, W_dim), np.inf)
        for u, v, length in A_clu.edges(data='length'):
            idx = i_from_n[u], i_from_n[v]
            # terminal-terminal distances are symmetric
            W_clu[idx] = W_clu[idx[::-1]] = length
            w_max = max(w_max, length)

        # depot distances are asymmetric
        # fill the distances from depot
        W_clu[0, terminal_slice] = d2roots[n_from_i[terminal_slice], r]
        # make return to depot always free
        W_clu[:, 0] = 0.0

        if num_slack:
            # make the slack nodes only connect to all terminals and from depot
            # from slack to each terminal (same as depot to each terminal)
            W_clu[-num_slack:, terminal_slice] = W_clu[0, terminal_slice]
            # from depot to slack (free)
            W_clu[0, -num_slack:] = 0.0

        W_.append(W_clu)
        indices_.append(n_from_i)
    # only after preparing all matrices we have the actual w_max
    for W in W_:
        np.clip(W, a_min=None, a_max=clip_factor * w_max, out=W)
    return W_, indices_


def hgs_multiroot(
    A: nx.Graph,
    *,
    capacity: int,
    time_limit: float,
    balanced: bool = False,
    seed: int | None = None,
    keep_log: bool = False,
) -> nx.Graph:
    R, T = (A.graph[k] for k in 'RT')
    VertexC = A.graph['VertexC']

    # Partition location in clusters and get link lengths from A
    cluster_, num_slack_ = clusterize(A, capacity)
    W_, indices_ = _length_matrices(
        A, cluster_, num_slack_ if balanced else [0] * len(cluster_)
    )

    # HGS-CVRP parameters
    if seed is None:
        seed = random.getrandbits(63)
    solver_options = dict(
        timeLimit=time_limit,  # seconds
        # nbIter=2000,  # max iterations without improvement (20,000)
        seed=seed,
    )
    vehicles_ = [math.ceil(len(cluster) / capacity) for cluster in cluster_]

    # data preparation
    # Distance_matrix may be provided instead of coordinates, or in addition to
    # coordinates. Distance_matrix is used for cost calculation if provided.
    # The additional coordinates will be helpful in speeding up the algorithm.
    cluster_data = zip(
        W_,  # distance matrix
        [VertexC[indices].T for indices in indices_],  # coordinates
        vehicles_,  # vehicles
        [capacity] * R,  # vehicle capacity
        [solver_options] * R,  # to be **passed to `hgs.AlgorithmParameters()`
    )

    # Launch one parallel HGS-CVRP solver process per root.
    # TODO: do not assume there are more CPU cores than roots
    pool = Pool(R)
    results = pool.starmap(do_hgs, cluster_data)
    routes_, runtime_, solution_time_, cost_, log_, algo_params = zip(*results)

    if balanced:
        # remove the slack nodes from the routes
        for num_slack, routes, cluster in zip(num_slack_, routes_, cluster_):
            if num_slack != 0:
                num_nodes = len(cluster) + 1
                routes[:] = [[n for n in route if n < num_nodes] for route in routes]
    # create a topology graph S from the results
    S = nx.Graph(
        T=T,
        R=R,
        capacity=capacity,
        has_loads=True,
        objective=sum(cost_),
        creator='baselines.hgs',
        runtime=max(runtime_),
        solution_time=solution_time_,
        method_options=dict(
            solver_name='HGS-CVRP',
            complete=False,
            fun_fingerprint=_hgs_multiroot_fun_fingerprint,
        )
        | algo_params[0],
        solver_details=dict(
            vehicles=vehicles_,
            seed=seed,
        ),
    )
    # ensure roots are added, even if some are not connected
    S.add_nodes_from(range(-R, 0))
    if keep_log:
        S.graph['method_log'] = log_
    subtree_id_start = 0
    max_load = 0
    for r, (routes, indices) in enumerate(zip(routes_, indices_), start=-R):
        branches = (indices[route] for route in routes)
        for subtree_id, branch in enumerate(branches, start=subtree_id_start):
            branch_load = len(branch)
            max_load = max(max_load, branch_load)
            loads = range(branch_load, 0, -1)
            S.add_nodes_from(
                ((n, {'load': load}) for n, load in zip(branch.tolist(), loads)),
                subtree=subtree_id,
            )
            branch_roll = np.empty_like(branch)
            branch_roll[1:] = branch[:-1]
            branch_roll[0] = r
            reverses = tuple(bool(u < v) for u, v in zip(branch, branch_roll))
            edgeD = (
                {'load': load, 'reverse': reverse}
                for load, reverse in zip(loads, reverses)
            )
            S.add_edges_from(zip(branch_roll.tolist(), branch.tolist(), edgeD))
        subtree_id_start = subtree_id + 1
        root_load = sum(S.nodes[n]['load'] for n in S.neighbors(r))
        S.nodes[r]['load'] = root_load
    assert sum(S.nodes[r]['load'] for r in range(-R, 0)) == T, (
        'ERROR: root node load does not match T.'
    )
    S.graph['max_load'] = max_load
    return S


_hgs_multiroot_fun_fingerprint = fun_fingerprint(hgs_multiroot)
