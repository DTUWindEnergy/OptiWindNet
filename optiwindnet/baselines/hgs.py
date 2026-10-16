# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import random
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

import hybgensea
import networkx as nx
import numpy as np

from ..clustering import cluster_plan, plan_from_clusters
from ..constraints import NetworkBounds, NetworkConstraints
from ..identity import fingerprint_function
from ..interarraylib import _links_over_limit
from ..loads import (
    _orient_tree,
    calcload,
    split_rings_and_calc_loads,
)
from ..repair import repair_routeset_path
from ..types import OWNWarmupFailed, Topology
from ._core import (
    add_branches_to_S,
    balanced_capacity,
    linkset_identity,
    remove_offending_crossings,
    root_details,
    root_fleet_,
    routes_from_warmstart,
    scaled_length_block,
    vrp_bounds,
)

_lggr = logging.getLogger(__name__)
_warn = _lggr.warning


def _length_matrix(
    A: nx.Graph,
    r: int,
    num_slack: int,
    n_from_i: np.ndarray,
    *,
    complete: bool = False,
    closed: bool = False,
    clip_factor: float = 5.0,
    link_length_ub: float | None = None,
) -> np.ndarray:
    """Build the HGS-CVRP distance matrix (depot at index 0).

    By default the return leg to the depot is free (``W[:, 0] = 0``), turning the
    closed CVRP that HGS-CVRP solves into the Open-CVRP used for radial layouts.
    With ``closed=True`` the return leg costs the feeder distance, so every route
    pays for both of its feeder legs: HGS then finds cycles (rings). Depot
    clones (slack nodes) stay at the depot, so their return leg is always free.

    With ``complete=True``, use Euclidean distances for terminal pairs
    absent from ``A``.

    HGS-CVRP requires finite entries. Clip the matrix to ``clip_factor`` times
    the largest terminal-to-terminal length, including the Euclidean fill.
    This gives missing links and unreachable slack-node entries a finite
    penalty. Links and feeders longer than ``link_length_ub`` get the same.
    """
    terminal_slice = slice(1, -num_slack if num_slack else None)
    terminals = n_from_i[terminal_slice].tolist()
    W = np.full((len(n_from_i), len(n_from_i)), np.inf)
    W[terminal_slice, terminal_slice], fill_max, edge_max = scaled_length_block(
        A, terminals, scale=1.0, complete=complete, absent=np.inf, dtype=np.float64
    )
    w_max = max(fill_max, edge_max)

    d2roots = A.graph['d2roots'][n_from_i[terminal_slice], r]
    if link_length_ub is not None:
        W[W > link_length_ub] = np.inf
        d2roots = np.where(d2roots > link_length_ub, np.inf, d2roots)
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
    complete: bool = False,
    closed: bool = False,
    link_length_ub: float | None = None,
) -> tuple[list, list]:
    R = A.graph['R']
    W_ = []
    indices_ = []
    for r, (cluster, num_slack) in enumerate(zip(cluster_, num_slack_), start=-R):
        n_from_i = np.array([r] + sorted(cluster) + [r] * num_slack, dtype=int)
        A_clu = nx.subgraph_view(A, filter_node=lambda n, cluster=cluster: n in cluster)
        W = _length_matrix(
            A_clu,
            r,
            num_slack,
            n_from_i,
            complete=complete,
            closed=closed,
            link_length_ub=link_length_ub,
        )
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


def _initial_routes(
    routes: list[list[int]], n_from_i: np.ndarray, num_slack: int, capacity: int
) -> list[list[int]]:
    """Translate one root's warm-start routes into an HGS-CVRP initial solution.

    HGS-CVRP refers to a client by its row in the distance matrix (the depot is
    row 0, see ``n_from_i``). In a balanced solve, each route is topped up to
    ``capacity`` with slack nodes, which open the route since only the depot
    reaches them.

    Raises:
        optiwindnet.types.OWNWarmupFailed: a balanced solve's slack nodes cannot
            top up exactly the warm start's routes (i.e. the route counts
            differ).
    """
    end = len(n_from_i) - num_slack
    i_from_n = {n: i for i, n in enumerate(n_from_i[1:end].tolist(), start=1)}
    initial = [[i_from_n[n] for n in route] for route in routes]
    if num_slack:
        deficits = [capacity - len(route) for route in initial]
        if sum(deficits) != num_slack:
            raise OWNWarmupFailed(
                f'Balanced solve of root {n_from_i[0]} needs '
                f'{(end - 1 + num_slack) // capacity} routes, but the warm start '
                f'has {len(initial)}.'
            )
        slack = iter(range(end, len(n_from_i)))
        initial = [
            [next(slack) for _ in range(deficit)] + route
            for deficit, route in zip(deficits, initial)
        ]
    return initial


def _do_hgs(W, coordinates, vehicles, capacity, hgs_options, log_callback=None,
            demands=None, initial_routes=None):  # fmt: skip
    """Multithreading worker function that calls the external library"""
    n = coordinates.shape[1]
    if demands is None:
        demands = np.ones(n, dtype=np.float64)
        demands[0] = 0.0  # depot demand = 0

    num_vehicles = vehicles if vehicles is not None else -1
    try:
        result = hybgensea.solve_cvrp_dist_mtx(
            W,
            demands,
            float(capacity),
            x_coords=coordinates[0],
            y_coords=coordinates[1],
            num_vehicles=num_vehicles,
            initial_solutions=None if initial_routes is None else [initial_routes],
            log_callback=log_callback,
            **hgs_options,
        )
    except ValueError as exc:
        # HybGenSea validates the initial solutions before solving
        if initial_routes is None or not str(exc).startswith('Input solution'):
            raise
        raise OWNWarmupFailed(f'HGS-CVRP rejected the warm start: {exc}') from exc

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
    complete=False,
    warm_routes_=None,
    link_length_ub=None,
    inflow_by_node=None,
):
    T, VertexC = A.graph['T'], A.graph['VertexC']
    if balanced:
        capacity, vehicles, num_slack = balanced_capacity(T, capacity, vehicles)
    else:
        num_slack = 0
    n_from_i = np.array([-1] + list(range(T)) + [-1] * num_slack, dtype=int)
    distance_matrix = _length_matrix(
        A,
        -1,
        num_slack,
        n_from_i,
        complete=complete,
        closed=closed,
        link_length_ub=link_length_ub,
    )
    rootC = VertexC[-1:].T
    coordinates = np.hstack((rootC, VertexC[:T].T, *((rootC,) * num_slack)))

    demands = None
    if inflow_by_node:
        # The depot precedes the terminals in the demand array.
        # hgs_cvrp() rejects balanced solves with nonunitary inflow.
        demands = np.ones(T + 1 + num_slack, dtype=np.float64)
        demands[0] = 0.0
        for t, inflow in inflow_by_node.items():
            demands[t + 1] = inflow

    initial_routes = (
        None
        if warm_routes_ is None
        else _initial_routes(warm_routes_[0], n_from_i, num_slack, capacity)
    )
    outputs = _do_hgs(
        distance_matrix,
        coordinates,
        vehicles,
        capacity,
        hgs_options,
        log_callback,
        demands,
        initial_routes,
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
    A,
    capacity,
    hgs_options,
    plan,
    fleet_,
    log_callback,
    closed=False,
    complete=False,
    warm_routes_=None,
    link_length_ub=None,
):
    R, VertexC = A.graph['R'], A.graph['VertexC']
    cluster_ = plan.cluster_
    len_cluster_ = tuple(len(cluster) for cluster in cluster_)
    capacity_, vehicles_, num_slack_ = [], [], []
    for len_cluster, routes, pinned, fleet in zip(
        len_cluster_, plan.budget_, plan.pinned_, fleet_
    ):
        if pinned:
            # the slack nodes fill every route: exactly `routes` of them
            capacity_c, vehicles_c, num_slack_c = balanced_capacity(
                len_cluster, capacity, routes
            )
        else:
            capacity_c, vehicles_c, num_slack_c = capacity, fleet, 0
        capacity_.append(capacity_c)
        vehicles_.append(vehicles_c)
        num_slack_.append(num_slack_c)
    W_, indices_ = _length_matrices(
        A,
        cluster_,
        num_slack_,
        complete=complete,
        closed=closed,
        link_length_ub=link_length_ub,
    )
    populated_ = [c for c, len_cluster in enumerate(len_cluster_) if len_cluster]
    cluster_data = [
        (
            W_[c],
            VertexC[indices_[c]].T,
            vehicles_[c],
            capacity_[c],
            hgs_options,
            None,
            None,
            None
            if warm_routes_ is None
            else _initial_routes(
                warm_routes_[c], indices_[c], num_slack_[c], capacity_[c]
            ),
        )
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


def _process_results(A, keep_log, pinned, inputs_, outputs_):
    R = A.graph['R']
    routes_, runtime_, solution_time_, cost_, log_, algo_params = zip(*outputs_)
    vehicles_, len_cluster_, indices_, num_slack_, capacity_ = inputs_

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
                {'capacity_effective': capacity_ if R > 1 else capacity_[0]}
                if pinned
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
        # rings, too, are built as paths here (one feeder each), so that the
        # path-based repair machinery applies; split_rings_and_calc_loads closes
        # them afterwards
        sub_max_load, subtree_id_start = add_branches_to_S(
            S, subtrees, root=r, subtree_id_start=subtree_id_start
        )
        max_load = max(max_load, sub_max_load)
        root_load = sum(S.nodes[n]['load'] for n in S.neighbors(r))
        S.nodes[r]['load'] = root_load

    S.graph['max_load'] = max_load
    return _orient_tree(S)


def hgs_cvrp(
    A: nx.Graph,
    constraints: NetworkConstraints | NetworkBounds,
    *,
    time_limit: float,
    power_rtol: float | None = None,
    seed: int | None = None,
    keep_log: bool = False,
    repair: bool = True,
    max_retries: int = 10,
    complete: bool = False,
    log_callback: Callable | None = None,
    warmstart: nx.Graph | None = None,
) -> nx.DiGraph:
    """Solves the O/CVRP using HGS-CVRP with links from ``A``.

    Wraps HybGenSea, which provides bindings to the HGS-CVRP library (Hybrid
    Genetic Search solver for Capacitated Vehicle Routing Problems). By default
    this function solves an Open-CVRP (vehicles do not return to the depot),
    yielding radial layouts. For a RINGED topology it solves the closed CVRP
    instead: every route returns to the depot, forming a ring. The ring capacity
    is doubled internally (``2 * capacity``) so each of the ring's two arms holds
    at most ``capacity`` terminals.

    ``constraints`` give the cable capacity, the topology (RADIAL or BRANCHED
    yield radial layouts) and the count and load bounds, which are resolved after
    the turbine power is quantized (see
    :class:`~optiwindnet.constraints.NetworkBounds`), or are the
    bounds already resolved.
    Each vehicle is one route: a subtree with one feeder, or a ring with two (one
    per arm). Feeder routes are not kept straight (``FeederRoute.STRAIGHT`` is
    refused).

    HGS-CVRP bounds the route count from above. A route count that must be exact
    (fixed above the minimum, or fixed for a root) is held by balancing the loads
    of those routes: slack nodes, depot clones of unit demand, fill every route
    up to an even load, so no route is left empty. A balanced solution also
    satisfies unbalanced bounds. An exact ring count leaves every ring two or
    more terminals.

    A ``max_link_length`` in ``constraints`` is priced, not forbidden: a link or
    feeder over it costs what a link absent from ``A`` costs. A solution that
    still uses one raises ``ValueError``.

    Normalization of input graph is recommended before calling this function.

    For single-root problems, the solver runs on the full graph. For multi-root
    problems, the terminals are clustered within the bounds (see
    :func:`~optiwindnet.clustering.clusterize`), each root getting a budget of
    routes, and each cluster is solved concurrently. Under balance, the routes of
    every cluster have loads within those balanced for the whole farm.

    A ``warmstart`` solution topology seeds the HGS-CVRP population with its
    routes (one per feeder subtree, or one per ring), and, for multi-root
    problems, its assignment of terminals to roots and its route counts replace
    the clustering. Only the first solve of the repair loop is seeded, since a
    retry runs on fewer links. An unusable warm start raises
    ``OWNWarmupFailed``: one that does not cover every terminal exactly once with
    routes hanging from a single root each, that breaks the bounds, or whose
    routes HGS-CVRP rejects (over capacity, more routes than the fleet, or, for
    an exact count, a route count other than it).

    Unequal turbine power (nonunitary inflow) is only honoured by a single-root,
    radial solve without a count fixed above the minimum or balance.

    If ``repair=True`` (the default), the solution is iteratively repaired
    until no crossings remain (or ``max_retries`` is reached). This may cause the
    actual runtime to be up to ``(max_retries + 1)`` times the given ``time_limit``.
    Repair requires ``A``'s planar mesh. Set ``repair=False`` when using
    ``complete=True``.

    Args:
        A: available-links graph. No edges implies ``complete=True``
        constraints: network constraints, cable capacity included, or the bounds
            resolved from them
        time_limit: [s] solver run time limit
        power_rtol: quantization tolerance of unequal turbine power (only with
            ``constraints``)
        seed: random seed for reproducibility
        keep_log: attach solver log to the solution graph
        repair: iteratively fix crossings (default True)
        max_retries: maximum repair iterations
        complete: allow every terminal pair, using Euclidean distances for
            links absent from ``A``. Enabled automatically when ``A`` has no
            edges. Requires ``repair=False`` because mesh-based crossing
            detection and repair cannot handle the added links. Use this
            option to assess how missing-link penalties affect solution
            quality. Pass the meshed ``A`` to preserve stored lengths
            for links that route around obstacles.
        log_callback: callback to receive each log line produced by HGS-CVRP
            (only for single-root instances)
        warmstart: solution topology S (radial, or ringed for a RINGED topology)
            to seed the search and, for multi-root instances, to cluster the
            terminals

    Returns:
        Solution topology S

    Raises:
        NotImplementedError: the bounds ask for straight feeders, for something
            unequal turbine power rules out (see below), or for an odd exact
            feeder count of a root with rings.
        ValueError: the bounds ask for more exact rings than ``T // 2``, or no
            clustering satisfies them; or the solution exceeds the
            constraints' ``max_link_length``.
        optiwindnet.types.OWNWarmupFailed: the warm start is unusable.
    """
    R = A.graph['R']
    T = A.graph['T']
    bounds, complete = vrp_bounds(
        'hgs_cvrp',
        A,
        constraints,
        power_rtol,
        complete=complete,
        repair=repair,
        counted_by='the balanced slack nodes are counted',
    )
    inflow_by_node = bounds.nonunit_inflow
    ringed = bounds.topology is Topology.RINGED
    balanced = bounds.balanced
    solve_capacity = bounds.group_capacity
    groups_lb, groups_ub = bounds.groups_lb, bounds.groups_ub

    plan = fleet_ = None
    if R == 1:
        # the root's bounds are within those of the farm
        pinned = balanced or bounds.fixed_above_min
        vehicles = groups_lb if pinned else groups_ub
    else:
        pinned, vehicles = False, None

    warm_routes_ = None
    if warmstart is not None:
        warm_routes_ = routes_from_warmstart(warmstart, R, T)
        if R > 1:
            try:
                plan = plan_from_clusters(
                    bounds,
                    [{n for route in routes for n in route} for routes in warm_routes_],
                    tuple(len(routes) for routes in warm_routes_),
                )
            except ValueError as exc:
                raise OWNWarmupFailed(
                    f'The warm start breaks the bounds: {exc}'
                ) from exc
    if R > 1 and plan is None:
        plan = cluster_plan(A, bounds)
    if plan is not None:
        fleet_ = root_fleet_(bounds, plan)
        pinned = any(plan.pinned_)

    if seed is None:
        seed = random.randrange(0, 2**31)
    hgs_options = {
        'timeLimit': time_limit,
        'seed': seed,
    }

    def _solve(warm_routes_=None):
        if plan is None:
            results_ = _solve_single_root(
                A,
                solve_capacity,
                hgs_options,
                vehicles,
                pinned,
                log_callback,
                ringed,
                complete,
                warm_routes_=warm_routes_,
                link_length_ub=bounds.link_length_ub,
                inflow_by_node=inflow_by_node,
            )
        else:
            results_ = _solve_multi_root(
                A,
                solve_capacity,
                hgs_options,
                plan,
                fleet_,
                log_callback,
                ringed,
                complete,
                warm_routes_=warm_routes_,
                link_length_ub=bounds.link_length_ub,
            )
        S = _process_results(A, keep_log, pinned, *results_)
        assert sum(S.nodes[r]['load'] for r in range(-R, 0)) == T, (
            'ERROR: root node load does not match T.'
        )
        return S

    # iterative repair loop
    A_orig = A  # the loop may rebind A to a pruned copy
    # Without repair, A need not contain mesh diagonals.
    diagonals = A.graph['diagonals'] if repair else A.graph.get('diagonals', {})
    crossings = []
    i = 0
    if not repair:
        S = _solve(warm_routes_)
    else:
        while True:
            # retries run on a pruned A, so only the first solve is seeded
            S = _solve(warm_routes_ if i == 0 else None)
            S = repair_routeset_path(S, A, ringed=ringed)
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
    over_limit = (
        []
        if bounds.link_length_ub is None
        else _links_over_limit(S, A_orig, bounds.link_length_ub)
    )
    if over_limit:
        raise ValueError(
            f'HGS-CVRP found no solution within the link length limit: '
            f'{len(over_limit)} link(s) exceed it, e.g. {over_limit[0]}.'
        )
    nx.set_node_attributes(S, inflow_by_node, 'inflow')
    if ringed:
        S.graph['topology'] = Topology.RINGED
        S = split_rings_and_calc_loads(S, A_orig)
    else:
        S.graph['topology'] = Topology.RADIAL
        calcload(S)

    # Encode against the original link set, before repair removed any links.
    S.graph.update(
        T=T,
        R=R,
        capacity=bounds.capacity,
        **linkset_identity(S, A_orig),
        creator='baselines.hgs',
        method_options=dict(
            solver_name='HGS-CVRP',
            complete=complete,
            feeders_above_min=None
            if groups_ub is None
            else groups_ub - bounds.groups_min,
            feeders_exact=groups_lb == groups_ub and pinned,
            balanced=balanced,
            ringed=ringed,
            repair=repair,
            max_retries=max_retries,
            fun_fingerprint=_hgs_cvrp_fun_fingerprint,
            **S.graph['method_options'],
        ),
        solver_details=dict(
            seed=seed,
            **S.graph['solver_details'],
            **root_details(bounds),
        ),
        **bounds.power_attrs,
    )
    return S


_hgs_cvrp_fun_fingerprint = fingerprint_function(hgs_cvrp)
