# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import math
import os
import random
import re
import subprocess
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import chain, islice, pairwise
from pathlib import Path
from typing import Any

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
debug, info, warn, error = _lggr.debug, _lggr.info, _lggr.warning, _lggr.error


# TODO: this belongs in interarraylib
def _prune_links(A: nx.Graph, max_blockable_per_link: int):
    # remove links that are likely to block feeders (heuristic pruning)
    unfeas_links = []
    closest_root = A.graph['closest_root']
    d2roots = A.graph['d2roots']
    for u, v, edgeD in A.edges(data=True):
        if u < 0 or v < 0:
            continue
        extent = edgeD['length']
        r_u, r_v = closest_root[u], closest_root[v]
        if (  # prune if savings are negative both ways
            extent > d2roots[u, r_u] and extent > d2roots[v, r_v]
        ) or (  # prune if link blocks line-of-sight to too many terminals
            r_u == r_v and edgeD['blocked__'][r_u].count() > max_blockable_per_link
        ):
            unfeas_links.append((u, v) if u < v else (v, u))
    debug('links removed in pre-processing: %s', unfeas_links)
    A.remove_edges_from(unfeas_links)
    diagonals = A.graph['diagonals']
    for link in unfeas_links:
        if link in diagonals:
            del diagonals[link]


def _extrema(pattern: str, line: str) -> tuple[float, ...] | None:
    """Parse the (min, max) pair out of one of LKH's summary lines.

    Returns ``None`` if the line does not match, since these per-cluster stats
    are optional and a format change should not abort an otherwise good solve.
    """
    match = re.match(pattern, line)
    if match is None:
        warn('Could not parse LKH summary line: %s', line)
        return None
    return tuple(float(v) for v in match.groups())


def _solution_time(log, objective) -> float:
    sol_repr = f'{objective}'
    time = 0.0
    for line in log.splitlines():
        if not line or line[0] == '*':
            continue
        if line[:4] == 'Run ':
            cost_, time_ = line.split(': ')[1].split(', ')
            time += float(time_.split(' = ')[1].split(' ')[0])
            if cost_.split('_')[1] == sol_repr:
                break
    return time


def _build_weight_matrix(
    A: nx.Graph,
    terminals: list[int],
    root: int,
    *,
    scale: float,
    complete: bool,
    w_clip: int,
    link_length_ub: float | None = None,
) -> np.ndarray:
    """Build LKH weight matrix ``L`` with depot at last index.

    The matrix has shape ``(T_c+1, T_c+1)``.

    Args:
        A: source graph (provides edge lengths, VertexC, d2roots).
        terminals: list of terminal node ids included (in matrix order 0..T_c-1).
        root: depot node id (negative integer).
        scale: factor to scale lengths.
        complete: if True, fill missing edges with Euclidean distances.
        w_clip: integer used for non-existing/clipped edges.

    Raises:
        OverflowError: a scaled length exceeds ``w_clip``. LKH multiplies our
            stored cost by ``PRECISION`` internally and works in 32-bit ints,
            so the budget per entry is ``int32_max // (2 * PRECISION)`` —
            which the caller passes here as ``w_clip``. Exceeding it usually
            means the input graph is not normalized (call
            :func:`~optiwindnet.transforming.as_normalized` before solving), or
            that ``scale`` is too large for the coordinate magnitudes.
    """
    T_c = len(terminals)

    def _check(value: float, source: str) -> None:
        if value > w_clip:
            raise OverflowError(
                f'LKH weight matrix overflows the per-entry budget: scaled '
                f'{source} reaches {value:.3e} > w_clip={w_clip} (= '
                f'int32_max // (2 * precision)). Normalize the input graph '
                f'(`as_normalized()`) or reduce `scale` (currently {scale:g}).'
            )

    R = A.graph['R']
    root_col = R + root  # convert negative root id (-R..-1) to d2roots column index
    d2root = A.graph['d2roots'][terminals, root_col]
    d2root_scaled = np.round(d2root * scale)
    _check(float(d2root_scaled.max(initial=0.0)), 'depot distance')

    block, fill_max, edge_max = scaled_length_block(
        A,
        terminals,
        scale=scale,
        complete=complete,
        absent=w_clip,
        dtype=np.int32,
        link_length_ub=link_length_ub,
    )
    _check(fill_max, 'pairwise distance')
    _check(edge_max, 'edge length')
    if link_length_ub is not None:
        # feeders over the limit get what an absent link gets
        d2root_scaled[d2root > link_length_ub] = w_clip

    # only the upper triangle is passed to LKH, so the depot row and the diagonal
    # are left at the sentinel
    L = np.full((T_c + 1, T_c + 1), w_clip, dtype=np.int32)
    L[:-1, :-1] = block
    L[:-1, -1] = d2root_scaled.astype(np.int32)
    return L


def _w_clip(precision: int) -> int:
    """Price of a link missing from ``A`` (in ``L``'s scaled units).

    It is the same expression LKH uses to define its big-M value, since the
    problem file never declares which links exist (see the specs in
    :func:`_do_lkh`).
    """
    return np.iinfo(np.int32).max // (2 * precision)


def _route_from_tour(tour_fpath: str, L: np.ndarray) -> tuple[list[int], int]:
    """Recover the single open route of a 1-vehicle solution from a TOUR_FILE.

    LKH-3 writes MTSP_SOLUTION_FILE only for two or more salesmen. With
    ``VEHICLES=1`` it still solves the OVRP, but reports the result as a plain
    closed tour over all ``T + 1`` nodes, with the depot's closing edge free.

    Cutting the cycle at the depot leaves the route; the cycle can be walked in
    either direction, which puts the feeder at one end or the other, so both
    orientations are costed against ``L`` and the cheaper one is returned.

    Returns:
        (route, cost) with ``route`` a list of 0-based terminal indices.
    """
    nodes: list[int] = []
    in_section = False
    for line in Path(tour_fpath).read_text().splitlines():
        line = line.strip()
        if line.startswith('TOUR_SECTION'):
            in_section = True
        elif in_section:
            if line == '-1' or line.startswith('EOF'):
                break
            nodes.append(int(line))
    depot = L.shape[0]  # 1-based id of the depot (last matrix index + 1)
    cut = nodes.index(depot)
    seq = [n - 1 for n in chain(nodes[cut + 1 :], nodes[:cut])]
    hops = sum(int(L[u, v]) for u, v in pairwise(seq))
    cost_fwd = int(L[seq[0], -1]) + hops
    cost_rev = int(L[seq[-1], -1]) + hops
    if cost_rev < cost_fwd:
        return seq[::-1], cost_rev
    return seq, cost_fwd


def _do_lkh(
    L: np.ndarray,
    *,
    capacity: int,
    vehicles: int,
    min_route_size: int,
    demands: Sequence[int] = (),
    time_limit: float,
    scale: float,
    runs: int,
    per_run_limit: float,
    precision: int,
    seed: int,
    initial_tour_nodes: list[int] | None,
    name: str,
    ringed: bool = False,
    asymmetric: bool = False,
) -> dict:
    """Run LKH-3 on a precomputed weight matrix.

    ``L`` has shape ``(T+1, T+1)`` with the depot at the last index. Only its
    upper triangle is passed to LKH, unless ``asymmetric``, which passes the
    whole matrix (see :func:`_with_slack_nodes`).

    With ``ringed=False`` (default) LKH solves an Open-CVRP (``TYPE=OVRP``): the
    return leg to the depot is free and routes are radial. With ``ringed=True``
    it solves the closed CVRP (``TYPE=CVRP``): every route returns to the depot,
    so the symmetric weight matrix charges both feeder legs and each route is a
    ring.

    Returns a dict containing routes (list of lists of 0-based terminal indices
    in the matrix), penalty, minimum, log, ``elapsed_time``, ``solution_time``, plus
    parsed run statistics.
    """
    T = L.shape[0] - 1
    N = T + 1
    if asymmetric:
        edge_weights = '\n'.join(' '.join(str(d) for d in row) for row in L)
    else:
        edge_weights = '\n'.join(
            ' '.join(str(d) for d in row[i + 1 :]) for i, row in enumerate(L[:-1])
        )

    problem_fname = 'problem.txt'
    params_fname = 'params.txt'
    output_fname = 'solution.out'
    tour_fname = 'solution.tour'
    initial_tour_fname = 'initial.tour'

    specs: dict[str, str | int | float] = {
        'NAME': name,
        'TYPE': 'CVRP' if ringed else 'OVRP',
        'DIMENSION': N,  # CVRP number of nodes and depots
        # For CAPACITY to be enforced, a DEMAND section is required.
        # MTSP_MAX_SIZE should work for unitary demand, but did not.
        'CAPACITY': capacity,
        'EDGE_WEIGHT_TYPE': 'EXPLICIT',
        'EDGE_WEIGHT_FORMAT': 'FULL_MATRIX' if asymmetric else 'UPPER_ROW',
        # The available-links set reaches LKH only through the weights: a link
        # absent from A is priced at the big-M of _build_cluster_weight_matrices,
        # never declared missing. Declaring it was tried and only made the solver
        # fail. LKH does offer EDGE_DATA_SECTION (with EDGE_DATA_FORMAT={ADJ_LIST,
        # EDGE_LIST}), but it is very hard to use with the VRP problem types used
        # here: LKH seems to expect the edges with respect to the transformed
        # problem, and no reference documents that transformation (the C code
        # lacks comments). EDGE_FILE, the alternative way to define a candidate
        # set, applies to the transformed problem as well and follows the Concorde
        # format, whose indices start at 0.
    }
    if not ringed:
        # LKH penalizes a route longer than DISTANCE, and any penalty outranks
        # any cost. A route with a link missing from A is longer than that link's
        # price, so the cap keeps such links out of every solution that has no
        # penalty. It is not imposed on rings (untried for TYPE=CVRP).
        specs['DISTANCE'] = _w_clip(precision)  # maximum route length
    if not demands:
        demands = [1] * T
    data = {
        'EDGE_WEIGHT_SECTION': edge_weights,
        'DEMAND_SECTION': '\n'.join(
            chain(
                (f'{i} {demand}' for i, demand in enumerate(demands, start=1)),
                (f'{N} 0',),
            )
        ),
    }
    params: dict[str, str | int | float | None] = {
        # SPECIAL is a shorthand for a bundle of large-neighborhood move settings
        # (MOVE_TYPE='5 SPECIAL', MAX_SWAPS=0, ...). Without it, LKH-3 often ends
        # with a penalty (counts or sizes off the bounds) and rings use rogue
        # links; with it, both topologies come out shorter.
        'SPECIAL': None,  # None -> output only the key
        'DEPOT': N,
        'SEED': seed,  # 0 means pick a random seed
        'PRECISION': precision,  # d[i][j] = PRECISION*c[i][j] + pi[i] + pi[j]
        'TOTAL_TIME_LIMIT': time_limit,
        'TIME_LIMIT': per_run_limit,
        'RUNS': runs,  # default: 10
        # MAX_TRIALS=100,  # default: number of nodes (DIMENSION)
        # TRACE_LEVEL=1,  # default is 1, 0 supresses output
        #  INITIAL_TOUR_ALGORITHM='GREEDY',  # { … | CVRP | MTSP | SOP } Default: WALK
        'VEHICLES': vehicles,
        # penalized for TYPE=CVRP (rings), ignored for TYPE=OVRP, whose routes
        # are filled up with slack nodes instead (see _with_slack_nodes)
        'MTSP_MIN_SIZE': min_route_size,
        # a node cap, so it is CAPACITY over the lightest demand a route can hold
        'MTSP_MAX_SIZE': capacity // min(demands, default=1),
        'MTSP_OBJECTIVE': 'MINSUM',  # [ MINMAX | MINMAX_SIZE | MINSUM ]
        'MTSP_SOLUTION_FILE': output_fname,
        # LKH-3 only writes MTSP_SOLUTION_FILE for 2+ salesmen. With a single
        # vehicle it solves the problem but reports it as a plain tour, so ask
        # for TOUR_FILE too and recover the lone route from it (see below).
        'TOUR_FILE': tour_fname,
        #  MOVE_TYPE='5 SPECIAL',  # <integer> [ SPECIAL ]
        #  GAIN23='NO',
        #  KICKS=1,
        #  KICK_TYPE=4,
        #  MAX_SWAPS=0,
        #  POPULATION_SIZE=12,  # default 10
        #  PATCHING_A=
        #  PATCHING_C=
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        problem_fpath = os.path.join(tmpdir, problem_fname)
        Path(problem_fpath).write_text(
            '\n'.join(
                chain(
                    (f'{k}: {v}' for k, v in specs.items()),
                    (f'{k}\n{v}' for k, v in data.items()),
                    ('EOF',),
                )
            )
        )
        params['PROBLEM_FILE'] = problem_fpath
        params_fpath = os.path.join(tmpdir, params_fname)
        params['MTSP_SOLUTION_FILE'] = os.path.join(tmpdir, output_fname)
        params['TOUR_FILE'] = os.path.join(tmpdir, tour_fname)
        if initial_tour_nodes is not None:
            initial_tour_fpath = os.path.join(tmpdir, initial_tour_fname)
            Path(initial_tour_fpath).write_text(
                '\n'.join(
                    (
                        f'NAME: {name}',
                        'TYPE: TOUR',
                        'TOUR_SECTION',
                        *(str(node) for node in initial_tour_nodes),
                        '-1',
                        'EOF',
                    )
                )
            )
            params['INITIAL_TOUR_FILE'] = initial_tour_fpath
            params['INITIAL_TOUR_FRACTION'] = 1.0
        Path(params_fpath).write_text(
            '\n'.join((f'{k} = {v}' if v is not None else k) for k, v in params.items())
        )
        start_time = time.perf_counter()
        try:
            # LKH's exit code is not a reliable success signal; output is parsed
            result = subprocess.run(
                ['LKH', params_fpath],
                capture_output=True,
                check=False,
                # TOTAL_TIME_LIMIT counts preprocessing; if that alone takes up the
                # limit, LKH completes no run and hangs instead of exiting. The
                # margin covers wall time exceeding CPU time on a busy machine.
                timeout=1.5 * time_limit + 2,
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            result = subprocess.CompletedProcess(
                exc.cmd, -9, exc.stdout or b'', exc.stderr or b''
            )
            timed_out = True
        elapsed_time = time.perf_counter() - start_time
        output_fpath = os.path.join(tmpdir, output_fname)
        tour_fpath = os.path.join(tmpdir, tour_fname)
        solution_parsed = not timed_out and Path(output_fpath).is_file()
        if solution_parsed:
            with open(output_fpath, 'r') as f_sol:
                penalty, minimum = next(f_sol).split(':')[-1][:-1].split('_')
                next(f_sol)  # discard second line
                routes = [
                    [int(node) - 1 for node in line.split(' ')[1:-5]] for line in f_sol
                ]
        elif not timed_out and Path(tour_fpath).is_file():
            # single-vehicle solve: LKH wrote a plain tour instead (it only
            # writes MTSP_SOLUTION_FILE for 2+ salesmen). LKH omits the tour
            # file altogether when it cannot satisfy the constraints, so its
            # presence means the solution is feasible.
            route, cost = _route_from_tour(tour_fpath, L)
            routes = [route]
            penalty = '0'
            minimum = str(cost)
            solution_parsed = True
        else:
            penalty = '0'
            minimum = 'inf'
            routes = []

    log = result.stdout.decode('utf8')
    output: dict[str, Any] = {
        'routes': routes,
        'penalty': int(penalty),
        'minimum': minimum,
        'cost': float(minimum) / scale,
        'log': log,
        'stderr': result.stderr.decode('utf8'),
        'elapsed_time': elapsed_time,
        'solution_time': _solution_time(log, minimum),
        'vehicles': vehicles,
        'seed': seed,
        'timed_out': timed_out,
    }

    if not solution_parsed or result.stderr:
        info('===stdout===\n%s', log)
        error('===stderr===\n%s', output['stderr'])
        return output

    tail = result.stdout[result.stdout.rfind(b'Successes/') :].decode()
    entries = iter(tail.splitlines())
    next(entries)  # skip successes line
    output['cost_extrema'] = _extrema(
        r'Cost\.min = (-?\d+), Cost\.avg = -?\d+\.?\d*, Cost\.max = -?(\d+)',
        next(entries),
    )
    next(entries)  # skip gap line
    output['penalty_extrema'] = _extrema(
        r'Penalty\.min = (\d+), Penalty\.avg = \d+\.?\d*, Penalty\.max = (\d+)',
        next(entries),
    )
    output['trials_extrema'] = _extrema(
        r'Trials\.min = (\d+), Trials\.avg = \d+\.?\d*, Trials\.max = (\d+)',
        next(entries),
    )
    output['runtime_extrema'] = _extrema(
        r'Time\.min = (\d+\.?\d*) sec., Time\.avg = \d+\.?\d* sec.,'
        r' Time\.max = (\d+\.?\d*) sec.',
        next(entries),
    )
    return output


def _with_slack_nodes(L: np.ndarray, num_slack: int, w_clip: int) -> np.ndarray:
    """Add slack nodes to a radial weight matrix, making it asymmetric.

    Slack nodes are depot clones of unit demand that fill every route up to the
    capacity, so that no route is left empty (see
    :func:`~optiwindnet.baselines._core.balanced_capacity`). They take the rows
    and columns between the terminals and the depot. Only the depot leads to a
    slack node, at no cost, and a slack node leads to a terminal at the price of
    the terminal's feeder. A terminal leading to a slack node, or a slack node
    to another, costs ``w_clip``, which the route-length limit of
    :func:`_do_lkh` penalizes. Hence a slack node can only open a route. With a
    symmetric matrix, LKH would place a slack node between two terminals: an
    extra feeder that no vehicle accounts for.

    Args:
        L: symmetric weight matrix of :func:`_build_weight_matrix`, depot last.
        num_slack: number of slack nodes.
        w_clip: price of a link missing from ``A``.

    Returns:
        The full matrix, of shape ``(T_c + num_slack + 1,) * 2``, depot last.
    """
    T_c = L.shape[0] - 1
    n = T_c + num_slack + 1
    d2root = L[:T_c, -1]
    M = np.full((n, n), w_clip, dtype=L.dtype)
    M[:T_c, :T_c] = L[:T_c, :T_c]
    # LKH frees the leg back to the depot itself (TYPE=OVRP)
    M[:T_c, -1] = M[-1, :T_c] = d2root
    M[T_c:-1, :T_c] = d2root
    M[T_c:-1, -1] = M[-1, T_c:-1] = 0
    np.fill_diagonal(M, 0)
    return M


def _build_cluster_weight_matrices(
    A: nx.Graph,
    terminals_: list[list[int]],
    *,
    scale: float,
    complete: bool,
    precision: int,
    link_length_ub: float | None = None,
) -> list[np.ndarray]:
    """Build one LKH weight matrix per cluster.

    Computes the ``w_clip`` sentinel (used for missing/clipped edges) from
    ``precision`` once, then calls :func:`_build_weight_matrix` for each
    (terminals, root) pair in root order (-R..-1).
    """
    R = A.graph['R']
    w_clip = _w_clip(precision)
    return [
        _build_weight_matrix(
            A,
            terminals,
            r,
            scale=scale,
            complete=complete,
            w_clip=w_clip,
            link_length_ub=link_length_ub,
        )
        for r, terminals in zip(range(-R, 0), terminals_)
    ]


def _solve_cluster(
    L: np.ndarray,
    *,
    capacity: int,
    vehicles: int,
    balanced: bool,
    demands: Sequence[int] = (),
    time_limit: float,
    scale: float,
    runs: int,
    per_run_limit: float,
    precision: int,
    seed: int,
    initial_tour_nodes: list[int] | None,
    name: str,
    ringed: bool = False,
    num_slack: int = 0,
    route_sizes: tuple[int, int] | None = None,
) -> dict:
    """Run LKH-3 on a pre-built cluster weight matrix.

    ``L`` is shape ``(T_c+1, T_c+1)`` with the depot at the last index. Derives
    ``min_route_size`` from ``vehicles``/``capacity``/``balanced`` and dispatches to
    :func:`_do_lkh`. The matrix is built by :func:`_build_cluster_weight_matrices`,
    decoupled from this call so it can be reused across iterations that do
    not mutate the underlying graph.

    A radial solve with ``num_slack`` slack nodes (see :func:`_with_slack_nodes`)
    returns its routes without them, and counts in ``'stray_slack'`` the slack
    nodes found anywhere but at the start of a route. ``route_sizes`` bounds the
    terminals of every ring (``TYPE=CVRP`` enforces ``MTSP_MIN_SIZE``).
    """
    T_c = L.shape[0] - 1
    if num_slack:
        L = _with_slack_nodes(L, num_slack, _w_clip(precision))
        min_route_size = 0
    elif route_sizes is not None:
        min_route_size = route_sizes[0]
    elif ringed or demands:
        # MTSP_MIN_SIZE for TYPE=CVRP would require a FULL_MATRIX (asymmetric)
        # formulation; the symmetric ring solve does not impose a minimum size.
        # It counts nodes, too, which bounds nothing once the demands differ.
        min_route_size = 0
    elif balanced:
        min_route_size = T_c // vehicles
    elif vehicles == math.ceil(T_c / capacity):
        min_route_size = (T_c % capacity) or capacity
    else:
        min_route_size = 0
    output = _do_lkh(
        L,
        capacity=capacity,
        vehicles=vehicles,
        min_route_size=min_route_size,
        demands=demands,
        time_limit=time_limit,
        scale=scale,
        runs=runs,
        per_run_limit=per_run_limit,
        precision=precision,
        seed=seed,
        initial_tour_nodes=initial_tour_nodes,
        name=name,
        ringed=ringed,
        asymmetric=bool(num_slack),
    )
    if num_slack:
        routes = output['routes']
        output['stray_slack'] = sum(
            1 for route in routes for k, i in enumerate(route) if i >= T_c and k > 0
        )
        output['routes'] = [[i for i in route if i < T_c] for route in routes]
    return output


def _initial_tours(
    routes_: list[list[list[int]]],
    terminals_: list[list[int]],
    vehicles_: list[int],
    num_slack_: Sequence[int] | None = None,
    capacity_: Sequence[int] | None = None,
) -> list[list[int] | None]:
    """Translate warm-start routes into one LKH initial tour per root.

    LKH refers to terminal ``n`` as customer ``i + 1``, where ``i`` is ``n``'s
    position in the cluster's sorted ``terminals`` list (its row in the LKH
    weight matrix). Slack nodes (see :func:`_with_slack_nodes`), if any, follow
    the terminals; then come the depot and its ``vehicles - 1`` clones. In
    LKH's tour, the depot nodes separate the routes, so a clone follows each
    route and the vehicles left unused are empty routes: consecutive clones,
    closed by the depot. Each route opens with the slack nodes that fill it up
    to its root's capacity, as far as they go; any left over open the next
    route.

    The tour only hints at the visiting order: LKH costs it from the weight
    matrix, so a warm-start link absent from it gets the ``w_clip`` price.

    Args:
        routes_: warm-start routes of each root (see
            :func:`~optiwindnet.baselines._core.routes_from_warmstart`), whose
            terminals belong to that root's cluster.
        terminals_: sorted terminals of each root's cluster.
        vehicles_: vehicles of each root, at least its number of routes.
        num_slack_: slack nodes of each root (none if not given).
        capacity_: capacity of each root's routes (needed with slack nodes).

    Returns:
        The tour of each root, or ``None`` for a root without routes.
    """
    tours: list[list[int] | None] = []
    for c, (routes, terminals, vehicles) in enumerate(
        zip(routes_, terminals_, vehicles_)
    ):
        if not routes:
            tours.append(None)
            continue
        T_c = len(terminals)
        num_slack = 0 if num_slack_ is None else num_slack_[c]
        n_customers = T_c + num_slack
        id_from_node = {n: i for i, n in enumerate(terminals, start=1)}
        slack = iter(range(T_c + 1, n_customers + 1))
        separators = [
            *range(n_customers + 2, n_customers + vehicles + 1),
            n_customers + 1,
        ]
        tour = []
        for route, separator in zip(routes, separators):
            if num_slack:
                assert capacity_ is not None
                tour.extend(islice(slack, max(capacity_[c] - len(route), 0)))
            tour.extend(id_from_node[n] for n in route)
            tour.append(separator)
        tour.extend(slack)
        tour.extend(separators[len(routes) :])
        tours.append(tour)
    return tours


def _build_solution(
    A: nx.Graph,
    *,
    capacity: int,
    outputs_: list[dict],
    terminals_: list[list[int]],
    keep_log: bool,
    method_options: dict,
    solver_details_extra: dict,
) -> nx.DiGraph:
    """Assemble the final solution graph S from per-root LKH outputs.

    Raises:
        ValueError: LKH-3 returned no routes for a root with terminals.
    """
    R, T = A.graph['R'], A.graph['T']
    for r, output, terminals in zip(range(-R, 0), outputs_, terminals_):
        if terminals and not output['routes']:
            raise ValueError(
                f'LKH-3 found no solution for root {r} ({len(terminals)} '
                f'terminals, {output["vehicles"]} vehicles).'
                + (
                    ' LKH-3 did not finish: `time_limit` may be shorter than its '
                    'preprocessing.'
                    if output.get('timed_out')
                    else ''
                )
            )
    multi = R > 1
    objective = sum(o['cost'] for o in outputs_)
    runtime = max(o['elapsed_time'] for o in outputs_)
    solution_times = [o['solution_time'] for o in outputs_]
    logs = [o['log'] for o in outputs_]
    vehicles_ = [o['vehicles'] for o in outputs_]
    penalties = [o['penalty'] for o in outputs_]

    S = nx.Graph(
        T=T,
        R=R,
        capacity=capacity,
        objective=objective,
        creator='baselines.lkh',
        runtime=runtime,
        solution_time=tuple(solution_times) if multi else solution_times[0],
        method_options=method_options,
        solver_details=dict(
            penalty=tuple(penalties) if multi else penalties[0],
            vehicles=tuple(vehicles_) if multi else vehicles_[0],
            **solver_details_extra,
        ),
    )
    if keep_log:
        S.graph['method_log'] = tuple(logs) if multi else logs[0]

    # extract optional per-cluster stats
    for key in ('cost_extrema', 'penalty_extrema', 'trials_extrema', 'runtime_extrema'):
        values = [o.get(key) for o in outputs_]
        if any(v is not None for v in values):
            S.graph['solver_details'][key] = tuple(values) if multi else values[0]

    S.add_nodes_from(range(-R, 0))
    subtree_id = 0
    max_load = 0
    for r, output, terminals in zip(range(-R, 0), outputs_, terminals_):
        # output['routes'] uses matrix indices (0..T_c-1) for terminals
        subtrees = [[terminals[i] for i in route] for route in output['routes']]
        # rings, too, are built as paths here (one feeder each), so that the
        # path-based repair machinery applies; split_rings_and_calc_loads closes
        # them afterwards
        sub_max_load, subtree_id = add_branches_to_S(
            S, subtrees, root=r, subtree_id_start=subtree_id
        )
        max_load = max(max_load, sub_max_load)
        root_load = sum(S.nodes[n]['load'] for n in S.neighbors(r))
        S.nodes[r]['load'] = root_load

    S.graph['max_load'] = max_load
    return _orient_tree(S)


def _lkh(
    A: nx.Graph,
    *,
    capacity: int,
    time_limit: float,
    scale: float = 1e5,
    vehicles: int | None = None,
    runs: int = 50,
    per_run_limit: float = 15.0,
    precision: int = 1000,
    complete: bool = False,
    keep_log: bool = False,
    seed: int | None = None,
    initial_tour_nodes: list[int] | None = None,
) -> nx.DiGraph:
    """Low-level single-root Lin-Kernighan-Helsgaun (LKH-3) solver.

    Open Capacitated Vehicle Routing Problem on a single depot. ``A`` must be
    normalized (use :func:`~optiwindnet.transforming.as_normalized` before
    calling) and have R == 1. For multi-root instances, use :func:`lkh3` instead.

    See :func:`lkh3` for a higher-level wrapper that handles multi-root,
    iterative repair, and parameter validation.

    Args:
      A: graph with allowed edges (if it has 0 edges, use ``complete=True``)
      capacity: maximum vehicle capacity
      time_limit: [s] solver run time limit
      scale: factor to scale lengths (should be < 1e6)
      vehicles: number of vehicles (if None, use the minimum feasible)
      runs: consult LKH manual
      per_run_limit: [s] consult LKH manual
      precision: consult LKH manual
      complete: make the full graph over A available (links not in A assumed direct)
      keep_log: save the LKH text output to graph attr ``'method_log'``
      seed: for the pseudo-random number generator (None or 0: random seed)
      initial_tour_nodes: optional initial tour for LKH (1-indexed nodes)

    Returns:
      Solution topology S
    """
    R, T = A.graph['R'], A.graph['T']
    assert R == 1, 'LKH allows only 1 depot'

    vehicles_min: int = math.ceil(T / capacity)
    if (vehicles is None) or (vehicles <= vehicles_min):
        if vehicles is not None and vehicles < vehicles_min:
            warn(
                f'Vehicle number ({vehicles}) too low for feasibility '
                f'with capacity ({capacity}). Setting to {vehicles_min}.'
            )
        vehicles = vehicles_min
        balanced = True
    else:
        balanced = False
    seed_for_lkh = 0 if seed is None else seed

    terminals = list(range(T))
    [L] = _build_cluster_weight_matrices(
        A, [terminals], scale=scale, complete=complete, precision=precision
    )
    output = _solve_cluster(
        L,
        capacity=capacity,
        vehicles=vehicles,
        balanced=balanced,
        time_limit=time_limit,
        scale=scale,
        runs=runs,
        per_run_limit=per_run_limit,
        precision=precision,
        seed=seed_for_lkh,
        initial_tour_nodes=initial_tour_nodes,
        name=A.graph.get('name', 'unnamed'),
    )

    method_options = {
        'solver_name': 'LKH-3',
        'time_limit': time_limit,
        'scale': scale,
        'runs': runs,
        'per_run_limit': per_run_limit,
        'complete': complete,
        'fun_fingerprint': _lkh_fun_fingerprint,
    }
    S = _build_solution(
        A,
        capacity=capacity,
        outputs_=[output],
        terminals_=[terminals],
        keep_log=keep_log,
        method_options=method_options,
        solver_details_extra={'seed': seed},
    )
    assert S.nodes[-1]['load'] == T, 'ERROR: root node load does not match T.'
    S.graph['has_loads'] = True
    return S


_lkh_fun_fingerprint = fingerprint_function(_lkh)


def _no_terminals_output(seed: int) -> dict:
    """Stand-in for :func:`_do_lkh`'s output on a root that got no terminals."""
    return {
        'routes': [],
        'penalty': 0,
        'minimum': '0',
        'cost': 0.0,
        'log': '',
        'stderr': '',
        'elapsed_time': 0.0,
        'solution_time': 0.0,
        'vehicles': 0,
        'seed': seed,
    }


def _run_lkh_per_cluster(
    L_: list[np.ndarray],
    *,
    name: str,
    capacity_: Sequence[int],
    time_limit: float,
    vehicles_: list[int],
    warmstart_tours: list[list[int] | None],
    balanced: bool,
    demands: Sequence[int] = (),
    scale: float,
    runs: int,
    per_run_limit: float,
    precision: int,
    seed: int,
    ringed: bool = False,
    num_slack_: Sequence[int] | None = None,
    route_sizes_: Sequence[tuple[int, int] | None] | None = None,
) -> list[dict]:
    """Solve every root cluster with LKH-3, sequentially or in parallel.

    Each root has its own capacity, slack nodes and route-size bounds (see
    :func:`_solve_cluster`).

    Single-root (R == 1) is solved synchronously; multi-root dispatches one
    :func:`_solve_cluster` per root through a ThreadPoolExecutor (one thread per
    root). Returns one LKH output dict per root, in root order (-R..-1).

    A root with an empty cluster is not dispatched (LKH rejects the resulting 1x1
    weight matrix) but still gets an entry, so that root ids stay aligned with
    cluster indices.
    """
    R = len(L_)
    num_slack_ = num_slack_ or (0,) * R
    route_sizes_ = route_sizes_ or (None,) * R
    job_kwargs_ = [
        {
            'L': L,
            'capacity': capacity_c,
            'vehicles': vehicles_c,
            'balanced': balanced,
            'demands': demands,
            'time_limit': time_limit,
            'scale': scale,
            'runs': runs,
            'per_run_limit': per_run_limit,
            'precision': precision,
            'seed': seed,
            'initial_tour_nodes': init_tour,
            'name': name if R == 1 else f'{name}_root{r}',
            'ringed': ringed,
            'num_slack': num_slack_c,
            'route_sizes': route_sizes_c,
        }
        for r, L, capacity_c, vehicles_c, init_tour, num_slack_c, route_sizes_c in zip(
            range(-R, 0),
            L_,
            capacity_,
            vehicles_,
            warmstart_tours,
            num_slack_,
            route_sizes_,
        )
    ]
    if R == 1:
        return [_solve_cluster(**job_kwargs_[0])]
    populated_ = [c for c, L in enumerate(L_) if L.shape[0] > 1]
    with ThreadPoolExecutor(max_workers=len(populated_) or 1) as executor:
        solved_ = list(
            executor.map(
                lambda kw: _solve_cluster(**kw),
                (job_kwargs_[c] for c in populated_),
            )
        )
    outputs_ = [_no_terminals_output(seed) for _ in range(R)]
    for c, output in zip(populated_, solved_):
        outputs_[c] = output
    return outputs_


#: Feeders worth offering a cluster that fits within ``capacity`` — i.e. one whose
#: capacity constraint is inactive. Says nothing about clusters at large: a cluster
#: that fills its feeders is bound by ``ceil(T_c / capacity)`` and never comes here.
#:
#: With capacity to spare, a second feeder beats extending a subtree by one more
#: terminal-terminal link only if it leaves the root at a significantly different
#: angle (upwards of pi/2) — otherwise the shorter move is to stay on the subtree
#: and join it at the root anyway, which the spare capacity allows. Only about
#: 2*pi / (pi/2) = 4 such directions fit around a root, so 4 is the practical
#: ceiling, and it does not grow with the cluster. Of the 85 single-root bundled
#: locations solved with capacity = T (capacity fully inactive), exactly one
#: (`horns3`, whose substation sits inside the array) does better with a 5th feeder
#: — by 0.48%, on a 2-turbine stub next to the root, where the feeder leg is nearly
#: free — and none does better with a 6th.
#:
#: Being generous past this point *hurts*: LKH's mTSP transformation adds
#: ``vehicles - 1`` depot clones, and the bloated search converges worse in the same
#: time. Offered T vehicles, `anglia` returns an 8-feeder solution 2.6% *longer*
#: than the one it finds when held to 4.
_MAX_FEEDERS_WITHIN_CAPACITY = 5


def _vehicles_within_capacity(T_c: int) -> int:
    """Vehicles to offer LKH for a cluster that fits within ``capacity``.

    Such a cluster needs only one feeder, but one is not a good number to ask for
    (see :func:`_setup_clusters`), so it gets an allowance instead of a pin. The
    allowance is only an *upper* bound: LKH-3 ignores ``MTSP_MIN_SIZE`` for
    ``TYPE=OVRP`` and leaves the surplus routes empty.
    """
    return min(T_c, _MAX_FEEDERS_WITHIN_CAPACITY)


def _setup_clusters(
    A: nx.Graph,
    bounds: NetworkBounds,
    *,
    warm_routes_: list[list[list[int]]] | None = None,
) -> tuple[list[list[int]], list[int], list[int | None], list[bool]]:
    """Compute per-root terminals, vehicle counts, vehicle caps and pins.

    For R == 1 the only cluster is ``range(T)``; for R > 1 the terminals are
    partitioned within ``bounds`` by :func:`~optiwindnet.clustering.cluster_plan`
    and each cluster's terminal list is sorted so that LKH customer ids
    ``[1..T_c]`` correspond to the cluster's nodes in sorted order (and
    :func:`_initial_tours` agrees on indexing). Warm-start routes, if given,
    replace the clustering of R > 1 with their own terminals and route counts,
    and every root is offered at least as many vehicles as its warm-start
    routes.

    A pinned root must use exactly its vehicles (see
    :func:`~optiwindnet.clustering.cluster_plan`): for R == 1, under balance or
    a route count fixed above the minimum. It is offered its budget in the
    clustering, or the route count for R == 1. Otherwise, if the wind farm's
    route count has a maximum, each root is offered its whole fleet: that
    maximum for R == 1, the root's budget in the clustering otherwise. If not, a
    root is offered its minimum feasible vehicle count, capped by its own feeder
    limit, except:

    - a cluster that fits within ``capacity`` (``T_c <= capacity``) gets
      :func:`_vehicles_within_capacity` instead of the minimum of 1. Its minimum
      is one only because the capacity constraint is inactive, and that same spare
      capacity is what makes extra feeders harmless: any routes LKH returns can be
      joined into a single subtree at the root without exceeding capacity, so the
      layout stays radial and no feeder budget is overspent. Pinning it to 1
      instead demands a single route, which over the sparse (near-planar) link set
      ``A`` means a Hamiltonian path — and one need not exist. LKH then finds no
      solution for the cluster.

    Returns:
        The terminals of each root, the vehicles offered to each root, the
        most vehicles each root may use (``None`` if unlimited) and whether
        each root is pinned.

    Raises:
        NotImplementedError: a root's exact feeder count is odd for rings.
        ValueError: no clustering satisfies ``bounds``.
        optiwindnet.types.OWNWarmupFailed: the warm-start routes break ``bounds``.
    """
    R, T, capacity = bounds.R, bounds.T, bounds.group_capacity
    if R == 1:
        terminals_ = [list(range(T))]
        # Nonunitary inflow is supported only for a single root.
        demand_ = [bounds.inflow_total]
        fleet_ = [bounds.groups_ub]
        groups_ub = bounds.groups_ub
        budget_ = [bounds.groups_lb]
        pinned_ = [bounds.balanced or bounds.fixed_above_min]
        if (
            warm_routes_ is not None
            and groups_ub is not None
            and len(warm_routes_[0]) > groups_ub
        ):
            raise OWNWarmupFailed(
                f'The warm start has {len(warm_routes_[0])} routes, more than '
                f'the maximum of {groups_ub}.'
            )
    else:
        if warm_routes_ is None:
            plan = cluster_plan(A, bounds)
        else:
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
        budget_, pinned_ = list(plan.budget_), list(plan.pinned_)
        terminals_ = [sorted(c) for c in plan.cluster_]
        demand_ = [len(c) for c in terminals_]
        fleet_ = root_fleet_(bounds, plan)
    vehicles_ = []
    for terminals, demand, fleet, budget, pinned in zip(
        terminals_, demand_, fleet_, budget_, pinned_
    ):
        if pinned:
            vehicles_.append(budget)
            continue
        vehicles_min = math.ceil(demand / capacity)
        vehicles = (
            _vehicles_within_capacity(len(terminals))
            if vehicles_min == 1
            else vehicles_min
        )
        if fleet is not None:
            vehicles = fleet if bounds.feeders_ub is not None else min(fleet, vehicles)
        vehicles_.append(vehicles)
    if warm_routes_ is not None:
        vehicles_ = [max(v, len(routes)) for v, routes in zip(vehicles_, warm_routes_)]
    return terminals_, vehicles_, fleet_, pinned_


def lkh3(
    A: nx.Graph,
    constraints: NetworkConstraints | NetworkBounds,
    *,
    time_limit: float,
    power_rtol: float | None = None,
    seed: int | None = None,
    keep_log: bool = False,
    repair: bool = True,
    max_retries: int = 10,
    scale: float = 1e5,
    runs: int = 50,
    per_run_limit: float = 15.0,
    precision: int = 1000,
    complete: bool = False,
    warmstart: nx.Graph | None = None,
) -> nx.DiGraph:
    """Solve the O/CVRP using LKH-3 with links from ``A``.

    Wraps the LKH-3 executable, which is not distributed with OptiWindNet.
    Get it from http://akira.ruc.dk/~keld/research/LKH-3/ and make sure the
    ``LKH`` executable is in the environment's PATH.

    Uses the Lin-Kernighan-Helsgaun meta-heuristic to solve an Open-CVRP
    (i.e., vehicles do not return to the depot), yielding radial layouts. For a
    RINGED topology it solves the closed CVRP instead (``TYPE=CVRP``): every
    route returns to the depot, forming a ring whose capacity is doubled
    internally (``2 * capacity``) so each of the two arms holds at most
    ``capacity`` terminals. Normalization of the input graph is recommended
    before calling this function (use :func:`~optiwindnet.transforming.as_normalized`).

    ``constraints`` give the cable capacity, the topology (RADIAL or BRANCHED
    yield radial layouts) and the count and load bounds, which are resolved after
    the turbine power is quantized (see
    :class:`~optiwindnet.constraints.NetworkBounds`), or are the
    bounds already resolved.
    Each vehicle is one route: a subtree with one feeder, or a ring with two (one
    per arm). Feeder routes are not kept straight (``FeederRoute.STRAIGHT`` is
    refused).

    LKH-3 bounds the route count from above only: it may leave vehicles unused.
    A route count that must be exact (fixed above the minimum, or fixed for a
    root) or balanced is held by other means. Radial routes are filled up with
    slack nodes, depot clones of unit demand that only the depot may lead to (the
    weight matrix becomes asymmetric), so every route comes out full and the
    loads are balanced, as in :func:`~optiwindnet.baselines.hgs.hgs_cvrp`. Rings
    are held by LKH-3's route-size bounds (``MTSP_MIN_SIZE`` and
    ``MTSP_MAX_SIZE``): two or more terminals per ring, balanced only under
    balance. LKH-3 may still return a solution off these bounds (its constraints
    are penalties), which raises ``ValueError``; a longer ``time_limit`` may
    help. A count fixed at the minimum needs none of this, since fewer routes
    are infeasible.

    For single-root problems, the solver runs on the full graph. For multi-root
    problems, the terminals are clustered within the bounds (see
    :func:`~optiwindnet.clustering.clusterize`) and each cluster is solved
    concurrently. Without a maximum route count, each root is offered its
    minimum feasible route count, except a root whose cluster fits within a
    single route, which is offered enough routes to use as many as lower the
    cable length (within the root's own limit).

    Unequal turbine power (nonunitary inflow) is only honoured by a single-root,
    radial solve without a count fixed above the minimum or balance.

    A link absent from ``A`` is priced, not forbidden, so LKH-3 may still use
    one: a solution with such rogue links raises ``ValueError`` (unless
    ``complete``).

    A ``warmstart`` solution topology gives LKH-3 its initial tour (its routes,
    one per feeder subtree or ring), and, for multi-root problems, its
    assignment of terminals to roots and its route counts replace the
    clustering. Each root is offered at least as many vehicles as its
    warm-start routes. Only the first solve of the repair loop is seeded, unless
    a retry only adds vehicles. An unusable warm start raises
    ``OWNWarmupFailed``: one that does not cover every terminal exactly once with
    routes hanging from a single root each, or that breaks the bounds.

    If ``repair=True`` (the default), the solution is iteratively repaired
    until no crossings remain (or ``max_retries`` is reached). This may cause
    the actual runtime to be up to ``(max_retries + 1)`` times the given
    ``time_limit``. Repair requires ``A``'s planar mesh. Set ``repair=False``
    when using ``complete=True``.

    A ``max_link_length`` in ``constraints`` is priced, not forbidden: a link or
    feeder over it costs what a link absent from ``A`` costs. A solution that
    still uses one raises ``ValueError``.

    Args:
        A: available-links graph. No edges implies ``complete=True``.
        constraints: network constraints, cable capacity included, or the bounds
            resolved from them.
        time_limit: [s] solver run time limit (per cluster).
        power_rtol: quantization tolerance of unequal turbine power (only with
            ``constraints``).
        seed: random seed for reproducibility (if None, picks a random one).
        keep_log: attach solver log to the solution graph.
        repair: iteratively fix crossings (default True).
        max_retries: maximum repair iterations.
        scale: factor to scale lengths (LKH manual).
        runs: number of LKH runs (LKH manual).
        per_run_limit: [s] LKH per-run time limit.
        precision: LKH precision parameter.
        complete: allow every terminal pair, using Euclidean distances for
            links absent from ``A``. Enabled automatically when ``A`` has no
            edges. Requires ``repair=False`` because mesh-based crossing
            detection and repair cannot handle the added links. Use this
            option to assess how missing-link penalties affect solution
            quality. Pass the meshed ``A`` to preserve stored lengths
            for links that route around obstacles.
        warmstart: solution topology S (radial, or ringed for a RINGED topology)
            to seed the search and, for multi-root instances, to cluster the
            terminals

    Returns:
        Solution topology S.

    Raises:
        NotImplementedError: the bounds ask for straight feeders, for
            something unequal turbine power rules out (see above), or for an
            odd exact feeder count of a root with rings.
        ValueError: the bounds ask for more exact rings than ``T // 2``, or no
            clustering satisfies them; LKH-3 finds no solution for a root, or
            none within the bounds or without rogue links, or none within the
            constraints' ``max_link_length``.
        optiwindnet.types.OWNWarmupFailed: the warm start is unusable.
    """
    R, T = A.graph['R'], A.graph['T']
    bounds, complete = vrp_bounds(
        'lkh3',
        A,
        constraints,
        power_rtol,
        complete=complete,
        repair=repair,
        counted_by="LKH-3's route-size bounds count nodes",
    )
    inflow_by_node = bounds.nonunit_inflow
    ringed = bounds.topology is Topology.RINGED
    balanced = bounds.balanced
    solve_capacity = bounds.group_capacity
    groups_lb, groups_ub = bounds.groups_lb, bounds.groups_ub
    fixed_above_min = bounds.fixed_above_min
    demands = [inflow_by_node.get(t, 1) for t in range(T)] if inflow_by_node else ()

    def _route_load(route: list[int]) -> int:
        """Return total terminal demand, using the terminal count for unitary inflow."""
        return sum(demands[i] for i in route) if demands else len(route)

    if seed is None:
        seed = random.randrange(0, 2**31)

    method_options = {
        'solver_name': 'LKH-3',
        'time_limit': time_limit,
        'scale': scale,
        'runs': runs,
        'per_run_limit': per_run_limit,
        'complete': complete,
        'feeders_above_min': (
            None if groups_ub is None else groups_ub - bounds.groups_min
        ),
        'feeders_exact': groups_lb == groups_ub and (balanced or fixed_above_min),
        'balanced': balanced,
        'ringed': ringed,
        'repair': repair,
        'max_retries': max_retries,
        'fun_fingerprint': _lkh3_fun_fingerprint,
    }
    solver_details_extra = {'seed': seed, **root_details(bounds)}

    A_iter = A.copy()
    # Without repair, A need not contain mesh diagonals.
    diagonals = A.graph['diagonals'].copy() if repair else A.graph.get('diagonals', {})
    A_iter.graph['diagonals'] = diagonals

    warm_routes_ = None
    if warmstart is not None:
        warm_routes_ = routes_from_warmstart(warmstart, R, T)
    terminals_, vehicles_, fleet_, pinned_ = _setup_clusters(
        A_iter, bounds, warm_routes_=warm_routes_
    )
    # A pinned root uses exactly its vehicles, which LKH-3 alone does not ensure
    # (it may leave a vehicle unused). Radial routes are filled up with slack
    # nodes (see _with_slack_nodes), as balanced as the routes can be. Rings
    # are held by LKH-3's route-size bounds, which TYPE=CVRP enforces; they are
    # balanced only under balance.
    capacity_ = [solve_capacity] * R
    num_slack_ = [0] * R
    route_sizes_: list[tuple[int, int] | None] = [None] * R
    for c, (terminals, vehicles_c, pinned) in enumerate(
        zip(terminals_, vehicles_, pinned_)
    ):
        T_c = len(terminals)
        if not pinned or T_c == 0:
            continue
        if ringed and not balanced:
            lo, hi = 2, solve_capacity
        else:
            lo, hi = T_c // vehicles_c, -(-T_c // vehicles_c)
            if not ringed:
                _, _, num_slack_[c] = balanced_capacity(T_c, solve_capacity, vehicles_c)
        route_sizes_[c] = (lo, hi)
        capacity_[c] = hi
    if warm_routes_ is None:
        warmstart_tours = [None] * R
    else:
        warmstart_tours = _initial_tours(
            warm_routes_, terminals_, vehicles_, num_slack_, capacity_
        )

    # Built once outside the retry loop. Rebuilt only after the crossings
    # branch (which mutates A_iter via remove_offending_crossings); the
    # over-capacity branch leaves A_iter intact and reuses the same matrices.
    L_ = _build_cluster_weight_matrices(
        A_iter,
        terminals_,
        scale=scale,
        complete=complete,
        precision=precision,
        link_length_ub=bounds.link_length_ub,
    )
    name = A_iter.graph.get('name', 'unnamed')

    def _solve_and_repair() -> tuple[nx.DiGraph, list[dict]]:
        outputs_ = _run_lkh_per_cluster(
            L_,
            name=name,
            capacity_=capacity_,
            time_limit=time_limit,
            vehicles_=vehicles_,
            warmstart_tours=warmstart_tours,
            balanced=balanced,
            demands=demands,
            scale=scale,
            runs=runs,
            per_run_limit=per_run_limit,
            precision=precision,
            seed=seed,
            ringed=ringed,
            num_slack_=num_slack_,
            route_sizes_=route_sizes_,
        )
        S = _build_solution(
            A_iter,
            capacity=bounds.capacity,
            outputs_=outputs_,
            terminals_=terminals_,
            keep_log=keep_log,
            method_options=method_options,
            solver_details_extra=solver_details_extra,
        )
        assert sum(S.nodes[r]['load'] for r in range(-R, 0)) == T, (
            'ERROR: root node load does not match T.'
        )
        return S, outputs_

    crossings: list = []
    over_capacity_clusters: list[int] = []
    i = 0
    if not repair:
        S, outputs_ = _solve_and_repair()
    else:
        while True:
            S, outputs_ = _solve_and_repair()
            S = repair_routeset_path(S, A_iter, ringed=ringed)
            crossings = S.graph.get('outstanding_crossings', [])
            over_capacity_clusters = [
                ic
                for ic, output in enumerate(outputs_)
                if max((_route_load(r) for r in output['routes']), default=0)
                > solve_capacity
            ]
            # a cluster at its fleet cap, or pinned, cannot get another vehicle
            bumpable = [
                ic
                for ic in over_capacity_clusters
                if not pinned_[ic]
                and ((fleet := fleet_[ic]) is None or vehicles_[ic] < fleet)
            ]
            if over_capacity_clusters:
                worst = max(
                    _route_load(route)
                    for ic in over_capacity_clusters
                    for route in outputs_[ic]['routes']
                )
                warn(
                    'Capacity violated in LKH solution: '
                    f'max_load ({worst}) > capacity ({solve_capacity}). '
                    'Retrying with increased vehicles.'
                )
            if (
                (not crossings and not over_capacity_clusters)
                or (over_capacity_clusters and not bumpable)
                or i == max_retries
            ):
                break
            i += 1
            if over_capacity_clusters:
                # Bump vehicles for the offending clusters and warmstart from S.
                # A_iter is not modified here, so L_ stays valid and the
                # warmstart tour does not refer to edges with the w_clip
                # sentinel weight.
                for ic in bumpable:
                    vehicles_[ic] += 1
                warmstart_tours = _initial_tours(
                    routes_from_warmstart(S, R, T),
                    terminals_,
                    vehicles_,
                    num_slack_,
                    capacity_,
                )
            else:
                # remove_offending_crossings shrinks A_iter; rebuild L_ so the
                # removed edges revert to the w_clip sentinel. Start cold (any
                # warmstart from S would now refer to removed edges).
                warmstart_tours = [None] * R
                remove_offending_crossings(A_iter, diagonals, crossings)
                L_ = _build_cluster_weight_matrices(
                    A_iter,
                    terminals_,
                    scale=scale,
                    complete=complete,
                    precision=precision,
                    link_length_ub=bounds.link_length_ub,
                )
    if i > 0:
        S.graph['retries'] = i
        if crossings or over_capacity_clusters:
            warn('Solution remains invalid (max_retries reached)')
    for r, output, terminals, vehicles_c, route_sizes in zip(
        range(-R, 0), outputs_, terminals_, vehicles_, route_sizes_
    ):
        if route_sizes is None:
            continue
        sizes = [len(route) for route in output['routes'] if route]
        lo, hi = route_sizes
        stray_slack = output.get('stray_slack', 0)
        flaws = []
        if output['penalty'] != 0:
            flaws.append(f'penalty {output["penalty"]}')
        if len(sizes) != vehicles_c:
            flaws.append(f'{len(sizes)} routes, not {vehicles_c}')
        if any(not lo <= size <= hi for size in sizes):
            flaws.append(f'route sizes {min(sizes)} to {max(sizes)}, not {lo} to {hi}')
        if stray_slack:
            flaws.append(f'{stray_slack} slack nodes off a route start')
        if flaws:
            raise ValueError(
                f'LKH-3 found no solution within the bounds for root {r} '
                f'({len(terminals)} terminals): {"; ".join(flaws)}. A longer '
                '`time_limit` may help.'
            )
    nx.set_node_attributes(S, inflow_by_node, 'inflow')
    if ringed:
        S.graph['topology'] = Topology.RINGED
        S = split_rings_and_calc_loads(S, A)
    else:
        S.graph['topology'] = Topology.RADIAL
        calcload(S)

    if not complete:
        rogue = [
            tuple(sorted((u, v)))
            for u, v in S.edges
            if u >= 0 and v >= 0 and not A.has_edge(u, v)
        ]
        if rogue:
            raise ValueError(
                f'LKH-3 solution uses rogue links absent from the mesh: {rogue}. A '
                'longer `time_limit` may help.'
            )
    over_limit = (
        []
        if bounds.link_length_ub is None
        else _links_over_limit(S, A, bounds.link_length_ub)
    )
    if over_limit:
        raise ValueError(
            f'LKH-3 found no solution within the link length limit: '
            f'{len(over_limit)} link(s) exceed it, e.g. {over_limit[0]}.'
        )

    if any(pinned_):
        S.graph['solver_details']['capacity_effective'] = (
            tuple(capacity_) if R > 1 else capacity_[0]
        )
    # Encode against the original link set, before repair removed any links.
    S.graph.update(linkset_identity(S, A), **bounds.power_attrs)
    return S


_lkh3_fun_fingerprint = fingerprint_function(lkh3)
