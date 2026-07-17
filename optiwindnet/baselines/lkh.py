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
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform

from ..clustering import clusterize
from ..interarraylib import add_link_blockmap, fun_fingerprint, ringify_S
from ..repair import repair_routeset_path
from ._core import (
    add_branches_to_S,
    clamp_vehicles_to_min,
    remove_offending_crossings,
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
            means the input graph is not normalized (call :func:`as_normalized`
            before solving), or that ``scale`` is too large for the coordinate
            magnitudes.
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
    d2root_scaled = np.round(A.graph['d2roots'][terminals, root_col] * scale)
    _check(float(d2root_scaled.max(initial=0.0)), 'depot distance')

    if complete:
        VertexC = A.graph['VertexC']
        coords = np.vstack([VertexC[terminals], VertexC[root].reshape(1, -1)])
        pd_scaled = np.round(pdist(coords) * scale)
        _check(float(pd_scaled.max(initial=0.0)), 'pairwise distance')
        L = squareform(pd_scaled.astype(np.int32))
    else:
        L = np.full((T_c + 1, T_c + 1), w_clip, dtype=np.int32)

    i_from_n = {n: i for i, n in enumerate(terminals)}
    for u, v, length in A.edges(data='length'):
        iu = i_from_n.get(u)
        iv = i_from_n.get(v)
        if iu is not None and iv is not None:
            scaled = round(length * scale)
            _check(scaled, 'edge length')
            L[iu, iv] = L[iv, iu] = scaled
    L[:-1, -1] = d2root_scaled.astype(np.int32)
    return L


def _distance_cap(L: np.ndarray, *, capacity: int, scale: float) -> float:
    """Maximum route length to impose on LKH (in ``L``'s scaled units).

    A route visits at most ``k = min(capacity, T)`` terminals, so its length is
    the feeder leg plus ``k - 1`` inter-terminal hops. Normalized coordinates
    put the site's concave hull at unit area, hence the terminal spacing scales
    as ``1/sqrt(T)`` and the hop total as ``k/sqrt(T)``. The feeder leg is
    bounded by the longest depot distance, read straight off ``L``'s depot
    column.

    The coefficients were calibrated against 25453 known-good routesets
    (HGS-CVRP and LKH-3, both path-shaped) from the bundled instance database:
    ``0.45`` minimizes the spread of longest-route/predictor (cv 0.063, against
    0.225 for the previous capacity-blind cap), and ``1.72`` is the smallest
    margin that still leaves every one of those routesets feasible, with 1.25x
    to spare on the worst of them.
    """
    T = L.shape[0] - 1
    d2root_max = float(L[:-1, -1].max())
    hops = scale * 0.45 * min(capacity, T) / math.sqrt(T)
    return 1.72 * (d2root_max + hops)


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
    hops = sum(int(L[u, v]) for u, v in zip(seq[:-1], seq[1:]))
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
    time_limit: float,
    scale: float,
    runs: int,
    per_run_limit: float,
    precision: int,
    seed: int,
    initial_tour_nodes: list[int] | None,
    name: str,
    ringed: bool = False,
) -> dict:
    """Run LKH-3 on a precomputed weight matrix.

    ``L`` has shape ``(T+1, T+1)`` with the depot at the last index.

    With ``ringed=False`` (default) LKH solves an Open-CVRP (``TYPE=OVRP``): the
    return leg to the depot is free and routes are radial. With ``ringed=True``
    it solves the closed CVRP (``TYPE=CVRP``): every route returns to the depot,
    so the symmetric weight matrix charges both feeder legs and each route is a
    ring (closed loop).

    Returns a dict containing routes (list of lists of 0-based terminal indices
    in the matrix), penalty, minimum, log, ``elapsed_time``, ``solution_time``, plus
    parsed run statistics.
    """
    T = L.shape[0] - 1
    N = T + 1
    edge_weights = '\n'.join(
        ' '.join(str(d) for d in row[i + 1 :]) for i, row in enumerate(L[:-1])
    )

    problem_fname = 'problem.txt'
    params_fname = 'params.txt'
    output_fname = 'solution.out'
    tour_fname = 'solution.tour'
    initial_tour_fname = 'initial.tour'

    distance_cap = _distance_cap(L, capacity=capacity, scale=scale)
    specs: dict[str, str | int | float] = dict(
        NAME=name,
        TYPE='CVRP' if ringed else 'OVRP',
        DIMENSION=N,  # CVRP number of nodes and depots
        # For CAPACITY to be enforced, a DEMAND section is required.
        # MTSP_MAX_SIZE should work for unitary demand, but did not.
        CAPACITY=capacity,
        EDGE_WEIGHT_TYPE='EXPLICIT',
        EDGE_WEIGHT_FORMAT='UPPER_ROW',
    )
    if not ringed and not math.isinf(distance_cap):
        # LKH treats DISTANCE as a hard constraint: too low and it reports the
        # problem infeasible and returns no solution at all. The cap is
        # calibrated for open (radial) routes; a closed ring pays a second feeder
        # leg, so the cap is not imposed for ringed solves.
        specs['DISTANCE'] = distance_cap  # maximum route length
    data = dict(
        EDGE_WEIGHT_SECTION=edge_weights,
        DEMAND_SECTION='\n'.join(chain((f'{i + 1} 1' for i in range(T)), (f'{N} 0',))),
    )
    params = dict(
        # SPECIAL is a shorthand for a bundle of large-neighborhood move settings
        # (MOVE_TYPE='5 SPECIAL', MAX_SWAPS=0, ...). It segfaults LKH-3 on
        # TYPE=CVRP (the ringed solve), so it is only enabled for the OVRP case.
        **({} if ringed else {'SPECIAL': None}),  # None -> output only the key
        DEPOT=N,
        SEED=seed,  # 0 means pick a random seed
        PRECISION=precision,  # d[i][j] = PRECISION*c[i][j] + pi[i] + pi[j]
        TOTAL_TIME_LIMIT=time_limit,
        TIME_LIMIT=per_run_limit,
        RUNS=runs,  # default: 10
        # MAX_TRIALS=100,  # default: number of nodes (DIMENSION)
        # TRACE_LEVEL=1,  # default is 1, 0 supresses output
        #  INITIAL_TOUR_ALGORITHM='GREEDY',  # { … | CVRP | MTSP | SOP } Default: WALK
        VEHICLES=vehicles,
        # FIXME: if TYPE=OVRP, LHK-3 does not apply penalties for MTSP_MIN_SIZE:
        #   MTSP_MIN_SIZE is enforced for TYPE=CVRP, but then an assymetric
        #   EDGE_WEIGHT_FORMAT='FULL_MATRIX' is required for open routes
        #   Notably, balanced=True is NOT enforced with the current parameters
        MTSP_MIN_SIZE=min_route_size,
        MTSP_MAX_SIZE=capacity,
        MTSP_OBJECTIVE='MINSUM',  # [ MINMAX | MINMAX_SIZE | MINSUM ]
        MTSP_SOLUTION_FILE=output_fname,
        # LKH-3 only writes MTSP_SOLUTION_FILE for 2+ salesmen. With a single
        # vehicle it solves the problem but reports it as a plain tour, so ask
        # for TOUR_FILE too and recover the lone route from it (see below).
        TOUR_FILE=tour_fname,
        #  MOVE_TYPE='5 SPECIAL',  # <integer> [ SPECIAL ]
        #  GAIN23='NO',
        #  KICKS=1,
        #  KICK_TYPE=4,
        #  MAX_SWAPS=0,
        #  POPULATION_SIZE=12,  # default 10
        #  PATCHING_A=
        #  PATCHING_C=
    )

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
        result = subprocess.run(['LKH', params_fpath], capture_output=True)
        elapsed_time = time.perf_counter() - start_time
        output_fpath = os.path.join(tmpdir, output_fname)
        tour_fpath = os.path.join(tmpdir, tour_fname)
        solution_parsed = Path(output_fpath).is_file()
        if solution_parsed:
            with open(output_fpath, 'r') as f_sol:
                penalty, minimum = next(f_sol).split(':')[-1][:-1].split('_')
                next(f_sol)  # discard second line
                routes = [
                    [int(node) - 1 for node in line.split(' ')[1:-5]] for line in f_sol
                ]
        elif Path(tour_fpath).is_file():
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
    output = dict(
        routes=routes,
        penalty=int(penalty),
        minimum=minimum,
        cost=float(minimum) / scale,
        log=log,
        stderr=result.stderr.decode('utf8'),
        elapsed_time=elapsed_time,
        solution_time=_solution_time(log, minimum),
        vehicles=vehicles,
        seed=seed,
    )

    if not solution_parsed or result.stderr:
        info('===stdout===\n%s', log)
        error('===stderr===\n%s', output['stderr'])
        return output

    tail = result.stdout[result.stdout.rfind(b'Successes/') :].decode()
    entries = iter(tail.splitlines())
    next(entries)  # skip successes line
    output['cost_extrema'] = tuple(
        float(v)
        for v in re.match(
            r'Cost\.min = (-?\d+), Cost\.avg = -?\d+\.?\d*,'
            r' Cost\.max = -?(\d+)',
            next(entries),
        ).groups()
    )
    next(entries)  # skip gap line
    output['penalty_extrema'] = tuple(
        float(v)
        for v in re.match(
            r'Penalty\.min = (\d+), Penalty\.avg = \d+\.?\d*,'
            r' Penalty\.max = (\d+)',
            next(entries),
        ).groups()
    )
    output['trials_extrema'] = tuple(
        float(v)
        for v in re.match(
            r'Trials\.min = (\d+), Trials\.avg = \d+\.?\d*,'
            r' Trials\.max = (\d+)',
            next(entries),
        ).groups()
    )
    output['runtime_extrema'] = tuple(
        float(v)
        for v in re.match(
            r'Time\.min = (\d+\.?\d*) sec., Time\.avg = \d+\.?\d* sec.,'
            r' Time\.max = (\d+\.?\d*) sec.',
            next(entries),
        ).groups()
    )
    return output


def _build_cluster_weight_matrices(
    A: nx.Graph,
    terminals_: list[list[int]],
    *,
    scale: float,
    complete: bool,
    precision: int,
) -> list[np.ndarray]:
    """Build one LKH weight matrix per cluster.

    Computes the ``w_clip`` sentinel (used for missing/clipped edges) from
    ``precision`` once, then calls :func:`_build_weight_matrix` for each
    (terminals, root) pair in root order (-R..-1).
    """
    R = A.graph['R']
    w_clip = np.iinfo(np.int32).max // (2 * precision)
    return [
        _build_weight_matrix(
            A, terminals, r, scale=scale, complete=complete, w_clip=w_clip
        )
        for r, terminals in zip(range(-R, 0), terminals_)
    ]


def _solve_cluster(
    L: np.ndarray,
    *,
    capacity: int,
    vehicles: int,
    balanced: bool,
    time_limit: float,
    scale: float,
    runs: int,
    per_run_limit: float,
    precision: int,
    seed: int,
    initial_tour_nodes: list[int] | None,
    name: str,
    ringed: bool = False,
) -> dict:
    """Run LKH-3 on a pre-built cluster weight matrix.

    ``L`` is shape ``(T_c+1, T_c+1)`` with the depot at the last index. Derives
    ``min_route_size`` from ``vehicles``/``capacity``/``balanced`` and dispatches to
    :func:`_do_lkh`. The matrix is built by :func:`_build_cluster_weight_matrices`,
    decoupled from this call so it can be reused across iterations that do
    not mutate the underlying graph.
    """
    T_c = L.shape[0] - 1
    if ringed:
        # MTSP_MIN_SIZE for TYPE=CVRP would require a FULL_MATRIX (asymmetric)
        # formulation; the symmetric ring solve does not impose a minimum size.
        min_route_size = 0
    elif balanced:
        min_route_size = T_c // vehicles
    elif vehicles == math.ceil(T_c / capacity):
        min_route_size = (T_c % capacity) or capacity
    else:
        min_route_size = 0
    return _do_lkh(
        L,
        capacity=capacity,
        vehicles=vehicles,
        min_route_size=min_route_size,
        time_limit=time_limit,
        scale=scale,
        runs=runs,
        per_run_limit=per_run_limit,
        precision=precision,
        seed=seed,
        initial_tour_nodes=initial_tour_nodes,
        name=name,
        ringed=ringed,
    )


def _initial_tours_from_warmstart(
    warmstart: nx.Graph,
    terminals_: list[list[int]],
    vehicles_: list[int],
) -> list[list[int] | None]:
    """Per-root LKH initial tours derived from a warmstart solution graph.

    For each root, walks the warmstart's branches in order. Each visited
    terminal ``n`` becomes the LKH customer id ``i + 1``, where ``i`` is ``n``'s
    position in the cluster's sorted ``terminals`` list (i.e., its row index
    in the LKH weight matrix). The depot clones (``vehicles - 1`` of them)
    and the final depot id are appended, as required by LKH-3 for OVRP.

    The walked tour is purely a hint about the *order* in which customers
    should be visited; LKH evaluates segment costs from the weight matrix.
    Therefore the warmstart's edges should still be present in ``A_iter``
    when this tour is fed back to LKH (or those segments will be charged
    the ``w_clip`` sentinel weight).

    Roots whose warmstart cluster is empty get ``None``.

    Raises:
        KeyError: a walked terminal is not in the corresponding cluster's
            ``terminals`` list (i.e., warmstart and clustering disagree).
    """
    R = warmstart.graph['R']
    out: list[list[int] | None] = []
    for r, terminals, vehicles in zip(range(-R, 0), terminals_, vehicles_):
        idx_from_node = {n: i + 1 for i, n in enumerate(terminals)}
        ordered_ids: list[int] = []
        for cur in warmstart.neighbors(r):
            rev = r
            while True:
                ordered_ids.append(idx_from_node[cur])
                nb = warmstart[cur]
                if len(nb) == 1:
                    break  # leaf
                a, b = nb
                rev, cur = cur, a if b == rev else b
        if not ordered_ids:
            out.append(None)
            continue
        T_c = len(terminals)
        depot_clones = range(T_c + 2, T_c + vehicles + 1)
        out.append(ordered_ids + list(depot_clones) + [T_c + 1])
    return out


def _build_solution(
    A: nx.Graph,
    *,
    capacity: int,
    outputs_: list[dict],
    terminals_: list[list[int]],
    keep_log: bool,
    method_options: dict,
    solver_details_extra: dict,
) -> nx.Graph:
    """Assemble the final solution graph S from per-root LKH outputs."""
    R, T = A.graph['R'], A.graph['T']
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
        # path-based repair machinery applies; ringify_S() closes them afterwards
        sub_max_load, subtree_id = add_branches_to_S(
            S, subtrees, root=r, subtree_id_start=subtree_id
        )
        max_load = max(max_load, sub_max_load)
        root_load = sum(S.nodes[n]['load'] for n in S.neighbors(r))
        S.nodes[r]['load'] = root_load

    S.graph['max_load'] = max_load
    return S


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
) -> nx.Graph:
    """Low-level single-root Lin-Kernighan-Helsgaun (LKH-3) solver.

    Open Capacitated Vehicle Routing Problem on a single depot. ``A`` must be
    normalized (use :func:`as_normalized` before calling) and have R == 1. For
    multi-root instances, use :func:`lkh3` instead.

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

    vehicles_min = math.ceil(T / capacity)
    if (vehicles is None) or (vehicles <= vehicles_min):
        if vehicles is not None and vehicles < vehicles_min:
            warn(
                f'Vehicle number ({vehicles}) too low for feasibilty '
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

    method_options = dict(
        solver_name='LKH-3',
        time_limit=time_limit,
        scale=scale,
        runs=runs,
        per_run_limit=per_run_limit,
        complete=complete,
        fun_fingerprint=_lkh_fun_fingerprint,
    )
    S = _build_solution(
        A,
        capacity=capacity,
        outputs_=[output],
        terminals_=[terminals],
        keep_log=keep_log,
        method_options=method_options,
        solver_details_extra=dict(seed=seed),
    )
    assert S.nodes[-1]['load'] == T, 'ERROR: root node load does not match T.'
    S.graph['has_loads'] = True
    return S


_lkh_fun_fingerprint = fun_fingerprint(_lkh)


def _no_terminals_output(seed: int) -> dict:
    """Stand-in for :func:`_do_lkh`'s output on a root that got no terminals."""
    return dict(
        routes=[],
        penalty=0,
        minimum='0',
        cost=0.0,
        log='',
        stderr='',
        elapsed_time=0.0,
        solution_time=0.0,
        vehicles=0,
        seed=seed,
    )


def _run_lkh_per_cluster(
    L_: list[np.ndarray],
    *,
    name: str,
    capacity: int,
    time_limit: float,
    vehicles_: list[int],
    warmstart_tours: list[list[int] | None],
    balanced: bool,
    scale: float,
    runs: int,
    per_run_limit: float,
    precision: int,
    seed: int,
    ringed: bool = False,
) -> list[dict]:
    """Solve every root cluster with LKH-3, sequentially or in parallel.

    Single-root (R == 1) is solved synchronously; multi-root dispatches one
    :func:`_solve_cluster` per root through a ThreadPoolExecutor (one thread per
    root). Returns one LKH output dict per root, in root order (-R..-1).

    A root with an empty cluster is not dispatched (LKH rejects the resulting 1x1
    weight matrix) but still gets an entry, so that root ids stay aligned with
    cluster indices.
    """
    R = len(L_)
    job_kwargs_ = [
        dict(
            L=L,
            capacity=capacity,
            vehicles=vehicles_c,
            balanced=balanced,
            time_limit=time_limit,
            scale=scale,
            runs=runs,
            per_run_limit=per_run_limit,
            precision=precision,
            seed=seed,
            initial_tour_nodes=init_tour,
            name=name if R == 1 else f'{name}_root{r}',
            ringed=ringed,
        )
        for r, L, vehicles_c, init_tour in zip(
            range(-R, 0), L_, vehicles_, warmstart_tours
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
    A: nx.Graph, *, capacity: int, vehicles: int | None
) -> tuple[list[list[int]], list[int]]:
    """Compute per-root terminals and vehicle counts.

    For R == 1 the only cluster is ``range(T)``; for R > 1 the terminals are
    partitioned by :func:`clusterize` and each cluster's terminal list is sorted
    so that LKH customer ids ``[1..T_c]`` correspond to the cluster's nodes in
    sorted order (and :func:`_initial_tours_from_warmstart` agrees on indexing).

    Clusters are given their minimum feasible vehicle count, except:

    - a cluster that fits within ``capacity`` (``T_c <= capacity``) gets
      :func:`_vehicles_within_capacity` instead of the minimum of 1. Its minimum
      is one only because the capacity constraint is inactive, and that same spare
      capacity is what makes extra feeders harmless: any routes LKH returns can be
      joined into a single subtree at the root without exceeding capacity, so the
      layout stays radial and no feeder budget is overspent. Pinning it to 1
      instead demands a single route, which over the sparse (near-planar) link set
      ``A`` means a Hamiltonian path — and one need not exist, all the more so
      after :func:`_prune_links` drops the links that would block feeders (a
      rationale that is vacuous when there is only one feeder to block). LKH then
      returns nothing at all for the cluster.
    - for R == 1, a user-supplied ``vehicles > vehicles_min`` is honoured (this
      knob is meaningless under multi-root clustering).
    """
    R, T = A.graph['R'], A.graph['T']
    if R == 1:
        terminals_ = [list(range(T))]
        len_cluster_ = [T]
    else:
        cluster_ = clusterize(A, capacity)
        terminals_ = [sorted(c) for c in cluster_]
        len_cluster_ = [len(c) for c in terminals_]
    vehicles_min_ = [math.ceil(n / capacity) for n in len_cluster_]
    if R == 1 and vehicles is not None and vehicles > vehicles_min_[0]:
        vehicles_ = [vehicles]
    else:
        vehicles_ = [
            _vehicles_within_capacity(T_c) if v_min == 1 else v_min
            for T_c, v_min in zip(len_cluster_, vehicles_min_)
        ]
    return terminals_, vehicles_


def lkh3(
    A: nx.Graph,
    *,
    capacity: int,
    time_limit: float,
    vehicles: int | None = None,
    seed: int | None = None,
    keep_log: bool = False,
    repair: bool = True,
    max_retries: int = 10,
    balanced: bool = False,
    scale: float = 1e5,
    runs: int = 50,
    per_run_limit: float = 15.0,
    precision: int = 1000,
    complete: bool = False,
    ringed: bool = False,
    warmstart: nx.Graph | None = None,
) -> nx.Graph:
    """Solve the O/CVRP using LKH-3 with links from ``A``.

    Wraps the LKH-3 executable, which is not distributed with OptiWindNet.
    Get it from http://akira.ruc.dk/~keld/research/LKH-3/ and make sure the
    ``LKH`` executable is in the environment's PATH.

    Uses the Lin-Kernighan-Helsgaun meta-heuristic to solve an Open-CVRP
    (i.e., vehicles do not return to the depot), yielding radial layouts. With
    ``ringed=True`` it solves the closed CVRP instead (``TYPE=CVRP``): every
    route returns to the depot, forming a ring whose capacity is doubled
    internally (``2 * capacity``) so each of the two arms holds at most
    ``capacity`` terminals. Normalization of the input graph is recommended
    before calling this function (use :func:`as_normalized`).

    For single-root problems, the solver runs on the full graph. For multi-root
    problems, the graph is clustered (one cluster per root) and each cluster is
    solved concurrently.

    For multi-root instances, the vehicles (feeders) parameter is forced per
    cluster (a warning is issued if a different value is requested): to the
    minimum feasible value, except for a cluster that fits within ``capacity``,
    which is offered enough feeders to use as many as lower the cable length.

    If ``repair=True`` (the default), the solution is iteratively repaired
    until no crossings remain (or ``max_retries`` is reached). This may cause
    the actual runtime to be up to ``(max_retries + 1)`` times the given
    ``time_limit``.

    Args:
        A: graph with allowed edges (if it has 0 edges, use ``complete=True``).
        capacity: maximum vehicle capacity.
        time_limit: [s] solver run time limit (per cluster).
        vehicles: number of vehicles (if None or at the minimum, use the
            per-cluster default described above; ignored for multi-root
            problems).
        seed: random seed for reproducibility (if None, picks a random one).
        keep_log: attach solver log to the solution graph.
        repair: iteratively fix crossings (default True).
        max_retries: maximum repair iterations.
        balanced: currently not implemented for this solver.
        scale: factor to scale lengths (LKH manual).
        runs: number of LKH runs (LKH manual).
        per_run_limit: [s] LKH per-run time limit.
        precision: LKH precision parameter.
        complete: make the full graph over A available (missing edges assumed
            direct).
        warmstart: optional previous solution graph used to seed the initial
            tour. For multi-root instances each cluster receives the portion of
            the warmstart attached to its root.

    Returns:
        Solution topology S.
    """
    R, T = A.graph['R'], A.graph['T']
    # a ring holds up to 2*capacity terminals (two arms of `capacity` each)
    solve_capacity = 2 * capacity if ringed else capacity
    if vehicles is not None:
        vehicles_min = math.ceil(T / solve_capacity)
        if vehicles != vehicles_min:
            if R > 1:
                warn(
                    'For multi-root instances, the parameter vehicles (feeders) can '
                    'only be None or the minimum feasible: setting to the minimum.'
                )
        vehicles = clamp_vehicles_to_min(vehicles, vehicles_min, capacity)
        feeders_above_min = vehicles - vehicles_min
    else:
        feeders_above_min = None

    if seed is None:
        seed = random.randrange(0, 2**31)

    method_options = dict(
        solver_name='LKH-3',
        time_limit=time_limit,
        scale=scale,
        runs=runs,
        per_run_limit=per_run_limit,
        complete=complete,
        feeders_above_min=feeders_above_min,
        fun_fingerprint=_lkh3_fun_fingerprint,
    )
    solver_details_extra = dict(seed=seed)

    A_iter = A.copy()
    diagonals = A.graph['diagonals'].copy()
    A_iter.graph['diagonals'] = diagonals
    # for clustering() and _prune_links() to index d2roots (-R offset is indifferent):
    if R > 1:
        A_iter.graph['closest_root'] = -R + A.graph['d2roots'][:T].argmin(axis=1)
    else:
        A_iter.graph['closest_root'] = np.full((T,), -1, dtype=np.int_)
    add_link_blockmap(A_iter)
    _prune_links(A_iter, math.ceil(2.4 * solve_capacity))

    terminals_, vehicles_ = _setup_clusters(
        A_iter, capacity=solve_capacity, vehicles=vehicles
    )
    if warmstart is not None:
        warmstart_tours = _initial_tours_from_warmstart(
            warmstart, terminals_, vehicles_
        )
    else:
        warmstart_tours = [None] * R

    # Built once outside the retry loop. Rebuilt only after the crossings
    # branch (which mutates A_iter via remove_offending_crossings); the
    # over-capacity branch leaves A_iter intact and reuses the same matrices.
    L_ = _build_cluster_weight_matrices(
        A_iter, terminals_, scale=scale, complete=complete, precision=precision
    )
    name = A_iter.graph.get('name', 'unnamed')

    def _solve_and_repair() -> tuple[nx.Graph, list[dict]]:
        outputs_ = _run_lkh_per_cluster(
            L_,
            name=name,
            capacity=solve_capacity,
            time_limit=time_limit,
            vehicles_=vehicles_,
            warmstart_tours=warmstart_tours,
            balanced=balanced,
            scale=scale,
            runs=runs,
            per_run_limit=per_run_limit,
            precision=precision,
            seed=seed,
            ringed=ringed,
        )
        S = _build_solution(
            A_iter,
            capacity=capacity,
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
        S, _ = _solve_and_repair()
    else:
        while True:
            S, outputs_ = _solve_and_repair()
            S = repair_routeset_path(S, A_iter, ringed=ringed)
            crossings = S.graph.get('outstanding_crossings', [])
            over_capacity_clusters = [
                ic
                for ic, output in enumerate(outputs_)
                if max((len(r) for r in output['routes']), default=0) > solve_capacity
            ]
            if over_capacity_clusters:
                warn(
                    'Capacity violated in LKH solution: '
                    f'max_load ({S.graph["max_load"]}) > capacity ({solve_capacity}). '
                    'Retrying with increased vehicles.'
                )
            if (not crossings and not over_capacity_clusters) or i == max_retries:
                break
            i += 1
            if over_capacity_clusters:
                # Bump vehicles for the offending clusters and warmstart from S.
                # A_iter is not modified here, so L_ stays valid and the
                # warmstart tour does not refer to edges with the w_clip
                # sentinel weight.
                for ic in over_capacity_clusters:
                    vehicles_[ic] += 1
                warmstart_tours = _initial_tours_from_warmstart(
                    S, terminals_, vehicles_
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
                )
    if i > 0:
        S.graph['retries'] = i
        if crossings or over_capacity_clusters:
            warn('Solution remains invalid (max_retries reached)')
    if ringed:
        # routes were built (and repaired) as open paths: close them into rings
        S.graph['topology'] = 'ringed'
        ringify_S(S, A)  # also sets 'has_loads'
    else:
        S.graph['topology'] = 'radial'
        S.graph['has_loads'] = True
    return S


_lkh3_fun_fingerprint = fun_fingerprint(lkh3)
