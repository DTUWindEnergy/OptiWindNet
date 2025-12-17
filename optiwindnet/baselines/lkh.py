# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

from itertools import chain
import logging
import math
import os
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform

from ..interarraylib import fun_fingerprint
from ..repair import repair_routeset_path
from ..geometric import angle_helpers

_lggr = logging.getLogger(__name__)
debug, info, warn, error = _lggr.debug, _lggr.info, _lggr.warning, _lggr.error


def _add_link_blockage(A: nx.Graph):
    """Experimental. Add edge attribute 'num_blocked'.

    Changes A in place. Assesses the number of terminals whose line-of-sight
    to the root is intersected by the edge.

    Only implemented for single-root sites.
    """
    VertexC = A.graph['VertexC']
    angle_, angle_rank_, _ = angle_helpers(A, include_borders=False)
    A.graph['angle_'] = angle_
    A.graph['angle_rank_'] = angle_rank_
    for u, v, edgeD in A.edges(data=True):
        if u < 0 or v < 0:
            edgeD['num_blocked'] = [0]
            continue
        uR, vR = angle_rank_[[u, v], 0].tolist()
        uv_angle = (angle_[v] - angle_[u]).item()
        if uv_angle < 0:
            uR, vR = vR, uR
        if abs(uv_angle) <= np.pi:
            inside_wedge = np.flatnonzero((uR < angle_rank_) & (angle_rank_ < vR))
        else:
            inside_wedge = np.flatnonzero((angle_rank_ < uR) | (vR < angle_rank_))
        if len(inside_wedge) > 0:
            vec = VertexC[v] - VertexC[u]
            wedge_vec_ = VertexC[inside_wedge] - VertexC[u]
            cross = wedge_vec_[:, 0] * vec[1] - wedge_vec_[:, 1] * vec[0]
            # ¡assuming a single root!
            root_vec = VertexC[-1] - VertexC[u]
            is_root_sign_pos = (root_vec[0] * vec[1] - root_vec[1] * vec[0]) > 0
            # ¡assuming a single root!
            edgeD['num_blocked'] = [
                sum((cross <= 0) if is_root_sign_pos else (cross >= 0)).item()
            ]
        else:
            edgeD['num_blocked'] = [0]


#  def prune_bad_links(A: nx.Graph, blockage_link_feeder_lim: float = 3.0):
def _prune_bad_links(A: nx.Graph, max_blockable_per_link: int):
    # remove links that have negative savings both ways from the start
    #  max_blockable_by_link = blockage_link_feeder_lim * capacity
    d2roots = A.graph['d2roots']
    unfeas_links = []
    for u, v, edgeD in A.edges(data=True):
        extent = edgeD['length']
        root = A.nodes[v]['root']
        if (
            extent > d2roots[u, A.nodes[u]['root']]
            and extent > d2roots[v, A.nodes[v]['root']]
        ) or (
            edgeD['num_blocked'][root] > max_blockable_per_link
            #  and edgeD['cos_'][root] < blockage_link_cos_lim
        ):
            unfeas_links.append((u, v) if u < v else (v, u))
    debug('links removed in pre-processing: %s', unfeas_links)
    A.remove_edges_from(unfeas_links)
    diagonals = A.graph['diagonals']
    for link in unfeas_links:
        if link in diagonals:
            del diagonals[link]


def lkh(
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
) -> nx.Graph:
    """
    Lin-Kernighan-Helsgaun via LKH-3 binary.
    Open Capacitated Vehicle Routing Problem.

    A must be normalized - use as_normalized() before calling this.

    http://akira.ruc.dk/~keld/research/LKH-3/

    Args:
      A: graph with allowed edges (if it has 0 edges, use complete graph)
      capacity: maximum vehicle capacity
      time_limit: [s] solver run time limit
      scale: factor to scale lengths (should be < 1e6)
      vehicles: number of vehicles (if None, use the minimum feasible)
      runs: consult LKH manual
      per_run_limit: [s] consult LKH manual
      precision: consult LKH manual
      complete: make the full graph over A available (links not in A assumed direct)
      keep_log: save the LKH text output to graph attr 'method_log'
      seed: for the pseudo-random number generator (if None: random seed)
    Returns:
      Solution topology S
    """
    # Notes for tinkering with this wrapper:
    # LKH offers the option to define the available edges set from a sparse graph:
    # EDGE_DATA_SECTION, along with EDGE_DATA_FORMAT={ADJ_LIST, EDGE_LIST}. This is
    # very hard to use with TYPE set to ACVRP or OVRP. It seems like LKH expects the
    # edges to be defined wrt the transformed problem, but no reference is available
    # as to how that transformation is carried out (and the C code lacks comments).
    #
    # An option to define the candidate set of edges is alternatively available through
    # EDGE_FILE. Besides applying to the transformed problem, it follows the Concorde
    # format, which has indices starting at 0.

    R, T = A.graph['R'], A.graph['T']
    assert R == 1, 'LKH allows only 1 depot'
    # problem dimension
    N = R + T
    problem_fname = 'problem.txt'
    params_fname = 'params.txt'

    # this is the same expression that LKH uses to define the big-M value
    w_clip = np.iinfo(np.int32).max // (2 * precision)

    # build weight matrix (only the upper triangle is passed)
    if complete:
        # Note: this is pure Euclidian 2D distance, neglecting contours
        VertexC = A.graph['VertexC']
        VertexCmod = np.vstack((VertexC[:T], VertexC[-R:]))
        L = squareform(np.round(pdist(VertexCmod) * scale).astype(np.int32))
    else:
        L = np.full((N, N), w_clip, dtype=np.int32)
    # this overwrites the direct distances (if using the complete graph)
    for u, v, length in A.edges(data='length'):
        L[u, v] = L[v, u] = round(length * scale)
    L[:-1, -1] = np.round(A.graph['d2roots'][:T, -1] * scale)
    edge_weights = '\n'.join(
        ' '.join(str(d) for d in row[i + 1 :]) for i, row in enumerate(L[:-1])
    )

    output_fname = 'solution.out'
    specs = dict(
        NAME=A.graph.get('name', 'unnamed'),
        TYPE='OVRP',
        DIMENSION=N,  # CVRP number of nodes and depots
        # For CAPACITY to be enforced, a DEMAND section is required.
        # MTSP_MAX_SIZE should work for unitary demand, but did not.
        CAPACITY=capacity,
        # This expression for limiting DISTANCE was empiricaly obtained.
        # It could be improved by replacing T with the aspect ratio of the border shape
        DISTANCE=scale * 16.0 / (math.log(T) + 0.1),  # maximum route length
        EDGE_WEIGHT_TYPE='EXPLICIT',
        EDGE_WEIGHT_FORMAT='UPPER_ROW',
    )
    data = dict(
        EDGE_WEIGHT_SECTION=edge_weights,
        DEMAND_SECTION='\n'.join(chain((f'{i + 1} 1' for i in range(T)), (f'{N} 0',))),
    )

    vehicles_min = math.ceil(T / capacity)
    if (vehicles is None) or (vehicles <= vehicles_min):
        # set to minimum feasible vehicle number
        if vehicles is not None and vehicles < vehicles_min:
            warn(
                f'Vehicle number ({vehicles}) too low for feasibilty '
                f'with capacity ({capacity}). Setting to {vehicles_min}.'
            )
        vehicles = vehicles_min
        min_route_size = (T % capacity) or capacity
    else:
        min_route_size = 0
    if seed is None:
        seed = 0
    params = dict(
        SPECIAL=None,  # None -> output only the key
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
        MTSP_MIN_SIZE=min_route_size,
        MTSP_MAX_SIZE=capacity,
        MTSP_OBJECTIVE='MINSUM',  # [ MINMAX | MINMAX_SIZE | MINSUM ]
        MTSP_SOLUTION_FILE=output_fname,
        #  MOVE_TYPE='5 SPECIAL',  # <integer> [ SPECIAL ]
        #  GAIN23='NO',
        #  KICKS=1,
        #  KICK_TYPE=4,
        #  MAX_SWAPS=0,
        #  POPULATION_SIZE=12,  # default 10
        #  PATCHING_A=
        #  PATCHING_C=
    )

    # run LKH
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
        Path(params_fpath).write_text(
            '\n'.join((f'{k} = {v}' if v is not None else k) for k, v in params.items())
        )
        start_time = time.perf_counter()
        result = subprocess.run(['LKH', params_fpath], capture_output=True)
        elapsed_time = time.perf_counter() - start_time
        output_fpath = os.path.join(tmpdir, output_fname)
        if Path(output_fpath).is_file():
            with open(output_fpath, 'r') as f_sol:
                penalty, minimum = next(f_sol).split(':')[-1][:-1].split('_')
                next(f_sol)  # discard second line
                branches = [
                    [int(node) - 1 for node in line.split(' ')[1:-5]] for line in f_sol
                ]
        else:
            penalty = 0
            minimum = 'inf'
            branches = []

    # create a topology graph S from the results
    log = result.stdout.decode('utf8')
    S = nx.Graph(
        T=T,
        R=R,
        capacity=capacity,
        has_loads=True,
        objective=float(minimum) / scale,
        creator='baselines.lkh',
        runtime=elapsed_time,
        solution_time=_solution_time(log, minimum),
        method_options=dict(
            solver_name='LKH-3',
            time_limit=time_limit,
            scale=scale,
            runs=runs,
            per_run_limit=per_run_limit,
            complete=complete,
            fun_fingerprint=_lkh_fun_fingerprint,
        ),
        solver_details=dict(
            penalty=int(penalty),
            vehicles=vehicles,
            seed=seed,
        ),
    )
    # ensure roots are added, even if some are not connected
    S.add_nodes_from(range(-R, 0))
    solver_details = S.graph['solver_details']
    if keep_log:
        S.graph['method_log'] = log
    if vehicles is not None:
        solver_details.update(vehicles=vehicles)

    if not penalty or result.stderr:
        info('===stdout===\n%s', result.stdout.decode('utf8'))
        error('===stderr===\n%s', result.stderr.decode('utf8'))
    else:
        tail = result.stdout[result.stdout.rfind(b'Successes/') :].decode()
        entries = iter(tail.splitlines())
        # Decision to drop avg. stats: unreliable, possibly due to time_limit
        next(entries)  # skip sucesses line
        solver_details['cost_extrema'] = tuple(
            float(v)
            for v in re.match(
                r'Cost\.min = (-?\d+), Cost\.avg = -?\d+\.?\d*,'
                r' Cost\.max = -?(\d+)',
                next(entries),
            ).groups()
        )
        next(entries)  # skip gap line
        solver_details['penalty_extrema'] = tuple(
            float(v)
            for v in re.match(
                r'Penalty\.min = (\d+), Penalty\.avg = \d+\.?\d*,'
                r' Penalty\.max = (\d+)',
                next(entries),
            ).groups()
        )
        solver_details['trials_extrema'] = tuple(
            float(v)
            for v in re.match(
                r'Trials\.min = (\d+), Trials\.avg = \d+\.?\d*,'
                r' Trials\.max = (\d+)',
                next(entries),
            ).groups()
        )
        solver_details['runtime_extrema'] = tuple(
            float(v)
            for v in re.match(
                r'Time\.min = (\d+\.?\d*) sec., Time\.avg = \d+\.?\d* sec.,'
                r' Time\.max = (\d+\.?\d*) sec.',
                next(entries),
            ).groups()
        )
    for subtree_id, branch in enumerate(branches):
        loads = range(len(branch), 0, -1)
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
    return S


_lkh_fun_fingerprint = fun_fingerprint(lkh)


def _solution_time(log, objective) -> float:
    sol_repr = f'{objective}'
    time = 0.0
    for line in log.splitlines():
        if not line or line[0] == '*':
            continue
        if line[:4] == 'Run ':
            # sum up the times of each Run, until the Run that found the objective
            # example: Run 4: Cost = 84_129583, Time = 2.87 sec.
            cost_, time_ = line.split(': ')[1].split(', ')
            # example time_: Time = 2.87 sec.
            time += float(time_.split(' = ')[1].split(' ')[0])
            # example cost_: Cost = 84_8724588
            if cost_.split('_')[1] == sol_repr:
                # this was the Run that found the objective: finished
                break
    return time


def iterative_lkh(
    Aʹ: nx.Graph,
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
    max_retries: int = 10,
) -> nx.Graph:
    """Iterate until crossing-free solution is found (`lkh()` wrapper).

    See the docstring of lkh() for details on the LKH-3 meta-heuristic.

    The vehicle-routing solver LKH-3 may produce routes with crossings, this
    wrapper ensures the output is crossing-free. Each time a solution with a
    crossing is produced, a repair attempt is made. Failing that, one of the
    offending edges is removed from `A` and the solver is called again. In the
    same way as `lkh()`, it is recommended to pass a normalized `A`.

    Args:
      *: see `lkh()`
      max_retries: maximum number of additional calls to lkh() to avoid crossings

    Returns:
      Solution topology S
    """
    A = Aʹ.copy()
    diagonals = Aʹ.graph['diagonals'].copy()
    A.graph['diagonals'] = diagonals
    nx.set_node_attributes(A, -1, 'root')
    _add_link_blockage(A)
    _prune_bad_links(A, math.ceil(2.4 * capacity))
    i = 0
    while True:
        # solve
        S = lkh(
            A,
            capacity=capacity,
            time_limit=time_limit,
            scale=scale,
            vehicles=vehicles,
            runs=runs,
            per_run_limit=per_run_limit,
            precision=precision,
            complete=complete,
            keep_log=keep_log,
            seed=seed,
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
