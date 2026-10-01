# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
from collections.abc import Iterator, Mapping
from itertools import chain
from types import MappingProxyType
from typing import Any

import highspy
import networkx as nx
import numpy as np
from bitarray import frozenbitarray
from scipy.sparse import csc_array, csr_array

from ..converting import G_from_S
from ..crossings import _feeder_crossings, edgeset_edgeXing_iter
from ..identity import fingerprint_function
from ..pathfinding import PathFinder
from ._core import (
    FeederLimit,
    FeederRoute,
    ModelMetadata,
    OWNSolutionNotFound,
    OWNWarmupFailed,
    PoolHandler,
    SolutionInfo,
    Solver,
    Topology,
    canonical_linksets,
    check_inflow_support,
    check_model_enums,
    check_warmstart_topology,
    feeder_and_load_bounds,
    nonclosest_feeders,
    physical_core_count,
    warmstart_links,
)

__all__ = ('make_min_length_model', 'warmup_model')

_lggr = logging.getLogger(__name__)
error, warn, info = _lggr.error, _lggr.warning, _lggr.info

qsum = highspy.Highs.qsum


class SolverHiGHS(Solver, PoolHandler):
    """HiGHS wrapper using the ``highspy`` API directly.

    HiGHS keeps every improving solution found in a search when the option
    ``mip_improving_solution_save`` is on, which :meth:`solve` enforces. These
    solutions form the pool that :meth:`.PoolHandler._investigate_pool` ranks.
    """

    name: str = 'highs'
    _solution_pool: list[tuple[float, np.ndarray]]

    def __init__(self):
        self.options = {
            'parallel': 'on',
            'threads': physical_core_count(),
        }

    # Variable values in a HiGHS solution may be slightly off of an integer:
    #   use round() to coerce the float to the nearest integer
    def _link_val(self, var: Any) -> int:
        return round(self._value_map[var.index])

    def _flow_val(self, var: Any) -> int:
        return round(self._value_map[var.index])

    def _set_model(self, warmstart: nx.Graph | None) -> None:
        model, metadata = make_min_length_model(
            self.A, self.capacity, **self.model_options
        )
        self.model, self.metadata = model, metadata
        if warmstart is not None:
            warmup_model(model, metadata, warmstart)

    def solve(
        self,
        time_limit: float,
        mip_gap: float,
        options: Mapping[str, Any] = MappingProxyType({}),
        verbose: bool = False,
    ) -> SolutionInfo:
        """Run the HiGHS search, saving every improving solution to a pool.

        Options are HiGHS option names (see HiGHS's documentation), passed
        unchanged to ``Highs.setOptionValue()``. Option
        ``mip_improving_solution_save`` is always on, since the pool depends on it.
        """
        try:
            model = self.model
        except AttributeError as exc:
            exc.args += ('.set_problem() must be called before .solve()',)
            raise
        applied_options = {**self.options, **options}
        self.stopping = {'mip_gap': mip_gap, 'time_limit': time_limit}
        for key, value in chain(
            applied_options.items(),
            (
                ('time_limit', time_limit),
                ('mip_rel_gap', mip_gap),
                ('output_flag', verbose),
                ('mip_improving_solution_save', True),
            ),
        ):
            if model.setOptionValue(key, value) != highspy.HighsStatus.kOk:
                raise ValueError(f'Invalid HiGHS option: {key}={value!r}')
        info(
            '>>> HiGHS options <<<\n%s\n',
            {key: model.getOptionValue(key)[1] for key in applied_options},
        )
        # HiGHS's thread pool is process-wide and fixed by the first search to run
        # in the process; a search requesting another thread count fails unless the
        # pool is torn down (it is rebuilt on demand).
        highspy.Highs.resetGlobalScheduler(True)
        # getRunTime() accumulates over the searches of a Highs instance
        start_time = model.getRunTime()
        model.run()
        runtime = model.getRunTime() - start_time
        termination = model.modelStatusToString(model.getModelStatus())
        # the saved solutions are those of the latest search only
        saved = model.getSavedMipSolutions()
        num_solutions = len(saved)
        if num_solutions == 0:
            raise OWNSolutionNotFound(
                f'Unable to find a solution. Solver {self.name} terminated'
                f' with: {termination}'
            )
        # PoolHandler relies on index zero being the lowest model objective.
        self._solution_pool = sorted(
            ((sol.objective, np.asarray(sol.col_value)) for sol in saved),
            key=lambda entry: entry[0],
        )
        self.num_solutions = num_solutions
        highs_info = model.getInfo()
        bound = highs_info.mip_dual_bound
        objective = highs_info.objective_function_value
        solution_info = SolutionInfo(
            runtime=runtime,
            bound=bound,
            objective=objective,
            relgap=1.0 - bound / objective,
            termination=termination,
        )
        return self._record_incumbent(solution_info, applied_options)

    def _read_incumbent_linkbits(self) -> frozenbitarray:
        return self._read_incumbent_linkbits_from_pool()

    def get_solution(self, A: nx.Graph | None = None) -> tuple[nx.Graph, nx.Graph]:
        if A is None:
            A = self.A
        P, model_options = self.P, self.model_options
        if model_options['feeder_route'] is FeederRoute.STRAIGHT:
            S = self._incumbent_topology()
            G = PathFinder(G_from_S(S, A), P, A).create_detours()
        else:
            S, G = self._investigate_pool(P, A)
        G.graph.update(self._make_graph_attributes())
        return S, G

    def _objective_at(self, index: int) -> float:
        objective_value, self._value_map = self._solution_pool[index]
        return objective_value


def make_min_length_model(
    A: nx.Graph,
    capacity: int,
    *,
    topology: Topology = Topology.BRANCHED,
    feeder_route: FeederRoute = FeederRoute.SEGMENTED,
    feeder_limit: FeederLimit = FeederLimit.UNLIMITED,
    balanced: bool = False,
    max_feeders: int = 0,
) -> tuple[highspy.Highs, ModelMetadata]:
    """Make discrete optimization model over link set A.

    Build HiGHS model for the collector system length minimization.

    Args:
      A: graph with the available links to choose from
      capacity: maximum link flow capacity
      topology: one of ``Topology.{BRANCHED, RADIAL, RINGED}``
      feeder_route:
        ``FeederRoute.SEGMENTED`` → feeder routes may be detoured around subtrees;
        ``FeederRoute.STRAIGHT`` → feeder routes must be straight, direct lines
      feeder_limit: one of ``FeederLimit.{MINIMUM, UNLIMITED, EXACTLY, SPECIFIED,
        MIN_PLUS1, MIN_PLUS2, MIN_PLUS3}``
      balanced: enforce subtree loads differing at most by one unit (only
        possible if ``feeder_limit`` pins the feeder count to a single value)
      max_feeders: upper bound if ``feeder_limit`` is ``FeederLimit.SPECIFIED``,
        exact count if it is ``FeederLimit.EXACTLY``, unused otherwise
    """
    check_model_enums(topology, feeder_route, feeder_limit)
    R = A.graph['R']
    T = A.graph['T']
    d2roots = A.graph['d2roots']
    A_terminals = nx.subgraph_view(A, filter_node=lambda n: n >= 0)
    inflow_total = check_inflow_support(A, topology)

    # For RINGED, double the internal capacity; store original for metadata.
    ring_capacity = capacity
    if topology is Topology.RINGED:
        capacity = 2 * capacity

    # Sets
    _T = range(T)
    _R = range(-R, 0)

    E, Eʹ, stars, starsʹ = canonical_linksets(A_terminals, R, T, topology)
    linkset = E + Eʹ + stars + starsʹ
    # flow variables only for edges with actual flow (no ring-backs)
    flowset = E + Eʹ + stars

    # Create model
    m = highspy.Highs()
    # solve() sets output_flag as requested; this silences the model building
    m.setOptionValue('output_flag', False)

    ##############
    # Parameters #
    ##############

    k = capacity
    feeder_weights = tuple(d2roots[t, r].item() for t, r in stars)
    weight_ = (
        2 * tuple(A[u][v]['length'] for u, v in E)
        + feeder_weights
        + (feeder_weights if topology is Topology.RINGED else ())
    )

    #############
    # Variables #
    #############

    link_ = {(u, v): m.addBinary(name=f'link_{u}~{v}') for u, v in chain(E, Eʹ)}
    link_ |= {(t, r): m.addBinary(name=f'link_{t}~r{-r}') for t, r in stars}
    if topology is Topology.RINGED:
        link_ |= {(r, t): m.addBinary(name=f'link_r{-r}~{t}') for r, t in starsʹ}
    # HiGHS has no fixing call: lb == ub is a fixed column, removed by presolve
    for t, r in nonclosest_feeders(A):
        m.changeColBounds(link_[t, r].index, 0, 0)
        if topology is Topology.RINGED:
            m.changeColBounds(link_[r, t].index, 0, 0)
    # continuous: single_out_link + flow_conserv pin flows to integers.
    # a link into v carries at most what leaves v less v's own inflow
    flow_ = {
        (u, v): m.addVariable(
            lb=0, ub=k - A.nodes[v].get('inflow', 1), name=f'flow_{u}~{v}'
        )
        for u, v in chain(E, Eʹ)
    }
    flow_ |= {
        (t, r): m.addVariable(lb=0, ub=k, name=f'flow_{t}~r{-r}') for t, r in stars
    }

    ###############
    # Constraints #
    ###############

    # total number of edges must equal number of terminal nodes (skip for RINGED)
    if topology is not Topology.RINGED:
        m.addConstr(qsum(link_.values()) == T, name='num_links_eq_T')

    # enforce a single directed edge between each node pair
    for u, v in E:
        m.addConstr(link_[(u, v)] + link_[(v, u)] <= 1, name=f'single_dir_link_{u}~{v}')

    # feeder-edge crossings
    if feeder_route is FeederRoute.STRAIGHT:
        for u, v, r, t in _feeder_crossings(A).tolist():
            if topology is Topology.RINGED:
                m.addConstr(
                    link_[(u, v)] + link_[(v, u)] + link_[t, r] + link_[r, t] <= 1,
                    name=f'feeder_link_cross_{u}~{v}_{t}~r{-r}',
                )
            else:
                m.addConstr(
                    link_[(u, v)] + link_[(v, u)] + link_[t, r] <= 1,
                    name=f'feeder_link_cross_{u}~{v}_{t}~r{-r}',
                )

    # edge-edge crossings
    for Xing in edgeset_edgeXing_iter(A.graph['diagonals']):
        m.addConstr(
            qsum(link_[a, b] for u, v in Xing for a, b in ((u, v), (v, u))) <= 1,
            name=f'link_link_cross_{"_".join(f"{u}~{v}" for u, v in Xing)}',
        )

    # bind flow to link activation (only for edges with flow variables)
    for t, n in flowset:
        _n = str(n) if n >= 0 else f'r{-n}'
        head_room = k if n < 0 else k - A.nodes[n].get('inflow', 1)
        m.addConstr(
            flow_[t, n] <= head_room * link_[t, n],
            name=f'flow_ub_{t}~{_n}',
        )
        m.addConstr(
            flow_[t, n] >= A.nodes[t].get('inflow', 1) * link_[t, n],
            name=f'flow_lb_{t}~{_n}',
        )

    # flow conservation with possibly nonunitary terminal inflow
    for t in _T:
        m.addConstr(
            qsum(
                chain(
                    (flow_[t, n] - flow_[n, t] for n in A_terminals.neighbors(t)),
                    (flow_[t, r] for r in _R),
                )
            )
            == A.nodes[t].get('inflow', 1),
            name=f'flow_conserv_{t}',
        )

    # feeder limits. A RINGED subtree is a cycle with two feeders, so the
    # user-facing feeder count is in substation connections (two per ring), while
    # the model counts rings (one flow-feeder var each): convert between them.
    feeders_per_subtree = 2 if topology is Topology.RINGED else 1
    feeders_lb, feeders_ub, load_lb, load_ub = feeder_and_load_bounds(
        T, k, feeder_limit, max_feeders, balanced, feeders_per_subtree, inflow_total
    )
    if feeders_ub is not None and feeder_limit.name.startswith('MIN_PLUS'):
        # derived from the minimum: surface it in the solution's metadata
        max_feeders = feeders_per_subtree * feeders_ub
    all_feeder_vars = [link_[t, r] for r in _R for t in _T]
    if feeders_lb == feeders_ub:
        m.addConstr(qsum(all_feeder_vars) == feeders_lb, name='feeder_limit_eq')
    else:
        # valid inequality: number of feeders is at least the minimum
        m.addConstr(qsum(all_feeder_vars) >= feeders_lb, name='feeder_limit_lb')
        if feeders_ub is not None:
            m.addConstr(qsum(all_feeder_vars) <= feeders_ub, name='feeder_limit_ub')

    # enforce balanced subtrees (subtree loads differ at most by one unit)
    if load_lb is not None:
        for t, r in stars:
            m.addConstr(
                flow_[t, r] >= load_lb * link_[t, r], name=f'balanced_lb_{t}~r{-r}'
            )
    if load_ub is not None:
        for t, r in stars:
            m.addConstr(
                flow_[t, r] <= load_ub * link_[t, r], name=f'balanced_ub_{t}~r{-r}'
            )

    # topology-specific incoming-edge constraints
    if topology is Topology.RADIAL:
        for t in _T:
            m.addConstr(
                qsum(link_[n, t] for n in A_terminals.neighbors(t)) <= 1,
                name=f'radial_{t}',
            )
    elif topology is Topology.RINGED:
        for t in _T:
            m.addConstr(
                qsum(
                    chain(
                        (link_[n, t] for n in A_terminals.neighbors(t)),
                        (link_[r, t] for r in _R),
                    )
                )
                == 1,
                name=f'ringed_{t}',
            )

    # assert all nodes are connected to some root
    m.addConstr(
        qsum(flow_[t, r] for r in _R for t in _T) == inflow_total,
        name='total_inflow_sank',
    )

    # valid inequalities
    for t in _T:
        # incoming flow limit
        m.addConstr(
            qsum(flow_[n, t] for n in A_terminals.neighbors(t))
            <= k - A.nodes[t].get('inflow', 1),
            name=f'inflow_limit_{t}',
        )
        # only one out-edge per terminal
        m.addConstr(
            qsum(link_[t, n] for n in chain(A_terminals.neighbors(t), _R)) == 1,
            name=f'single_out_link_{t}',
        )

    #############
    # Objective #
    #############

    m.setObjective(
        qsum(w * x for w, x in zip(weight_, link_.values())),
        sense=highspy.ObjSense.kMinimize,
    )

    ##################
    # Store metadata #
    ##################

    model_options = {
        'topology': topology,
        'feeder_route': feeder_route,
        'feeder_limit': feeder_limit,
        'max_feeders': max_feeders,
        'balanced': balanced,
    }
    metadata = ModelMetadata(
        R,
        T,
        ring_capacity,
        linkset,
        link_,
        flow_,
        model_options,
        _make_min_length_model_fingerprint,
        weight_=weight_,
    )

    return m, metadata


_make_min_length_model_fingerprint = fingerprint_function(make_min_length_model)


def _solution_violations(
    model: highspy.Highs, col_value: np.ndarray, tol: float = 1e-6
) -> Iterator[tuple[str, float, float, float]]:
    """Yield (constraint, value, lower_bound, upper_bound) per violated constraint.

    ``Highs.setSolution()`` accepts any vector of column values, and the search
    discards an infeasible one silently, so the rows are checked here.

    Args:
      model: model whose rows are to be checked.
      col_value: value for every column of the model.
      tol: slack allowed on each bound.

    Yields:
      One tuple per violated row, in row order.
    """
    lp = model.getLp()
    a_matrix = lp.a_matrix_
    # a model built row by row keeps its matrix rowwise
    sparse_array = (
        csc_array if a_matrix.format_ == highspy.MatrixFormat.kColwise else csr_array
    )
    activity = (
        sparse_array(
            (a_matrix.value_, a_matrix.index_, a_matrix.start_),
            shape=(lp.num_row_, lp.num_col_),
        )
        @ col_value
    )
    lower, upper = np.asarray(lp.row_lower_), np.asarray(lp.row_upper_)
    for row in np.flatnonzero((activity < lower - tol) | (activity > upper + tol)):
        _, name = model.getRowName(int(row))
        yield name, activity[row].item(), lower[row].item(), upper[row].item()


def warmup_model(
    model: highspy.Highs, metadata: ModelMetadata, S: nx.Graph
) -> highspy.Highs:
    """Set initial solution into ``model``.

    Changes ``model`` and ``metadata`` in-place.

    Args:
      model: HiGHS model to apply the solution to.
      metadata: indices to the model's variables.
      S: solution topology

    Returns:
      The same model instance that was provided, now with a solution.

    Raises:
      OWNWarmupFailed: if some link in S is not available in model or if S
        violates some model constraint.
    """
    check_warmstart_topology(metadata, S)
    # a complete solution spares HiGHS from having to fill in the gaps, so
    # initialize every variable to 0 and override the ones S activates.
    col_value = np.zeros(model.getNumCol())
    for link_var, flow_var, flow in warmstart_links(metadata, S):
        col_value[link_var.index] = 1
        if flow_var is not None:
            col_value[flow_var.index] = flow

    # Bounds need no separate check: every flow var has a linking constraint
    # `flow_ub_*` (f <= M*link) with M no looser than the var's own ub, and the
    # solution sets flow > 0 only where link is 1.
    violations = _solution_violations(model, col_value)
    first = next(violations, None)
    if first is not None:
        if _lggr.isEnabledFor(logging.INFO):
            for name, value, lb, ub in chain((first,), violations):
                info('solution violates %s: %g outside [%g, %g]', name, value, lb, ub)
        raise OWNWarmupFailed(
            f'warmup_model() failed: S violates model constraint {first[0]}'
        )
    solution = highspy.HighsSolution()
    solution.col_value = col_value.tolist()
    solution.value_valid = True
    status = model.setSolution(solution)
    if status != highspy.HighsStatus.kOk:
        raise OWNWarmupFailed('warmup_model() failed: HiGHS rejected the solution')
    metadata.warmed_by = S.graph['creator']
    return model
