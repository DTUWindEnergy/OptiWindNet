# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import threading
import weakref
from collections.abc import Mapping
from itertools import chain
from types import MappingProxyType
from typing import Any

import networkx as nx
from bitarray import frozenbitarray
from pyscipopt import SCIP_STAGE, Model

from ..crossings import _feeder_link_conflicts, edgeset_edgeXing_iter
from ..identity import fingerprint_function
from ..types import FeederRoute
from ._core import (
    ModelMetadata,
    NetworkBounds,
    OWNSolutionNotFound,
    OWNWarmupFailed,
    PoolHandler,
    SolutionInfo,
    Solver,
    Topology,
    balance_bounds,
    bound_rows,
    canonical_linksets,
    check_model_bounds,
    check_warmstart_topology,
    links_to_fix,
    passover_rows,
    physical_core_count,
    warmstart_links,
)

__all__ = ('make_min_length_model', 'warmup_model')

_lggr = logging.getLogger(__name__)
error, warn, info = _lggr.error, _lggr.warning, _lggr.info

# SCIP keeps a single process-wide thread pool for concurrent solves, yet every
# model that ran one tears that pool down when freed: two such models alive at
# once crash the interpreter when the second is freed.
_concurrent_holder: weakref.ref[Model] | None = None
_concurrent_lock = threading.Lock()


def _solve_concurrent(model: Model) -> None:
    """Run ``model.solveConcurrent()`` if no other live model holds the pool.

    Raises:
      RuntimeError: if another model that ran a concurrent solve is alive.
    """
    global _concurrent_holder
    with _concurrent_lock:
        holder = _concurrent_holder and _concurrent_holder()
        if holder is not None and holder is not model:
            raise RuntimeError(
                'Another SCIP model that ran a concurrent solve is still alive: '
                'call get_solution() on its solver (or delete the solver) first, '
                "or pass options={'concurrent': False}."
            )
        _concurrent_holder = weakref.ref(model)
    model.solveConcurrent()


class SolverSCIP(Solver, PoolHandler):
    name: str = 'scip'
    _solution_pool: list[tuple[float, dict]]

    def __init__(self):
        self.options = {
            'parallel/maxnthreads': physical_core_count(),
            'concurrent/scip-feas/prefprio': 0.6,
            'concurrent/scip/prefprio': 0.3,
            'concurrent/scip-cpsolver/prefprio': 0,
            'concurrent/scip-easycip/prefprio': 0,
            'concurrent/scip-opti/prefprio': 0,
        }

    # Variable values in a SCIP solution may be slightly off of an integer:
    #   use round() to coerce the float to the nearest integer
    def _link_val(self, var: Any) -> int:
        return round(self._value_map[var])

    def _flow_val(self, var: Any) -> int:
        return round(self._value_map[var])

    def _set_model(self, warmstart: nx.DiGraph | None) -> None:
        model, metadata = make_min_length_model(self.A, self.bounds)
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
        """Run SCIP search via concurrent multi-threading or standard optimization.

        Options:
          concurrent (bool):
            If ``True`` (default), launches SCIP's multi-threaded concurrent
            solvers via ``Model.solveConcurrent()``. Pass
            ``options={'concurrent': False}`` to run standard single-threaded
            ``Model.optimize()``, which is also used if
            ``'parallel/maxnthreads'`` is 1.

        Only one model at a time may have run a concurrent solve: call
        ``get_solution()``, which frees the model, before another solver solves
        concurrently. A forked process inherits its parent's models, so fork
        first, then solve -- not the other way around.

        Raises:
          RuntimeError: if another solver's model that ran a concurrent solve
            is still alive.
        """
        try:
            model = self.model
        except AttributeError as exc:
            exc.args += ('.set_problem() must be called before .solve()',)
            raise
        applied_options = {**self.options, **options}
        # a one-thread concurrent solve would still take SCIP's thread pool
        use_concurrent = (
            applied_options.pop('concurrent', True)
            and applied_options['parallel/maxnthreads'] > 1
        )
        if model.getStage() != SCIP_STAGE.PROBLEM:
            # SCIP refuses to (re)solve unless the model is back in the PROBLEM
            # stage; callers may solve the same model more than once (e.g. the
            # retry loop in api.py). Solutions added by warmup_model() survive.
            model.freeTransform()
        # this would be ideal for displaying the log in notebooks, but is killing python
        # model.redirectOutput()
        model.setParams(applied_options)
        model.setParam('limits/gap', mip_gap)
        model.setParam('limits/time', time_limit)
        self.stopping = {'mip_gap': mip_gap, 'time_limit': time_limit}
        if not verbose:
            model.setParam('display/verblevel', 1)  # 1: warnings; 0: no output
        info('>>> SCIP parameters <<<\n%s\n', model.getParams())
        if use_concurrent:
            _solve_concurrent(model)
        else:
            model.optimize()
        num_solutions = model.getNSols()
        if num_solutions == 0:
            raise OWNSolutionNotFound(
                f'Unable to find a solution. Solver {self.name} terminated'
                f' with: {model.getStatus()}'
            )
        bound = model.getDualbound()
        objective = model.getObjVal()
        self._solution_pool = [
            (model.getSolObjVal(sol), sol) for sol in model.getSols()
        ]
        # PoolHandler relies on index zero being the lowest model objective.
        self._solution_pool.sort(key=lambda entry: entry[0])
        self.num_solutions = num_solutions
        solution_info = SolutionInfo(
            runtime=model.getSolvingTime(),
            bound=bound,
            objective=objective,
            # SCIP offers model.getGap(), but its denominator is the bound
            relgap=1.0 - bound / objective,
            termination=model.getStatus(),
        )
        return self._record_incumbent(solution_info, applied_options)

    def _read_incumbent_linkbits(self) -> frozenbitarray:
        return self._read_incumbent_linkbits_from_pool()

    def get_solution(self, A: nx.Graph | None = None) -> tuple[nx.DiGraph, nx.DiGraph]:
        """Output solution topology S and routeset G, then free the SCIP model.

        Freeing the model releases SCIP's thread pool for the next concurrent
        solve (see :meth:`solve`). ``solve()`` cannot be called afterwards.
        """
        if A is None:
            A = self.A
        P, bounds = self.P, self.bounds
        try:
            if bounds.feeder_route is FeederRoute.STRAIGHT:
                S = self._incumbent_topology()
                G = self._create_detours(S, P, A)
            else:
                S, G = self._investigate_pool(P, A)
            G.graph.update(self._make_graph_attributes())
            return S, G
        finally:
            # drop every reference to the model's native objects, the last one
            # being the model itself, whose deallocation calls SCIPfree()
            self._solution_pool = []
            self.__dict__.pop('_value_map', None)
            self.__dict__.pop('model', None)

    def _objective_at(self, index: int) -> float:
        objective_value, self._value_map = self._solution_pool[index]
        return objective_value


def make_min_length_model(
    A: nx.Graph,
    bounds: NetworkBounds,
) -> tuple[Model, ModelMetadata]:
    """Make discrete optimization model over link set A.

    Build SCIP model for the collector system length minimization.

    Args:
      A: graph with the available links to choose from
      bounds: capacity, topology, feeder route, counts and loads to enforce
        (see :class:`~optiwindnet.constraints.NetworkBounds`)

    Raises:
      ValueError: the bounds limit the load of the roots of a RINGED model.
      NotImplementedError: the bounds are RINGED with nonunitary inflow.
    """
    check_model_bounds(bounds)
    topology, feeder_route = bounds.topology, bounds.feeder_route
    R = A.graph['R']
    T = A.graph['T']
    d2roots = A.graph['d2roots']
    A_terminals = nx.subgraph_view(A, filter_node=lambda n: n >= 0)
    inflow_total, inflow_ = bounds.inflow_total, bounds.inflow_

    # a ring holds two arms of `bounds.capacity` each
    capacity = bounds.group_capacity

    # Sets
    _T = range(T)
    _R = range(-R, 0)

    E, Eʹ, stars, starsʹ = canonical_linksets(A_terminals, R, T, topology)
    linkset = E + Eʹ + stars + starsʹ
    # flow variables only for edges with actual flow (no ring-backs)
    flowset = E + Eʹ + stars

    # Create model
    m = Model()

    ##############
    # Parameters #
    ##############

    k = capacity
    weight_ = (
        2 * tuple(A[u][v]['length'] for u, v in E)
        + tuple(d2roots[t, r] for t, r in stars)
        + (
            tuple(d2roots[t, r] for t, r in stars)
            if topology is Topology.RINGED
            else ()
        )
    )

    #############
    # Variables #
    #############

    link_ = {(u, v): m.addVar(f'link_{u}~{v}', 'B') for u, v in chain(E, Eʹ)}
    link_ |= {(t, r): m.addVar(f'link_{t}~r{-r}', 'B') for t, r in stars}
    if topology is Topology.RINGED:
        link_ |= {(r, t): m.addVar(f'link_r{-r}~{t}', 'B') for r, t in starsʹ}
    fixed = set(links_to_fix(A, bounds))
    for link in fixed:
        m.fixVar(link_[link], 0)
    # 'M' (implied integral): single_out_link + flow_conserv pin flows to integers.
    # a link into v carries at most what leaves v less v's own inflow
    flow_ = {
        (u, v): m.addVar(f'flow_{u}~{v}', 'M', lb=0, ub=k - inflow_[v])
        for u, v in chain(E, Eʹ)
        if (u, v) not in fixed
    }
    flow_ |= {
        (t, r): m.addVar(f'flow_{t}~r{-r}', 'M', lb=0, ub=k)
        for t, r in stars
        if (t, r) not in fixed
    }

    ###############
    # Constraints #
    ###############

    # total number of edges must equal number of terminal nodes (skip for RINGED)
    if topology is not Topology.RINGED:
        m.addCons(sum(link_.values()) == T, name='num_links_eq_T')

    # enforce a single directed edge between each node pair
    for u, v in E:
        if (u, v) in fixed:
            continue
        m.addConsSOS1((link_[(u, v)], link_[(v, u)]), name=f'single_dir_link_{u}~{v}')

    # feeder-edge crossings
    if feeder_route is FeederRoute.STRAIGHT:
        Xings, passovers = _feeder_link_conflicts(A)
        for u, v, r, t in Xings.tolist():
            if (u, v) in fixed or (t, r) in fixed:
                continue
            if topology is Topology.RINGED:
                m.addConsSOS1(
                    (link_[(u, v)], link_[(v, u)], link_[t, r], link_[r, t]),
                    name=f'feeder_link_cross_{u}~{v}_{t}~r{-r}',
                )
            else:
                m.addConsSOS1(
                    (link_[(u, v)], link_[(v, u)], link_[t, r]),
                    name=f'feeder_link_cross_{u}~{v}_{t}~r{-r}',
                )

        # a feeder over nodes conflicts with links on both of its sides
        for links, rhs, name in passover_rows(passovers, topology is Topology.RINGED):
            active_links = [link for link in links if link not in fixed]
            if len(active_links) <= rhs:
                continue
            m.addCons(sum(link_[link] for link in active_links) <= rhs, name=name)

    # edge-edge crossings
    for Xing in edgeset_edgeXing_iter(A.graph['diagonals']):
        active_Xing = [uv for uv in Xing if uv not in fixed]
        if len(active_Xing) <= 1:
            continue
        m.addConsSOS1(
            sum(((link_[u, v], link_[v, u]) for u, v in active_Xing), ()),
            name=f'link_link_cross_{"_".join(f"{u}~{v}" for u, v in Xing)}',
        )

    # bind flow to link activation (only for edges with flow variables)
    for t, n in flowset:
        if (t, n) in fixed:
            continue
        _n = str(n) if n >= 0 else f'r{-n}'
        head_room = k if n < 0 else k - inflow_[n]
        m.addCons(
            flow_[t, n] <= link_[t, n] * head_room,
            name=f'flow_ub_{t}~{_n}',
        )
        m.addCons(
            flow_[t, n] >= link_[t, n] * inflow_[t],
            name=f'flow_lb_{t}~{_n}',
        )

    # flow conservation with possibly nonunitary terminal inflow
    for t in _T:
        m.addCons(
            sum(flow_[t, n] for n in A_terminals.neighbors(t) if (t, n) in flow_)
            - sum(flow_[n, t] for n in A_terminals.neighbors(t) if (n, t) in flow_)
            + sum(flow_[t, r] for r in _R if (t, r) in flow_)
            == inflow_[t],
            name=f'flow_conserv_{t}',
        )

    # feeder count and per-root bounds
    for row in bound_rows(bounds, R, T):
        var_ = flow_ if row.flow else link_
        row_sum = sum(var_[link] for link in row.links if link in var_)
        if row.lb is not None and row.lb == row.ub:
            m.addCons(row_sum == row.lb, name=f'{row.name}_eq')
            continue
        if row.lb is not None:
            m.addCons(row_sum >= row.lb, name=f'{row.name}_lb')
        if row.ub is not None:
            m.addCons(row_sum <= row.ub, name=f'{row.name}_ub')

    load_lb, load_ub = balance_bounds(bounds)

    # enforce balanced subtrees (subtree loads differ at most by one unit)
    if load_lb is not None:
        for t, r in stars:
            if (t, r) in flow_:
                m.addCons(
                    flow_[t, r] >= link_[t, r] * load_lb, name=f'balanced_lb_{t}~r{-r}'
                )
    if load_ub is not None:
        for t, r in stars:
            if (t, r) in flow_:
                m.addCons(
                    flow_[t, r] <= link_[t, r] * load_ub, name=f'balanced_ub_{t}~r{-r}'
                )

    # topology-specific incoming-edge constraints
    if topology is Topology.RADIAL:
        for t in _T:
            active_nbrs = [n for n in A_terminals.neighbors(t) if (n, t) not in fixed]
            if len(active_nbrs) > 1:
                # SOS1 takes the *list* of incoming terminal-link variables (at most
                # one may be nonzero => simple paths), not their sum.
                m.addConsSOS1([link_[n, t] for n in active_nbrs], name=f'radial_{t}')
    elif topology is Topology.RINGED:
        for t in _T:
            active_nbrs = [n for n in A_terminals.neighbors(t) if (n, t) not in fixed]
            active_roots = [r for r in _R if (r, t) not in fixed]
            m.addCons(
                sum(link_[n, t] for n in active_nbrs)
                + sum(link_[r, t] for r in active_roots)
                == 1,
                name=f'ringed_{t}',
            )

    # assert all nodes are connected to some root
    m.addCons(
        sum(flow_[t, r] for r in _R for t in _T if (t, r) in flow_) == inflow_total,
        name='total_inflow_sank',
    )

    # valid inequalities
    for t in _T:
        # incoming flow limit
        in_flows = [flow_[n, t] for n in A_terminals.neighbors(t) if (n, t) in flow_]
        if in_flows:
            m.addCons(
                sum(in_flows) <= k - inflow_[t],
                name=f'inflow_limit_{t}',
            )
        # only one out-edge per terminal
        out_links = [
            n for n in chain(A_terminals.neighbors(t), _R) if (t, n) not in fixed
        ]
        m.addCons(
            sum(link_[t, n] for n in out_links) == 1,
            name=f'single_out_link_{t}',
        )

    #############
    # Objective #
    #############

    m.setObjective(
        sum(w * x for w, x in zip(weight_, link_.values())), sense='minimize'
    )

    ##################
    # Store metadata #
    ##################

    metadata = ModelMetadata(
        bounds,
        linkset,
        link_,
        flow_,
        _make_min_length_model_fingerprint,
    )

    return m, metadata


_make_min_length_model_fingerprint = fingerprint_function(make_min_length_model)


def warmup_model(model: Model, metadata: ModelMetadata, S: nx.DiGraph) -> Model:
    """Set initial solution into ``model``.

    Changes ``model`` and ``metadata`` in-place.

    Args:
      model: SCIP model to apply the solution to.
      metadata: indices to the model's variables.
      S: solution topology

    Returns:
      The same model instance that was provided, now with a solution.

    Raises:
      optiwindnet.types.OWNWarmupFailed: if some link in S is not available in model.
    """
    check_warmstart_topology(metadata, S)
    # createSol() zero-initializes every variable, so only the links S activates
    # need to be set.
    sol = model.createSol()
    for link_var, flow_var, flow in warmstart_links(metadata, S):
        model.setSolVal(sol, link_var, 1)
        if flow_var is not None:
            model.setSolVal(sol, flow_var, flow)

    # addSol() alone would take a solution that violates the model (e.g. a warm
    # start whose feeder count differs from the one the model pins), leaving SCIP
    # to search from an infeasible point, so check it against the original problem.
    # checkSol() covers bounds and integrality too (both default to True), on top of
    # the constraints -- although the bounds are already implied here: every flow var
    # has a linking constraint `flow_ub_*` (f <= M*link) with M no looser than the
    # var's own ub, and the solution sets flow > 0 only where link is 1.
    if not model.checkSol(sol, printreason=False, original=True):
        if _lggr.isEnabledFor(logging.INFO):
            # re-check to get the diagnosis: SCIP prints the violations to its own
            # output and `completely` keeps it going past the first one
            model.checkSol(sol, printreason=True, completely=True, original=True)
        model.freeSol(sol)
        raise OWNWarmupFailed('warmup_model() failed: S violates some model constraint')

    if not model.addSol(sol):
        raise OWNWarmupFailed(
            'warmup_model() failed: the feasible solution was not stored'
        )
    metadata.warmed_by = S.graph['creator']
    return model
