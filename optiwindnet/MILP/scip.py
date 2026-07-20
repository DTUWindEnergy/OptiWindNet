# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
from collections.abc import Mapping
from itertools import chain
from typing import Any

import networkx as nx
from pyscipopt import Model

from ..crossings import edgeset_edgeXing_iter, gateXing_iter
from ..interarraylib import G_from_S, fun_fingerprint
from ..pathfinding import PathFinder
from ._core import (
    FeederLimit,
    FeederRoute,
    ModelMetadata,
    ModelOptions,
    OWNSolutionNotFound,
    OWNWarmupFailed,
    PoolHandler,
    SolutionInfo,
    Solver,
    Topology,
    feeder_and_load_bounds,
    physical_core_count,
    ringed_warmstart_values,
    warmstart_topology,
)

__all__ = ('make_min_length_model', 'warmup_model')

_lggr = logging.getLogger(__name__)
error, warn, info = _lggr.error, _lggr.warning, _lggr.info


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

    def set_problem(
        self,
        P: nx.PlanarEmbedding,
        A: nx.Graph,
        capacity: int,
        model_options: Mapping[str, Any],
        warmstart: nx.Graph | None = None,
    ):
        self.P, self.A, self.capacity = P, A, capacity
        model_options = self.model_options = ModelOptions(**model_options)
        model, metadata = make_min_length_model(self.A, self.capacity, **model_options)
        self.model, self.metadata = model, metadata
        if warmstart is not None:
            warmup_model(model, metadata, warmstart)

    def solve(
        self,
        time_limit: float,
        mip_gap: float,
        options: dict[str, Any] = {},
        verbose: bool = False,
    ) -> SolutionInfo:
        """Wrapper for Model.solveConcurrent()."""
        try:
            model = self.model
        except AttributeError as exc:
            exc.args += ('.set_problem() must be called before .solve()',)
            raise
        applied_options = self.options | options
        # this would be ideal for displaying the log in notebooks, but is killing python
        # model.redirectOutput()
        model.setParams(applied_options)
        model.setParam('limits/gap', mip_gap)
        model.setParam('limits/time', time_limit)
        self.stopping = dict(mip_gap=mip_gap, time_limit=time_limit)
        if not verbose:
            model.setParam('display/verblevel', 1)  # 1: warnings; 0: no output
        info('>>> SCIP parameters <<<\n%s\n', model.getParams())
        model.solveConcurrent()
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
        # Prime _value_map with the best solution so the STRAIGHT get_solution()
        # path (which calls _topology_from_mip_pool without _objective_at) works;
        # the SEGMENTED path resets it per pool entry via _objective_at().
        _, self._value_map = self._solution_pool[0]
        self.num_solutions = num_solutions
        solution_info = SolutionInfo(
            runtime=model.getSolvingTime(),
            bound=bound,
            objective=objective,
            # SCIP offers model.getGap(), but its denominator is the bound
            relgap=1.0 - bound / objective,
            termination=model.getStatus(),
        )
        self.solution_info, self.applied_options = solution_info, applied_options
        info('>>> Solution <<<\n%s\n', solution_info)
        return solution_info

    def get_solution(self, A: nx.Graph | None = None) -> tuple[nx.Graph, nx.Graph]:
        if A is None:
            A = self.A
        P, model_options = self.P, self.model_options
        if model_options['feeder_route'] is FeederRoute.STRAIGHT:
            S = self._topology_from_mip_pool()
            G = PathFinder(G_from_S(S, A), P, A).create_detours()
        else:
            S, G = self._investigate_pool(P, A)
        G.graph.update(self._make_graph_attributes())
        return S, G

    def _objective_at(self, index: int) -> float:
        objective_value, self._value_map = self._solution_pool[index]
        return objective_value

    def _topology_from_mip_pool(self) -> nx.Graph:
        return self._topology_from_mip_sol()


def make_min_length_model(
    A: nx.Graph,
    capacity: int,
    *,
    topology: Topology = Topology.BRANCHED,
    feeder_route: FeederRoute = FeederRoute.SEGMENTED,
    feeder_limit: FeederLimit = FeederLimit.UNLIMITED,
    balanced: bool = False,
    max_feeders: int = 0,
) -> tuple[Model, ModelMetadata]:
    """Make discrete optimization model over link set A.

    Build SCIP model for the collector system length minimization.

    Args:
      A: graph with the available edges to choose from
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
    R = A.graph['R']
    T = A.graph['T']
    d2roots = A.graph['d2roots']
    A_terminals = nx.subgraph_view(A, filter_node=lambda n: n >= 0)
    W = sum(w for _, w in A_terminals.nodes(data='power', default=1))

    # For RINGED, double the internal capacity; store original for metadata.
    ring_capacity = capacity
    if topology is Topology.RINGED:
        capacity = 2 * capacity

    # Sets
    _T = range(T)
    _R = range(-R, 0)

    E = tuple(((u, v) if u < v else (v, u)) for u, v in A_terminals.edges())
    # using directed node-node links -> create the reversed tuples
    Eʹ = tuple((v, u) for u, v in E)
    # set of feeders to all roots
    stars = tuple((t, r) for t in _T for r in _R)
    if topology is Topology.RINGED:
        starsʹ = tuple((r, t) for t, r in stars)
    else:
        starsʹ = ()
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
    flow_ = {
        (u, v): m.addVar(f'flow_{u}~{v}', 'I', lb=0, ub=k - 1) for u, v in chain(E, Eʹ)
    }
    flow_ |= {(t, r): m.addVar(f'flow_{t}~r{-r}', lb=0, ub=k) for t, r in stars}

    ###############
    # Constraints #
    ###############

    # total number of edges must equal number of terminal nodes (skip for RINGED)
    if topology is not Topology.RINGED:
        m.addCons(sum(link_.values()) == T, name='num_links_eq_T')

    # enforce a single directed edge between each node pair
    for u, v in E:
        m.addConsSOS1((link_[(u, v)], link_[(v, u)]), name=f'single_dir_link_{u}~{v}')

    # feeder-edge crossings
    if feeder_route is FeederRoute.STRAIGHT:
        for (u, v), (r, t) in gateXing_iter(A):
            if u >= 0:
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
            else:
                # a feeder crossing another feeder (possible in multi-root instances)
                if topology is Topology.RINGED:
                    m.addConsSOS1(
                        (link_[(u, v)], link_[t, r], link_[r, t]),
                        name=f'feeder_feeder_cross_r{-u}~{v}_{t}~r{-r}',
                    )
                else:
                    m.addConsSOS1(
                        (link_[(u, v)], link_[t, r]),
                        name=f'feeder_feeder_cross_r{-u}~{v}_{t}~r{-r}',
                    )

    # edge-edge crossings
    for Xing in edgeset_edgeXing_iter(A.graph['diagonals']):
        m.addConsSOS1(
            sum(((link_[u, v], link_[v, u]) for u, v in Xing), ()),
            name=f'link_link_cross_{"_".join(f"{u}~{v}" for u, v in Xing)}',
        )

    # bind flow to link activation (only for edges with flow variables)
    for t, n in flowset:
        _n = str(n) if n >= 0 else f'r{-n}'
        m.addCons(
            flow_[t, n] <= link_[t, n] * (k if n < 0 else (k - 1)),
            name=f'flow_ub_{t}~{_n}',
        )
        m.addCons(flow_[t, n] >= link_[t, n], name=f'flow_lb_{t}~{_n}')

    # flow conservation with possibly non-unitary node power
    for t in _T:
        m.addCons(
            sum((flow_[t, n] - flow_[n, t]) for n in A_terminals.neighbors(t))
            + sum(flow_[t, r] for r in _R)
            == A.nodes[t].get('power', 1),
            name=f'flow_conserv_{t}',
        )

    # feeder limits. A RINGED subtree is a cycle with two feeders, so the
    # user-facing feeder count is in substation connections (two per ring), while
    # the model counts rings (one flow-feeder var each): convert between them.
    feeders_per_subtree = 2 if topology is Topology.RINGED else 1
    feeders_lb, feeders_ub, load_lb, load_ub = feeder_and_load_bounds(
        T, k, feeder_limit, max_feeders, balanced, feeders_per_subtree
    )
    if feeders_ub is not None and feeder_limit.name.startswith('MIN_PLUS'):
        # derived from the minimum: surface it in the solution's metadata
        max_feeders = feeders_per_subtree * feeders_ub
    all_feeder_vars_sum = sum(link_[t, r] for r in _R for t in _T)
    if feeders_lb == feeders_ub:
        m.addCons(all_feeder_vars_sum == feeders_lb, name='feeder_limit_eq')
    else:
        # valid inequality: number of feeders is at least the minimum
        m.addCons(all_feeder_vars_sum >= feeders_lb, name='feeder_limit_lb')
        if feeders_ub is not None:
            m.addCons(all_feeder_vars_sum <= feeders_ub, name='feeder_limit_ub')

    # enforce balanced subtrees (subtree loads differ at most by one unit)
    if load_lb is not None:
        for t, r in stars:
            m.addCons(
                flow_[t, r] >= link_[t, r] * load_lb, name=f'balanced_lb_{t}~r{-r}'
            )
    if load_ub is not None:
        for t, r in stars:
            m.addCons(
                flow_[t, r] <= link_[t, r] * load_ub, name=f'balanced_ub_{t}~r{-r}'
            )

    # topology-specific incoming-edge constraints
    if topology is Topology.RADIAL:
        for t in _T:
            # SOS1 takes the *list* of incoming terminal-link variables (at most
            # one may be nonzero => simple paths), not their sum.
            m.addConsSOS1(
                [link_[n, t] for n in A_terminals.neighbors(t)], name=f'radial_{t}'
            )
    elif topology is Topology.RINGED:
        for t in _T:
            m.addCons(
                sum(link_[n, t] for n in A_terminals.neighbors(t))
                + sum(link_[r, t] for r in _R)
                == 1,
                name=f'ringed_{t}',
            )

    # assert all nodes are connected to some root
    m.addCons(sum(flow_[t, r] for r in _R for t in _T) == W, name='total_power_sank')

    # valid inequalities
    for t in _T:
        # incoming flow limit
        m.addCons(
            sum(flow_[n, t] for n in A_terminals.neighbors(t))
            <= k - A.nodes[t].get('power', 1),
            name=f'inflow_limit_{t}',
        )
        # only one out-edge per terminal
        m.addCons(
            sum(link_[t, n] for n in chain(A_terminals.neighbors(t), _R)) == 1,
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

    model_options = dict(
        topology=topology,
        feeder_route=feeder_route,
        feeder_limit=feeder_limit,
        max_feeders=max_feeders,
        balanced=balanced,
    )
    metadata = ModelMetadata(
        R,
        T,
        ring_capacity,
        linkset,
        link_,
        flow_,
        model_options,
        _make_min_length_model_fingerprint,
    )

    return m, metadata


_make_min_length_model_fingerprint = fun_fingerprint(make_min_length_model)


def warmup_model(model: Model, metadata: ModelMetadata, S: nx.Graph) -> Model:
    """Set initial solution into ``model``.

    Changes ``model`` and ``metadata`` in-place.

    Args:
      model: SCIP model to apply the solution to.
      metadata: indices to the model's variables.
      S: solution topology

    Returns:
      The same model instance that was provided, now with a solution.

    Raises:
      OWNWarmupFailed: if some link in S is not available in model.
    """
    R, T = metadata.R, metadata.T
    topology = warmstart_topology(metadata, S)
    if topology is Topology.RINGED:
        # A ring is a single directed chain per the flow formulation; derive the
        # variable values from the (split) ringed solution graph directly.
        link_vals, flow_vals = ringed_warmstart_values(metadata, S)
        sol = model.createSol()
        for key, var in metadata.link_.items():
            model.setSolVal(sol, var, link_vals[key])
        for key, var in metadata.flow_.items():
            model.setSolVal(sol, var, flow_vals[key])
        if not model.addSol(sol):
            raise OWNWarmupFailed(
                'warmup_model() failed: S violates some model constraint'
            )
        metadata.warmed_by = S.graph['creator']
        return model
    in_S_not_in_model = S.edges - metadata.link_.keys()
    in_S_not_in_model -= {(v, u) for u, v in metadata.linkset[-R * T :]}
    if in_S_not_in_model:
        raise OWNWarmupFailed(
            f'warmup_model() failed: model lacks S links ({in_S_not_in_model})'
        )
    sol = model.createSol()
    for u, v in metadata.linkset[: (len(metadata.linkset) - R * T) // 2]:
        edgeD = S.edges.get((u, v))
        if edgeD is None:
            model.setSolVal(sol, metadata.link_[u, v], 0)
            model.setSolVal(sol, metadata.flow_[u, v], 0)
            model.setSolVal(sol, metadata.link_[v, u], 0)
            model.setSolVal(sol, metadata.flow_[v, u], 0)
        else:
            u, v = (u, v) if ((u < v) == edgeD['reverse']) else (v, u)
            model.setSolVal(sol, metadata.link_[u, v], 1)
            model.setSolVal(sol, metadata.flow_[u, v], edgeD['load'])
            model.setSolVal(sol, metadata.link_[v, u], 0)
            model.setSolVal(sol, metadata.flow_[v, u], 0)
    for t, r in metadata.linkset[-R * T :]:
        edgeD = S.edges.get((t, r))
        model.setSolVal(sol, metadata.link_[t, r], 0 if edgeD is None else 1)
        model.setSolVal(
            sol, metadata.flow_[t, r], 0 if edgeD is None else edgeD['load']
        )
    accepted = model.addSol(sol)
    if not accepted:
        raise OWNWarmupFailed('warmup_model() failed: S violates some model constraint')
    metadata.warmed_by = S.graph['creator']
    return model
