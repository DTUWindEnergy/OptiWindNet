# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import math
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
)

__all__ = ('make_min_length_model', 'warmup_model')

_lggr = logging.getLogger(__name__)
error, warn, info = _lggr.error, _lggr.warning, _lggr.info


class SolverSCIP(Solver, PoolHandler):
    name: str = 'scip'
    _solution_pool: list[tuple[float, dict]]

    def __init__(self):
        self.options = {
            # SCIP's concurrent solver uses min(8, cpu_count()) by default
            # but it may use a lower number if RAM usage is huge
            # 'parallel/maxnthreads': 16,
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
        model_options: ModelOptions,
        warmstart: nx.Graph | None = None,
    ):
        self.P, self.A, self.capacity = P, A, capacity
        self.model_options = model_options
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
                f'Unable to find a solution. Solver {self.name} terminated with: {model.getStatus()}'
            )
        bound = model.getDualbound()
        objective = model.getObjVal()
        self._solution_pool = [
            (model.getSolObjVal(sol), sol) for sol in model.getSols()
        ]
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
            G = PathFinder(
                G_from_S(S, A),
                P,
                A,
                branched=model_options['topology'] is Topology.BRANCHED,
            ).create_detours()
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
      topology: one of Topology.{BRANCHED, RADIAL}
      feeder_route:
        FeederRoute.SEGMENTED -> feeder routes may be detoured around subtrees;
        FeederRoute.STRAIGHT -> feeder routes must be straight, direct lines
      feeder_limit: one of FeederLimit.{MINIMUM, UNLIMITED, SPECIFIED,
        MIN_PLUS1, MIN_PLUS2, MIN_PLUS3}
      max_feeders: only used if feeder_limit is FeederLimit.SPECIFIED
    """
    R = A.graph['R']
    T = A.graph['T']
    d2roots = A.graph['d2roots']
    A_terminals = nx.subgraph_view(A, filter_node=lambda n: n >= 0)
    W = sum(w for _, w in A_terminals.nodes(data='power', default=1))

    # Sets
    _T = range(T)
    _R = range(-R, 0)

    E = tuple(((u, v) if u < v else (v, u)) for u, v in A_terminals.edges())
    # using directed node-node links -> create the reversed tuples
    Eʹ = tuple((v, u) for u, v in E)
    # set of feeders to all roots
    stars = tuple((t, r) for t in _T for r in _R)
    linkset = E + Eʹ + stars

    # Create model
    m = Model()

    ##############
    # Parameters #
    ##############

    k = capacity
    weight_ = 2 * tuple(A[u][v]['length'] for u, v in E) + tuple(
        d2roots[t, r] for t, r in stars
    )

    #############
    # Variables #
    #############

    link_ = {(u, v): m.addVar(f'link_{u}~{v}', 'B') for u, v in chain(E, Eʹ)}
    link_ |= {(t, r): m.addVar(f'link_{t}~r{-r}', 'B') for t, r in stars}
    flow_ = {
        (u, v): m.addVar(f'flow_{u}~{v}', 'I', lb=0, ub=k - 1) for u, v in chain(E, Eʹ)
    }
    flow_ |= {(t, r): m.addVar(f'flow_{t}~r{-r}', lb=0, ub=k) for t, r in stars}

    ###############
    # Constraints #
    ###############

    # total number of edges must be equal to number of terminal nodes
    m.addCons(sum(link_.values()) == T, name='num_links_eq_T')

    # enforce a single directed edge between each node pair
    for u, v in E:
        m.addConsSOS1((link_[(u, v)], link_[(v, u)]), name=f'single_dir_link_{u}~{v}')

    # feeder-edge crossings
    if feeder_route is FeederRoute.STRAIGHT:
        for (u, v), (r, t) in gateXing_iter(A):
            if u >= 0:
                m.addConsSOS1(
                    (link_[(u, v)], link_[(v, u)], link_[t, r]),
                    name=f'feeder_link_cross_{u}~{v}_{t}~r{-r}',
                )
            else:
                # a feeder crossing another feeder (possible in multi-root instances)
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

    # bind flow to link activation
    for t, n in linkset:
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

    # feeder limits
    min_feeders = math.ceil(T / k)
    all_feeder_vars_sum = sum(link_[t, r] for r in _R for t in _T)
    if feeder_limit is FeederLimit.UNLIMITED:
        # valid inequality: number of feeders is at least the minimum
        m.addCons(all_feeder_vars_sum >= min_feeders, name='feeder_limit_lb')
        if balanced:
            warn(
                'Model option <balanced = True> is incompatible with <feeder_limit'
                ' = UNLIMITED>: model will not enforce balanced subtrees.'
            )
    else:
        is_equal_not_range = False
        if feeder_limit is FeederLimit.SPECIFIED:
            if max_feeders == min_feeders:
                is_equal_not_range = True
            elif max_feeders < min_feeders:
                raise ValueError('max_feeders is below the minimum necessary')
        elif feeder_limit is FeederLimit.MINIMUM:
            is_equal_not_range = True
        elif feeder_limit is FeederLimit.MIN_PLUS1:
            max_feeders = min_feeders + 1
        elif feeder_limit is FeederLimit.MIN_PLUS2:
            max_feeders = min_feeders + 2
        elif feeder_limit is FeederLimit.MIN_PLUS3:
            max_feeders = min_feeders + 3
        else:
            raise NotImplementedError('Unknown value:', feeder_limit)
        if is_equal_not_range:
            m.addCons(all_feeder_vars_sum == min_feeders, name='feeder_limit_eq')
        else:
            m.addCons(all_feeder_vars_sum >= min_feeders, name='feeder_limit_lb')
            m.addCons(all_feeder_vars_sum <= max_feeders, name='feeder_limit_ub')
        # enforce balanced subtrees (subtree loads differ at most by one unit)
        if balanced:
            if is_equal_not_range:
                feeder_min_load = T // min_feeders
                if feeder_min_load < capacity:
                    for t, r in stars:
                        m.addCons(
                            flow_[t, r] >= link_[t, r] * feeder_min_load,
                            name=f'balanced_{t}~r{-r}',
                        )
            else:
                warn(
                    'Model option <balanced = True> is incompatible with '
                    'having a range of possible feeder counts: model will '
                    'not enforce balanced subtrees.'
                )

    # radial or branched topology
    if topology is Topology.RADIAL:
        for t in _T:
            m.addConsSOS1(
                sum(link_[n, t] for n in A_terminals.neighbors(t)), name=f'radial_{t}'
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
        k,
        linkset,
        link_,
        flow_,
        model_options,
        _make_min_length_model_fingerprint,
    )

    return m, metadata


_make_min_length_model_fingerprint = fun_fingerprint(make_min_length_model)


def warmup_model(model: Model, metadata: ModelMetadata, S: nx.Graph) -> Model:
    """Set initial solution into `model`.

    Changes `model` and `metadata` in-place.

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
