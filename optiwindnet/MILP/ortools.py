# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import math
from itertools import chain
from typing import Any

import networkx as nx
from ortools.sat.python import cp_model

from ..crossings import edgeset_edgeXing_iter, gateXing_iter
from ..interarraylib import G_from_S, fun_fingerprint
from ..pathfinding import PathFinder
from ._core import (
    FeederLimit,
    FeederRoute,
    ModelMetadata,
    ModelOptions,
    PoolHandler,
    SolutionInfo,
    Solver,
    Topology,
    investigate_pool,
)

__all__ = ('make_min_length_model', 'warmup_model', 'topology_from_mip_sol')

_lggr = logging.getLogger(__name__)
error, warn, info = _lggr.error, _lggr.warning, _lggr.info


class _SolutionStore(cp_model.CpSolverSolutionCallback):
    """Ad hoc implementation of a callback that stores solutions to a pool."""

    solutions: list[tuple[float, dict]]

    def __init__(self, model: cp_model.CpModel):
        super().__init__()
        self.solutions = []
        int_lits = []
        bool_lits = []
        for var in model._CpModel__var_list._VariableList__var_list:
            if var.is_boolean:
                bool_lits.append(var)
            elif var.is_integer():
                int_lits.append(var)
        self.bool_lits = bool_lits
        self.int_lits = int_lits

    def on_solution_callback(self):
        solution = {var.index: self.boolean_value(var) for var in self.bool_lits}
        solution |= {var.index: self.value(var) for var in self.int_lits}
        self.solutions.append((self.objective_value, solution))


class SolverORTools(Solver, PoolHandler):
    """OR-Tools CpSolver wrapper.

    This class wraps and changes the behavior of CpSolver in order to save all
    solutions found to a pool. Meant to be used with `investigate_pool()`.
    """

    name: str = 'ortools'
    solution_pool: list[tuple[float, dict]]
    solver: cp_model.CpSolver

    def __init__(self):
        self.solver = cp_model.CpSolver()

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
        if warmstart is not None:
            warmup_model(model, metadata, warmstart)
        self.model, self.metadata = model, metadata

    def solve(
        self,
        time_limit: int,
        mip_gap: float,
        options: dict[str, Any] = {},
        verbose: bool = False,
    ) -> SolutionInfo:
        """Wrapper for CpSolver.solve() that saves all solutions.

        This method uses a custom CpSolverSolutionCallback to fill a solution
        pool stored in the attribute self.solutions.
        """
        try:
            model, solver = self.model, self.solver
        except AttributeError as exc:
            exc.args += ('.set_problem() must be called before .solve()',)
            raise
        storer = _SolutionStore(model)
        for key, val in options.items():
            setattr(solver.parameters, key, val)
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.relative_gap_limit = mip_gap
        solver.parameters.log_search_progress = verbose
        info('>>> ORTools CpSat parameters <<<\n%s\n', solver.parameters)
        _ = solver.solve(model, storer)
        storer.solutions.reverse()
        self.solution_pool = storer.solutions
        _, self._value_map = storer.solutions[0]
        self.num_solutions = len(storer.solutions)
        bound = solver.best_objective_bound
        objective = solver.objective_value
        solution_info = SolutionInfo(
            runtime=solver.wall_time,
            bound=bound,
            objective=objective,
            relgap=1.0 - bound / objective,
            termination=solver.status_name(),
        )
        self.solution_info, self.solver_options = solution_info, options
        info('>>> Solution <<<\n%s\n', solution_info)
        return solution_info

    def get_solution(self, A: nx.Graph | None = None) -> tuple[nx.Graph, nx.Graph]:
        if A is None:
            A = self.A
        P, model_options = self.P, self.model_options
        if model_options['feeder_route'] is FeederRoute.STRAIGHT:
            S = self.topology_from_mip_pool()
            G = PathFinder(
                G_from_S(S, A),
                P,
                A,
                branched=model_options['topology'] is Topology.BRANCHED,
            ).create_detours()
        else:
            S, G = investigate_pool(P, A, self)
        G.graph.update(self._make_graph_attributes())
        G.graph['solver_details'].update(strategy=self.solver.solution_info())
        return S, G

    def boolean_value(self, literal: cp_model.IntVar) -> bool:
        return self._value_map[literal.index]

    def value(self, literal: cp_model.IntVar) -> int:
        return self._value_map[literal.index]

    def objective_at(self, index: int) -> float:
        objective_value, self._value_map = self.solution_pool[index]
        return objective_value

    def topology_from_mip_pool(self) -> nx.Graph:
        return topology_from_mip_sol(metadata=self.metadata, solver=self)

    def topology_from_mip_sol(self):
        return topology_from_mip_sol(metadata=self.metadata, solver=self)


def make_min_length_model(
    A: nx.Graph,
    capacity: int,
    *,
    topology: Topology = Topology.BRANCHED,
    feeder_route: FeederRoute = FeederRoute.SEGMENTED,
    feeder_limit: FeederLimit = FeederLimit.UNLIMITED,
    balanced: bool = False,
    max_feeders: int = 0,
) -> tuple[cp_model.CpModel, ModelMetadata]:
    """Make discrete optimization model over link set A.

    Build OR-tools CP-SAT model for the collector system length minimization.

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
    A_nodes = nx.subgraph_view(A, filter_node=lambda n: n >= 0)
    W = sum(w for _, w in A_nodes.nodes(data='power', default=1))

    # Sets
    _T = range(T)
    _R = range(-R, 0)

    E = tuple(((u, v) if u < v else (v, u)) for u, v in A_nodes.edges())
    # using directed node-node links -> create the reversed tuples
    Eʹ = tuple((v, u) for u, v in E)
    # set of feeders to all roots
    stars = tuple((t, r) for t in _T for r in _R)
    linkset = E + Eʹ + stars

    # Create model
    m = cp_model.CpModel()

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

    link_ = {e: m.new_bool_var(f'link_{e}') for e in linkset}
    flow_ = {e: m.new_int_var(0, k - 1, f'flow_{e}') for e in chain(E, Eʹ)}
    flow_ |= {e: m.new_int_var(0, k, f'flow_{e}') for e in stars}

    ###############
    # Constraints #
    ###############

    # total number of edges must be equal to number of terminal nodes
    m.add(sum(link_.values()) == T)

    # enforce a single directed edge between each node pair
    for u, v in E:
        m.add_at_most_one(link_[(u, v)], link_[(v, u)])

    # feeder-edge crossings
    if feeder_route is FeederRoute.STRAIGHT:
        for (u, v), tr in gateXing_iter(A):
            m.add_at_most_one(link_[(u, v)], link_[(v, u)], link_[tr])

    # edge-edge crossings
    for Xing in edgeset_edgeXing_iter(A.graph['diagonals']):
        m.add_at_most_one(sum(((link_[u, v], link_[v, u]) for u, v in Xing), ()))

    # bind flow to link activation
    for t, n in linkset:
        m.add(flow_[t, n] == 0).only_enforce_if(link_[t, n].Not())
        #  m.add(flow_[t, n] <= link_[t, n]*(k if n < 0 else (k - 1)))
        m.add(flow_[t, n] > 0).only_enforce_if(link_[t, n])
        #  m.add(flow_[t, n] >= link_[t, n])

    # flow conservation with possibly non-unitary node power
    for t in _T:
        m.add(
            sum((flow_[t, n] - flow_[n, t]) for n in A_nodes.neighbors(t))
            + sum(flow_[t, r] for r in _R)
            == A.nodes[t].get('power', 1)
        )

    # feeder limits
    min_feeders = math.ceil(T / k)
    all_feeder_vars_sum = sum(link_[t, r] for r in _R for t in _T)
    is_equal_not_bounded = False
    if feeder_limit is FeederLimit.UNLIMITED:
        # valid inequality: number of gates is at least the minimum
        m.add(all_feeder_vars_sum >= min_feeders)
        if balanced:
            warn(
                'Model option <balanced = True> is incompatible with <feeder_limit'
                ' = UNLIMITED>: model will not enforce balanced subtrees.'
            )
    else:
        if feeder_limit is FeederLimit.SPECIFIED:
            if max_feeders == min_feeders:
                is_equal_not_bounded = True
            elif max_feeders < min_feeders:
                raise ValueError('max_feeders is below the minimum necessary')
        elif feeder_limit is FeederLimit.MINIMUM:
            is_equal_not_bounded = True
        elif feeder_limit is FeederLimit.MIN_PLUS1:
            max_feeders = min_feeders + 1
        elif feeder_limit is FeederLimit.MIN_PLUS2:
            max_feeders = min_feeders + 2
        elif feeder_limit is FeederLimit.MIN_PLUS3:
            max_feeders = min_feeders + 3
        else:
            raise NotImplementedError('Unknown value:', feeder_limit)
        if is_equal_not_bounded:
            m.add(all_feeder_vars_sum == min_feeders)
        else:
            m.add_linear_constraint(all_feeder_vars_sum, min_feeders, max_feeders)
        # enforce balanced subtrees (subtree loads differ at most by one unit)
        if balanced:
            if not is_equal_not_bounded:
                warn(
                    'Model option <balanced = True> is incompatible with '
                    'having a range of possible feeder counts: model will '
                    'not enforce balanced subtrees.'
                )
            else:
                feeder_min_load = T // min_feeders
                if feeder_min_load < capacity:
                    for t, r in stars:
                        m.add(flow_[t, r] >= link_[t, r] * feeder_min_load)

    # radial or branched topology
    if topology is Topology.RADIAL:
        for t in _T:
            m.add(sum(link_[n, t] for n in A_nodes.neighbors(t)) <= 1)

    # assert all nodes are connected to some root
    m.add(sum(flow_[t, r] for r in _R for t in _T) == W)

    # valid inequalities
    for t in _T:
        # incoming flow limit
        m.add(
            sum(flow_[n, t] for n in A_nodes.neighbors(t))
            <= k - A.nodes[t].get('power', 1)
        )
        # only one out-edge per terminal
        m.add(sum(link_[t, n] for n in chain(A_nodes.neighbors(t), _R)) == 1)

    #############
    # Objective #
    #############

    m.minimize(cp_model.LinearExpr.WeightedSum(tuple(link_.values()), weight_))

    ##################
    # Store metadata #
    ##################

    model_options = dict(
        topology=topology,
        feeder_route=feeder_route,
        feeder_limit=feeder_limit,
        max_feeders=max_feeders,
    )
    metadata = ModelMetadata(
        R, T, k, linkset, link_, flow_, model_options, fun_fingerprint()
    )

    return m, metadata


def warmup_model(
    model: cp_model.CpModel, metadata: ModelMetadata, S: nx.Graph
) -> cp_model.CpModel:
    """Set initial solution into `model`.

    Changes `model` in-place.

    Args:
      model: CP-SAT model to apply the solution to.
      metadata: indices to the model's variables.
      S: solution topology

    Returns:
      The same model instance that was provided, now with a solution.
    """
    R, T = metadata.R, metadata.T
    model.ClearHints()
    for u, v in metadata.linkset[: (len(metadata.linkset) - R * T) // 2]:
        edgeD = S.edges.get((u, v))
        if edgeD is None:
            model.add_hint(metadata.link_[u, v], False)
            model.add_hint(metadata.flow_[u, v], 0)
            model.add_hint(metadata.link_[v, u], False)
            model.add_hint(metadata.flow_[v, u], 0)
        else:
            u, v = (u, v) if ((u < v) == edgeD['reverse']) else (v, u)
            model.add_hint(metadata.link_[u, v], True)
            model.add_hint(metadata.flow_[u, v], edgeD['load'])
            model.add_hint(metadata.link_[v, u], False)
            model.add_hint(metadata.flow_[v, u], 0)
    for t, r in metadata.linkset[-R * T :]:
        edgeD = S.edges.get((t, r))
        model.add_hint(metadata.link_[t, r], edgeD is not None)
        model.add_hint(metadata.flow_[t, r], 0 if edgeD is None else edgeD['load'])
    metadata.warmed_by = S.graph['creator']
    return model


def topology_from_mip_sol(
    *, metadata: ModelMetadata, solver: SolverORTools | cp_model.CpSolver, **kwargs
) -> nx.Graph:
    """Create a topology graph from the OR-tools solution to the MILP model.

    Args:
      metadata: attributes of the solved model
      solver: solver instance that solved the model
      kwargs: not used (signature compatibility)
    Returns:
      Graph topology `S` from the solution.
    """
    # in ortools, the solution is in the solver instance not in the model
    S = nx.Graph(R=metadata.R, T=metadata.T)
    # Get active links and if flow is reversed (i.e. from small to big)
    rev_from_link = {
        (u, v): u < v
        for (u, v), use in metadata.link_.items()
        if solver.boolean_value(use)
    }
    S.add_weighted_edges_from(
        ((u, v, solver.value(metadata.flow_[u, v])) for (u, v) in rev_from_link.keys()),
        weight='load',
    )
    # set the 'reverse' edge attribute
    nx.set_edge_attributes(S, rev_from_link, name='reverse')
    # propagate loads from edges to nodes
    subtree = -1
    max_load = 0
    for r in range(-metadata.R, 0):
        for u, v in nx.edge_dfs(S, r):
            S.nodes[v]['load'] = S[u][v]['load']
            if u == r:
                subtree += 1
            S.nodes[v]['subtree'] = subtree
        rootload = 0
        for nbr in S.neighbors(r):
            subtree_load = S.nodes[nbr]['load']
            max_load = max(max_load, subtree_load)
            rootload += subtree_load
        S.nodes[r]['load'] = rootload
    S.graph.update(
        capacity=metadata.capacity,
        max_load=max_load,
        has_loads=True,
        creator='MILP.' + __name__,
        solver_details={},
    )
    return S
