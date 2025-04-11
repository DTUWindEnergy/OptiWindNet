# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import math
import logging
from typing import Any
from itertools import chain
import networkx as nx

from ortools.sat.python import cp_model

from .core import Solver, PoolHandler, investigate_pool
from ..pathfinding import PathFinder
from ..interarraylib import G_from_S
from ..crossings import edgeset_edgeXing_iter, gateXing_iter
from ..interarraylib import fun_fingerprint

logger = logging.getLogger(__name__)
info = logger.info


class _SolutionStore(cp_model.CpSolverSolutionCallback):
    '''Ad hoc implementation of a callback that stores solutions to a pool.'''
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


class SolverORTools(cp_model.CpSolver, Solver, PoolHandler):
    '''OR-Tools CpSolver wrapper.

    This class wraps and changes the behavior of CpSolver in order to save all
    solutions found to a pool. Meant to be used with `investigate_pool()`.
    '''
    name: str = 'ortools'
    solution_pool: list[tuple[float, dict]]

    def __init__(self):
        super().__init__()

    def set_problem(self, P: nx.PlanarEmbedding, A: nx.Graph, capacity: int,
                     warmstart: nx.Graph | None = None, **kwargs):
        'kwargs contains problem options'
        self.P = P
        self.A = A
        self.capacity = capacity
        model = make_min_length_model(self.A, self.capacity, **kwargs)
        if warmstart is not None:
            warmup_model(model, warmstart)
        self.model = model

    def solve(self, timelimit: int, mipgap: float,
              options: dict[str, Any] = {}, verbose: bool = False) -> tuple:
    #  def solve(self, model: cp_model.CpModel) -> cp_model.cp_model_pb2.CpSolverStatus:
        '''Wrapper for CpSolver.solve() that saves all solutions.

        This method uses a custom CpSolverSolutionCallback to fill a solution
        pool stored in the attribute self.solutions.
        '''
        model = self.model
        storer = _SolutionStore(model)
        self.parameters.max_time_in_seconds = timelimit
        self.parameters.relative_gap_limit = mipgap
        for key, val in options:
            setattr(self.parameters, key, val)
        self.log_callback = print
        self.parameters.log_search_progress = verbose
        result = super().solve(model, storer)
        storer.solutions.reverse()
        self.solution_pool = storer.solutions
        _, self._value_map = storer.solutions[0]
        self.num_solutions = len(storer.solutions)
        return result

    def get_solution(self) -> nx.Graph:
        P, A, model = self.P, self.A, self.model
        if model.method_options['gateXings_constraint']:
            S = self.S_from_pool()
            G = PathFinder(G_from_S(S, A), P, A).create_detours()
        else:
            G = investigate_pool(P, A, self)
        return G

    def boolean_value(self, literal: cp_model.IntVar) -> bool:
        return self._value_map[literal.index]

    def value(self, literal: cp_model.IntVar) -> int:
        return self._value_map[literal.index]

    def objective_at(self, index:int) -> float:
        objective_value, self._value_map = self.solution_pool[index]
        return objective_value

    def S_from_pool(self) -> nx.Graph:
        return S_from_solution(self.model, solver=self)


def make_min_length_model(A: nx.Graph, capacity: int, *,
                          gateXings_constraint: bool = False,
                          gates_limit: bool | int = False,
                          branching: bool = True) -> cp_model.CpModel:
    '''
    Build ILP CP OR-tools model for the collector system length minimization.
    `A` is the graph with the available edges to choose from.

    `capacity`: cable capacity

    `gateXing_constraint`: if gates and edges are forbidden to cross.

    `gates_limit`: if True, use the minimum feasible number of gates
    (total for all roots); if False, no limit is imposed; if a number,
    use it as the limit.

    `branching`: if root branches are paths (False) or can be trees (True).
    '''
    R = A.graph['R']
    T = A.graph['T']
    d2roots = A.graph['d2roots']
    A_nodes = nx.subgraph_view(A, filter_node=lambda n: n >= 0)
    W = sum(w for _, w in A_nodes.nodes(data='power', default=1))

    # Sets
    _T = range(T)
    _R = range(-R, 0)

    E = tuple(((u, v) if u < v else (v, u))
              for u, v in A_nodes.edges())
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
    weight_ = (2*tuple(A[u][v]['length'] for u, v in E)
              + tuple(d2roots[t, r] for t, r in stars))

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

    # gate-edge crossings
    if gateXings_constraint:
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
        m.add(sum((flow_[t, n] - flow_[n, t]) for n in A_nodes.neighbors(t))
              + sum(flow_[t, r] for r in _R)
              == A.nodes[t].get('power', 1))

    # gates limit
    min_gates = math.ceil(T/k)
    all_gate_vars_sum = sum(link_[t, r] for r in _R for t in _T)
    if gates_limit:
        if isinstance(gates_limit, bool) or gates_limit == min_gates:
            # fixed number of gates
            m.add(all_gate_vars_sum == min_gates)
        else:
            assert min_gates < gates_limit, (
                    f'Infeasible: T/k > gates_limit (T = {T}, k = {k},'
                    f' gates_limit = {gates_limit}).')
            # number of gates within range
            m.add_linear_constraint(all_gate_vars_sum, min_gates, gates_limit)
    else:
        # valid inequality: number of gates is at least the minimum
        m.add(all_gate_vars_sum >= min_gates)

    # non-branching
    if not branching:
        for t in _T:
            m.add(sum(link_[n, t] for n in A_nodes.neighbors(t)) <= 1)

    # assert all nodes are connected to some root
    m.add(sum(flow_[t, r] for r in _R for t in _T) == W)

    # valid inequalities
    for t in _T:
        # incoming flow limit
        m.add(sum(flow_[n, t] for n in A_nodes.neighbors(t))
              <= k - A.nodes[t].get('power', 1))
        # only one out-edge per terminal
        m.add(sum(link_[t, n] for n in chain(A_nodes.neighbors(t), _R)) == 1)

    #############
    # Objective #
    #############

    m.minimize(cp_model.LinearExpr.WeightedSum(tuple(link_.values()), weight_))

    # save data structure as model attributes
    m.link_, m.linkset, m.flow_, m.R, m.T, m.k = link_, linkset, flow_, R, T, k

    ##################
    # Store metadata #
    ##################

    m.handle = A.graph['handle']
    m.name = A.graph.get('name', 'unnamed')
    m.method_options = dict(gateXings_constraint=gateXings_constraint,
                            gates_limit=gates_limit,
                            branching=branching)
    m.fun_fingerprint = fun_fingerprint()
    m.warmed_by = None
    return m


def warmup_model(model: cp_model.CpModel, S: nx.Graph) -> cp_model.CpModel:
    '''
    Changes `model` in-place.
    '''
    R, T = model.R, model.T
    model.ClearHints()
    for u, v in model.linkset[:(len(model.linkset) - R*T)//2]:
        edgeD = S.edges.get((u, v))
        if edgeD is None:
            model.add_hint(model.link_[u, v], False)
            model.add_hint(model.flow_[u, v], 0)
            model.add_hint(model.link_[v, u], False)
            model.add_hint(model.flow_[v, u], 0)
        else:
            u, v = (u, v) if ((u < v) == edgeD['reverse']) else (v, u)
            model.add_hint(model.link_[u, v], True)
            model.add_hint(model.flow_[u, v], edgeD['load'])
            model.add_hint(model.link_[v, u], False)
            model.add_hint(model.flow_[v, u], 0)
    for t, r in model.linkset[-R*T:]:
        edgeD = S.edges.get((t, r))
        model.add_hint(model.link_[t, r], edgeD is not None)
        model.add_hint(model.flow_[t, r], 0 if edgeD is None else edgeD['load'])
    model.warmed_by = S.graph['creator']
    return model


def S_from_solution(model: cp_model.CpModel,
                    solver: cp_model.CpSolver, result: int = 0) -> nx.Graph:
    '''Create a topology `S` from the OR-tools solution to the MILP model.

    Args:
        model: passed to the solver.
        solver: used to solve the model.
        result: irrelevant, exists only to mirror the Pyomo alternative.
    Returns:
        Graph topology from the solution.
    '''
    # the solution is in the solver object not in the model

    # Metadata
    R, T = model.R, model.T
    bound = solver.best_objective_bound
    objective = solver.objective_value
    S = nx.Graph(
        R=R, T=T,
        handle=model.handle,
        capacity=model.k,
        objective=objective,
        bound=bound,
        runtime=solver.wall_time,
        termination=solver.status_name(),
        gap=1. - bound/objective,
        creator='MILP.' + solver.name,
        has_loads=True,
        method_options=dict(
            solver_name=solver.name,
            mipgap=solver.parameters.relative_gap_limit,
            timelimit=solver.parameters.max_time_in_seconds,
            fun_fingerprint=model.fun_fingerprint,
            **model.method_options,
        ),
        solver_details=dict(
            strategy=solver.solution_info(),
        )
    )

    if model.warmed_by is not None:
        S.graph['warmstart'] = model.warmed_by

    # Graph data
    # Get active links and if flow is reversed (i.e. from small to big)
    rev_from_link = {(u, v): u < v
                     for (u, v), use in model.link_.items()
                     if solver.boolean_value(use)}
    S.add_weighted_edges_from(
        ((u, v, solver.value(model.flow_[u, v]))
         for (u, v) in rev_from_link.keys()), weight='load'
    )
    # set the 'reverse' edge attribute
    nx.set_edge_attributes(S, rev_from_link, name='reverse')
    # propagate loads from edges to nodes
    subtree = -1
    for r in range(-R, 0):
        for u, v in nx.edge_dfs(S, r):
            S.nodes[v]['load'] = S[u][v]['load']
            if u == r:
                subtree += 1
            S.nodes[v]['subtree'] = subtree
        rootload = 0
        for nbr in S.neighbors(r):
            rootload += S.nodes[nbr]['load']
        S.nodes[r]['load'] = rootload
    return S
