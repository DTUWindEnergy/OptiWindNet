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
    _solution_pool: list[tuple[float, dict]]
    solver: cp_model.CpSolver

    def __init__(self):
        self.solver = cp_model.CpSolver()
        # set default options for ortools
        self.options = {}

    def _link_val(self, var: Any) -> int:
        return self._value_map[var.index]

    def _flow_val(self, var: Any) -> int:
        return self._value_map[var.index]

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
        applied_options = self.options | options
        for key, val in applied_options.items():
            setattr(solver.parameters, key, val)
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.relative_gap_limit = mip_gap
        self.stopping = dict(mip_gap=mip_gap, time_limit=time_limit)
        solver.parameters.log_search_progress = verbose
        info('>>> ORTools CpSat parameters <<<\n%s\n', solver.parameters)
        _ = solver.solve(model, storer)
        num_solutions = len(storer.solutions)
        if num_solutions == 0:
            raise OWNSolutionNotFound(
                f'Unable to find a solution. Solver {self.name} terminated with: {solver.status_name()}'
            )
        storer.solutions.reverse()
        self._solution_pool = storer.solutions
        _, self._value_map = storer.solutions[0]
        self.num_solutions = num_solutions
        bound = solver.best_objective_bound
        objective = solver.objective_value
        solution_info = SolutionInfo(
            runtime=solver.wall_time,
            bound=bound,
            objective=objective,
            relgap=1.0 - bound / objective,
            termination=solver.status_name(),
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
        G.graph['solver_details'].update(strategy=self.solver.solution_info())
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

    link_ = {(u, v): m.new_bool_var(f'link_{u}~{v}') for u, v in chain(E, Eʹ)}
    link_ |= {(t, r): m.new_bool_var(f'link_{t}~r{-r}') for t, r in stars}
    flow_ = {(u, v): m.new_int_var(0, k - 1, f'flow_{u}~{v}') for u, v in chain(E, Eʹ)}
    flow_ |= {(t, r): m.new_int_var(0, k, f'flow_{t}~r{-r}') for t, r in stars}

    ###############
    # Constraints #
    ###############

    # total number of edges must be equal to number of terminal nodes
    m.add(sum(link_.values()) == T).with_name('num_links_eq_T')

    # enforce a single directed edge between each node pair
    for u, v in E:
        m.add_at_most_one(link_[(u, v)], link_[(v, u)]).with_name(
            f'single_dir_link_{u}~{v}'
        )

    # feeder-edge crossings
    if feeder_route is FeederRoute.STRAIGHT:
        for (u, v), (r, t) in gateXing_iter(A):
            if u >= 0:
                m.add_at_most_one(link_[(u, v)], link_[(v, u)], link_[t, r]).with_name(
                    f'feeder_link_cross_{u}~{v}_{t}~r{-r}'
                )
            else:
                # a feeder crossing another feeder (possible in multi-root instances)
                m.add_at_most_one(link_[(u, v)], link_[t, r]).with_name(
                    f'feeder_feeder_cross_r{-u}~{v}_{t}~r{-r}'
                )

    # edge-edge crossings
    for Xing in edgeset_edgeXing_iter(A.graph['diagonals']):
        m.add_at_most_one(
            sum(((link_[u, v], link_[v, u]) for u, v in Xing), ())
        ).with_name(f'link_link_cross_{"_".join(f"{u}~{v}" for u, v in Xing)}')

    # bind flow to link activation
    for t, n in linkset:
        _n = str(n) if n >= 0 else f'r{-n}'
        m.add(flow_[t, n] == 0).only_enforce_if(link_[t, n].Not()).with_name(
            f'flow_zero_{t}~{_n}'
        )
        #  m.add(flow_[t, n] <= link_[t, n]*(k if n < 0 else (k - 1)))
        m.add(flow_[t, n] > 0).only_enforce_if(link_[t, n]).with_name(
            f'flow_nonzero_{t}~{_n}'
        )
        #  m.add(flow_[t, n] >= link_[t, n])

    # flow conservation with possibly non-unitary node power
    for t in _T:
        m.add(
            sum((flow_[t, n] - flow_[n, t]) for n in A_terminals.neighbors(t))
            + sum(flow_[t, r] for r in _R)
            == A.nodes[t].get('power', 1)
        ).with_name(f'flow_conserv_{t}')

    # feeder limits
    min_feeders = math.ceil(T / k)
    all_feeder_vars_sum = sum(link_[t, r] for r in _R for t in _T)
    if feeder_limit is FeederLimit.UNLIMITED:
        # valid inequality: number of feeders is at least the minimum
        m.add(all_feeder_vars_sum >= min_feeders).with_name('feeder_limit_lb')
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
            m.add(all_feeder_vars_sum == min_feeders).with_name('feeder_limit_eq')
        else:
            m.add_linear_constraint(
                all_feeder_vars_sum, min_feeders, max_feeders
            ).with_name('feeder_limit_interval')
        # enforce balanced subtrees (subtree loads differ at most by one unit)
        if balanced:
            if is_equal_not_range:
                feeder_min_load = T // min_feeders
                if feeder_min_load < capacity:
                    for t, r in stars:
                        m.add(flow_[t, r] >= link_[t, r] * feeder_min_load).with_name(
                            f'balanced_{t}~r{-r}'
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
            m.add(sum(link_[n, t] for n in A_terminals.neighbors(t)) <= 1).with_name(
                f'radial_{t}'
            )

    # assert all nodes are connected to some root
    m.add(sum(flow_[t, r] for r in _R for t in _T) == W).with_name('total_power_sank')

    # valid inequalities
    for t in _T:
        # incoming flow limit
        m.add(
            sum(flow_[n, t] for n in A_terminals.neighbors(t))
            <= k - A.nodes[t].get('power', 1)
        ).with_name(f'inflow_limit_{t}')
        # only one out-edge per terminal
        m.add(
            sum(link_[t, n] for n in chain(A_terminals.neighbors(t), _R)) == 1
        ).with_name(f'single_out_link_{t}')

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


def warmup_model(
    model: cp_model.CpModel, metadata: ModelMetadata, S: nx.Graph
) -> cp_model.CpModel:
    """Set initial solution into `model`.

    Changes `model` and `metadata` in-place.

    Args:
      model: CP-SAT model to apply the solution to.
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
