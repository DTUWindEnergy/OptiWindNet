# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import math
import os
from datetime import timedelta
from itertools import chain
from typing import Any

import networkx as nx
from ortools.math_opt.python import mathopt

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
_SOLVER_TYPES = {
    'cp_sat': mathopt.SolverType.CP_SAT,
    'gscip': mathopt.SolverType.GSCIP,
    'gurobi': mathopt.SolverType.GUROBI,
    'highs': mathopt.SolverType.HIGHS,
}
_CALLBACK_BACKENDS = ('cp_sat', 'gurobi')



class _SolutionStore:
    """Ad hoc callback implementation that stores solutions to a pool."""

    solutions: list[tuple[float, dict]]

    def __init__(self, metadata: ModelMetadata, weight_by_link_id: dict[int, float]):
        self.metadata = metadata
        self.weight_by_link_id = weight_by_link_id
        self.solutions = []

    def on_solution_callback(
        self, cb_data: mathopt.CallbackData
    ) -> mathopt.CallbackResult:
        if cb_data.event is not mathopt.Event.MIP_SOLUTION or cb_data.solution is None:
            return mathopt.CallbackResult()
        solution = {
            var.id: round(cb_data.solution[var]) for var in self.metadata.link_.values()
        }
        solution |= {
            var.id: round(cb_data.solution[var]) for var in self.metadata.flow_.values()
        }
        objective_value = sum(
            weight * solution[var_id] for var_id, weight in self.weight_by_link_id.items()
        )
        self.solutions.append((objective_value, solution))
        return mathopt.CallbackResult()


class SolverORTools(Solver, PoolHandler):
    """OR-Tools MathOpt wrapper using the selected backend.

    This class wraps and changes the behavior of MathOpt in order to save all
    solutions found to a pool. Meant to be used with `investigate_pool()`.
    """

    name: str
    backend: str
    log_callback: Any
    _solution_pool: list[tuple[float, dict]]
    _solve_result: mathopt.SolveResult | None

    def __init__(self, backend: str):
        if backend not in _SOLVER_TYPES:
            raise ValueError(f'Unsupported OR-Tools MathOpt backend: {backend}')
        self.backend = backend
        self.name = f'ortools.{backend}'
        self.log_callback = None
        self._solve_result = None
        self.options = {}

    def _link_val(self, var: Any) -> int:
        return self._value_map[var.id]

    def _flow_val(self, var: Any) -> int:
        return self._value_map[var.id]

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
        """Wrapper for MathOpt solve() that saves all solutions.

        This method uses a MIP_SOLUTION callback to fill a solution pool stored
        in the attribute self.solutions.
        """
        try:
            model = self.model
        except AttributeError as exc:
            exc.args += ('.set_problem() must be called before .solve()',)
            raise
        tracked_vars = tuple(chain(self.metadata.link_.values(), self.metadata.flow_.values()))
        storer = _SolutionStore(
            self.metadata,
            {
                var.id: weight
                for var, weight in zip(
                    self.metadata.link_.values(), self.metadata.weight_
                )
            },
        )
        applied_options = self.options | options
        self.stopping = dict(mip_gap=mip_gap, time_limit=time_limit)
        solve_params = self._make_solve_parameters(
            time_limit, mip_gap, applied_options, verbose
        )
        info('>>> ORTools %s parameters <<<\n%s\n', self.backend, solve_params)
        model_params = mathopt.ModelSolveParameters(
            variable_values_filter=mathopt.VariableFilter(filtered_items=tracked_vars),
            solution_hints=(
                [mathopt.SolutionHint(variable_values=self.metadata.solution_hint)]
                if self.metadata.solution_hint
                else []
            ),
        )
        solve_kwargs = dict(
            opt_model=model,
            solver_type=_SOLVER_TYPES[self.backend],
            params=solve_params,
            model_params=model_params,
            msg_cb=self._msg_cb if self.log_callback is not None else None,
        )
        if self.backend in _CALLBACK_BACKENDS:
            solve_kwargs['callback_reg'] = mathopt.CallbackRegistration(
                events={mathopt.Event.MIP_SOLUTION},
                mip_solution_filter=mathopt.VariableFilter(filtered_items=tracked_vars),
            )
            solve_kwargs['cb'] = storer.on_solution_callback
        result = mathopt.solve(**solve_kwargs)
        self._solve_result = result
        if len(storer.solutions) == 0 and result.has_primal_feasible_solution():
            solution = {
                var.id: round(value)
                for var, value in zip(tracked_vars, result.variable_values(tracked_vars))
            }
            storer.solutions.append((result.objective_value(), solution))
        num_solutions = len(storer.solutions)
        termination = result.termination.reason.name
        if num_solutions == 0:
            raise OWNSolutionNotFound(
                f'Unable to find a solution. Solver {self.name} terminated with: {termination}'
            )
        storer.solutions.reverse()
        self._solution_pool = storer.solutions
        _, self._value_map = storer.solutions[0]
        self.num_solutions = num_solutions
        bound = result.best_objective_bound()
        objective = result.objective_value()
        solution_info = SolutionInfo(
            runtime=result.solve_stats.solve_time.total_seconds(),
            bound=bound,
            objective=objective,
            relgap=1.0 - bound / objective,
            termination=termination,
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
        G.graph['solver_details'].update(strategy=self._solver_termination_detail())
        return S, G

    def _objective_at(self, index: int) -> float:
        objective_value, self._value_map = self._solution_pool[index]
        return objective_value

    def _topology_from_mip_pool(self) -> nx.Graph:
        return self._topology_from_mip_sol()

    def _solver_termination_detail(self) -> str:
        if self._solve_result is None:
            return ''
        termination = self._solve_result.termination
        detail = f' ({termination.detail})' if termination.detail else ''
        return f'{termination.reason.name}{detail}'

    def _msg_cb(self, messages: list[str]) -> None:
        for line in messages:
            self.log_callback(line)

    def _make_solve_parameters(
        self,
        time_limit: float,
        mip_gap: float,
        applied_options: dict[str, Any],
        verbose: bool,
    ) -> mathopt.SolveParameters:
        threads = applied_options.pop('threads', len(os.sched_getaffinity(0)))
        # SolveParameters.threads is only honoured by cp_sat and gscip;
        # other backends need it injected into their own parameter sub-messages.
        solve_params = mathopt.SolveParameters(
            time_limit=timedelta(seconds=time_limit),
            relative_gap_tolerance=mip_gap,
            enable_output=verbose,
            threads=threads if self.backend in ('cp_sat', 'gscip') else None,
        )
        match self.backend:
            case 'cp_sat':
                for key, val in applied_options.items():
                    setattr(solve_params.cp_sat, key, val)
                solve_params.cp_sat.log_search_progress = verbose
            case 'gurobi':
                solve_params.gurobi.param_values['Threads'] = str(threads)
                for key, val in applied_options.items():
                    solve_params.gurobi.param_values[key] = str(val)
            case 'gscip':
                for key, val in applied_options.items():
                    setattr(solve_params.gscip, key, val)
            case 'highs':
                solve_params.highs.int_options['threads'] = threads
                for key, val in applied_options.items():
                    setattr(solve_params.highs, key, val)
            case _:
                raise ValueError(f'Unsupported OR-Tools MathOpt backend: {self.backend}')
        return solve_params


def make_min_length_model(
    A: nx.Graph,
    capacity: int,
    *,
    topology: Topology = Topology.BRANCHED,
    feeder_route: FeederRoute = FeederRoute.SEGMENTED,
    feeder_limit: FeederLimit = FeederLimit.UNLIMITED,
    balanced: bool = False,
    max_feeders: int = 0,
) -> tuple[mathopt.Model, ModelMetadata]:
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
    m = mathopt.Model(name='optiwindnet')

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

    link_ = {
        (u, v): m.add_binary_variable(name=f'link_{u}~{v}')
        for u, v in chain(E, Eʹ)
    }
    link_ |= {(t, r): m.add_binary_variable(name=f'link_{t}~r{-r}') for t, r in stars}
    flow_ = {
        (u, v): m.add_integer_variable(lb=0, ub=k - 1, name=f'flow_{u}~{v}')
        for u, v in chain(E, Eʹ)
    }
    flow_ |= {
        (t, r): m.add_integer_variable(lb=0, ub=k, name=f'flow_{t}~r{-r}')
        for t, r in stars
    }

    ###############
    # Constraints #
    ###############

    # total number of edges must be equal to number of terminal nodes
    m.add_linear_constraint(sum(link_.values()) == T, name='num_links_eq_T')

    # enforce a single directed edge between each node pair
    for u, v in E:
        m.add_linear_constraint(
            link_[(u, v)] + link_[(v, u)] <= 1,
            name=f'single_dir_link_{u}~{v}',
        )

    # feeder-edge crossings
    if feeder_route is FeederRoute.STRAIGHT:
        for (u, v), (r, t) in gateXing_iter(A):
            if u >= 0:
                m.add_linear_constraint(
                    link_[(u, v)] + link_[(v, u)] + link_[t, r] <= 1,
                    name=f'feeder_link_cross_{u}~{v}_{t}~r{-r}',
                )
            else:
                # a feeder crossing another feeder (possible in multi-root instances)
                m.add_linear_constraint(
                    link_[(u, v)] + link_[t, r] <= 1,
                    name=f'feeder_feeder_cross_r{-u}~{v}_{t}~r{-r}',
                )

    # edge-edge crossings
    for Xing in edgeset_edgeXing_iter(A.graph['diagonals']):
        m.add_linear_constraint(
            sum(link_[a, b] for u, v in Xing for a, b in ((u, v), (v, u))) <= 1,
            name=f'link_link_cross_{"_".join(f"{u}~{v}" for u, v in Xing)}',
        )

    # bind flow to link activation
    for t, n in linkset:
        _n = str(n) if n >= 0 else f'r{-n}'
        m.add_linear_constraint(
            expr=flow_[t, n] - (k if n < 0 else (k - 1)) * link_[t, n],
            ub=0,
            name=f'flow_zero_{t}~{_n}',
        )
        m.add_linear_constraint(
            expr=flow_[t, n] - link_[t, n],
            lb=0,
            name=f'flow_nonzero_{t}~{_n}',
        )

    # flow conservation with possibly non-unitary node power
    for t in _T:
        m.add_linear_constraint(
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
        m.add_linear_constraint(
            all_feeder_vars_sum >= min_feeders, name='feeder_limit_lb'
        )
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
            m.add_linear_constraint(
                all_feeder_vars_sum == min_feeders, name='feeder_limit_eq'
            )
        else:
            m.add_linear_constraint(
                expr=all_feeder_vars_sum,
                lb=min_feeders,
                ub=max_feeders,
                name='feeder_limit_interval',
            )
        # enforce balanced subtrees (subtree loads differ at most by one unit)
        if balanced:
            if is_equal_not_range:
                feeder_min_load = T // min_feeders
                if feeder_min_load < capacity:
                    for t, r in stars:
                        m.add_linear_constraint(
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
            m.add_linear_constraint(
                expr=sum(link_[n, t] for n in A_terminals.neighbors(t)),
                ub=1,
                name=f'radial_{t}',
            )

    # assert all nodes are connected to some root
    m.add_linear_constraint(
        sum(flow_[t, r] for r in _R for t in _T) == W, name='total_power_sank'
    )

    # valid inequalities
    for t in _T:
        # incoming flow limit
        m.add_linear_constraint(
            expr=sum(flow_[n, t] for n in A_terminals.neighbors(t)),
            ub=k - A.nodes[t].get('power', 1),
            name=f'inflow_limit_{t}',
        )
        # only one out-edge per terminal
        m.add_linear_constraint(
            sum(link_[t, n] for n in chain(A_terminals.neighbors(t), _R)) == 1,
            name=f'single_out_link_{t}',
        )

    #############
    # Objective #
    #############

    m.minimize(sum(weight * var for weight, var in zip(weight_, link_.values())))

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
        weight_=weight_,
    )

    return m, metadata


_make_min_length_model_fingerprint = fun_fingerprint(make_min_length_model)


def warmup_model(
    model: mathopt.Model, metadata: ModelMetadata, S: nx.Graph
) -> mathopt.Model:
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
    hint_values = {}
    for u, v in metadata.linkset[: (len(metadata.linkset) - R * T) // 2]:
        edgeD = S.edges.get((u, v))
        if edgeD is None:
            hint_values[metadata.link_[u, v]] = 0
            hint_values[metadata.flow_[u, v]] = 0
            hint_values[metadata.link_[v, u]] = 0
            hint_values[metadata.flow_[v, u]] = 0
        else:
            u, v = (u, v) if ((u < v) == edgeD['reverse']) else (v, u)
            hint_values[metadata.link_[u, v]] = 1
            hint_values[metadata.flow_[u, v]] = edgeD['load']
            hint_values[metadata.link_[v, u]] = 0
            hint_values[metadata.flow_[v, u]] = 0
    for t, r in metadata.linkset[-R * T :]:
        edgeD = S.edges.get((t, r))
        hint_values[metadata.link_[t, r]] = 0 if edgeD is None else 1
        hint_values[metadata.flow_[t, r]] = 0 if edgeD is None else edgeD['load']
    metadata.solution_hint = hint_values
    metadata.warmed_by = S.graph['creator']
    return model
