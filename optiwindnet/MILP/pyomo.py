# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import math
import os
from collections import namedtuple
from itertools import chain
from typing import Any

import networkx as nx
import pyomo.environ as pyo
from pyomo.util.infeasible import find_infeasible_constraints

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
    SolutionInfo,
    Solver,
    Topology,
)

__all__ = ('make_min_length_model', 'warmup_model')

_lggr = logging.getLogger(__name__)
error, warn, info = _lggr.error, _lggr.warning, _lggr.info

# solver option name mapping (pyomo should have taken care of this)
_common_options = namedtuple('common_options', 'mip_gap time_limit')
_optkey = dict(
    cplex=_common_options('mipgap', 'timelimit'),
    gurobi=_common_options('mipgap', 'timelimit'),
    cbc=_common_options('ratioGap', 'seconds'),
    highs=_common_options('mip_gap', 'time_limit'),
)
# usage: _optname[solver_name].mipgap

_default_options = dict(
    cbc=dict(
        threads=os.cpu_count(),
        timeMode='elapsed',
        # the parameters below and more can be experimented with
        # http://www.decom.ufop.br/haroldo/files/cbcCommandLine.pdf
        nodeStrategy='downFewest',
        # Heuristics
        Dins='on',
        VndVariableNeighborhoodSearch='on',
        Rens='on',
        Rins='on',
        pivotAndComplement='off',
        proximitySearch='off',
        # Cuts
        gomoryCuts='on',
        mixedIntegerRoundingCuts='on',
        flowCoverCuts='on',
        cliqueCuts='off',
        twoMirCuts='off',
        knapsackCuts='off',
        probingCuts='off',
        zeroHalfCuts='off',
        liftAndProjectCuts='off',
        residualCapacityCuts='off',
    ),
    highs={},
    scip={},
)


class SolverPyomo(Solver):
    def __init__(self, name, prefix='', suffix='', **kwargs) -> None:
        self.name = name
        self.options = _default_options[name]
        self.solver = pyo.SolverFactory(prefix + name + suffix, **kwargs)

    def _link_val(self, var: Any) -> int:
        return round(var.value)

    def _flow_val(self, var: Any) -> int:
        return round(var.value)

    def set_problem(
        self,
        P: nx.PlanarEmbedding,
        A: nx.Graph,
        capacity: int,
        model_options: ModelOptions,
        warmstart: nx.Graph | None = None,
    ):
        self.P, self.A, self.capacity = P, A, capacity
        model, metadata = make_min_length_model(A, capacity, **model_options)
        self.model, self.model_options, self.metadata = model, model_options, metadata
        if warmstart is not None and self.solver.warm_start_capable():
            warmup_model(model, metadata, warmstart)
            self.solve_kwargs = {'warmstart': True}
        else:
            self.solve_kwargs = {}
            if warmstart is not None:
                warn('Solver <%s> is not capable of warm-starting.', self.name)

    def solve(
        self,
        time_limit: float,
        mip_gap: float,
        options: dict[str, Any] = {},
        verbose: bool = False,
    ) -> SolutionInfo:
        try:
            solver, name, model = self.solver, self.name, self.model
        except AttributeError as exc:
            exc.args += ('.set_problem() must be called before .solve()',)
            raise
        applied_options = self.options | options
        self.stopping = dict(mip_gap=mip_gap, time_limit=time_limit)
        solver.options.update(applied_options)
        solver.options.update(
            {
                _optkey[name].time_limit: time_limit,
                _optkey[name].mip_gap: mip_gap,
            }
        )
        info('>>> %s solver options <<<\n%s\n', self.name, solver.options)
        result = solver.solve(
            model, **self.solve_kwargs, tee=verbose, load_solutions=False
        )
        termination = result['Solver'][0]['Termination condition'].name
        if len(result.solution) == 0:
            raise OWNSolutionNotFound(
                f'Unable to find a solution. Solver {self.name} terminated with: {termination}'
            )
        self.result = result
        if self.name != 'scip':
            objective = result['Problem'][0]['Upper bound']
            bound = result['Problem'][0]['Lower bound']
            runtime = result['Solver'][0]['Wallclock time']
        else:
            objective = result['Solver'][0]['Primal bound']
            bound = result['Solver'][0]['Dual bound']
            runtime = result['Solver'][0]['Time']
        solution_info = SolutionInfo(
            runtime=runtime,
            bound=bound,
            objective=objective,
            relgap=1.0 - bound / objective,
            termination=termination,
        )
        self.solution_info, self.applied_options = solution_info, applied_options
        info('>>> Solution <<<\n%s\n', solution_info)
        return solution_info

    def get_solution(self, A: nx.Graph | None = None) -> tuple[nx.Graph, nx.Graph]:
        P, model, model_options = self.P, self.model, self.model_options
        result = self.result
        # hack to prevent warning about the solver not reaching the desired mip_gap
        result.solver.status = pyo.SolverStatus.ok
        model.solutions.load_from(result)
        if A is None:
            A = self.A
        S = self._topology_from_mip_sol()
        S.graph['fun_fingerprint'] = _make_min_length_model_fingerprint
        G = PathFinder(
            G_from_S(S, A),
            P,
            A,
            branched=model_options['topology'] is Topology.BRANCHED,
        ).create_detours()
        G.graph.update(self._make_graph_attributes())
        return S, G


class SolverPyomoAppsi(Solver):
    """As of Pyomo v3.9.4, a new solver inverface (v3) is being introduced. HiGHS is the
    only solver using v3 at that point."""

    def __init__(self, name, solver_cls, **kwargs) -> None:
        self.name = name
        self.options = _default_options[name]
        self.solver = solver_cls(**kwargs)

    def _link_val(self, var: Any) -> int:
        # work-around for HiGHS: use round() to coerce link_ value (should be binary)
        #   values for link_ variables are floats and may be slightly off of 0
        return round(var.value)

    def _flow_val(self, var: Any) -> int:
        return round(var.value)

    def set_problem(
        self,
        P: nx.PlanarEmbedding,
        A: nx.Graph,
        capacity: int,
        model_options: ModelOptions,
        warmstart: nx.Graph | None = None,
    ):
        self.P, self.A, self.capacity = P, A, capacity
        model, metadata = make_min_length_model(A, capacity, **model_options)
        self.model, self.model_options, self.metadata = model, model_options, metadata
        if warmstart is not None and self.solver.warm_start_capable():
            warmup_model(model, metadata, warmstart)
            self.solver.config.warmstart = True
        else:
            if warmstart is not None:
                warn('Solver <%s> is not capable of warm-starting.', self.name)

    def solve(
        self,
        time_limit: float,
        mip_gap: float,
        options: dict[str, Any] = {},
        verbose: bool = False,
    ) -> SolutionInfo:
        try:
            solver, name, model = self.solver, self.name, self.model
        except AttributeError as exc:
            exc.args += ('.set_problem() must be called before .solve()',)
            raise
        applied_options = self.options | options
        stopping = {
            _optkey[name].time_limit: time_limit,
            _optkey[name].mip_gap: mip_gap,
        }
        self.stopping = stopping
        for k, v in (applied_options | stopping).items():
            solver.config[k] = v
        solver.config.load_solution = False
        solver.config.stream_solver = verbose
        info('>>> %s solver options <<<\n%s\n', self.name, solver.config)
        result = solver.solve(model)
        self.result = result
        objective = result.best_feasible_objective
        termination = result.termination_condition.name
        if objective is None:
            raise OWNSolutionNotFound(
                f'Unable to find a solution. Solver {self.name} terminated with: {termination}'
            )
        bound = result.best_objective_bound
        solution_info = SolutionInfo(
            runtime=result.wallclock_time,
            bound=bound,
            objective=objective,
            relgap=1.0 - bound / objective,
            termination=termination,
        )
        self.solution_info, self.applied_options = solution_info, applied_options
        info('>>> Solution <<<\n%s\n', solution_info)
        return solution_info

    def get_solution(self, A: nx.Graph | None = None) -> tuple[nx.Graph, nx.Graph]:
        P, model_options = self.P, self.model_options
        result = self.result
        result.solution_loader.load_vars()
        #  model.solutions.load_from(result)
        if A is None:
            A = self.A
        S = self._topology_from_mip_sol()
        S.graph['fun_fingerprint'] = _make_min_length_model_fingerprint
        G = PathFinder(
            G_from_S(S, A),
            P,
            A,
            branched=model_options['topology'] is Topology.BRANCHED,
        ).create_detours()
        G.graph.update(self._make_graph_attributes())
        return S, G


def make_min_length_model(
    A: nx.Graph,
    capacity: int,
    *,
    topology: Topology = Topology.BRANCHED,
    feeder_route: FeederRoute = FeederRoute.SEGMENTED,
    feeder_limit: FeederLimit = FeederLimit.UNLIMITED,
    balanced: bool = False,
    max_feeders: int = 0,
) -> tuple[pyo.ConcreteModel, ModelMetadata]:
    """Make discrete optimization model over link set A.

    Build ILP Pyomo model for the collector system length minimization.

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

    # Create model
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T - 1)
    m.R = pyo.RangeSet(-R, -1)
    m.linkset = pyo.Set(initialize=E + Eʹ + stars)

    ##############
    # Parameters #
    ##############

    m.k = pyo.Param(domain=pyo.PositiveIntegers, name='capacity', default=capacity)
    m.weight_ = pyo.Param(
        m.linkset,
        domain=pyo.PositiveReals,
        name='link_weight',
        initialize=lambda m, u, v: (
            A.edges[(u, v)]['length'] if v >= 0 else d2roots[u, v]
        ),
    )

    #############
    # Variables #
    #############

    m.link_ = pyo.Var(m.linkset, domain=pyo.Binary, initialize=0)

    def flow_bounds(m, u, v):
        return (0, (m.k if v < 0 else m.k - 1))

    m.flow_ = pyo.Var(
        m.linkset, domain=pyo.NonNegativeIntegers, bounds=flow_bounds, initialize=0
    )

    ###############
    # Constraints #
    ###############

    # total number of edges must be equal to number of non-root nodes
    m.cons_num_links_eq_T = pyo.Constraint(
        rule=(lambda m: sum(m.link_.values()) == T), name='num_links_eq_T'
    )

    # enforce a single directed edge between each node pair
    m.cons_single_dir_link = pyo.Constraint(
        E,
        rule=(lambda m, u, v: m.link_[u, v] + m.link_[v, u] <= 1),
        name='single_dir_link',
    )

    # feeder-edge crossings
    if feeder_route is FeederRoute.STRAIGHT:

        def feederXedge_rule(m, u, v, r, t):
            if u >= 0:
                return m.link_[u, v] + m.link_[v, u] + m.link_[t, r] <= 1
            else:
                # a feeder crossing another feeder (possible in multi-root instances)
                return m.link_[u, v] + m.link_[t, r] <= 1

        m.cons_feeder_cross = pyo.Constraint(
            gateXing_iter(A), rule=feederXedge_rule, name='feeder_cross'
        )

    # edge-edge crossings
    def edgeXedge_rule(m, *vertices):
        lhs = sum(
            (m.link_[u, v] + m.link_[v, u])
            for u, v in zip(vertices[::2], vertices[1::2])
        )
        return lhs <= 1

    doubleXings = []
    tripleXings = []
    for Xing in edgeset_edgeXing_iter(A.graph['diagonals']):
        if len(Xing) == 2:
            doubleXings.append(Xing)
        else:
            tripleXings.append(Xing)
    if doubleXings:
        m.cons_link_pair_cross = pyo.Constraint(
            doubleXings, rule=edgeXedge_rule, name='link_pair_cross'
        )
    if tripleXings:
        m.cons_link_trio_cross = pyo.Constraint(
            tripleXings, rule=edgeXedge_rule, name='link_trio_cross'
        )

    # bind flow to link activation
    m.cons_flow_ub = pyo.Constraint(
        m.linkset,
        rule=(
            lambda m, u, v: m.flow_[(u, v)]
            <= m.link_[(u, v)] * (m.k if v < 0 else (m.k - 1))
        ),
        name='flow_ub',
    )
    m.cons_flow_lb = pyo.Constraint(
        m.linkset,
        rule=(lambda m, u, v: m.link_[(u, v)] <= m.flow_[(u, v)]),
        name='flow_lb',
    )

    # flow conservation with possibly non-unitary node power
    m.cons_flow_conserv = pyo.Constraint(
        m.T,
        rule=(
            lambda m, u: (
                sum((m.flow_[u, v] - m.flow_[v, u]) for v in A_terminals.neighbors(u))
                + sum(m.flow_[u, r] for r in _R)
                == A.nodes[u].get('power', 1)
            )
        ),
        name='flow_conserv',
    )

    # feeder limits
    min_feeders = math.ceil(T / m.k)
    if feeder_limit is FeederLimit.UNLIMITED:
        # valid inequality: number of feeders is at least the minimum
        m.cons_feeder_limit_lb = pyo.Constraint(
            rule=(lambda m: sum(m.link_[t, r] for r in _R for t in _T) >= min_feeders),
            name='feeder_limit_lb',
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
            m.cons_feeder_limit_eq = pyo.Constraint(
                rule=(
                    lambda m: sum(m.link_[t, r] for r in _R for t in _T) == min_feeders
                ),
                name='feeder_limit_eq',
            )
        else:
            m.cons_feeder_limit_lb = pyo.Constraint(
                rule=(
                    lambda m: sum(m.link_[t, r] for r in _R for t in _T) >= min_feeders
                ),
                name='feeder_limit_lb',
            )
            m.cons_feeder_limit_ub = pyo.Constraint(
                rule=(
                    lambda m: sum(m.link_[t, r] for r in _R for t in _T) <= max_feeders
                ),
                name='feeder_limit_ub',
            )
        # enforce balanced subtrees (subtree loads differ at most by one unit)
        if balanced:
            if is_equal_not_range:
                feeder_min_load = T // min_feeders
                if feeder_min_load < capacity:
                    m.cons_balanced = pyo.Constraint(
                        m.T,
                        m.R,
                        rule=(
                            lambda m, t, r: m.flow_[t, r]
                            >= m.link_[t, r] * feeder_min_load
                        ),
                        name='balanced',
                    )
            else:
                warn(
                    'Model option <balanced = True> is incompatible with '
                    'having a range of possible feeder counts: model will '
                    'not enforce balanced subtrees.'
                )

    # radial or branched topology
    if topology is Topology.RADIAL:
        # just need to limit incoming edges since the outgoing are
        # limited by the m.cons_one_out_edge
        m.cons_radial = pyo.Constraint(
            m.T,
            rule=(
                lambda m, u: sum(m.link_[v, u] for v in A_terminals.neighbors(u)) <= 1
            ),
            name='radial',
        )

    # assert all nodes are connected to some root
    m.cons_total_power_sank = pyo.Constraint(
        rule=(lambda m: sum(m.flow_[t, r] for r in _R for t in _T) == W),
        name='total_power_sank',
    )

    # valid inequalities
    m.cons_inflow_limit = pyo.Constraint(
        m.T,
        rule=(
            lambda m, u: sum(m.flow_[v, u] for v in A_terminals.neighbors(u))
            <= m.k - A.nodes[u].get('power', 1)
        ),
        name='inflow_limit',
    )
    m.cons_single_out_link = pyo.Constraint(
        m.T,
        rule=(
            lambda m, u: sum(m.link_[u, v] for v in chain(A_terminals.neighbors(u), _R))
            == 1
        ),
        name='single_out_link',
    )

    #############
    # Objective #
    #############

    m.length = pyo.Objective(
        expr=lambda m: pyo.sum_product(m.weight_, m.link_),
        sense=pyo.minimize,
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
        capacity,
        m.linkset,
        m.link_,
        m.flow_,
        model_options,
        _make_min_length_model_fingerprint,
    )

    return m, metadata


_make_min_length_model_fingerprint = fun_fingerprint(make_min_length_model)


def warmup_model(
    model: pyo.ConcreteModel, metadata: ModelMetadata, S: nx.Graph
) -> pyo.ConcreteModel:
    """Set initial solution into `model`.

    Changes `model` and `metadata` in-place.

    Args:
      model: pyomo model to apply the solution to.
      metadata: indices to the model's variables.
      S: solution topology

    Returns:
      The same model instance that was provided, now with a solution.

    Raises:
      OWNWarmupFailed: if some link in S is not available in model.
    """
    for u, v, reverse in S.edges(data='reverse'):
        u, v = (u, v) if ((u < v) == reverse) else (v, u)
        try:
            model.link_[(u, v)] = 1
        except KeyError:
            raise OWNWarmupFailed(
                f'warmup_model() failed: model lacks S link ({u, v})'
            ) from None
        model.flow_[(u, v)] = S[u][v]['load']
    # check if solution violates any constraints:
    # checking the bounds seem redundant, but the way to do it would be:
    # next(find_infeasible_bounds(model), False)
    if next(find_infeasible_constraints(model), False):
        raise OWNWarmupFailed('warmup_model() failed: S violates some model constraint')
    metadata.warmed_by = S.graph['creator']
    return model
