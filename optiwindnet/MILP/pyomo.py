# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
from collections import namedtuple
from collections.abc import Mapping
from itertools import chain
from typing import Any

import networkx as nx
import pyomo.environ as pyo
from pyomo.util.infeasible import (
    find_infeasible_constraints,
    log_infeasible_constraints,
)

from ..crossings import edgeset_edgeXing_iter, gateXing_iter
from ..fingerprint import fingerprint_function
from ..interarraylib import G_from_S
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
    feeder_and_load_bounds,
    physical_core_count,
    warmstart_links,
)

__all__ = ('make_min_length_model', 'warmup_model')

_lggr = logging.getLogger(__name__)
error, warn, info = _lggr.error, _lggr.warning, _lggr.info

# solver option name mapping (pyomo should have taken care of this)
_common_options = namedtuple('common_options', 'mip_gap time_limit')
_optkey = {
    'pyomo.cplex': _common_options('mipgap', 'timelimit'),
    'pyomo.gurobi': _common_options('mipgap', 'timelimit'),
    'pyomo.cbc': _common_options('ratioGap', 'seconds'),
    'pyomo.highs': _common_options('mip_gap', 'time_limit'),
}
# usage: _optname[solver_name].mipgap

_default_options = dict(
    cbc=dict(
        threads=physical_core_count(),
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
    highs=dict(
        parallel='on',
        # threads=0,  # 0 means automatic and is HiGHS's default
    ),
    scip={},
)


class SolverPyomo(Solver):
    def __init__(self, name, prefix='', suffix='', **kwargs) -> None:
        self.name = 'pyomo.' + name
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
        model_options: Mapping[str, Any],
        warmstart: nx.Graph | None = None,
    ):
        self.P, self.A, self.capacity = P, A, capacity
        model_options = ModelOptions(**model_options)
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
                f'Unable to find a solution. Solver {self.name} terminated'
                f' with: {termination}'
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
        P, model = self.P, self.model
        result = self.result
        # hack to prevent warning about the solver not reaching the desired mip_gap
        result.solver.status = pyo.SolverStatus.ok
        model.solutions.load_from(result)
        if A is None:
            A = self.A
        S = self._topology_from_mip_sol()
        S.graph['fun_fingerprint'] = _make_min_length_model_fingerprint
        G = PathFinder(G_from_S(S, A), P, A).create_detours()
        G.graph.update(self._make_graph_attributes())
        return S, G


class SolverPyomoAppsi(Solver):
    """As of Pyomo v3.9.4, a new solver inverface (v3) is being introduced. HiGHS is the
    only solver using v3 at that point."""

    def __init__(self, name, solver_cls, **kwargs) -> None:
        self.name = 'pyomo.' + name
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
        model_options: Mapping[str, Any],
        warmstart: nx.Graph | None = None,
    ):
        self.P, self.A, self.capacity = P, A, capacity
        model_options = ModelOptions(**model_options)
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
        for key, value in applied_options.items():
            if key in solver.config:
                solver.config[key] = value
            else:
                solver._solver_options[key] = value
        stopping = {
            _optkey[name].time_limit: time_limit,
            _optkey[name].mip_gap: mip_gap,
        }
        self.stopping = stopping
        for k, v in stopping.items():
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
                f'Unable to find a solution. Solver {self.name} terminated'
                f' with: {termination}'
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
        P = self.P
        result = self.result
        result.solution_loader.load_vars()
        #  model.solutions.load_from(result)
        if A is None:
            A = self.A
        S = self._topology_from_mip_sol()
        S.graph['fun_fingerprint'] = _make_min_length_model_fingerprint
        G = PathFinder(G_from_S(S, A), P, A).create_detours()
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
      topology: one of ``Topology.{BRANCHED, RADIAL}``
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

    # For RINGED, double the internal capacity so each ring can hold up to
    # 2×capacity turbines (capacity per arm); store original for metadata.
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
        # append the links leaving the roots
        starsʹ = tuple((r, t) for t, r in stars)
    else:
        starsʹ = ()

    # Create model
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T - 1)
    m.R = pyo.RangeSet(-R, -1)
    m.linkset = pyo.Set(initialize=E + Eʹ + stars + starsʹ)

    ##############
    # Parameters #
    ##############

    m.k = pyo.Param(domain=pyo.PositiveIntegers, name='capacity', default=capacity)
    m.weight_ = pyo.Param(
        m.linkset,
        domain=pyo.PositiveReals,
        name='link_weight',
        initialize=lambda m, u, v: (
            A.edges[(u, v)]['length']
            if v >= 0 and u >= 0
            else d2roots[(u, v) if u >= 0 else (v, u)]
        ),
    )

    #############
    # Variables #
    #############

    m.link_ = pyo.Var(m.linkset, domain=pyo.Binary, initialize=0)

    def flow_bounds(m, u, v):
        return (0, (m.k if v < 0 else m.k - 1))

    m.flow_ = pyo.Var(
        m.linkset if topology != Topology.RINGED else E + Eʹ + stars,
        domain=pyo.NonNegativeIntegers,
        bounds=flow_bounds,
        initialize=0,
    )

    ###############
    # Constraints #
    ###############

    # total number of edges must be equal to number of non-root nodes
    if topology != Topology.RINGED:
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
        if topology == Topology.RINGED:

            def feederXedge_rule(m, u, v, r, t):
                if u >= 0:
                    return (
                        m.link_[u, v] + m.link_[v, u] + m.link_[t, r] + m.link_[r, t]
                        <= 1
                    )
                else:
                    # feeder-feeder crossing (possible in multi-root instances)
                    return (
                        m.link_[u, v] + m.link_[v, u] + m.link_[t, r] + m.link_[r, t]
                        <= 1
                    )
        else:

            def feederXedge_rule(m, u, v, r, t):
                if u >= 0:
                    return m.link_[u, v] + m.link_[v, u] + m.link_[t, r] <= 1
                else:
                    # feeder-feeder crossing (possible in multi-root instances)
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
        m.linkset if topology != Topology.RINGED else E + Eʹ + stars,
        rule=(
            lambda m, u, v: (
                m.flow_[(u, v)] <= m.link_[(u, v)] * (m.k if v < 0 else (m.k - 1))
            )
        ),
        name='flow_ub',
    )
    m.cons_flow_lb = pyo.Constraint(
        m.linkset if topology != Topology.RINGED else E + Eʹ + stars,
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

    # feeder limits. A RINGED subtree is a cycle with two feeders, so the
    # user-facing feeder count is in substation connections (two per ring), while
    # the model counts rings (one flow-feeder var each): convert between them.
    feeders_per_subtree = 2 if topology is Topology.RINGED else 1
    feeders_lb, feeders_ub, load_lb, load_ub = feeder_and_load_bounds(
        T,
        capacity,
        feeder_limit,
        max_feeders,
        balanced,
        feeders_per_subtree,
        total_power=W,
    )
    if feeders_ub is not None and feeder_limit.name.startswith('MIN_PLUS'):
        # derived from the minimum: surface it in the solution's metadata
        max_feeders = feeders_per_subtree * feeders_ub
    if feeders_lb == feeders_ub:
        m.cons_feeder_limit_eq = pyo.Constraint(
            rule=(lambda m: sum(m.link_[t, r] for r in _R for t in _T) == feeders_lb),
            name='feeder_limit_eq',
        )
    else:
        # valid inequality: number of feeders is at least the minimum
        m.cons_feeder_limit_lb = pyo.Constraint(
            rule=(lambda m: sum(m.link_[t, r] for r in _R for t in _T) >= feeders_lb),
            name='feeder_limit_lb',
        )
        if feeders_ub is not None:
            m.cons_feeder_limit_ub = pyo.Constraint(
                rule=(
                    lambda m: sum(m.link_[t, r] for r in _R for t in _T) <= feeders_ub
                ),
                name='feeder_limit_ub',
            )

    # enforce balanced subtrees (subtree loads differ at most by one unit)
    if load_lb is not None:
        m.cons_balanced_lb = pyo.Constraint(
            m.T,
            m.R,
            rule=(lambda m, t, r: m.flow_[t, r] >= m.link_[t, r] * load_lb),
            name='balanced_lb',
        )
    if load_ub is not None:
        m.cons_balanced_ub = pyo.Constraint(
            m.T,
            m.R,
            rule=(lambda m, t, r: m.flow_[t, r] <= m.link_[t, r] * load_ub),
            name='balanced_ub',
        )

    # only for radial or ringed topology
    if topology is Topology.RADIAL:
        # just need to limit incoming edges since the outgoing are
        # limited by the m.cons_single_out_link
        m.cons_radial = pyo.Constraint(
            m.T,
            rule=(
                lambda m, u: sum(m.link_[v, u] for v in A_terminals.neighbors(u)) <= 1
            ),
            name='radial',
        )
    elif topology is Topology.RINGED:
        # just need to limit incoming edges since the outgoing are
        # limited by the m.cons_one_out_edge
        m.cons_ringed = pyo.Constraint(
            m.T,
            rule=(
                lambda m, t: (
                    sum(m.link_[v, t] for v in A_terminals.neighbors(t))
                    + sum(m.link_[r, t] for r in m.R)
                    == 1
                )
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
            lambda m, u: (
                sum(m.flow_[v, u] for v in A_terminals.neighbors(u))
                <= m.k - A.nodes[u].get('power', 1)
            )
        ),
        name='inflow_limit',
    )
    m.cons_single_out_link = pyo.Constraint(
        m.T,
        rule=(
            lambda m, u: (
                sum(m.link_[u, v] for v in chain(A_terminals.neighbors(u), _R)) == 1
            )
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
        ring_capacity,
        m.linkset,
        m.link_,
        m.flow_,
        model_options,
        _make_min_length_model_fingerprint,
    )

    return m, metadata


_make_min_length_model_fingerprint = fingerprint_function(make_min_length_model)


def warmup_model(
    model: pyo.ConcreteModel, metadata: ModelMetadata, S: nx.Graph
) -> pyo.ConcreteModel:
    """Set initial solution into ``model``.

    Changes ``model`` and ``metadata`` in-place.

    Args:
      model: pyomo model to apply the solution to.
      metadata: indices to the model's variables.
      S: solution topology

    Returns:
      The same model instance that was provided, now with a solution.

    Raises:
      OWNWarmupFailed: if some link in S is not available in model.
    """
    mt = metadata.model_options['topology']
    st = S.graph['topology']
    if not (st is mt or (mt is Topology.BRANCHED and st is Topology.RADIAL)):
        raise OWNWarmupFailed(
            f'warmup_model() failed: {st} network cannot seed a {mt} model'
        )
    # Pyomo Vars are initialize=0, so the inactive baseline is already in place;
    # only the active links need to be set.
    for link_var, flow_var, flow in warmstart_links(metadata, S):
        link_var.value = 1
        if flow_var is not None:
            flow_var.value = flow

    # check if solution violates any constraints:
    # checking the bounds seem redundant, but the way to do it would be:
    # next(find_infeasible_bounds(model), False)
    log_infeasible_constraints(model, log_variables=True)
    if next(find_infeasible_constraints(model), False):
        raise OWNWarmupFailed('warmup_model() failed: S violates some model constraint')
    metadata.warmed_by = S.graph['creator']
    return model
