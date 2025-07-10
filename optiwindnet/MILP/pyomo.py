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

from ..crossings import edgeset_edgeXing_iter, gateXing_iter
from ..interarraylib import G_from_S, fun_fingerprint
from ..pathfinding import PathFinder
from ._core import (
    FeederLimit,
    FeederRoute,
    ModelMetadata,
    ModelOptions,
    SolutionInfo,
    Solver,
    Topology,
)

__all__ = ('make_min_length_model', 'warmup_model', 'topology_from_mip_sol')

_lggr = logging.getLogger(__name__)
error, warn, info = _lggr.error, _lggr.warning, _lggr.info

# NOTE: SCIP has solution pool which can be accessed with PySCIPOpt: getSols()
# TODO: move scip to scip.py and implement:
#       - class SolverSCIP(SolverPyomo, PoolHandler)
#       - SCIP-specific make_min_length_model, warmup_model, topology_from_mip_sol
#       - warmup_model: model.createSol(), model.setSolVal(), model.addSol()

# solver option name mapping (pyomo should have taken care of this)
_common_options = namedtuple('common_options', 'mip_gap time_limit')
_optkey = dict(
    cplex=_common_options('mipgap', 'timelimit'),
    gurobi=_common_options('mipgap', 'timelimit'),
    cbc=_common_options('ratioGap', 'seconds'),
    highs=_common_options('mip_rel_gap', 'time_limit'),
    scip=_common_options('limits/gap', 'limits/time'),
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
        self.model, self.model_options = model, model_options
        if warmstart is not None and self.solver.warm_start_capable():
            self.solve_kwargs = {'warmstart': True}
            warmup_model(model, warmstart)
        else:
            self.solve_kwargs = {}
            if warmstart is not None:
                warn(f'Solver <{self.name}> is not capable of warm-starting.')
        self.model, self.metadata = model, metadata

    def solve(
        self,
        time_limit: int,
        mip_gap: float,
        options: dict[str, Any] = {},
        verbose: bool = False,
    ) -> SolutionInfo:
        try:
            solver, name, model = self.solver, self.name, self.model
        except AttributeError as exc:
            exc.args += ('.set_problem() must be called before .solve()',)
            raise
        solver.options.update(
            self.options
            | options
            | {_optkey[name].time_limit: time_limit, _optkey[name].mip_gap: mip_gap}
        )
        info('>>> %s solver options <<<\n%s\n', self.name, solver.options)
        result = solver.solve(model, **self.solve_kwargs, tee=verbose)
        self.result = result
        if self.name != 'scip':
            objective = result['Problem'][0]['Upper bound']
            bound = result['Problem'][0]['Lower bound']
        else:
            objective = result['Solver'][0]['Primal bound']
            bound = result['Solver'][0]['Dual bound']
        solution_info = SolutionInfo(
            runtime=result['Solver'][0]['Wallclock time'],
            bound=bound,
            objective=objective,
            relgap=1.0 - bound / objective,
            termination=result['Solver'][0]['Termination condition'].name,
        )
        self.solution_info, self.solver_options = solution_info, options
        info('>>> Solution <<<\n%s\n', solution_info)
        return solution_info

    def get_solution(self, A: nx.Graph | None = None) -> tuple[nx.Graph, nx.Graph]:
        if A is None:
            A = self.A
        P, model_options = self.P, self.model_options
        S = topology_from_mip_sol(model=self.model)
        S.graph['creator'] += self.name
        G = PathFinder(
            G_from_S(S, A),
            P,
            A,
            branched=model_options['topology'] is Topology.BRANCHED,
        ).create_detours()
        G.graph.update(self._make_graph_attributes())
        return S, G

    def topology_from_mip_sol(self):
        return topology_from_mip_sol(model=self.model)


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
    m.cons_edges_eq_nodes = pyo.Constraint(rule=lambda m: sum(m.link_.values()) == T)

    # enforce a single directed edge between each node pair
    m.cons_one_diEdge = pyo.Constraint(
        E, rule=lambda m, u, v: m.link_[u, v] + m.link_[v, u] <= 1
    )

    # feeder-edge crossings
    if feeder_route is FeederRoute.STRAIGHT:
        m.cons_feederXedge = pyo.Constraint(
            gateXing_iter(A),
            rule=lambda m, u, v, r, n: (
                m.link_[u, v] + m.link_[v, u] + m.link_[r, n] <= 1
            ),
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
        m.cons_edgeXedge = pyo.Constraint(doubleXings, rule=edgeXedge_rule)
    if tripleXings:
        m.cons_edgeXedgeXedge = pyo.Constraint(tripleXings, rule=edgeXedge_rule)

    # bind flow to link activation
    m.cons_link_used_iff_demand_lb = pyo.Constraint(
        m.linkset,
        rule=(
            lambda m, u, v: m.flow_[(u, v)]
            <= m.link_[(u, v)] * (m.k if v < 0 else (m.k - 1))
        ),
    )
    m.cons_link_used_iff_demand_ub = pyo.Constraint(
        m.linkset, rule=lambda m, u, v: m.link_[(u, v)] <= m.flow_[(u, v)]
    )

    # flow conservation with possibly non-unitary node power
    m.cons_flow_conservation = pyo.Constraint(
        m.T,
        rule=(
            lambda m, u: (
                sum((m.flow_[u, v] - m.flow_[v, u]) for v in A_nodes.neighbors(u))
                + sum(m.flow_[u, r] for r in _R)
                == A.nodes[u].get('power', 1)
            )
        ),
    )

    # feeder limits
    min_feeders = math.ceil(T / m.k)
    if feeder_limit is not FeederLimit.UNLIMITED:

        def feeder_limit_eq_rule(m):
            return sum(m.link_[t, r] for r in _R for t in _T) == min_feeders

        def feeder_limit_ub_rule(m):
            return sum(m.link_[t, r] for r in _R for t in _T) <= max_feeders

        feeder_rule = feeder_limit_ub_rule
        if feeder_limit is FeederLimit.SPECIFIED:
            if max_feeders == min_feeders:
                feeder_rule = feeder_limit_eq_rule
            elif max_feeders < min_feeders:
                raise ValueError('max_feeders is below the minimum necessary')
        elif feeder_limit is FeederLimit.MINIMUM:
            feeder_rule = feeder_limit_eq_rule
        elif feeder_limit is FeederLimit.MIN_PLUS1:
            max_feeders = min_feeders + 1
        elif feeder_limit is FeederLimit.MIN_PLUS2:
            max_feeders = min_feeders + 2
        elif feeder_limit is FeederLimit.MIN_PLUS3:
            max_feeders = min_feeders + 3
        else:
            raise NotImplementedError('Unknown value:', feeder_limit)
        m.cons_feeder_limit = pyo.Constraint(rule=feeder_rule)

        # enforce balanced subtrees (subtree loads differ at most by one unit)
        if balanced:
            if feeder_rule is not feeder_limit_eq_rule:
                warn(
                    'Model option <balanced = True> is incompatible with '
                    'having a range of possible feeder counts: model will '
                    'not enforce balanced subtrees.'
                )
            else:
                feeder_min_load = T // min_feeders
                if feeder_min_load < capacity:
                    m.cons_balanced = pyo.Constraint(
                        m.T,
                        m.R,
                        rule=(
                            lambda m, t, r: m.flow_[t, r]
                            >= m.link_[t, r] * feeder_min_load
                        ),
                    )
    elif balanced:
        warn(
            'Model option <balanced = True> is incompatible with <feeder_limit'
            ' = UNLIMITED>: model will not enforce balanced subtrees.'
        )

        # auxiliary variables
    # radial or branched topology
    if topology is Topology.RADIAL:
        # just need to limit incoming edges since the outgoing are
        # limited by the m.cons_one_out_edge
        m.cons_radial = pyo.Constraint(
            m.T,
            rule=lambda m, u: (sum(m.link_[v, u] for v in A_nodes.neighbors(u)) <= 1),
        )

    # assert all nodes are connected to some root
    m.cons_all_nodes_connected = pyo.Constraint(
        rule=lambda m: sum(m.flow_[t, r] for r in _R for t in _T) == W
    )

    # valid inequalities
    m.cons_min_gates_required = pyo.Constraint(
        rule=lambda m: (
            sum(m.link_[t, r] for r in _R for t in _T) >= math.ceil(T / m.k)
        )
    )
    m.cons_incoming_demand_limit = pyo.Constraint(
        m.T,
        rule=lambda m, u: (
            sum(m.flow_[v, u] for v in A_nodes.neighbors(u))
            <= m.k - A.nodes[u].get('power', 1)
        ),
    )
    m.cons_one_out_edge = pyo.Constraint(
        m.T,
        rule=lambda m, u: (
            sum(m.link_[u, v] for v in chain(A_nodes.neighbors(u), _R)) == 1
        ),
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
    )
    metadata = ModelMetadata(
        R, T, capacity, m.linkset, m.link_, m.flow_, model_options, fun_fingerprint()
    )

    return m, metadata


def warmup_model(model: pyo.ConcreteModel, S: nx.Graph) -> pyo.ConcreteModel:
    """Set initial solution into `model`.

    Changes `model` in-place.

    Args:
      model: pyomo model to apply the solution to.
      S: solution topology

    Returns:
      The same model instance that was provided, now with a solution.
    """
    for u, v, reverse in S.edges(data='reverse'):
        u, v = (u, v) if ((u < v) == reverse) else (v, u)
        model.link_[(u, v)] = 1
        model.flow_[(u, v)] = S[u][v]['load']
    model.warmed_by = S.graph['creator']
    return model


def topology_from_mip_sol(*, model: pyo.ConcreteModel, **kwargs) -> nx.Graph:
    """Create a topology graph from the solution to the Pyomo MILP model.

    Args:
      model: pyomo model instance
      kwargs: not used (signature compatibility)
    Returns:
      Graph topology `S` from the solution.
    """
    # in pyomo, the solution is in the model instance not in the solver
    S = nx.Graph(R=len(model.R), T=len(model.T))
    # Get active links and if flow is reversed (i.e. from small to big)
    rev_from_link = {
        (u, v): u < v for (u, v), use in model.link_.items() if use.value > 0.5
    }
    S.add_weighted_edges_from(
        ((u, v, round(model.flow_[u, v].value)) for (u, v) in rev_from_link.keys()),
        weight='load',
    )
    # set the 'reverse' edge attribute
    nx.set_edge_attributes(S, rev_from_link, name='reverse')
    # propagate loads from edges to nodes
    subtree = -1
    max_load = 0
    for r in range(model.R.first(), 0):
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
        capacity=model.k.value,
        max_load=max_load,
        has_loads=True,
        creator='MILP.' + __name__,
        solver_details={},
    )
    return S
