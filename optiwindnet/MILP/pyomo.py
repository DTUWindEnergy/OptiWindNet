# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import math
from collections import namedtuple, defaultdict
from itertools import chain
import networkx as nx

import pyomo.environ as pyo
from pyomo.contrib.solver.base import SolverBase
from pyomo.opt import SolverResults

from ..crossings import edgeset_edgeXing_iter, gateXing_iter
from ..interarraylib import fun_fingerprint, G_from_S
from ..pathfinding import PathFinder


# solver option name mapping (pyomo should have taken care of this)
_common_options = namedtuple('common_options', 'mipgap timelimit')
_optname = defaultdict(lambda: _common_options(*_common_options._fields))
_optname['cbc'] = _common_options('ratioGap', 'seconds')
_optname['highs'] = _common_options('mip_rel_gap', 'time_limit')
_optname['scip'] = _common_options('limits/gap', 'limits/time')


def make_min_length_model(A: nx.Graph, capacity: int, *,
                          gateXings_constraint: bool = False,
                          gates_limit: bool | int = False,
                          branching: bool = True) -> pyo.ConcreteModel:
    '''
    Build ILP Pyomo model for the collector system length minimization.
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

    # Create model
    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T - 1)
    m.R = pyo.RangeSet(-R, -1)
    m.linkset = pyo.Set(initialize=E + Eʹ + stars)

    ##############
    # Parameters #
    ##############

    m.k = pyo.Param(domain=pyo.PositiveIntegers,
                    name='capacity', default=capacity)
    m.weight = pyo.Param(
        m.linkset,
        domain=pyo.PositiveReals,
        name='link_weight',
        initialize=lambda m, u, v: (A.edges[(u, v)]['length'] if v >= 0
                                    else d2roots[u, v])
    )

    #############
    # Variables #
    #############

    m.link_ = pyo.Var(m.linkset, domain=pyo.Binary, initialize=0)
    def flow_bounds(m, u, v):
        return (0, (m.k if v < 0 else m.k - 1))
    m.flow_ = pyo.Var(m.linkset, domain=pyo.NonNegativeIntegers,
                     bounds=flow_bounds, initialize=0)

    ###############
    # Constraints #
    ###############

    # total number of edges must be equal to number of non-root nodes
    m.cons_edges_eq_nodes = pyo.Constraint(
        rule=lambda m: sum(m.link_.values()) == T
    )

    # enforce a single directed edge between each node pair
    m.cons_one_diEdge = pyo.Constraint(
        E,
        rule=lambda m, u, v: m.link_[u, v] + m.link_[v, u] <= 1
    )

    # gate-edge crossings
    if gateXings_constraint:
        m.cons_gateXedge = pyo.Constraint(
            gateXing_iter(A),
            rule=lambda m, u, v, r, n: (m.link_[u, v]
                                        + m.link_[v, u]
                                        + m.link_[r, n] <= 1)
        )

    # edge-edge crossings
    def edgeXedge_rule(m, *vertices):
        lhs = sum((m.link_[u, v] + m.link_[v, u])
                  for u, v in zip(vertices[::2],
                                  vertices[1::2]))
        return lhs <= 1
    doubleXings = []
    tripleXings = []
    for Xing in edgeset_edgeXing_iter(A.graph['diagonals']):
        if len(Xing) == 2:
            doubleXings.append(Xing)
        else:
            tripleXings.append(Xing)
    if doubleXings:
        m.cons_edgeXedge = pyo.Constraint(doubleXings,
                                          rule=edgeXedge_rule)
    if tripleXings:
        m.cons_edgeXedgeXedge = pyo.Constraint(tripleXings,
                                               rule=edgeXedge_rule)

    # bind flow to link activation
    m.cons_link_used_iff_demand_lb = pyo.Constraint(
        m.linkset,
        rule=(lambda m, u, v:
              m.flow_[(u, v)] <= m.link_[(u, v)]*(m.k if v < 0 else (m.k - 1)))
    )
    m.cons_link_used_iff_demand_ub = pyo.Constraint(
        m.linkset,
        rule=lambda m, u, v: m.link_[(u, v)] <= m.flow_[(u, v)]
    )

    # flow conservation with possibly non-unitary node power
    m.cons_flow_conservation = pyo.Constraint(
        m.T,
        rule=(lambda m, u:
              (sum((m.flow_[u, v] - m.flow_[v, u])
                   for v in A_nodes.neighbors(u))
               + sum(m.flow_[u, r] for r in _R)
              == A.nodes[u].get('power', 1))
        )
    )

    # gates limit
    if gates_limit:
        def gates_limit_eq_rule(m):
            return (sum(m.link_[t, r] for r in _R for t in _T)
                    == math.ceil(T/m.k))

        def gates_limit_ub_rule(m):
            return (sum(m.link_[t, r] for r in _R for t in _T)
                    <= gates_limit)

        m.gates_limit = pyo.Constraint(rule=(gates_limit_eq_rule
                                             if isinstance(gates_limit, bool)
                                             else gates_limit_ub_rule))

    # non-branching
    if not branching:
        # just need to limit incoming edges since the outgoing are
        # limited by the m.cons_one_out_edge
        m.non_branching = pyo.Constraint(
            m.T,
            rule=lambda m, u: (sum(m.link_[v, u] for v in A_nodes.neighbors(u))
                               <= 1)
        )

    # assert all nodes are connected to some root
    m.cons_all_nodes_connected = pyo.Constraint(
        rule=lambda m: sum(m.flow_[t, r] for r in _R for t in _T) == W
    )

    # valid inequalities
    m.cons_min_gates_required = pyo.Constraint(
        rule=lambda m: (sum(m.link_[t, r] for r in _R for t in _T)
                        >= math.ceil(T/m.k))
    )
    m.cons_incoming_demand_limit = pyo.Constraint(
        m.T,
        rule=lambda m, u: (sum(m.flow_[v, u] for v in A_nodes.neighbors(u))
                           <= m.k - A.nodes[u].get('power', 1))
    )
    m.cons_one_out_edge = pyo.Constraint(
        m.T,
        rule=lambda m, u:(
            sum(m.link_[u, v] for v in chain(A_nodes.neighbors(u), _R)) == 1)
    )


    #############
    # Objective #
    #############

    m.length = pyo.Objective(
        expr=lambda m: pyo.sum_product(m.weight, m.link_),
        sense=pyo.minimize,
    )

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


def warmup_model(model: pyo.ConcreteModel, S: nx.Graph) \
        -> pyo.ConcreteModel:
    '''
    Changes `model` in-place.
    '''
    for u, v, reverse in S.edges(data='reverse'):
        u, v = (u, v) if ((u < v) == reverse) else (v, u)
        model.link_[(u, v)] = 1
        model.flow_[(u, v)] = S[u][v]['load']
    model.warmed_by = S.graph['creator']
    return model


def S_from_solution(model: pyo.ConcreteModel, solver: SolverBase,
                    result: SolverResults) -> nx.Graph:
    '''
    Create a topology `S` with the solution in `model` by `solver`.
    '''

    # Metadata
    R, T, k = len(model.R), len(model.T), model.k.value
    #  if 'highs.Highs' in str(solver):
    if hasattr(solver, 'highs_options'):
        solver_name = 'highs'
    elif solver.name.endswith('direct'):
        solver_name = solver.name[:-6].rstrip('_')
    elif solver.name.endswith('persistent'):
        solver_name = solver.name[:-10].rstrip('_')
    else:
        solver_name = solver.name
    bound = result['Problem'][0]['Lower bound']
    objective = result['Problem'][0]['Upper bound']
    # create a topology graph S from the solution
    S = nx.Graph(
        R=R, T=T,
        handle=model.handle,
        capacity=k,
        objective=objective,
        bound=bound,
        runtime=result['Solver'][0]['Wallclock time'],
        termination=result['Solver'][0]['Termination condition'].name,
        gap=1. - bound/objective,
        creator='MILP.pyomo.' + solver_name,
        has_loads=True,
        method_options=dict(
            solver_name=solver_name,
            mipgap=solver.options[_optname[solver_name].mipgap],
            timelimit=solver.options[_optname[solver_name].timelimit],
            fun_fingerprint=model.fun_fingerprint,
            **model.method_options,
        ),
        #  solver_details=dict(
        #  )
    )

    if model.warmed_by is not None:
        S.graph['warmstart'] = model.warmed_by
    
    # Graph data
    # Get active links and if flow is reversed (i.e. from small to big)
    rev_from_link = {
        (u, v): u < v for (u, v), use in model.link_.items() if use.value > 0.5
    }
    S.add_weighted_edges_from(
        ((u, v, round(model.flow_[u, v].value))
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


def gurobi_investigate_pool(P, A, model, solver, result):
    '''Go through the Gurobi's solutions checking which has the shortest length
    after applying the detours with PathFinder.'''
    # initialize incumbent total length
    solver_model = solver._solver_model
    Λ = float('inf')
    num_solutions = solver_model.getAttr('SolCount')
    print(f'Solution pool has {num_solutions} solutions.')
    # Pool = iter(sorted((cplex.solution.pool.get_objective_value(i), i)
                       # for i in range(cplex.solution.pool.get_num()))[1:])
    # model comes loaded with minimal-length undetoured solution
    for i in range(num_solutions):
        solver_model.setParam('SolutionNumber', i)
        λ = solver_model.getAttr('PoolObjVal')
        if λ > Λ:
            print(f'Pool investigation over - next best undetoured length: {λ:.3f}')
            break
        for omovar, gurvar in solver._pyomo_var_to_solver_var_map.items():
            omovar.set_value(round(gurvar.Xn), skip_validation=True)
        S = S_from_solution(model, solver=solver, result=result)
        G = G_from_S(S, A)
        Hʹ = PathFinder(G, planar=P, A=A).create_detours()
        Λʹ = Hʹ.size(weight='length')
        if Λʹ < Λ:
            H, Λ = Hʹ, Λʹ
            pool_index = i
            pool_objective = λ
            print(f'Incumbent has (detoured) length: {Λ:.3f}')
    H.graph['solver_details']['pool_index'] = pool_index
    if pool_index > 0:
        H.graph['solver_details']['pool_objective'] = pool_objective
    return H


def cplex_load_solution_from_pool(solver, soln):
    cplex = solver._solver_model
    vals = cplex.solution.pool.get_values(soln)
    vars_to_load = solver._pyomo_var_to_ndx_map.keys()
    for pyomo_var, val in zip(vars_to_load, vals):
        if solver._referenced_variables[pyomo_var] > 0:
            pyomo_var.set_value(val, skip_validation=True)


def cplex_investigate_pool(P, A, model, solver, result):
    '''Go through the CPLEX solutions checking which has the shortest length
    after applying the detours with PathFinder.'''
    cplex = solver._solver_model
    # initialize incumbent total length
    Λ = float('inf')
    print(f'Solution pool has {cplex.solution.pool.get_num()} solutions.')
    Pool = iter(sorted((cplex.solution.pool.get_objective_value(i), i)
                       for i in range(cplex.solution.pool.get_num()))[1:])
    # model comes loaded with minimal-length undetoured solution
    while True:
        S = S_from_solution(model, solver=solver, result=result)
        G = G_from_S(S, A)
        Hʹ = PathFinder(G, planar=P, A=A).create_detours()
        Λʹ = Hʹ.size(weight='length')
        if Λʹ < Λ:
            H, Λ = Hʹ, Λʹ
            print(f'Incumbent has (detoured) length: {Λ:.3f}')
        # check if next best solution is worth processing
        try:
            λ, soln = next(Pool)
        except StopIteration:
            print('Pool exhausted.')
            break
        if λ > Λ:
            print(f'Done with pool - next best undetoured length: {λ:.3f}')
            break
        cplex_load_solution_from_pool(solver, soln)
    return H
