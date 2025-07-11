# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
from typing import Any

import networkx as nx
import pyomo.environ as pyo

from ..interarraylib import G_from_S
from ..pathfinding import PathFinder
from ._core import FeederRoute, PoolHandler, SolutionInfo, Topology, investigate_pool
from .pyomo import SolverPyomo, topology_from_mip_sol

__all__ = ()

_lggr = logging.getLogger(__name__)
error, info = _lggr.error, _lggr.info


class SolverCplex(SolverPyomo, PoolHandler):
    name: str = 'cplex'
    # default options to pass to Pyomo solver
    options: dict = dict(
        # default solution pool size limit is 2100000000
        # mip_pool_replace=1,  # irrelevant with the default pool size
        parallel=-1,  # opportunistic parallelism (non-deterministic)
        emphasis_mip=4,  # focus on producing solutions
    )

    def __init__(self) -> None:
        self.solver = pyo.SolverFactory('cplex', solver_io='python')

    def solve(
        self,
        time_limit: int,
        mip_gap: float,
        options: dict[str, Any] = {},
        verbose: bool = False,
    ) -> SolutionInfo:
        solution_info = super().solve(time_limit, mip_gap, options, verbose)
        cplex = self.solver._solver_model
        num_solutions = cplex.solution.pool.get_num()
        self.num_solutions, self.cplex = num_solutions, cplex
        self.sorted_index_ = sorted(
            range(num_solutions), key=cplex.solution.pool.get_objective_value
        )
        self.vars = self.solver._pyomo_var_to_ndx_map.keys()
        return solution_info

    def get_solution(self, A: nx.Graph | None = None) -> tuple[nx.Graph, nx.Graph]:
        if A is None:
            A = self.A
        P, model_options = self.P, self.model_options
        if model_options['feeder_route'] is FeederRoute.STRAIGHT:
            S = self.topology_from_mip_pool()
            S.graph['creator'] += '.' + self.name
            G = PathFinder(
                G_from_S(S, A),
                P,
                A,
                branched=model_options['topology'] is Topology.BRANCHED,
            ).create_detours()
        else:
            S, G = investigate_pool(P, A, self)
        G.graph.update(self._make_graph_attributes())
        return S, G

    def objective_at(self, index: int) -> float:
        soln = self.sorted_index_[index]
        objective = self.cplex.solution.pool.get_objective_value(soln)
        self.soln = soln
        return objective

    def topology_from_mip_pool(self) -> nx.Graph:
        solver, vars = self.solver, self.vars
        vals = solver._solver_model.solution.pool.get_values(self.soln)
        for pyomo_var, val in zip(vars, vals):
            if solver._referenced_variables[pyomo_var] > 0:
                pyomo_var.set_value(val, skip_validation=True)
        return topology_from_mip_sol(model=self.model)
