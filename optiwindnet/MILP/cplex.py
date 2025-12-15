# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
from typing import Any

import networkx as nx
import pyomo.environ as pyo

from ..interarraylib import G_from_S
from ..pathfinding import PathFinder
from ._core import FeederRoute, PoolHandler, Topology
from .pyomo import SolverPyomo

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
        # emphasis_mip:
        #   0|BALANCED|(default) Balance optimality and feasibility; default
        #   1|FEASIBILITY|Emphasize feasibility over optimality
        #   2|OPTIMALITY|Emphasize optimality over feasibility
        #   3|BESTBOUND|Emphasize moving best bound
        #   4|HIDDENFEAS|Emphasize finding hidden feasible solutions
        #   5|HEURISTIC|Emphasize finding high quality feasible solutions earlier
        emphasis_mip=4,
    )

    def __init__(self) -> None:
        self.solver = pyo.SolverFactory('cplex', solver_io='python')

    def _link_val(self, var: Any) -> int:
        return round(self._value_map[var.name])

    def _flow_val(self, var: Any) -> int:
        return round(self._value_map[var.name])

    def get_solution(self, A: nx.Graph | None = None) -> tuple[nx.Graph, nx.Graph]:
        cplex = self.solver._solver_model
        num_solutions = cplex.solution.pool.get_num()
        self.num_solutions, self.cplex = num_solutions, cplex
        # make the ranked soln list (position 0 holds the lowest objective)
        self.sorted_index_ = sorted(
            range(num_solutions), key=cplex.solution.pool.get_objective_value
        )
        # set the selected (last visited) soln to the best one
        self.soln = self.sorted_index_[0]
        self.vars = self.solver._pyomo_var_to_ndx_map.keys()
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
        soln = self.sorted_index_[index]
        objective = self.cplex.solution.pool.get_objective_value(soln)
        self.soln = soln
        return objective

    def _topology_from_mip_pool(self) -> nx.Graph:
        self._value_map = {
            var.name: val
            for var, val in zip(
                self.vars, self.solver._solver_model.solution.pool.get_values(self.soln)
            )
        }
        return self._topology_from_mip_sol()
