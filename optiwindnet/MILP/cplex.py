# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import networkx as nx
from typing import Any
import pyomo.environ as pyo
from .core import Solver, PoolHandler, summarize_result, investigate_pool
from .pyomo import make_min_length_model, warmup_model, S_from_solution

logger = logging.getLogger(__name__)
error, info = logger.error, logger.info


class SolverCplex(Solver, PoolHandler):
    name: str = 'cplex'
    sorted_index: int
    options: dict = {}

    def __init__(self) -> None:
        self.solver = pyo.SolverFactory('cplex', solver_io='python')

    def set_problem(self, P: nx.PlanarEmbedding, A: nx.Graph, capacity: int,
                    warmstart: nx.Graph | None = None, **kwargs):
        'kwargs contains problem options'
        self.P, self.A, self.capacity = P, A, capacity
        model = make_min_length_model(self.A, self.capacity, **kwargs)
        self.model = model
        if warmstart is not None:
            self.warmstart = True
            warmup_model(model, warmstart)
        else:
            self.warmstart = False

    def solve(self, timelimit: int, mipgap: float,
              options: dict[str, Any] = {}, verbose: bool = False) -> tuple:
        solver, model = self.solver, self.model
        base_options = self.options | dict(timelimit=timelimit, mipgap=mipgap)
        solver.options.update(base_options | options)
        result = solver.solve(model, warmstart=self.warmstart, tee=verbose)
        self.result = result
        cplex = solver._solver_model
        num_solutions = cplex.solution.pool.get_num()
        self.num_solutions, self.cplex = num_solutions, cplex
        self.sorted_index_ = sorted(
            range(num_solutions), key=cplex.solution.pool.get_objective_value
        )
        self.vars = solver._pyomo_var_to_ndx_map.keys()
        return summarize_result(result)

    def get_solution(self) -> nx.Graph:
        P, A, model = self.P, self.A, self.model
        if model.method_options['gateXings_constraint']:
            S = self.S_from_pool()
            G = PathFinder(G_from_S(S, A), P, A).create_detours()
        else:
            G = investigate_pool(P, A, self)
        return G

    def objective_at(self, index: int) -> float:
        soln = self.sorted_index_[index]
        objective = self.cplex.solution.pool.get_objective_value(soln)
        self.soln = soln
        return objective

    def S_from_pool(self) -> nx.Graph:
        solver, vars = self.solver, self.vars
        vals = solver._solver_model.cplex.solution.pool.get_values(self.soln)
        for pyomo_var, val in zip(vars, vals):
            if solver._referenced_variables[pyomo_var] > 0:
                pyomo_var.set_value(val, skip_validation=True)
        S = S_from_solution(self.model, solver, self.result)
        S.graph['method_options'] = dict(
            solver_name=self.solver_name,
            mipgap=self.mipgap,
            timelimit=self.timelimit,
        )
        S.graph['creator'] += self.solver_name
        return S
