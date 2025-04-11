# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import networkx as nx
from typing import Any
import pyomo.environ as pyo
from .core import Solver, PoolHandler, summarize_result, investigate_pool
from .pyomo import make_min_length_model, S_from_solution, warmup_model
from ..pathfinding import PathFinder
from ..interarraylib import G_from_S

logger = logging.getLogger(__name__)
error, info = logger.error, logger.info


class SolverGurobi(Solver, PoolHandler):
    name: str = 'gurobi'
    # default options to pass to Pyomo solver
    options: dict = dict(
        mipfocus=1,
    )

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
        '''
        This will keep the Gurobi license in use until a call to `get_solution()`.
        '''
        model = self.model
        base_options = self.options | dict(timelimit=timelimit, mipgap=mipgap)
        solver = pyo.SolverFactory('gurobi', solver_io='python', manage_env=True,
                                   options=base_options|options)
        self.solver = solver
        result = solver.solve(model, warmstart=self.warmstart, tee=verbose)
        self.num_solutions = solver._solver_model.getAttr('SolCount')
        self.result, self.timelimit, self.mipgap = result, timelimit, mipgap
        return summarize_result(result)

    def get_solution(self) -> nx.Graph:
        P, A, model = self.P, self.A, self.model
        try:
            if model.method_options['gateXings_constraint']:
                S = self.S_from_pool()
                G = PathFinder(G_from_S(S, A), P, A).create_detours()
            else:
                G = investigate_pool(P, A, self)
        except Exception as exc:
            raise exc
        else: 
            return G
        finally:
            self.solver.close()

    def objective_at(self, index: int) -> float:
        solver_model = self.solver._solver_model
        solver_model.setParam('SolutionNumber', index)
        return solver_model.getAttr('PoolObjVal')

    def S_from_pool(self) -> nx.Graph:
        solver = self.solver
        for omovar, gurvar in solver._pyomo_var_to_solver_var_map.items():
            omovar.set_value(round(gurvar.Xn), skip_validation=True)
        S = S_from_solution(self.model, solver, self.result)
        S.graph['method_options'].update(
            solver_name=self.name,
            mipgap=self.mipgap,
            timelimit=self.timelimit,
        )
        S.graph['creator'] += self.name
        return S
