# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
from types import SimpleNamespace
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


class SolverGurobi(SolverPyomo, PoolHandler):
    name: str = 'gurobi'
    # default options to pass to Pyomo solver
    options: dict = dict(
        mipfocus=1,
    )

    def __init__(self):
        # dummy attribute `solver` to be used by SolverPyomo.set_problem()
        self.solver = SimpleNamespace(warm_start_capable=lambda: True)

    def solve(
        self,
        time_limit: int,
        mip_gap: float,
        options: dict[str, Any] = {},
        verbose: bool = False,
    ) -> SolutionInfo:
        """
        This will keep the Gurobi license in use until a call to `get_solution()`.
        """
        model = self.model
        try:
            model = self.model
        except AttributeError as exc:
            exc.args += ('.set_problem() must be called before .solve()',)
            raise
        base_options = self.options | dict(timelimit=time_limit, mipgap=mip_gap)
        solver = pyo.SolverFactory(
            'gurobi',
            solver_io='python',
            manage_env=True,
            options=base_options | options,
        )
        self.solver = solver
        info('>>> %s solver options <<<\n%s\n', self.name, solver.options)
        result = solver.solve(model, **self.solve_kwargs, tee=verbose)
        self.result = result
        objective = result['Problem'][0]['Upper bound']
        bound = result['Problem'][0]['Lower bound']
        solution_info = SolutionInfo(
            runtime=result['Solver'][0]['Wallclock time'],
            bound=bound,
            objective=objective,
            relgap=1.0 - bound / objective,
            termination=result['Solver'][0]['Termination condition'].name,
        )
        self.solution_info, self.solver_options = solution_info, options
        info('>>> Solution <<<\n%s\n', solution_info)
        self.num_solutions = solver._solver_model.getAttr('SolCount')
        return solution_info

    def get_solution(self, A: nx.Graph | None = None) -> tuple[nx.Graph, nx.Graph]:
        if A is None:
            A = self.A
        P, model_options = self.P, self.model_options
        try:
            if self.model_options['feeder_route'] is FeederRoute.STRAIGHT:
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
        except Exception as exc:
            raise exc
        else:
            G.graph.update(self._make_graph_attributes())
            return S, G
        finally:
            self.solver.close()

    def objective_at(self, index: int) -> float:
        solver_model = self.solver._solver_model
        solver_model.setParam('SolutionNumber', index)
        return solver_model.getAttr('PoolObjVal')

    def topology_from_mip_pool(self) -> nx.Graph:
        solver = self.solver
        for omovar, gurvar in solver._pyomo_var_to_solver_var_map.items():
            omovar.set_value(round(gurvar.Xn), skip_validation=True)
        return topology_from_mip_sol(model=self.model)
