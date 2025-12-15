# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
from types import SimpleNamespace
from typing import Any

import networkx as nx
import pyomo.environ as pyo

from ..interarraylib import G_from_S
from ..pathfinding import PathFinder
from ._core import (
    FeederRoute,
    OWNSolutionNotFound,
    PoolHandler,
    SolutionInfo,
    Topology,
)
from .pyomo import SolverPyomo

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

    def _link_val(self, var: Any) -> int:
        return self._value_map[var.name]

    def _flow_val(self, var: Any) -> int:
        return self._value_map[var.name]

    def solve(
        self,
        time_limit: float,
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
        applied_options = self.options | options
        self.stopping = dict(mip_gap=mip_gap, time_limit=time_limit)
        solver = pyo.SolverFactory(
            'gurobi',
            solver_io='python',
            manage_env=True,
            options=applied_options | dict(timelimit=time_limit, mipgap=mip_gap),
        )
        self.solver = solver
        info('>>> %s solver options <<<\n%s\n', self.name, solver.options)
        result = solver.solve(
            model, **self.solve_kwargs, tee=verbose, load_solutions=False
        )
        num_solutions = solver._solver_model.getAttr('SolCount')
        termination = result['Solver'][0]['Termination condition'].name
        if num_solutions == 0:
            raise OWNSolutionNotFound(
                f'Unable to find a solution. Solver gurobi terminated with: {termination}'
            )
        self.result = result
        objective = result['Problem'][0]['Upper bound']
        bound = result['Problem'][0]['Lower bound']
        solution_info = SolutionInfo(
            runtime=result['Solver'][0]['Wallclock time'],
            bound=bound,
            objective=objective,
            relgap=1.0 - bound / objective,
            termination=termination,
        )
        self.solution_info, self.applied_options = solution_info, applied_options
        self.num_solutions = num_solutions
        info('>>> Solution <<<\n%s\n', solution_info)
        return solution_info

    def get_solution(self, A: nx.Graph | None = None) -> tuple[nx.Graph, nx.Graph]:
        if A is None:
            A = self.A
        P, model_options = self.P, self.model_options
        try:
            if self.model_options['feeder_route'] is FeederRoute.STRAIGHT:
                S = self._topology_from_mip_pool()
                G = PathFinder(
                    G_from_S(S, A),
                    P,
                    A,
                    branched=model_options['topology'] is Topology.BRANCHED,
                ).create_detours()
            else:
                S, G = self._investigate_pool(P, A)
        except Exception as exc:
            raise exc
        else:
            G.graph.update(self._make_graph_attributes())
            return S, G
        finally:
            self.solver.close()

    def _objective_at(self, index: int) -> float:
        solver_model = self.solver._solver_model
        solver_model.setParam('SolutionNumber', index)
        return solver_model.getAttr('PoolObjVal')

    def _topology_from_mip_pool(self) -> nx.Graph:
        self._value_map = {
            omovar.name: round(gurvar.Xn)
            for omovar, gurvar in self.solver._pyomo_var_to_solver_var_map.items()
        }
        return self._topology_from_mip_sol()
