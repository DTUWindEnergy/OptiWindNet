# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import os
import subprocess
import tempfile
from itertools import chain
from typing import Any

import networkx as nx

from ..interarraylib import G_from_S
from ..pathfinding import PathFinder
from .scip import make_min_length_model, warmup_model
from ._core import (
    FeederRoute,
    ModelOptions,
    OWNSolutionNotFound,
    PoolHandler,
    SolutionInfo,
    Solver,
    Topology,
)

__all__ = ('make_min_length_model', 'warmup_model')

_lggr = logging.getLogger(__name__)
error, warn, info = _lggr.error, _lggr.warning, _lggr.info


class SolverFSCIP(Solver, PoolHandler):
    name: str = 'fscip'
    solution_pool: list[tuple[float, dict]]

    def __init__(self):
        self.options = {}

    def _link_val(self, var: Any) -> int:
        # work-around for SCIP: use round() to coerce link_ value (should be binary)
        #   values for link_ variables are floats and may be slightly off of 0
        return round(self._value_map[var])

    def _flow_val(self, var: Any) -> int:
        return self._value_map[var]

    def set_problem(
        self,
        P: nx.PlanarEmbedding,
        A: nx.Graph,
        capacity: int,
        model_options: ModelOptions,
        warmstart: nx.Graph | None = None,
    ):
        self.P, self.A, self.capacity = P, A, capacity
        self.model_options = model_options
        model, metadata = make_min_length_model(self.A, self.capacity, **model_options)
        self.model, self.metadata = model, metadata
        if warmstart is not None:
            warmup_model(model, metadata, warmstart)

    def solve(
        self,
        time_limit: float,
        mip_gap: float,
        options: dict[str, Any] = {},
        verbose: bool = False,
    ) -> SolutionInfo:
        """Wrapper that calls the fscip executable."""
        try:
            model = self.model
        except AttributeError as exc:
            exc.args += ('.set_problem() must be called before .solve()',)
            raise
        applied_options = self.options | options

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = '.'
            mps_path = os.path.join(tmpdir, 'problem.mps')
            fscip_params_path = os.path.join(tmpdir, 'fscip.prm')
            settings_path = os.path.join(tmpdir, 'options.set')
            sol_path = os.path.join(tmpdir, 'best.sol')
            model.writeProblem(mps_path)
            with open(fscip_params_path, 'w') as f:
                f.write(
                    f'Quiet = FALSE\nTimeLimit = {time_limit}\n'
                    'LocalBranching = TRUE\n'  # RampUpPhaseProcess = 0\n'
                    'RacingRampUpTerminationCriteria = 1\n'
                    'StopRacingTimeLimit = 20\n'
                )
            with open(settings_path, 'w') as f:
                f.writelines(
                    chain(
                        (f'limits/gap = {mip_gap}\n',),
                        (f'{k} = {v}\n' for k, v in applied_options.items()),
                    )
                )
                if not verbose:
                    f.write('display/verblevel = 1\n')  # 1: warnings; 0: no output
            cmd = [
                'fscip',
                fscip_params_path,
                mps_path,
                '-s',
                settings_path,
                '-sth',
                '6',
                '-fsol',
                sol_path,
            ]
            if model.getNSols() > 0:
                isol_path = os.path.join(tmpdir, 'warmstart.sol')
                model.writeBestSol(isol_path)
                cmd.extend(['-isol', isol_path])
                #  with open(isol_path, 'r') as f:
                #      for line in f.readlines():
                #          print(line, end='')
            result = subprocess.run(cmd)
            print('DONE')
            with open(sol_path, 'r') as f:
                for line in f.readlines():
                    print(line, end='')
            print(result)
            model.readSol(sol_path)
        num_solutions = model.getNSols()
        if num_solutions == 0:
            raise OWNSolutionNotFound(
                f'Unable to find a solution. Solver {self.name} terminated with: {model.getStatus()}'
            )
        bound = model.getDualbound()
        objective = model.getObjVal()
        self.solution_pool = [(model.getSolObjVal(sol), sol) for sol in model.getSols()]
        self.num_solutions = num_solutions
        solution_info = SolutionInfo(
            runtime=model.getSolvingTime(),
            bound=bound,
            objective=objective,
            # SCIP offers model.getGap(), but its denominator is the bound
            relgap=1.0 - bound / objective,
            termination=model.getStatus(),
        )
        self.solution_info, self.applied_options = solution_info, applied_options
        info('>>> Solution <<<\n%s\n', solution_info)
        return solution_info

    def get_solution(self, A: nx.Graph | None = None) -> tuple[nx.Graph, nx.Graph]:
        if A is None:
            A = self.A
        P, model_options = self.P, self.model_options
        if model_options['feeder_route'] is FeederRoute.STRAIGHT:
            S = self.topology_from_mip_pool()
            G = PathFinder(
                G_from_S(S, A),
                P,
                A,
                branched=model_options['topology'] is Topology.BRANCHED,
            ).create_detours()
        else:
            S, G = self.investigate_pool(P, A)
        G.graph.update(self._make_graph_attributes())
        return S, G

    def objective_at(self, index: int) -> float:
        objective_value, self._value_map = self.solution_pool[index]
        return objective_value

    def topology_from_mip_pool(self) -> nx.Graph:
        return self.topology_from_mip_sol()
