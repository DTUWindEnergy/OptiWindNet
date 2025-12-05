# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import os
import re
import subprocess
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
    _regexp_objective = re.compile(r'^objective value:\s+([0-9]+(?:\.[0-9]+)?)$')
    _regexp_var_value = re.compile(r'^(\S+)\s+([0-9]+(?:\.[0-9]+)?)\s+\(obj:\S+\)$')

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
        self.var_from_name = {
            var.name: var
            for var in chain(metadata.link_.values(), metadata.flow_.values())
        }
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

        #  with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = '.'
        mps_path = os.path.join(tmpdir, 'problem.mps')
        fscip_params_path = os.path.join(tmpdir, 'fscip.prm')
        settings_path = os.path.join(tmpdir, 'options.set')
        sol_path = os.path.join(tmpdir, 'best.sol')
        log_path = os.path.join(tmpdir, 'fscip.log')
        model.writeProblem(mps_path)
        if 'load_coordinator' in applied_options:
            options_LC = applied_options.pop('load_coordinator')
        else:
            options_LC = dict(
                Quiet='TRUE',
                #  LocalBranching='TRUE',
                #  RampUpPhaseProcess=0,
                #  RacingRampUpTerminationCriteria=1,
                StopRacingTimeLimit=15,
            )
        options_LC['TimeLimit'] = str(time_limit)
        with open(fscip_params_path, 'w') as f:
            f.write('\n'.join(f'{k} = {v}' for k, v in options_LC.items()))
        with open(settings_path, 'w') as f:
            if not verbose:
                f.write('display/verblevel = 1\n')  # 1: warnings; 0: no output
            f.write(
                f'limits/gap = {mip_gap}\n'
                + '\n'.join(f'{k} = {v}\n' for k, v in applied_options.items())
            )
        cmd = [
            'fscip',
            fscip_params_path,
            mps_path,
            '-s',
            settings_path,
            '-sth',
            '8',  # number of parallel scip instances
            '-fsol',
            sol_path,
            '-l',
            log_path,
        ]
        if model.getNSols() > 0:
            isol_path = os.path.join(tmpdir, 'warmstart.sol')
            model.writeBestSol(isol_path)
            cmd.extend(['-isol', isol_path])
        subprocess.run(cmd)
        # non-zero variable values are 2*T (T for link_ and T for flow_)
        table_range = range(2 * self.A.graph['T'])
        var_from_name = self.var_from_name
        with open(sol_path, 'r') as f:
            while True:
                first_line = f.readline()
                if not first_line:
                    break
                assert first_line in ('\n', '[ Final Solution ]\n')
                solution = model.createOrigSol()
                objective = self._regexp_objective.match(f.readline()).group(1)
                print('objective:', objective)
                for _, line in zip(table_range, f.readlines()):
                    m = self._regexp_var_value.match(line)
                    if m is not None:
                        name, value = m.groups()
                        model.setSolVal(
                            solution, var_from_name[name], round(float(value))
                        )
                    else:
                        error('Unexpected line in', sol_path, ':\n', line)
                print(objective, model.addSol(solution))
        #  model.readSol(sol_path)
        num_solutions = model.getNSols()
        if num_solutions == 0:
            raise OWNSolutionNotFound(
                f'Unable to find a solution. Solver {self.name} terminated with: {model.getStatus()}'
            )
        # TODO: find a way to get that from the logs
        #  bound = model.getDualbound()
        bound = objective
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
