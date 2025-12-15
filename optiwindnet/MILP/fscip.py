# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import os
import re
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
    _solution_pool: list[tuple[float, dict]]
    _regexp_objective = re.compile(r'^objective value:\s+([0-9]+(?:\.[0-9]+)?)$')
    _regexp_var_value = re.compile(r'^(\S+)\s+([0-9]+(?:\.[0-9]+)?)\s+\(obj:\S+\)$')
    _termination_from_status = {
        # fscip has a very non-standard status description
        'problem is solved': 'optimal',
    }

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

        #  tmpdir = './log/'
        with tempfile.TemporaryDirectory() as tmpdir:
            ug_params_file = 'ug.prm'
            lc_settings_file = 'options_lc.set'
            #  root_settings_file = 'options_root.set'
            settings_file = 'options.set'
            #  settings_path = './feasibility.set'
            sol_file = 'best.sol'
            problem_file = 'problem.cip'
            problem_path = os.path.join(tmpdir, problem_file)
            ug_params_path = os.path.join(tmpdir, ug_params_file)
            lc_settings_path = os.path.join(tmpdir, lc_settings_file)
            #  root_settings_path = os.path.join(tmpdir, root_settings_file)
            settings_path = os.path.join(tmpdir, 'options.set')
            #  #  settings_path = './feasibility.set'
            sol_path = os.path.join(tmpdir, sol_file)
            model.writeProblem(problem_path, verbose=False)
            if 'parallel/maxnthreads' in applied_options:
                # instead of setting n_threads for SCIP instances, set it for fscip
                n_threads = applied_options.pop('parallel/maxnthreads')
            else:
                n_threads = os.cpu_count()
            if 'ubiquity_generator' in applied_options:
                ug_params = applied_options.pop('ubiquity_generator')
            else:
                ug_params = dict(
                    Quiet='FALSE',
                    #  OutputParaParams = 2,
                    OutputParaParams=0,
                    OutputTabularSolvingStatus='TRUE',
                    TabularSolvingStatusInterval=1,
                    LogSolvingStatus='TRUE',
                    # There is a bug with fscip in scipoptsuite 10.0.0:
                    #   When the ramp-up process ends, the program terminates.
                    #   The only way to make it search for a given duration
                    #   is to skip the ramp up entirely. Quite poor performance
                    #   when compared to SCIP's solveConcurrent().
                    RampUpPhaseProcess=0,
                    #  RacingRampUpTerminationCriteria=1,
                    #  StopRacingTimeLimit=15,
                    RacingStatBranching='FALSE',
                    # not clear how TRUE and FALSE in LocalBranching compare
                    #  LocalBranching='TRUE',
                    LogSolvingStatusFilePath='"./"',
                    LogTasksTransferFilePath='"./"',
                    SolutionFilePath='"./"',
                    CheckpointFilePath='"./"',
                    TempFilePath='"./"',
                )
            ug_params['TimeLimit'] = str(time_limit)
            with open(ug_params_path, 'w') as f:
                f.write('\n'.join(f'{k} = {v}' for k, v in ug_params.items()))
            options_lc = dict()
            options_lc['limits/gap'] = str(mip_gap)
            with open(lc_settings_path, 'w') as f:
                f.write('\n'.join(f'{k} = {v}' for k, v in options_lc.items()))
            with open(settings_path, 'w') as f:
                if not verbose:
                    f.write('display/verblevel = 1\n')  # 1: warnings; 0: no output
                f.write(
                    #  f'limits/gap = {mip_gap}\n' +
                    '\n'.join(f'{k} = {v}\n' for k, v in applied_options.items())
                )
            cmd = [
                'fscip',
                ug_params_file,
                problem_file,
                '-sl',
                lc_settings_file,
                #  '-sr',
                #  root_settings_file,
                '-s',
                settings_file,
                '-sth',
                str(n_threads),  # number of parallel scip instances
                '-fsol',
                sol_file,
            ]
            if model.getNSols() > 0:
                isol_path = str(os.path.join(tmpdir, 'warmstart.sol')).replace(
                    '\\', '/'
                )
                model.writeBestSol(isol_path)
                cmd.extend(['-isol', isol_path])
            subprocess.run(cmd, cwd=tmpdir)
            # non-zero variable values are 2*T (T for link_ and T for flow_)
            table_range = range(2 * self.A.graph['T'])
            var_from_name = self.var_from_name
            objective = float('inf')
            with open(os.path.join(tmpdir, sol_path), 'r') as f:
                while True:
                    try:
                        first_line = next(f)
                    except StopIteration:
                        break
                    assert first_line in ('\n', '[ Final Solution ]\n')
                    solution = model.createOrigSol()
                    objective = float(self._regexp_objective.match(next(f)).group(1))
                    for _, line in zip(table_range, f):
                        m = self._regexp_var_value.match(line)
                        if m is not None:
                            name, value = m.groups()
                            model.setSolVal(
                                solution, var_from_name[name], round(float(value))
                            )
                        else:
                            error('Unexpected line in %s:\n%s', sol_path, line)
                    model.addSol(solution)
                    #  accepted = model.addSol(solution)
                    #  print(f'<obj={objective}> accepted?:', accepted)
                    #  if not accepted:
                    #      # model.checkSol() cause a segfault
                    #      print(model.checkSol(solution))
            num_solutions = model.getNSols()
            if num_solutions == 0:
                raise OWNSolutionNotFound(
                    f'Unable to find a solution. Solver {self.name} terminated with: {model.getStatus()}'
                )
            termination = 'unknown'
            solving_time = float('nan')
            with open(os.path.join(tmpdir, 'problem_LC0_T.status'), 'r') as f:
                last_data_row = ''
                messages = []
                for line in f:
                    if not line.endswith('%\n'):
                        messages.append(line)
                        continue
                    last_data_row = line
                # look for summary fields in messages
                for line in messages:
                    if line.startswith('SCIP Status        : '):
                        status = line[21:].strip()
                        termination = self._termination_from_status.get(status, status)
                        continue
                    elif line.startswith('Total Time         : '):
                        solving_time = float(line[21:])
                        continue
                    elif line.startswith('  Dual Bound       : '):
                        bound = float(line[21:])
                        break
                else:
                    # summary was not saved in status file, use data from the last row
                    termination = 'truncated after ramp up'
                    col_offset = int(last_data_row[0] == '*')
                    last_status = last_data_row.split()
                    solving_time = float(last_status[0 + col_offset])
                    bound = float(last_status[5 + col_offset])
                for line in messages:
                    print(line)
        solution_pool = [(model.getSolObjVal(sol), sol) for sol in model.getSols()]
        solution_pool.sort()
        self._solution_pool = solution_pool
        self.num_solutions = num_solutions
        solution_info = SolutionInfo(
            runtime=solving_time,
            bound=bound,
            objective=solution_pool[0][0],
            # SCIP offers model.getGap(), but its denominator is the bound
            relgap=1.0 - bound / objective,
            termination=termination,
        )
        self.stopping = dict(mip_gap=mip_gap, time_limit=time_limit)
        self.solution_info, self.applied_options = solution_info, applied_options
        info('>>> Solution <<<\n%s\n', solution_info)
        return solution_info

    def get_solution(self, A: nx.Graph | None = None) -> tuple[nx.Graph, nx.Graph]:
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
        objective_value, self._value_map = self._solution_pool[index]
        return objective_value

    def _topology_from_mip_pool(self) -> nx.Graph:
        return self._topology_from_mip_sol()
