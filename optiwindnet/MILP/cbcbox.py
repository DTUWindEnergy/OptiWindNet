# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from itertools import chain
from types import MappingProxyType
from typing import Any

from ortools.math_opt.io.python import (
    mps_converter,  # pyrefly: ignore[missing-module-attribute]
)

from ._core import OWNSolutionNotFound, SolutionInfo, physical_core_count
from .ortools import SolverORTools, make_min_length_model, warmup_model

__all__ = ('SolverCbcbox', 'make_min_length_model', 'warmup_model')

_lggr = logging.getLogger(__name__)

_RESULT_PREFIX = 'Result - '
_TERMINATION_FROM_RESULT = (
    ('Optimal solution found', 'optimal'),
    ('Stopped on time', 'max_time'),
    ('Stopped on iterations', 'max_iterations'),
    ('Problem proven infeasible', 'infeasible'),
    ('Linear relaxation infeasible', 'infeasible'),
    ('Linear relaxation unbounded', 'unbounded'),
)
# CBC prints solution values with 15 significant digits
_INTEGRALITY_TOL = 1e-6


def _read_cbc_log(
    lines: Iterable[str], log: Callable[[str], Any] | None
) -> tuple[str, dict[str, float]]:
    """Consume CBC's output, keeping only the final result summary.

    The log is scanned in a single pass without being stored, so its size does
    not matter. The summary is the ``Result - <status>`` line and the
    ``<label>: <number>`` lines after it (e.g. ``Lower bound``, ``Total time
    (Wallclock seconds)``). It is absent if CBC stops before branch-and-cut
    (e.g. time limit reached while solving the root LP relaxation).

    Args:
      lines: CBC's output lines.
      log: function to forward each line to, if not None.

    Returns:
      Tuple of the result status (empty string if absent) and the summary figures
      by label.
    """
    result, summary = '', {}
    for line in lines:
        line = line.rstrip('\n')
        if log is not None:
            log(line)
        if line.startswith(_RESULT_PREFIX):
            result, summary = line[len(_RESULT_PREFIX) :], {}
        elif result:
            label, sep, figures = line.partition(':')
            if sep:
                try:
                    summary[label] = float(figures.split(maxsplit=1)[0])
                except (IndexError, ValueError):
                    pass
    return result, summary


def _read_cbc_solution(
    path: str, var_from_name: Mapping[str, Any]
) -> tuple[float, dict[int, int]] | None:
    """Read the variable values of a CBC solution file (``-solu`` output).

    Only non-zero variables are listed in the file. If CBC stops without an
    integer solution, the file holds the LP relaxation instead, whose header is
    unreliable (it may read ``Optimal``), so integrality is checked on the values.

    Args:
      path: solution file written by CBC.
      var_from_name: MathOpt variables to read, by name.

    Returns:
      Tuple of the objective value and the rounded values by variable id, or None
      if the file is missing or the solution is not integer.
    """
    if not os.path.exists(path):
        return None
    with open(path) as f:
        header = f.readline()
        values = {var.id: 0 for var in var_from_name.values()}
        for line in f:
            # line fields: [**] <index> <name> <value> <objective coefficient>
            *_, name, value, _ = line.split()
            var = var_from_name.get(name)
            if var is None:
                continue
            value = float(value)
            rounded = round(value)
            if var.integer and abs(value - rounded) > _INTEGRALITY_TOL:
                return None
            values[var.id] = rounded
    # header format: <status> - objective value <objective>
    return float(header.rpartition(' ')[2]), values


class SolverCbcbox(SolverORTools):
    """OR-Tools MathOpt model solved by the cbc executable from package cbcbox.

    The model is exported in Free MPS format and solved in a subprocess. Only the
    final solution is retrieved (pool of size one).

    Args:
      cbc_bin: path to the cbc executable.
    """

    def __init__(self, cbc_bin: str):
        self.cbc_bin = cbc_bin
        self.backend = 'cbcbox'
        self.name = 'ortools.cbcbox'
        self.log_callback = None
        self.options = {
            'Dins': 'on',
        }
        self._cbc_result = ''

    def solve(
        self,
        time_limit: float,
        mip_gap: float,
        options: Mapping[str, Any] = MappingProxyType({}),
        verbose: bool = False,
    ) -> SolutionInfo:
        """Solve the model with the cbc executable.

        Option ``threads`` defaults to the number of physical cores. Other items
        in ``options`` become cbc command-line parameters: ``{'key': value}`` is
        passed as ``-key value``, or as ``-key`` if value is True or None (the
        item is dropped if value is False).
        """
        try:
            model = self.model
        except AttributeError as exc:
            exc.args += ('.set_problem() must be called before .solve()',)
            raise
        metadata = self.metadata
        applied_options = {**self.options, **options}
        var_from_name = {
            var.name: var
            for var in chain(metadata.link_.values(), metadata.flow_.values())
        }
        cmd = [self.cbc_bin, 'problem.mps']
        if math.isfinite(time_limit):
            cmd += ['-sec', str(time_limit)]
        if math.isfinite(mip_gap):
            cmd += ['-ratio', str(mip_gap)]
        if metadata.solution_hint:
            cmd += ['-mipstart', 'warmstart.mst']
        for key, val in {'threads': physical_core_count(), **applied_options}.items():
            if val is not False:
                cmd.append('-' + key.lstrip('-'))
                if val is not True and val is not None:
                    cmd.append(str(val))
        cmd += ['-solve', '-solu', 'problem.soln', '-quit']
        log = self.log_callback or (print if verbose else None)

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'problem.mps'), 'w') as f:
                f.write(mps_converter.model_proto_to_mps(model.export_model()))
            if metadata.solution_hint:
                with open(os.path.join(tmpdir, 'warmstart.mst'), 'w') as f:
                    f.writelines(
                        f'{var.name} {round(val)}\n'
                        for var, val in metadata.solution_hint.items()
                    )
            start_time = time.perf_counter()
            with subprocess.Popen(
                cmd,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding='utf-8',
                errors='replace',
            ) as proc:
                # pyrefly: ignore[bad-argument-type]
                result, summary = _read_cbc_log(proc.stdout, log)
            elapsed_time = time.perf_counter() - start_time
            solution = (
                _read_cbc_solution(os.path.join(tmpdir, 'problem.soln'), var_from_name)
                if result
                else None
            )

        self._cbc_result = result
        termination = next(
            (term for prefix, term in _TERMINATION_FROM_RESULT
             if result.startswith(prefix)),
            result or 'unknown',
        )  # fmt: skip
        self.stopping = {'mip_gap': mip_gap, 'time_limit': time_limit}
        if solution is None:
            raise OWNSolutionNotFound(
                f'Unable to find a solution. Solver {self.name} terminated'
                f' with: {termination}'
            )
        objective = solution[0]
        # CBC prints the bound with 6 significant digits, which may round it above
        # the objective
        bound = min(
            summary.get(
                'Lower bound', objective if termination == 'optimal' else -math.inf
            ),
            objective,
        )
        self._solution_pool = [solution]
        self.num_solutions = 1
        solution_info = SolutionInfo(
            runtime=summary.get('Total time (Wallclock seconds)', elapsed_time),
            bound=bound,
            objective=objective,
            relgap=1.0 - bound / objective,
            termination=termination,
        )
        return self._record_incumbent(solution_info, applied_options)

    def _solver_termination_detail(self) -> str:
        return self._cbc_result
