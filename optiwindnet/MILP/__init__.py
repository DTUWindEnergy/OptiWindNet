# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import shutil
import sys
from importlib.util import find_spec

from ._core import (
    FeederLimit,
    FeederRoute,
    ModelMetadata,
    ModelOptions,
    OWNSolutionNotFound,
    OWNWarmupFailed,
    SolutionInfo,
    Solver,
    Topology,
)

__all__ = (
    'FeederLimit',
    'FeederRoute',
    'ModelMetadata',
    'ModelOptions',
    'OWNSolutionNotFound',
    'OWNWarmupFailed',
    'SolutionInfo',
    'Solver',
    'Topology',
    'solver_factory',
)


def _reject_loaded_rivals(solver_name: str, package: str, *rivals: str) -> None:
    """Refuse `package` if a rival wrapping the same native library is loaded.

    Args:
      solver_name: solver name as requested, for the error message.
      package: package the solver backend is about to load.
      rivals: packages that must not already be in this interpreter instance.

    Raises:
      RuntimeError: if any of `rivals` has been imported.
    """
    if loaded := [rival for rival in rivals if rival in sys.modules]:
        raise RuntimeError(
            f"Solver '{solver_name}' requires package '{package}', which must not share"
            f' a Python interpreter instance with: {", ".join(loaded)}. '
            "Package 'ortools' bundles its own copies of the HiGHS and SCIP libraries "
            'and their versions may differ from the ones in the standalone packages. '
            'Use only one of the rival solvers per process: restart the interpreter '
            '(or Jupyter kernel) to switch, or run each solver in its own subprocess.'
        )


def solver_factory(solver_name: str) -> Solver:
    """Create a Solver object tied to the specified external MILP solver.

    Note that the only solver that is a dependency of OptiWindNet is ``'ortools'``.
    Check OptiWindNet's documentation on how to install optional solvers.

    Legacy compatibility: if ``solver_name == 'ortools'`` then the CP-SAT backend
    is used.

    Args:
      solver_name: one of ``'ortools.cp_sat'``, ``'ortools.gscip'``,
        ``'ortools.highs'``, ``'cplex'``, ``'gurobi'``, ``'cbc'``, ``'scip'``,
        ``'highs'``.

    Returns:
      Solver instance that can produce solutions for the cable routing problem.

    Raises:
      RuntimeError: if the solver's backend package shares a native library with a
        backend already in use in this interpreter instance (``'ortools*'`` clashes
        with both ``'highs'`` and ``'scip'``/``'fscip'``).
    """
    requested = solver_name
    solver_name, *backend = solver_name.split('.', maxsplit=1)
    match solver_name:
        case 'ortools':
            if find_spec('ortools'):
                _reject_loaded_rivals(requested, 'ortools', 'highspy', 'pyscipopt')
                from .ortools import SolverORTools

                return SolverORTools(backend[0] if backend else 'cp_sat')
            raise ModuleNotFoundError(
                "Package 'ortools' not found. Try 'pip install ortools'."
            )
        case 'cplex':
            if find_spec('cplex'):
                from .cplex import SolverCplex

                return SolverCplex()
            raise ModuleNotFoundError(
                "Package 'cplex' not found. Try 'pip install cplex' or "
                "'conda install -c IBMDecisionOptimization cplex'."
            )
        case 'gurobi':
            if find_spec('gurobipy'):
                from .gurobi import SolverGurobi

                return SolverGurobi()
            raise ModuleNotFoundError(
                "Package 'gurobipy' not found. Try 'pip install gurobipy' or "
                "'conda install -c Gurobi gurobi'."
            )
        case 'cbc':
            if shutil.which('cbc'):
                from .pyomo import SolverPyomo

                return SolverPyomo(solver_name)
            raise FileNotFoundError(
                "Executable 'cbc' not found. Ensure the system PATH includes the "
                "path to 'cbc' or try 'conda install -c conda-forge coin-or-cbc'."
            )
        case 'scip':
            if find_spec('pyscipopt'):
                _reject_loaded_rivals(requested, 'pyscipopt', 'ortools')
                from .scip import SolverSCIP

                return SolverSCIP()
            raise ModuleNotFoundError(
                "Package 'pyscipopt' not found. Try 'pip install pyscipopt' or "
                "'conda install -c conda-forge pyscipopt'."
            )
        case 'fscip':
            if find_spec('pyscipopt'):
                if shutil.which('fscip'):
                    _reject_loaded_rivals(requested, 'pyscipopt', 'ortools')
                    from .fscip import SolverFSCIP

                    return SolverFSCIP()
                else:
                    raise FileNotFoundError(
                        "Executable 'fscip' not found. Ensure the system PATH includes"
                        " the path to 'fscip' (part of scipoptsuite from"
                        ' https://www.scipopt.org).'
                    )
            else:
                raise ModuleNotFoundError(
                    "Package 'pyscipopt' not found. Try 'pip install pyscipopt' or "
                    "'conda install -c conda-forge pyscipopt'."
                )
        case 'highs':
            if find_spec('highspy'):
                _reject_loaded_rivals(requested, 'highspy', 'ortools')
                # pyomo defers its own 'highspy' import to solve time; import it here
                # so that a later 'ortools*' request sees it in sys.modules
                import highspy  # noqa: F401
                from pyomo.contrib.appsi.solvers.highs import Highs

                from .pyomo import SolverPyomoAppsi

                return SolverPyomoAppsi(solver_name, Highs)
            raise ModuleNotFoundError(
                "Package 'highspy' not found. Try 'pip install highspy' or "
                "'conda install -c conda-forge highspy'."
            )
        case _:
            raise ValueError(f'Unsupported solver: {solver_name}')
