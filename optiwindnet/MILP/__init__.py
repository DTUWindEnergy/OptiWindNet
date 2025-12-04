# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import shutil
from importlib.util import find_spec

from ._core import (
    FeederLimit,
    FeederRoute,
    ModelMetadata,
    ModelOptions,
    OWNWarmupFailed,
    OWNSolutionNotFound,
    SolutionInfo,
    Solver,
    Topology,
)

__all__ = (
    'Solver',
    'Topology',
    'FeederRoute',
    'FeederLimit',
    'ModelOptions',
    'ModelMetadata',
    'OWNWarmupFailed',
    'OWNSolutionNotFound',
    'SolutionInfo',
    'solver_factory',
)


def solver_factory(solver_name: str) -> Solver:
    """Create a Solver object tied to the specified external MILP solver.

    Note that the only solver that is a dependency of OptiWindNet is 'ortools'.
    Check OptiWindNet's documentation on how to install optional solvers.

    Args:
      solver_name: one of 'ortools', 'cplex', 'gurobi', 'cbc', 'scip', 'highs'.

    Returns:
      Solver instance that can produce solutions for the cable routing problem.
    """
    match solver_name:
        case 'ortools':
            if find_spec('ortools'):
                from .ortools import SolverORTools

                return SolverORTools()
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
                from .scip import SolverSCIP

                return SolverSCIP()
            raise ModuleNotFoundError(
                "Package 'pyscipopt' not found. Try 'pip install pyscipopt' or "
                "'conda install -c conda-forge pyscipopt'."
            )
        case 'fscip':
            if find_spec('pyscipopt'):
                if shutil.which('fscip'):
                    from .fscip import SolverFSCIP

                    return SolverFSCIP()
                else:
                    raise FileNotFoundError(
                        "Executable 'fscip' not found. Ensure the system PATH includes the "
                        "path to 'fscip' (part of scipoptsuite from https://www.scipopt.org)."
                    )
            else:
                raise ModuleNotFoundError(
                    "Package 'pyscipopt' not found. Try 'pip install pyscipopt' or "
                    "'conda install -c conda-forge pyscipopt'."
                )
        case 'highs':
            if find_spec('highspy'):
                from .pyomo import SolverPyomoAppsi
                from pyomo.contrib.appsi.solvers.highs import Highs

                return SolverPyomoAppsi(solver_name, Highs)
            raise ModuleNotFoundError(
                "Package 'highspy' not found. Try 'pip install highspy' or "
                "'conda install -c conda-forge highspy'."
            )
        case _:
            raise ValueError(f'Unsupported solver: {solver_name}')
