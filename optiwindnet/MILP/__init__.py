# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging

from ._core import (
    FeederLimit,
    FeederRoute,
    ModelMetadata,
    ModelOptions,
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
    'SolutionInfo',
    'solver_factory',
)

_lggr = logging.getLogger(__name__)


def solver_factory(solver_name: str):
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
            from .ortools import SolverORTools

            return SolverORTools()
        case 'cplex':
            from .cplex import SolverCplex

            return SolverCplex()
        case 'gurobi':
            from .gurobi import SolverGurobi

            return SolverGurobi()
        case 'cbc' | 'scip':
            from .pyomo import SolverPyomo

            return SolverPyomo(solver_name)
        case 'highs':
            # Pyomo's appsi solvers
            from .pyomo import SolverPyomo

            return SolverPyomo(solver_name, prefix='appsi_')
        case _:
            _lggr.error('Unsupported solver: %s', solver_name)
            return None
