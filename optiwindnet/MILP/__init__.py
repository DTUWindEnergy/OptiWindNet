# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/


def solver_factory(solver_name: str):
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
            error('Unsupported solver: %s', solver_name)
