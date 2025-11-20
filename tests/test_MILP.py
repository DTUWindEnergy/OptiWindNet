import pytest
import numpy as np

from optiwindnet.synthetic import toyfarm
from optiwindnet.mesh import make_planar_embedding
from optiwindnet.MILP import solver_factory, ModelOptions
from optiwindnet.interarraylib import terse_links_from_S


# topology in terse links for toy_farm at capacity=5
_terse_toy_farm_5 = np.array([2, -1, 1, 2, -1, -1, 3, 4, -1, 5, 8, 8])
_CAPACITY = 5


@pytest.fixture(scope='module')
def P_A_toy():
    L = toyfarm()
    P, A = make_planar_embedding(L)
    return P, A


@pytest.mark.parametrize(
    'solver_name',
    ['ortools', 'gurobi', 'cplex', 'highs', 'scip', 'cbc'],
)
def test_MILP_solvers(P_A_toy, solver_name):
    try:
        solver = solver_factory(solver_name)
    except (FileNotFoundError, ModuleNotFoundError):
        pytest.skip(f'{solver_name} not available')

    solver.set_problem(
        *P_A_toy,
        capacity=_CAPACITY,
        model_options=ModelOptions(),
    )
    solution_info = solver.solve(time_limit=1, mip_gap=0.001)
    assert solution_info.termination.lower() == 'optimal'
    S, _ = solver.get_solution()
    assert (terse_links_from_S(S) == _terse_toy_farm_5).all()
