import pytest
import numpy as np
import multiprocessing

from optiwindnet.synthetic import toyfarm
from optiwindnet.mesh import make_planar_embedding
from optiwindnet.MILP import solver_factory, ModelOptions
from optiwindnet.interarraylib import terse_links_from_S


# topology in terse links for toy_farm at capacity=5
_terse_toy_farm_5 = np.array([2, -1, 1, 2, -1, -1, 3, 4, -1, 5, 8, 8])
_CAPACITY = 5
_RUNTIME = 1
_GAP = 0.001


# Loading solvers ortools and scip within the same python instance causes DLL hell
# (ortools contains a SCIP library). Typical usage of OWN is with a single solver.
# Use a workaround for tests: spawn a new python process for each solver.
def _worker_MILP_solver(P, A, solver_name, queue) -> None | tuple:
    solver = solver_factory(solver_name)
    solver.set_problem(P, A, capacity=_CAPACITY, model_options=ModelOptions())
    solution_info = solver.solve(time_limit=_RUNTIME, mip_gap=_GAP)
    queue.put((solution_info, solver.get_solution()))


@pytest.fixture(scope='module')
def P_A_toy():
    L = toyfarm()
    P, A = make_planar_embedding(L)
    return P, A


@pytest.mark.parametrize(
    'solver_name',
    ['ortools', 'gurobi', 'cplex', 'highs', 'scip', 'cbc', 'fscip'],
)
def test_MILP_solvers(P_A_toy, solver_name):
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue(maxsize=1)
    p = ctx.Process(target=_worker_MILP_solver, args=(*P_A_toy, solver_name, queue))

    try:
        p.start()
    except (FileNotFoundError, ModuleNotFoundError):
        pytest.skip(f'{solver_name} not available')
    finally:
        p.join()

    solution_info, (S, _) = queue.get()
    assert solution_info.termination.lower() == 'optimal'
    assert (terse_links_from_S(S) == _terse_toy_farm_5).all()
