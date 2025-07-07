
import pytest
import numpy as np
from optiwindnet.api import WindFarmNetwork, MILP

@pytest.fixture
def simple_coordinates():
    turbines = np.array([[0, 0], [1, 0], [0, 1]])
    substations = np.array([[2, 2]])
    return turbines, substations

@pytest.fixture
def basic_cables():
    return [(100, 1.0), (200, 1.5)]

def test_milp_optimizer_runs(simple_coordinates, basic_cables):
    turbines, substations = simple_coordinates
    wfn = WindFarmNetwork(cables=basic_cables, turbinesC=turbines, substationsC=substations)
    router = MILP(solver_name='cbc', time_limit=10, mip_gap=0.05)
    terse_links = wfn.optimize(router=router)
    assert isinstance(terse_links, np.ndarray)
