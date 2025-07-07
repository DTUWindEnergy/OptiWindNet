
import pytest
import numpy as np
from optiwindnet.api import WindFarmNetwork, MetaHeuristic

@pytest.fixture
def simple_coordinates():
    turbines = np.array([[0, 0], [1, 0], [0, 1]])
    substations = np.array([[2, 2]])
    return turbines, substations

@pytest.fixture
def basic_cables():
    return [(100, 1.0), (200, 1.5)]

def test_metaheuristic_optimizer_runs(simple_coordinates, basic_cables):
    turbines, substations = simple_coordinates
    wfn = WindFarmNetwork(cables=basic_cables, turbinesC=turbines, substationsC=substations)
    router = MetaHeuristic(solver='HGS', time_limit=5, max_iter=5)
    terse_links = wfn.optimize(router=router)
    assert isinstance(terse_links, np.ndarray)

def test_metaheuristic_raises_with_unknown_solver(simple_coordinates, basic_cables):
    turbines, substations = simple_coordinates
    network = WindFarmNetwork(cables=basic_cables, turbinesC=turbines, substationsC=substations)
    router = MetaHeuristic(solver='Invalid', time_limit=5)
    with pytest.raises(ValueError):
        network.optimize(router=router)

