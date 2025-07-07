
import pytest
import numpy as np
from optiwindnet.api import WindFarmNetwork, Heuristic

@pytest.fixture
def simple_coordinates():
    turbines = np.array([[0, 0], [1, 0], [0, 1]])
    substations = np.array([[2, 2]])
    return turbines, substations

@pytest.fixture
def basic_cables():
    return [(100, 1.0), (200, 1.5)]

def test_init_creates_valid_network(simple_coordinates, basic_cables):
    turbines, substations = simple_coordinates
    network = WindFarmNetwork(cables=basic_cables, turbinesC=turbines, substationsC=substations)
    assert network.L is not None
    assert network.cables_capacity == 200

def test_invalid_cables_raises_error(simple_coordinates):
    turbines, substations = simple_coordinates
    with pytest.raises(Exception):
        WindFarmNetwork(cables='invalid', turbinesC=turbines, substationsC=substations)

def test_optimize_runs_and_returns_terse_links(simple_coordinates, basic_cables):
    turbines, substations = simple_coordinates
    network = WindFarmNetwork(cables=basic_cables, turbinesC=turbines, substationsC=substations)
    terse_links = network.optimize()
    assert isinstance(terse_links, np.ndarray)
    assert len(terse_links) == len(turbines)

def test_cost_and_length_are_nonzero_after_optimize(simple_coordinates, basic_cables):
    turbines, substations = simple_coordinates
    network = WindFarmNetwork(cables=basic_cables, turbinesC=turbines, substationsC=substations)
    network.optimize()
    assert network.cost() > 0
    assert network.length() > 0

def test_update_from_terse_links(simple_coordinates, basic_cables):
    turbines, substations = simple_coordinates
    network = WindFarmNetwork(cables=basic_cables, turbinesC=turbines, substationsC=substations)
    terse_links = network.optimize()
    updated_G = network.update_from_terse_links(terse_links)
    assert updated_G.number_of_edges() == len(terse_links)

def test_get_network_returns_structured_array(simple_coordinates, basic_cables):
    turbines, substations = simple_coordinates
    network = WindFarmNetwork(cables=basic_cables, turbinesC=turbines, substationsC=substations)
    network.optimize()
    net_array = network.get_network()
    assert isinstance(net_array, np.ndarray)
    assert 'src' in net_array.dtype.names
    assert 'cost' in net_array.dtype.names
