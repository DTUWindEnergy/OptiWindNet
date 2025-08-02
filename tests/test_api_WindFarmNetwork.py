
import pytest
import numpy as np
from optiwindnet.api import WindFarmNetwork


# ========== Fixtures ==========

@pytest.fixture
def dummy_coordinates():
    turbines = np.array([[0, 0], [1, 0], [2, 0]])
    substations = np.array([[1, 1]])
    return turbines, substations

@pytest.fixture
def dummy_cables():
    return [(10, 100), (20, 150)]  # (capacity, cost)

@pytest.fixture
def simple_network(dummy_coordinates, dummy_cables):
    turbines, substations = dummy_coordinates
    return WindFarmNetwork(
        cables=dummy_cables,
        turbinesC=turbines,
        substationsC=substations,
        name='Test Farm'
    )

# ========== Tests ==========

def test_initialization(simple_network):
    assert isinstance(simple_network, WindFarmNetwork)
    assert simple_network.L is not None
    assert simple_network.router is not None

def test_optimize_returns_valid_terse_links(simple_network):
    terse = simple_network.optimize()
    T = simple_network.L.graph['T']

    assert isinstance(terse, np.ndarray)
    assert terse.shape == (T,)
    assert np.issubdtype(terse.dtype, np.integer)
    assert (terse != np.arange(T)).all()  # No self-loops

def test_cost_and_length(simple_network):
    simple_network.optimize()
    cost = simple_network.cost()
    length = simple_network.length()

    assert isinstance(cost, (int, float))
    assert isinstance(length, (int, float))
    assert cost >= 0
    assert length >= 0

def test_get_network(simple_network):
    simple_network.optimize()
    net = simple_network.get_network()

    assert isinstance(net, np.ndarray)
    assert net.dtype.names is not None  # structured array expected
    assert {'src', 'tgt', 'length', 'load', 'reverse', 'cable', 'cost'}.issubset(net.dtype.names)

def test_gradient_outputs(simple_network):
    simple_network.optimize()
    grad_t, grad_s = simple_network.gradient()

    assert isinstance(grad_t, np.ndarray)
    assert isinstance(grad_s, np.ndarray)
    assert grad_t.shape[1] == 2
    assert grad_s.shape[1] == 2
    assert np.isfinite(grad_t).all()
    assert np.isfinite(grad_s).all()

def test_gradient_type_variants(simple_network):
    simple_network.optimize()

    for kind in ['length', 'cost']:
        grad_t, grad_s = simple_network.gradient(gradient_type=kind)
        assert grad_t.shape[1] == 2

    with pytest.raises(ValueError):
        simple_network.gradient(gradient_type='invalid')

def test_update_from_terse_links(simple_network):
    terse = simple_network.optimize()
    new_G = simple_network.update_from_terse_links(terse)

    assert new_G is not None
    assert isinstance(new_G.graph['VertexC'], np.ndarray)

def test_terse_links_correctness(simple_network):
    terse = simple_network.optimize()
    T = simple_network.L.graph['T']

    assert terse.shape == (T,)
    assert all(isinstance(x, (int, np.integer)) for x in terse)

def test_map_detour_vertex(simple_network):
    simple_network.optimize()
    mapping = simple_network.map_detour_vertex()

    if mapping:  # Only test if detours exist
        assert isinstance(mapping, dict)
        for k, v in mapping.items():
            assert isinstance(k, int)
            assert isinstance(v, int)

def test_plot_methods_do_not_crash(simple_network):
    simple_network.optimize()

    # These should render but not return data (no assert needed)
    simple_network.plot()
    simple_network.plot_location()
    simple_network.plot_available_links()
    simple_network.plot_selected_links()
    simple_network.plot_navigation_mesh()
