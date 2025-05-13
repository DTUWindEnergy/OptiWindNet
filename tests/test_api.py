import pytest
import numpy as np
from optiwindnet.api import WindFarmNetwork

# ---------- Helper Fixtures ----------

@pytest.fixture
def wf_network_fixture():
    import os, pickle
    path = os.path.join(os.path.dirname(__file__), "fixtures", "wf_network.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------- Initialization & Validation ----------

def test_valid_initialization():
    cables = [(100, 1.0)]
    turbinesC = np.array([[0, 0]])
    substationsC = np.array([[1, 1]])
    wf_network = WindFarmNetwork(cables, turbinesC=turbinesC, substationsC=substationsC)
    assert wf_network.cables == cables

def test_invalid_cables_type():
    with pytest.raises(Exception):
        WindFarmNetwork(cables="invalid")

def test_missing_coordinates_raises():
    with pytest.raises(ValueError):
        WindFarmNetwork(cables=[(100, 1.0)])


# ---------- Network Properties ----------

def test_cost_and_length(wf_network_fixture):
    wf_network_fixture.G = wf_network_fixture.A
    assert isinstance(wf_network_fixture.cost(), (int, float))
    assert isinstance(wf_network_fixture.length(), (int, float))


# ---------- File-based Constructors ----------

def test_from_yaml_invalid_type():
    with pytest.raises(TypeError):
        WindFarmNetwork.from_yaml(123)

def test_from_pbf_invalid_type():
    with pytest.raises(Exception):
        WindFarmNetwork.from_pbf(123)

def test_from_windIO_invalid_type():
    with pytest.raises(TypeError):
        WindFarmNetwork.from_windIO(123)


# ---------- Link Management ----------

def test_terse_links_missing_reverse(wf_network_fixture):
    wf_network_fixture.S = wf_network_fixture.A  # simulate bad graph without reverse
    with pytest.raises(Exception):
        wf_network_fixture.terse_links()

def test_update_from_terse_links_valid(wf_network_fixture):
    terse_links = np.array([0])
    wf_network_fixture.S = wf_network_fixture.A  # simulate graph
    wf_network_fixture.update_from_terse_links(terse_links)

def test_update_from_terse_links_invalid_type(wf_network_fixture):
    with pytest.raises(ValueError):
        wf_network_fixture.update_from_terse_links(np.array([1.5]))

def test_update_from_terse_links_invalid_shape(wf_network_fixture):
    with pytest.raises(ValueError):
        wf_network_fixture.update_from_terse_links(np.array([[1, 2]]))


# ---------- Gradient Calculation ----------

def test_gradient_invalid_type(wf_network_fixture):
    with pytest.raises(ValueError):
        wf_network_fixture.gradient(gradient_type="invalid")

def test_gradient_valid_output_shapes(wf_network_fixture):
    wt_grad, ss_grad = wf_network_fixture.gradient()
    assert wt_grad.shape[1] == 2
    assert ss_grad.shape[1] == 2


# ---------- Optimization ----------

def test_optimize_runs_and_returns_terse_links(wf_network_fixture):
    terse_links = wf_network_fixture.optimize()
    assert isinstance(terse_links, np.ndarray)


# ---------- Visualization ----------

def test_plot_methods(wf_network_fixture):
    assert wf_network_fixture.plot() is not None
    assert wf_network_fixture.plot_location() is not None
    assert wf_network_fixture.plot_available_links() is not None
    assert wf_network_fixture.plot_navigation_mesh() is not None
    assert wf_network_fixture.plot_selected_links() is not None

def test_plot_original_vs_buffered_executes(wf_network_fixture):
    # Only run if your fixture has _borderC_original, otherwise skip
    if hasattr(wf_network_fixture, '_borderC_original'):
        wf_network_fixture.plot_original_vs_buffered()


# ---------- Export ----------

def test_get_network_returns_array(wf_network_fixture):
    wf_network_fixture.G = wf_network_fixture.A
    network = wf_network_fixture.get_network()
    assert isinstance(network, np.ndarray)


# ---------- Representation ----------

def test_repr_svg_returns_string(wf_network_fixture):
    svg = wf_network_fixture._repr_svg_()
    assert isinstance(svg, str) or svg is not None
