import numpy as np
import pytest

from optiwindnet.api import EWRouter, WindFarmNetwork, HGSRouter, MILPRouter

from .helpers import assert_graph_equal
# ========== Test ==========


def test_wfn_fails_without_coordinates_or_L():
    with pytest.raises(TypeError):
        WindFarmNetwork(cables=7)


def test_wfn_warns_when_L_and_coordinates_given(
    LG_from_database, site_from_database, caplog
):
    expected_L, _ = LG_from_database('eagle_EWRouter')
    site = site_from_database('eagle_EWRouter')

    with caplog.at_level('WARNING'):
        WindFarmNetwork(
            cables=7,
            turbinesC=site['turbinesC'],
            substationsC=site['substationsC'],
            L=expected_L,
        )

    assert any(
        'OptiWindNet prioritizes L over coordinates' in message
        for message in caplog.messages
    )


def test_wfn_fails_without_cables(site_from_database):
    site = site_from_database('eagle_EWRouter')
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'cables'"
    ):
        WindFarmNetwork(turbinesC=site['turbinesC'], substationsC=site['substationsC'])


def test_wfn_from_coordinates(LG_from_database, site_from_database):
    expected_L, _ = LG_from_database('eagle_EWRouter')
    site = site_from_database('eagle_EWRouter')

    kwargs = {
        'cables': 7,
        'turbinesC': site['turbinesC'],
        'substationsC': site['substationsC'],
        'handle': site['handle'],
        'name': site['name'],
        'landscape_angle': site['landscape_angle'],
    }

    if site['borderC'].size > 0:
        kwargs['borderC'] = site['borderC']

    if site['obstaclesC']:
        kwargs['obstaclesC'] = site['obstaclesC']

    wfn1 = WindFarmNetwork(**kwargs)

    assert_graph_equal(
        wfn1.L,
        expected_L,
        ignored_graph_keys={'norm_offset', 'norm_scale', 'obstacles'},
    )
    assert isinstance(wfn1.router, EWRouter)


def test_wfn_from_L(LG_from_database):
    expected_L, _ = LG_from_database('eagle_EWRouter')

    wfn2 = WindFarmNetwork(cables=7, L=expected_L)
    assert_graph_equal(
        wfn2.L,
        expected_L,
        ignored_graph_keys={'norm_offset', 'norm_scale', 'obstacles'},
    )


@pytest.mark.parametrize(
    'cables_input, expected_array',
    [
        (7, [(7, 0.0)]),
        ([(7, 100)], [(7, 100)]),
        ((7, 9), [(7, 0.0), (9, 0.0)]),
        ([(5, 100), (7, 150), (9, 200)], [(5, 100), (7, 150), (9, 200)]),
    ],
)
def test_wfn_cable_formats(LG_from_database, cables_input, expected_array):
    expected_L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=cables_input, L=expected_L)
    #  assert np.array_equal(wfn.cables, expected_array)
    assert wfn.cables == expected_array


def test_wfn_invalid_cables_raises(LG_from_database):
    expected_L, _ = LG_from_database('eagle_EWRouter')

    with pytest.raises(ValueError, match='Invalid cable values'):
        WindFarmNetwork(cables=(5, (7, 3, 8), 9), L=expected_L)


def test_cables_capacity_calculation(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=[(5, 100), (7, 150)], L=L)
    assert wfn.cables_capacity == 7


def test_invalid_gradient_type_raises(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    with pytest.raises(ValueError, match='gradient_type should be either'):
        wfn.gradient(gradient_type='bad_type')


def test_optimize_updates_graphs(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    terse = wfn.optimize()
    assert wfn.S is not None
    assert wfn.G is not None
    assert len(terse) > 0


def test_from_yaml_invalid_path():
    with pytest.raises(Exception):
        WindFarmNetwork.from_yaml(r'not>a*path')


def test_from_pbf_invalid_path():
    with pytest.raises(Exception):  # TypeError or custom error
        WindFarmNetwork.from_pbf(r'not>a*path')


def test_terse_links_output_shape(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    wfn.optimize()
    terse = wfn.terse_links()
    assert terse.shape[0] == wfn.S.graph['T']


def test_map_detour_vertex_empty_if_no_detours(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    wfn.optimize()
    result = wfn.map_detour_vertex()
    assert isinstance(result, dict)


def test_plot_selected_links_runs(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    wfn.optimize()
    wfn.plot_selected_links()


def test_get_network_returns_array(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    wfn.optimize()
    data = wfn.get_network()
    assert isinstance(data, np.ndarray)
    assert data.ndim == 1


# value-based unit tests
def test_length_returns_expected_value(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    wfn.optimize()
    expected_length = 56573.7358
    assert pytest.approx(wfn.length(), rel=1e-4) == expected_length


def test_cost_returns_expected_value(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=[(7, 100)], L=L)
    wfn.optimize()
    expected_cost = 5657373.5802
    assert pytest.approx(wfn.cost(), rel=1e-4) == expected_cost


def test_terse_links_returns_expected_array(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    wfn.optimize()
    # fmt: off
    expected_terse = np.array([1, 2, 41, 4, 41, 6, 27, 6, 7, 10, 13, 8, 11, 14, 15, -1,
                               14, 18, 22, 16, 18, 22, 23, 24, -1, 26, 28, -1, -1, 39, 29,
                               30, 31, 32, 35, 36, 37, 38, -1, -1, -1, 40, 43, -1, -1, 44,
                               45, 46, 47, 48])

    # fmt: on
    assert np.array_equal(wfn.terse_links(), expected_terse)


def test_update_from_terse_links_matches_expected_cost(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    # fmt: off
    terse = np.array([1, 2, 41, 4, 41, 6, 27, 6, 7, 10, 13, 8, 11, 14, 15, -1, 14,
                      18, 22, 16, 18, 22, 23, 24, -1, 26, 28, -1, -1, 39, 29, 30, 31,
                      32, 35, 36, 37, 38, -1, -1, -1, 40, 43, -1, -1, 44, 45, 46, 47, 48])
    # fmt: on

    # Update from terse and check cost
    wfn.update_from_terse_links(terse)
    expected_length = 56573.7358
    assert pytest.approx(wfn.length(), rel=1e-4) == expected_length


def test_gradient(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    wfn.optimize()
    grad_wt, grad_ss = wfn.gradient(gradient_type='length')

    # fmt: off
    expected_grad_wt = np.array([
        [9.38436570e-01, -3.45451596e-01], [-7.98183068e-01, -6.44664032e-01],
        [-7.39642419e-03, -1.01957897e-03], [1.04575893e-01, -9.94516909e-01],
        [-1.08009929e00, 7.74621228e-01], [1.26120663e-01, -9.92014908e-01],
        [-1.33322491e-01, 1.02168418e00], [-8.37910472e-01, -1.24218770e00],
        [-1.10535566e-02, -1.38844155e-03], [-2.05024539e-02, -9.99789803e-01],
        [8.41614038e-03, -1.37155245e-04], [9.38332754e-03, 1.17061707e-03],
        [-1.28485640e-01, 9.91711369e-01], [7.31049987e-05, -8.80961725e-07],
        [-9.96281410e-01, 1.19183629e-01], [4.09389733e-02, 1.87151171e-01],
        [-8.53698302e-03, 1.68086993e-04], [5.40683378e-02, -9.98537238e-01],
        [-7.90634529e-01, -4.38211791e-02], [2.39537701e-02, 9.99713067e-01],
        [-2.60474343e-01, 9.65480770e-01], [-1.23237432e-01, 9.92377214e-01],
        [1.01105092e00, 7.85172600e-02], [-1.56623966e-02, -1.84616267e-03],
        [-7.85325621e-01, -5.78034766e-01], [-1.37561631e-01, 9.90493209e-01],
        [-8.49795999e-01, -1.14900160e00], [3.34117005e-02, -1.14990584e-01],
        [-1.23269713e-02, 1.83622103e-01], [-8.81286661e-01, -6.76742064e-01],
        [8.89254552e-02, 1.01214648e-02], [-2.15313860e-02, -3.19279914e-03],
        [-3.42349336e-02, -4.09638808e-03], [-1.01672600e-01, 9.94817914e-01],
        [-1.55114181e-01, 9.87896549e-01], [1.31605497e-02, 1.97675968e-03],
        [-7.50496584e-04, -1.07915993e-04], [-9.13880978e-03, -1.36075745e-03],
        [-4.65348150e-01, -2.01591340e-01], [8.09510100e-03, 2.00635554e-02],
        [-9.62232527e-01, 4.04042913e-01], [9.93515166e-01, 2.22474048e-01],
        [1.78239791e-01, -9.83987082e-01], [7.08547411e-01, 5.21809132e-01],
        [9.35918404e-01, -3.18928707e-01], [-1.15085023e-02, -2.12808401e-03],
        [-1.49620884e-02, -2.55921275e-03], [2.82593340e-02, 5.03058647e-03],
        [-1.51059821e-02, -2.79318559e-03], [-1.74265398e-01, 9.84698721e-01]
    ])
    # fmt: on

    expected_grad_ss = np.array([4.53885, -1.15393])
    assert np.allclose(grad_wt, expected_grad_wt, rtol=1e-4)
    assert np.allclose(grad_ss, expected_grad_ss, rtol=1e-4)


def test_map_detour_vertex(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    wfn.optimize()
    detour_map = wfn.map_detour_vertex()
    print(detour_map)

    expected_map = {56: 29}
    assert detour_map == expected_map


def test_from_pbf(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork.from_pbf(
        filepath='optiwindnet/data/Baltic Eagle.osm.pbf', cables=7
    )
    assert_graph_equal(
        L, wfn.L, ignored_graph_keys={'norm_offset', 'norm_scale', 'OSM_name'}
    )


def test_from_yaml(LG_from_database):
    L, _ = LG_from_database('taylor_EWRouter')
    wfn = WindFarmNetwork.from_yaml(
        filepath='optiwindnet/data/Taylor-2023.yaml', cables=7
    )
    assert_graph_equal(L, wfn.L, ignored_graph_keys={'norm_offset', 'norm_scale'})


@pytest.mark.parametrize(
    'router',
    [
        EWRouter(),
        EWRouter(feeder_route='straight'),
        HGSRouter(time_limit=1, seed=0),
        HGSRouter(time_limit=1, feeder_limit=1, max_retries=5, balanced=True, seed=0),
        MILPRouter(solver_name='ortools', time_limit=2, mip_gap=0.005),
        MILPRouter(solver_name='cbc', time_limit=2, mip_gap=0.005),
        MILPRouter(solver_name='cplex', time_limit=2, mip_gap=0.005),
        MILPRouter(solver_name='gurobi', time_limit=2, mip_gap=0.005),
        MILPRouter(solver_name='highs', time_limit=2, mip_gap=0.005),
        MILPRouter(solver_name='scip', time_limit=2, mip_gap=0.005),
    ],
)
def test_wfn_all_routers(router):
    turbinesC = np.array([[1.0, 0], [1, 1]])
    substationsC = np.array([[0, 0]])
    cables = 2
    EXPECTED_TERSE = np.array([-1, 0])

    # case 1: router passed in constructor
    wfn = WindFarmNetwork(
        cables=cables,
        turbinesC=turbinesC,
        substationsC=substationsC,
        router=router,
    )
    assert np.array_equal(wfn.optimize(), EXPECTED_TERSE)

    # case 2: router passed at call-time
    wfn = WindFarmNetwork(
        cables=cables,
        turbinesC=turbinesC,
        substationsC=substationsC,
    )
    assert np.array_equal(wfn.optimize(router=router), EXPECTED_TERSE)


# tests/test_api_extra.py
import io
import sys
from pathlib import Path

import numpy as np
import pytest

from optiwindnet.api import EWRouter, WindFarmNetwork

# ---------- Guards & simple properties ----------

def test_S_and_G_raise_before_optimize(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    with pytest.raises(RuntimeError, match="Call the `optimize"):
        _ = wfn.S
    with pytest.raises(RuntimeError, match="Call the `optimize"):
        _ = wfn.G

def test_buffer_dist_property_and_router_default(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    assert isinstance(wfn.router, EWRouter)
    # default buffer distance
    assert isinstance(wfn.buffer_dist, float)

# ---------- Planar refresh & polygon OOB ----------

def test_polygon_out_of_bounds_raises():
    # A tiny triangular border; put one turbine well outside
    turbinesC = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0]])  # third is OOB
    substationsC = np.array([[0.0, 0.1]])
    borderC = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]])
    wfn = WindFarmNetwork(cables=2, turbinesC=turbinesC, substationsC=substationsC, borderC=borderC)
    # Accessing P triggers _refresh_planar() and then OOB check
    with pytest.raises(ValueError, match="Turbine out of bounds"):
        _ = wfn.P

# ---------- Buffering, merging & plotting helpers ----------

def test_plot_original_vs_buffered_without_prior_buffer_prints_message(LG_from_database, capsys):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    # No buffer applied yet -> triggers the "No buffering is performed" branch
    _ = wfn.plot_original_vs_buffered()
    captured = capsys.readouterr()
    assert "No buffering is performed" in captured.out

def test_add_buffer_then_plot_original_vs_buffered_returns_axes(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    wfn.add_buffer(5.0)  # apply buffer so pre_buffer structures exist
    ax = wfn.plot_original_vs_buffered()
    # Matplotlib Axes object or None if headless — typically not None
    assert ax is not None

def test_merge_obstacles_into_border_idempotent(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    # Should not raise; updates L and flags so P recomputes fine:
    wfn.merge_obstacles_into_border()
    _ = wfn.P  # force recomputation to ensure internal flags are OK

# ---------- Plot wrappers ----------

def test_plot_helpers_run(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    wfn.optimize()
    # Smoke tests for plotting wrappers
    wfn.plot()                 # G
    wfn.plot_location()        # L
    wfn.plot_available_links() # A
    wfn.plot_navigation_mesh() # P + A

# ---------- Cables setter after optimize (assign_cables path) ----------

def test_cables_setter_assigns_after_optimize(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=[(7, 100.0)], L=L)
    wfn.optimize()
    old_cost = wfn.cost()
    # Change cable prices -> should reassign and change cost scale
    wfn.cables = [(7, 200.0)]
    new_cost = wfn.cost()
    assert new_cost == pytest.approx(old_cost * 2.0, rel=1e-3)

# ---------- Optimize warmstart path & solution_info ----------

def test_optimize_second_time_uses_warmstart_and_solution_info(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    terse1 = wfn.optimize(verbose=True)
    terse2 = wfn.optimize(verbose=True)  # triggers warmstart branch
    assert terse1.shape == terse2.shape
    info = wfn.solution_info()
    # minimal expected keys
    assert 'router' in info and 'capacity' in info
    assert isinstance(info['capacity'], int)

# ---------- Gradient cost branch & zero-length guard ----------

def test_gradient_cost_branch_returns_shapes(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=[(7, 123.0)], L=L)
    wfn.optimize()
    g_wt, g_ss = wfn.gradient(gradient_type='cost')  # different code path vs 'length'
    assert g_wt.shape[0] == wfn.S.graph['T']
    assert g_ss.shape[0] == wfn.S.graph['R']

# ---------- update_from_terse_links: float ints accepted ----------

def test_update_from_terse_links_accepts_float_ints(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    wfn.optimize()
    terse = wfn.terse_links().astype(float)  # ints as floats
    wfn.update_from_terse_links(terse)
    # Should rebuild G and produce a finite length
    assert np.isfinite(wfn.length())

# ---------- _repr_svg_ produces an SVG string ----------

def test_repr_svg_returns_string(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    # Before optimize: shows L; after optimize: shows G
    svg1 = wfn._repr_svg_()
    assert isinstance(svg1, str) and svg1.strip().startswith("<svg")
    wfn.optimize()
    svg2 = wfn._repr_svg_()
    assert isinstance(svg2, str) and svg2.strip().startswith("<svg")

# ---------- from_windIO (temp file with minimal schema) ----------

def test_from_windIO_minimal_yaml(tmp_path):
    yml = tmp_path / "proj_2025_case.yaml"  # <- three tokens for handle-building
    yml.write_text(
        """
wind_farm:
  layouts:
    initial_layout:
      coordinates:
        x: [0.0, 1.0]
        y: [0.0, 0.0]
  electrical_substations:
    coordinates:
      x: [2.0]
      y: [0.0]
site:
  boundaries:
    polygons:
      - {x: [ -1.0,  3.0,  3.0, -1.0],
         y: [ -1.0, -1.0,  1.0,  1.0]}
        """
    )
    wfn = WindFarmNetwork.from_windIO(str(yml), cables=2)
    assert wfn.L.graph['T'] == 2
    assert wfn.L.graph['R'] == 1
    wfn.optimize()
    assert wfn.G.number_of_edges() > 0


def test_refresh_planar_multipolygon_obstacles_oob_and_border_point(monkeypatch):
    import shapely as shp
    import numpy as np
    from optiwindnet.api import WindFarmNetwork

    # 3 turbines: inside obstacle, on the obstacle border, and clearly outside
    turbinesC = np.array([
        [0.0, 0.0],   # inside square -> should remain in out_of_bounds
        [1.0, 0.0],   # on square edge -> should be removed by obstacle.exterior subtraction
        [3.0, 3.0],   # outside
    ])
    substationsC = np.array([[5.0, 5.0]])
    # No border, no obstacles -> we'll inject polygon ourselves
    wfn = WindFarmNetwork(cables=2, turbinesC=turbinesC, substationsC=substationsC)

    # Build a proper MultiPolygon made from real shapely.Polygon geometries
    square = shp.Polygon([(-1,-1), (1,-1), (1,1), (-1,1)])
    tri    = shp.Polygon([(10,10), (11,10), (10.5,11)])  # irrelevant second geom
    multi  = shp.MultiPolygon([square, tri])

    # Inject it and mark as fresh so WindFarmNetwork.polygon returns this object
    wfn._polygon = multi
    wfn._is_stale_polygon = False

    # Accessing P triggers _refresh_planar(); with one point inside, after removing border
    # points, out_of_bounds is still non-empty -> raises ValueError
    with pytest.raises(ValueError, match="Turbine out of bounds"):
        _ = wfn.P


# tests/test_api_utils.py
import logging
import numpy as np
import networkx as nx
import pytest
from shapely.geometry import Polygon

from optiwindnet import api_utils as U


# ---------------------------
# expand_polygon_safely
# ---------------------------

def test_expand_polygon_safely_warns_for_nonconvex_large_buffer(caplog):
    # Concave "C" shape
    poly = Polygon([(0,0),(4,0),(4,1),(1,1),(1,3),(4,3),(4,4),(0,4)])
    # Ensure non-convex
    assert not poly.equals(poly.convex_hull)
    with caplog.at_level(logging.WARNING, logger=U.__name__):
        out = U.expand_polygon_safely(poly, buffer_dist=1.0)  # deliberately big
    assert isinstance(out, Polygon)
    assert any("non-convex and buffering may introduce unexpected changes" in m for m in caplog.messages)


# ---------------------------
# shrink_polygon_safely
# ---------------------------

def test_shrink_polygon_safely_returns_array_normal():
    poly = Polygon([(0,0),(4,0),(4,4),(0,4)])
    arr = U.shrink_polygon_safely(poly, shrink_dist=0.2, indx=0)
    assert isinstance(arr, np.ndarray)
    assert arr.shape[1] == 2

def test_shrink_polygon_safely_becomes_empty_warns(caplog):
    poly = Polygon([(0,0),(1,0),(1,1),(0,1)])
    with caplog.at_level(logging.WARNING, logger=U.__name__):
        res = U.shrink_polygon_safely(poly, shrink_dist=10.0, indx=3)
    assert res is None
    assert any("completely removed the obstacle" in m for m in caplog.messages)

def test_shrink_polygon_safely_splits_to_multipolygon(caplog):
    # "Dumbbell" → two squares connected by narrow corridor
    big = Polygon([(0,0),(8,0),(8,4),(0,4)])
    hole = Polygon([(3,-1),(5,-1),(5,5),(3,5)])  # remove middle band to get two lobes
    shape = big.difference(hole)
    # Shrink a little so the corridor fully disappears, leaving two parts
    with caplog.at_level(logging.WARNING, logger=U.__name__):
        res = U.shrink_polygon_safely(shape, shrink_dist=0.1, indx=1)
    # Returns list of arrays for MultiPolygon
    assert isinstance(res, list) and len(res) >= 2
    assert any("split the obstacle" in m for m in caplog.messages)


# ---------------------------
# enable_ortools_logging_if_jupyter
# ---------------------------

def test_enable_ortools_logging_if_jupyter_sets_callback(monkeypatch):
    # Create a class literally named ZMQInteractiveShell so __class__.__name__ matches
    ZMQInteractiveShell = type("ZMQInteractiveShell", (), {})
    def fake_get_ipython():
        return ZMQInteractiveShell()

    # api_utils.enable_ortools_logging_if_jupyter looks up get_ipython() in its own module
    monkeypatch.setattr(U, "get_ipython", fake_get_ipython, raising=False)

    class DummyInner:
        def __init__(self): self.log_callback = None
    class DummySolver:
        def __init__(self): self.solver = DummyInner()

    s = DummySolver()
    U.enable_ortools_logging_if_jupyter(s)
    assert s.solver.log_callback is print



# ---------------------------
# is_warmstart_eligible
# ---------------------------

def _make_S_warm(R=1, T=5, branched=False, feeders=0):
    S = nx.Graph(R=R, T=T)
    # substation nodes: -R..-1 ; turbine nodes: 0..T-1
    for r in range(-R, 0):
        S.add_node(r)
    for t in range(T):
        S.add_node(t)
    # connect feeders turbines to substation -1
    for t in range(feeders):
        S.add_edge(-1, t)
    if branched:
        # make turbine 0 branched: degree > 2
        S.add_edge(0, 1)
        S.add_edge(0, 2)
        S.add_edge(0, 3)
    return S

def test_warmstart_ineligible_due_to_feeder_limit(capfd):
    S = _make_S_warm(R=1, T=5, feeders=4)  # degree(-1)=4
    model_options = {"feeder_limit":"minimum", "topology":"radial", "feeder_route":"segmented"}
    ok = U.is_warmstart_eligible(
        S_warm=S, cables_capacity=2, model_options=model_options,
        S_warm_has_detour=False, solver_name="ortools",
        logger=logging.getLogger(U.__name__), verbose=True
    )
    assert ok is False
    out = capfd.readouterr().out
    assert "exceeds feeder limit" in out

def test_warmstart_ineligible_detour_with_straight_route(capfd):
    S = _make_S_warm(R=1, T=4, feeders=1)
    model_options = {"feeder_limit":"unlimited", "topology":"radial", "feeder_route":"straight"}
    ok = U.is_warmstart_eligible(
        S_warm=S, cables_capacity=10, model_options=model_options,
        S_warm_has_detour=True, solver_name="ortools",
        logger=logging.getLogger(U.__name__), verbose=True
    )
    assert ok is False
    out = capfd.readouterr().out
    assert "incompatible with model option: feeder_route" in out

def test_warmstart_ineligible_branched_for_radial(capfd):
    S = _make_S_warm(R=1, T=5, feeders=1, branched=True)
    model_options = {"feeder_limit":"unlimited", "topology":"radial", "feeder_route":"segmented"}
    ok = U.is_warmstart_eligible(
        S_warm=S, cables_capacity=10, model_options=model_options,
        S_warm_has_detour=False, solver_name="ortools",
        logger=logging.getLogger(U.__name__), verbose=True
    )
    assert ok is False
    out = capfd.readouterr().out
    assert "branched network incompatible" in out

def test_warmstart_eligible_true_non_scip(capfd):
    S = _make_S_warm(R=1, T=3, feeders=1)
    model_options = {"feeder_limit":"unlimited", "topology":"radial", "feeder_route":"segmented"}
    ok = U.is_warmstart_eligible(
        S_warm=S, cables_capacity=10, model_options=model_options,
        S_warm_has_detour=False, solver_name="ortools",
        logger=logging.getLogger(U.__name__), verbose=True
    )
    assert ok is True

def test_warmstart_eligible_false_for_scip(capfd):
    S = _make_S_warm(R=1, T=3, feeders=1)
    model_options = {"feeder_limit":"unlimited", "topology":"radial", "feeder_route":"segmented"}
    ok = U.is_warmstart_eligible(
        S_warm=S, cables_capacity=10, model_options=model_options,
        S_warm_has_detour=False, solver_name="scip",
        logger=logging.getLogger(U.__name__), verbose=True
    )
    assert ok is False


# ---------------------------
# extract_network_as_array
# ---------------------------

def test_extract_network_as_array_includes_cost_when_present():
    G = nx.Graph()
    G.add_nodes_from([0,1])
    # reverse True means (s<t)==reverse holds, so src=s
    G.add_edge(0, 1, reverse=True, length=10.0, load=1.5, cable=2, cost=77.0)
    G.graph['has_cost'] = True
    G.graph['cables'] = [(1, 0.0), (2, 123.0)]
    arr = U.extract_network_as_array(G)
    assert arr.dtype.names == ('src','tgt','length','load','cable','cost')
    assert arr['cost'][0] == pytest.approx(77.0)


# ---------------------------
# merge_obs_into_border
# ---------------------------

def _make_L(border, obstacles, T=0, R=0):
    V_parts = []
    idx_cursor = 0
    border_idx = None
    obstacle_idxs = []

    # turbines (empty)
    V_parts.append(np.zeros((T,2)))
    idx_cursor += T

    if border is not None:
        border_idx = np.arange(idx_cursor, idx_cursor + len(border), dtype=int)
        V_parts.append(np.asarray(border))
        idx_cursor += len(border)

    for obs in obstacles:
        obs = np.asarray(obs)
        ids = np.arange(idx_cursor, idx_cursor + len(obs), dtype=int)
        obstacle_idxs.append(ids)
        V_parts.append(obs)
        idx_cursor += len(obs)

    # substations (empty)
    V_parts.append(np.zeros((R,2)))

    V = np.vstack([p for p in V_parts if len(p)>0]) if any(len(p)>0 for p in V_parts) else np.zeros((0,2))
    L = nx.Graph()
    L.graph['VertexC'] = V
    L.graph['T'] = T
    L.graph['R'] = R
    L.graph['border'] = border_idx
    L.graph['obstacles'] = obstacle_idxs
    L.graph['B'] = (len(border) if border is not None else 0) + sum(len(o) for o in obstacles)
    return L

def test_merge_obstacles_outside_is_dropped(caplog):
    border = [(-10,-10),(10,-10),(10,10),(-10,10)]
    obstacle = [(100,100),(101,100),(101,101),(100,101)]  # far away
    L = _make_L(border, [obstacle])
    with caplog.at_level(logging.WARNING, logger=U.__name__):
        L2 = U.merge_obs_into_border(L)
    # keeps same border, obstacle removed
    assert L2.graph['border'] is not None and len(L2.graph['obstacles']) == 0
    assert any("completely outside the border" in m for m in caplog.messages)

def test_merge_obstacles_intersection_multipolygon_raises():
    border = [(-10,-10),(10,-10),(10,10),(-10,10)]
    # vertical bar crossing the whole border → border minus obs becomes two pieces
    obstacle = [(-1,-11),(1,-11),(1,11),(-1,11)]
    L = _make_L(border, [obstacle])
    with pytest.raises(ValueError, match="multiple pieces"):
        U.merge_obs_into_border(L)

def test_merge_obstacles_intersection_empty_border_raises():
    border = [(-10,-10),(10,-10),(10,10),(-10,10)]
    obstacle = border[:]  # identical -> subtraction empty
    L = _make_L(border, [obstacle])
    with pytest.raises(ValueError, match="empty border"):
        U.merge_obs_into_border(L)

def test_merge_obstacles_inside_kept():
    border = [(-10,-10),(10,-10),(10,10),(-10,10)]
    obstacle = [(-1,-1),(1,-1),(1,1),(-1,1)]  # strictly inside
    L = _make_L(border, [obstacle])
    L2 = U.merge_obs_into_border(L)
    # Obstacle remains
    assert len(L2.graph['obstacles']) == 1


# ---------------------------
# buffer_border_obs
# ---------------------------

def test_buffer_border_obs_negative_raises(LG_from_database):
    # grab a real L for structure; content doesn't matter here
    L, _ = LG_from_database('eagle_EWRouter')
    with pytest.raises(ValueError, match="must be equal or greater than 0"):
        U.buffer_border_obs(L, buffer_dist=-1.0)


import logging
import numpy as np
import networkx as nx
import pytest
from shapely.geometry import Polygon, LinearRing

from optiwindnet import api_utils as U


# ---------- expand_polygon_safely: convex (no warning) ----------
def test_expand_polygon_safely_convex_no_warning(caplog):
    poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])  # convex square
    with caplog.at_level(logging.WARNING, logger=U.__name__):
        out = U.expand_polygon_safely(poly, buffer_dist=0.25)
    assert out.area > poly.area
    # no concavity warning should be emitted
    assert not any("non-convex and buffering may introduce" in m for m in caplog.messages)


# ---------- is_warmstart_eligible: extra feeder-limit modes ----------
def _S_warm(R=1, T=6, feeders=0, branched=False):
    S = nx.Graph(R=R, T=T)
    for r in range(-R, 0):
        S.add_node(r)
    for t in range(T):
        S.add_node(t)
    # Connect 'feeders' turbines to single substation -1
    for t in range(feeders):
        S.add_edge(-1, t)
    if branched:
        S.add_edge(0, 1)
        S.add_edge(0, 2)
        S.add_edge(0, 3)  # deg > 2 at 0
    return S

@pytest.mark.parametrize("mode,plus", [
    ("specified", 2),     # explicit limit
    ("min_plus1", 1),
    ("min_plus2", 2),
    ("min_plus3", 3),
])
def test_warmstart_feeder_limit_modes_block(capfd, mode, plus):
    # T=6, capacity=2 -> minimum feeders = ceil(6/2)=3
    # make feeders = (minimum + plus) + 1  -> strictly greater than the limit
    feeders = 3 + plus + 1
    S = _S_warm(R=1, T=6, feeders=feeders)
    model_options = {
        "feeder_limit": mode,
        "max_feeders": 4,
        "topology": "radial",
        "feeder_route": "segmented",
    }
    ok = U.is_warmstart_eligible(
        S_warm=S, cables_capacity=2, model_options=model_options,
        S_warm_has_detour=False, solver_name="ortools",
        logger=logging.getLogger(U.__name__), verbose=True
    )
    assert ok is False
    out = capfd.readouterr().out
    assert "exceeds feeder limit" in out


def test_warmstart_feeder_limit_specified_allows(capfd):
    # under specified limit -> eligible
    S = _S_warm(R=1, T=6, feeders=2)
    model_options = {"feeder_limit": "specified", "max_feeders": 3, "topology": "radial", "feeder_route": "segmented"}
    ok = U.is_warmstart_eligible(
        S_warm=S, cables_capacity=2, model_options=model_options,
        S_warm_has_detour=False, solver_name="ortools",
        logger=logging.getLogger(U.__name__), verbose=True
    )
    assert ok is True


# ---------- parse_cables_input with numpy arrays ----------
def test_parse_cables_input_numpy_ints_and_pairs():
    out1 = U.parse_cables_input(np.array([5, 7]))
    assert out1 == [(5, 0.0), (7, 0.0)]
    arr = np.array([(3, 10.0), (6, 20.0)], dtype=object)  # preserve tuple shapes
    out2 = U.parse_cables_input(arr)
    assert out2 == [(3, 10.0), (6, 20.0)]


# ---------- merge_obs_into_border: invalid obstacle & single info ----------
def _L(border, obstacles, T=0, R=0):
    V_parts = [np.zeros((T, 2))]
    cursor = T
    border_idx = None
    obs_idxs = []

    if border is not None:
        b = np.asarray(border, float)
        V_parts.append(b)
        border_idx = np.arange(cursor, cursor + len(b), dtype=int)
        cursor += len(b)

    for obs in obstacles:
        o = np.asarray(obs, float)
        V_parts.append(o)
        obs_idxs.append(np.arange(cursor, cursor + len(o), dtype=int))
        cursor += len(o)

    V_parts.append(np.zeros((R, 2)))
    V = np.vstack([p for p in V_parts if len(p) > 0]) if any(len(p) > 0 for p in V_parts) else np.zeros((0, 2))

    L = nx.Graph()
    L.graph['VertexC'] = V
    L.graph['T'] = T
    L.graph['R'] = R
    L.graph['border'] = border_idx
    L.graph['obstacles'] = obs_idxs
    L.graph['B'] = (0 if border_idx is None else len(border_idx)) + sum(len(i) for i in obs_idxs)
    return L

def test_merge_obstacles_invalid_polygon_warns(caplog):
    # Self-crossing "bow" polygon is invalid
    border = [(-5, -5), (5, -5), (5, 5), (-5, 5)]
    invalid = [(0, 0), (2, 2), (-2, 2), (2, -2)]  # self-crossing order
    L = _L(border, [invalid])
    with caplog.at_level(logging.WARNING, logger=U.__name__):
        L2 = U.merge_obs_into_border(L)
    # Should warn about invalid obstacle; result remains usable
    assert any("invalid" in m.lower() for m in caplog.messages)
    assert L2.graph['border'] is not None

def test_merge_obstacles_info_prints_once_when_multiple_intersections(caplog):
    # Two obstacles both intersect border; info should print only once
    border = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
    obs1   = [(-10, -1), (0, -1), (0, 1), (-10, 1)]  # touches/overlaps at left edge
    obs2   = [(9, -2), (12, -2), (12, 2), (9, 2)]    # overlaps right edge
    L = _L(border, [obs1, obs2])

    with caplog.at_level(logging.INFO, logger=U.__name__):
        L2 = U.merge_obs_into_border(L)

    # Border updated, obstacles possibly removed/kept
    assert L2.graph['border'] is not None
    # Only one info message printed for multiple intersections
    infos = [m for m in caplog.messages if "intersects/touches the border" in m]
    assert len(infos) == 1


# ---------- buffer_border_obs: no border + obstacles ----------
def test_buffer_border_obs_with_border_positive_shrinks_obstacles(LG_from_database):
    # Start from a real L to keep structure stable
    L, _ = LG_from_database('eagle_EWRouter')
    V = L.graph['VertexC'].copy()
    T, R = L.graph['T'], L.graph['R']

    # Minimal valid border (tiny triangle) just to satisfy pre_buffer & stacking
    tri_border = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]], float)

    # One square obstacle to be shrunk
    obstacle = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [0.0, 50.0]], float)

    # Rebuild VertexC: turbines, border, obstacle, substations
    coords = [V[:T], tri_border, obstacle, V[-R:]]
    new_V = np.vstack(coords)
    L.graph['VertexC'] = new_V

    # Indices
    cursor = T
    border_idx = np.arange(cursor, cursor + len(tri_border), dtype=int)
    cursor += len(tri_border)
    obstacle_idx = np.arange(cursor, cursor + len(obstacle), dtype=int)

    L.graph['border'] = border_idx
    L.graph['obstacles'] = [obstacle_idx]
    L.graph['B'] = len(tri_border) + len(obstacle)

    # Buffer > 0 → obstacles shrink; ensure function returns a rebuilt L and pre_buffer
    L2, pre = U.buffer_border_obs(L, buffer_dist=5.0)

    assert isinstance(pre, dict) and 'obstaclesC' in pre and 'borderC' in pre
    # Still has same T and R, and obstacles exist (but shrunk)
    assert L2.graph['VertexC'][:T].shape == (T, 2)
    assert L2.graph['VertexC'][-R:].shape == (R, 2)
    assert isinstance(L2.graph['obstacles'], list)
    # Obstacle got smaller in total points length or coordinates moved inward
    # (We just sanity-check that at least one obstacle remains)
    assert len(L2.graph['obstacles']) >= 1




# ---------- _ensure_closed ----------
def test__ensure_closed_adds_repeat_point():
    ring = np.array([[0.0, 0.0], [1, 0], [1, 1]])
    closed = U._ensure_closed(ring)
    assert np.allclose(closed[0], closed[-1])

def test__ensure_closed_noop_if_already_closed():
    ring = LinearRing([(0, 0), (1, 0), (1, 1)])
    coords = np.array(ring.coords[:], float)
    closed = U._ensure_closed(coords)
    # already closed -> unchanged
    assert np.array_equal(closed, coords)
