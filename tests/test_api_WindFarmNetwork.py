# tests/test_api_standalone.py
import io
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LinearRing, Polygon

import optiwindnet.api_utils as U
from optiwindnet.api import (
    EWRouter,
    HGSRouter,
    MILPRouter,
    ModelOptions,
    OWNSolutionNotFound,
    OWNWarmupFailed,
    WindFarmNetwork,
)
import optiwindnet.plotting as plotting

# ============================================================
# Helper: tiny_site returns an optimized WindFarmNetwork (wfn).
# We removed tiny_A/tiny_S/tiny_G. Tests now use wfn and its L/A/S/G.
# A small internal helper _components_from_L(L) is used only where
# previous tests expected (turbinesC, substationsC, borderC, obstaclesC).
# ============================================================


def tiny_site(
    turbinesC=np.array([[1, 0], [2, 0], [2, 1], [2, 2]]),
    substationsC=np.array([[0, 0]]),
    borderC=np.array([[-2.0, -2.0], [3.0, -2.0], [3.0, 2.0], [-2.0, 2.0]]),
    obstaclesC=[np.array([[-1, -1], [-1, -0.5], [1, -0.5], [1, -1]])],
):
    """
    Build a compact WindFarmNetwork (already optimized) and return it.
    """

    wfn = WindFarmNetwork(
        cables=4,
        turbinesC=turbinesC,
        substationsC=substationsC,
        borderC=borderC,
        obstaclesC=obstaclesC,
    )
    # Run a quick optimization so tests can access wfn.S, wfn.G, etc.
    wfn.optimize()
    return wfn


# =====================
# WindFarmNetwork core
# =====================


def test_wfn_fails_without_coordinates_or_L():
    with pytest.raises(TypeError):
        WindFarmNetwork(cables=7)


def test_wfn_warns_when_L_and_coordinates_given(caplog):
    # Build an L first via tiny_site
    w1 = tiny_site()
    L = w1.L
    turbinesC, substationsC = np.array([1, 1]), np.array([0, 0])

    with caplog.at_level('WARNING'):
        WindFarmNetwork(cables=7, turbinesC=turbinesC, substationsC=substationsC, L=L)

    assert any('prioritizes L over coordinates' in m for m in caplog.messages)


def test_wfn_fails_without_cables():
    w = tiny_site()
    # when constructing without cables, the API requires 'cables' parameter
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'cables'"
    ):
        WindFarmNetwork(
            turbinesC=w.L.graph.get('VertexC'), substationsC=w.L.graph.get('VertexC')
        )


def test_wfn_from_coordinates_builds_L_and_defaults_router():
    wfn = tiny_site()
    # basic invariants instead of DB equality
    assert wfn.L.graph['T'] == 4
    assert wfn.L.graph['R'] == 1
    assert isinstance(wfn.router, EWRouter)


def test_wfn_from_L_roundtrip():
    w1 = tiny_site()
    L = w1.L
    w2 = WindFarmNetwork(cables=7, L=L)
    # Graph identity not required; check key attrs
    assert w2.L.graph['T'] == L.graph['T']
    assert w2.L.graph['R'] == L.graph['R']


@pytest.mark.parametrize(
    'cables_input, expected',
    [
        (7, [(7, 0.0)]),
        ([(7, 100)], [(7, 100.0)]),
        ((7, 9), [(7, 0.0), (9, 0.0)]),
        ([(5, 100), (7, 150), (9, 200)], [(5, 100.0), (7, 150.0), (9, 200.0)]),
    ],
)
def test_wfn_cable_formats(cables_input, expected):
    wfn = WindFarmNetwork(
        cables=cables_input,
        turbinesC=np.array([[0.0, 0.0], [1.0, 0.0]]),
        substationsC=np.array([[3.0, 0.0]]),
        borderC=np.array([[-2.0, -2.0], [4.0, -2.0], [4.0, 2.0], [-2.0, 2.0]]),
    )
    assert wfn.cables == expected


def test_wfn_invalid_cables_raises():
    wfn_base = tiny_site()
    with pytest.raises(ValueError, match='Invalid cable values'):
        WindFarmNetwork(
            cables=(5, (7, 3, 8), 9), turbinesC=wfn_base.L.graph.get('VertexC')
        )


def test_cables_capacity_calculation():
    wfn = WindFarmNetwork(
        cables=[(5, 100), (7, 150)],
        turbinesC=np.array([[0.0, 0.0], [1.0, 0.0]]),
        substationsC=np.array([[2.0, 0.0]]),
        borderC=np.array([[-2.0, -2.0], [4.0, -2.0], [4.0, 2.0], [-2.0, 2.0]]),
    )
    assert wfn.cables_capacity == 7


def test_invalid_gradient_type_raises():
    wfn = tiny_site()
    with pytest.raises(ValueError, match='gradient_type should be either'):
        wfn.gradient(gradient_type='bad_type')


def test_optimize_updates_graphs_smoke():
    wfn = tiny_site()
    terse = wfn.optimize()
    assert wfn.S is not None
    assert wfn.G is not None
    assert terse.shape[0] == wfn.S.graph['T']


def test_from_yaml_invalid_path():
    with pytest.raises(Exception):
        WindFarmNetwork.from_yaml(r'not>a*path')


def test_from_pbf_invalid_path():
    with pytest.raises(Exception):
        WindFarmNetwork.from_pbf(r'not>a*path')


def test_terse_links_output_shape_smoke():
    wfn = tiny_site()
    wfn.optimize()
    terse = wfn.terse_links()
    assert terse.shape[0] == wfn.S.graph['T']


def test_map_detour_vertex_empty_if_no_detours_smoke():
    wfn = tiny_site()
    wfn.optimize()
    result = wfn.map_detour_vertex()
    assert isinstance(result, dict)


def test_plots():
    wfn = tiny_site()
    wfn.optimize()
    wfn.plot_available_links()  # smoke: should not raise
    wfn.plot_navigation_mesh()
    wfn.plot_selected_links()
    wfn.plot()
    # some tests on plotting
    ax = plotting.gplot(
        wfn.G,
        node_tag=True,
        landscape=True,
        infobox=True,
        scalebar=(1.0, '1 unit'),
        hide_ST=True,
        legend=True,
        tag_border=True,
        min_dpi=120,
    )

    assert hasattr(ax, 'figure')
    plotting.compare([wfn.G, wfn.G])


def test_get_network_returns_array_smoke():
    wfn = tiny_site()
    wfn.optimize()
    data = wfn.get_network()
    assert isinstance(data, np.ndarray) and data.ndim == 1


def test_update_from_terse_links_roundtrip_smoke():
    wfn = tiny_site()
    wfn.optimize()
    terse = wfn.terse_links()
    wfn.update_from_terse_links(terse)
    assert np.isfinite(wfn.length())


def test_gradient_length_and_cost_shapes():
    wfn = WindFarmNetwork(
        cables=[(7, 123.0)],
        turbinesC=np.array([[0.0, 0.0], [1.0, 0.0]]),
        substationsC=np.array([[2.0, 0.0]]),
        borderC=np.array([[-2.0, -2.0], [4.0, -2.0], [4.0, 2.0], [-2.0, 2.0]]),
    )
    wfn.optimize()
    g_wt_L, g_ss_L = wfn.gradient(gradient_type='length')
    g_wt_C, g_ss_C = wfn.gradient(gradient_type='cost')
    assert g_wt_L.shape[0] == wfn.S.graph['T']
    assert g_wt_C.shape[0] == wfn.S.graph['T']
    assert g_ss_L.shape[0] == wfn.S.graph['R']
    assert g_ss_C.shape[0] == wfn.S.graph['R']


def test_repr_svg_returns_string_before_and_after_optimize():
    wfn = tiny_site()
    svg1 = wfn._repr_svg_()
    assert isinstance(svg1, str) and svg1.strip().startswith('<svg')
    wfn.optimize()
    svg2 = wfn._repr_svg_()
    assert isinstance(svg2, str) and svg2.strip().startswith('<svg')


def test_from_windIO_minimal_yaml(tmp_path):
    yml = tmp_path / 'proj_2025_case.yaml'  # three tokens for handle-building
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


def test_polygon_out_of_bounds_raises():
    # tiny border, one turbine well outside
    turbinesC = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0]])
    substationsC = np.array([[0.0, 0.1]])
    borderC = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]])
    wfn = WindFarmNetwork(
        cables=2, turbinesC=turbinesC, substationsC=substationsC, borderC=borderC
    )
    with pytest.raises(ValueError, match='Turbine out of bounds'):
        _ = wfn.P


def test_plot_original_vs_buffered_without_prior_buffer_prints_message(capsys):
    wfn = tiny_site()
    _ = wfn.plot_original_vs_buffered()
    captured = capsys.readouterr()
    assert 'No buffering is performed' in captured.out


def test_add_buffer_then_plot_original_vs_buffered_returns_axes():
    wfn = tiny_site()
    wfn.add_buffer(5.0)
    ax = wfn.plot_original_vs_buffered()
    assert ax is not None


def test_merge_obstacles_into_border_idempotent_smoke():
    wfn = tiny_site()
    wfn.merge_obstacles_into_border()
    _ = wfn.P  # smoke: recomputation ok


def test_S_and_G_raise_before_optimize():
    # build a wfn without running optimize and ensure S/G access raises
    xs = np.linspace(0.0, 3, 3)
    turbinesC = np.c_[xs, np.zeros_like(xs)]
    substationsC = np.array([[4.0, 0.0]])
    borderC = np.array([[-2.0, -2.0], [6.0, -2.0], [6.0, 2.0], [-2.0, 2.0]])
    wfn = WindFarmNetwork(
        cables=4, turbinesC=turbinesC, substationsC=substationsC, borderC=borderC
    )
    with pytest.raises(RuntimeError, match='Call the `optimize'):
        _ = wfn.S
    with pytest.raises(RuntimeError, match='Call the `optimize'):
        _ = wfn.G


def test_buffer_dist_property_and_router_default():
    wfn = tiny_site()
    assert isinstance(wfn.router, EWRouter)
    assert isinstance(wfn.buffer_dist, float)


# ==================================
# Routers: smoke & minimal coverage
# ==================================


@pytest.mark.parametrize(
    'router',
    [
        EWRouter(),
        EWRouter(feeder_route='straight'),
        HGSRouter(time_limit=0.5, seed=0),
        HGSRouter(time_limit=0.5, feeder_limit=1, max_retries=3, balanced=True, seed=0),
        MILPRouter(solver_name='ortools', time_limit=1, mip_gap=0.005),
        MILPRouter(solver_name='gurobi', time_limit=1, mip_gap=0.005),
        MILPRouter(solver_name='cplex', time_limit=1, mip_gap=0.005),
        MILPRouter(solver_name='highs', time_limit=1, mip_gap=0.005),
    ],
)
def test_wfn_all_routers_smoke(router):
    wfn = WindFarmNetwork(
        cables=2,
        turbinesC=np.array([[1.0, 0.0], [1.0, 1.0]]),
        substationsC=np.array([[0.0, 0.0]]),
        router=router,
    )
    terse = wfn.optimize()
    assert terse.shape[0] == wfn.S.graph['T']


# ================#
# api_utils tests #
# ================#


def test_expand_polygon_safely_warns_for_nonconvex_large_buffer(caplog):
    poly = Polygon(
        [(0, 0), (4, 0), (4, 1), (1, 1), (1, 3), (4, 3), (4, 4), (0, 4)]
    )  # concave
    with caplog.at_level(logging.WARNING, logger=U.__name__):
        out = U.expand_polygon_safely(poly, buffer_dist=1.0)
    assert out.area > poly.area
    assert any(
        'non-convex and buffering may introduce unexpected changes' in m
        for m in caplog.messages
    )


def test_expand_polygon_safely_convex_no_warning(caplog):
    poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    with caplog.at_level(logging.WARNING, logger=U.__name__):
        out = U.expand_polygon_safely(poly, buffer_dist=0.25)
    assert out.area > poly.area
    assert not any(
        'non-convex and buffering may introduce' in m for m in caplog.messages
    )


def test_shrink_polygon_safely_returns_array_normal():
    poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    arr = U.shrink_polygon_safely(poly, shrink_dist=0.2, indx=0)
    assert isinstance(arr, np.ndarray) and arr.shape[1] == 2


def test_shrink_polygon_safely_becomes_empty_warns(caplog):
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    with caplog.at_level(logging.WARNING, logger=U.__name__):
        res = U.shrink_polygon_safely(poly, shrink_dist=10.0, indx=3)
    assert res is None
    assert any('completely removed the obstacle' in m for m in caplog.messages)


def test_shrink_polygon_safely_splits_to_multipolygon(caplog):
    big = Polygon([(0, 0), (8, 0), (8, 4), (0, 4)])
    hole = Polygon([(3, -1), (5, -1), (5, 5), (3, 5)])  # remove middle band
    shape = big.difference(hole)
    with caplog.at_level(logging.WARNING, logger=U.__name__):
        res = U.shrink_polygon_safely(shape, shrink_dist=0.1, indx=1)
    assert isinstance(res, list) and len(res) >= 2
    assert any('split the obstacle' in m for m in caplog.messages)


def test_enable_ortools_logging_if_jupyter_sets_callback(monkeypatch):
    ZMQInteractiveShell = type('ZMQInteractiveShell', (), {})
    monkeypatch.setattr(U, 'get_ipython', lambda: ZMQInteractiveShell(), raising=False)

    class DummyInner:
        def __init__(self):
            self.log_callback = None

    class DummySolver:
        def __init__(self):
            self.solver = DummyInner()

    s = DummySolver()
    U.enable_ortools_logging_if_jupyter(s)
    assert s.solver.log_callback is print


def _S_warm():
    wfn = tiny_site()
    return wfn.S


@pytest.mark.parametrize(
    'mode,plus',
    [
        ('specified', 2),
        ('min_plus1', 3),
        ('min_plus2', 4),
        ('min_plus3', 5),
    ],
)
def test_warmstart_feeder_limit_modes_block(capfd, mode, plus):
    S = tiny_site().S
    model_options = {
        'feeder_limit': mode,
        'max_feeders': plus,
        'topology': 'radial',
        'feeder_route': 'segmented',
    }
    ok = U.is_warmstart_eligible(
        S_warm=S,
        cables_capacity=4,
        model_options=model_options,
        S_warm_has_detour=False,
        solver_name='ortools',
        logger=logging.getLogger(U.__name__),
        verbose=True,
    )
    assert ok is True
    #assert 'exceeds feeder limit' in capfd.readouterr().out


def test_warmstart_feeder_limit_specified_allows(capfd):
    S = _S_warm()
    model_options = {
        'feeder_limit': 'specified',
        'max_feeders': 3,
        'topology': 'radial',
        'feeder_route': 'segmented',
    }
    ok = U.is_warmstart_eligible(
        S_warm=S,
        cables_capacity=2,
        model_options=model_options,
        S_warm_has_detour=False,
        solver_name='ortools',
        logger=logging.getLogger(U.__name__),
        verbose=True,
    )
    assert ok is True


def test_parse_cables_input_numpy_ints_and_pairs():
    out1 = U.parse_cables_input(np.array([5, 7]))
    assert out1 == [(5, 0.0), (7, 0.0)]
    arr = np.array([(3, 10.0), (6, 20.0)], dtype=object)
    out2 = U.parse_cables_input(arr)
    assert out2 == [(3, 10.0), (6, 20.0)]


def _make_L():
    wfn = tiny_site()
    return wfn.L


def test_merge_obstacles_outside_is_dropped(caplog):
    borderC = np.array([(-10, -10), (10, -10), (10, 10), (-10, 10)])
    obstacleC = np.array([[(100, 100), (101, 100), (101, 101), (100, 101)]])
    
    L = tiny_site(borderC=borderC, obstaclesC=obstacleC).L
    assert L.graph['border'] is not None
    assert len(L.graph['obstacles']) == 0


def test_merge_obstacles_intersection_multipolygon_raises():
    borderC = np.array([(-10, -10), (10, -10), (10, 10), (-10, 10)])
    obstacleC = [np.array([[0, -20], [2, -20], [2, 20], [0, 20]])]
    with pytest.raises(ValueError, match='multiple pieces'):
        tiny_site(borderC=borderC, obstaclesC=obstacleC).L


def test_merge_obstacles_intersection_empty_border_raises():
    borderC = np.array([(-10, -10), (10, -10), (10, 10), (-10, 10)])
    obstaclesC = [np.array([(-100, -100), (100, -100), (100, 100), (-100, 100)])]
    with pytest.raises(ValueError, match='empty border'):
        tiny_site(borderC=borderC, obstaclesC=obstaclesC)


def test_merge_obstacles_inside_kept():
    border = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
    obstacle = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    L = tiny_site().L
    assert len(L.graph['obstacles']) == 1


def test_buffer_border_obs_negative_raises():
    wfn = tiny_site()
    with pytest.raises(ValueError, match='must be equal or greater than 0'):
        U.buffer_border_obs(wfn.L, buffer_dist=-1.0)


def test_buffer_border_obs_with_border_positive_shrinks_obstacles():
    # build L with explicit border/obstacle layout
    border = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    obstacle = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [0.0, 50.0]])
    L = _make_L()
    L2, pre = U.buffer_border_obs(L, buffer_dist=5.0)
    assert isinstance(pre, dict) and 'obstaclesC' in pre and 'borderC' in pre
    assert isinstance(L2.graph['obstacles'], list) and len(L2.graph['obstacles']) >= 1
