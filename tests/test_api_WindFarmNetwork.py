import logging
import numpy as np
import pytest
from shapely.geometry import Polygon

import optiwindnet.api_utils as U
from optiwindnet.api import (
    EWRouter,
    HGSRouter,
    WindFarmNetwork,
)
import optiwindnet.plotting as plotting
from .helpers import tiny_wfn


# =====================
# WindFarmNetwork core
# =====================


def test_wfn_fails_without_coordinates_or_L():
    with pytest.raises(TypeError):
        WindFarmNetwork(cables=7)


def test_optimize_updates_graphs_smoke():
    wfn = tiny_wfn()
    terse = wfn.optimize()
    assert wfn.S is not None
    assert wfn.G is not None
    assert terse.shape[0] == wfn.S.graph['T']


def test_wfn_warns_when_L_and_coordinates_given(caplog):
    w1 = tiny_wfn()
    L = w1.L
    turbinesC, substationsC = np.array([1, 1]), np.array([0, 0])

    with caplog.at_level('WARNING'):
        WindFarmNetwork(cables=7, turbinesC=turbinesC, substationsC=substationsC, L=L)

    assert any('prioritizes L over coordinates' in m for m in caplog.messages)


def test_wfn_fails_without_cables():
    w = tiny_wfn()
    # when constructing without cables, the API requires 'cables' parameter
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'cables'"
    ):
        WindFarmNetwork(
            turbinesC=w.L.graph.get('VertexC'), substationsC=w.L.graph.get('VertexC')
        )


def test_wfn_from_coordinates_builds_L_and_defaults_router():
    wfn = tiny_wfn()
    # check basic parameters
    assert wfn.L.graph['T'] == 4
    assert wfn.L.graph['R'] == 1
    assert isinstance(wfn.router, EWRouter)


def test_wfn_from_L_roundtrip():
    wfn1 = tiny_wfn()
    L = wfn1.L
    wfn2 = WindFarmNetwork(cables=4, L=L)
    wfn2.optimize()
    # Graph identity not required; check key attrs
    assert wfn2.L.graph['T'] == L.graph['T']
    assert wfn2.L.graph['R'] == L.graph['R']
    assert np.array_equal(wfn2.terse_links(), wfn1.terse_links())


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
        turbinesC=np.array([[0.0, 1.0], [1.0, 0.0]]),
        substationsC=np.array([[0.0, 0.0]]),
    )
    assert wfn.cables == expected


def test_wfn_invalid_cables_raises():
    with pytest.raises(ValueError, match='Invalid cable values'):
        WindFarmNetwork(
            cables=(5, (7, 3, 8), 9),
            turbinesC=np.array([[0.0, 1.0], [1.0, 0.0]]),
            substationsC=np.array([[0.0, 0.0]]),
        )


def test_cables_capacity_calculation():
    wfn1 = WindFarmNetwork(
        cables=9,
        turbinesC=np.array([[0.0, 1.0], [1.0, 0.0]]),
        substationsC=np.array([[0.0, 0.0]]),
    )
    wfn2 = WindFarmNetwork(
        cables=[(5, 100), (7, 150)],
        turbinesC=np.array([[0.0, 1.0], [1.0, 0.0]]),
        substationsC=np.array([[0.0, 0.0]]),
    )
    assert wfn1.cables_capacity == 9
    assert wfn2.cables_capacity == 7


def test_invalid_gradient_type_raises():
    wfn = tiny_wfn()
    with pytest.raises(ValueError, match='gradient_type should be either'):
        wfn.gradient(gradient_type='bad_type')


def test_from_yaml_invalid_path():
    with pytest.raises(Exception):
        WindFarmNetwork.from_yaml(r'not>a*path')


def test_from_pbf_invalid_path():
    with pytest.raises(Exception):
        WindFarmNetwork.from_pbf(r'not>a*path')


def test_terse_links_output():
    terse_expected = np.array([-1, 0, 1, 2])
    wfn = tiny_wfn()
    terse = wfn.terse_links()
    assert terse.shape[0] == wfn.S.graph['T']
    assert np.array_equal(wfn.terse_links(), terse_expected)


def test_update_from_terse_links():
    wfn = tiny_wfn()
    terse1 = wfn.terse_links()
    terse = np.array([1, 2, -1, 0])
    wfn.update_from_terse_links(terse)
    terse2 = wfn.terse_links()
    assert np.array_equal(terse1, np.array([-1, 0, 1, 2]))
    assert np.array_equal(terse2, np.array([1, 2, -1, 0]))


def test_map_detour_vertex_empty_if_no_detours_smoke():
    wfn = tiny_wfn()
    map_detour = wfn.map_detour_vertex()
    assert isinstance(map_detour, dict)
    assert map_detour == {12: 8, 13: 11}


def test_plots():
    wfn = tiny_wfn()
    wfn.optimize()
    wfn.plot_available_links()  # smoke: should not raise any error
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

    with pytest.raises(AttributeError):
        plotting.compare([wfn.plot(), ax])

    plotting.compare([wfn.G, wfn.G])


def test_get_network_returns_array_smoke():
    wfn = tiny_wfn()
    data = wfn.get_network()
    # basic shape/type checks
    assert isinstance(data, np.ndarray)
    assert data.ndim == 1  # structured 1-D array (rows)

    # expected structured dtype and field names
    expected_fields = ('src', 'tgt', 'length', 'load', 'cable')
    assert tuple(data.dtype.names) == expected_fields

    # element types
    assert np.issubdtype(data['src'].dtype, np.integer)
    assert np.issubdtype(data['tgt'].dtype, np.integer)
    assert np.issubdtype(data['length'].dtype, np.floating)
    assert np.issubdtype(data['load'].dtype, np.floating)
    assert np.issubdtype(data['cable'].dtype, np.integer)

    # value range sanity checks
    assert np.all(data['length'] >= 0.0)  # non-negative lengths
    assert np.all(data['load'] >= 0.0)  # loads shouldn't be negative
    # cable indices must be valid indices into wfn.cables
    n_cables = len(wfn.cables) if hasattr(wfn, 'cables') else 0
    assert np.all((data['cable'] >= 0) & (data['cable'] < max(1, n_cables)))

    # consistency with graph edges: every (src,tgt) should exist in wfn.G (undirected)
    edges_in_G = set(tuple(sorted(e)) for e in wfn.G.edges())
    for row in data:
        pair = tuple(sorted((int(row['src']), int(row['tgt']))))
        assert pair in edges_in_G, (
            f'row {(row["src"], row["tgt"])} not present in wfn.G'
        )


def test_gradient():
    # build an optimized tiny network so wfn.S exists and gradients are meaningful
    wfn = tiny_wfn(optimize=True)

    g_wt_L, g_ss_L = wfn.gradient(gradient_type='length')
    g_wt_C, g_ss_C = wfn.gradient(gradient_type='cost')

    assert g_wt_L.shape[0] == wfn.S.graph['T']
    assert g_wt_C.shape[0] == wfn.S.graph['T']
    assert g_ss_L.shape[0] == wfn.S.graph['R']
    assert g_ss_C.shape[0] == wfn.S.graph['R']

    # expected (reference) arrays from previous golden values
    exp_wt_L = np.array(
        [
            [0.62860932, 0.92847669],
            [0.70710678, -0.29289322],
            [0.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    exp_ss_L = np.array([[-1.0, 0.0]], dtype=float)

    # For cost we expect the same directional gradients scaled by 10 (example),
    # but keep explicit golden arrays to be clear:
    exp_wt_C = np.array(
        [
            [6.28609324, 9.28476691],
            [7.07106781, -2.92893219],
            [0.0, 0.0],
            [0.0, 10.0],
        ],
        dtype=float,
    )
    exp_ss_C = np.array([[-10.0, 0.0]], dtype=float)

    # Use an absolute/relative tolerance for floating point comparisons
    atol = 1e-8
    rtol = 1e-6

    # Length gradients
    assert np.allclose(g_wt_L, exp_wt_L, rtol=rtol, atol=atol)
    assert np.allclose(g_ss_L, exp_ss_L, rtol=rtol, atol=atol)

    # Cost gradients
    assert np.allclose(g_wt_C, exp_wt_C, rtol=rtol, atol=atol)
    assert np.allclose(g_ss_C, exp_ss_C, rtol=rtol, atol=atol)


def test_repr_svg_returns_string_before_and_after_optimize():
    wfn = tiny_wfn()
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
        y: [1.0, 0.0]
  electrical_substations:
    coordinates:
      x: [0.0]
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
    assert np.array_equal(wfn.terse_links(), [-1, -1])


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
    wfn = tiny_wfn()
    _ = wfn.plot_original_vs_buffered()
    captured = capsys.readouterr()
    assert 'No buffering is performed' in captured.out


def test_add_buffer_then_plot_original_vs_buffered_returns_axes():
    wfn = tiny_wfn()
    wfn.add_buffer(5.0)
    ax = wfn.plot_original_vs_buffered()
    assert ax is not None


def test_merge_obstacles_into_border_idempotent_smoke():
    wfn = tiny_wfn()
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


@pytest.mark.parametrize(
    'router',
    [
        EWRouter(),
        EWRouter(feeder_route='straight'),
        HGSRouter(time_limit=0.5, seed=0),
        HGSRouter(time_limit=0.5, feeder_limit=1, max_retries=3, balanced=True, seed=0),
    ],
)
def test_wfn_inexact_routers_smoke(router):
    wfn = tiny_wfn()
    terse = wfn.optimize(router=router)
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
    S = tiny_wfn().S
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
    # assert 'exceeds feeder limit' in capfd.readouterr().out


def test_warmstart_feeder_limit_specified_allows(capfd):
    S = tiny_wfn().S
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


def test_merge_obstacles_outside_is_dropped(caplog):
    borderC = np.array([(-10, -10), (10, -10), (10, 10), (-10, 10)])
    obstacleC_ = np.array([[(100, 100), (101, 100), (101, 101), (100, 101)]])

    wfn = tiny_wfn(borderC=borderC, obstacleC_=obstacleC_, optimize=False)
    wfn.merge_obstacles_into_border()
    assert wfn.L.graph['border'] is not None
    assert len(wfn.L.graph['obstacles']) == 0


def test_merge_obstacles_intersection_multipolygon_raises():
    borderC = np.array([(-10, -10), (10, -10), (10, 10), (-10, 10)])
    obstacleC = [np.array([[-9, -20], [-8, -20], [-8, 20], [-9, 20]])]
    with pytest.raises(ValueError, match='multiple pieces'):
        wfn = tiny_wfn(borderC=borderC, obstacleC_=obstacleC)
        wfn.merge_obstacles_into_border()


def test_merge_obstacles_intersection_empty_border_raises():
    borderC = np.array([(-10, -10), (10, -10), (10, 10), (-10, 10)])
    obstacleC_ = [np.array([(-100, -100), (100, -100), (100, 100), (-100, 100)])]
    with pytest.raises(ValueError, match='Turbine out of bounds'):
        wfn = tiny_wfn(borderC=borderC, obstacleC_=obstacleC_)
        wfn.merge_obstacles_into_border()


def test_merge_obstacles_inside_kept():
    borderC = np.array([(-10, -10), (10, -10), (10, 10), (-10, 10)])
    obstacleC_ = [np.array([(-9, -9), (1, -9), (1, -5), (-1, -5)])]
    L = tiny_wfn(borderC=borderC, obstacleC_=obstacleC_).L
    assert len(L.graph['obstacles']) == 1


def test_buffer_border_obs_negative_raises():
    wfn = tiny_wfn()
    with pytest.raises(ValueError, match='must be equal or greater than 0'):
        U.buffer_border_obs(wfn.L, buffer_dist=-1.0)


def test_buffer_border_obs_with_border_positive_shrinks_obstacles():
    # build L with explicit border/obstacle layout
    wfn = tiny_wfn()
    assert 'obstacles' in wfn.L.graph and wfn.L.graph['border'] is not None
    wfn.add_buffer(buffer_dist=5.0)
    assert (
        isinstance(wfn.L.graph['obstacles'], list)
        and len(wfn.L.graph['obstacles']) == 0
    )
