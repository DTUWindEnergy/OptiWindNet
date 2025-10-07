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

# ============================================================
# Helpers: tiny synthetic site, plus a few tiny graph builders
# ============================================================


def tiny_site(T=4, R=1):
    """Return (turbinesC, substationsC, borderC, obstaclesC)."""
    # simple line of turbines, one substation to the right
    xs = np.linspace(0.0, T - 1, T)
    turbinesC = np.c_[xs, np.zeros_like(xs)]
    substationsC = np.array([[T + 1.0, 0.0]])
    borderC = np.array([[-2.0, -2.0], [T + 3.0, -2.0], [T + 3.0, 2.0], [-2.0, 2.0]])
    obstaclesC = []  # none by default
    return turbinesC, substationsC, borderC, obstaclesC


def tiny_A(R=1, T=2):
    A = nx.Graph(R=R, T=T)
    for i in range(T + R):
        A.add_node(i)
    if T >= 2:
        A.add_edge(0, 1)
    return A


def tiny_S(T=2, R=1):
    S = nx.Graph(R=R, T=T)
    if T >= 2:
        S.add_edge(0, 1)
    return S


def tiny_G(T=2, R=1):
    G = nx.Graph(R=R, T=T, VertexC=np.zeros((T + R, 2)))
    if T >= 2:
        G.add_edge(0, 1, length=1.0, cost=1.0, cable=0)
    G.graph['cables'] = [(1, 1.0)]
    G.graph['method_options'] = {}
    return G


# =====================
# WindFarmNetwork core
# =====================


def test_wfn_fails_without_coordinates_or_L():
    with pytest.raises(TypeError):
        WindFarmNetwork(cables=7)


def test_wfn_warns_when_L_and_coordinates_given(caplog):
    tC, sC, bC, _ = tiny_site(T=3, R=1)
    # Build an L first
    w1 = WindFarmNetwork(cables=7, turbinesC=tC, substationsC=sC, borderC=bC)
    L = w1.L
    with caplog.at_level('WARNING'):
        WindFarmNetwork(cables=7, turbinesC=tC, substationsC=sC, L=L)
    assert any('prioritizes L over coordinates' in m for m in caplog.messages)


def test_wfn_fails_without_cables():
    tC, sC, bC, _ = tiny_site(T=3)
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'cables'"
    ):
        WindFarmNetwork(turbinesC=tC, substationsC=sC, borderC=bC)


def test_wfn_from_coordinates_builds_L_and_defaults_router():
    tC, sC, bC, _ = tiny_site(T=5)
    wfn = WindFarmNetwork(
        cables=7, turbinesC=tC, substationsC=sC, borderC=bC, handle='h', name='n'
    )
    # basic invariants instead of DB equality
    assert wfn.L.graph['T'] == len(tC)
    assert wfn.L.graph['R'] == len(sC)
    assert isinstance(wfn.router, EWRouter)


def test_wfn_from_L_roundtrip():
    tC, sC, bC, _ = tiny_site(T=4)
    w1 = WindFarmNetwork(cables=7, turbinesC=tC, substationsC=sC, borderC=bC)
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
    tC, sC, bC, _ = tiny_site(T=3)
    wfn = WindFarmNetwork(
        cables=cables_input, turbinesC=tC, substationsC=sC, borderC=bC
    )
    assert wfn.cables == expected


def test_wfn_invalid_cables_raises():
    tC, sC, bC, _ = tiny_site(T=3)
    with pytest.raises(ValueError, match='Invalid cable values'):
        WindFarmNetwork(
            cables=(5, (7, 3, 8), 9), turbinesC=tC, substationsC=sC, borderC=bC
        )


def test_cables_capacity_calculation():
    tC, sC, bC, _ = tiny_site(T=3)
    wfn = WindFarmNetwork(
        cables=[(5, 100), (7, 150)], turbinesC=tC, substationsC=sC, borderC=bC
    )
    assert wfn.cables_capacity == 7


def test_invalid_gradient_type_raises():
    tC, sC, bC, _ = tiny_site(T=4)
    wfn = WindFarmNetwork(cables=7, turbinesC=tC, substationsC=sC, borderC=bC)
    with pytest.raises(ValueError, match='gradient_type should be either'):
        wfn.gradient(gradient_type='bad_type')


def test_optimize_updates_graphs_smoke():
    tC, sC, bC, _ = tiny_site(T=6)
    wfn = WindFarmNetwork(cables=7, turbinesC=tC, substationsC=sC, borderC=bC)
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
    tC, sC, bC, _ = tiny_site(T=5)
    wfn = WindFarmNetwork(cables=5, turbinesC=tC, substationsC=sC, borderC=bC)
    wfn.optimize()
    terse = wfn.terse_links()
    assert terse.shape[0] == wfn.S.graph['T']


def test_map_detour_vertex_empty_if_no_detours_smoke():
    tC, sC, bC, _ = tiny_site(T=5)
    wfn = WindFarmNetwork(cables=5, turbinesC=tC, substationsC=sC, borderC=bC)
    wfn.optimize()
    result = wfn.map_detour_vertex()
    assert isinstance(result, dict)


def test_plot_selected_links_runs_smoke():
    tC, sC, bC, _ = tiny_site(T=5)
    wfn = WindFarmNetwork(cables=5, turbinesC=tC, substationsC=sC, borderC=bC)
    wfn.optimize()
    wfn.plot_selected_links()  # smoke: should not raise


def test_get_network_returns_array_smoke():
    tC, sC, bC, _ = tiny_site(T=4)
    wfn = WindFarmNetwork(cables=4, turbinesC=tC, substationsC=sC, borderC=bC)
    wfn.optimize()
    data = wfn.get_network()
    assert isinstance(data, np.ndarray) and data.ndim == 1


def test_update_from_terse_links_roundtrip_smoke():
    tC, sC, bC, _ = tiny_site(T=6)
    wfn = WindFarmNetwork(cables=6, turbinesC=tC, substationsC=sC, borderC=bC)
    wfn.optimize()
    terse = wfn.terse_links()
    wfn.update_from_terse_links(terse)
    assert np.isfinite(wfn.length())


def test_gradient_length_and_cost_shapes():
    tC, sC, bC, _ = tiny_site(T=6)
    wfn = WindFarmNetwork(
        cables=[(7, 123.0)], turbinesC=tC, substationsC=sC, borderC=bC
    )
    wfn.optimize()
    g_wt_L, g_ss_L = wfn.gradient(gradient_type='length')
    g_wt_C, g_ss_C = wfn.gradient(gradient_type='cost')
    assert g_wt_L.shape[0] == wfn.S.graph['T']
    assert g_wt_C.shape[0] == wfn.S.graph['T']
    assert g_ss_L.shape[0] == wfn.S.graph['R']
    assert g_ss_C.shape[0] == wfn.S.graph['R']


def test_repr_svg_returns_string_before_and_after_optimize():
    tC, sC, bC, _ = tiny_site(T=4)
    wfn = WindFarmNetwork(cables=4, turbinesC=tC, substationsC=sC, borderC=bC)
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
    tC, sC, bC, _ = tiny_site(T=5)
    wfn = WindFarmNetwork(cables=7, turbinesC=tC, substationsC=sC, borderC=bC)
    _ = wfn.plot_original_vs_buffered()
    captured = capsys.readouterr()
    assert 'No buffering is performed' in captured.out


def test_add_buffer_then_plot_original_vs_buffered_returns_axes():
    tC, sC, bC, _ = tiny_site(T=5)
    wfn = WindFarmNetwork(cables=7, turbinesC=tC, substationsC=sC, borderC=bC)
    wfn.add_buffer(5.0)
    ax = wfn.plot_original_vs_buffered()
    assert ax is not None


def test_merge_obstacles_into_border_idempotent_smoke():
    tC, sC, bC, _ = tiny_site(T=5)
    wfn = WindFarmNetwork(cables=7, turbinesC=tC, substationsC=sC, borderC=bC)
    wfn.merge_obstacles_into_border()
    _ = wfn.P  # smoke: recomputation ok


def test_S_and_G_raise_before_optimize_guards():
    tC, sC, bC, _ = tiny_site(T=4)
    wfn = WindFarmNetwork(cables=4, turbinesC=tC, substationsC=sC, borderC=bC)
    with pytest.raises(RuntimeError, match='Call the `optimize'):
        _ = wfn.S
    with pytest.raises(RuntimeError, match='Call the `optimize'):
        _ = wfn.G


def test_buffer_dist_property_and_router_default():
    tC, sC, bC, _ = tiny_site(T=4)
    wfn = WindFarmNetwork(cables=4, turbinesC=tC, substationsC=sC, borderC=bC)
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
    turbinesC = np.array([[1.0, 0.0], [1.0, 1.0]])
    substationsC = np.array([[0.0, 0.0]])
    wfn = WindFarmNetwork(
        cables=2, turbinesC=turbinesC, substationsC=substationsC, router=router
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


def _S_warm(R=1, T=6, feeders=0, branched=False):
    S = nx.Graph(R=R, T=T)
    for r in range(-R, 0):
        S.add_node(r)
    for t in range(T):
        S.add_node(t)
    for t in range(feeders):
        S.add_edge(-1, t)
    if branched:
        S.add_edge(0, 1)
        S.add_edge(0, 2)
        S.add_edge(0, 3)
    return S


@pytest.mark.parametrize(
    'mode,plus',
    [
        ('specified', 2),
        ('min_plus1', 1),
        ('min_plus2', 2),
        ('min_plus3', 3),
    ],
)
def test_warmstart_feeder_limit_modes_block(capfd, mode, plus):
    feeders = 3 + plus + 1  # exceeds limit
    S = _S_warm(R=1, T=6, feeders=feeders)
    model_options = {
        'feeder_limit': mode,
        'max_feeders': 4,
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
    assert ok is False
    assert 'exceeds feeder limit' in capfd.readouterr().out


def test_warmstart_feeder_limit_specified_allows(capfd):
    S = _S_warm(R=1, T=6, feeders=2)
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


def _make_L(border, obstacles, T=0, R=0):
    V_parts, cursor = [np.zeros((T, 2))], T
    border_idx, obs_idxs = None, []
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
    V = (
        np.vstack([p for p in V_parts if len(p) > 0])
        if any(len(p) > 0 for p in V_parts)
        else np.zeros((0, 2))
    )
    L = nx.Graph()
    L.graph.update(
        VertexC=V,
        T=T,
        R=R,
        border=border_idx,
        obstacles=obs_idxs,
        B=(0 if border_idx is None else len(border_idx))
        + sum(len(i) for i in obs_idxs),
    )
    return L


def test_merge_obstacles_outside_is_dropped(caplog):
    border = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
    obstacle = [(100, 100), (101, 100), (101, 101), (100, 101)]
    L = _make_L(border, [obstacle])
    with caplog.at_level(logging.WARNING, logger=U.__name__):
        L2 = U.merge_obs_into_border(L)
    assert L2.graph['border'] is not None and len(L2.graph['obstacles']) == 0
    assert any('completely outside the border' in m for m in caplog.messages)


def test_merge_obstacles_intersection_multipolygon_raises():
    border = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
    obstacle = [(-1, -11), (1, -11), (1, 11), (-1, 11)]
    L = _make_L(border, [obstacle])
    with pytest.raises(ValueError, match='multiple pieces'):
        U.merge_obs_into_border(L)


def test_merge_obstacles_intersection_empty_border_raises():
    border = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
    obstacle = border[:]
    L = _make_L(border, [obstacle])
    with pytest.raises(ValueError, match='empty border'):
        U.merge_obs_into_border(L)


def test_merge_obstacles_inside_kept():
    border = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
    obstacle = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    L = _make_L(border, [obstacle])
    L2 = U.merge_obs_into_border(L)
    assert len(L2.graph['obstacles']) == 1


def test_buffer_border_obs_negative_raises():
    tC, sC, bC, _ = tiny_site(T=3)
    w = WindFarmNetwork(cables=3, turbinesC=tC, substationsC=sC, borderC=bC)
    with pytest.raises(ValueError, match='must be equal or greater than 0'):
        U.buffer_border_obs(w.L, buffer_dist=-1.0)


def test_buffer_border_obs_with_border_positive_shrinks_obstacles():
    # build L with explicit border/obstacle layout
    border = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    obstacle = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [0.0, 50.0]])
    L = _make_L(border, [obstacle], T=0, R=0)
    L2, pre = U.buffer_border_obs(L, buffer_dist=5.0)
    assert isinstance(pre, dict) and 'obstaclesC' in pre and 'borderC' in pre
    assert isinstance(L2.graph['obstacles'], list) and len(L2.graph['obstacles']) >= 1


def test_ensure_closed_adds_repeat_point():
    ring = np.array([[0.0, 0.0], [1, 0], [1, 1]])
    closed = U._ensure_closed(ring)
    assert np.allclose(closed[0], closed[-1])


def test_ensure_closed_noop_if_already_closed():
    ring = LinearRing([(0, 0), (1, 0), (1, 1)])
    coords = np.array(ring.coords[:], float)
    closed = U._ensure_closed(coords)
    assert np.array_equal(closed, coords)


# ==========================================================
# MILP/HGS warmstart & control-flow (monkeypatched stubs)
# ==========================================================



def test_milprouter_route_warmstart_branched_segmented_and_retries(monkeypatch):
    from optiwindnet import api as api_mod

    calls = {'eligible': 0, 'set_problem': 0, 'solve': 0, 'assign': 0}

    def fake_is_warmstart_eligible(**kwargs):
        calls['eligible'] += 1

    def fake_EW_presolver(A, capacity, maxiter=None):
        return tiny_S(T=A.graph['T'], R=A.graph['R'])

    class DummySolver:
        options = {}

        def set_problem(self, P, A, capacity, model_options, warmstart=None):
            calls['set_problem'] += 1
            if calls['set_problem'] == 1:
                raise OWNWarmupFailed('boom')

        def solve(self, time_limit, mip_gap, options, verbose):
            calls['solve'] += 1
            if calls['solve'] == 1:
                raise OWNSolutionNotFound('no sol yet')

        def get_solution(self):
            return tiny_S(), tiny_G()

    monkeypatch.setattr(api_mod, 'solver_factory', lambda name: DummySolver())
    monkeypatch.setattr(api_mod, 'is_warmstart_eligible', fake_is_warmstart_eligible)
    monkeypatch.setattr(api_mod, 'EW_presolver', fake_EW_presolver)
    monkeypatch.setattr(
        api_mod,
        'assign_cables',
        lambda G, cables: calls.__setitem__('assign', calls['assign'] + 1),
    )

    mo = ModelOptions(topology='branched', feeder_route='segmented')
    router = MILPRouter(
        solver_name='ortools', time_limit=1, mip_gap=0.1, model_options=mo
    )

    tC, sC, bC, _ = tiny_site(T=2)
    wfn = WindFarmNetwork(
        cables=2, turbinesC=tC, substationsC=sC, borderC=bC, router=router
    )
    wfn.optimize()
    assert (
        calls['eligible'] == 1
        and calls['set_problem'] >= 2
        and calls['solve'] == 2
        and calls['assign'] == 1
    )


def test_milprouter_route_warmstart_nonbranched_single_and_multi(monkeypatch):
    from optiwindnet import api as api_mod

    calls = {'single': 0, 'multi': 0}

    def fake_iterative_hgs(A_norm, capacity, time_limit, **k):
        calls['single'] += 1
        return tiny_S(T=A_norm.graph['T'], R=A_norm.graph['R'])

    def fake_hgs_multi(A_norm, capacity, time_limit, **k):
        calls['multi'] += 1
        return tiny_S(T=A_norm.graph['T'], R=A_norm.graph['R'])

    def fake_as_norm(A):
        return A

    class DummySolverWarmFailThenOK:
        options = {}

        def __init__(self, R):
            self._R = R
            self._raised_once = False

        def set_problem(self, P, A, capacity, model_options, warmstart=None):
            if not self._raised_once:
                self._raised_once = True
                raise OWNWarmupFailed('warmup fail')

        def solve(self, *a, **k):
            pass

        def get_solution(self):
            return tiny_S(), tiny_G()

    monkeypatch.setattr(api_mod, 'assign_cables', lambda G, cables: None)
    monkeypatch.setattr(api_mod, 'as_normalized', fake_as_norm)
    monkeypatch.setattr(api_mod, 'iterative_hgs_cvrp', fake_iterative_hgs)
    monkeypatch.setattr(api_mod, 'hgs_multiroot', fake_hgs_multi)
    monkeypatch.setattr(api_mod, 'is_warmstart_eligible', lambda **k: None)

    # single substation
    tC, sC, bC, _ = tiny_site(T=2)
    monkeypatch.setattr(
        api_mod, 'solver_factory', lambda name: DummySolverWarmFailThenOK(R=1)
    )
    mo = ModelOptions(topology='radial')
    router = MILPRouter(
        solver_name='ortools', time_limit=1, mip_gap=0.1, model_options=mo
    )
    w1 = WindFarmNetwork(
        cables=2, turbinesC=tC, substationsC=sC, borderC=bC, router=router
    )
    w1.optimize()
    assert calls['single'] == 1

    # multi substation path
    tC2 = tC
    sC2 = np.array([[3.0, 0.0], [3.0, 1.0]])
    monkeypatch.setattr(
        api_mod, 'solver_factory', lambda name: DummySolverWarmFailThenOK(R=2)
    )
    router2 = MILPRouter(
        solver_name='ortools', time_limit=1, mip_gap=0.1, model_options=mo
    )
    w2 = WindFarmNetwork(
        cables=2, turbinesC=tC2, substationsC=sC2, borderC=bC, router=router2
    )
    w2.optimize()
    assert calls['multi'] == 1
