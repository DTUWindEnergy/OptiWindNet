from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import optiwindnet.api as api
import optiwindnet.plotting as plotting
from optiwindnet.api import (
    EWRouter,
    HGSRouter,
    MILPRouter,
    Router,
    WindFarmNetwork,
)
from optiwindnet.MILP import ModelOptions

from .helpers import tiny_wfn

_LOCATION_FILE = Path(__file__).parent / 'locations' / 'example_location.yaml'


class _RecordingRouter(Router):
    _summary_attrs = ()

    def __init__(self, result):
        self.result = result
        self.calls = []

    def route(self, P, A, cables, cables_capacity, verbose=False, **kwargs):
        self.calls.append(
            {
                'P': P,
                'A': A,
                'cables': cables,
                'cables_capacity': cables_capacity,
                'verbose': verbose,
                **kwargs,
            }
        )
        return self.result


def test_optimize_forwards_inputs_and_warmstart_state():
    solved = tiny_wfn()
    wfn = tiny_wfn(optimize=False)
    router = _RecordingRouter((solved.S, solved.G))

    wfn.optimize(router=router, verbose=True)
    first = router.calls[0]
    assert first['P'] is wfn.P
    assert first['A'] is wfn.A
    assert first['cables'] == wfn.cables
    assert first['cables_capacity'] == wfn.cables_capacity
    assert first['verbose'] is True
    assert 'S_warm' not in first
    assert wfn.S is solved.S and wfn.G is solved.G

    wfn.optimize()
    second = router.calls[1]
    assert second['S_warm'] is solved.S
    assert second['S_warm_has_detour'] is bool(solved.G.graph.get('D', 0))


@pytest.mark.parametrize('feeder_route', ('segmented', 'straight'))
def test_ewrouter_forwards_constructor_options(monkeypatch, feeder_route):
    solved = tiny_wfn()
    calls = []

    def fake_constructor(A, **kwargs):
        calls.append((A, kwargs))
        return solved.S

    monkeypatch.setattr(api, 'constructor', fake_constructor)
    monkeypatch.setattr(api, 'G_from_S', lambda S, A: solved.G)
    monkeypatch.setattr(
        api,
        'PathFinder',
        lambda *args, **kwargs: type(
            'PF', (), {'create_detours': lambda self: solved.G}
        )(),
    )
    monkeypatch.setattr(api, 'assign_cables', lambda G, cables: None)
    router = EWRouter(
        method='ringed',
        maxiter=17,
        bias_margin=0.25,
        feeder_route=feeder_route,
    )

    router.route(solved.P, solved.A, solved.cables, solved.cables_capacity)

    assert calls == [
        (
            solved.A,
            {
                'capacity': solved.cables_capacity,
                'method': 'ringed',
                'maxiter': 17,
                'bias_margin': 0.25,
                'weigh_detours': feeder_route == 'segmented',
                'straight_feeder_route': feeder_route == 'straight',
            },
        )
    ]


def test_hgsrouter_forwards_low_level_options(monkeypatch):
    solved = tiny_wfn()
    captured = {}

    def fake_hgs(A, **kwargs):
        captured.update(kwargs)
        return solved.S

    monkeypatch.setattr(api, 'hgs_cvrp', fake_hgs)
    monkeypatch.setattr(api, 'G_from_S', lambda S, A: solved.G)
    monkeypatch.setattr(
        api,
        'PathFinder',
        lambda *args, **kwargs: type(
            'PF', (), {'create_detours': lambda self: solved.G}
        )(),
    )
    monkeypatch.setattr(api, 'assign_cables', lambda G, cables: None)
    router = HGSRouter(
        time_limit=0.4,
        feeder_limit=3,
        feeder_exact=True,
        max_retries=7,
        balanced=True,
        ringed=True,
        seed=11,
    )

    router.route(solved.P, solved.A, solved.cables, solved.cables_capacity)

    assert captured == {
        'capacity': solved.cables_capacity,
        'time_limit': 0.4,
        'max_retries': 7,
        'vehicles': 3,
        'vehicles_exact': True,
        'balanced': True,
        'ringed': True,
        'seed': 11,
    }


def test_milprouter_constructs_and_forwards_to_solver(monkeypatch):
    solved = tiny_wfn()

    class FakeSolver:
        options = {'default': True}

        def __init__(self):
            self.problem_calls = []
            self.solve_calls = []

        def set_problem(self, *args, **kwargs):
            self.problem_calls.append((args, kwargs))

        def solve(self, **kwargs):
            self.solve_calls.append(kwargs)

        def get_solution(self):
            return solved.S, solved.G

    fake = FakeSolver()
    factory_calls = []
    monkeypatch.setattr(
        api,
        'solver_factory',
        lambda name: factory_calls.append(name) or fake,
    )
    monkeypatch.setattr(api, 'assign_cables', lambda G, cables: None)
    options = ModelOptions(topology='ringed', feeder_route='straight')
    router = MILPRouter(
        solver_name='fake',
        time_limit=9,
        mip_gap=0.2,
        solver_options={'threads': 1},
        model_options=options,
    )

    S, G = router.route(
        solved.P,
        solved.A,
        solved.cables,
        solved.cables_capacity,
        S_warm=solved.S,
        num_retries=4,
        verbose=True,
    )

    assert factory_calls == ['fake']
    assert fake.problem_calls == [
        (
            (solved.P, solved.A),
            {
                'capacity': solved.cables_capacity,
                'model_options': options,
                'warmstart': solved.S,
            },
        )
    ]
    assert fake.solve_calls == [
        {
            'time_limit': 9,
            'mip_gap': 0.2,
            'options': {'threads': 1},
            'verbose': True,
        }
    ]
    assert S is solved.S and G is solved.G


def _run_ortools_warmstart_cases():
    router = MILPRouter(
        solver_name='ortools.cp_sat', time_limit=2, mip_gap=0.005, verbose=True
    )
    wfn = tiny_wfn()
    wfn.optimize(router=router)
    results = [list(wfn.optimize(router=router))]

    wfn.G.add_edge(-1, 11)
    results.append(list(wfn.optimize(router=router)))

    wfn = tiny_wfn(cables=1)
    wfn.optimize(router=EWRouter())
    router = MILPRouter(
        solver_name='ortools.cp_sat', time_limit=2, mip_gap=0.005, verbose=True
    )
    results.append(list(wfn.optimize(router=router)))

    wfn.G.add_edges_from([(0, 12), (12, 13)])
    wfn.G.remove_edge(0, -1)
    results.append(list(wfn.optimize(router=router)))
    return results


def test_ortools_warmstart_behavior(ortools_worker):
    result = ortools_worker.run(_run_ortools_warmstart_cases, (), timeout=30)
    if isinstance(result, (FileNotFoundError, ModuleNotFoundError)):
        pytest.skip('ortools.cp_sat not available')
    if isinstance(result, BaseException):
        raise result
    assert result == [
        [-1, 0, 1, 2],
        [-1, 0, 1, 2],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
    ]


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
    assert len(terse) == wfn.S.graph['T']


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
        WindFarmNetwork(  # pyrefly: ignore[missing-argument]
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
            cables=(5, (7, 3, 8), 9),  # pyrefly: ignore[bad-argument-type]
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


def test_from_own_yaml_loads_own_yaml_file():
    wfn = WindFarmNetwork.from_own_yaml(str(_LOCATION_FILE), cables=4)

    assert wfn.L.graph['T'] == 12
    assert wfn.L.graph['R'] == 1


def test_deprecated_from_yaml_warns():
    with pytest.warns(DeprecationWarning, match='from_own_yaml'):
        wfn = WindFarmNetwork.from_yaml(str(_LOCATION_FILE), cables=4)

    assert wfn.L.graph['T'] == 12


def test_from_own_yaml_invalid_path():
    with pytest.raises(Exception):
        WindFarmNetwork.from_own_yaml(r'not>a*path')


def test_from_pbf_invalid_path():
    with pytest.raises(Exception):
        WindFarmNetwork.from_pbf(r'not>a*path')


def test_terse_links_output():
    terse_expected = np.array([-1, 0, 1, 2])
    wfn = tiny_wfn()
    terse = wfn.terse_links()
    assert len(terse) == wfn.S.graph['T']
    assert np.array_equal(wfn.terse_links(), terse_expected)


def test_update_from_terse_links():
    wfn = tiny_wfn()
    terse1 = wfn.terse_links()
    terse = np.array([1, 2, -1, 0])
    wfn.update_from_terse_links(terse)
    terse2 = wfn.terse_links()
    assert np.array_equal(terse1, np.array([-1, 0, 1, 2]))
    assert np.array_equal(terse2, np.array([1, 2, -1, 0]))


def test_update_from_terse_links_preserves_topology_architecture():
    wfn = tiny_wfn()
    terse = wfn.terse_links()
    topology = wfn.S.graph['topology']

    wfn.update_from_terse_links(terse)

    assert wfn.S.graph['topology'] == terse.topology == topology

    # Serialized links are plain arrays; the public API also accepts the
    # architecture in its legacy string form.
    wfn.update_from_terse_links(np.asarray(terse), topology=topology)
    assert wfn.S.graph['topology'] == topology


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


def test_plots_matplotlib_backend():
    """ax=None routes all plot methods to the matplotlib backend."""
    wfn = tiny_wfn()
    wfn.optimize()
    for method in ('plot', 'plot_location', 'plot_available_links'):
        ax = getattr(wfn, method)(ax=None)
        assert hasattr(ax, 'figure')
    ax = wfn.plot_navigation_mesh(ax=None)
    assert hasattr(ax, 'figure')
    ax = wfn.plot_selected_links(ax=None)
    assert hasattr(ax, 'figure')
    plt.close('all')


def test_get_network_returns_array_smoke():
    wfn = tiny_wfn()
    data = wfn.get_network()
    # basic shape/type checks
    assert isinstance(data, np.ndarray)
    assert data.ndim == 1  # structured 1-D array (rows)

    # expected structured dtype and field names
    expected_fields = ('src', 'tgt', 'length', 'load', 'cable')
    assert data.dtype.names is not None
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


def test_plot_original_vs_buffered_without_prior_buffer_logs_message(caplog):
    import logging

    wfn = tiny_wfn()
    with caplog.at_level(logging.INFO, logger='optiwindnet.api'):
        _ = wfn.plot_original_vs_buffered()
    assert 'No buffering is performed' in caplog.text


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
    assert len(terse) == wfn.S.graph['T']


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


def test_buffer_border_obs_with_border_positive_shrinks_obstacles():
    # build L with explicit border/obstacle layout
    wfn = tiny_wfn()
    assert 'obstacles' in wfn.L.graph and wfn.L.graph['border'] is not None
    wfn.add_buffer(buffer_dist=5.0)
    assert (
        isinstance(wfn.L.graph['obstacles'], list)
        and len(wfn.L.graph['obstacles']) == 0
    )


def test_wfn_repr_and_repr_svg():
    unsolved = tiny_wfn(optimize=False)
    repr_unsolved = repr(unsolved)
    assert 'WindFarmNetwork' in repr_unsolved
    assert 'unsolved' in repr_unsolved

    svg_unsolved = unsolved._repr_svg_()
    assert isinstance(svg_unsolved, str)
    assert '<svg' in svg_unsolved

    solved = tiny_wfn(optimize=True)
    repr_solved = repr(solved)
    assert 'length=' in repr_solved

    svg_solved = solved._repr_svg_()
    assert isinstance(svg_solved, str)
    assert '<svg' in svg_solved


def test_wfn_solution_info():
    wfn = tiny_wfn(optimize=True)
    info = wfn.solution_info()
    assert isinstance(info, dict)
    assert info.get('router') == 'EWRouter'
    assert info.get('capacity') == wfn.cables_capacity


def test_milp_router_warmup_fallback_and_retry_exhaustion(monkeypatch):
    from optiwindnet.MILP import OWNSolutionNotFound, OWNWarmupFailed

    # Create a fake solver that raises OWNWarmupFailed on set_problem with warmstart
    class FakeSolver:
        def __init__(self):
            self.model_options = ModelOptions()
            self.metadata = type('Meta', (), {'warmed_by': ''})()

        def set_problem(self, P, A, capacity, model_options, warmstart=None):
            if warmstart is not None and not getattr(self, '_passed_warmup', False):
                self._passed_warmup = True
                raise OWNWarmupFailed('Warmup failed')

        def solve(self, **kwargs):
            raise OWNSolutionNotFound('No solution')

    monkeypatch.setattr(api, 'solver_factory', lambda name: FakeSolver())

    router = MILPRouter(solver_name='ortools.cp_sat', time_limit=0.1, mip_gap=1e-3)
    solved = tiny_wfn()
    with pytest.raises(OWNSolutionNotFound, match='after 1 retries'):
        router.route(
            solved.P, solved.A, solved.cables, solved.cables_capacity, num_retries=1
        )


def test_wfn_handles_sweep():
    from .sitecache import SELECTED_HANDLES, get_location

    for handle in SELECTED_HANDLES:
        wfn = WindFarmNetwork(cables=[(5, 0.0)], L=get_location(handle), handle=handle)
        assert wfn.handle == handle
        assert wfn.L.graph['handle'] == handle
