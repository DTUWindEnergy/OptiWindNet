import pickle

import numpy as np
import pytest

from optiwindnet.api import EWRouter, WindFarmNetwork, HGSRouter, MILPRouter

from .helpers import assert_graph_equal
# ========== Test ==========


def test_wfn_fails_without_coordinates_or_L():
    with pytest.raises(
        ValueError,
        match='Both turbinesC and substationsC must be provided! Or alternatively L should be given.',
    ):
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
        'OptiWindNet prioritizes coordinates over L' in message
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
        (7, [(7, 1)]),
        ((7, 100), [(7, 100)]),
        ([(5, 100), (7, 150), (9, 200)], [(5, 100), (7, 150), (9, 200)]),
    ],
)
def test_wfn_cable_formats(LG_from_database, cables_input, expected_array):
    expected_L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=cables_input, L=expected_L)
    assert np.array_equal(wfn.cables, expected_array)


def test_wfn_invalid_cables_raises(LG_from_database):
    expected_L, _ = LG_from_database('eagle_EWRouter')

    with pytest.raises(ValueError, match='Invalid cable values'):
        WindFarmNetwork(cables=(5, 7, 9), L=expected_L)


def test_cables_capacity_calculation(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=[(5, 100), (7, 150)], L=L)
    assert wfn.cables_capacity == 7


def test_update_from_terse_links_invalid_type(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    with pytest.raises(ValueError, match='terse_links must be an array of integers'):
        wfn.update_from_terse_links(np.array([0.1, 2.0]))


def test_update_from_terse_links_invalid_shape(LG_from_database):
    L, _ = LG_from_database('eagle_EWRouter')
    wfn = WindFarmNetwork(cables=7, L=L)
    with pytest.raises(ValueError, match='1D array'):
        wfn.update_from_terse_links(np.array([[0, 1], [2, 3]]))


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


def test_from_yaml_invalid_path_type():
    with pytest.raises(TypeError, match='Filepath must be a string'):
        WindFarmNetwork.from_yaml(123)


def test_from_pbf_invalid_path_type():
    with pytest.raises(Exception):  # TypeError or custom error
        WindFarmNetwork.from_pbf(123)


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
    "router",
    [
        EWRouter(),
        EWRouter(feeder_route='straight'),
        HGSRouter(time_limit=1),
        HGSRouter(time_limit=1, feeder_limit=1, max_retries=5, balanced=True),
        MILPRouter(solver_name="ortools", time_limit=2, mip_gap=0.005),
        MILPRouter(solver_name="cbc", time_limit=2, mip_gap=0.005),
        MILPRouter(solver_name="cplex", time_limit=2, mip_gap=0.005),
        MILPRouter(solver_name="gurobi", time_limit=2, mip_gap=0.005),
        MILPRouter(solver_name="highs", time_limit=2, mip_gap=0.005),
        MILPRouter(solver_name="scip", time_limit=2, mip_gap=0.005),
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
    print(wfn.optimize())
    assert np.array_equal(wfn.optimize(), EXPECTED_TERSE)

    # case 2: router passed at call-time
    wfn = WindFarmNetwork(
        cables=cables,
        turbinesC=turbinesC,
        substationsC=substationsC,
    )
    assert np.array_equal(wfn.optimize(router=router), EXPECTED_TERSE)
