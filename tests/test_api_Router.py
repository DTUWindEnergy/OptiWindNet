import pickle
from unittest.mock import MagicMock

import numpy as np
import pytest

from optiwindnet.api import EWRouter, HGSRouter, MILPRouter, WindFarmNetwork, ModelOptions

# from tests.assertions import assert_graph_equal
# ========== Assertions ==========


def assert_graph_equal(G1, G2, ignored_graph_keys=None):
    if ignored_graph_keys is None:
        ignored_graph_keys = set()

    assert set(G1.nodes) == set(G2.nodes), 'Node sets differ'
    assert set(G1.edges) == set(G2.edges), 'Edge sets differ'

    for n in G1.nodes:
        attrs1 = G1.nodes[n]
        attrs2 = G2.nodes[n]
        filtered1 = {k: v for k, v in attrs1.items() if k != 'label'}
        filtered2 = {k: v for k, v in attrs2.items() if k != 'label'}
        assert filtered1 == filtered2, (
            f'Node {n} attributes differ: {filtered1} != {filtered2}'
        )

    keys1 = set(G1.graph.keys()) - ignored_graph_keys
    keys2 = set(G2.graph.keys()) - ignored_graph_keys
    assert keys1 == keys2, f'Graph keys mismatch: {keys1.symmetric_difference(keys2)}'

    for k in keys1:
        v1 = G1.graph[k]
        v2 = G2.graph[k]
        if isinstance(v1, np.ndarray):
            assert np.array_equal(v1, v2), f"Mismatch in graph['{k}']"
        elif isinstance(v1, list):
            assert isinstance(v2, list) and len(v1) == len(v2), (
                f"Mismatch in list length for graph['{k}']"
            )
            for a, b in zip(v1, v2):
                if isinstance(a, np.ndarray):
                    assert np.array_equal(a, b), (
                        f"Mismatch in list of arrays in graph['{k}']"
                    )
                else:
                    assert a == b, f"Mismatch in list values in graph['{k}']"
        else:
            assert v1 == v2, f"Mismatch in graph['{k}']: {v1} != {v2}"


# ========== Test Routers ==========


@pytest.mark.parametrize(
    'label, router, ignored_keys',
    [
        ('eagle_EWRouter', None, {'runtime'}),
        ('eagle_EWRouter_straight', EWRouter(feeder_route='straight'), {'runtime'}),
        ('taylor_EWRouter', None, {'runtime'}),
        ('taylor_EWRouter_straight', EWRouter(feeder_route='straight'), {'runtime'}),
        ('eagle_HGSRouter', HGSRouter(time_limit=2), {'solution_time', 'runtime'}),
        (
            'eagle_HGSRouter_feeder_limit',
            HGSRouter(time_limit=2, feeder_limit=0),
            {'solution_time', 'runtime'},
        ),
        ('taylor_HGSRouter', HGSRouter(time_limit=2), {'solution_time', 'runtime'}),
        (
            'taylor_HGSRouter_feeder_limit',
            HGSRouter(time_limit=2, feeder_limit=0),
            {'solution_time', 'runtime'},
        ),
        (
            'eagle_MILPRouter',
            MILPRouter(solver_name='ortools', time_limit=5, mip_gap=0.005),
            {'runtime', 'bound', 'pool_count', 'relgap'},
        ),
    ],
)
def test_router_variants(LG_from_database, label, router, ignored_keys):
    expected_L, G = LG_from_database(label)
    print(G.graph['capacity'])

    wfn = WindFarmNetwork(
        cables=G.graph['capacity'],
        L=expected_L,
    )

    wfn.optimize(router=router)

    ignored_keys = ignored_keys or set()
    assert_graph_equal(wfn.G, G, ignored_graph_keys=ignored_keys)


@pytest.mark.parametrize(
    'router_class, init_kwargs, expected_attrs',
    [
        # --- EWRouter Tests ---
        (
            EWRouter,
            {},
            {'maxiter': 10000, 'feeder_route': 'segmented', 'verbose': False},
        ),
        (
            EWRouter,
            {'maxiter': 5000, 'feeder_route': 'straight', 'verbose': True},
            {'maxiter': 5000, 'feeder_route': 'straight', 'verbose': True},
        ),
        # --- HGSRouter Tests ---
        (
            HGSRouter,
            {'time_limit': 60},
            {
                'time_limit': 60,
                'max_reruns': 10,
                'feeder_limit': None,
                'balanced': False,
                'seed': 0,
                'verbose': False,
            },
        ),
        (
            HGSRouter,
            {
                'time_limit': 120,
                'feeder_limit': 3,
                'max_reruns': 20,
                'balanced': True,
                'seed': 42,
                'verbose': True,
            },
            {
                'time_limit': 120,
                'feeder_limit': 3,
                'max_reruns': 20,
                'balanced': True,
                'seed': 42,
                'verbose': True,
            },
        ),
        # --- MILPRouter Tests ---
        (
            MILPRouter,
            {'solver_name': 'cbc', 'time_limit': 100, 'mip_gap': 0.05},
            {'solver_name': 'cbc', 'time_limit': 100, 'mip_gap': 0.05},
        ),
        (
            MILPRouter,
            {
                'solver_name': 'cbc',
                'time_limit': 200,
                'mip_gap': 0.01,
                'solver_options': {'threads': 2},
                'model_options': ModelOptions(balanced=True),
                'verbose': True,
            },
            {
                'solver_name': 'cbc',
                'time_limit': 200,
                'mip_gap': 0.01,
                'solver_options': {'threads': 2},
                'verbose': True,
            },
        ),
    ],
)
def test_router_initialization(router_class, init_kwargs, expected_attrs):
    router = router_class(**init_kwargs)
    for attr, expected in expected_attrs.items():
        actual = getattr(router, attr)
        if isinstance(expected, dict):
            assert actual == expected
        else:
            assert actual == expected

@pytest.mark.parametrize(
    "router_class, init_kwargs, call_args",
    [
        (
            EWRouter,
            {},
            {
                "L": "L_test",
                "A": "A_test",
                "P": "P_test",
                "cables": "c_test",
                "cables_capacity": 10,
            },
        ),
        (
            HGSRouter,
            {"time_limit": 10},
            {
                "A": "A_test",
                "P": "P_test",
                "cables": "c_test",
                "cables_capacity": 10,
            },
        ),
        (
            MILPRouter,
            {"solver_name": "cbc", "time_limit": 10, "mip_gap": 0.1},
            {
                "A": "A_test",
                "P": "P_test",
                "cables": "c_test",
                "cables_capacity": 10,
            },
        ),
    ],
)
def test_router_call_delegates_to_optimize(router_class, init_kwargs, call_args):
    router = router_class(**init_kwargs)

    # Mock optimize
    router.optimize = MagicMock(return_value="mock_output")

    # Call router like a function
    result = router(**call_args)

    # Fill in default values expected by __call__
    expected_args = {
        "L": None,
        "A": None,
        "P": None,
        "cables": None,
        "cables_capacity": None,
        "S_warm": None,
        "S_warm_has_detour": False,
        "verbose": False,
    }
    expected_args.update(call_args)

    # Assert optimize was called correctly
    router.optimize.assert_called_once_with(**expected_args)
    assert result == "mock_output"

