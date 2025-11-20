# test end to end
import pytest
from optiwindnet.api import WindFarmNetwork, EWRouter, MILPRouter
from .helpers import tiny_wfn, router_factory, assert_graph_equal, load_dill
from .paths import END_TO_END_DILL


def pytest_generate_tests(metafunc):
    if 'routed_instance' in metafunc.fixturenames:
        blob = load_dill(END_TO_END_DILL)
        routed_instances = []
        ids = []
        for case in blob['Cases']:
            key = case['key']
            routed_instances.append(
                dict(
                    location=case['location'],
                    router_spec=blob['Routers'][case['router']],
                    G=blob['Graphs'][key],
                )
            )
            ids.append(key)
        metafunc.parametrize('routed_instance', routed_instances, ids=ids)


def test_expected_router_graphs_match(routed_instance, locations):
    router_spec = routed_instance['router_spec']
    try:
        router = router_factory(router_spec)
    except (FileNotFoundError, ModuleNotFoundError):
        pytest.skip(f'{router_spec["params"]["solver_name"]} not available')
    G_ref = routed_instance['G']
    L = getattr(locations, routed_instance['location'])
    wfn = WindFarmNetwork(L=L, cables=router_spec['cables'])
    wfn.optimize(router=router)
    ignored_keys = {'solution_time', 'runtime', 'pool_count'}
    assert_graph_equal(wfn.G, G_ref, ignored_graph_keys=ignored_keys, verbose=False)


def test_ortools_with_warmstart():
    try:
        router_ortools = MILPRouter(
            solver_name='ortools', time_limit=2, mip_gap=0.005, verbose=True
        )
    except (FileNotFoundError, ModuleNotFoundError):
        pytest.skip('ortools not available')
    wfn = tiny_wfn()
    wfn.optimize(router=EWRouter())
    terse_links = wfn.optimize(router=router_ortools)
    expected = [-1, 0, 1, 2]
    assert list(terse_links) == expected

    # invalid warmstart
    wfn.G.add_edge(-1, 11)
    router_ortools = MILPRouter(
        solver_name='ortools', time_limit=2, mip_gap=0.005, verbose=True
    )
    terse_links = wfn.optimize(router=router_ortools)
    expected = [-1, 0, 1, 2]
    assert list(terse_links) == expected

    # --- with detours
    wfn = tiny_wfn(cables=1)
    wfn.optimize(router=EWRouter())
    router_ortools = MILPRouter(
        solver_name='ortools', time_limit=2, mip_gap=0.005, verbose=True
    )
    terse_links = wfn.optimize(router=router_ortools)
    expected = [-1, -1, -1, -1]
    assert list(terse_links) == expected

    # invalid warmstart
    wfn.G.add_edge(0, 12)
    wfn.G.add_edge(12, 13)
    wfn.G.remove_edge(0, -1)
    router_ortools = MILPRouter(
        solver_name='ortools', time_limit=2, mip_gap=0.005, verbose=True
    )
    terse_links = wfn.optimize(router=router_ortools)
    expected = [-1, -1, -1, -1]
    assert list(terse_links) == expected
