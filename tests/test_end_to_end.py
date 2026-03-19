# test end to end
import pytest
from optiwindnet.api import WindFarmNetwork, EWRouter, MILPRouter
from .helpers import tiny_wfn, router_factory, load_instances
from .paths import SOLUTIONS_FILE


def pytest_generate_tests(metafunc):
    if 'routed_instance' in metafunc.fixturenames:
        if not SOLUTIONS_FILE.exists():
            metafunc.parametrize(
                'routed_instance',
                [
                    pytest.param(
                        None,
                        marks=pytest.mark.skip(
                            reason=(
                                f'Missing expected test data: {SOLUTIONS_FILE}\n'
                                'To (re)generate run: python update_expected_values.py\n'
                                'Or run pytest with --regen-expected.'
                            ),
                        ),
                    )
                ],
            )
            return
        all_instances = load_instances(SOLUTIONS_FILE)
        routed_instances = []
        ids = []
        for key in sorted(all_instances):
            routed_instances.append(all_instances[key])
            ids.append(key)
        metafunc.parametrize('routed_instance', routed_instances, ids=ids)


def test_expected_router_graphs_match(routed_instance, locations):
    router_spec = routed_instance['router_spec']
    try:
        router = router_factory(router_spec)
    except (FileNotFoundError, ModuleNotFoundError):
        pytest.skip(f'{router_spec["params"]["solver_name"]} not available')
    terse_ref = routed_instance['terse_links']
    L = getattr(locations, routed_instance['location'])
    wfn = WindFarmNetwork(L=L, cables=router_spec['cables'])
    wfn.optimize(router=router)
    assert tuple(wfn.terse_links().tolist()) == terse_ref


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
