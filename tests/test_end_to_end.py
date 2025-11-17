# test end to ends
from __future__ import annotations

from typing import Sequence
import shutil
import pytest
import dill
from optiwindnet.api import WindFarmNetwork, EWRouter, MILPRouter
from .helpers import tiny_wfn
from .helpers import assert_graph_equal, is_package_installed
from . import paths

UNITTESTS_DILL = paths.UNITTESTS_DILL
END_TO_END_DILL = paths.END_TO_END_DILL
TEST_FILES_DIR = paths.TEST_FILES_DIR
SITES_DIR = paths.SITES_DIR
GEN_UNITS_SCRIPT = paths.GEN_UNITS_SCRIPT
GEN_END2END_SCRIPT = paths.GEN_END2END_SCRIPT

# Note: use fixtures provided by tests/conftest.py:
# - expected_end_to_end (session-scoped)
# - router_factory (factory to build routers)
# - locations (repository-backed sites)
#


def pytest_generate_tests(metafunc):
    # Keep existing dynamic parametrization but use the expected_end_to_end fixture path
    if 'key' not in metafunc.fixturenames:
        return

    blob = None
    try:
        # Use fixture-like load without instantiating fixtures at collection time:
        EXPECTED_PATH = TEST_FILES_DIR / 'expected_end_to_end.dill'
        if EXPECTED_PATH.exists():
            with EXPECTED_PATH.open('rb') as f:
                blob = dill.load(f)
    except Exception:
        blob = None

    if blob is None:
        metafunc.parametrize('key', [])
        return

    stored = blob.get('Cases', [])
    graphs = blob.get('RouterGraphs', {})
    sites: Sequence[str] = tuple(blob.get('Sites', ()))
    routers = blob.get('Routers', {})

    keys = [
        c['key']
        for c in stored
        if c.get('key') in graphs
        and c.get('site') in sites
        and c.get('router') in routers
    ]

    metafunc.parametrize('key', sorted(keys))


@pytest.fixture(scope='session')
def expected_blob(expected_end_to_end):
    # simple wrapper to make the fixture name explicit for tests
    return expected_end_to_end


def test_expected_router_graphs_match(expected_blob, key, router_factory, locations):
    graphs = expected_blob['RouterGraphs']
    # sites: Sequence[str] = tuple(expected_blob["Sites"])
    routers = expected_blob['Routers']

    case_meta = next(c for c in expected_blob['Cases'] if c['key'] == key)

    site_name = case_meta['site']
    router_name = case_meta['router']
    expected_G = graphs[key]

    router_spec = routers[router_name]
    cables = int(router_spec['cables'])

    if router_spec['class'] == 'MILPRouter':
        package_from_solver_name = {
            'ortools': 'ortools',
            'cplex': 'cplex',
            'gurobi': 'gurobipy',
            'highs': 'highspy',
            'scip': 'pyscipopt',
        }
        solver_name = router_spec['params']['solver_name']
        # Skip if the solver is not available in the test environment
        if solver_name in package_from_solver_name:
            if not is_package_installed(package_from_solver_name[solver_name]):
                pytest.skip(f'{solver_name} not available')
        elif solver_name in ('cbc'):
            if not shutil.which(solver_name):
                pytest.skip(f'{solver_name} not available')

    # build router via central factory
    router = router_factory(router_spec)
    # Load site (from central fixture 'locations')
    L = getattr(locations, site_name)  # unchanged semantics from generator
    wfn = WindFarmNetwork(L=L, cables=cables)
    wfn.optimize(router=router)

    ignored_keys = {'solution_time', 'runtime', 'pool_count'}
    assert_graph_equal(
        wfn.G, expected_G, ignored_graph_keys=ignored_keys, verbose=False
    )


@pytest.mark.skipif(not is_package_installed('ortools'), reason='ortools not available')
def test_ortools_with_warmstart():
    wfn = tiny_wfn()
    wfn.optimize(router=EWRouter())
    router_ortools = MILPRouter(
        solver_name='ortools', time_limit=2, mip_gap=0.005, verbose=True
    )
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
