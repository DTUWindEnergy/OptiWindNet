# test end to ends
from __future__ import annotations

from typing import Dict, Any, Sequence

import pytest
import dill
from optiwindnet.api import WindFarmNetwork
from optiwindnet.MILP import ModelOptions
from .helpers import assert_graph_equal
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
# This removes duplicated loader/constructor code.

def _make_model_options_from_spec(spec: Dict[str, Any]) -> ModelOptions:
    return ModelOptions(**spec)


def _make_router_from_spec_via_factory(spec: Dict[str, Any], router_factory):
    # router_factory already expands ModelOptions as needed; use it.
    return router_factory(spec)


def pytest_generate_tests(metafunc):
    # Keep existing dynamic parametrization but use the expected_end_to_end fixture path
    if "key" not in metafunc.fixturenames:
        return

    blob = None
    try:
        # Use fixture-like load without instantiating fixtures at collection time:
        EXPECTED_PATH = TEST_FILES_DIR / "expected_end_to_end.dill"
        if EXPECTED_PATH.exists():
            with EXPECTED_PATH.open("rb") as f:
                blob = dill.load(f)
    except Exception:
        blob = None

    if blob is None:
        metafunc.parametrize("key", [])
        return

    stored = blob.get("Cases", [])
    graphs = blob.get("RouterGraphs", {})
    sites: Sequence[str] = tuple(blob.get("Sites", ()))
    routers = blob.get("Routers", {})

    keys = [
        c["key"]
        for c in stored
        if c.get("key") in graphs
        and c.get("site") in sites
        and c.get("router") in routers
    ]

    metafunc.parametrize("key", sorted(keys))


@pytest.fixture(scope="session")
def expected_blob(expected_end_to_end):
    # simple wrapper to make the fixture name explicit for tests
    return expected_end_to_end


def test_expected_router_graphs_match(expected_blob, key, router_factory, locations):
    graphs = expected_blob["RouterGraphs"]
    sites: Sequence[str] = tuple(expected_blob["Sites"])
    routers = expected_blob["Routers"]

    case_meta = next(c for c in expected_blob["Cases"] if c["key"] == key)

    site_name = case_meta["site"]
    router_name = case_meta["router"]
    expected_G = graphs[key]

    router_spec = routers[router_name]
    cables = int(router_spec["cables"])
    # build router via central factory
    router = router_factory(router_spec)

    # Skip MILP if OR-Tools isn't available in the test environment
    if router_spec.get("class") == "MILPRouter":
        pytest.importorskip("ortools", reason="MILPRouter requires OR-Tools")

    # Load site (from central fixture 'locations')
    L = getattr(locations, site_name)  # unchanged semantics from generator
    wfn = WindFarmNetwork(L=L, cables=cables)
    if router is None:
        wfn.optimize()
    else:
        wfn.optimize(router=router)

    if site_name == 'example_1ss_300wt' or site_name == 'example_4ss_300wt':
        ignored_keys = {"solution_time", "runtime", "pool_count", "norm_scale"}
    else:
        ignored_keys = {"solution_time", "runtime", "pool_count"}

    assert_graph_equal(wfn.G, expected_G, ignored_graph_keys=ignored_keys, verbose=False)
