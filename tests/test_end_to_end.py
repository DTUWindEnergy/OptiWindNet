# tests/test_with_DTU_letters.py
"""
Replay-and-verify tests using repository-backed sites.

This test reads a dill blob produced by scripts/generate_expected_end_to_end.py.
Blob structure (new format):
  - "Sites":         (site_name, ...)                  # tuple/list of site names
  - "Routers":       {router_name: {"class": ..., "params": {...}, "cables": int}}
  - "Cases":         [{"key": "<site>_<router>", "site": site_name, "router": router_name}, ...]
  - "RouterGraphs":  {key: expected_networkx_graph}
  - "Meta":          environment info (informational)

For each case, we rebuild the WindFarmNetwork with the repository site
(load_repository(); getattr(locations, site_name)), recreate the router from
its spec (incl. ModelOptions for MILPRouter), run the optimization, and
assert the produced graph equals the stored one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Sequence

import dill
import pytest

from optiwindnet.api import EWRouter, HGSRouter, MILPRouter, WindFarmNetwork
from optiwindnet.MILP import ModelOptions
from optiwindnet.importer import load_repository

# Uses your existing helper assertion
from .helpers import assert_graph_equal

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
HERE = Path(__file__).parent
EXPECTED_PATH = HERE / "test_files" / "expected_end_to_end.dill"


# ---------------------------------------------------------------------
# Router (re)construction helpers
# ---------------------------------------------------------------------
def _make_model_options_from_spec(spec: Dict[str, Any]) -> ModelOptions:
    """Convert a plain dict into a ModelOptions instance."""
    return ModelOptions(**spec)


def _make_router_from_spec(spec: Dict[str, Any]):
    """
    Instantiate a router from a spec dict:
      {"class": "EWRouter"|"HGSRouter"|"MILPRouter"|None, "params": {...}, "cables": int}
    """
    clsname = spec.get("class")
    params = dict(spec.get("params", {}))

    # Expand nested ModelOptions if present for MILPRouter
    if clsname == "MILPRouter" and isinstance(params.get("model_options"), dict):
        params["model_options"] = _make_model_options_from_spec(params["model_options"])

    if clsname is None:
        return None
    if clsname == "EWRouter":
        return EWRouter(**params)
    if clsname == "HGSRouter":
        return HGSRouter(**params)
    if clsname == "MILPRouter":
        return MILPRouter(**params)

    raise ValueError(f"Unknown router class {clsname!r}")


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def _load_expected_blob():
    """Load the dill blob or return None if missing/unreadable."""
    if not EXPECTED_PATH.exists():
        return None
    with EXPECTED_PATH.open("rb") as f:
        return dill.load(f)


# ---------------------------------------------------------------------
# Pytest collection hook: build the param list from the dill blob
# ---------------------------------------------------------------------
def pytest_generate_tests(metafunc):
    if "key" not in metafunc.fixturenames:
        return

    blob = _load_expected_blob()
    if blob is None:
        metafunc.parametrize("key", [])
        return

    stored = blob.get("Cases", [])
    graphs = blob.get("RouterGraphs", {})
    sites: Sequence[str] = tuple(blob.get("Sites", ()))  # sequence of names
    routers = blob.get("Routers", {})

    keys = [
        c["key"]
        for c in stored
        if c.get("key") in graphs
        and c.get("site") in sites
        and c.get("router") in routers
    ]

    metafunc.parametrize("key", sorted(keys))


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def expected_blob():
    blob = _load_expected_blob()
    if blob is None:
        pytest.skip(f"Expected file not found: {EXPECTED_PATH}")
    return blob


# ---------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------
def test_expected_router_graphs_match(expected_blob, key):
    graphs = expected_blob["RouterGraphs"]
    sites: Sequence[str] = tuple(expected_blob["Sites"])  # names only
    routers = expected_blob["Routers"]

    # Find metadata for this key
    try:
        case_meta = next(c for c in expected_blob["Cases"] if c["key"] == key)
    except StopIteration:
        pytest.fail(f"Case metadata not found for key {key!r}")

    site_name = case_meta["site"]
    router_name = case_meta["router"]
    expected_G = graphs[key]

    assert site_name in sites, f"Site {site_name!r} not listed in blob Sites"

    # Build router
    router_spec = routers[router_name]
    cables = int(router_spec["cables"])
    router = _make_router_from_spec(router_spec)

    # Skip MILP if OR-Tools isn't available in the test environment
    if router_spec.get("class") == "MILPRouter":
        pytest.importorskip("ortools", reason="MILPRouter requires OR-Tools")

    # Load site from repository by name and optimize
    locations = load_repository(path=r'tests\test_files\sites')              # assumed valid per generator contract
    L = getattr(locations, site_name)          # no validation
    wfn = WindFarmNetwork(L=L, cables=cables)
    if router is None:
        wfn.optimize()
    else:
        wfn.optimize(router=router)

    # Compare graphs; ignore volatile per-run keys
    ignored_keys = {"solution_time", "runtime", "pool_count"}
    assert_graph_equal(wfn.G, expected_G, ignored_graph_keys=ignored_keys, verbose=False)
