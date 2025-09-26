import pytest
from pathlib import Path
import dill

from optiwindnet.api import (
    EWRouter,
    HGSRouter,
    MILPRouter,
    WindFarmNetwork,
    ModelOptions,
)

from .helpers import assert_graph_equal


# -------------------------------
# Test data & router definitions
# -------------------------------
HERE = Path(__file__).parent
EXPECTED_PATH = HERE / "test_files" / "expected_DTU_letters.dill"

# Mirror the site name -> file path mapping used to generate the dill
SITES = {
    # sites_1
    "DTU_1ss_10wt": HERE / "test_files" / "DTU_tests_1ss_10wt.osm.pbf",
    # sites_2
    "DTU_1ss_40wt": HERE / "test_files" / "DTU_tests_1ss_40wt.osm.pbf",
    "DTU_2ss_40wt": HERE / "test_files" / "DTU_tests_2ss_40wt.osm.pbf",
    "DTU_4ss_40wt": HERE / "test_files" / "DTU_tests_4ss_40wt.osm.pbf",
    "DTU_1ss_100wt": HERE / "test_files" / "DTU_tests_1ss_100wt.osm.pbf",
}

# Mirror the router configs used when saving expected values
ROUTERS = {
    # routers_1
    "EWRouter_cap1": {"router": None, "cables": 1},
    "EWRouter_cap3": {"router": None, "cables": 3},
    "EWRouter_cap10": {"router": None, "cables": 10},
    "EWRouter_straight_cap1": {"router": EWRouter(feeder_route="straight"), "cables": 1},
    "EWRouter_straight_cap4": {"router": EWRouter(feeder_route="straight"), "cables": 4},
    "EWRouter_straight_cap10": {"router": EWRouter(feeder_route="straight"), "cables": 10},
    "HGSRouter_cap1": {"router": HGSRouter(time_limit=2, seed=0), "cables": 1},
    "HGSRouter_cap3": {"router": HGSRouter(time_limit=2, seed=0), "cables": 3},
    "HGSRouter_cap10": {"router": HGSRouter(time_limit=2, seed=0), "cables": 10},
    "HGSRouter_feeder_limit_cap1": {
        "router": HGSRouter(time_limit=0.5, feeder_limit=0, seed=0),
        "cables": 1,
    },
    "HGSRouter_feeder_limit_cap4": {
        "router": HGSRouter(time_limit=0.5, feeder_limit=0, seed=0),
        "cables": 4,
    },
    "HGSRouter_feeder_limit_cap10": {
        "router": HGSRouter(time_limit=0.5, feeder_limit=0, seed=0),
        "cables": 10,
    },
    "MILPRouter_cap10": {
        "router": MILPRouter(solver_name="ortools", time_limit=5, mip_gap=0.001),
        "cables": 10,
    },
    "MILPRouter_cap10_modeloptions": {
        "router": MILPRouter(
            solver_name="ortools",
            time_limit=5,
            mip_gap=0.001,
            model_options=ModelOptions(
                topology="radial",
                feeder_limit="minimum",
                feeder_route="straight",
            ),
        ),
        "cables": 10,
    },

    # routers_2
    "EWRouter_cap5": {"router": None, "cables": 5},
    "EWRouter_cap10": {"router": None, "cables": 10},
    "EWRouter_cap100": {"router": None, "cables": 100},
    "EWRouter_straight_cap7": {"router": EWRouter(feeder_route="straight"), "cables": 7},
    "EWRouter_straight_cap15": {"router": EWRouter(feeder_route="straight"), "cables": 15},
    "EWRouter_straight_cap100": {"router": EWRouter(feeder_route="straight"), "cables": 100},
}


def _split_key(key: str):
    """
    Given keys formed as f'{site_name}_{router_name}', recover site_name/router_name
    by matching the router_name suffix from ROUTERS.
    """
    for rname in ROUTERS:
        suffix = f"_{rname}"
        if key.endswith(suffix):
            site = key[: -len(suffix)]
            return site, rname
    raise KeyError(f"Unrecognized router suffix in key: {key!r}")


import numpy as np

# ========== Graph Assertion Helpers ==========



@pytest.mark.parametrize("key", sorted([
    # Load keys from dill once at import-time; if file is missing, pytest will show a clear error.
    *(
        dill.load(open(EXPECTED_PATH, "rb"))["RouterGraphs"].keys()
        if EXPECTED_PATH.exists()
        else []
    )
]))
def test_expected_router_graphs_match(key):
    if not EXPECTED_PATH.exists():
        pytest.skip(f"Expected file not found: {EXPECTED_PATH}")

    with open(EXPECTED_PATH, "rb") as f:
        expected = dill.load(f)

    expected_G = expected["RouterGraphs"][key]

    site_name, router_name = _split_key(key)

    # Resolve inputs
    try:
        pbf_path = SITES[site_name]
    except KeyError as e:
        pytest.fail(f"Unknown site_name {site_name!r} (from key {key!r}). Update SITES mapping.")

    try:
        router_cfg = ROUTERS[router_name]
    except KeyError:
        pytest.fail(f"Unknown router_name {router_name!r} (from key {key!r}). Update ROUTERS mapping.")

    cables = router_cfg["cables"]
    router = router_cfg["router"]

    # Build & optimize
    wfn = WindFarmNetwork.from_pbf(filepath=str(pbf_path), cables=cables)
    wfn.optimize(router=router)

    # Compare graphs; customize ignored keys if you need to
    ignored_keys = {'solution_time', 'runtime', 'pool_count'}
    assert_graph_equal(wfn.G, expected_G, ignored_graph_keys=ignored_keys)
