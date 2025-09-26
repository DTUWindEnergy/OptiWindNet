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
# Paths
# -------------------------------
HERE = Path(__file__).parent
EXPECTED_PATH = HERE / "test_files" / "expected_DTU_letters.dill"


# -------------------------------
# Router definitions (factories!)
# -------------------------------
ROUTERS = {
    # routers_1
    "EWRouter_cap1":               {"router_factory": lambda: None,                              "cables": 1},
    "EWRouter_cap3":               {"router_factory": lambda: None,                              "cables": 3},
    "EWRouter_cap10":              {"router_factory": lambda: None,                              "cables": 10},
    "EWRouter_straight_cap1":      {"router_factory": lambda: EWRouter(feeder_route="straight"), "cables": 1},
    "EWRouter_straight_cap4":      {"router_factory": lambda: EWRouter(feeder_route="straight"), "cables": 4},
    "EWRouter_straight_cap10":     {"router_factory": lambda: EWRouter(feeder_route="straight"), "cables": 10},

    "HGSRouter_cap1":              {"router_factory": lambda: HGSRouter(time_limit=0.5, seed=0),                         "cables": 1},
    "HGSRouter_cap3":              {"router_factory": lambda: HGSRouter(time_limit=0.5, seed=0),                         "cables": 3},
    "HGSRouter_cap10":             {"router_factory": lambda: HGSRouter(time_limit=0.5, seed=0),                         "cables": 10},
    "HGSRouter_feeder_limit_cap1": {"router_factory": lambda: HGSRouter(time_limit=0.5, feeder_limit=0, seed=0),         "cables": 1},
    "HGSRouter_feeder_limit_cap4": {"router_factory": lambda: HGSRouter(time_limit=0.5, feeder_limit=0, seed=0),         "cables": 4},
    "HGSRouter_feeder_limit_cap10": {"router_factory": lambda: HGSRouter(time_limit=0.5, feeder_limit=0, seed=0),        "cables": 10},

    "MILPRouter_cap10": {
        "router_factory": lambda: MILPRouter(solver_name="ortools", time_limit=5, mip_gap=0.001),
        "cables": 10,
    },
    "MILPRouter_cap10_modeloptions": {
        "router_factory": lambda: MILPRouter(
            solver_name="ortools", time_limit=5, mip_gap=0.001,
            model_options=ModelOptions(topology="radial", feeder_limit="minimum", feeder_route="straight"),
        ),
        "cables": 10,
    },

    # routers_2
    "EWRouter_cap5":               {"router_factory": lambda: None,                              "cables": 5},
    "EWRouter_cap10":              {"router_factory": lambda: None,                              "cables": 10},
    "EWRouter_cap100":             {"router_factory": lambda: None,                              "cables": 100},
    "EWRouter_straight_cap7":      {"router_factory": lambda: EWRouter(feeder_route="straight"), "cables": 7},
    "EWRouter_straight_cap15":     {"router_factory": lambda: EWRouter(feeder_route="straight"), "cables": 15},
    "EWRouter_straight_cap100":    {"router_factory": lambda: EWRouter(feeder_route="straight"), "cables": 100},
    "HGSRouter_cap5":              {"router_factory": lambda: HGSRouter(time_limit=0.5, seed=0), "cables": 5},
}


def _split_key(key: str):
    """
    Keys are f'{site_name}_{router_name}'. The router name starts with EWRouter/HGSRouter/MILPRouter.
    """
    for prefix in ("EWRouter", "HGSRouter", "MILPRouter"):
        idx = key.find(f"_{prefix}")
        if idx != -1:
            return key[:idx], key[idx + 1 :]
    raise KeyError(f"Unrecognized router suffix in key: {key!r}")


def _pbf_path_from_site(site_name: str) -> Path:
    """
    'DTU_1ss_10wt' -> tests/test_files/DTU_tests_1ss_10wt.osm.pbf
    """
    assert site_name.startswith("DTU_"), f"Unexpected site_name: {site_name!r}"
    rest = site_name[len("DTU_"):]
    return HERE / "test_files" / f"DTU_tests_{rest}.osm.pbf"


# ========== Graph tests ==========

@pytest.mark.parametrize(
    "key",
    sorted(list(dill.load(open(EXPECTED_PATH, "rb"))["RouterGraphs"].keys())) if EXPECTED_PATH.exists() else [],
)
def test_expected_router_graphs_match(key):
    if not EXPECTED_PATH.exists():
        pytest.skip(f"Expected file not found: {EXPECTED_PATH}")

    with open(EXPECTED_PATH, "rb") as f:
        expected = dill.load(f)

    expected_G = expected["RouterGraphs"][key]

    site_name, router_name = _split_key(key)

    # Resolve inputs from key
    pbf_path = _pbf_path_from_site(site_name)
    if not pbf_path.exists():
        pytest.fail(f"PBF not found: {pbf_path} (from key {key!r})")

    try:
        router_cfg = ROUTERS[router_name]
    except KeyError:
        pytest.fail(f"Unknown router_name {router_name!r} (from key {key!r}). Update ROUTERS mapping.")

    cables = router_cfg["cables"]
    router = router_cfg["router_factory"]()  # fresh instance per test

    # Skip MILP case if OR-Tools isn't installed
    if isinstance(router, MILPRouter):
        pytest.importorskip("ortools", reason="MILPRouter requires OR-Tools")

    # Build & optimize
    wfn = WindFarmNetwork.from_pbf(filepath=str(pbf_path), cables=cables)
    wfn.optimize(router=router)

    # Compare graphs; ignore volatile graph-level keys
    ignored_keys = {"solution_time", "runtime", "pool_count"} #, "method_options"}
    assert_graph_equal(wfn.G, expected_G, ignored_graph_keys=ignored_keys)
