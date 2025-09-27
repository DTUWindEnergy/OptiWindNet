#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate expected values (graphs) for DTU letters.

Key improvements vs the original:
- Stores *specs* (not callables/instances) for routers → robust, pickle-friendly.
- Saves Sites, Routers (specs), Cases, RouterGraphs, and Meta (env info).
- Single factory to create routers from router-specs (incl. nested ModelOptions).
- Deterministic defaults (e.g., seed=0 where applicable).
- Safer path handling (Pathlib, repo-root–relative paths).
- Clear logging and progress.
"""

from __future__ import annotations

import gc
import os
import sys
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import dill

# ---- project imports ----
from optiwindnet.MILP import ModelOptions
from optiwindnet.api import WindFarmNetwork, EWRouter, HGSRouter, MILPRouter

# ======================================================================================
# Configuration
# ======================================================================================

REPO_ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).parent.name == "scripts") else Path.cwd()
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "test_files" / "expected_DTU_letters.dill"

# Keep test files relative to repo root to avoid absolute-path drift in CI
TEST_FILES = REPO_ROOT / "tests" / "test_files"


def r_spec(cls: Optional[str], params: Optional[Dict[str, Any]] = None, cables: int = 1) -> Dict[str, Any]:
    """
    Build a pickle-friendly router spec. `cls` is a string key mapping to a class in make_router_from_spec.
    If cls is None, we interpret this as "no router" (use WFN defaults).
    """
    return {"class": cls, "params": (params or {}), "cables": int(cables)}


# --------------------------------------------------------------------------------------
# Sites & Routers (as serializable specs)
# --------------------------------------------------------------------------------------

SITES_1: Dict[str, str] = {
    "DTU_1ss_10wt": str(TEST_FILES / "DTU_tests_1ss_10wt.osm.pbf"),
}

ROUTERS_1: Dict[str, Dict[str, Any]] = {
    "EWRouter_cap1":               r_spec("EWRouter", cables=1),
    "EWRouter_cap3":               r_spec("EWRouter", cables=3),
    "EWRouter_cap10":              r_spec("EWRouter", cables=10),
    "EWRouter_straight_cap1":      r_spec("EWRouter", {"feeder_route": "straight"}, cables=1),
    "EWRouter_straight_cap4":      r_spec("EWRouter", {"feeder_route": "straight"}, cables=4),
    "EWRouter_straight_cap10":     r_spec("EWRouter", {"feeder_route": "straight"}, cables=10),
    "HGSRouter_cap1":              r_spec("HGSRouter", {"time_limit": 0.5, "seed": 0}, cables=1),
    "HGSRouter_cap3":              r_spec("HGSRouter", {"time_limit": 0.5, "seed": 0}, cables=3),
    "HGSRouter_cap10":             r_spec("HGSRouter", {"time_limit": 0.5, "seed": 0}, cables=10),
    "HGSRouter_feeder_limit_cap1": r_spec("HGSRouter", {"time_limit": 0.5, "feeder_limit": 0, "seed": 0}, cables=1),
    "HGSRouter_feeder_limit_cap4": r_spec("HGSRouter", {"time_limit": 0.5, "feeder_limit": 0, "seed": 0}, cables=4),
    "HGSRouter_feeder_limit_cap10":r_spec("HGSRouter", {"time_limit": 0.5, "feeder_limit": 0, "seed": 0}, cables=10),
    "MILPRouter_cap10":            r_spec("MILPRouter", {"solver_name": "ortools", "time_limit": 5, "mip_gap": 0.001}, cables=10),
    "MILPRouter_cap10_modeloptions": r_spec(
        "MILPRouter",
        {
            "solver_name": "ortools",
            "time_limit": 5,
            "mip_gap": 0.001,
            "model_options": {"topology": "radial", "feeder_limit": "minimum", "feeder_route": "straight"},
        },
        cables=10,
    ),
}

SITES_2: Dict[str, str] = {
    "DTU_1ss_40wt":  str(TEST_FILES / "DTU_tests_1ss_40wt.osm.pbf"),
    "DTU_2ss_40wt":  str(TEST_FILES / "DTU_tests_2ss_40wt.osm.pbf"),
    "DTU_4ss_40wt":  str(TEST_FILES / "DTU_tests_4ss_40wt.osm.pbf"),
    "DTU_1ss_100wt": str(TEST_FILES / "DTU_tests_1ss_100wt.osm.pbf"),
    "DTU_1ss_300wt": str(TEST_FILES / "DTU_tests_1ss_300wt.osm.pbf"),
}

ROUTERS_2: Dict[str, Dict[str, Any]] = {
    "EWRouter_cap5":            r_spec("EWRouter", cables=5),
    "EWRouter_cap10":           r_spec("EWRouter", cables=10),
    "EWRouter_cap100":          r_spec("EWRouter", cables=100),
    "EWRouter_straight_cap7":   r_spec("EWRouter", {"feeder_route": "straight"}, cables=7),
    "EWRouter_straight_cap15":  r_spec("EWRouter", {"feeder_route": "straight"}, cables=15),
    "EWRouter_straight_cap300": r_spec("EWRouter", {"feeder_route": "straight"}, cables=300),
    "HGSRouter_cap5":           r_spec("HGSRouter", {"time_limit": 20, "seed": 0}, cables=5),
    # uncomment/add more when needed:
    # "HGSRouter_cap10":        r_spec("HGSRouter", {"time_limit": 0.5, "seed": 0}, cables=10),
    # ...
}

# ======================================================================================
# Utilities
# ======================================================================================

def make_model_options_from_spec(spec: Dict[str, Any]) -> ModelOptions:
    """Turn a plain dict into a ModelOptions object."""
    return ModelOptions(**spec)

def make_router_from_spec(spec: Optional[Dict[str, Any]]):
    """Instantiate a router from a simple spec dict produced by r_spec()."""
    if spec is None:
        return None
    clsname = spec["class"]
    params = dict(spec.get("params", {}))

    # expand nested ModelOptions if provided as dict
    if clsname == "MILPRouter" and "model_options" in params and isinstance(params["model_options"], dict):
        params["model_options"] = make_model_options_from_spec(params["model_options"])

    if clsname is None:
        return None
    if clsname == "EWRouter":
        return EWRouter(**params)
    if clsname == "HGSRouter":
        return HGSRouter(**params)
    if clsname == "MILPRouter":
        return MILPRouter(**params)

    raise ValueError(f"Unknown router class: {clsname!r}")


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in dicts:
        out.update(d)
    return out


def print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def environment_meta() -> Dict[str, Any]:
    meta = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "package_versions": {},  # fill selectively to avoid heavy imports
    }
    # Optional: capture a few known libs if installed
    def _maybe_version(modname: str):
        try:
            mod = __import__(modname)
            ver = getattr(mod, "__version__", None)
            if ver:
                meta["package_versions"][modname] = ver
        except Exception:
            pass
    for name in ("networkx", "numpy", "ortools", "pyomo", "gurobipy"):
        _maybe_version(name)

    return meta


# ======================================================================================
# Core runner
# ======================================================================================

def generate_expected(output_path: Path) -> None:
    """
    Build networks and serialize expected results.

    Output structure:
      {
        "Sites":         {site_name: pbf_path, ...},
        "Routers":       {router_name: spec, ...},   # class/params/cables only
        "Cases":         [{"key": ..., "site": ..., "router": ...}, ...],
        "RouterGraphs":  {key: networkx.Graph, ...},
        "Meta":          {...}
      }
    """
    print_header("Generating expected DTU letters")

    # (Re)move file first to ensure a clean write
    try:
        output_path.unlink(missing_ok=True)
        print(f"🗑️  Removed (if existed): {output_path}")
    except Exception as e:
        print(f"⚠️  Error removing old file: {e}")

    sites = merge_dicts(SITES_1, SITES_2)
    routers = merge_dicts(ROUTERS_1, ROUTERS_2)

    cases: List[Dict[str, str]] = []
    router_graphs: Dict[str, Any] = {}

    # Helper to run a batch
    def run_batch(batch_sites: Dict[str, str], batch_routers: Dict[str, Dict[str, Any]], label: str) -> None:
        print_header(f"Running {label} ({len(batch_sites)} sites × {len(batch_routers)} routers)")
        for site_idx, (site_name, pbf_file) in enumerate(batch_sites.items(), 1):
            pbf_path = Path(pbf_file)
            if not pbf_path.exists():
                print(f"❌  PBF not found for site '{site_name}': {pbf_path}")
                continue

            for router_idx, (router_name, spec) in enumerate(batch_routers.items(), 1):
                key = f"{site_name}_{router_name}"
                cases.append({"key": key, "site": site_name, "router": router_name})

                # Create router instance from spec
                router = make_router_from_spec(spec)
                cables = int(spec["cables"])
                print(f"[{site_idx}/{len(batch_sites)}] [{router_idx}/{len(batch_routers)}] → {key} (cables={cables})")

                wfn = None
                try:
                    wfn = WindFarmNetwork.from_pbf(filepath=str(pbf_path), cables=cables)
                    # If MILP router requires optional deps, user may not have them installed.
                    # Let it raise; that's fine for a generator script. Alternatively, skip:
                    # if isinstance(router, MILPRouter):
                    #     try:
                    #         __import__("ortools")
                    #     except Exception:
                    #         print("  ↪ skipping MILPRouter case (ortools not installed).")
                    #         continue

                    wfn.optimize(router=router)
                    router_graphs[key] = wfn.G.copy()
                    print("  ✅ done")
                except Exception as e:
                    print(f"  ⚠️  failed: {e}")
                finally:
                    # Explicit cleanup to keep memory steady on big runs
                    try:
                        del wfn
                    except Exception:
                        pass
                    del router
                    gc.collect()

    # Run both sets
    run_batch(SITES_1, ROUTERS_1, "sites_1 × routers_1")
    run_batch(SITES_2, ROUTERS_2, "sites_2 × routers_2")

    expected = {
        "Sites": sites,
        "Routers": routers,
        "Cases": cases,
        "RouterGraphs": router_graphs,
        "Meta": environment_meta(),
    }

    # Ensure parent exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        dill.dump(expected, f, protocol=dill.HIGHEST_PROTOCOL)

    print_header("Completed")
    print(f"✅ Saved expected values to: {output_path}")
    print(f"📦 Cases stored: {len(cases)}; Graphs stored: {len(router_graphs)}")


# ======================================================================================
# CLI
# ======================================================================================

def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Optional: allow custom output path via CLI: `python script.py path/to/file.dill`
    out = Path(argv[0]).resolve() if argv else DEFAULT_OUTPUT
    generate_expected(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
