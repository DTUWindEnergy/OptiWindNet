"""
Generate expected graphs for specified sites-routers.
"""

import copy
import gc
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import dill

from optiwindnet.api import EWRouter, HGSRouter, MILPRouter, WindFarmNetwork
from optiwindnet.heuristics import CPEW, EW_presolver
from optiwindnet.importer import load_repository
from optiwindnet.interarraylib import (
    G_from_S,
    S_from_G,
    as_normalized,
    assign_cables,
    calcload,
)
from optiwindnet.mesh import make_planar_embedding
from optiwindnet.MILP import ModelOptions, solver_factory
from optiwindnet.pathfinding import PathFinder

# import paths used for testing
HERE = Path(__file__).resolve()
TOP_LEVEL = HERE.parent.parent
sys.path.insert(0, str(TOP_LEVEL))
import paths


def generate_expected_values_end_to_end_tests():
    """
    Generate the end-to-end expected dill file.
    """
    SITES_DIR = paths.SITES_DIR
    DEFAULT_OUTPUT = paths.END_TO_END_DILL
    output_path = DEFAULT_OUTPUT

    # -----------------------
    # Local helpers / specs
    # -----------------------
    def r_spec(cls: Optional[str], params: Optional[Dict[str, Any]] = None, cables: int = 1) -> Dict[str, Any]:
        return {"class": cls, "params": (params or {}), "cables": int(cables)}

    # === Sites as plain names ===
    SITES_1: Sequence[str] = ("example_location",)  # small default
    SITES_2: Sequence[str] = ("hornsea", "london", "taylor_2023", "yi_2019")
    SITES_3: Sequence[str] = ("example_1ss_300wt", "example_4ss_300wt",)

    model_options_strict = {
        "topology": "radial",
        "feeder_limit": "minimum",
        "feeder_route": "straight",
    }

    ROUTERS_1: Dict[str, Dict[str, Any]] = {
        "EWRouter1_cap1": r_spec("EWRouter", cables=1),
        "EWRouter1_cap3": r_spec("EWRouter", cables=3),
        "EWRouter1_cap10": r_spec("EWRouter", cables=10),
        "EWRouter1_straight_cap1": r_spec("EWRouter", {"feeder_route": "straight"}, cables=1),
        "EWRouter1_straight_cap4": r_spec("EWRouter", {"feeder_route": "straight"}, cables=4),
        "EWRouter1_straight_cap10": r_spec("EWRouter", {"feeder_route": "straight"}, cables=10),
        "HGSRouter1_cap1": r_spec("HGSRouter", {"time_limit": 0.5, "seed": 0}, cables=1),
        "HGSRouter1_cap3": r_spec("HGSRouter", {"time_limit": 0.5, "seed": 0}, cables=3),
        "HGSRouter1_cap10": r_spec("HGSRouter", {"time_limit": 0.5, "seed": 0}, cables=10),
        "HGSRouter1_feeder_limit_cap1": r_spec("HGSRouter", {"time_limit": 0.5, "feeder_limit": 0, "seed": 0}, cables=1),
        "HGSRouter1_feeder_limit_cap4": r_spec("HGSRouter", {"time_limit": 0.5, "feeder_limit": 0, "seed": 0}, cables=4),
        "HGSRouter1_feeder_limit_cap10": r_spec("HGSRouter", {"time_limit": 0.5, "feeder_limit": 0, "seed": 0}, cables=10),
        "MILPRouter1_ortools_cap5": r_spec("MILPRouter", {"solver_name": "ortools", "time_limit": 5, "mip_gap": 1e-3}, cables=5),
        "MILPRouter1_gurobi_cap4": r_spec("MILPRouter", {"solver_name": "gurobi", "time_limit": 5, "mip_gap": 1e-3}, cables=4),
        "MILPRouter1_highs_cap3": r_spec("MILPRouter", {"solver_name": "highs", "time_limit": 5, "mip_gap": 1e-3}, cables=3),
        "MILPRouter1_cplex_cap2": r_spec("MILPRouter", {"solver_name": "cplex", "time_limit": 5, "mip_gap": 1e-3}, cables=2),
        "MILPRouter1_ortools_cap10_modeloptions": r_spec("MILPRouter", {"solver_name": "ortools", "time_limit": 5, "mip_gap": 1e-3, "model_options": model_options_strict}, cables=10),
        "MILPRouter1_gurobi_cap9_modeloptions": r_spec("MILPRouter", {"solver_name": "gurobi", "time_limit": 5, "mip_gap": 1e-3, "model_options": model_options_strict}, cables=9),
        "MILPRouter1_highs_cap8_modeloptions": r_spec("MILPRouter", {"solver_name": "highs", "time_limit": 5, "mip_gap": 1e-3, "model_options": model_options_strict}, cables=8),
        "MILPRouter1_cplex_cap7_modeloptions": r_spec("MILPRouter", {"solver_name": "cplex", "time_limit": 5, "mip_gap": 1e-3, "model_options": model_options_strict}, cables=7),
    }

    ROUTERS_2: Dict[str, Dict[str, Any]] = {
        "EWRouter2_cap1": r_spec("EWRouter", cables=1),
        "EWRouter2_cap10": r_spec("EWRouter", cables=10),
        "EWRouter2_cap100": r_spec("EWRouter", cables=100),
        "EWRouter2_straight_cap7": r_spec("EWRouter", {"feeder_route": "straight"}, cables=7),
        "EWRouter2_straight_cap15": r_spec("EWRouter", {"feeder_route": "straight"}, cables=15),
        "EWRouter2_straight_cap200": r_spec("EWRouter", {"feeder_route": "straight"}, cables=200),
        "HGSRouter2_cap2": r_spec("HGSRouter", {"time_limit": 2, "seed": 0}, cables=2),
    }

    ROUTERS_3: Dict[str, Dict[str, Any]] = {
        "EWRouter3_cap3": r_spec("EWRouter", cables=3),
        "EWRouter3_cap25": r_spec("EWRouter", cables=25),
        "EWRouter3_cap300": r_spec("EWRouter", cables=300),
        "EWRouter3_straight_cap3": r_spec("EWRouter", {"feeder_route": "straight"}, cables=3),
        "EWRouter3_straight_cap30": r_spec("EWRouter", {"feeder_route": "straight"}, cables=30),
        "EWRouter3_straight_cap300": r_spec("EWRouter", {"feeder_route": "straight"}, cables=300),}

    # -----------------------
    # Small helpers
    # -----------------------
    def make_model_options_from_spec(spec: Dict[str, Any]) -> ModelOptions:
        return ModelOptions(**spec)

    def make_router_from_spec(spec: Optional[Dict[str, Any]]):
        if spec is None:
            return None
        clsname = spec["class"]
        params = dict(spec.get("params", {}))
        if clsname == "MILPRouter" and isinstance(params.get("model_options"), dict):
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

    def merge_router_specs(*spec_maps: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for m in spec_maps:
            out.update(m)
        return out

    def print_header(title: str) -> None:
        print("\n" + "=" * 10)
        print(title)
        print("=" * 10)

    def environment_meta() -> Dict[str, Any]:
        meta = {
            "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
        }
        for name in ("networkx", "numpy", "ortools", "pyomo", "gurobipy"):
            try:
                mod = __import__(name)
                ver = getattr(mod, "__version__", None)
                if ver:
                    meta.setdefault("package_versions", {})[name] = ver
            except Exception:
                pass
        return meta

    # -----------------------
    # Prepare plan and output
    # -----------------------
    sites_union: List[str] = list(SITES_1) + list(SITES_2) + list(SITES_3)
    routers_union = merge_router_specs(ROUTERS_1, ROUTERS_2, ROUTERS_3)

    cases: List[Dict[str, str]] = []
    router_graphs: Dict[str, Any] = {}

    print_header("Generating expected graphs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.unlink(missing_ok=True)
        print(f"Removed (if existed): {output_path}")
    except Exception as e:
        print(f"Error removing old file: {e}")

    # -----------------------
    # Load repository (sites)
    # -----------------------
    print("Loading repository from:", SITES_DIR)
    locations = load_repository(path=str(SITES_DIR))


    S1, S2, R1, R2 = SITES_1, SITES_2, ROUTERS_1, ROUTERS_2

    def run_batch(batch_sites: Sequence[str], batch_routers: Dict[str, Dict[str, Any]], label: str) -> None:
        if not batch_sites or not batch_routers:
            return
        print_header(f"Running {label} ({len(batch_sites)} sites x {len(batch_routers)} routers)")
        for si, site_name in enumerate(batch_sites, 1):
            L = getattr(locations, site_name)  # no validation (matches generator contract)
            for ri, (router_name, spec) in enumerate(batch_routers.items(), 1):
                key = f"{site_name}_{router_name}"
                cases.append({"key": key, "site": site_name, "router": router_name})
                router = make_router_from_spec(spec)
                cables = int(spec["cables"])
                print(f"[{si}/{len(batch_sites)}] [{ri}/{len(batch_routers)}]: {key} (cables={cables})")

                wfn = WindFarmNetwork(L=L, cables=cables)
                if router is None:
                    wfn.optimize()
                else:
                    wfn.optimize(router=router)

                router_graphs[key] = wfn.G.copy()
                del wfn, router
                gc.collect()

    run_plan = [("sites_1 x routers_1", S1, R1), ("sites_2 x routers_2", S2, R2), ("sites_3 x routers_3", SITES_3, ROUTERS_3)]
    for label, s, r in run_plan:
        run_batch(s, r, label)

    expected = {
        "Sites": tuple(sites_union),
        "Routers": routers_union,
        "Cases": cases,
        "RouterGraphs": router_graphs,
        "Meta": environment_meta(),
    }

    with output_path.open("wb") as f:
        dill.dump(expected, f, protocol=dill.HIGHEST_PROTOCOL)

    print_header("Completed")
    print(f"Saved expected values to: {output_path}")
    print(f"Cases stored: {len(cases)}; Graphs stored: {len(router_graphs)}")

###########################
#############################
# unit tests
##########################

def generate_expected_values_unit_tests():
    # ===============================
    # Remove previous expected file
    # ===============================
    
    file_path = paths.UNITTESTS_DILL
    try:
        os.remove(file_path)
        print(f'Removed file: {file_path}')
    except FileNotFoundError:
        print(f'File not found (so nothing removed): {file_path}')
    except Exception as e:
        print(f'Error removing file: {e}')

    # ===============================
    # Load repository
    # ===============================
    data_dir = paths.SITES_DIR
    locations = load_repository(path=str(data_dir))
    L = locations.albatros

    # ===============================
    # Initialize expected dict
    # ===============================
    expected = {}
    expected['L'] = L

    # -------------------------------
    # Planar Embedding
    # -------------------------------
    P, A = make_planar_embedding(L)
    expected['P'] = copy.deepcopy(P)
    expected['A'] = copy.deepcopy(A)

    # -------------------------------
    # Normalization
    # -------------------------------
    A_norm = as_normalized(A)
    expected['A_norm'] = A_norm

    # -------------------------------
    # EW Presolver
    # -------------------------------
    S_ew = EW_presolver(A, capacity=7)
    expected['S_ew'] = S_ew

    # -------------------------------
    # G from S, then add load + cables
    # -------------------------------
    G_tentative = G_from_S(S_ew, A)
    expected['G_tentative'] = copy.deepcopy(G_tentative)
    G = PathFinder(G_tentative, planar=P, A=A).create_detours()
    expected['G'] = copy.deepcopy(G)
    expected['S_from_G'] = S_from_G(expected['G'])

    calcload(G)
    expected['G_calcload'] = copy.deepcopy(G)

    cables_assign = [(3, 1500.00), (5, 1800.0), (7, 2000.0)]
    assign_cables(G, cables_assign)
    expected['cables'] = cables_assign
    expected['G_assign_cables'] = copy.deepcopy(G)

    # -------------------------------
    # CPEW
    # -------------------------------
    G_cpew = CPEW(L, capacity=7)
    expected['G_CPEW'] = G_cpew

    # -------------------------------
    # ModelOptions
    # -------------------------------
    model_opts = ModelOptions()
    expected['ModelOptions'] = dict(model_opts)

    # -------------------------------
    # Solver types
    # -------------------------------
    solver_names = [
        'ortools',
        'cplex',
        'gurobi',
        'cbc',
        'scip',
        'highs',
        'unknown_solver',
    ]

    def safe_solver_name(name):
        try:
            s = solver_factory(name)
            return type(s).__name__ if s else None
        except ValueError as e:
            return f'ERROR: {e}'

    solver_types = {name: safe_solver_name(name) for name in solver_names}

    expected['SolverTypes'] = solver_types

    # -------------------------------
    # Extra Graphs via Routers
    # -------------------------------
    sites = {
        'albatros': locations.albatros,
        'taylor': locations.taylor_2023,
    }

    routers = {
        'EWRouter': {'router': None, 'cables': 7},
        'EWRouter_straight': {'router': EWRouter(feeder_route='straight'), 'cables': 7},
        'HGSRouter': {'router': HGSRouter(time_limit=2, seed=0), 'cables': 7,},
        'HGSRouter_feeder_limit': {
            'router': HGSRouter(time_limit=2, feeder_limit=0, seed=0),
            'cables': 7,
        },
        'MILPRouter': {
            'router': MILPRouter(solver_name='ortools', time_limit=10, mip_gap=0.005),
            'cables': 2,
        },
    }

    router_graphs = {}

    for site_name, location in sites.items():
        for router_name, config in routers.items():
            cables = config['cables']
            router = config['router']

            wfn = WindFarmNetwork(L=location, cables=cables)
            wfn.optimize(router=router)

            key = f'{site_name}_{router_name}'
            router_graphs[key] = wfn.G

    expected['RouterGraphs'] = router_graphs

    # -------------------------------
    # Save everything to dill
    # -------------------------------
    with open(file_path, 'wb') as f:
        dill.dump(expected, f)

    print('All expected values saved to:', file_path)


if __name__ == '__main__': 
    print("\n" + "=" * 50)
    print("Starting end_to_end expected values generation...")
    generate_expected_values_end_to_end_tests()

    
    print("\n" + "=" * 50)
    print("Starting unit test expected values generation...")
    generate_expected_values_unit_tests()


