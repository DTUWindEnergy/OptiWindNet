"""
Generate expected graphs for specified sites-routers.
"""

import platform
import sys
from importlib import metadata, util
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import dill
from optiwindnet.api import WindFarmNetwork
from optiwindnet.importer import load_repository

import paths
from helpers import router_factory

# -----------------------
# Small helpers
# -----------------------


def merge_router_specs(
    *spec_maps: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for m in spec_maps:
        out.update(m)
    return out


def print_header(title: str) -> None:
    print('\n' + '=' * 10)
    print(title)
    print('=' * 10)


def environment_meta() -> Dict[str, Any]:
    meta = {
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'platform': platform.platform(),
        'python': sys.version.split()[0],
        'package_versions': {},
    }
    for name in ('networkx', 'numpy', 'ortools', 'pyomo', 'gurobipy', 'pyscipopt'):
        if util.find_spec(name):
            meta['package_versions'][name] = metadata.version(name)
    return meta


def generate_expected_values_end_to_end_tests():
    """
    Generate the end-to-end expected dill file.
    """
    SITES_DIR = paths.SITES_DIR

    # -----------------------
    # Local helpers / specs
    # -----------------------
    def r_spec(
        cls: Optional[str], params: Optional[Dict[str, Any]] = None, cables: int = 1
    ) -> Dict[str, Any]:
        return {'class': cls, 'params': (params or {}), 'cables': int(cables)}

    # === Sites as plain names ===
    SITES_1: Sequence[str] = ('example_location',)  # small default
    SITES_2: Sequence[str] = ('hornsea', 'london', 'taylor_2023', 'yi_2019', 'borkum2')
    SITES_3: Sequence[str] = ('hornsea',)

    model_options_strict = {
        'topology': 'radial',
        'feeder_limit': 'minimum',
        'feeder_route': 'straight',
    }

    ROUTERS_1: Dict[str, Dict[str, Any]] = {
        'EWRouter1_cap1': r_spec('EWRouter', cables=1),
        'EWRouter1_cap3': r_spec('EWRouter', cables=3),
        'EWRouter1_cap10': r_spec('EWRouter', cables=10),
        'EWRouter1_straight_cap1': r_spec(
            'EWRouter', {'feeder_route': 'straight'}, cables=1
        ),
        'EWRouter1_straight_cap4': r_spec(
            'EWRouter', {'feeder_route': 'straight'}, cables=4
        ),
        'EWRouter1_straight_cap10': r_spec(
            'EWRouter', {'feeder_route': 'straight'}, cables=10
        ),
        'HGSRouter1_cap1': r_spec(
            'HGSRouter', {'time_limit': 0.5, 'seed': 0}, cables=1
        ),
        'HGSRouter1_cap3': r_spec(
            'HGSRouter', {'time_limit': 0.5, 'seed': 0}, cables=3
        ),
        'HGSRouter1_cap10': r_spec(
            'HGSRouter', {'time_limit': 0.5, 'seed': 0}, cables=10
        ),
        'HGSRouter1_feeder_limit_cap1': r_spec(
            'HGSRouter', {'time_limit': 0.5, 'feeder_limit': 0, 'seed': 0}, cables=1
        ),
        'HGSRouter1_feeder_limit_cap4': r_spec(
            'HGSRouter', {'time_limit': 0.5, 'feeder_limit': 0, 'seed': 0}, cables=4
        ),
        'HGSRouter1_feeder_limit_cap10': r_spec(
            'HGSRouter', {'time_limit': 0.5, 'feeder_limit': 0, 'seed': 0}, cables=10
        ),
        'MILPRouter1_ortools_cap5': r_spec(
            'MILPRouter',
            {'solver_name': 'ortools', 'time_limit': 5, 'mip_gap': 1e-3},
            cables=5,
        ),
        'MILPRouter1_gurobi_cap4': r_spec(
            'MILPRouter',
            {'solver_name': 'gurobi', 'time_limit': 5, 'mip_gap': 1e-3},
            cables=4,
        ),
        'MILPRouter1_highs_cap3': r_spec(
            'MILPRouter',
            {'solver_name': 'highs', 'time_limit': 5, 'mip_gap': 1e-3},
            cables=3,
        ),
        'MILPRouter1_cplex_cap2': r_spec(
            'MILPRouter',
            {'solver_name': 'cplex', 'time_limit': 5, 'mip_gap': 1e-3},
            cables=2,
        ),
        'MILPRouter1_ortools_cap10_modeloptions': r_spec(
            'MILPRouter',
            {
                'solver_name': 'ortools',
                'time_limit': 5,
                'mip_gap': 1e-3,
                'model_options': model_options_strict,
            },
            cables=10,
        ),
        'MILPRouter1_gurobi_cap9_modeloptions': r_spec(
            'MILPRouter',
            {
                'solver_name': 'gurobi',
                'time_limit': 5,
                'mip_gap': 1e-3,
                'model_options': model_options_strict,
            },
            cables=9,
        ),
        'MILPRouter1_highs_cap8_modeloptions': r_spec(
            'MILPRouter',
            {
                'solver_name': 'highs',
                'time_limit': 5,
                'mip_gap': 1e-3,
                'model_options': model_options_strict,
            },
            cables=8,
        ),
        'MILPRouter1_cplex_cap7_modeloptions': r_spec(
            'MILPRouter',
            {
                'solver_name': 'cplex',
                'time_limit': 5,
                'mip_gap': 1e-3,
                'model_options': model_options_strict,
            },
            cables=7,
        ),
    }

    ROUTERS_2: Dict[str, Dict[str, Any]] = {
        'EWRouter2_cap1': r_spec('EWRouter', cables=1),
        'EWRouter2_cap10': r_spec('EWRouter', cables=10),
        'EWRouter2_cap100': r_spec('EWRouter', cables=100),
        'EWRouter2_straight_cap4': r_spec(
            'EWRouter', {'feeder_route': 'straight'}, cables=4
        ),
        'EWRouter2_straight_cap15': r_spec(
            'EWRouter', {'feeder_route': 'straight'}, cables=15
        ),
        'EWRouter2_straight_cap50': r_spec(
            'EWRouter', {'feeder_route': 'straight'}, cables=50
        ),
    }

    ROUTERS_3: Dict[str, Dict[str, Any]] = {
        'HGSRouter3_cap4': r_spec('HGSRouter', {'time_limit': 2, 'seed': 0}, cables=4),
    }

    # -----------------------
    # Prepare plan and output
    # -----------------------
    sites_union: List[str] = list(SITES_1) + list(SITES_2) + list(SITES_3)
    routers_union = merge_router_specs(ROUTERS_1, ROUTERS_2, ROUTERS_3)

    cases: List[Dict[str, str]] = []
    router_graphs: Dict[str, Any] = {}

    print_header('Generating expected graphs')

    # -----------------------
    # Load repository (sites)
    # -----------------------
    print('Loading repository from:', SITES_DIR)
    locations = load_repository(path=str(SITES_DIR))

    S1, S2, R1, R2 = SITES_1, SITES_2, ROUTERS_1, ROUTERS_2

    def run_batch(
        batch_sites: Sequence[str], batch_routers: Dict[str, Dict[str, Any]], label: str
    ) -> None:
        if not batch_sites or not batch_routers:
            return
        print_header(
            f'Running {label} ({len(batch_sites)} locations x {len(batch_routers)} routers)'
        )
        for si, site_name in enumerate(batch_sites, 1):
            L = getattr(locations, site_name)
            for ri, (router_name, spec) in enumerate(batch_routers.items(), 1):
                key = f'{site_name}_{router_name}'
                cases.append({'key': key, 'location': site_name, 'router': router_name})
                router = router_factory(spec)
                cables = int(spec['cables'])
                print(
                    f'[{si}/{len(batch_sites)}] [{ri}/{len(batch_routers)}]: {key} (cables={cables})'
                )

                wfn = WindFarmNetwork(L=L, cables=cables)
                if router is None:
                    wfn.optimize()
                else:
                    wfn.optimize(router=router)

                router_graphs[key] = wfn.G.copy()
                del wfn, router

    run_plan = [
        ('sites_1 x routers_1', S1, R1),
        ('sites_2 x routers_2', S2, R2),
        ('sites_3 x routers_3', SITES_3, ROUTERS_3),
    ]
    for label, s, r in run_plan:
        run_batch(s, r, label)

    expected = {
        'Locations': tuple(sites_union),
        'Routers': routers_union,
        'Cases': cases,
        'Graphs': router_graphs,
        'Meta': environment_meta(),
    }

    print_header('Completed')
    print(f'Cases generated: {len(cases)}; Number of graphs: {len(router_graphs)}')
    return expected


if __name__ == '__main__':
    print_header('Generating end_to_end expected values...')

    expected = generate_expected_values_end_to_end_tests()
    output_path = paths.END_TO_END_DILL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.unlink(missing_ok=True)
        print(f'Removed (if existed): {output_path}')
    except Exception as e:
        print(f'Error removing old file: {e}')
    with output_path.open('wb') as f:
        dill.dump(expected, f, protocol=dill.HIGHEST_PROTOCOL)

    print_header(f'Saved expected values to: {output_path}')
