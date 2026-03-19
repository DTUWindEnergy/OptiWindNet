"""
Generate expected graphs for specified sites-routers.
"""

import pickle
from typing import Any, Dict, Optional, Sequence

from optiwindnet.api import WindFarmNetwork
from optiwindnet.importer import L_from_yaml

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


def generate_expected_values_end_to_end_tests():
    """
    Generate the end-to-end expected instances file.
    """
    LOCATIONS_DIR = paths.LOCATIONS_DIR

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
    routers_union = merge_router_specs(ROUTERS_1, ROUTERS_2, ROUTERS_3)

    cases: list[Dict[str, str]] = []
    router_graphs: Dict[str, Any] = {}

    print_header('Generating expected graphs')

    # -----------------------
    # Load locations by explicit file name
    # -----------------------
    from collections import namedtuple

    data_dir = paths.DATA_DIR
    location_files = {
        'hornsea': (L_from_yaml, data_dir / 'Hornsea One.yaml'),
        'london': (L_from_yaml, data_dir / 'London Array.yaml'),
        'taylor_2023': (L_from_yaml, data_dir / 'Taylor-2023.yaml'),
        'yi_2019': (L_from_yaml, data_dir / 'Yi-2019.yaml'),
        'borkum2': (L_from_yaml, data_dir / 'Borkum Riffgrund 2.yaml'),
        'example_location': (L_from_yaml, LOCATIONS_DIR / 'example_location.yaml'),
    }
    print('Loading locations:', ', '.join(location_files.keys()))
    loaded = {handle: loader(path) for handle, (loader, path) in location_files.items()}
    Locations = namedtuple('Locations', loaded.keys())
    locations = Locations(**loaded)

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

                router_graphs[key] = tuple(wfn.terse_links().tolist())
                del wfn, router

    run_plan = [
        ('sites_1 x routers_1', S1, R1),
        ('sites_2 x routers_2', S2, R2),
        ('sites_3 x routers_3', SITES_3, ROUTERS_3),
    ]
    for label, s, r in run_plan:
        run_batch(s, r, label)

    print_header('Completed')
    print(f'Cases generated: {len(cases)}; Number of graphs: {len(router_graphs)}')

    # Build per-instance dicts keyed by case key
    instances = {}
    for case in cases:
        key = case['key']
        instances[key] = {
            'location': case['location'],
            'router_spec': routers_union[case['router']],
            'terse_links': router_graphs[key],
        }
    return instances


if __name__ == '__main__':
    print_header('Generating end_to_end expected values...')

    instances = generate_expected_values_end_to_end_tests()
    output_path = paths.SOLUTIONS_FILE
    with output_path.open('wb') as f:
        pickle.dump(instances, f)

    print_header(f'Saved {len(instances)} instances to: {output_path}')
