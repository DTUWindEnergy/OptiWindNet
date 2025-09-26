if __name__ == '__main__':
    import os
    import copy
    import dill

    from optiwindnet.importer import load_repository
    from optiwindnet.interarraylib import (
        assign_cables,
        G_from_S,
        S_from_G,
        as_normalized,
        calcload,
    )
    from optiwindnet.pathfinding import PathFinder
    from optiwindnet.heuristics import EW_presolver, CPEW
    from optiwindnet.mesh import make_planar_embedding
    from optiwindnet.MILP import ModelOptions, solver_factory
    from optiwindnet.api import WindFarmNetwork, EWRouter, HGSRouter, MILPRouter

    # ===============================
    # Remove previous expected file
    # ===============================
    file_path = 'tests/test_files/expected_DTU_letters.dill'
    try:
        os.remove(file_path)
        print(f'🗑️ Removed file: {file_path}')
    except FileNotFoundError:
        print(f'📁 File not found (so nothing removed): {file_path}')
    except Exception as e:
        print(f'⚠️ Error removing file: {e}')

    #res_warmstart= wfn.optimize(router=EWRouter())

    # -------------------------------
    # Extra Graphs via Routers
    # -------------------------------
    sites_1 = {
        'DTU_1ss_10wt': 'tests/test_files/DTU_tests_1ss_10wt.osm.pbf',
    }

    routers_1 = {
        'EWRouter_cap1': {'router': None, 'cables': 1},
        'EWRouter_cap3': {'router': None, 'cables': 3},
        'EWRouter_cap10': {'router': None, 'cables': 10},
        'EWRouter_straight_cap1': {'router': EWRouter(feeder_route='straight'), 'cables': 1},
        'EWRouter_straight_cap4': {'router': EWRouter(feeder_route='straight'), 'cables': 4},
        'EWRouter_straight_cap10': {'router': EWRouter(feeder_route='straight'), 'cables': 10},
        'HGSRouter_cap1': {'router': HGSRouter(time_limit=2, seed=0), 'cables': 1},
        'HGSRouter_cap3': {'router': HGSRouter(time_limit=2, seed=0), 'cables': 3},
        'HGSRouter_cap10': {'router': HGSRouter(time_limit=2, seed=0), 'cables': 10},
        'HGSRouter_feeder_limit_cap1': {
            'router': HGSRouter(time_limit=0.5, feeder_limit=0, seed=0),
            'cables': 1,
        },
        'HGSRouter_feeder_limit_cap4': {
            'router': HGSRouter(time_limit=0.5, feeder_limit=0, seed=0),
            'cables': 4,
        },
        'HGSRouter_feeder_limit_cap10': {
            'router': HGSRouter(time_limit=0.5, feeder_limit=0, seed=0),
            'cables': 10,
        },
        'MILPRouter_cap10': {
            'router': MILPRouter(solver_name='ortools', time_limit=5, mip_gap=0.001),
            'cables': 10,
        },
        'MILPRouter_cap10_modeloptions': {
            'router': MILPRouter(solver_name='ortools', time_limit=5, mip_gap=0.001, model_options= ModelOptions(topology='radial', feeder_limit='minimum', feeder_route='straight',)),
            'cables': 10,
        },
    }

    sites_2 = {
        'DTU_1ss_40wt': 'tests/test_files/DTU_tests_1ss_40wt.osm.pbf',
        'DTU_2ss_40wt': 'tests/test_files/DTU_tests_2ss_40wt.osm.pbf',
        'DTU_4ss_40wt': 'tests/test_files/DTU_tests_4ss_40wt.osm.pbf',
        'DTU_1ss_100wt': 'tests/test_files/DTU_tests_1ss_100wt.osm.pbf',
    }

    routers_2 = {
        'EWRouter_cap5': {'router': None, 'cables': 5},
        'EWRouter_cap10': {'router': None, 'cables': 10},
        'EWRouter_cap100': {'router': None, 'cables': 100},
        'EWRouter_straight_cap7': {'router': EWRouter(feeder_route='straight'), 'cables': 7},
        'EWRouter_straight_cap15': {'router': EWRouter(feeder_route='straight'), 'cables': 15},
        'EWRouter_straight_cap100': {'router': EWRouter(feeder_route='straight'), 'cables': 100},

    }

    router_graphs = {}
    expected = {}

    for site_name, pbf_file in sites_1.items():
        for router_name, config in routers_1.items():
            cables = config['cables']
            router = config['router']

            wfn = WindFarmNetwork.from_pbf(filepath=pbf_file, cables=cables)
            wfn.optimize(router=router)

            key = f'{site_name}_{router_name}'
            router_graphs[key] = wfn.G

    for site_name, pbf_file in sites_2.items():
        for router_name, config in routers_2.items():
            cables = config['cables']
            router = config['router']

            wfn = WindFarmNetwork.from_pbf(filepath=pbf_file, cables=cables)
            wfn.optimize(router=router)

            key = f'{site_name}_{router_name}'
            router_graphs[key] = wfn.G

    expected['RouterGraphs'] = router_graphs

    # -------------------------------
    # Save everything to dill
    # -------------------------------
    with open(file_path, 'wb') as f:
        dill.dump(expected, f)

    print('✅ All expected values saved to:', file_path)
