if __name__ == '__main__':
    import os
    import dill
    import gc

    from optiwindnet.MILP import ModelOptions
    from optiwindnet.api import WindFarmNetwork, EWRouter, HGSRouter, MILPRouter

    file_path = 'tests/test_files/expected_DTU_letters.dill'
    try:
        os.remove(file_path)
        print(f'🗑️ Removed file: {file_path}')
    except FileNotFoundError:
        print(f'📁 File not found (so nothing removed): {file_path}')
    except Exception as e:
        print(f'⚠️ Error removing file: {e}')

    sites_1 = {
        'DTU_1ss_10wt': 'tests/test_files/DTU_tests_1ss_10wt.osm.pbf',
    }

    # Use factories so each iteration gets a brand-new router object
    routers_1 = {
        'EWRouter_cap1':               {'router_factory': lambda: None,                              'cables': 1},
        'EWRouter_cap3':               {'router_factory': lambda: None,                              'cables': 3},
        'EWRouter_cap10':              {'router_factory': lambda: None,                              'cables': 10},
        'EWRouter_straight_cap1':      {'router_factory': lambda: EWRouter(feeder_route='straight'), 'cables': 1},
        'EWRouter_straight_cap4':      {'router_factory': lambda: EWRouter(feeder_route='straight'), 'cables': 4},
        'EWRouter_straight_cap10':     {'router_factory': lambda: EWRouter(feeder_route='straight'), 'cables': 10},
        'HGSRouter_cap1':              {'router_factory': lambda: HGSRouter(time_limit=0.5, seed=0), 'cables': 1},
        'HGSRouter_cap3':              {'router_factory': lambda: HGSRouter(time_limit=0.5, seed=0), 'cables': 3},
        'HGSRouter_cap10':             {'router_factory': lambda: HGSRouter(time_limit=0.5, seed=0), 'cables': 10},
        'HGSRouter_feeder_limit_cap1': {'router_factory': lambda: HGSRouter(time_limit=0.5, feeder_limit=0, seed=0), 'cables': 1},
        'HGSRouter_feeder_limit_cap4': {'router_factory': lambda: HGSRouter(time_limit=0.5, feeder_limit=0, seed=0), 'cables': 4},
        'HGSRouter_feeder_limit_cap10':{'router_factory': lambda: HGSRouter(time_limit=0.5, feeder_limit=0, seed=0), 'cables': 10},
        'MILPRouter_cap10':            {'router_factory': lambda: MILPRouter(solver_name='ortools', time_limit=5, mip_gap=0.001), 'cables': 10},
        'MILPRouter_cap10_modeloptions': {
            'router_factory': lambda: MILPRouter(
                solver_name='ortools', time_limit=5, mip_gap=0.001,
                model_options=ModelOptions(topology='radial', feeder_limit='minimum', feeder_route='straight')
            ),
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
        'EWRouter_cap5':               {'router_factory': lambda: None,                              'cables': 5},
        'EWRouter_cap10':              {'router_factory': lambda: None,                              'cables': 10},
        'EWRouter_cap100':             {'router_factory': lambda: None,                              'cables': 100},
        'EWRouter_straight_cap7':      {'router_factory': lambda: EWRouter(feeder_route='straight'), 'cables': 7},
        'EWRouter_straight_cap15':     {'router_factory': lambda: EWRouter(feeder_route='straight'), 'cables': 15},
        'EWRouter_straight_cap100':    {'router_factory': lambda: EWRouter(feeder_route='straight'), 'cables': 100},
        'HGSRouter_cap5':              {'router_factory': lambda: HGSRouter(time_limit=0.5, seed=0), 'cables': 5},
        #'HGSRouter_cap10':             {'router_factory': lambda: HGSRouter(time_limit=0.5, seed=0), 'cables': 10},
        # 'HGSRouter_cap100':            {'router_factory': lambda: HGSRouter(time_limit=0.5, seed=0), 'cables': 100},
        # 'HGSRouter_feeder_limit_cap7': {'router_factory': lambda: HGSRouter(time_limit=0.5, feeder_limit=0, seed=0), 'cables': 7},
        # 'HGSRouter_feeder_limit_cap15':{'router_factory': lambda: HGSRouter(time_limit=0.5, feeder_limit=0, seed=0), 'cables': 15},
        # 'HGSRouter_feeder_limit_cap100':{'router_factory': lambda: HGSRouter(time_limit=0.5, feeder_limit=0, seed=0), 'cables': 100},
    }

    router_graphs = {}
    expected = {}

    # --- run sites_1 ---
    for site_name, pbf_file in sites_1.items():
        for router_name, config in routers_1.items():
            cables = config['cables']
            router = config['router_factory']()

            try:
                wfn = WindFarmNetwork.from_pbf(filepath=pbf_file, cables=cables)
                wfn.optimize(router=router)
                # copy the graph so it doesn't reference wfn internals
                G_copy = wfn.G.copy()  # networkx copy
                key = f'{site_name}_{router_name}'
                router_graphs[key] = G_copy
            finally:
                # drop big objects ASAP
                try:
                    del wfn
                except NameError:
                    pass
                del router
                gc.collect()

    # --- run sites_2 ---
    for site_name, pbf_file in sites_2.items():
        for router_name, config in routers_2.items():
            cables = config['cables']
            router = config['router_factory']()

            try:
                wfn = WindFarmNetwork.from_pbf(filepath=pbf_file, cables=cables)
                wfn.optimize(router=router)
                G_copy = wfn.G.copy()
                key = f'{site_name}_{router_name}'
                router_graphs[key] = G_copy
            finally:
                try:
                    del wfn
                except NameError:
                    pass
                del router
                gc.collect()

    expected['RouterGraphs'] = router_graphs

    with open(file_path, 'wb') as f:
        dill.dump(expected, f)

    print('✅ All expected values saved to:', file_path)
