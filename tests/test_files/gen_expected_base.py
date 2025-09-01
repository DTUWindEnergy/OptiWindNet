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
    file_path = 'tests/test_files/expected_base.dill'
    try:
        os.remove(file_path)
        print(f'🗑️ Removed file: {file_path}')
    except FileNotFoundError:
        print(f'📁 File not found (so nothing removed): {file_path}')
    except Exception as e:
        print(f'⚠️ Error removing file: {e}')

    # ===============================
    # Load repository
    # ===============================
    locations = load_repository()
    L = locations.eagle

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
        'eagle': locations.eagle,
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

    print('✅ All expected values saved to:', file_path)
