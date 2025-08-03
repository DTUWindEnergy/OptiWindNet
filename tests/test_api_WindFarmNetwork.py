
import pytest
import numpy as np
from typing import Any
from pony.orm import db_session
from optiwindnet.api import WindFarmNetwork, EWRouter, HGSRouter, MILPRouter
from optiwindnet.db import modelv2
from optiwindnet.db.storagev2 import G_from_routeset
from optiwindnet.interarraylib import L_from_G, L_from_site

# ========== Database Fixture ==========

@pytest.fixture(scope="module")
def db():
    db = modelv2.open_database('tests/test_files/db_testing.sqlite')
    yield db
    # No explicit `db.close()` needed if Pony handles context
    del db


# ========== Factory Fixture for G, L ==========

@pytest.fixture
def LG_from_database(db):
    def _load(rs_id: int):
        with db_session:
            rs = db.RouteSet[rs_id]
            G = G_from_routeset(rs)
            L = L_from_G(G)
            G_info = {
                'id': rs.id,
                'capacity': rs.capacity,
                'detextra': rs.detextra,
                'length': rs.length,
                'creator': rs.creator,
                'gap': rs.misc.get('gap', None),
            }
            return G, L, G_info
    return _load


# ========== Factory Fixture for Coordinate-Based Site ==========

@pytest.fixture
def site_from_database(db):
    def _load(rs_id: int) -> dict[str, Any]:
        with db_session:
            rs = db.RouteSet[rs_id]
            G = G_from_routeset(rs)
            VertexC = G.graph['VertexC']
            T = G.graph['T']
            R = G.graph['R']

            return {
                "turbinesC": VertexC[:T],
                "substationsC": VertexC[-R:] if R > 0 else np.empty((0, 2)),
                "borderC": VertexC[G.graph.get('border', [])] if 'border' in G.graph else np.empty((0, 2)),
                "obstaclesC": [VertexC[o] for o in G.graph.get('obstacles', [])],
                "name": G.graph.get("name", f"routeset_{rs_id}"),
                "handle": G.graph.get("handle", f"rs_{rs_id}"),
            }
    return _load


# ========== Graph Assertion Helpers ==========

def assert_graph_equal(G1, G2, ignored_graph_keys={'label', 'landscape_angle', 'obstacles'}):
    # Structural checks
    assert set(G1.nodes) == set(G2.nodes), "Node sets differ"
    assert set(G1.edges) == set(G2.edges), "Edge sets differ"

    # Node attribute checks
    for n in G1.nodes:
        attrs1 = G1.nodes[n]
        attrs2 = G2.nodes[n]
        filtered1 = {k: v for k, v in attrs1.items() if k != 'label'}
        filtered2 = {k: v for k, v in attrs2.items() if k != 'label'}
        assert filtered1 == filtered2, f"Node {n} attributes differ: {filtered1} != {filtered2}"

    # Graph-level attribute comparison (filtered)
    keys1 = set(G1.graph.keys()) - ignored_graph_keys
    keys2 = set(G2.graph.keys()) - ignored_graph_keys
    assert keys1 == keys2, f"Graph keys mismatch: {keys1.symmetric_difference(keys2)}"

    for k in keys1:
        v1 = G1.graph[k]
        v2 = G2.graph[k]
        if isinstance(v1, np.ndarray):
            assert np.array_equal(v1, v2), f"Mismatch in graph['{k}']"
        elif isinstance(v1, list):
            assert isinstance(v2, list) and len(v1) == len(v2), f"Mismatch in list length for graph['{k}']"
            for a, b in zip(v1, v2):
                if isinstance(a, np.ndarray):
                    assert np.array_equal(a, b), f"Mismatch in list of arrays in graph['{k}']"
                else:
                    assert a == b, f"Mismatch in list values in graph['{k}']"
        else:
            assert v1 == v2, f"Mismatch in graph['{k}']: {v1} != {v2}"


def test_wfn_initialization(LG_from_database, site_from_database, caplog):

    rs_id = 10

    _, expected_L, G_info = LG_from_database(rs_id)
    site = site_from_database(rs_id)

    # Test that missing both L and coordinates raises error
    with pytest.raises(ValueError, match="Both turbinesC and substationsC must be provided! Or alternatively L should be given."):
        WindFarmNetwork(cables=5)

    with caplog.at_level("WARNING"):
        WindFarmNetwork(
            cables=[(10, 100)],
            turbinesC=site['turbinesC'],
            substationsC=site['substationsC'],
            L=expected_L
        )

    assert any("OptiWindNet prioritizes coordinates over L" in message for message in caplog.messages)
    
    # Test that missing cables raises error
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'cables'"):
        WindFarmNetwork()  # missing cables

    wfn1 = WindFarmNetwork(
        cables=[(G_info['capacity'], 100)],  # simplify cable input for test
        turbinesC=site['turbinesC'],
        substationsC=site['substationsC'],
        borderC=site['borderC'],
        obstaclesC=site['obstaclesC'],
        name=site['name'],
        handle=site['handle']
    )

    assert_graph_equal(wfn1.L, expected_L)
    assert isinstance(wfn1.router, EWRouter)

    # test for from L
    wfn2 = WindFarmNetwork(
        cables=[G_info['capacity']],  # simplify cable input for test
        L=expected_L,
    )

    assert_graph_equal(wfn2.L, expected_L)

    # test cables formatting
    wfn = WindFarmNetwork(
        cables=7,  # simplify cable input for test
        L=expected_L,
    )
    assert np.array_equal(wfn.cables, [(7, 1)])

    wfn = WindFarmNetwork(
    cables=(7, 100),  # simplify cable input for test
    L=expected_L,
    )
    assert np.array_equal(wfn.cables, [(7, 100)])

    wfn = WindFarmNetwork(
        cables=[(5, 100), (7, 150), (9, 200)] ,  # simplify cable input for test
        L=expected_L,
        )
    assert np.array_equal(wfn.cables, [(5, 100), (7, 150), (9, 200)])

    
    with pytest.raises(ValueError, match="Invalid cable values"):
        WindFarmNetwork(cables=(5, 7, 9), L=expected_L)

def test_hgs_router(LG_from_database, site_from_database, caplog):

    rs_id = 3617

    expected_G, expected_L, G_info = LG_from_database(rs_id)

    wfn = WindFarmNetwork(
        cables=G_info['capacity'] ,  # simplify cable input for test
        L=expected_L,
        )
    
    wfn.optimize(router=HGSRouter(time_limit=1))

    assert_graph_equal(wfn.G, expected_G)

def test_milp_router(LG_from_database, site_from_database, caplog):

    rs_id = 6

    expected_G, expected_L, G_info = LG_from_database(rs_id)

    wfn = WindFarmNetwork(
        cables=G_info['capacity'] ,  # simplify cable input for test
        L=expected_L,
        )
    
    wfn.optimize(router=MILPRouter(solver_name='cplex', time_limit=20, mip_gap=0.005))

    assert_graph_equal(wfn.G, expected_G)


    #############################################
    # TO ADD
    # wfn = WindFarmNetwork.from_yaml()
    # wfn = WindFarmNetwork.from_pbf()
    # wfn = WindFarmNetwork.from_windIO()
    ############################################


# def test_optimize_returns_valid_terse_links(simple_network):
#     terse = simple_network.optimize()
#     T = simple_network.L.graph['T']

#     assert isinstance(terse, np.ndarray)
#     assert terse.shape == (T,)
#     assert np.issubdtype(terse.dtype, np.integer)
#     assert (terse != np.arange(T)).all()  # No self-loops

# def test_cost_and_length(simple_network):
#     simple_network.optimize()
#     cost = simple_network.cost()
#     length = simple_network.length()

#     assert isinstance(cost, (int, float))
#     assert isinstance(length, (int, float))
#     assert cost >= 0
#     assert length >= 0

# def test_get_network(simple_network):
#     simple_network.optimize()
#     net = simple_network.get_network()

#     assert isinstance(net, np.ndarray)
#     assert net.dtype.names is not None  # structured array expected
#     assert {'src', 'tgt', 'length', 'load', 'reverse', 'cable', 'cost'}.issubset(net.dtype.names)

# def test_gradient_outputs(simple_network):
#     simple_network.optimize()
#     grad_t, grad_s = simple_network.gradient()

#     assert isinstance(grad_t, np.ndarray)
#     assert isinstance(grad_s, np.ndarray)
#     assert grad_t.shape[1] == 2
#     assert grad_s.shape[1] == 2
#     assert np.isfinite(grad_t).all()
#     assert np.isfinite(grad_s).all()

# def test_gradient_type_variants(simple_network):
#     simple_network.optimize()

#     for kind in ['length', 'cost']:
#         grad_t, grad_s = simple_network.gradient(gradient_type=kind)
#         assert grad_t.shape[1] == 2

#     with pytest.raises(ValueError):
#         simple_network.gradient(gradient_type='invalid')

# def test_update_from_terse_links(simple_network):
#     terse = simple_network.optimize()
#     new_G = simple_network.update_from_terse_links(terse)

#     assert new_G is not None
#     assert isinstance(new_G.graph['VertexC'], np.ndarray)

# def test_terse_links_correctness(simple_network):
#     terse = simple_network.optimize()
#     T = simple_network.L.graph['T']

#     assert terse.shape == (T,)
#     assert all(isinstance(x, (int, np.integer)) for x in terse)

# def test_map_detour_vertex(simple_network):
#     simple_network.optimize()
#     mapping = simple_network.map_detour_vertex()

#     if mapping:  # Only test if detours exist
#         assert isinstance(mapping, dict)
#         for k, v in mapping.items():
#             assert isinstance(k, int)
#             assert isinstance(v, int)

# def test_plot_methods_do_not_crash(simple_network):
#     simple_network.optimize()

#     # These should render but not return data (no assert needed)
#     simple_network.plot()
#     simple_network.plot_location()
#     simple_network.plot_available_links()
#     simple_network.plot_selected_links()
#     simple_network.plot_navigation_mesh()
