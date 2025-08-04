import pytest
import numpy as np
import pickle
from optiwindnet.api import WindFarmNetwork, EWRouter, HGSRouter, MILPRouter
from optiwindnet.db.storagev2 import G_from_routeset
from optiwindnet.interarraylib import L_from_G, L_from_site

# ========== Database Fixture ==========

@pytest.fixture(scope="module")
def db():
    with open("tests/test_files/G_tests.pkl", "rb") as f:
        db = pickle.load(f)
    yield db

# ========== Fixture Factory for G, L ==========

@pytest.fixture
def LG_from_database(db):
    def _factory(label):
        G = db[label]
        L = L_from_G(G)
        return L, G
    return _factory

# ========== Fixture Factory for Coordinate-Based Site ==========

@pytest.fixture
def site_from_database(db):
    def _factory(label):
        G = db[label]

        VertexC = G.graph['VertexC']
        T = G.graph['T']
        R = G.graph['R']
        print([VertexC[o] for o in G.graph.get('obstacles', [])])

        return {
            "turbinesC": VertexC[:T],
            "substationsC": VertexC[-R:] if R > 0 else np.empty((0, 2)),
            "borderC": VertexC[G.graph.get('border', [])] if 'border' in G.graph else np.empty((0, 2)),
            "obstaclesC": [VertexC[o] for o in G.graph.get('obstacles', [])],
            "handle": G.graph['handle'],
            "name": G.graph['name'],
            "landscape_angle": G.graph['landscape_angle'],
        }
    return _factory

# ========== Graph Assertion Helpers ==========

def assert_graph_equal(G1, G2, ignored_graph_keys=None):
    if ignored_graph_keys is None:
        ignored_graph_keys = set()

    assert set(G1.nodes) == set(G2.nodes), "Node sets differ"
    assert set(G1.edges) == set(G2.edges), "Edge sets differ"

    for n in G1.nodes:
        attrs1 = G1.nodes[n]
        attrs2 = G2.nodes[n]
        filtered1 = {k: v for k, v in attrs1.items() if k != 'label'}
        filtered2 = {k: v for k, v in attrs2.items() if k != 'label'}
        assert filtered1 == filtered2, f"Node {n} attributes differ: {filtered1} != {filtered2}"

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

# ========== Test ==========

def test_wfn_fails_without_coordinates_or_L():
    with pytest.raises(ValueError, match="Both turbinesC and substationsC must be provided! Or alternatively L should be given."):
        WindFarmNetwork(cables=7)


def test_wfn_warns_when_L_and_coordinates_given(LG_from_database, site_from_database, caplog):
    expected_L, _ = LG_from_database("eagle_EWRouter")
    site = site_from_database("eagle_EWRouter")

    with caplog.at_level("WARNING"):
        WindFarmNetwork(
            cables=7,
            turbinesC=site['turbinesC'],
            substationsC=site['substationsC'],
            L=expected_L
        )

    assert any("OptiWindNet prioritizes coordinates over L" in message for message in caplog.messages)


def test_wfn_fails_without_cables(site_from_database):
    site = site_from_database("eagle_EWRouter")
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'cables'"):
        WindFarmNetwork(
            turbinesC=site['turbinesC'],
            substationsC=site['substationsC']
        )


def test_wfn_from_coordinates_matches_expected_L(LG_from_database, site_from_database):
    expected_L, _ = LG_from_database("eagle_EWRouter")
    site = site_from_database("eagle_EWRouter")

    kwargs = {
        "cables": 7,
        "turbinesC": site["turbinesC"],
        "substationsC": site["substationsC"],
        "handle": site['handle'],
        "name": site['name'],
        "landscape_angle": site['landscape_angle'],
    }

    if site["borderC"].size > 0:
        kwargs["borderC"] = site["borderC"]

    if site["obstaclesC"]:
        kwargs["obstaclesC"] = site["obstaclesC"]

    wfn1 = WindFarmNetwork(**kwargs)

    assert_graph_equal(wfn1.L, expected_L, ignored_graph_keys={'norm_offset', 'norm_scale', 'obstacles'})
    assert isinstance(wfn1.router, EWRouter)


def test_wfn_from_L_matches_expected_L(LG_from_database):
    expected_L, _ = LG_from_database("eagle_EWRouter")

    wfn2 = WindFarmNetwork(cables=7, L=expected_L)
    assert_graph_equal(wfn2.L, expected_L, ignored_graph_keys={'norm_offset', 'norm_scale', 'obstacles'})


@pytest.mark.parametrize("cables_input, expected_array", [
    (7, [(7, 1)]),
    ((7, 100), [(7, 100)]),
    ([(5, 100), (7, 150), (9, 200)], [(5, 100), (7, 150), (9, 200)]),
])
def test_wfn_cable_formats(LG_from_database, cables_input, expected_array):
    expected_L, _ = LG_from_database("eagle_EWRouter")
    wfn = WindFarmNetwork(cables=cables_input, L=expected_L)
    assert np.array_equal(wfn.cables, expected_array)


def test_wfn_invalid_cables_raises(LG_from_database):
    expected_L, _ = LG_from_database("eagle_EWRouter")

    with pytest.raises(ValueError, match="Invalid cable values"):
        WindFarmNetwork(cables=(5, 7, 9), L=expected_L)


@pytest.mark.parametrize("label, router, ignored_keys", [
    ("eagle_EWRouter", None, {'runtime'}),
    ("eagle_EWRouter_straight", EWRouter(feeder_route='straight'), {'runtime'}),
    ("taylor_EWRouter", None, {'runtime'}),
    ("taylor_EWRouter_straight", EWRouter(feeder_route='straight'), {'runtime'}),
    ("eagle_HGSRouter", HGSRouter(time_limit=2), {'solution_time'}),
    ("eagle_HGSRouter_feeder_limit", HGSRouter(time_limit=2, feeder_limit=0), {'solution_time'}),
    ("taylor_HGSRouter", HGSRouter(time_limit=2), {'solution_time'}),
    ("taylor_HGSRouter_feeder_limit", HGSRouter(time_limit=2, feeder_limit=0), {'solution_time'}),
    ("eagle_MILPRouter", MILPRouter(solver_name='ortools', time_limit=5, mip_gap=0.005), {'runtime', 'bound', 'pool_count', 'relgap'}),
])


def test_router_variants(LG_from_database, label, router, ignored_keys):
    expected_L, G = LG_from_database(label)
    print(G.graph['capacity'])

    wfn = WindFarmNetwork(
        cables=G.graph['capacity'],
        L=expected_L,
    )

    wfn.optimize(router=router)

    ignored_keys = ignored_keys or set()
    assert_graph_equal(wfn.G, G, ignored_graph_keys=ignored_keys)

