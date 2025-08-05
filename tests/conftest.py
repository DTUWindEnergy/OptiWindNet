import pytest
import numpy as np
import pickle
from optiwindnet.interarraylib import L_from_G

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

# ========== Assertions ==========

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