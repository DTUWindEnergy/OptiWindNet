import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")


import numba  # make sure the runtime flag is also set
numba.config.DISABLE_JIT = True

import pytest
import numpy as np
import dill
from optiwindnet.interarraylib import L_from_G

from .helpers import assert_graph_equal


# ========== Core Fixtures ==========

@pytest.fixture
def expected():
    """Loads all expected values from the dill file for single test use."""
    with open("tests/test_files/expected_base.dill", "rb") as f:
        return dill.load(f)

@pytest.fixture(scope="module")
def db():
    """Module-scoped database fixture for shared access across tests."""
    with open("tests/test_files/expected_base.dill", "rb") as f:
        data = dill.load(f)
    yield data["RouterGraphs"]

# ========== Factory Fixtures ==========

@pytest.fixture
def LG_from_database(db):
    """Factory that returns (L, G) pair reconstructed from a saved graph."""
    def _factory(label):
        G = db[label]
        L = L_from_G(G)
        return L, G
    return _factory

@pytest.fixture
def site_from_database(db):
    """Factory that extracts coordinate-based site components from a graph."""
    def _factory(label):
        G = db[label]
        VertexC = G.graph['VertexC']
        T = G.graph['T']
        R = G.graph['R']

        return {
            "turbinesC": VertexC[:T],
            "substationsC": VertexC[-R:] if R > 0 else np.empty((0, 2)),
            "borderC": VertexC[G.graph.get('border', [])] if 'border' in G.graph else np.empty((0, 2)),
            "obstaclesC": [VertexC[o] for o in G.graph.get('obstacles', [])],
            "handle": G.graph.get('handle'),
            "name": G.graph.get('name'),
            "landscape_angle": G.graph.get('landscape_angle'),
        }
    return _factory

