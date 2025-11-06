# tests/test_repair.py
import networkx as nx
import numpy as np
import pytest

from optiwindnet.api import WindFarmNetwork
import optiwindnet.repair as repair
from optiwindnet.interarraylib import calcload
from .helpers import assert_graph_equal


# =========================
# Fixtures & tiny helpers
# =========================

@pytest.fixture(scope="module")
def LA():
    """Small deterministic site: 1 substation + 8 turbines on a grid."""
    substationsC = np.array([[0.0, 0.0]])
    turbinesC = np.array(
        [
            [0, 1],
            [0, 2],
            [1, 2],
            [2, 2],
            [2, 1],
            [2, 0],
            [1, 0],
            [1, 1],
        ],
        dtype=float,
    )
    wfn = WindFarmNetwork(turbinesC=turbinesC, substationsC=substationsC, cables=5)
    return wfn.L, wfn.A  # (L has nodes/attrs incl. -1; A is available links)


def create_S_from_edges(L: nx.Graph, edges: list[tuple[int, int]]) -> nx.Graph:
    """
    Build S using L's node set & attributes, then add the given undirected edges.
    """
    S = nx.Graph()
    S.graph.update(R=L.graph["R"], T=L.graph["T"])
    S.add_nodes_from(L.nodes(data=True))  # keep -1 root and node attrs
    S.add_edges_from(edges)
    calcload(S)  # populate 'load'/'subtree' used by repair
    return S


def assert_repair(L, A, edges_in, edges_expected, ignored_graph_keys):
    S_in = create_S_from_edges(L, edges_in)
    S_expected = create_S_from_edges(L, edges_expected)
    S_out = repair.repair_routeset_path(S_in, A)
    assert_graph_equal(S_out, S_expected, ignored_graph_keys=ignored_graph_keys)


# =========================
# Parametrized tests
# =========================

@pytest.mark.parametrize(
    "edges_in, edges_expected",
    [
        # Case 1: repairable crossing → rewires to expected minimal edit
        (
            [(-1, 0), (0, 1), (1, 7), (7, 5),
             (-1, 6), (6, 4), (4, 3), (3, 2)],
            [(0, -1), (0, 1), (1, 7), (2, 7),
             (3, 4), (4, 5), (5, 6), (6, -1)],
        ),
        # Case 2: unrepairable → graph unchanged
        (
            [(-1, 0), (0, 7), (7, 3), (3, 4),
             (-1, 6), (6, 5), (5, 2), (2, 1)],
            [(-1, 0), (0, 7), (7, 3), (3, 4),
             (-1, 6), (6, 5), (5, 2), (2, 1)],
        ),
    ],
)
def test_repair_routeset_path(LA, edges_in, edges_expected):
    """
    Case 1 should be repaired to edges_expected.
    Case 2 should remain unchanged (unrepairable for current version of repair function).
    """
    L, A = LA
    ignored_graph_keys = {"solution_time", "runtime", "repaired"}
    assert_repair(L, A, edges_in, edges_expected, ignored_graph_keys)
