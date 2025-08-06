import numpy as np
import networkx as nx
import pytest
import pickle

from optiwindnet.baselines.hgs import hgs_multiroot, iterative_hgs_cvrp
from optiwindnet.heuristics import CPEW, EW_presolver
from optiwindnet.importer import L_from_pbf, L_from_site, L_from_yaml, load_repository
from optiwindnet.interarraylib import G_from_S, S_from_G, as_normalized, calcload
from optiwindnet.interface import assign_cables
from optiwindnet.mesh import make_planar_embedding
from optiwindnet.MILP import ModelOptions, solver_factory
from optiwindnet.pathfinding import PathFinder
from optiwindnet.plotting import gplot, pplot
from optiwindnet.svg import svgplot
#from tests.assertions import assert_graph_equal
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

# ========== Test functionss ==========


with open("tests/test_files/expected_values_imported_functions.pkl", "rb") as f:
    expected = pickle.load(f)

P, A = make_planar_embedding(expected["L"])
as_normalized(A)
expected["A"] = A
expected["P"] = P

# def test_as_normalized_returns_expected_result():
#     A_norm = as_normalized(expected["A"])
#     assert_graph_equal(A_norm, expected["A_norm"])

def test_g_from_s_returns_expected_edges():
    G = G_from_S(expected["S_ew"], expected["A"])
    assert_graph_equal(G, expected["G_from_S"])


def test_s_from_g_extracts_edges():
    S = S_from_G(expected["G_from_S"])
    assert_graph_equal(S, expected["S_from_G"])


def test_calcload_assigns_loads_correctly():
    G = expected["G_from_S"]
    calcload(G)
    assert_graph_equal(G, expected["G_calcload"])


def test_assign_cables_applies_expected_cost():
    G = expected["G_calcload"]
    assign_cables(G, expected['cables'])
    assert_graph_equal(G, expected["G_assign_cables"])


def test_ew_presolver():
    S = EW_presolver(expected["A"], capacity=7)
    assert_graph_equal(S, expected["S_ew"], ignored_graph_keys={'runtime'})


def test_cpew_creates_expected_graph():
    G = CPEW(expected["L"], capacity=7)
    assert_graph_equal(G, expected["G_CPEW"], ignored_graph_keys={'runtime'})


def test_l_from_site_returns_expected_graph():
    L = expected["L"]
    turbinesC = np.array([L.graph['VertexC'][n] for n, data in L.nodes(data=True) if data['kind'] == 'wtg'])
    substationsC = np.array([L.graph['VertexC'][n] for n, data in L.nodes(data=True) if data['kind'] == 'oss'])

    vertexC = L.graph['VertexC'] #np.vstack((turbinesC, substationsC))
    R = substationsC.shape[0]
    T = turbinesC.shape[0]

    L_site = L_from_site(
        R=R, T=T, B=6,
        VertexC=vertexC,
        name="Baltic Eagle",
        handle="eagle"
    )
    assert_graph_equal(L_site, L, ignored_graph_keys={'border', 'OSM_name', 'landscape_angle'})

def test_make_planar_embedding_expected_output():
    P, A = make_planar_embedding(expected["L"])
    assert_graph_equal(P, expected["P"])
    assert_graph_equal(A, expected["A"], ignored_graph_keys={"planar"})


def test_model_options():
    opts = ModelOptions()
    for key, val in expected["ModelOptions"].items():
        assert opts[key] == val, f"Mismatch in ModelOptions[{key}]: {opts[key]} != {val}"


@pytest.mark.parametrize("solver_name", ["ortools", "cplex", "gurobi", "cbc", "scip", "highs", "unknown_solver"])
def test_solver_factory_returns_expected_solver(solver_name):
    solver = solver_factory(solver_name)
    expected_type = expected["SolverTypes"].get(solver_name)

    if expected_type is None:
        assert solver is None, f"Expected None for unsupported solver '{solver_name}'"
    else:
        assert type(solver).__name__ == expected_type, (
            f"For solver '{solver_name}', expected type '{expected_type}', "
            f"but got '{type(solver).__name__}'"
        )

