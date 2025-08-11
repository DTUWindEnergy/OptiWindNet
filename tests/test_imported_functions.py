import copy
import dill
import pickle

import networkx as nx
import numpy as np
import pytest

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

# ========== Assertions ==========

def assert_graph_equal(G1, G2, ignored_graph_keys=None):
    if ignored_graph_keys is None:
        ignored_graph_keys = set()

    ignored_graph_keys.update({"fun_fingerprint.funfile"})

    assert set(G1.nodes) == set(G2.nodes), 'Node sets differ'
    assert set(G1.edges) == set(G2.edges), 'Edge sets differ'

    for n in G1.nodes:
        attrs1 = G1.nodes[n]
        attrs2 = G2.nodes[n]
        filtered1 = {k: v for k, v in attrs1.items() if k != 'label'}
        filtered2 = {k: v for k, v in attrs2.items() if k != 'label'}
        assert filtered1 == filtered2, (
            f'Node {n} attributes differ: {filtered1} != {filtered2}'
        )

    keys1 = set(G1.graph.keys()) - ignored_graph_keys
    keys2 = set(G2.graph.keys()) - ignored_graph_keys
    assert keys1 == keys2, f'Graph keys mismatch: {keys1.symmetric_difference(keys2)}'

    for k in keys1:
        v1 = G1.graph[k]
        v2 = G2.graph[k]
        if isinstance(v1, np.ndarray):
            assert np.array_equal(v1, v2), f"Mismatch in graph['{k}']"
        elif isinstance(v1, list):
            assert isinstance(v2, list) and len(v1) == len(v2), (
                f"Mismatch in list length for graph['{k}']"
            )
            for a, b in zip(v1, v2):
                if isinstance(a, np.ndarray):
                    assert np.array_equal(a, b), (
                        f"Mismatch in list of arrays in graph['{k}']"
                    )
                else:
                    assert a == b, f"Mismatch in list values in graph['{k}']"
        else:
            assert v1 == v2, f"Mismatch in graph['{k}']: {v1} != {v2}"


# ========== Test functions ==========

def test_make_planar_embedding(expected):
    P_expected = copy.deepcopy(expected['P'])
    A_expected = copy.deepcopy(expected['A'])

    P_test, A_test = make_planar_embedding(expected['L'])

    assert_graph_equal(P_test, P_expected)

    assert set(A_test.graph['planar'].nodes) == set(A_expected.graph['planar'].nodes), \
        "PlanarEmbedding nodes mismatch"
    assert set(A_test.graph['planar'].edges) == set(A_expected.graph['planar'].edges), \
        "PlanarEmbedding edges mismatch"

    A_test.graph.pop('planar', None)
    A_expected.graph.pop('planar', None)

    assert_graph_equal(A_test, A_expected)


def test_as_normalized(expected):
    A_norm_test = as_normalized(expected['A'])
    A_norm_expected = expected["A_norm"]
    assert_graph_equal(A_norm_test, A_norm_expected, ignored_graph_keys={'planar'} )


def test_g_from_s_(expected):
    G_tentative_test = G_from_S(expected['S_ew'], expected['A'])
    G_tentative_expected = expected['G_tentative']
    assert_graph_equal(G_tentative_test, G_tentative_expected, ignored_graph_keys={'is_normalized'})

def test_pathfinder(expected):
    G_test = PathFinder(expected['G_tentative'], planar=expected['P'], A=expected['A']).create_detours()
    G_expected = expected['G']
    assert_graph_equal(G_test, G_expected)


def test_s_from_g(expected):
    S_test = S_from_G(expected['G'])
    S_expected = expected['S_from_G']
    assert_graph_equal(S_test, S_expected)


def test_calcload(expected):
    G_test = expected['G']
    calcload(G_test)
    G_expected = expected['G_calcload']
    assert_graph_equal(G_test, G_expected)


def test_assign_cables(expected):
    G_test = expected['G_calcload']
    assign_cables(G_test, expected['cables'])
    G_expected = expected['G_assign_cables']
    assert_graph_equal(G_test, G_expected)


def test_ew_presolver(expected):
    S_test = EW_presolver(expected['A'], capacity=7)
    S_expected = expected['S_ew']
    assert_graph_equal(S_test, S_expected, ignored_graph_keys={'runtime', "fun_fingerprint.funfile"})


def test_cpew(expected):
    G_test = CPEW(expected['L'], capacity=7)
    G_expected = expected['G_CPEW']
    assert_graph_equal(G_test, G_expected, ignored_graph_keys={'runtime', "fun_fingerprint.funfile"})


def test_l_from_site(expected):
    L_expected = expected['L']
    turbinesC_test = np.array([
        L_expected.graph['VertexC'][n]
        for n, data in L_expected.nodes(data=True)
        if data['kind'] == 'wtg'
    ])
    substationsC_test = np.array([
        L_expected.graph['VertexC'][n]
        for n, data in L_expected.nodes(data=True)
        if data['kind'] == 'oss'
    ])

    vertexC_test = L_expected.graph['VertexC']
    R_test = substationsC_test.shape[0]
    T_test = turbinesC_test.shape[0]

    L_test = L_from_site(
        R=R_test, T=T_test, B=6, VertexC=vertexC_test, name='Baltic Eagle', handle='eagle'
    )
    assert_graph_equal(
        L_test, L_expected,
        ignored_graph_keys={'border', 'OSM_name', 'landscape_angle'}
    )


def test_model_options(expected):
    options_test = ModelOptions()
    for key, value_expected in expected['ModelOptions'].items():
        assert options_test[key] == value_expected, (
            f'Mismatch in ModelOptions[{key}]: {options_test[key]} != {value_expected}'
        )


solver_names = ['ortools', 'cplex', 'gurobi', 'cbc', 'scip', 'highs', 'unknown_solver']

@pytest.mark.parametrize('solver_name', solver_names)
def test_solver_factory_returns_expected_solver(solver_name, expected):
    try:
        s = solver_factory(solver_name)
        actual_val = type(s).__name__ if s else None
    except ValueError as e:
        actual_val = f"ERROR: {e}"

    assert actual_val == expected['SolverTypes'][solver_name]
