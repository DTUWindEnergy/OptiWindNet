import copy

import numpy as np
import pytest

from optiwindnet.heuristics import CPEW, EW_presolver
from optiwindnet.importer import L_from_site
from optiwindnet.interarraylib import (
    assign_cables,
    G_from_S,
    S_from_G,
    as_normalized,
    calcload,
)
from optiwindnet.mesh import make_planar_embedding
from optiwindnet.MILP import ModelOptions, solver_factory
from optiwindnet.pathfinding import PathFinder
from .helpers import assert_graph_equal
# ========== Test functions ==========


def test_make_planar_embedding(expected):
    P_expected = copy.deepcopy(expected['P'])
    A_expected = copy.deepcopy(expected['A'])

    P_test, A_test = make_planar_embedding(expected['L'])

    assert_graph_equal(P_test, P_expected)

    assert set(A_test.graph['planar'].nodes) == set(A_expected.graph['planar'].nodes), (
        'PlanarEmbedding nodes mismatch'
    )
    assert set(A_test.graph['planar'].edges) == set(A_expected.graph['planar'].edges), (
        'PlanarEmbedding edges mismatch'
    )

    A_test.graph.pop('planar', None)
    A_expected.graph.pop('planar', None)

    assert_graph_equal(A_test, A_expected)


def test_as_normalized(expected):
    A_norm_test = as_normalized(expected['A'])
    A_norm_expected = expected['A_norm']
    assert_graph_equal(A_norm_test, A_norm_expected, ignored_graph_keys={'planar'})


def test_g_from_s_(expected):
    G_tentative_test = G_from_S(expected['S_ew'], expected['A'])
    G_tentative_expected = expected['G_tentative']
    assert_graph_equal(
        G_tentative_test, G_tentative_expected, ignored_graph_keys={'is_normalized'}
    )


def test_pathfinder(expected):
    G_test = PathFinder(
        expected['G_tentative'], planar=expected['P'], A=expected['A']
    ).create_detours()
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
    assert_graph_equal(S_test, S_expected, ignored_graph_keys={'runtime'})


def test_cpew(expected):
    G_test = CPEW(expected['L'], capacity=7)
    G_expected = expected['G_CPEW']
    assert_graph_equal(G_test, G_expected, ignored_graph_keys={'runtime'})


def test_l_from_site(expected):
    L_expected = expected['L']
    turbinesC_test = np.array(
        [
            L_expected.graph['VertexC'][n]
            for n, data in L_expected.nodes(data=True)
            if data['kind'] == 'wtg'
        ]
    )
    substationsC_test = np.array(
        [
            L_expected.graph['VertexC'][n]
            for n, data in L_expected.nodes(data=True)
            if data['kind'] == 'oss'
        ]
    )

    vertexC_test = L_expected.graph['VertexC']
    R_test = substationsC_test.shape[0]
    T_test = turbinesC_test.shape[0]

    L_test = L_from_site(
        R=R_test,
        T=T_test,
        B=6,
        VertexC=vertexC_test,
        name='Baltic Eagle',
        handle='eagle',
    )
    assert_graph_equal(
        L_test, L_expected, ignored_graph_keys={'border', 'OSM_name', 'landscape_angle'}
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
        actual_val = f'ERROR: {e}'

    assert actual_val == expected['SolverTypes'][solver_name]
