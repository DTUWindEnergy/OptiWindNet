# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import math

import networkx as nx
import numpy as np
import pytest

from optiwindnet.interarraylib import (
    add_link_blockmap,
    add_link_cosines,
    add_terminal_closest_root,
    assign_cables,
    count_diagonals,
    describe_G,
    make_remap,
    pathdist,
    scaffolded,
    update_lengths,
)

from .helpers import tiny_wfn

# ----------
# tests
# ----------


def test_assign_cables():
    # use tiny_wfn
    wfn = tiny_wfn()
    original_G = wfn.G

    # 1) check defaults
    G = original_G.copy()
    cables1 = [(1, 100.0), (2, 150.0), (4, 200.0)]
    assign_cables(G, cables1)

    # graph-level checks
    assert G.graph['cables'] == cables1
    assert G.graph['currency'] == '€'
    assert G.graph['capacity'] == 4

    wfn2 = tiny_wfn(cables=1)
    G2 = wfn2.G

    # 1) check defaults
    cables2 = [(1, 0.0)]
    assign_cables(G2, cables2)

    # graph-level checks
    assert G2.graph['cables'] == cables2
    # assert G2.graph['currency'] == '€'
    assert G2.graph['capacity'] == 1

    def compare_cable_and_cost(G, edges_expected):
        # Optional: check lengths match
        assert len(edges_expected) == G.number_of_edges(), 'Number of edges mismatch'

        # Iterate pairwise
        for u, v, expectedD in edges_expected:
            actualD = G[u][v]

            # Check cable type
            assert expectedD['cable'] == actualD['cable'], (
                f'Edge {u, v} cable mismatch:'
                f' {expectedD["cable"]} != {actualD["cable"]}'
            )

            # Check cost (approximate)
            assert math.isclose(
                expectedD['cost'], actualD['cost'], rel_tol=1e-7, abs_tol=1e-9
            ), f'Edge {u, v} cost mismatch: {expectedD["cost"]} != {actualD["cost"]}'

    expected_1 = [
        (0, 12, {'cable': 2, 'cost': 107.70329614269008}),
        (-1, 0, {'cable': 2, 'cost': 200.0}),
        (1, 13, {'cable': 2, 'cost': 141.4213562373095}),
        (1, 2, {'cable': 1, 'cost': 150.0}),
        (2, 3, {'cable': 0, 'cost': 200.0}),
        (12, 13, {'cable': 2, 'cost': 60.0}),
    ]

    compare_cable_and_cost(G, expected_1)

    # 2) Assign again with a different cable set: currency should update, cables update
    cables2 = [(10, 1000.0), (20, 1500.0), (30, 2000.0)]

    assign_cables(G, cables2, currency='Any Currency')

    assert G.graph['cables'] == cables2
    assert G.graph['currency'] == 'Any Currency'

    # 3) All-zero-costs case:
    G_zero_cost = original_G.copy()
    cables3 = [(1, 0.0), (4, 0.0)]
    assign_cables(G_zero_cost, cables3, currency='IgnoredCurrency')
    assert G_zero_cost.graph['cables'] == cables3
    # since all costs zero, no cost value
    assert 'cost' not in G_zero_cost.graph

    # 4) Error case: raise ValueError when G.graph['max_load'] > max_capacity
    G4 = original_G.copy()
    small_cables = [(1, 10.0), (2, 20.0)]
    with pytest.raises(ValueError):
        assign_cables(G4, small_cables)

    # 5) test without capacity
    G4.graph.pop('capacity', None)
    cables4 = [(1, 100.0), (2, 150.0), (5, 200.0)]
    assign_cables(G4, cables4)
    assert G4.graph['capacity'] == 5


def test_assign_cables_prices_ring_zero_load_link():
    G = nx.Graph(max_load=1)
    G.add_edge(-1, 0, load=1, length=2.0)
    G.add_edge(0, 1, load=0, length=3.0)

    assign_cables(G, [(1, 4.0)])

    assert G[0][1]['cable'] == 0
    assert G[0][1]['cost'] == 12.0


def test_describe_G():
    wfn = tiny_wfn()
    G = wfn.G

    desc = describe_G(G)
    expected = ['κ = 4, T = 4', '(+0) [-1]: 1', 'Σλ = 5.5456\u00a0m', '55\u00a0€']

    assert desc == expected, f'Output mismatch:\nGot: {desc}\nExpected: {expected}'


def test_scaffolded():
    wfn = tiny_wfn()
    G = wfn.G.copy()
    P = wfn.P.copy()
    scaff = scaffolded(G, P)

    # Check that returned graph is undirected
    assert not scaff.is_directed()

    # All nodes from G should be in scaff
    for n in G.nodes():
        assert n in scaff.nodes()
        for k, v in G.nodes[n].items():
            assert scaff.nodes[n][k] == v

    # fnT should exist and match expected length: G's primes, P's 3
    # supertriangle placeholders, and R roots (scaff never gains G's clone
    # nodes, since P has no clone concept and those ids alias supertriangle)
    assert 'fnT' in scaff.graph
    T, B, R = (G.graph[k] for k in 'TBR')
    assert len(scaff.graph['fnT']) == T + B + 3 + R

    # VertexC should contain G's VertexC plus P's supertriangle
    R = G.graph.get('R', 0)
    supertriangleC = P.graph['supertriangleC']
    VertexC_expected = np.vstack(
        (G.graph['VertexC'][:-R], supertriangleC, G.graph['VertexC'][-R:])
    )
    assert np.allclose(scaff.graph['VertexC'], VertexC_expected)


def test_update_lengths():
    wfn = tiny_wfn()
    G = wfn.G.copy()

    expected_lengths = [
        (0, 12, 0.5385164807134504),
        (0, -1, 1.0),
        (1, 13, 0.7071067811865476),
        (1, 2, 1.0),
        (2, 3, 2.0),
        (12, 13, 0.30000000000000004),
    ]

    # remove some of lengths from G
    del G.edges[0, -1]['length']
    del G.edges[2, 3]['length']

    update_lengths(G)

    # check all lengths are available in G
    for u, v, expected in expected_lengths:
        actual_length = G.edges[u, v].get('length')
        assert actual_length == pytest.approx(expected), (
            f'Edge {(u, v)} length {actual_length} != expected {expected}'
        )


def test_count_diagonals():
    wfn = tiny_wfn()
    diagonals = count_diagonals(wfn.S, wfn.A)
    assert diagonals == 0


# --- add_link_blockmap ---


def test_add_link_blockmap():
    wfn = tiny_wfn()
    A = wfn.A
    add_link_blockmap(A)
    # Should add 'blocked__' to edges
    for _, _, d in A.edges(data=True):
        assert 'blocked__' in d
        assert len(d['blocked__']) == A.graph['R']
    # Should add angle arrays to graph
    assert 'angle__' in A.graph
    assert 'angle_rank__' in A.graph


# --- add_link_cosines ---


def test_add_link_cosines():
    wfn = tiny_wfn()
    A = wfn.A
    add_link_cosines(A)
    for _, _, d in A.edges(data=True):
        assert 'cos_' in d
        assert len(d['cos_']) == A.graph['R']


# --- pathdist ---


def test_pathdist_simple():
    wfn = tiny_wfn()
    G = wfn.G
    # turbines 0=(1,0), 1=(2,0), 2=(2,1) in tiny_wfn
    # path 0→1: distance = 1.0; 1→2: distance = 1.0; total = 2.0
    dist = pathdist(G, [0, 1, 2])
    assert np.isclose(dist, 2.0)


def test_pathdist_single_step():
    wfn = tiny_wfn()
    G = wfn.G
    # just two nodes
    VertexC = G.graph['VertexC']
    expected = np.hypot(*(VertexC[0] - VertexC[1]))
    assert np.isclose(pathdist(G, [0, 1]), expected)


# --- add_terminal_closest_root ---


def test_add_terminal_closest_root():
    wfn = tiny_wfn()
    A = wfn.A.copy()
    add_terminal_closest_root(A)
    T, R = A.graph['T'], A.graph['R']
    # every terminal gets a 'root' attribute
    for n in range(T):
        assert 'root' in A.nodes[n]
        assert -R <= A.nodes[n]['root'] < 0
    # rootmask__ is a list of R bitarrays, each of length T
    assert 'rootmask__' in A.graph
    assert len(A.graph['rootmask__']) == R
    assert all(len(bm) == T for bm in A.graph['rootmask__'])


# --- make_remap ---


def test_make_remap_identity():
    """Remapping a graph to itself should produce an identity map."""
    wfn = tiny_wfn()
    G = wfn.G
    T = G.graph['T']
    remap = make_remap(G, [0, 1], G, [0, 1])
    # Every terminal maps to itself (within floating-point alignment)
    for i in range(T):
        assert remap[i] == i
