# SPDX-License-Identifier: MIT
# tests/test_interarraylib.py

import math
import numpy as np
import networkx as nx
import pytest

from optiwindnet.interarraylib import (
    assign_cables,
    update_lengths,
    describe_G,
    S_from_G,
    terse_links_from_S,
    as_normalized,
    as_rescaled,
    as_undetoured,
    as_hooked_to_nearest,
    as_hooked_to_head,
    scaffolded,
)
from .helpers import tiny_wfn


# ----------------------------
# helpers
# ----------------------------
TOLERANCE = 1e-6



# ----------------------------
# assign_cables
# ----------------------------

def test_assign_cables():
    # get a network from tiny_wfn
    wfn = tiny_wfn()
    original_G = wfn.G

    # 1) Normal case: non-zero costs -> currency set, capacity added (if missing),
    #    edges get 'cable' and 'cost'
    G = original_G.copy()
    cables1 = [(1, 100.0), (2, 150.0), (4, 200.0)]
    assign_cables(G, cables1)

    # graph-level checks
    assert G.graph['cables'] == cables1
    assert G.graph['currency'] == "€"
    #expected_capacity_1 = _recompute_kind_and_costs(cables1)[2]
    assert G.graph['capacity'] == 4
    def compare_cable_and_cost(edges_expected, edges_actual):
        # Convert EdgeDataView to a list
        edges_actual = list(edges_actual)

        # Optional: check lengths match
        assert len(edges_expected) == len(edges_actual), "Number of edges mismatch"

        # Iterate pairwise
        for edge_1, edge_2 in zip(edges_expected, edges_actual):
            u1, v1, attr1 = edge_1
            u2, v2, attr2 = edge_2

            # Check cable type
            assert attr1['cable'] == attr2['cable'], f"Edge ({u1}, {v1}) cable mismatch: {attr1['cable']} != {attr2['cable']}"

            # Check cost (approximate)
            assert math.isclose(attr1['cost'], attr2['cost'], rel_tol=1e-7, abs_tol=1e-9), \
                f"Edge ({u1}, {v1}) cost mismatch: {attr1['cost']} != {attr2['cost']}"

    expected_1 = [
        (0, 12, {'cable': 2, 'cost': 107.70329614269008}),
        (0, -1, {'cable': 2, 'cost': 200.0}),
        (1, 13, {'cable': 2, 'cost': 141.4213562373095}),
        (1, 2, {'cable': 1, 'cost': 150.0}),
        (2, 3, {'cable': 0, 'cost': 200.0}),
        (12, 13, {'cable': 2, 'cost': 60.0})
    ]
    actual_1 = [(u, v, {'cable': d['cable'], 'cost': d['cost']})
         for u, v, d in G.edges(data=True)]

    compare_cable_and_cost(expected_1, actual_1)
   
    # 2) Assign again with a different cable set: currency should update, cables update
    cables2 = [(10, 1000.0), (20, 1500.0), (30, 2000.0)]

    assign_cables(G, cables2, currency="Any Currency")

    assert G.graph['cables'] == cables2
    assert G.graph['currency'] == "Any Currency"
    # function sets capacity only if missing!
    #assert G.graph['capacity'] == 10

    # 3) All-zero-costs case:
    G_zero_cost = original_G.copy()
    cables3 = [(1, 0.0), (4, 0.0)]
    assign_cables(G_zero_cost, cables3, currency="IgnoredCurrency")
    assert G_zero_cost.graph['cables'] == cables3
    # since all costs zero, no cost value
    assert 'cost' not in G_zero_cost.graph


    # 4) Error case: raise ValueError when G.graph['max_load'] > max_capacity
    G4= original_G.copy()
    small_cables = [(1, 10.0), (2, 20.0)]
    with pytest.raises(ValueError):
        assign_cables(G4, small_cables)

    # test without capacity
    G4.graph.pop('capacity', None)
    cables4 = [(1, 100.0), (2, 150.0), (5, 200.0)]
    assign_cables(G4, cables4)
    assert G4.graph['capacity'] == 5


def test_describe_G():
    # Get a tiny test graph
    wfn = tiny_wfn()
    G = wfn.G

    # Run the description function
    desc = describe_G(G)
    # Expected output
    expected = ['κ = 4, T = 4', '(+0) [-1]: 1', 'Σλ = 5.5456 m', '55 €']

    # Assert equality
    assert desc == expected, f"Output mismatch:\nGot: {desc}\nExpected: {expected}"


def test_assign_cables_capacity_error():
    G = nx.Graph(R=1, T=2)
    G.add_node(-1, kind="oss")
    G.add_node(0, kind="wtg")
    G.add_node(1, kind="wtg")
    G.add_edge(0, 1, load=3)
    G.graph["max_load"] = 3

    with pytest.raises(ValueError, match="Maximum cable capacity"):
        assign_cables(G, [(1, 0.0), (2, 0.0)])


# ----------------------------
# update_lengths + describe_G
# ----------------------------

def test_update_lengths_and_describe_G_rounding():
    # Manual tiny graph
    wfn = tiny_wfn(optimize=True)
    G = wfn.G.copy()

    #G.graph['VertexC'] = np.array([[0, -1], [0, 0], [3, 4]], float)
    update_lengths(G)

    for u, v, data in G.edges(data=True):
        assert isinstance(data["length"], float)

    desc = describe_G(G)
    assert any("κ" in line or "Σλ" in line for line in desc)


# ----------------------------
# S_from_G + terse_links
# ----------------------------

def test_S_from_G_roundtrip_and_terse_links():
    wfn = tiny_wfn(optimize=True)
    G = wfn.G.copy()
    # ensure loads exist
    for u, v in G.edges:
        G.edges[u, v]['load'] = 1
    G.graph['has_loads'] = True

    S = S_from_G(G)
    terse = terse_links_from_S(S)
    assert hasattr(terse, "shape")


# ----------------------------
# normalize/rescale
# ----------------------------

def test_as_normalized_and_as_rescaled_invert():
    wfn = tiny_wfn(optimize=True)
    G = wfn.G.copy()

    # Provide d2roots explicitly
    G.graph['d2roots'] = np.ones((len(G.nodes), G.graph['R']))
    G.graph['norm_offset'] = np.array([0.0, 0.0])
    G.graph['norm_scale'] = 0.5

    A_norm = as_normalized(G)
    for u, v, data in A_norm.edges(data=True):
        assert math.isclose(data['length'], 0.5 * G.edges[u, v]['length'])

    G_rescaled = as_rescaled(A_norm, G)
    for u, v, data in G_rescaled.edges(data=True):
        assert math.isclose(data['length'], G.edges[u, v]['length'])


# ----------------------------
# undetour
# ----------------------------

def test_as_undetoured_removes_detour_chain():
    G = tiny_wfn().G

    G2 = as_undetoured(G)
    assert detour not in G2.nodes
    assert (-1, 0) in G2.edges


# ----------------------------
# rehooking (nearest / head)
# ----------------------------

def test_as_hooked_to_nearest_and_head():
    wfn = tiny_wfn()
    G = wfn.G
    # G = nx.Graph(R=1, T=2, has_loads=True)
    # G.add_node(-1, kind="oss", load=3)
    # G.add_node(0, kind="wtg", load=1, subtree=0)
    # G.add_node(1, kind="wtg", load=1, subtree=0)
    # G.add_edge(-1, 0)
    # G.add_edge(0, 1)
    # # tentative edge
    # G.add_edge(-1, 1, kind="tentative", load=2)
    # G.graph["tentative"] = [(-1, 1)]

    # distance to root
    d2roots = np.array([[0.0], [1.0]])

    G_nearest = as_hooked_to_nearest(G, d2roots)
    assert (-1, 0) in G_nearest.edges
    assert (-1, 1) not in G_nearest.edges

    G_head = as_hooked_to_head(G, d2roots)
    assert (-1, 0) in G_head.edges
    assert (-1, 1) not in G_head.edges


# ----------------------------
# scaffolded
# ----------------------------

def test_scaffolded_merges_edges():
    G = nx.Graph(R=1, T=1)
    G.add_node(-1, kind="oss")
    G.add_node(0, kind="wtg")
    G.add_edge(-1, 0)

    from networkx import PlanarEmbedding
    P = PlanarEmbedding()
    P.add_node(-1); P.add_node(0)
    P.add_half_edge(-1, 0)
    P.add_half_edge(0, -1)
    P.graph["supertriangleC"] = np.array([[-10, -10], [10, -10], [0, 10]])
    G.graph["VertexC"] = np.array([[0, -1], [0, 0]])

    scaff = scaffolded(G, P)
    assert (-1, 0) in scaff.edges
    assert 'kind' not in scaff.edges[-1, 0]
