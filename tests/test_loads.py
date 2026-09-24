# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Cable-load computation over solution topologies and routesets."""

import itertools
import math
from fractions import Fraction

import networkx as nx
import pytest

from optiwindnet.api_utils import extract_network_as_array
from optiwindnet.converting import _rings_from_S
from optiwindnet.loads import (
    _add_ring_to_S,
    bfs_subtree_loads,
    calcload,
    split_rings_and_calc_loads,
    terminal_inflow,
    validate_terminal_power,
)
from optiwindnet.MILP import Topology
from optiwindnet.presenting import _nominal_loads
from optiwindnet.validating import validate_topology

from .helpers import tiny_wfn


@pytest.mark.parametrize('n', range(1, 13))
def test_private_add_ring_to_S_canonical_shape(n):
    S = nx.Graph(R=1, T=n)
    S.add_node(-1)
    _add_ring_to_S(S, (-1, -1), list(range(n)), subtree=0, A=None)

    feeders = [data['load'] for u, v, data in S.edges(data=True) if min(u, v) < 0]
    zero_load_links = [
        (u, v) for u, v, data in S.edges(data=True) if data.get('load') == 0
    ]
    arm_load = math.ceil(n / 2)
    assert max(data['load'] for *_, data in S.edges(data=True)) == arm_load
    if n == 1:
        assert feeders == [1] and zero_load_links == []
    else:
        assert sorted(feeders) == [n - arm_load, arm_load]
        assert len(zero_load_links) == 1
    assert sum(S.nodes[t]['load'] for t in S.neighbors(-1)) == n
    assert {S.nodes[t]['subtree'] for t in range(n)} == {0}


def _path_form_S(R, paths):
    T = sum(len(path) for _, path in paths)
    S = nx.Graph(R=R, T=T)
    S.add_nodes_from(range(-R, 0))
    for root, ordered in paths:
        S.add_edge(root, ordered[0])
        S.add_edges_from(itertools.pairwise(ordered))
    return S


def test_split_rings_and_calc_loads_single_and_multi_root():
    one = _path_form_S(1, [(-1, [0, 1, 2]), (-1, [3, 4])])
    split_rings_and_calc_loads(one, nx.Graph())
    assert validate_topology(one, capacity=2) == []
    assert one.graph['topology'] is Topology.RINGED
    assert one.nodes[-1]['load'] == 5

    multi = _path_form_S(2, [(-2, [0, 1]), (-1, [2, 3, 4, 5]), (-1, [6])])
    split_rings_and_calc_loads(multi, nx.Graph())
    assert validate_topology(multi, capacity=3) == []
    by_root = {
        root: {
            frozenset(ordered)
            for roots, ordered in _rings_from_S(multi)
            if roots == (root, root)
        }
        for root in (-2, -1)
    }
    assert by_root[-2] == {frozenset({0, 1})}
    assert by_root[-1] == {frozenset({2, 3, 4, 5}), frozenset({6})}


def test_split_rings_rejects_branching_subtree():
    S = nx.Graph(R=1, T=3)
    S.add_edges_from([(-1, 0), (0, 1), (0, 2)])
    with pytest.raises(ValueError):
        split_rings_and_calc_loads(S, nx.Graph())


def test_calcload():
    wfn = tiny_wfn()
    G = wfn.G

    G.graph.pop('has_loads', None)
    G.graph.pop('max_load', None)

    calcload(G)

    assert G.graph['has_loads']
    assert G.graph['max_load'] == 4


def _chain_S(T, inflow=None):
    """One root feeding a single chain of ``T`` terminals."""
    S = nx.Graph(R=1, T=T)
    S.add_node(-1)
    S.add_edge(-1, 0)
    S.add_edges_from(itertools.pairwise(range(T)))
    if inflow is not None:
        nx.set_node_attributes(S, dict(enumerate(inflow)), 'inflow')
    return S


def test_calcload_sources_declared_terminal_inflow():
    inflow = (2, 3, 1, 4)
    S = _chain_S(4, inflow)

    calcload(S)

    # each link carries the inflow of everything beyond it, feeder included
    assert [S[u][v]['load'] for u, v in ((2, 3), (1, 2), (0, 1), (-1, 0))] == [
        4, 5, 8, 10,
    ]  # fmt: skip
    assert S.nodes[-1]['load'] == sum(inflow)
    assert S.graph['max_load'] == sum(inflow)


def test_calcload_without_inflow_attributes_counts_terminals():
    plain, unitary = _chain_S(4), _chain_S(4, (1, 1, 1, 1))

    calcload(plain)
    calcload(unitary)

    assert plain.nodes[-1]['load'] == unitary.nodes[-1]['load'] == 4
    assert {(u, v): d['load'] for u, v, d in plain.edges(data=True)} == {
        (u, v): d['load'] for u, v, d in unitary.edges(data=True)
    }


def test_calcload_reports_a_terminal_no_root_reaches():
    """Traversal coverage is checked by terminal count, independently of inflow."""
    S = _chain_S(4, (2, 3, 1, 4))
    S.remove_edge(2, 3)  # terminal 3 is now cut off from every root

    with pytest.raises(ValueError, match='reached 3 terminals, not T = 4'):
        calcload(S)


def test_bfs_subtree_loads_sources_declared_terminal_inflow():
    S = _chain_S(3, (2, 3, 1))

    assert bfs_subtree_loads(S, -1, [0], subtree=0) == 6
    assert S.nodes[0]['load'] == 6
    assert S.nodes[1]['load'] == 4


def test_terminal_inflow_reports_only_the_departures_from_unitary_inflow():
    S = _chain_S(4, (1, 2, 1, 3))
    S.add_node(-1, inflow=99)  # a root sources nothing; it is never reported

    assert terminal_inflow(S) == {1: 2, 3: 3}
    assert terminal_inflow(_chain_S(4)) == {}


def test_nominal_export_preserves_integer_loads_with_clones_and_open_ring():
    G = nx.Graph(R=2, T=4, power_per_inflow=Fraction(1, 2))
    G.add_edges_from([(-2, 4), (4, 0), (0, 1), (0, 2), (-1, 3)])
    G.add_edge(2, 3, load=0, reverse=True)
    nx.set_node_attributes(G, {0: 2, 1: 2, 2: 4, 3: 3}, 'inflow')
    nx.set_node_attributes(G, {0: 1.01, 1: 1.0, 2: 2.01, 3: 1.49}, 'power')
    for _, _, attrs in G.edges(data=True):
        attrs.update(length=10.0, cable=0)
    validate_terminal_power(G)
    calcload(G)
    original = G.copy()

    network = extract_network_as_array(G)

    loads = {(row['src'], row['tgt']): row['load'] for row in network}
    assert loads == pytest.approx(
        {
            (4, -2): 4.02,
            (0, 4): 4.02,
            (1, 0): 1.0,
            (2, 0): 2.01,
            (3, -1): 1.49,
            (2, 3): 0.0,
        }
    )
    assert G.nodes[-2]['load_nominal'] == pytest.approx(4.02)
    assert G.nodes[4]['load_nominal'] == pytest.approx(4.02)
    for _, attrs in G.nodes(data=True):
        attrs.pop('load_nominal')
    for _, _, attrs in G.edges(data=True):
        attrs.pop('load_nominal')
    assert nx.utils.graphs_equal(G, original)


def test_exact_nominal_loads_scale_integer_loads_without_touching_the_graph(
    monkeypatch,
):
    G = _chain_S(2, (2, 3))
    G.graph['power_per_inflow'] = Fraction(1, 2)
    validate_terminal_power(G)
    calcload(G)
    original = G.copy()

    def unexpected_traversal(*args, **kwargs):
        pytest.fail('Exact nominal loads must use existing integer loads.')

    monkeypatch.setattr('optiwindnet.loads._bfs_loads_walk', unexpected_traversal)
    nominal = _nominal_loads(G)
    assert nominal == {-1: Fraction(5, 2), 0: Fraction(5, 2), 1: Fraction(3, 2)}
    assert nx.utils.graphs_equal(G, original)


def test_inexact_nominal_loads_use_partial_declarations_and_refresh():
    G = _chain_S(2, (1, 1))
    G.nodes[0]['power'] = Fraction(101, 100)
    validate_terminal_power(G)
    assert G.graph['power_quantization_inexact'] is True
    calcload(G)
    assert _nominal_loads(G) == {-1: Fraction(201, 100), 0: Fraction(201, 100), 1: 1}
    assert G[-1][0]['load_nominal'] == Fraction(201, 100)
    assert G[0][1]['load_nominal'] == 1

    G.remove_edge(0, 1)
    G.add_edge(-1, 1)
    calcload(G)
    calcload(G, nominal=True)
    assert G[-1][0]['load_nominal'] == Fraction(101, 100)
    assert G[-1][1]['load_nominal'] == 1
    assert G.nodes[-1]['load_nominal'] == Fraction(201, 100)
    assert G.graph['max_load'] == 1


def test_inexact_flag_detects_errors_that_cancel():
    G = _chain_S(2, (1, 1))
    G.nodes[0]['power'] = Fraction(101, 100)
    G.nodes[1]['power'] = Fraction(99, 100)
    validate_terminal_power(G)
    assert G.graph['power_quantization_inexact'] is True
    calcload(G)
    calcload(G, nominal=True)
    assert G.nodes[-1]['load_nominal'] == 2
    assert G[0][1]['load_nominal'] == Fraction(99, 100)

    G.nodes[0]['power'] = G.nodes[1]['power'] = Fraction(1)
    validate_terminal_power(G)
    assert G.graph['power_quantization_inexact'] is False


def test_nominal_calcload_refreshes_and_falls_back_to_inflow():
    G = _chain_S(2, (2, 3))
    G.graph['power_per_inflow'] = Fraction(1, 2)
    calcload(G)
    nx.set_node_attributes(G, {0: Fraction(101, 100), 1: Fraction(149, 100)}, 'power')
    calcload(G, nominal=True)
    G.nodes[1]['power'] = Fraction(3, 2)

    calcload(G, nominal=True)

    assert G[-1][0]['load_nominal'] == Fraction(251, 100)
    assert G[-1][0]['load'] == G.graph['max_load'] == 5
    # an undeclared terminal is worth its inflow times power_per_inflow
    del G.nodes[1]['power']
    calcload(G, nominal=True)
    assert G[-1][0]['load_nominal'] == Fraction(101, 100) + 3 * Fraction(1, 2)
