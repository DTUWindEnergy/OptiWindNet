# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Cable-load computation over solution topologies and routesets."""

import itertools
import math
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from optiwindnet.api_utils import extract_network_as_array
from optiwindnet.loads import (
    _add_ring_to_S,
    _branch_loads,
    calcload,
    nonunit_inflow,
    split_rings_and_calc_loads,
    validate_terminal_power,
)
from optiwindnet.MILP import Topology
from optiwindnet.presenting import _nominal_loads
from optiwindnet.terse import _walk_ring_routes
from optiwindnet.validating import validate_routeset, validate_topology

from .helpers import tiny_wfn


@pytest.mark.parametrize('n', range(1, 13))
def test_private_add_ring_to_S_canonical_shape(n):
    S = nx.DiGraph(R=1, T=n)
    S.add_node(-1)
    _add_ring_to_S(S, (-1, -1), list(range(n)), subtree=0, A=None)

    feeders = [data['load'] for u, v, data in S.edges(data=True) if min(u, v) < 0]
    zero_load_links = [
        (u, v) for u, v, data in S.edges(data=True) if data.get('load') == 0
    ]
    arm_load = math.ceil(n / 2)
    assert max(data['load'] for *_, data in S.edges(data=True)) == arm_load
    if n == 1:
        assert sorted(feeders) == [0, 1] and len(zero_load_links) == 1
    else:
        assert sorted(feeders) == [n - arm_load, arm_load]
        assert len(zero_load_links) == 1
    assert sum(data['load'] for _, _, data in S.in_edges(-1, data=True)) == n
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
    one = split_rings_and_calc_loads(one, nx.Graph())
    assert validate_topology(one, capacity=2) == []
    assert one.graph['topology'] is Topology.RINGED
    assert one.nodes[-1]['load'] == 5

    multi = _path_form_S(2, [(-2, [0, 1]), (-1, [2, 3, 4, 5]), (-1, [6])])
    multi = split_rings_and_calc_loads(multi, nx.Graph())
    assert validate_topology(multi, capacity=3) == []
    by_root = {
        root: {
            frozenset(walk)
            for open_, walk, close in _walk_ring_routes(multi)
            if open_ == close == root
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
    S = nx.DiGraph(R=1, T=T)
    S.add_node(-1)
    S.add_edge(0, -1)
    S.add_edges_from((v, u) for u, v in itertools.pairwise(range(T)))
    if inflow is not None:
        nx.set_node_attributes(S, dict(enumerate(inflow)), 'inflow')
    return S


def test_calcload_sources_declared_terminal_inflow():
    inflow = (2, 3, 1, 4)
    S = _chain_S(4, inflow)

    calcload(S)

    # each link carries the inflow of everything beyond it, feeder included
    assert [S[u][v]['load'] for u, v in ((3, 2), (2, 1), (1, 0), (0, -1))] == [
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
    S.remove_edge(3, 2)  # terminal 3 is now cut off from every root

    with pytest.raises(ValueError, match='one outgoing positive-load arc'):
        calcload(S)


def test_private_branch_loads_sources_declared_terminal_inflow():
    S = _chain_S(3, (2, 3, 1))
    S.nodes[-1]['load'] = 0

    assert _branch_loads(S, 0, -1, subtree=0) == 3
    assert S.nodes[-1]['load'] == 6
    assert S.nodes[0]['load'] == 6
    assert S.nodes[1]['load'] == 4


def test_calcload_rejects_a_loaded_clone_without_an_outgoing_cable():
    G = nx.DiGraph([(0, 2), (1, 0)], R=1, T=2)
    G.add_node(-1)
    with pytest.raises(ValueError, match='node 2 must have one outgoing'):
        calcload(G)


@pytest.mark.parametrize('cycle', [[(0, 1), (1, 0)], [(2, 3), (3, 2)]])
def test_calcload_rejects_disconnected_positive_load_cycles(cycle):
    G = _chain_S(2)
    if cycle[0][0] == 0:
        G.remove_edge(0, -1)
    G.add_edges_from(cycle)
    with pytest.raises(ValueError, match='positive-load arcs contain a directed cycle'):
        calcload(G)


def test_calcload_preserves_clones_inside_a_zero_load_cable():
    G = nx.DiGraph(T=2, R=1, power_per_inflow=Fraction(1, 2))
    G.add_edges_from([(0, -1), (1, -1)])
    nx.add_path(G, (0, 2, 3, 1), load=0)
    G.nodes[2].update(kind='contour', load=99, subtree=7)
    G.nodes[3].update(kind='contour', load=99, subtree=7)

    calcload(G)
    calcload(G, nominal=True)

    for node in (2, 3):
        assert G.nodes[node] == {
            'kind': 'contour',
            'load': 0,
            'load_nominal': 0,
            'subtree': 7,
        }
    for source, sink in ((0, 2), (2, 3), (3, 1)):
        assert G[source][sink] == {'load': 0, 'load_nominal': 0}
    assert G.nodes[-1]['load'] == 2
    assert G.nodes[-1]['load_nominal'] == 1
    assert G.graph['max_load'] == 1


def test_terminal_inflow_reports_only_the_departures_from_unitary_inflow():
    S = _chain_S(4, (1, 2, 1, 3))
    S.add_node(-1, inflow=99)  # a root sources nothing; it is never reported

    assert nonunit_inflow(S) == {1: 2, 3: 3}
    assert nonunit_inflow(_chain_S(4)) == {}


@pytest.mark.parametrize('topology', (Topology.RADIAL, Topology.BRANCHED))
def test_calcload_carries_flow_and_nominal_loads_through_a_feeder_clone(topology):
    G = nx.DiGraph(
        T=1,
        R=1,
        B=1,
        C=1,
        capacity=1,
        topology=topology,
        power_per_inflow=2,
        VertexC=np.array([(3.0, 0.0), (1.0, 1.0), (0.0, 0.0)]),
        fnT=np.array([0, 1, 1, -1]),
        name='feeder_clone',
        handle='feeder_clone',
        border=np.array([], dtype=int),
        creator='test',
        runtime=0.0,
    )
    G.add_node(-1, kind='oss')
    G.add_node(0, kind='wtg', power=2)
    G.add_node(2, kind='contour')
    G.add_edge(0, 2, length=math.sqrt(5), kind='contour')
    G.add_edge(2, -1, length=math.sqrt(2), kind='contour')

    calcload(G)
    calcload(G, nominal=True)

    assert G.nodes[-1]['load'] == 1
    assert G.nodes[-1]['load_nominal'] == 2
    assert G.number_of_edges() == 2
    assert validate_routeset(G) == []


def test_nominal_export_preserves_integer_loads_with_clones_and_open_ring():
    G = nx.DiGraph(R=2, T=4, power_per_inflow=Fraction(1, 2))
    G.add_edges_from([(4, -2), (0, 4), (1, 0), (2, 0), (3, -1)])
    G.add_edge(2, 3, load=0)
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

    monkeypatch.setattr('optiwindnet.presenting.calcload', unexpected_traversal)
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
    assert G[0][-1]['load_nominal'] == Fraction(201, 100)
    assert G[1][0]['load_nominal'] == 1

    G.remove_edge(1, 0)
    G.add_edge(1, -1)
    calcload(G)
    calcload(G, nominal=True)
    assert G[0][-1]['load_nominal'] == Fraction(101, 100)
    assert G[1][-1]['load_nominal'] == 1
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
    assert G[1][0]['load_nominal'] == Fraction(99, 100)

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

    assert G[0][-1]['load_nominal'] == Fraction(251, 100)
    assert G[0][-1]['load'] == G.graph['max_load'] == 5
    # an undeclared terminal is worth its inflow times power_per_inflow
    del G.nodes[1]['power']
    calcload(G, nominal=True)
    assert G[0][-1]['load_nominal'] == Fraction(101, 100) + 3 * Fraction(1, 2)
