"""Cable-load computation over solution topologies and routesets."""

import itertools
import math

import networkx as nx
import pytest

from optiwindnet.loads import (
    add_ring_to_S,
    calcload,
    rings_from_S,
    split_rings_and_calc_loads,
)
from optiwindnet.MILP import Topology
from optiwindnet.validating import validate_topology

from .helpers import tiny_wfn


@pytest.mark.parametrize('n', range(1, 13))
def test_add_ring_to_S_canonical_shape(n):
    S = nx.Graph(R=1, T=n)
    S.add_node(-1)
    add_ring_to_S(S, (-1, -1), list(range(n)), subtree=0, A=None)

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


@pytest.mark.parametrize('n', range(1, 13))
@pytest.mark.parametrize('bridging', (False, True), ids=('one-root', 'bridging'))
def test_rings_from_S_roundtrip(n, bridging):
    R = 2 if bridging else 1
    S = nx.Graph(R=R, T=n)
    S.add_nodes_from(range(-R, 0))
    roots = (-1, -2) if bridging else (-1, -1)
    add_ring_to_S(S, roots, list(range(n)), subtree=0, A=None)

    recovered_roots, ordered = rings_from_S(S)[0]
    assert set(recovered_roots) == set(roots)
    assert set(ordered) == set(range(n))


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
            for roots, ordered in rings_from_S(multi)
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
