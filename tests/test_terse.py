import json

import networkx as nx
import numpy as np
import pytest

from optiwindnet.fingerprint import fingerprint_coordinates
from optiwindnet.interarraylib import add_ring_to_S, calcload, validate_routeset
from optiwindnet.terse import LinkScope, TerseLinks
from optiwindnet.types import Topology

from .helpers import canonical_edges, tiny_wfn


@pytest.mark.parametrize('n', range(1, 11))
def test_topology_encoding_preserves_bridging_ring(n):
    S = nx.Graph(R=2, T=n, topology=Topology.RINGED)
    S.add_nodes_from((-2, -1))
    add_ring_to_S(S, (-1, -2), list(range(n)), subtree=0, A=None)
    calcload(S)

    encoded = TerseLinks.from_topology(S)
    restored = encoded.to_topology()

    assert set(map(frozenset, restored.edges)) == set(map(frozenset, S.edges))
    assert TerseLinks.from_topology(restored) == encoded


def _ring_with_cloned_open_link() -> tuple[nx.Graph, nx.Graph]:
    """Return a site and routed ring whose open cable follows a contour clone."""
    VertexC = np.array([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.5, 0.5), (0.0, 1.0)])
    L = nx.Graph(R=1, T=3, B=1, VertexC=VertexC)
    L.add_nodes_from(range(3), kind='wtg')
    L.add_node(-1, kind='oss')

    G = nx.create_empty_copy(L, with_data=True)
    G.graph.update(
        C=1,
        D=0,
        topology=Topology.RINGED,
        capacity=2,
        has_loads=True,
        fnT=np.array([0, 1, 2, 3, 3, -1]),
    )
    G.add_node(4, kind='contour', load=0)
    G.add_edges_from(
        [
            (-1, 0, {'load': 1, 'reverse': False}),
            (0, 4, {'load': 0, 'reverse': False, 'kind': 'contour'}),
            (4, 1, {'load': 0, 'reverse': False, 'kind': 'contour'}),
            (1, 2, {'load': 2, 'reverse': False}),
            (2, -1, {'load': 2, 'reverse': False}),
        ]
    )
    return L, G


def test_topology_encoding_keeps_unused_roots():
    S = nx.Graph(R=3, T=2, topology=Topology.RADIAL)
    S.add_nodes_from(range(-3, 0))
    S.add_edges_from([(-1, 0), (0, 1)])
    calcload(S)

    encoding = TerseLinks.from_topology(S)
    restored = encoding.to_topology()

    assert encoding.R == 3
    assert set(range(-3, 0)) <= restored.nodes
    assert set(map(frozenset, restored.edges)) == set(map(frozenset, S.edges))


def test_topology_encoding_roundtrips_nodeset_digest():
    source = tiny_wfn()
    digest = fingerprint_coordinates(source.L.graph['VertexC'])[0]

    encoding = TerseLinks.from_topology(source.S, nodeset_digest=digest)
    restored = TerseLinks.from_dict(json.loads(json.dumps(encoding.to_dict())))

    assert encoding.nodeset_digest == digest
    assert restored == encoding


def test_repr_summarizes_topology_encoding():
    encoding = tiny_wfn().terse_links()

    assert repr(encoding) == (
        "TerseLinks(scope='topology', topology='branched', T=4, R=1, links=4)"
    )


def test_routeset_encoding_roundtrips_clone_on_ring_open_link():
    L, G = _ring_with_cloned_open_link()

    encoding = TerseLinks.from_routeset(G)
    restored = encoding.to_routeset(L, capacity=2)

    assert encoding.scope is LinkScope.ROUTESET
    assert encoding.clone2prime == (3,)
    assert set(map(frozenset, restored.edges)) == set(map(frozenset, G.edges))
    assert {
        frozenset((u, v))
        for u, v, data in restored.edges(data=True)
        if data['load'] == 0
    } == {frozenset((0, 4)), frozenset((4, 1))}
    assert restored.graph['fnT'].tolist() == G.graph['fnT'].tolist()
    assert validate_routeset(restored) == []


def test_repr_summarizes_routeset_encoding():
    _, G = _ring_with_cloned_open_link()
    encoding = TerseLinks.from_routeset(G)

    assert repr(encoding) == (
        "TerseLinks(scope='routeset', topology='ringed', "
        'T=3, R=1, B=1, C=1, D=0, links=5)'
    )


def test_routeset_encoding_rejects_a_different_site():
    L, G = _ring_with_cloned_open_link()
    encoding = TerseLinks.from_routeset(G)
    other = nx.create_empty_copy(L, with_data=True)
    other.graph['VertexC'] = L.graph['VertexC'].copy()
    other.graph['VertexC'][0, 0] += 1

    with pytest.raises(ValueError, match='geometry does not match'):
        encoding.to_routeset(other)


def test_encoding_roundtrips_through_json():
    _, G = _ring_with_cloned_open_link()
    encoding = TerseLinks.from_routeset(G)

    restored = TerseLinks.from_dict(json.loads(json.dumps(encoding.to_dict())))

    assert restored == encoding


def test_wind_farm_network_restores_routed_encoding_directly():
    source = tiny_wfn()
    encoding = source.terse_links(routed=True)
    restored = tiny_wfn()

    restored.update_from_terse_links(encoding)

    assert encoding.scope is LinkScope.ROUTESET
    assert canonical_edges(restored.G) == canonical_edges(source.G)
    assert np.isclose(restored.length(), source.length())
