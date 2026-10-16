"""Single-turbine rings keep both physical cables throughout the library."""

import math

import networkx as nx
import numpy as np
import pytest

from optiwindnet.api import EWRouter
from optiwindnet.api_utils import extract_network_as_array
from optiwindnet.constraints import NetworkConstraints
from optiwindnet.converting import G_from_S, S_from_G, linkbits_from_S
from optiwindnet.crossings import (
    find_routeset_crossings,
)
from optiwindnet.db import G_from_routeset, RouteSet, database_connection, store_G
from optiwindnet.db.storage import pack_G
from optiwindnet.interarraylib import assign_cables
from optiwindnet.loads import _add_ring_to_S
from optiwindnet.pathfinding import PathFinder
from optiwindnet.plotting import gplot
from optiwindnet.presenting import describe_G
from optiwindnet.svg import svgplot
from optiwindnet.terse import TerseLinks
from optiwindnet.types import Topology
from optiwindnet.validating import validate_routeset, validate_topology

from .helpers import tiny_wfn


def _singleton(*, kind=None):
    """Return a routed same-root singleton ring, optionally through clones."""
    G = nx.DiGraph(
        T=1,
        R=1,
        B=1,
        capacity=1,
        max_load=1,
        has_loads=True,
        topology=Topology.RINGED,
        VertexC=np.array([(3.0, 0.0), (1.0, 1.0), (0.0, 0.0)]),
        name='singleton',
        handle='singleton',
        border=np.array([], dtype=int),
        creator='test',
        runtime=0.0,
    )
    G.add_node(-1, load=1, kind='oss')
    G.add_node(0, load=1, subtree=0, kind='wtg')
    if kind:
        G.graph.update(
            C=2 * int(kind == 'contour'),
            D=2 * int(kind == 'detour'),
            fnT=np.array([0, 1, 1, 1, -1]),
        )
        G.add_node(2, load=1, subtree=0, kind=kind)
        G.add_edge(0, 2, load=1, length=math.sqrt(5), kind=kind)
        G.add_edge(2, -1, load=1, length=math.sqrt(2), kind=kind)
        G.add_node(3, load=0, subtree=0, kind=kind)
        G.add_edge(-1, 3, load=0, length=math.sqrt(2), kind=kind)
        G.add_edge(3, 0, load=0, length=math.sqrt(5), kind=kind)
    else:
        G.add_edge(0, -1, load=1, length=3.0)
        G.add_edge(-1, 0, load=0, length=3.0)
    return G


@pytest.mark.parametrize('kind', (None, 'contour', 'detour'))
def test_singleton_has_two_physical_cables_through_roundtrips(kind):
    G = _singleton(kind=kind)
    assign_cables(G, [(1, 4.0)])
    expected_length = 2 * (math.sqrt(5) + math.sqrt(2)) if kind else 6.0

    assert G.degree[-1] == 2
    assert G.size(weight='length') == pytest.approx(expected_length)
    assert G.size(weight='cost') == pytest.approx(4 * expected_length)
    assert all('reverse' not in data for _, _, data in G.edges(data=True))
    assert validate_routeset(G) == []
    assert find_routeset_crossings(G) == []
    assert set(S_from_G(G).edges) == {(0, -1), (-1, 0)}
    assert validate_topology(S_from_G(G)) == []
    encoded = TerseLinks.from_routeset(G)
    decoded = TerseLinks.from_dict(encoded.to_dict()).to_routeset(G)
    assert set(decoded.edges) == set(G.edges)
    assert decoded.size(weight='length') == pytest.approx(expected_length)
    assert validate_routeset(decoded) == []
    assert len(extract_network_as_array(G)) == G.number_of_edges()
    assert sorted(extract_network_as_array(G)['load']) == [0.0] * (1 + bool(kind)) + [
        1.0
    ] * (1 + bool(kind))
    assert '[-1]: 2' in describe_G(G)[1]
    packed = pack_G(G)
    assert packed['length'] == pytest.approx(expected_length)
    assert packed['feeders_per_root'] == [2]


@pytest.mark.parametrize('kind', (None, 'contour', 'detour'))
def test_singleton_rendering(kind):
    G = _singleton(kind=kind)
    assign_cables(G, [(1, 4.0)])
    assert 'svg' in svgplot(G).data
    ax = gplot(G)
    assert ax is not None


def test_ringed_router_keeps_mesh_undirected_and_counts_singleton_twice():
    wfn = tiny_wfn(
        turbinesC=[[1.0, 0.0], [1.0, 1.0], [1.0, 3.0]],
        cables=[(1, 10.0)],
        obstacleC_=[],
        optimize=False,
    )
    wfn.constraints = NetworkConstraints('ringed')
    wfn.optimize(router=EWRouter(method='ringed'))
    assert not wfn.A.is_directed()
    assert wfn.S.number_of_edges() == 5
    assert wfn.G.degree[-1] == 4
    assert wfn.length() == pytest.approx(
        sum(
            wfn.A.graph['d2roots'][t, r] if r < 0 else wfn.A[t][r]['length']
            for t, r in wfn.S.edges
        )
    )
    assert wfn.cost() == pytest.approx(10 * wfn.length())
    assert validate_routeset(wfn.G) == []
    assert linkbits_from_S(wfn.A, wfn.S).count() == 4
    length_gradient, _ = wfn.gradient()
    cost_gradient, _ = wfn.gradient(gradient_type='cost')
    np.testing.assert_allclose(cost_gradient, 10 * length_gradient)
    coords = wfn.G.graph['VertexC'].copy()

    def length_at(x):
        coords[0, 0] = x
        return sum(np.linalg.norm(coords[u] - coords[v]) for u, v in wfn.G.edges)

    epsilon = 1e-6
    derivative = (length_at(1 + epsilon) - length_at(1 - epsilon)) / (2 * epsilon)
    assert length_gradient[0, 0] == pytest.approx(derivative)


def test_pathfinder_preserves_both_cables_around_obstacles():
    wfn = tiny_wfn(optimize=False)
    S = nx.DiGraph(T=4, R=1, capacity=1, max_load=1, has_loads=True)
    S.add_node(-1, load=4)
    for t in range(4):
        _add_ring_to_S(S, (-1, -1), [t], subtree=t)
    tentative = G_from_S(S, wfn.A)
    G = PathFinder(tentative, wfn.P, wfn.A).create_detours()
    assert G.graph.get('D', 0) > 0
    assert G.degree[-1] == 8
    assert validate_routeset(G) == []
    active = sum(d['length'] for _, _, d in G.edges(data=True) if d['load'])
    assert G.size(weight='length') == pytest.approx(2 * active)
    assert set(S_from_G(G).edges) == set(S.edges)
    encoded = TerseLinks.from_routeset(G)
    assert validate_routeset(encoded.to_routeset(wfn.L, capacity=1)) == []
    if 'detextra' in G.graph:
        assert G.graph['detextra'] == pytest.approx(
            G.size(weight='length') / tentative.size(weight='length') - 1
        )


def test_database_roundtrip_preserves_singleton_totals(tmp_path):
    wfn = tiny_wfn(
        turbinesC=[[1.0, 0.0], [1.0, 1.0], [1.0, 3.0]],
        cables=[(1, 10.0)],
        obstacleC_=[],
        optimize=False,
    )
    wfn.constraints = NetworkConstraints('ringed')
    wfn.optimize(router=EWRouter(method='ringed'))
    wfn.G.graph['method_options']['solver_name'] = 'constructor'
    with database_connection(tmp_path / 'singleton.sqlite', create_db=True):
        row = RouteSet.get_by_id(store_G(wfn.G))
        restored = G_from_routeset(row)
        assert set(restored.edges) == set(wfn.G.edges)
        assert restored.size(weight='length') == pytest.approx(wfn.length())
        assert row.feeders_per_root == [4]
        assert 'length_mismatch_on_db_read' not in restored.graph
        assert validate_routeset(restored) == []


def test_database_roundtrip_tags_both_feeders_of_a_tentative_singleton(tmp_path):
    wfn = tiny_wfn(
        turbinesC=[[1.0, 0.0], [1.0, 1.0], [1.0, 3.0]],
        cables=[(1, 10.0)],
        obstacleC_=[],
        optimize=False,
    )
    wfn.constraints = NetworkConstraints('ringed')
    wfn.optimize(router=EWRouter(method='ringed'))
    G = wfn.G.copy()
    G.graph['method_options']['solver_name'] = 'constructor'
    singleton = next(t for t in G.predecessors(-1) if G.has_edge(-1, t))
    # both feeders of the singleton share the one stored link pair
    G.graph['tentative'] = [(-1, singleton)]
    with database_connection(tmp_path / 'singleton.sqlite', create_db=True):
        restored = G_from_routeset(RouteSet.get_by_id(store_G(G)))
    assert restored[singleton][-1]['kind'] == 'tentative'
    assert restored[-1][singleton]['kind'] == 'tentative'
