import networkx as nx
import numpy as np
import pytest

from optiwindnet.crossings import (
    find_geometric_crossings,
    find_routeset_crossings,
    get_interferences_list,
)
from optiwindnet.interarraylib import validate_routeset

from .helpers import tiny_wfn


def _graph_with_clones(T, B, C, D, VertexC, edges):
    G = nx.Graph(T=T, B=B, C=C, D=D, R=1, VertexC=np.array(VertexC, dtype=float))
    G.add_nodes_from(range(T), kind='wtg')
    G.add_nodes_from(range(T, T + B), kind='border')
    G.add_nodes_from(range(T + B, T + B + C), kind='contour')
    G.add_nodes_from(range(T + B + C, T + B + C + D), kind='detour')
    G.add_node(-1, kind='oss')
    for edge in edges:
        if len(edge) == 2:
            u, v = edge
            data = {}
        else:
            u, v, data = edge
        G.add_edge(u, v, **data)
    return G


def test_get_interferences_list():
    def check_interferences(value, expected):
        return all(
            (set(valXing) == set(expXing) and (valtouch == exptouch))
            for (valXing, valtouch), (expXing, exptouch) in zip(value, expected)
        )

    wfn = tiny_wfn()
    G = wfn.G
    VertexC = np.array(G.graph['VertexC'])

    Edge = np.array(G.edges)

    with pytest.raises(IndexError):
        get_interferences_list(Edge=Edge, VertexC=VertexC)

    fnT = G.graph['fnT']

    crossings_0 = get_interferences_list(Edge=Edge, VertexC=VertexC, fnT=fnT)
    assert not crossings_0

    G.add_edge(-1, 11)
    crossings_1 = get_interferences_list(
        Edge=np.array(G.edges), VertexC=VertexC, fnT=fnT
    )
    expected = [((0, 12, -1, 11), None)]
    assert check_interferences(crossings_1, expected)

    # with detours
    wfn2 = tiny_wfn(cables=1)
    G2 = wfn2.G
    VertexC2 = np.array(G2.graph['VertexC'])
    Edge2 = np.array(G2.edges)
    fnT2 = G2.graph['fnT']

    crossings_2 = get_interferences_list(Edge=Edge2, VertexC=VertexC2, fnT=fnT2)
    assert not crossings_2

    crossings_3 = get_interferences_list(Edge=Edge2, VertexC=VertexC2, fnT=fnT)
    expected = [((-1, 13, 1, 12), None)]
    assert check_interferences(crossings_3, expected)


def test_find_routeset_crossings():
    G = tiny_wfn().G
    assert find_routeset_crossings(G) == []

    # this feeder crosses the 0–12 link
    G.add_edge(-1, 11)
    Xings = find_routeset_crossings(G)
    assert len(Xings) == 1
    assert set(Xings[0]) == {0, 12, -1, 11}

    # with detours
    assert find_routeset_crossings(tiny_wfn(cables=1).G) == []


def test_validate_routeset():
    wfn = tiny_wfn()
    G = wfn.G
    assert validate_routeset(G) == []

    # the added feeder crosses 0–12 and leaves the loads no longer matching
    G.add_edge(-1, 11)
    violations = validate_routeset(G)
    assert any('crosses' in v for v in violations)

    # with detours
    assert validate_routeset(tiny_wfn(cables=1).G) == []


def test_validate_routeset_reports_a_disconnected_terminal():
    """Removing a link strands a terminal: reported, not raised."""
    G = tiny_wfn().G
    G.remove_edge(0, 12)

    violations = validate_routeset(G)
    assert violations
    assert not any('crosses' in v for v in violations)


def test_validate_routeset_reports_wrong_loads_instead_of_fixing_them():
    """Loads are verified against the links, not recomputed over them."""
    G = tiny_wfn().G
    u, v, edgeD = next(iter(G.edges(data=True)))
    stated = edgeD['load'] + 7
    edgeD['load'] = stated

    violations = validate_routeset(G)
    assert any(f'states load {stated}' in v for v in violations)
    # G is left exactly as it was handed over
    assert G[u][v]['load'] == stated


def test_find_geometric_crossings_detects_simple_cross():
    """Two feeders from a single root cross once at a non-vertex point."""
    G = nx.Graph(
        T=4,
        B=0,
        C=0,
        D=0,
        R=1,
        VertexC=np.array(
            [
                [0.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
                [2.0, 0.0],
                [-3.0, 0.0],
            ]
        ),
    )
    G.add_node(-1, kind='oss')
    G.add_nodes_from(range(4), kind='wtg')
    G.add_edges_from([(-1, 0), (0, 1), (-1, 2), (2, 3)])

    crossings = find_geometric_crossings(G)
    assert len(crossings) == 1
    finding = crossings[0]
    assert finding['kind'] == 'cross'
    assert finding['path_a'] == (1, 0)
    assert finding['path_b'] == (3, 2)
    assert finding['geometry'] == 'POINT (1 1)'


def test_find_geometric_crossings_multi_root():
    """Crossings between feeders rooted at different substations are detected."""
    G = nx.Graph(
        T=4,
        B=0,
        C=0,
        D=0,
        R=2,
        VertexC=np.array(
            [
                [1.0, 1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
                [-2.0, 0.0],
                [2.0, 0.0],
            ]
        ),
    )
    G.add_nodes_from([-2, -1], kind='oss')
    G.add_nodes_from(range(4), kind='wtg')
    G.add_edges_from([(-2, 0), (-2, 1), (-1, 2), (-1, 3)])

    crossings = find_geometric_crossings(G)
    kinds = sorted(c['kind'] for c in crossings)
    assert kinds == ['cross', 'cross']


def test_find_geometric_crossings_include_touches_clean_case():
    """A clean routeset returns no findings even with include_touches=True."""
    wfn = tiny_wfn()
    G = wfn.G
    assert find_geometric_crossings(G) == []
    assert find_geometric_crossings(G, include_touches=True) == []


def test_find_geometric_crossings_clean_tiny_wfn():
    """An optimized tiny_wfn (with and without detours) is crossing-free."""
    wfn = tiny_wfn()
    assert find_geometric_crossings(wfn.G) == []
    wfn_with_detours = tiny_wfn(cables=1)
    assert find_geometric_crossings(wfn_with_detours.G) == []


def test_find_geometric_crossings_detects_broken_routeset():
    """Adding a feeder that crosses an existing edge is reported as 'cross'."""
    wfn = tiny_wfn()
    G = wfn.G
    G.add_edge(-1, 11)
    crossings = find_geometric_crossings(G)
    # validate_routeset reports the same edge crossing
    assert validate_routeset(G)
    assert any(c['kind'] == 'cross' for c in crossings)


def test_find_geometric_crossings_detects_clone_overlap():
    T, B, C, D, R = 4, 3, 3, 3, 1
    VertexC = np.array(
        [
            [-1.0, -0.2],
            [3.0, 0.2],
            [3.0, -0.2],
            [-1.0, 0.2],
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [-2.0, -0.2],
        ]
    )
    G = nx.Graph(T=T, B=B, C=C, D=D, R=R, VertexC=VertexC)
    G.add_nodes_from(range(T), kind='wtg')
    G.add_nodes_from(range(T, T + B), kind='border')
    G.add_node(-1, kind='oss')
    G.add_nodes_from(range(T + B, T + B + C), kind='contour')
    G.add_nodes_from(range(T + B + C, T + B + C + D), kind='detour')
    G.add_edges_from([(0, 7), (7, 8), (8, 9), (9, 1)], kind='contour')
    G.add_edges_from([(2, 10), (10, 11), (11, 12), (12, 3)], kind='detour')
    G.add_edge(-1, 3)
    G.graph['fnT'] = np.array([0, 1, 2, 3, 4, 5, 6, 4, 5, 6, 6, 5, 4, -1])

    crossings = find_geometric_crossings(G)

    assert any(
        crossing['kind'] == 'overlap_cross'
        and set(crossing['path_a']) == {0, 4, 5, 6, 1}
        and set(crossing['path_b']) == {2, 6, 5, 4, 3}
        for crossing in crossings
    )


def test_find_geometric_crossings_ignores_common_trunk_overlap():
    G = nx.Graph(
        T=4,
        B=0,
        C=0,
        D=0,
        R=1,
        VertexC=np.array([[0.0, 1.0], [1.0, 1.0], [0.0, 2.0], [1.0, 2.0], [0.5, 0.0]]),
    )
    G.add_node(-1, kind='oss')
    G.add_nodes_from(range(4), kind='wtg')
    G.add_edges_from([(-1, 0), (0, 1), (1, 2), (1, 3)])

    assert find_geometric_crossings(G) == []


def test_find_geometric_crossings_ignores_endpoint_touch():
    G = nx.Graph(
        T=4,
        B=0,
        C=0,
        D=0,
        R=1,
        VertexC=np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0], [0.0, -2.0]]
        ),
    )
    G.add_node(-1, kind='oss')
    G.add_nodes_from(range(4), kind='wtg')
    G.add_edges_from([(-1, 0), (0, 1), (2, 3)])

    assert find_geometric_crossings(G) == []


def test_find_geometric_crossings_detects_detour_branch_split():
    # Real terminal 1 has branch rays toward 2 and 3.  Detour clone 5 maps to
    # terminal 1 and its route passes through that coordinate, splitting them.
    G = _graph_with_clones(
        T=5,
        B=0,
        C=0,
        D=1,
        VertexC=[
            [0.0, -1.0],
            [0.0, 0.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [0.0, -2.0],
        ],
        edges=[
            (-1, 0),
            (0, 1),
            (1, 2),
            (1, 3),
            (5, -1, {'kind': 'detour'}),
            (5, 4, {'kind': 'detour'}),
        ],
    )
    G.graph['fnT'] = np.array([0, 1, 2, 3, 4, 1, -1])

    crossings = find_geometric_crossings(G)

    assert [
        (crossing['kind'], crossing['path_a'], crossing['path_b'])
        for crossing in crossings
    ] == [('branch_split', (4, 1), (1, 1, 3, 2))]
