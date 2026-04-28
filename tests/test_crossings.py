import numpy as np
import pytest
import networkx as nx
from .helpers import tiny_wfn
from optiwindnet.crossings import (
    full_geometric_crossings,
    get_interferences_list,
    validate_routeset,
)


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


def test_validate_routeset():
    wfn = tiny_wfn()
    G = wfn.G

    validate_0 = validate_routeset(G)
    assert validate_0 == []

    G.add_edge(-1, 11)
    validate_1 = validate_routeset(G)
    expected = [(0, 12, -1, 11)]
    assert all(set(val) == set(exp) for val, exp in zip(validate_1, expected))

    G.remove_edge(0, 12)
    with pytest.raises(AssertionError):
        validate_routeset(G)

    # with detours
    wfn2 = tiny_wfn(cables=1)
    G2 = wfn2.G
    assert validate_routeset(G2) == []


def test_full_geometric_crossings_detects_clone_overlap():
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

    crossings = full_geometric_crossings(G)

    assert any(
        crossing['kind'] == 'overlap_cross'
        and set(crossing['path_a']) == {0, 4, 5, 6, 1}
        and set(crossing['path_b']) == {2, 6, 5, 4, 3}
        for crossing in crossings
    )


def test_full_geometric_crossings_ignores_common_trunk_overlap():
    G = nx.Graph(
        T=4,
        B=0,
        C=0,
        D=0,
        R=1,
        VertexC=np.array(
            [[0.0, 1.0], [1.0, 1.0], [0.0, 2.0], [1.0, 2.0], [0.5, 0.0]]
        ),
    )
    G.add_node(-1, kind='oss')
    G.add_nodes_from(range(4), kind='wtg')
    G.add_edges_from([(-1, 0), (0, 1), (1, 2), (1, 3)])

    assert full_geometric_crossings(G) == []


def test_full_geometric_crossings_ignores_endpoint_touch():
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

    assert full_geometric_crossings(G) == []


def test_full_geometric_crossings_detects_detour_branch_split():
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

    crossings = full_geometric_crossings(G)

    assert [
        (crossing['kind'], crossing['path_a'], crossing['path_b'])
        for crossing in crossings
    ] == [('branch_split', (4, 1), (1, 1, 3, 2))]
