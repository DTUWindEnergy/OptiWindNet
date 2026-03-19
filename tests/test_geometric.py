import numpy as np
import pytest
from .helpers import tiny_wfn
from optiwindnet.geometric import (
    add_link_blockmap,
    add_link_cosines,
    angle,
    any_pairs_opposite_edge,
    complete_graph,
    is_crossing,
    is_crossing_numpy,
    minimum_spanning_forest,
    perimeter,
    point_d2line,
    rotate,
    rotating_calipers,
)


def test_minimum_spanning_forest():
    wfn = tiny_wfn()
    S = minimum_spanning_forest(wfn.A)
    Edges = np.array(list(S.edges()))
    expected = np.array([(0, 1), (0, -1), (1, 2), (2, 3)])
    assert np.array_equal(Edges, expected)

    # with capacity = 1, there will be detours in G
    wfn2 = tiny_wfn(cables=1)
    S2 = minimum_spanning_forest(wfn2.A)
    Edges2 = np.array(list(S2.edges()))
    expected2 = np.array([(0, 1), (0, -1), (1, 2), (2, 3)])
    assert np.array_equal(Edges2, expected2)


def test_rotate():
    wfn = tiny_wfn()
    G = wfn.G

    vertexC = G.graph['VertexC']
    rotated_vertexC = rotate(coords=vertexC, angle=5)
    expected = np.array(
        [
            [0.9961947, 0.08715574],
            [1.9923894, 0.17431149],
            [1.90523365, 1.17050618],
            [1.73092217, 3.16289558],
            [-1.81807791, -2.16670088],
            [2.16670088, -1.81807791],
            [1.64376643, 4.15909028],
            [-2.34101237, 3.81046731],
            [1.23901151, -0.39351046],
            [1.10827789, 1.10078159],
            [1.74957259, 0.65497769],
            [1.53786992, -0.36736373],
            [0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(rotated_vertexC, expected, atol=1e-6)


# --- point_d2line ---


def test_point_d2line_on_line():
    p = np.array([1.0, 0.0])
    u = np.array([0.0, 0.0])
    v = np.array([2.0, 0.0])
    assert np.isclose(point_d2line(p, u, v), 0.0)


def test_point_d2line_perpendicular():
    p = np.array([1.0, 3.0])
    u = np.array([0.0, 0.0])
    v = np.array([2.0, 0.0])
    assert np.isclose(point_d2line(p, u, v), 3.0)


def test_point_d2line_diagonal():
    p = np.array([0.0, 1.0])
    u = np.array([0.0, 0.0])
    v = np.array([1.0, 1.0])
    expected = np.sqrt(2) / 2
    assert np.isclose(point_d2line(p, u, v), expected, atol=1e-10)


# --- angle and angle_numpy ---


def test_angle_straight():
    a = np.array([1.0, 0.0])
    pivot = np.array([0.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert np.isclose(abs(angle(a, pivot, b)), np.pi)


def test_angle_right_angle():
    a = np.array([1.0, 0.0])
    pivot = np.array([0.0, 0.0])
    b = np.array([0.0, 1.0])
    assert np.isclose(angle(a, pivot, b), np.pi / 2)


def test_angle_zero():
    a = np.array([1.0, 0.0])
    pivot = np.array([0.0, 0.0])
    assert np.isclose(angle(a, pivot, a), 0.0)


def test_angle_negative():
    a = np.array([0.0, 1.0])
    pivot = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    # clockwise from a to b -> negative
    assert angle(a, pivot, b) < 0


# --- any_pairs_opposite_edge ---


def test_any_pairs_opposite_edge_true():
    nodesC = np.array([[0.0, 1.0], [0.0, -1.0]])
    uC = np.array([-1.0, 0.0])
    vC = np.array([1.0, 0.0])
    assert any_pairs_opposite_edge(nodesC, uC, vC)


def test_any_pairs_opposite_edge_false():
    nodesC = np.array([[0.0, 1.0], [0.0, 2.0]])
    uC = np.array([-1.0, 0.0])
    vC = np.array([1.0, 0.0])
    assert not any_pairs_opposite_edge(nodesC, uC, vC)


def test_any_pairs_opposite_edge_single_point():
    nodesC = np.array([[0.0, 1.0]])
    uC = np.array([-1.0, 0.0])
    vC = np.array([1.0, 0.0])
    assert not any_pairs_opposite_edge(nodesC, uC, vC)


# --- is_crossing_numpy ---


def test_is_crossing_numpy_crossing():
    u = np.array([0.0, 0.0])
    v = np.array([1.0, 1.0])
    s = np.array([1.0, 0.0])
    t = np.array([0.0, 1.0])
    assert is_crossing_numpy(u, v, s, t)


def test_is_crossing_numpy_no_crossing():
    u = np.array([0.0, 0.0])
    v = np.array([1.0, 0.0])
    s = np.array([2.0, 0.0])
    t = np.array([3.0, 0.0])
    assert not is_crossing_numpy(u, v, s, t)


def test_is_crossing_numpy_parallel():
    u = np.array([0.0, 0.0])
    v = np.array([1.0, 0.0])
    s = np.array([0.0, 1.0])
    t = np.array([1.0, 1.0])
    assert not is_crossing_numpy(u, v, s, t)


# --- is_crossing ---


def test_is_crossing_cross():
    u = np.array([0.0, 0.0])
    v = np.array([1.0, 1.0])
    s = np.array([1.0, 0.0])
    t = np.array([0.0, 1.0])
    assert is_crossing(u, v, s, t)


def test_is_crossing_no_cross():
    u = np.array([0.0, 0.0])
    v = np.array([1.0, 0.0])
    s = np.array([2.0, 2.0])
    t = np.array([3.0, 3.0])
    assert not is_crossing(u, v, s, t)


def test_is_crossing_touch_is_cross():
    u = np.array([0.0, 0.0])
    v = np.array([1.0, 0.0])
    s = np.array([1.0, 0.0])
    t = np.array([1.0, 1.0])
    # touch_is_cross=True (default): touching counts
    assert is_crossing(u, v, s, t, touch_is_cross=True)
    # touch_is_cross=False: touching does not count
    assert not is_crossing(u, v, s, t, touch_is_cross=False)


# --- perimeter ---


def test_perimeter_square():
    VertexC = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    vertices_ordered = np.array([0, 1, 2, 3])
    result = perimeter(VertexC, vertices_ordered)
    assert np.isclose(result, 4.0)


def test_perimeter_triangle():
    VertexC = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    vertices_ordered = np.array([0, 1, 2])
    result = perimeter(VertexC, vertices_ordered)
    expected = 1.0 + 1.0 + np.sqrt(2)
    assert np.isclose(result, expected)


# --- complete_graph ---


def test_complete_graph_basic():
    wfn = tiny_wfn()
    A = wfn.A
    G = complete_graph(A)
    T = A.graph['T']
    # Should have T nodes (no roots by default)
    assert G.number_of_nodes() == T
    # All edges should have 'length' attribute
    for _, _, d in G.edges(data=True):
        assert 'length' in d
        assert 'root' in d


def test_complete_graph_include_roots():
    wfn = tiny_wfn()
    A = wfn.A
    G = complete_graph(A, include_roots=True)
    T, R = A.graph['T'], A.graph['R']
    assert G.number_of_nodes() == T + R


def test_complete_graph_no_prune():
    wfn = tiny_wfn()
    A = wfn.A
    G_pruned = complete_graph(A, prune=True)
    G_unpruned = complete_graph(A, prune=False)
    # Unpruned should have at least as many edges as pruned
    assert G_unpruned.number_of_edges() >= G_pruned.number_of_edges()


# --- rotating_calipers ---


def test_rotating_calipers_square():
    hull = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    calipers, angle_val, metric, bbox = rotating_calipers(hull, metric='height')
    assert np.isclose(metric, 1.0)


def test_rotating_calipers_rectangle():
    hull = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    calipers, angle_val, metric, bbox = rotating_calipers(hull, metric='height')
    assert np.isclose(metric, 1.0)


def test_rotating_calipers_area_metric():
    hull = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    calipers, angle_val, metric, bbox = rotating_calipers(hull, metric='area')
    assert np.isclose(metric, 2.0)


def test_rotating_calipers_unknown_metric():
    hull = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match='Unknown metric'):
        rotating_calipers(hull, metric='invalid')


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
