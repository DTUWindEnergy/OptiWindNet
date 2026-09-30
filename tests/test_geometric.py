# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import numpy as np
import pytest
import shapely

from optiwindnet.geometric import (
    angle,
    any_pairs_opposite_edge,
    area_from_polygon_vertices,
    complete_graph,
    find_segments_crossing_any,
    is_bunch_split_by_corner,
    is_crossing,
    is_crossing_no_bbox,
    is_crossing_numpy,
    is_same_side,
    is_triangle_pair_a_convex_quadrilateral,
    is_triangle_pair_a_convex_quadrilateral_XY,
    minimum_spanning_forest,
    perimeter,
    point_d2line,
    point_to_segment_distance,
    rotate,
    rotating_calipers,
    rotation_checkers_factory,
    triangle_AR,
    unique_rays,
)

from .helpers import tiny_wfn


def test_area_from_polygon_vertices():
    # Square 1x1
    X = np.array([0, 1, 1, 0])
    Y = np.array([0, 0, 1, 1])
    assert area_from_polygon_vertices(X, Y) == 1.0

    # Square 1x1 reverse order
    assert area_from_polygon_vertices(X[::-1], Y[::-1]) == 1.0

    # Triangle base 2, height 2 -> area 2
    X_tri = np.array([0, 2, 0])
    Y_tri = np.array([0, 0, 2])
    assert area_from_polygon_vertices(X_tri, Y_tri) == 2.0

    # Negative coordinates square 2x2
    X_neg = np.array([-1, 1, 1, -1])
    Y_neg = np.array([-1, -1, 1, 1])
    assert area_from_polygon_vertices(X_neg, Y_neg) == 4.0


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
    expected = np.array([
        [0.9961947, 0.08715574], [1.9923894, 0.17431149],
        [1.90523365, 1.17050618], [1.73092217, 3.16289558],
        [-1.81807791, -2.16670088], [2.16670088, -1.81807791],
        [1.64376643, 4.15909028], [-2.34101237, 3.81046731],
        [1.23901151, -0.39351046], [1.10827789, 1.10078159],
        [1.74957259, 0.65497769], [1.53786992, -0.36736373],
        [0.0, 0.0],
    ])  # fmt: skip

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


def test_crossings_corner_cases():
    # Crossing
    u, v = np.array([0, 0]), np.array([2, 2])
    s, t = np.array([0, 2]), np.array([2, 0])
    assert is_crossing_numpy(u, v, s, t) is True
    assert is_crossing_no_bbox(u, v, s, t) is True
    assert is_crossing(u, v, s, t) is True

    # Touch (endpoint on segment)
    u, v = np.array([0, 0]), np.array([2, 2])
    s, t = np.array([1, 1]), np.array([1, 0])
    assert is_crossing_numpy(u, v, s, t) is True
    assert is_crossing_no_bbox(u, v, s, t) is True
    assert is_crossing(u, v, s, t, touch_is_cross=True) is True
    assert is_crossing(u, v, s, t, touch_is_cross=False) is False

    # Parallel (no overlap)
    u, v = np.array([0, 0]), np.array([2, 0])
    s, t = np.array([0, 1]), np.array([2, 1])
    assert is_crossing_numpy(u, v, s, t) is False
    assert is_crossing_no_bbox(u, v, s, t) is False
    assert is_crossing(u, v, s, t) is False

    # Superposition (overlap)
    u, v = np.array([0, 0]), np.array([2, 0])
    s, t = np.array([1, 0]), np.array([3, 0])
    assert is_crossing_numpy(u, v, s, t) is False
    assert is_crossing_no_bbox(u, v, s, t) is False
    assert is_crossing(u, v, s, t) is False


# --- find_segments_crossing_any ---


def test_find_segments_crossing_any_cases():
    segmentsC = np.array([[[0.0, 0.0], [2.0, 0.0]], [[5.0, 5.0], [6.0, 6.0]]])
    probesC = np.array(
        [
            [[1.0, -1.0], [1.0, 1.0]],  # proper crossing
            [[2.0, 0.0], [3.0, 1.0]],  # shares an endpoint
            [[1.0, 0.0], [1.0, 1.0]],  # endpoint on the segment's interior
            [[1.0, 0.0], [3.0, 0.0]],  # collinear overlap
            [[0.0, 1.0], [2.0, 1.0]],  # parallel
            [[3.0, -1.0], [3.0, 1.0]],  # disjoint
        ]
    )
    assert find_segments_crossing_any(probesC, segmentsC).tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]


def test_find_segments_crossing_any_shared_endpoint_is_a_touch():
    # A probe ending exactly at a segment's endpoint (a root-to-border-vertex
    # line of sight at Race Bank). Is a touch, not a crossing; the parametric
    # test of is_crossing_no_bbox() misses by one ulp and would flag it.
    probesC = np.array(
        [
            [
                [-0.07603445744258204, 0.08381560382070354],
                [-0.1499433763903498, 0.020403148550092887],
            ]
        ]
    )
    segmentsC = np.array(
        [
            [
                [-0.06931984687711679, -0.12457352221799783],
                [-0.1499433763903498, 0.020403148550092887],
            ]
        ]
    )
    assert not find_segments_crossing_any(probesC, segmentsC)[0]


@pytest.mark.parametrize(
    'uC, vC, sC, tC',
    [
        # orientation signs certified by the float filter
        (
            [0.10033256982245087, 1.2471068907925238],
            [0.6258883699967324, 0.8936754814926915],
            [0.625095466604667, 0.8972138009695755],
            [0.7756856902451935, 0.22520718999059186],
        ),
        # orientation signs not certified by the float filter
        (
            [0.2752636022214574, 0.1460726925164133],
            [0.8303115232139779, 0.3159018758546133],
            [0.8713393766928806, 0.3612640590141576],
            [0.5981840672072131, 0.05925164234550362],
        ),
    ],
)
def test_find_segments_crossing_any_rounded_touch(uC, vC, sC, tC):
    # vC is within rounding of the interior of ⟨sC, tC⟩, beyond it in exact
    # arithmetic, but the intersection point GEOS computes is vC itself: a touch
    probe = shapely.LineString([uC, vC])
    segment = shapely.LineString([sC, tC])
    assert probe.touches(segment) and not probe.crosses(segment)
    assert not find_segments_crossing_any(np.array([[uC, vC]]), np.array([[sC, tC]]))[0]


def test_find_segments_crossing_any_matches_shapely_near_degenerate():
    rng = np.random.default_rng(0)
    n = 500
    sC, tC = rng.random((n, 2)), rng.random((n, 2))
    uC = rng.random((n, 2)) * 2 - 0.5
    # probe ends within rounding of the segment's line
    vC = sC + rng.random((n, 1)) * (tC - sC)
    probesC = np.stack((uC, vC), axis=1)
    segmentsC = np.stack((sC, tC), axis=1)
    expected = shapely.crosses(
        shapely.linestrings(probesC), shapely.linestrings(segmentsC)
    )
    got = [
        find_segments_crossing_any(probesC[i : i + 1], segmentsC[i : i + 1])[0]
        for i in range(n)
    ]
    assert got == expected.tolist()


@pytest.mark.parametrize(
    'uC, vC, sC, tC, expected',
    [
        # square
        ([0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0], True),
        # v inside triangle ⟨u, s, t⟩
        ([0.0, 0.0], [0.3, 0.3], [1.0, 0.0], [0.0, 1.0], False),
        # u on ⟨s, t⟩ in decimal, but float cross product is ~-3e-18
        ([0.06, 0.46], [0.16, 0.36], [0.11, 0.51], [0.01, 0.41], False),
        # same, far from the origin
        (
            [6e6 + 60.0, 5e5 + 460.0],
            [6e6 + 160.0, 5e5 + 360.0],
            [6e6 + 110.0, 5e5 + 510.0],
            [6e6 + 10.0, 5e5 + 410.0],
            False,
        ),
        # thin but convex: sine of the angle at u is 2e-8
        ([0.0, 1e-8], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0], True),
        # angle at u straight up to 2e-12
        ([0.0, 1e-12], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0], False),
        # u coincides with s
        ([1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0], False),
    ],
)
def test_is_triangle_pair_a_convex_quadrilateral(uC, vC, sC, tC, expected):
    XY = [uC, vC, sC, tC]
    assert is_triangle_pair_a_convex_quadrilateral_XY(XY, 0, 1, 2, 3) is expected
    assert (
        is_triangle_pair_a_convex_quadrilateral(*np.array(XY, dtype=float)) == expected
    )
    # the answer does not depend on the side taken as ⟨u, v⟩'s first vertex
    assert is_triangle_pair_a_convex_quadrilateral_XY(XY, 1, 0, 3, 2) is expected


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
    _calipers, _angle_val, metric, _bbox = rotating_calipers(hull, metric='height')
    assert np.isclose(metric, 1.0)


def test_rotating_calipers_rectangle():
    hull = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    _calipers, _angle_val, metric, _bbox = rotating_calipers(hull, metric='height')
    assert np.isclose(metric, 1.0)


def test_rotating_calipers_area_metric():
    hull = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    _calipers, _angle_val, metric, _bbox = rotating_calipers(hull, metric='area')
    assert np.isclose(metric, 2.0)


def test_rotating_calipers_unknown_metric():
    hull = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match='Unknown metric'):
        rotating_calipers(hull, metric='invalid')


# --- triangle_AR ---


def test_triangle_AR_basic():
    base1 = np.array([0.0, 0.0])
    base2 = np.array([2.0, 0.0])
    top = np.array([1.0, 1.0])
    # base_sqr = 4, den = |0*1 - 2*1 + 2*0 - 0*0| = 2 → AR = 2
    assert np.isclose(triangle_AR(base1, base2, top), 2.0)


def test_triangle_AR_collinear():
    base1 = np.array([0.0, 0.0])
    base2 = np.array([2.0, 0.0])
    top = np.array([1.0, 0.0])  # on the baseline
    assert np.isinf(triangle_AR(base1, base2, top))


# --- is_same_side ---


def test_is_same_side_opposite():
    u = np.array([-1.0, 0.0])
    v = np.array([1.0, 0.0])
    s = np.array([0.0, 1.0])
    t = np.array([0.0, -1.0])
    assert not is_same_side(u, v, s, t, touch_is_cross=False)


def test_is_same_side_same():
    u = np.array([-1.0, 0.0])
    v = np.array([1.0, 0.0])
    s = np.array([0.0, 1.0])
    t = np.array([0.0, 2.0])
    assert is_same_side(u, v, s, t, touch_is_cross=False)


def test_is_same_side_touch_counts():
    u = np.array([-1.0, 0.0])
    v = np.array([1.0, 0.0])
    s = np.array([0.0, 1.0])
    t = np.array([0.0, 0.0])  # on the line
    assert is_same_side(u, v, s, t, touch_is_cross=True)
    assert not is_same_side(u, v, s, t, touch_is_cross=False)


def test_is_same_side_vertical_line():
    # vertical line x=1 → uses the denom==0 branch
    u = np.array([1.0, -1.0])
    v = np.array([1.0, 1.0])
    s = np.array([2.0, 0.0])  # right of x=1
    t = np.array([0.0, 0.0])  # left of x=1
    assert not is_same_side(u, v, s, t, touch_is_cross=False)


# --- point_to_segment_distance ---


def test_point_to_segment_distance_perpendicular():
    p = np.array([1.0, 1.0])
    a = np.array([0.0, 0.0])
    b = np.array([2.0, 0.0])
    assert np.isclose(point_to_segment_distance(p, a, b), 1.0)


def test_point_to_segment_distance_beyond_endpoint():
    p = np.array([3.0, 0.0])
    a = np.array([0.0, 0.0])
    b = np.array([2.0, 0.0])
    assert np.isclose(point_to_segment_distance(p, a, b), 1.0)


def test_point_to_segment_distance_degenerate_segment():
    p = np.array([3.0, 4.0])
    a = np.array([0.0, 0.0])
    b = np.array([0.0, 0.0])  # zero-length segment
    assert np.isclose(point_to_segment_distance(p, a, b), 5.0)


# --- unique_rays ---


def test_unique_rays_parallel_same_direction():
    rays = [np.array([1.0, 0.0]), np.array([2.0, 0.0])]
    result = unique_rays(rays, angle_tol=1e-6)
    assert len(result) == 1


def test_unique_rays_anti_parallel():
    rays = [np.array([1.0, 0.0]), np.array([-1.0, 0.0])]
    result = unique_rays(rays, angle_tol=1e-6)
    assert len(result) == 2


def test_unique_rays_zero_norm_dropped():
    rays = [np.array([0.0, 0.0]), np.array([1.0, 0.0])]
    result = unique_rays(rays, angle_tol=1e-6)
    assert len(result) == 1


def test_unique_rays_empty():
    result = unique_rays([], angle_tol=1e-6)
    assert result == []


# --- rotation_checkers_factory ---


def test_rotation_checkers_factory():
    # CCW triangle: 0=(0,0), 1=(1,0), 2=(0,1)
    VertexC = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cw, ccw, cross = rotation_checkers_factory(VertexC)

    assert ccw(0, 1, 2)
    assert not cw(0, 1, 2)
    assert cross(0, 1, 2) > 0

    # Reverse order → CW
    assert cw(0, 2, 1)
    assert not ccw(0, 2, 1)
    assert cross(0, 2, 1) < 0


def test_rotation_checkers_factory_collinear():
    VertexC = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    cw, ccw, cross = rotation_checkers_factory(VertexC)

    assert not cw(0, 1, 2)
    assert not ccw(0, 1, 2)
    assert cross(0, 1, 2) == 0.0


# --- is_bunch_split_by_corner ---


def test_is_bunch_split_by_corner_true():
    o = np.array([0.0, 0.0])
    a = np.array([1.0, 1.0])
    b = np.array([1.0, -1.0])
    # points: one inside the cone (right), one outside (left)
    bunch = np.array([[0.5, 0.0], [-1.0, 0.0]])
    split, inside, outside = is_bunch_split_by_corner(bunch, a, o, b)
    assert split
    assert len(inside) > 0
    assert len(outside) > 0


def test_is_bunch_split_by_corner_false():
    o = np.array([0.0, 0.0])
    a = np.array([1.0, 1.0])
    b = np.array([1.0, -1.0])
    # both points outside the rightward cone
    bunch = np.array([[-1.0, 0.5], [-1.0, -0.5]])
    split, _inside, _outside = is_bunch_split_by_corner(bunch, a, o, b)
    assert not split
