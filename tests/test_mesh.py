from typing import cast

import condeltri as cdt  # pyrefly: ignore[missing-import]
import networkx as nx
import numpy as np
import pytest
import shapely as shp

from optiwindnet.geometric import CoordPairs, is_crossing
from optiwindnet.identity import linkset_id
from optiwindnet.mesh import (
    _build_edge_line_tree,
    _edges_and_hull_from_cdt,
    _index,
    _record_nonstraight_root_distance,
    make_planar_embedding,
    planar_flipped_by_routeset,
)

from .helpers import tiny_wfn
from .sitecache import get_bundle


def test_small_mesh_helpers_cover_empty_and_repeated_distance_paths():
    assert _index(np.array([1, 2]), np.int_(9)) == 0
    empty_coordinates = cast(CoordPairs, np.empty((0, 2)))
    edges, lines, tree = _build_edge_line_tree(empty_coordinates, [])
    assert edges == () and lines.size == 0 and tree is None

    A = nx.Graph()
    A.add_node(0)
    distances = np.array([[2.0]])
    _record_nonstraight_root_distance(A, distances, 0, 0, 3.0)
    _record_nonstraight_root_distance(A, distances, 0, 0, 4.0)
    assert A.nodes[0]['los_d2root'] == {0: 3.0}
    assert distances[0, 0] == 4.0


def test_planar_flipped_by_routeset_skips_edge_without_two_shared_neighbors():
    planar = nx.PlanarEmbedding()
    planar.set_data({0: [1], 1: [2, 0], 2: [1]})
    planar.graph['triangles'] = []

    adjusted = planar_flipped_by_routeset(
        {(0, 2)},
        planar=planar,
        VertexC=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        ST=3,
    )

    assert set(adjusted.edges) == set(planar.edges)


# ----------------------- tests -----------------------
def test_make_planar_embedding_basic():
    wfn = tiny_wfn()

    P = wfn.P
    A = wfn.A
    assert isinstance(P, nx.PlanarEmbedding)
    assert isinstance(A, nx.Graph)
    assert A.number_of_nodes() > 0
    assert A.number_of_edges() > 0

    # check basic keys in A
    for key in ('T', 'R', 'B', 'VertexC', 'hull'):
        assert key in A.graph

    terminal_links = A.graph['_canonical_terminal_links']
    expected_links = sorted(
        (u, v) if u < v else (v, u)
        for u, v in A.edges
        if 0 <= u < A.graph['T'] and 0 <= v < A.graph['T']
    )
    assert terminal_links.dtype == np.uint32
    assert terminal_links.tolist() == [list(link) for link in expected_links]
    assert A.graph['_linkset_id'] == linkset_id(A)


@pytest.mark.parametrize(
    'handle',
    (
        # cazzaro_2022 drops 16 A edges in the shortest-path pruning loop;
        # G-140 is the smallest bundled location whose mesh also reaches Part
        # MN's late candidate-edge additions (2 of them).
        'cazzaro_2022',
        'cazzaro_2022G140',
    ),
)
def test_canonical_terminal_links_track_late_mesh_edge_changes(handle):
    A = get_bundle(handle).A
    T = A.graph['T']
    expected_links = sorted(
        (u, v) if u < v else (v, u) for u, v in A.edges if 0 <= u < T and 0 <= v < T
    )

    terminal_links = A.graph['_canonical_terminal_links']
    assert terminal_links.tolist() == [list(link) for link in expected_links]


def test_edges_and_hull_from_cdt_all():
    mesh = cdt.Triangulation(
        cdt.VertexInsertionOrder.AUTO,
        cdt.IntersectingConstraintEdges.NOT_ALLOWED,
        0.0,
    )
    pts = [cdt.V2d(0.0, 0.0), cdt.V2d(1.0, 0.0), cdt.V2d(1.0, 1.0), cdt.V2d(0.0, 1.0)]
    mesh.insert_vertices(pts)

    triangles = mesh.triangles
    vertmap = np.arange(3 + len(pts), dtype=int)

    ebunch, hull = _edges_and_hull_from_cdt(triangles, vertmap)
    assert isinstance(ebunch, list) and len(ebunch) > 0
    assert isinstance(hull, list) and len(hull) > 0
    assert all(isinstance(n, (int, np.integer)) for n in hull)


def test_build_edge_line_tree_query_crosses():
    VertexC = np.array([
        [0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0],
        [0.0, 1.0], [2.0, 1.0], [3.0, 3.0], [4.0, 4.0],
    ], dtype=float)  # fmt: skip
    constraint_edges, _, tree = _build_edge_line_tree(VertexC, {(4, 5), (6, 7)})
    assert tree is not None
    probe = [shp.LineString(VertexC[[0, 1]]), shp.LineString(VertexC[[2, 3]])]
    pairs = tree.query(probe, predicate='crosses')

    assert constraint_edges == ((4, 5), (6, 7))
    assert pairs.shape == (2, 2)
    assert sorted(zip(*pairs.tolist(), strict=False)) == [(0, 0), (1, 0)]


def test_make_planar_embedding_marks_exact_los_crossing_without_border_path(
    monkeypatch,
):
    import optiwindnet.mesh as mesh_mod

    L = tiny_wfn(optimize=False).L
    original = mesh_mod.nx.single_source_dijkstra

    def fake_single_source_dijkstra(G, source, *args, **kwargs):
        lengths, paths = original(G, source, *args, **kwargs)
        if source == -1 and 1 in paths:
            assert isinstance(paths, dict)
            paths = dict(paths)
            paths[1] = [-1, 0, 1]  # pyrefly: ignore[unsupported-operation]
        return lengths, paths

    monkeypatch.setattr(
        mesh_mod.nx, 'single_source_dijkstra', fake_single_source_dijkstra
    )

    P, A = make_planar_embedding(L)
    VertexC = A.graph['VertexC']
    blocked = any(
        is_crossing(
            VertexC[-1],
            VertexC[1],
            VertexC[u],
            VertexC[v],
            touch_is_cross=False,
        )
        for u, v in P.graph['constraint_edges']
        if 1 not in (u, v)
    )

    assert blocked
    assert A.nodes[1]['los_d2root'][-1] > 0.0
