# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import math
from collections import defaultdict
from itertools import chain, combinations, tee
from typing import Literal, NewType

import condeltri as cdt
import networkx as nx
import numba as nb
import numpy as np
import shapely as shp
from bidict import bidict
from scipy.spatial.distance import cdist

from .geometric import (
    CoordPairs,
    Indices,
    apply_edge_exemptions,
    assign_root,
    complete_graph,
    find_edges_bbox_overlaps,
    is_crossing_no_bbox,
    is_triangle_pair_a_convex_quadrilateral,
    rotation_checkers_factory,
    triangle_AR,
)
from .utils import F

__all__ = ('make_planar_embedding', 'planar_flipped_by_routeset', 'delaunay')

_lggr = logging.getLogger(__name__)
debug, info, warn = _lggr.debug, _lggr.info, _lggr.warning

NULL = np.iinfo(int).min
_MAX_TRIANGLE_ASPECT_RATIO = 50.0

IndexTrios = NewType(
    'IndexTrios', np.ndarray[tuple[int, Literal[3]], np.dtype[np.int_]]
)


@nb.njit(cache=True)
def _index(array: Indices, item: np.int_) -> int:
    """Find the index of first occurrence of `item` in `array`.

    Equivalent of the method `index()` of Python lists for numpy arrays.

    Returns:
      index
    """
    for idx, val in enumerate(array):
        if val == item:
            return idx
    # value not found (must not happen, maybe should throw exception)
    # raise ValueError('value not found in array')
    return 0


@nb.njit(cache=True)
def _halfedges_from_triangulation(
    triangles: IndexTrios,
    neighbors: IndexTrios,
    halfedges: IndexTrios,
    ref_is_cw_: np.ndarray[tuple[int], np.dtype[np.bool_]],
) -> None:
    """Lists the neighbor-aware half-edges that represent a triangulation.

    Meant to be called from `mesh._planar_from_cdt_triangles()`. Inputs are
    derived from `PythonCDT.Triangulation().triangles`.

    Args:
        triangles: array of triangle.vertices for triangle in triangles
        neighbors: array of triangle.neighbors for triangle in triangles

    Returns:
        3 lists of half-edges to be passed to `networkx.PlanarEmbedding`
    """
    NULL_ = nb.int_(NULL)
    nodes_done = set()
    # add the first three nodes to process
    nodes_todo = {n: nb.int_(0) for n in triangles[0]}
    i = nb.int_(0)
    while nodes_todo:
        pivot, tri_idx_start = nodes_todo.popitem()
        tri = triangles[tri_idx_start]
        tri_nb = neighbors[tri_idx_start]
        pivot_idx = _index(tri, pivot)
        succ_start = tri[(pivot_idx + 1) % 3]
        nb_idx_start_reverse = (pivot_idx - 1) % 3
        succ_end = tri[(pivot_idx - 1) % 3]
        # first half-edge from `pivot`
        #  print('INIT', [pivot, succ_start])
        halfedges[i] = pivot, succ_start, NULL_
        i += 1
        nb_idx = pivot_idx
        ref = succ_start
        ref_is_cw = False
        while True:
            tri_idx = tri_nb[nb_idx]
            if tri_idx == NULL_:
                if not ref_is_cw:
                    # revert direction
                    ref_is_cw = True
                    #  print('REVE', [pivot, succ_end, ref], cw)
                    ref_is_cw_[i] = ref_is_cw
                    halfedges[i] = pivot, succ_end, succ_start
                    i += 1
                    ref = succ_end
                    tri_nb = neighbors[tri_idx_start]
                    nb_idx = nb_idx_start_reverse
                    continue
                else:
                    break
            tri = triangles[tri_idx]
            tri_nb = neighbors[tri_idx]
            pivot_idx = _index(tri, pivot)
            succ = tri[(pivot_idx - 1) % 3] if ref_is_cw else tri[(pivot_idx + 1) % 3]
            nb_idx = ((pivot_idx - 1) % 3) if ref_is_cw else pivot_idx
            #  print('NORM', [pivot, succ, ref], cw)
            ref_is_cw_[i] = ref_is_cw
            halfedges[i] = pivot, succ, ref
            i += 1
            if succ not in nodes_todo and succ not in nodes_done:
                nodes_todo[succ] = tri_idx
            if succ == succ_end:
                break
            ref = succ
        nodes_done.add(pivot)
    return


def _edges_and_hull_from_cdt(
    triangles: list[cdt.Triangle], vertmap: Indices
) -> list[tuple[int, int]]:
    """Get edges/hull from triangulation.

    THIS FUNCTION MAY BE IRRELEVANT, AS WE TYPICALLY NEED THE
    NetworkX.PlanarEmbedding ANYWAY, SO IT IS BETTER TO USE
    `_planar_from_cdt_triangles()` DIRECTLY, followed by `_hull_processor()`.

    Produces all the edges and a the convex hull (nodes) from a constrained
    Delaunay triangulation (via PythonCDT).

    Args:
      triangles: is a `PythonCDT.Triangulation().triangles` list
      vertmap: is a node number translation table, from CDT numbers to NetworkX

    Returns:
      list of edges that are sides of the triangles
      list of nodes of the convex hull (counter-clockwise)
    """
    tri_visited = set()
    hull_edges = {}

    def edges_from_tri(edge, tri_idx):
        # recursive function
        tri_visited.add(tri_idx)
        tri = triangles[tri_idx]
        b = (set(tri.vertices) - edge).pop()
        idx_b = tri.vertices.index(b)
        idx_c = (idx_b + 1) % 3
        idx_a = (idx_b - 1) % 3
        a = tri.vertices[idx_a]
        AB = tri.neighbors[idx_a]
        c = tri.vertices[idx_c]
        BC = tri.neighbors[idx_b]
        check_hull = (a < 3, b < 3, c < 3)
        if sum(check_hull) == 1:
            if check_hull[0]:
                hull_edges[c] = b
            elif check_hull[1]:
                hull_edges[a] = c
            else:
                hull_edges[b] = a
        branches = [
            (new_edge, nb_idx)
            for new_edge, nb_idx in (((a, b), AB), ((b, c), BC))
            if nb_idx not in tri_visited
        ]
        for new_edge, nb_idx in branches:
            yield tuple(vertmap[new_edge,])
            if nb_idx not in tri_visited and nb_idx != cdt.NO_NEIGHBOR:
                yield from edges_from_tri(frozenset(new_edge), nb_idx)

    # TODO: Not sure if the starting triangle actually matters.
    #       Maybe it is ok to just remove the for-loop and use
    #       tri = triangles[0], tri_idx = 0
    for tri_idx, tri in enumerate(triangles):
        # make sure to start with a triangle not on the edge of supertriangle
        if cdt.NO_NEIGHBOR not in tri.neighbors:
            break
    edge_start = tuple(vertmap[tri.vertices[:2]])
    ebunch = [edge_start]
    ebunch.extend(edges_from_tri(frozenset(edge_start), tri_idx))
    assert len(tri_visited) == len(triangles)
    # Convert a sequence of hull edges into a sequence of hull nodes
    start, fwd = hull_edges.popitem()
    convex_hull = [vertmap[start]]
    while fwd != start:
        convex_hull.append(vertmap[fwd])
        fwd = hull_edges[fwd]
    return ebunch, convex_hull


def _planar_from_cdt_triangles(
    mesh: cdt.Triangulation, vertmap: Indices
) -> tuple[
    tuple[IndexTrios, np.ndarray[tuple[int], np.dtype[np.bool_]]], set[tuple[int, int]]
]:
    """Convert from a PythonCDT.Triangulation to NetworkX.PlanarEmbedding.

    For use within `make_planar_embedding()`. Wraps the numba-compiled
    `_halfedges_from_triangulation()`, which does the intensive work.

    Args:
      triangles: `PythonCDT.Triangulation().triangles` list
      vertmap: node number translation table, from CDT numbers to NetworkX

    Returns:
      planar embedding
    """
    num_tri = mesh.triangles_count()
    triangleI = np.empty((num_tri, 3), dtype=np.int_)
    neighborI = np.empty((num_tri, 3), dtype=np.int_)

    for i, tri in enumerate(mesh.triangles):
        triangleI[i] = vertmap[tri.vertices]
        neighborI[i] = tuple(
            (NULL if n == cdt.NO_NEIGHBOR else n) for n in tri.neighbors
        )
    # formula for number of triangulation's edges is: 3*V - H - 3
    # H = 3 since CDT's Hull is always the supertriangle
    # and because we count half-edges, use expression × 2
    num_half_edges = 6 * mesh.vertices_count() - 12
    halfedges = np.empty((num_half_edges, 3), dtype=np.int_)
    ref_is_cw_ = np.empty((num_half_edges,), dtype=np.bool_)
    _halfedges_from_triangulation(triangleI, neighborI, halfedges, ref_is_cw_)
    edges = set((u.item(), v.item()) for u, v in halfedges[:, :2] if u < v)
    return (halfedges, ref_is_cw_), edges


def _P_from_halfedge_pack(
    halfedge_pack: tuple[np.ndarray, np.ndarray],
) -> nx.PlanarEmbedding:
    """Build a nx.PlanarEmbedding from a pack of half-edges.

    For use within `make_planar_embedding()`.
    """
    halfedges, ref_is_cw_ = halfedge_pack
    P = nx.PlanarEmbedding()
    for (u, v, ref), ref_is_cw in zip(halfedges, ref_is_cw_):
        if ref == NULL:
            P.add_half_edge(u.item(), v.item())
        else:
            P.add_half_edge(
                u.item(), v.item(), **{('cw' if ref_is_cw else 'ccw'): ref.item()}
            )
    return P


def _hull_processor(
    P: nx.PlanarEmbedding,
    T: int,
    supertriangle: tuple[int, int, int],
    vertex2conc_id_map: dict[int, int],
    num_holes: int,
) -> tuple[list[int], list[tuple[int, int]], set[tuple[int, int]]]:
    """Find the convex hull and supertriangle-incident edges to remove.

    Iterate over the edges that form a triangle with one of supertriangle's
    vertices. This produces the node sequence that form the convex hull and
    marks for removal the portals that would enable a path to go around the
    outside of a concavity.

    Args:
      P: planar embedding to process
      T: number of terminals of P
      supertriangle: indices to the supertriangles' vertices
      vertex2cons_id_map: vertex index to concavity id map

    Returns:
      The convex hull, P edges to be removed and outer edges of concavities
    """
    a, b, c = supertriangle
    convex_hull = []
    conc_outer_edges = set()
    to_remove = []
    for pivot, begin, end in ((a, c, b), (b, a, c), (c, b, a)):
        debug('==== pivot %d ====', pivot)
        source, target = tee(P.neighbors_cw_order(pivot))
        outer = begin
        for u, v in zip(source, chain(target, (next(target),))):
            if u >= T and v >= T:
                if u == outer:
                    to_remove.append((pivot, u))
                    debug('del_sup %d %d', pivot, u)
                    outer = v
                elif v == end:
                    to_remove.append((pivot, v))
                    debug('del_sup %d %d', pivot, v)
                conc_id_u = vertex2conc_id_map.get(u)
                if (conc_id_u == vertex2conc_id_map.get(v, -1)) and (
                    conc_id_u >= num_holes
                ):
                    to_remove.append((u, v))
                    conc_outer_edges.add((u, v) if u < v else (v, u))
                    debug('del_int %d %d', u, v)
                    outer = v
            if u != begin and u != end and v != end:
                # if u is not in supertriangle, it is convex_hull
                convex_hull.append(u)
    return convex_hull, to_remove, conc_outer_edges


def make_planar_embedding(
    L: nx.Graph,
    offset_scale: float = 1e-4,
    max_tri_AR: float = _MAX_TRIANGLE_ASPECT_RATIO,
) -> tuple[nx.PlanarEmbedding, nx.Graph]:
    """Triangulate a location and produce graphs P and A for it.

    P is the planar embedding mesh and A is the available-edges graph.
    TODO: change the name of this function.

    Args:
      L: locations graph
      offset_scale: Fraction of the diagonal of the site's bbox to use as
        spacing between border and nodes in concavities (only where nodes
        are the border).
      max_tri_AR: maximum aspect ratio allowed for triangles on the hull of
        A as the algorithm removes flat triangles (higher values keep more).

    Returns:
      P - the planar embedding graph - and A - the available-edges graph.
    """

    # ######
    # Steps:
    # ######
    # A) Scale the coordinates to avoid CDT errors.
    # B) Transform border concavities in polygons.
    # C) Check if concavities' vertices coincide with wtg. Where they do,
    #    create stunt concavity vertices to the inside of the concavity.
    # D) Create a miriad of indices and mappings.
    # E) Get Delaunay triangulation of the wtg+oss nodes only.
    # F) Build the available-edges graph A and its planar embedding.
    # G) Build the hull-concave.
    # H) Insert the obstacles' constraint edges.
    # I) Insert the hull's and concavities' constraint edges.
    # J) Add coordinates for stunts, supertriangle and scale back.
    # K) Build the planar embedding of the constrained triangulation.
    # L) Build P_paths.
    # M) Revisit A to update edges crossing borders with P_path contours.
    # N) Revisit A to update d2roots according to lengths along P_paths.
    # O) Calculate the area of the concave hull.
    # P) Set A's graph attributes.

    R, T, B, VertexCʹ = (L.graph[k] for k in 'R T B VertexC'.split())
    border = L.graph.get('border')
    obstacles = L.graph.get('obstacles', [])

    # #############################################
    # A) Scale the coordinates to avoid CDT errors.
    # #############################################
    # Since the initialization of the Triangulation class is made with only
    # the wtg coordinates, there are cases where the later addition of border
    # coordinates generate an error. (e.g. Horns Rev 3)
    # This is caused by the supertriangle being calculated from the first
    # batch of vertices (wtg only), which turns out to be too small to fit the
    # border polygon.
    # CDT's supertriangle calculation has a fallback reference value of 1.0 for
    # vertex sets that fall within a small area. The way to circunvent the
    # error described above is to scale all coordinates down so that CDT will
    # use the fallback and this fallback is enough to cover the scaled borders.
    debug('PART A')
    mean = VertexCʹ.mean(axis=0)
    scale = 2.0 * max(VertexCʹ.max(axis=0) - VertexCʹ.min(axis=0))

    VertexS = (VertexCʹ - mean) / scale

    # ############################################
    # B) Transform border concavities in polygons.
    # ############################################
    debug('PART B')
    node_xy_ = tuple(
        (x.item(), y.item()) for (x, y) in chain(VertexS[:T], VertexS[-R:])
    )
    nodeset_xy_ = set(node_xy_[:-R])
    root_pts = shp.MultiPoint(node_xy_[-R:])
    if border is None:
        hull_minus_border = shp.MultiPolygon()
        border_vertex_from_xy = {}
        out_root_pts = shp.MultiPoint()
    else:
        border_vertex_from_xy = {
            tuple(VertexS[i].tolist()): i.item() for i in chain(border, *obstacles)
        }

        border_poly = shp.Polygon(shell=VertexS[border])
        out_root_pts = root_pts - border_poly
        hull_poly = (border_poly | root_pts).convex_hull
        hull_ring = hull_poly.boundary

        hull_border_xy_ = {
            xy for xy in hull_ring.coords[:-1] if xy in border_vertex_from_xy
        }
        hull_border_vertices = [border_vertex_from_xy[xy] for xy in hull_border_xy_]

        # Turn the main border's concave zones into concavity polygons.
        hull_minus_border = hull_poly - border_poly

    concavities = []
    if hull_minus_border.is_empty:
        assert out_root_pts.is_empty
    elif isinstance(hull_minus_border, shp.MultiPolygon):
        # MultiPolygon
        for p in hull_minus_border.geoms:
            if p.intersects(out_root_pts):
                border_poly |= p
            else:
                concavities.append(p.exterior)
    elif out_root_pts.is_empty:
        # single Polygon is a concavity
        concavities.append(hull_minus_border.exterior)
    else:
        # single Polygon in hull_minus_border includes a root -> no concavity
        border_poly = hull_poly

    holes = [shp.LinearRing(VertexS[obstacle]) for obstacle in obstacles]

    # ###################################################################
    # C) Check if concavities' vertices coincide with wtg. Where they do,
    #    create stunt concavity vertices to the inside of the concavity.
    # ###################################################################
    debug('PART C')
    offset = offset_scale * np.hypot(*(VertexS.max(axis=0) - VertexS.min(axis=0)))
    #  debug(f'offset: {offset}')
    stuntS = []
    stunts_primes = []
    remove_from_border_xy_map = set()
    B_old = B
    # replace coinciding vertices with stunts and save concavities here
    for rings, is_hole in ((concavities, False), (holes, True)):
        for i, ring in enumerate(rings):
            changed = False
            debug(
                'is_hole: %s, ring: %d, num_vertices: %d',
                is_hole,
                i,
                len(ring.coords) - 1,
            )
            stunt_coords = []
            new_ring_xy_ = []
            old_ring_xy_ = tuple(
                xy
                for xy in (
                    ring.coords[:-1]
                    if (not ring.is_ccw if is_hole else ring.is_ccw)
                    else ring.coords[-2::-1]
                )
            )
            debug('%s', old_ring_xy_)
            rev = old_ring_xy_[-1]
            X = border_vertex_from_xy[rev]
            X_is_hull = X in hull_border_vertices
            cur = old_ring_xy_[0]
            Y = border_vertex_from_xy[cur]
            Y_is_hull = Y in hull_border_vertices
            # X->Y->Z is in ccw direction
            for fwd in chain(old_ring_xy_[1:], (cur,)):
                Z = border_vertex_from_xy[fwd]
                Z_is_hull = fwd in hull_border_xy_
                if cur in nodeset_xy_:
                    # Concavity border vertex coincides with node.
                    # Therefore, create a stunt vertex for the border.
                    XY = VertexS[Y] - VertexS[X]
                    YZ = VertexS[Z] - VertexS[Y]
                    _XY_ = np.hypot(*XY)
                    _YZ_ = np.hypot(*YZ)
                    nXY = XY[::-1] / _XY_
                    nYZ = YZ[::-1] / _YZ_
                    # normal to XY, pointing inward
                    nXY[0] = -nXY[0]
                    # normal to YZ, pointing inward
                    nYZ[0] = -nYZ[0]
                    angle = np.arccos(np.dot(-XY, YZ) / _XY_ / _YZ_)
                    if abs(angle) < np.pi / 2:
                        # XYZ acute
                        debug('acute')
                        # project nXY on YZ
                        proj = YZ / _YZ_ / max(0.5, np.sin(abs(angle)))
                    else:
                        # XYZ obtuse
                        debug('obtuse')
                        # project nXY on YZ
                        proj = YZ * np.dot(nXY, YZ) / _YZ_**2
                    if Y_is_hull:
                        if X_is_hull:
                            debug('XY hull')
                            # project nYZ on XY
                            S = offset * (-XY / _XY_ / max(0.5, np.sin(angle)) - nXY)
                        else:
                            assert Z_is_hull
                            # project nXY on YZ
                            S = offset * (YZ / _YZ_ / max(0.5, np.sin(angle)) - nYZ)
                            debug('YZ hull')
                    else:
                        S = offset * (nYZ + proj)
                    debug('translation: %s', S)
                    # to extract stunts' coordinates:
                    # stuntsC = VertexS[T + B - len(stunts_primes): T + B]
                    stunts_primes.append(Y)
                    stunt_coord = VertexS[Y] + S
                    stunt_coords.append(stunt_coord)
                    stunt_xy = (stunt_coord[0].item(), stunt_coord[1].item())
                    new_ring_xy_.append(stunt_xy)
                    remove_from_border_xy_map.add(cur)
                    border_vertex_from_xy[stunt_xy] = T + B
                    B += 1
                    changed = True
                else:
                    new_ring_xy_.append(cur)
                X, X_is_hull = Y, Y_is_hull
                Y, Y_is_hull = Z, Z_is_hull
                Y_is_hull = fwd in hull_border_xy_
                cur = fwd
            if changed:
                info('Concavities changed: %d stunts', len(stunt_coords))
                new_conc = shp.LinearRing(new_ring_xy_)
                if is_hole:
                    holes[i] = new_conc
                else:
                    concavities[i] = new_conc
                #  stuntC.append(mean + scale*np.array(stunt_coords))
                stuntS.append(np.array(stunt_coords))
    # Stunts are added to the B range and they should be saved with routesets.
    # Alternatively, one could convert stunts to clones of their primes, but
    # this could create some small interferences between edges.
    if stuntS:
        info(
            'stuntS lengths: %s; former B: %d; new B: %d',
            [len(nc) for nc in stuntS],
            B_old,
            B,
        )

    for xy in remove_from_border_xy_map:
        info('removing %d', border_vertex_from_xy[xy])
        del border_vertex_from_xy[xy]

    # ###########################################
    # D) Create a miriad of indices and mappings.
    # ###########################################
    debug('PART D')

    node_vertex_from_xy = {
        (x.item(), y.item()): i
        for i, (x, y) in chain(
            enumerate(VertexS[:T]), enumerate(VertexS[-R:], start=-R)
        )
    }

    if not out_root_pts.is_empty:
        border = np.array(
            [
                (border_vertex_from_xy.get(xy) or node_vertex_from_xy[xy])
                for xy in border_poly.exterior.coords[:-1]
            ],
            dtype=int,
        )

    # count the points actually used in concavities and obstacles
    num_pt_concavities = sum(shp.count_coordinates(c) for c in concavities)
    num_pt_holes = sum(shp.count_coordinates(h) for h in holes)

    # account for the supertriangle vertices that cdt.Triangulation() adds
    supertriangle = (T + B, T + B + 1, T + B + 2)
    vertex_from_iCDT = np.full(
        (3 + T + R + num_pt_concavities + num_pt_holes,), NULL, dtype=int
    )
    vertex_from_iCDT[:3] = supertriangle
    V2d_nodes = []
    iCDT = NULL
    for iCDT, (xy, n) in enumerate(node_vertex_from_xy.items(), start=3):
        V2d_nodes.append(cdt.V2d(*xy))
        vertex_from_iCDT[iCDT] = n

    # Create a map vertex -> concavity/hole id
    # holes first
    vertex2conc_id_map = {
        border_vertex_from_xy[xy]: i
        for i, ring in enumerate(holes)
        for xy in ring.coords[:-1]
    }
    num_holes = len(holes)
    # then concavities
    if len(concavities) <= 1:
        vertex2conc_id_map |= {
            border_vertex_from_xy[xy]: i
            for i, ring in enumerate(concavities, start=num_holes)
            for xy in ring.coords[:-1]
        }
    else:  # multiple concavities
        # Concavities that share a common point are assigned the same id.
        stack = [
            (set(conc.coords[:-1]), tuple(conc.coords[:-1])) for conc in concavities
        ]
        ready = []
        while stack:
            refset, ref_xy_ = stack.pop()
            stable = True
            for iconc, (testset, test_xy_) in enumerate(stack):
                common = refset & testset
                if common:
                    (common,) = common
                    iref, itst = ref_xy_.index(common), test_xy_.index(common)
                    joined = (
                        ref_xy_[:iref]
                        + test_xy_[itst:]
                        + test_xy_[:itst]
                        + ref_xy_[iref:]
                    )
                    debug('common vertex: %s -> new contour: %s', common, joined)
                    del stack[iconc]
                    stack.append((refset | testset, joined))
                    stable = False
                    break
            if stable:
                ready.append(ref_xy_)
        vertex2conc_id_map |= {
            border_vertex_from_xy[xy]: i
            for i, xy_ in enumerate(ready, start=len(holes))
            for xy in xy_
        }

    # ########################################################
    # E) Get Delaunay triangulation of the wtg+oss nodes only.
    # ########################################################
    debug('PART E')
    # Create triangulation and add vertices and edges
    mesh = cdt.Triangulation(
        cdt.VertexInsertionOrder.AUTO, cdt.IntersectingConstraintEdges.NOT_ALLOWED, 0.0
    )
    mesh.insert_vertices(V2d_nodes)

    P_A_halfedge_pack, P_A_edges = _planar_from_cdt_triangles(mesh, vertex_from_iCDT)
    P_A = _P_from_halfedge_pack(P_A_halfedge_pack)
    P_A_edges.difference_update((u, v) for v in supertriangle for u in P_A[v])

    # ##############################################################
    # F) Build the available-edges graph A and its planar embedding.
    # ##############################################################
    debug('PART F')
    convex_hull_A = []
    a, b, c = supertriangle
    for pivot, begin, end in ((a, c, b), (b, a, c), (c, b, a)):
        # Circles pivot in cw order -> hull becomes ccw order.
        source, target = tee(P_A.neighbors_cw_order(pivot))
        for u, v in zip(source, chain(target, (next(target),))):
            if u != begin and u != end and v != end:
                convex_hull_A.append(u)
    debug('convex_hull_A: %s', '–'.join(F[n] for n in convex_hull_A))
    P_A.remove_nodes_from(supertriangle)

    # Prune flat triangles from P_A (criterion is aspect_ratio > `max_tri_AR`).
    # Also create a `hull_prunned`, a hull without the triangles (ccw order)
    # and a set of prunned hull edges.
    queue = list(
        zip(convex_hull_A[::-1], chain(convex_hull_A[0:1], convex_hull_A[:0:-1]))
    )
    hull_prunned = []
    hull_prunned_edges = set()
    while queue:
        u, v = queue.pop()
        n = P_A[u][v]['ccw']
        # P_A is a DiGraph, so there are 2 degrees per undirected edge
        if (
            P_A.degree[u] > 4
            and P_A.degree[v] > 4
            and triangle_AR(*VertexS[[u, v, n]]) > max_tri_AR
        ):
            P_A.remove_edge(u, v)
            queue.extend(((n, v), (u, n)))
            uv = (u, v) if u < v else (v, u)
            P_A_edges.remove(uv)
            continue
        hull_prunned.append(u)
        uv = (u, v) if u < v else (v, u)
        hull_prunned_edges.add(uv)
    u, v = hull_prunned[0], hull_prunned[-1]
    uv = (u, v) if u < v else (v, u)
    hull_prunned_edges.add(uv)
    debug('hull_prunned: %s', '–'.join(F[n] for n in hull_prunned))
    debug(
        'hull_prunned_edges: %s',
        ','.join(f'{F[u]}–{F[v]}' for u, v in hull_prunned_edges),
    )

    A = nx.Graph(P_A_edges)
    nx.set_edge_attributes(A, 'delaunay', name='kind')
    # TODO: ¿do we really need node attr kind? separate with test: node < 0
    nx.set_node_attributes(A, 'wtg', name='kind')
    for r in range(-R, 0):
        A.nodes[r]['kind'] = 'oss'

    # Extend A with diagonals.
    diagonals = bidict()
    for u, v in P_A_edges - hull_prunned_edges:
        uvD = P_A[u][v]
        s, t = uvD['cw'], uvD['ccw']

        # SANITY check (if hull edges were skipped, this should always hold)
        vuD = P_A[v][u]
        assert s == vuD['ccw'] and t == vuD['cw']

        if is_triangle_pair_a_convex_quadrilateral(*VertexS[[u, v, s, t]]):
            s, t = (s, t) if s < t else (t, s)
            diagonals[(s, t)] = (u, v)
            A.add_edge(s, t, kind='extended')

    # ##########################
    # G) Build the hull-concave.
    # ##########################
    debug('PART G')
    # prevent edges that cross the boudaries from going into PlanarEmbedding
    # an exception is made for edges that include a root node
    hull_concave = []
    if border is not None:
        hull_prunned_poly = shp.Polygon(shell=VertexS[hull_prunned])
        shp.prepare(hull_prunned_poly)
        shp.prepare(border_poly)
        pushed = 0
        if not border_poly.covers(hull_prunned_poly):
            hull_stack = hull_prunned[0:1] + hull_prunned[::-1]
            u, v = hull_prunned[-1], hull_stack.pop()
            while hull_stack:
                edge_line = shp.LineString(VertexS[[u, v]])
                if not border_poly.covers(edge_line):
                    t = P_A[u][v]['ccw']
                    #  print(f'[{pushed}]', F[u], F[v], f'⟨{F[t]}⟩', [F[n] for n in hull_stack[::-1]])
                    if t == u:
                        # degenerate case 1
                        hull_concave.append(v)
                        t, v, u = v, u, t
                        continue
                    pushed += 1
                    hull_stack.append(v)
                    if pushed and not any(n in A[t] for n in hull_stack[-pushed:]):
                        # TODO: figure out how to avoid repeated outlier nodes
                        warn(
                            'unable to include in hull_concave: %s',
                            ' '.join(F[n] for n in hull_stack[-pushed:]),
                        )
                        hull_outliers = A.graph.get('hull_outliers')
                        if hull_outliers is not None:
                            hull_outliers.extend(hull_stack[-pushed:])
                        else:
                            A.graph['hull_outliers'] = hull_stack[-pushed:]
                        del hull_stack[-pushed:]
                        pushed = 0
                        while hull_stack:
                            v = hull_stack.pop()
                            if v not in hull_concave:
                                break
                        continue
                    v = t
                else:
                    #  print(f'[{pushed}]', F[u], F[v], [F[n] for n in hull_stack[::-1]])
                    hull_concave.append(v)
                    u = v
                    if pushed:
                        pushed -= 1
                    v = hull_stack.pop()
    if not hull_concave:
        hull_concave = hull_prunned
    debug('hull_concave: %s', '–'.join(F[n] for n in hull_concave))

    # ##########################################
    # H) Insert the obstacles' constraint edges.
    # ##########################################
    debug('PART H')
    constraint_edges = set()
    edgesCDT_obstacles = []
    #  hard_constraints_xy_ = set()
    V2d_holes = []
    # add obstacles' edges
    debug('holes')
    for ring in holes:
        for s_xy, t_xy in zip(ring.coords[:-1], ring.coords[1:]):
            s, t = border_vertex_from_xy[s_xy], border_vertex_from_xy[t_xy]
            edge = []
            for n, xy in ((s, s_xy), (t, t_xy)):
                if xy not in node_vertex_from_xy:
                    iCDT += 1
                    vertex_from_iCDT[iCDT] = n
                    edge.append(iCDT - 3)
                    node_vertex_from_xy[xy] = n
                    V2d_holes.append(cdt.V2d(*xy))
                else:
                    n = node_vertex_from_xy[xy]
                    edge.append(np.flatnonzero(vertex_from_iCDT[3:] == n)[0].item())
            debug('s: %d, t: %d, sC: %s, tC: %s', s, t, s_xy, t_xy)
            st = (s, t) if s < t else (t, s)
            constraint_edges.add(st)
            edgesCDT_obstacles.append(cdt.Edge(*edge))

    debug('%s', constraint_edges)
    # if adding obstacles, crossing-free edges might be removed from the mesh
    justly_removed = set()
    soft_constraints = set()
    if edgesCDT_obstacles:
        VertexS = np.vstack(
            (
                VertexS[:-R],
                *stuntS,
                np.array([(v.x, v.y) for v in mesh.vertices[:3]]),
                VertexS[-R:],
            )
        )
        mesh.insert_vertices(V2d_holes)
        mesh.insert_edges(edgesCDT_obstacles)
        _, P_edges = _planar_from_cdt_triangles(mesh, vertex_from_iCDT)
        # Here we use the changes in CDT triangulation to identify the P_A
        # edges that cross obstacles or lay in their vicinity.
        edges_to_examine = P_A_edges - P_edges
        edges_check = np.array(list(constraint_edges))
        while edges_to_examine:
            u, v = edges_to_examine.pop()
            uC, vC = VertexS[[u, v]]
            # if ⟨u, v⟩ does not cross any constraint_edges, add it to edgesCDT
            ovlap = find_edges_bbox_overlaps(VertexS, u, v, edges_check)
            if not any(
                is_crossing_no_bbox(uC, vC, *VertexS[edge])
                for edge in edges_check[ovlap]
            ):
                # ⟨u, v⟩ was removed from the triangulation but does not cross
                soft_constraints.add((u, v))
            else:
                # ⟨u, v⟩ crosses some constraint_edge
                justly_removed.add((u, v))
                # enlist for examination the up to 4 edges surrounding ⟨u, v⟩
                for s, t in ((u, v), (v, u)):
                    nb = P_A[s][t]['cw']
                    if nb == P_A[t][s]['cw']:
                        for p, q in (
                            (nb, s) if nb < s else (s, nb),
                            ((nb, t) if nb < t else (t, nb)),
                        ):
                            if (p, q) not in soft_constraints and (
                                p,
                                q,
                            ) not in justly_removed:
                                edges_to_examine.add((p, q))
        if soft_constraints:
            # add the crossing-free edges around obstacles as constraints
            edgesCDT_soft = [
                cdt.Edge(u if u >= 0 else T + R + u, v if v >= 0 else T + R + v)
                for u, v in soft_constraints
            ]
            mesh.insert_edges(edgesCDT_soft)

    # #######################################################
    # I) Insert the hull's and concavities' constraint edges.
    # #######################################################
    debug('PART I')
    # create the PythonCDT edges
    edgesCDT_P_A = []

    # Add A's hull_concave as soft constraints to ensure A's edges remain in P.
    debug('hull_concave')
    for s, t in zip(hull_concave, hull_concave[1:] + [hull_concave[0]]):
        s, t = (s, t) if s < t else (t, s)
        if (s, t) in justly_removed or (s, t) in soft_constraints:
            # skip if ⟨s, t⟩ is known to cross an obstacle or was added earlier
            continue
        edgesCDT_P_A.append(
            cdt.Edge(s if s >= 0 else T + R + s, t if t >= 0 else T + R + t)
        )
    mesh.insert_edges(edgesCDT_P_A)

    edgesCDT_concavities = []
    V2d_concavities = []
    # add concavities' edges
    debug('concavities')
    for ring in concavities:
        for s_xy, t_xy in zip(ring.coords[:-1], ring.coords[1:]):
            s, t = border_vertex_from_xy[s_xy], border_vertex_from_xy[t_xy]
            edge = []
            for n, xy in ((s, s_xy), (t, t_xy)):
                if xy not in node_vertex_from_xy:
                    iCDT += 1
                    vertex_from_iCDT[iCDT] = n
                    edge.append(iCDT - 3)
                    node_vertex_from_xy[xy] = n
                    V2d_concavities.append(cdt.V2d(*xy))
                else:
                    n = node_vertex_from_xy[xy]
                    edge.append(np.flatnonzero(vertex_from_iCDT[3:] == n)[0])
            st = (s, t) if s < t else (t, s)
            constraint_edges.add(st)
            edgesCDT_concavities.append(cdt.Edge(*edge))

    if edgesCDT_concavities:
        mesh.insert_vertices(V2d_concavities)
        mesh.insert_edges(edgesCDT_concavities)

    # ############################################################
    # J) Add coordinates for stunts, supertriangle and scale back.
    # ############################################################
    # add any newly created plus the supertriangle's vertices to VertexC
    # note: B has already been increased by all stuntC lengths within the loop
    debug('PART J')
    supertriangleC = mean + scale * np.array([(v.x, v.y) for v in mesh.vertices[:3]])
    VertexC = np.vstack(
        (
            VertexCʹ[:-R],
            *(mean + scale * coord for coord in stuntS),
            supertriangleC,
            VertexCʹ[-R:],
        )
    )

    # Add length attribute to A's edges.
    A_edges = np.array((*P_A_edges, *diagonals))
    length_ = np.hypot(*(VertexC[A_edges[:, 0]] - VertexC[A_edges[:, 1]]).T)
    is_terminal_ = A_edges >= 0
    inter_terminal_mask = np.logical_and(is_terminal_[:, 0], is_terminal_[:, 1])
    inter_terminal_clearance_ = length_[inter_terminal_mask]
    inter_terminal_clearance_min = np.min(inter_terminal_clearance_).item()
    inter_terminal_clearance_safe = np.quantile(inter_terminal_clearance_, 0.1).item()

    A_edge_length = dict(
        zip(map(tuple, A_edges), (length.item() for length in length_))
    )
    nx.set_edge_attributes(A, A_edge_length, name='length')

    # ###############################################################
    # K) Build the planar embedding of the constrained triangulation.
    # ###############################################################
    debug('PART K')
    P_halfedge_pack, P_edges = _planar_from_cdt_triangles(mesh, vertex_from_iCDT)
    P = _P_from_halfedge_pack(P_halfedge_pack)

    # Remove edges inside the concavities
    for ring in chain(concavities, holes):
        cw_or_ccw = 'ccw' if ring.is_ccw else 'cw'
        vertices = tuple(border_vertex_from_xy[xy] for xy in ring.coords)
        rev = vertices[-2]
        for cur, fwd in zip(vertices[:-1], vertices[1:]):
            while P[cur][fwd][cw_or_ccw] != rev:
                u, v = cur, P[cur][fwd][cw_or_ccw]
                P.remove_edge(u, v)
                P_edges.remove((u, v) if u < v else (v, u))
            rev = cur

    # adjust flat triangles around concavities
    #  changes_super = _flip_triangles_obstacles_super(
    #          P, T, B + 3, VertexC, max_tri_AR=max_tri_AR)

    convex_hull, to_remove, conc_outer_edges = _hull_processor(
        P, T, supertriangle, vertex2conc_id_map, num_holes
    )
    P.remove_edges_from(to_remove)
    P_edges.difference_update((u, v) if u < v else (v, u) for u, v in to_remove)
    constraint_edges -= conc_outer_edges
    P.graph.update(
        R=R,
        T=T,
        B=B,
        constraint_edges=constraint_edges,
        supertriangleC=supertriangleC,
    )

    #  changes_obstacles = _flip_triangles_near_obstacles(P, T, B + 3,
    #                                                       VertexC)
    #  P.check_structure()
    #  print('changes_super', [(F[a], F[b]) for a, b in changes_super])
    #  print('changes_obstacles',
    #        [(F[a], F[b]) for a, b in changes_obstacles])

    #  print('&'*80 + '\n', P_A.edges - P.edges, '\n' + '&'*80)
    #  print('\n' + '&'*80)
    #
    #  # Favor the triagulation in P_A over the one in P where possible.
    #  for u, v in P_A.edges - P.edges:
    #      print(F[u], F[v])
    #      s, t = P_A[u][v]['cw'], P_A[u][v]['ccw']
    #      if (s == P_A[v][u]['ccw']
    #              and t == P_A[v][u]['cw']
    #              and (s, t) in P.edges):
    #          w, x = P[s][t]['cw'], P[s][t]['ccw']
    #          if (w == u and x == v
    #                  and w == P[t][s]['ccw']
    #                  and x == P[t][s]['cw']):
    #              print(F[u], F[v], 'replaces', F[s], F[t])
    #              P.add_half_edge(u, v, ccw=t)
    #              P.add_half_edge(v, u, ccw=s)
    #              P.remove_edge(s, t)
    #      #  else:
    #      #      print(F[u], F[v], 'not in P, but', F[s], F[t],
    #      #            'not available for flipping')
    #  print('&'*80)

    # #################
    # L) Build P_paths.
    # #################
    debug('PART L')
    P_edges.difference_update((u, v) for v in supertriangle for u in P[v])
    P_paths = nx.Graph(P_edges)

    # this adds diagonals to P_paths, but not diagonals that cross constraints
    for u, v in P_edges - hull_prunned_edges:
        if (u, v) in constraint_edges:
            continue
        uvD = P[u][v]
        s, t = uvD['cw'], uvD['ccw']
        if is_triangle_pair_a_convex_quadrilateral(*VertexC[[u, v, s, t]]):
            P_paths.add_edge(s, t)

    nx.set_edge_attributes(P_paths, A_edge_length, name='length')
    for u, v, edgeD in P_paths.edges(data=True):
        if 'length' not in edgeD:
            edgeD['length'] = np.hypot(*(VertexC[u] - VertexC[v])).item()

    # ###################################################################
    # M) Revisit A to update edges crossing borders with P_path contours.
    # ###################################################################
    debug('PART M')
    cw, ccw = rotation_checkers_factory(VertexC)
    corner_to_A_edges = defaultdict(list)
    A_edges_to_revisit = []
    for u, v in A.edges - P_paths.edges:
        # For the edges in A that are not in P, we find their corresponding
        # shortest path in P_path and update the length attribute in A.
        length, path = nx.bidirectional_dijkstra(P_paths, u, v, weight='length')
        debug('A_edge: %s–%s length: %.3f; path: %s', F[u], F[v], length, path)
        if all(n >= T for n in path[1:-1]):
            # keep only paths that only have border vertices between nodes
            edgeD = A[path[0]][path[-1]]
            midpath = path[1:-1].copy() if u < v else path[-2:0:-1].copy()
            i = 0
            while i <= len(path) - 3:
                # Check if each vertex at the border is necessary.
                # The vertex is kept if the border angle and the path angle
                # point to the same side. Otherwise, remove the vertex.
                s, b, t = path[i : i + 3]
                b_conc_id = vertex2conc_id_map[b]
                # skip to shortcut if b is in a concavity and is a neighbor of
                # the supertriangle
                if b_conc_id < num_holes or all(n not in P[b] for n in supertriangle):
                    debug(
                        's: %s; b: %s; t: %s; b_conc_id: %s',
                        F[s],
                        F[b],
                        F[t],
                        b_conc_id,
                    )
                    debug([(F[n], vertex2conc_id_map.get(n)) for n in P.neighbors(b)])
                    nbs = P.neighbors_cw_order(b)
                    skip_test = True
                    for a in nbs:
                        if vertex2conc_id_map.get(a, -1) == b_conc_id:
                            skip_test = False
                            break
                    if skip_test:
                        i += 1
                        debug('Took the 1st continue.')
                        continue
                    skip_test = True
                    for c in nbs:
                        if vertex2conc_id_map.get(c, -1) == b_conc_id and c not in P[a]:
                            if P[b][a]['cw'] == c:
                                skip_test = False
                                break
                            a = c
                    if c == a:
                        # no nb remaining after making a = c, c <- first nb
                        c = next(P.neighbors_cw_order(b))
                        if P[b][a]['cw'] == c:
                            skip_test = False
                    debug(
                        'a: %d %s; c: %d %s; s: %d %s, t: %d %s; %s',
                        a,
                        F[a],
                        c,
                        F[c],
                        s,
                        F[s],
                        t,
                        F[t],
                        skip_test,
                    )
                    if skip_test or not (
                        cw(a, b, c) or ((a == s or cw(a, b, s)) == cw(s, b, t))
                    ):
                        i += 1
                        debug('Took the 2nd continue.')
                        continue
                # PERFORM SHORTCUT
                # TODO: The entire new path should go for a 2nd pass if it
                #       changed here. Unlikely to change in the 2nd pass.
                #       Reason: a shortcut may change the geometry in such
                #       way as to make additional shortcuts possible.
                del path[i + 1]
                length -= P_paths[s][b]['length'] + P_paths[b][t]['length']
                shortcut_length = np.hypot(*(VertexC[s] - VertexC[t]).T).item()
                length += shortcut_length
                # changing P_paths for the case of revisiting this block
                P_paths.add_edge(s, t, length=shortcut_length)
                shortcuts = edgeD.get('shortcuts')
                if shortcuts is None:
                    edgeD['shortcuts'] = [b]
                else:
                    shortcuts.append(b)
                debug('(%d) %s %s %s shortcut', i, F[s], F[b], F[t])
            edgeD.update(  # midpath-> which P edges the A edge maps to
                # (so that PathFinder works)
                midpath=midpath,
                # contour_... edges may include direct ones that are
                # diverted because P_paths does not include them
                kind='contour_' + edgeD['kind'],
            )
            if len(path) > 2:
                edgeD['length'] = length
                u, v = (u, v) if u < v else (v, u)
                for p in path[1:-1]:
                    corner_to_A_edges[p].append((u, v))
        else:
            # remove edge because the path goes through some wtg node
            u, v = (u, v) if u < v else (v, u)
            A.remove_edge(u, v)
            if (u, v) in diagonals:
                del diagonals[(u, v)]
            else:
                # Some edges will need revisiting to maybe promote their
                # diagonals to delaunay edges.
                A_edges_to_revisit.append((u, v))

    # Diagonals in A which have a missing origin Delaunay edge become edges.
    promoted_diagonal_from_parent_node = {}
    P_A_edges_to_remove = []
    for uv in A_edges_to_revisit:
        st = diagonals.inv.get(uv)
        if st is not None:
            # prevent promotion of two diagonals of the same triangle
            promote_st = True
            for n in uv:
                promoted = promoted_diagonal_from_parent_node.get(n)
                if promoted is not None:
                    (w, y), o = promoted
                    if (
                        (y, n) in P_A.edges
                        or (y, o) in P_A.edges
                        or (w, n) in P_A.edges
                        or (w, o) in P_A.edges
                    ) and (w in uv or y in uv):
                        # st & promoted are diagonals of the same triangle
                        if (w, y) not in diagonals.inv:
                            diagonals[st] = w, y
                        else:
                            warn(
                                'Delaunay edge %s already has a diagonal. '
                                'Unable to add «%d–%d» as its diagonal',
                                st,
                                w,
                                y,
                            )
                        promote_st = False
            if promote_st:
                edgeD = A.edges[st]
                edgeD['kind'] = 'contour_delaunay' if 'midpath' in edgeD else 'delaunay'
                del diagonals[st]
                u, v = uv
                promoted_diagonal_from_parent_node[u] = st, v
                promoted_diagonal_from_parent_node[v] = st, u
                s, t = st
                w, y = (u, v) if P_A[u][v]['cw'] == s else (v, u)
                P_A.add_half_edge(s, t, cw=y)
                P_A.add_half_edge(t, s, cw=w)
        P_A_edges_to_remove.append(uv)
    for uv in P_A_edges_to_remove:
        P_A.remove_edge(*uv)

    # ##################################################################
    # N) Revisit A to update d2roots according to lengths along P_paths.
    # ##################################################################
    debug('PART N')
    d2roots = cdist(VertexC[: T + B + 3], VertexC[-R:])
    # d2roots may not be the plain Euclidean distance if there are obstacles.
    if concavities or obstacles:
        # Use P_paths to obtain estimates of d2roots taking into consideration
        # the concavities and obstacle zones.
        for r in range(-R, 0):
            lengths, paths = nx.single_source_dijkstra(P_paths, r, weight='length')
            for n, path in paths.items():
                if n >= T or n < 0 or all(p < T for p in path[1:-1]):
                    # skip border and root vertices and paths without borders
                    continue
                debug('updating d2root of ⟨%d, %d⟩ (path %s)', r, n, path)
                node_d2roots = A.nodes[n].get('d2roots')
                if node_d2roots is None:
                    A.nodes[n]['d2roots'] = {r: d2roots[n, r]}
                else:
                    node_d2roots.update({r: d2roots[n, r]})
                new_length = 0.0
                s = path[0]
                for t in (p for p in path[1:-1] if p >= T):
                    # only add the lengths between constraint vertices
                    new_length += np.hypot(*(VertexC[s] - VertexC[t]).T).item()
                    s = t
                new_length += np.hypot(*(VertexC[s] - VertexC[n]).T).item()
                d2roots[n, r] = new_length

    # ##########################################
    # O) Calculate the area of the concave hull.
    # ##########################################
    debug('PART O')
    if border is None:
        bX, bY = VertexC[convex_hull_A].T
    else:
        # for the bounding box, use border and roots
        bX, bY = np.vstack((VertexC[border], VertexC[-R:])).T
    # assuming that coordinates are UTM -> min() as bbox's offset to origin
    norm_offset = np.array((bX.min(), bY.min()), dtype=np.float64)
    hull_concaveC = VertexC[hull_concave + hull_concave[0:1]]
    semi_perimeter = np.hypot(*(hull_concaveC[1:] - hull_concaveC[:-1]).T).sum() / 2
    # Shoelace formula for area (https://stackoverflow.com/a/30408825/287217).
    area_hull = 0.5 * abs(
        np.dot(hull_concaveC[:-1, 0], hull_concaveC[1:, 1])
        - np.dot(hull_concaveC[:-1, 1], hull_concaveC[1:, 0])
    )
    sqrt_area_hull = math.sqrt(area_hull)
    # Derive a scaling factor from some property of the concave hull
    if sqrt_area_hull < 1e-4 * semi_perimeter:
        # the concave hull is essentially a line with area close to zero
        # derive the scaling factor of coordinates from the semi-perimeter
        norm_scale = 1.0 / semi_perimeter
    else:
        # derive the scaling factor of coordinates so that the scaled area is 1
        norm_scale = 1.0 / sqrt_area_hull

    # ############################
    # P) Set A's graph attributes.
    # ############################
    debug('PART P')
    A.graph.update(
        T=T,
        R=R,
        B=B,
        VertexC=VertexC,
        border=border,
        name=L.name,
        handle=L.graph.get('handle', 'handleless'),
        planar=P_A,
        diagonals=diagonals,
        d2roots=d2roots,
        corner_to_A_edges=corner_to_A_edges,
        # TODO: make these 2 attribute names consistent across the code
        hull=convex_hull_A,
        hull_prunned=hull_prunned,
        hull_concave=hull_concave,
        # experimental attr
        norm_offset=norm_offset,
        norm_scale=norm_scale,
        inter_terminal_clearance_min=inter_terminal_clearance_min,
        inter_terminal_clearance_safe=inter_terminal_clearance_safe,
    )
    if obstacles:
        A.graph['obstacles'] = obstacles
    if stunts_primes:
        A.graph['stunts_primes'] = stunts_primes
    landscape_angle = L.graph.get('landscape_angle')
    if landscape_angle is not None:
        A.graph['landscape_angle'] = landscape_angle
    # products:
    # P: PlanarEmbedding
    # A: Graph (carries the updated VertexC)
    #   P_A: PlanarEmbedding
    #   diagonals: bidict
    return P, A


def delaunay(L: nx.Graph, bind2root: bool = False) -> nx.Graph:
    # TODO: deprecate the use of delaunay()
    """DEPRECATED. Create the extended-Delaunay-based available-edges graph A.

    Args:
      L: location
      bind2root: assign edge attribute 'root' (used by legacy heuristics)

    Returns:
      A - available-edges graph
    """
    _, A = make_planar_embedding(L)
    if bind2root:
        assign_root(A)
        R = L.graph['R']
        # assign each edge to the root closest to the edge's middle point
        VertexC = A.graph['VertexC']
        for u, v, edgeD in A.edges(data=True):
            edgeD['root'] = -R + np.argmin(
                cdist(((VertexC[u] + VertexC[v]) / 2)[np.newaxis, :], VertexC[-R:])
            )
    return A


def A_graph(G_base, delaunay_based=True, weightfun=None, weight_attr='weight'):
    """DEPRECATED. Create the available-edges graph A.

    Migrate to `make_planar_embedding()`.

    Return the "available edges" graph that is the base for edge search in
    Esau-Williams. If `delaunay_based` is True, the edges are the expanded
    Delaunay triangulation, otherwise a complete graph is returned.

    This function is kept for backward-compatibility.
    """
    if delaunay_based:
        A = delaunay(G_base)
        if weightfun is not None:
            apply_edge_exemptions(A)
    else:
        A = complete_graph(G_base, include_roots=True)
        # intersections
        # I = get_crossings_list(np.array(A.edges()), VertexC)

    if weightfun is not None:
        for u, v, data in A.edges(data=True):
            data[weight_attr] = weightfun(data)

    # remove all gates from A
    # TODO: decide about this line
    # A.remove_edges_from(list(A.edges(range(-R, 0))))
    return A


def _deprecated_planar_flipped_by_routeset(
    G: nx.Graph, *, A: nx.Graph, planar: nx.PlanarEmbedding
) -> nx.PlanarEmbedding:
    """
    DEPRECATED

    Returns a modified PlanarEmbedding based on `planar`, where all edges used
    in `G` are edges of the output embedding. For this to work, all non-gate
    edges of `G` must be either edges of `planar` or one of `G`'s
    graph attribute 'diagonals'. In addition, `G` must be free of edge×edge
    crossings.
    """
    R, T, B, D, VertexC, border, obstacles = (
        G.graph.get(k) for k in ('R', 'T', 'B', 'D', 'VertexC', 'border', 'obstacles')
    )

    P = planar.copy()
    diagonals = A.graph['diagonals']
    P_A = A.graph['planar']
    seen_endpoints = set()
    for u, v in G.edges - P.edges:
        # update the planar embedding to include any Delaunay diagonals
        # used in G; the corresponding crossing Delaunay edge is removed
        u, v = (u, v) if u < v else (v, u)
        if u >= T:
            # we are in a redundant segment of a multi-segment path
            continue
        if v >= T and u not in seen_endpoints:
            uvA = G[u][v]['A_edge']
            seen_endpoints.add(uvA[0] if uvA[1] == u else uvA[1])
            print('path_uv:', F[u], F[v], '->', F[uvA[0]], F[uvA[1]])
            u, v = uvA if uvA[0] < uvA[1] else uvA[::-1]
            path_uv = [u] + A[u][v]['path'] + [v]
            # now ⟨u, v⟩ represents the corresponding edge in A
        else:
            path_uv = None
        st = diagonals.get((u, v))
        if st is not None:
            # ⟨u, v⟩ is a diagonal of Delaunay edge ⟨s, t⟩
            s, t = st
            path_st = A[s][t].get('path')
            if path_st is not None:
                # pick a proxy segment for checking existance of path in G
                source, target = (s, t) if s < t else (t, s)
                st = source, path_st[0]
                path_st = [source] + path_st + [target]
                # now st represents a corresponding segment in G of A's ⟨s, t⟩
            if st in G.edges and s >= 0:
                if u >= 0:
                    print(
                        'ERROR: both Delaunay st and diagonal uv are in G, '
                        'but uv is not gate. Edge×edge crossing!'
                    )
                # ⟨u, v⟩ & ⟨s, t⟩ are in G (i.e. a crossing). This means
                # the diagonal ⟨u, v⟩ is a gate and ⟨s, t⟩ should remain
                continue
            if u < 0:
                # uv is a gate: any diagonals crossing it should prevail.
                # ensure u–s–v–t is ccw
                u, v = (
                    (u, v)
                    if (P_A[u][t]['cw'] == s and P_A[v][s]['cw'] == t)
                    else (v, u)
                )
                # examine the two triangles ⟨s, t⟩ belongs to
                crossings = False
                for a, b, c in ((s, t, u), (t, s, v)):
                    # this is for diagonals crossing diagonals
                    d = planar[c][b]['ccw']
                    diag_da = (a, d) if a < d else (d, a)
                    if (
                        d == planar[b][c]['cw']
                        and diag_da in diagonals
                        and diag_da[0] >= 0
                    ):
                        path_da = A[d][a].get('path')
                        if path_da is not None:
                            diag_da = ((d if d < a else a), path_da[0])
                        crossings = crossings or diag_da in G.edges
                    e = planar[a][c]['ccw']
                    diag_eb = (e, b) if e < b else (b, e)
                    if (
                        e == planar[c][a]['cw']
                        and diag_eb in diagonals
                        and diag_eb[0] >= 0
                    ):
                        path_eb = A[e][b].get('path')
                        if path_eb is not None:
                            diag_eb = ((e if e < b else b), path_eb[0])
                        crossings = crossings or diag_eb in G.edges
                if crossings:
                    continue
            # ⟨u, v⟩ is not crossing any edge in G
            # TODO: THIS NEEDS CHANGES: use paths
            #       it gets really complicated if the paths overlap!
            if path_st is None:
                P.remove_edge(s, t)
            else:
                for s, t in zip(path_st[:-1], path_st[1:]):
                    P.remove_edge(s, t)
            if path_uv is None:
                P.add_half_edge(u, v, ccw=s)
                P.add_half_edge(v, u, ccw=t)
            else:
                for u, v in zip(path_uv[:-1], path_uv[1:]):
                    P.add_half_edge(u, v, ccw=s)
                    P.add_half_edge(v, u, ccw=t)
    return P


def planar_flipped_by_routeset(
    G: nx.Graph,
    *,
    planar: nx.PlanarEmbedding,
    VertexC: CoordPairs,
    diagonals: bidict | None = None,
) -> nx.PlanarEmbedding:
    """Ajust `planar` to include the edges actually used by routeset `G`.

    Copies `planar` and flips the edges to their diagonal if the latter is an
    edge of `G`. Ideally, the returned PlanarEmbedding includes all `G` edges
    (an expected discrepancy are `G`'s gates).

    If `diagonals` is provided, some diagonal gates may become `planar`'s edges
    if they are not crossing any edge in `G`. Otherwise gates are ignored.

    Important: `G` must be free of edge×edge crossings.
    """
    R, T, B, C, D = (G.graph.get(k, 0) for k in ('R', 'T', 'B', 'C', 'D'))
    fnT = G.graph.get('fnT')
    if fnT is None:
        fnT = np.arange(R + T + B + 3 + C + D)
        fnT[-R:] = range(-R, 0)

    P = planar.copy()
    if diagonals is not None:
        diags = diagonals.copy()
    else:
        diags = ()
    debug('differences between G and P:')
    # get G's edges in terms of node range -R : T + B
    edges_G = {
        ((u, v) if u < v else (v, u)) for u, v in (fnT[edge,] for edge in G.edges)
    }
    ST = T + B
    edges_P = {((u, v) if u < v else (v, u)) for u, v in P.edges if u < ST and v < ST}
    stack = list(edges_G - edges_P)
    # gates to the bottom of the stack
    stack.sort()
    while stack:
        u, v = stack.pop()
        if u < 0 and (u, v) not in diags:
            continue
        debug('%d–%d', u, v)
        intersection = set(planar[u]) & set(planar[v])
        if len(intersection) < 2:
            debug('share %d neighbors.', len(intersection))
            continue
        diagonal_found = False
        for s, t in combinations(intersection, 2):
            s, t = (s, t) if s < t else (t, s)
            if (s, t) in edges_P and is_triangle_pair_a_convex_quadrilateral(
                *VertexC[[s, t, u, v]]
            ):
                diagonal_found = True
                break
        if not diagonal_found:
            if u >= 0:
                # only warn if the non-planar is not a gate
                warn('Failed to find flippable for non-planar %d–%d', u, v)
            continue
        if (s, t) in edges_G and u < 0:
            # not replacing edge with gate
            continue
        if planar[u][s]['ccw'] == t and planar[v][t]['ccw'] == s:
            # u-s-v-t already in ccw orientation
            pass
        elif planar[u][s]['cw'] == t and planar[v][t]['cw'] == s:
            # reassign so that u-s-v-t is in ccw orientation
            s, t = t, s
        else:
            debug('%d–%d–%d–%d is not in two triangles.', u, s, v, t)
            continue
        #  if not (s == planar[v][u]['ccw']
        #          and t == planar[v][u]['cw']):
        #      print(f'{F[u]}–{F[v]} is not in two triangles')
        #      continue
        #  if (s, t) not in planar:
        #      print(f'{F[s]}–{F[t]} is not in planar')
        #      continue
        debug('flipping %d–%d to %d–%d', s, t, u, v)
        P.remove_edge(s, t)
        if diags:
            # diagonal (u_, v_) is added to P -> forbid diagonals that cross it
            for w, y in ((u, s), (s, v), (v, t), (t, u)):
                wy = (w, y) if w < y else (y, w)
                diags.inv.pop(wy, None)
        P.add_half_edge(u, v, cw=s)
        P.add_half_edge(v, u, cw=t)
    return P
