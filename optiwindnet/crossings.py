import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
from bidict import bidict
import shapely as shp

from .geometric import angle_helpers, is_bunch_split_by_corner, is_same_side
from .interarraylib import calcload


@dataclass(frozen=True)
class _RoutePolyline:
    nodes: tuple[int, ...]
    role: str
    non_root_end: int | None = None


def get_interferences_list(
    Edge: np.ndarray, VertexC: np.ndarray, fnT: np.ndarray | None = None, EPSILON=1e-15
) -> list[tuple[tuple[int, int, int, int], int | None]]:
    """List all crossings between edges in the `Edge` (E×2) numpy array.

    Coordinates must be provided in the `VertexC` (V×2) array.

    `Edge` contains indices to VertexC. If `Edge` includes detour nodes
    (i.e. indices go beyond `VertexC`'s length), `fnT` translation table
    must be provided.

    Should be used when edges are not limited to the expanded Delaunay set.

    Returns:
      list of interferences, where each interference is:
        ((4 vertices of the two edges involved), one of the vertices or None)
        the last tuple element indicates the index (0..3) of the vertex that
        lays exactly on the edge in cases of touching (not crossing)
    """
    crossings = []
    if fnT is None:
        V = VertexC[Edge[:, 1]] - VertexC[Edge[:, 0]]
    else:
        V = VertexC[fnT[Edge[:, 1]]] - VertexC[fnT[Edge[:, 0]]]
    for i, ((UVx, UVy), (u, v)) in enumerate(zip(V[:-1], Edge[:-1].tolist())):
        u_, v_ = (u, v) if fnT is None else fnT[[u, v]]
        (uCx, uCy), (vCx, vCy) = VertexC[[u_, v_]]
        for (STx, STy), (s, t) in zip(-V[i + 1 :], Edge[i + 1 :].tolist()):
            s_, t_ = (s, t) if fnT is None else fnT[[s, t]]
            if s_ == u_ or t_ == u_ or s_ == v_ or t_ == v_:
                # <edges have a common node>
                continue
            # bounding box check
            (sCx, sCy), (tCx, tCy) = VertexC[[s_, t_]]

            # X
            lo, hi = (vCx, uCx) if UVx < 0 else (uCx, vCx)
            if STx > 0:  # s - t > 0 -> hi: s, lo: t
                if hi < tCx or sCx < lo:
                    continue
            else:  # s - t < 0 -> hi: t, lo: s
                if hi < sCx or tCx < lo:
                    continue

            # Y
            lo, hi = (vCy, uCy) if UVy < 0 else (uCy, vCy)
            if STy > 0:
                if hi < tCy or sCy < lo:
                    continue
            else:
                if hi < sCy or tCy < lo:
                    continue

            # TODO: save the edges that have interfering bounding boxes
            #       to be checked in a vectorized implementation of
            #       the math below
            UV = UVx, UVy
            ST = STx, STy

            # denominator
            f = STx * UVy - STy * UVx
            # TODO: verify if this arbitrary tolerance is appropriate
            if math.isclose(f, 0.0, abs_tol=1e-5):
                # segments are parallel
                # TODO: there should be check for branch splitting in parallel
                #       cases with touching points
                continue

            C = uCx - sCx, uCy - sCy
            touch_found = []
            Xcount = 0
            for k, num in enumerate(
                (Px * Qy - Py * Qx) for (Px, Py), (Qx, Qy) in ((C, ST), (UV, C))
            ):
                if f > 0:
                    if -EPSILON <= num <= f + EPSILON:  # num < 0 or f < num:
                        Xcount += 1
                        if math.isclose(num, 0, abs_tol=EPSILON):
                            touch_found.append(2 * k)
                        if math.isclose(num, f, abs_tol=EPSILON):
                            touch_found.append(2 * k + 1)
                else:
                    if f - EPSILON <= num <= EPSILON:  # 0 < num or num < f:
                        Xcount += 1
                        if math.isclose(num, 0, abs_tol=EPSILON):
                            touch_found.append(2 * k)
                        if math.isclose(num, f, abs_tol=EPSILON):
                            touch_found.append(2 * k + 1)

            if Xcount == 2:
                # segments cross or touch
                uvst = (u, v, s, t)
                if touch_found:
                    assert len(touch_found) == 1, 'ERROR: too many touching points.'
                    #  p = uvst[touch_found[0]]
                    p = touch_found[0]
                else:
                    p = None
                crossings.append((uvst, p))
    return crossings


def edge_crossings(
    u: int, v: int, G: nx.Graph, diagonals: bidict
) -> list[tuple[int, int]]:
    u, v = (u, v) if u < v else (v, u)
    st = diagonals.get((u, v))
    conflicting = []
    if st is None:
        # ⟨u, v⟩ is a Delaunay edge
        st = diagonals.inv.get((u, v))
        if st is not None and st[0] >= 0:
            conflicting.append(st)
    else:
        # ⟨u, v⟩ is a diagonal of Delanay edge ⟨s, t⟩
        s, t = st
        # crossing with Delaunay edge
        conflicting.append(st)

        # two triangles may contain ⟨s, t⟩, each defined by their non-st vertex
        for hat in (u, v):
            for diag in (
                diagonals.inv.get((w, y) if w < y else (y, w))
                for w, y in ((s, hat), (hat, t))
            ):
                if diag is not None and diag[0] >= 0:
                    conflicting.append(diag)
    return [edge for edge in conflicting if edge in G.edges]


def edgeset_edgeXing_iter(diagonals: bidict) -> Iterator[list[tuple[int, int]]]:
    """Iterator over all edge crossings in an expanded Delaunay edge set `A`.

    Each crossing is a 2 or 3-tuple of (u, v) edges. Does not include gates.
    """
    checked = set()
    for (u, v), (s, t) in diagonals.items():
        # ⟨u, v⟩ is a diagonal of Delaunay ⟨s, t⟩
        if u < 0:
            # diagonal is a gate
            continue
        uv = (u, v)
        if s >= 0:
            # crossing with Delaunay edge
            yield [(s, t), uv]
        # two triangles may contain ⟨s, t⟩, each defined by their non-st vertex
        for hat in uv:
            triangle = tuple(sorted((s, t, hat)))
            if triangle in checked:
                continue
            checked.add(triangle)
            conflicting = [uv]
            for diag in (
                diagonals.inv.get((w, y) if w < y else (y, w))
                for w, y in ((s, hat), (hat, t))
            ):
                if diag is not None and diag[0] >= 0:
                    conflicting.append(diag)
            if len(conflicting) > 1:
                yield conflicting


def gateXing_iter(
    G: nx.Graph,
    *,
    hooks: Iterable | None = None,
    touch_is_cross: bool = True,
) -> Iterator[tuple[tuple[int, int], tuple[int, int]]]:
    """Iterate over all crossings between gates and edges/borders in G.

    If `hooks` is `None`, all nodes that are not a root neighbor are
    considered. Used in constraint generation for ILP model.

    Args:
      G: Routeset or edgeset (A) to examine.
      hooks: Nodes to check, grouped by root in sub-sequences from root `-R`
        to `-1`. If `None`, all non-root nodes are checked using `'root'`
        node attribute.
      touch_is_cross: If `True`, count as crossing a gate going over a node.

    Yields:
      Pair of (edge, gate) that cross (each a 2-tuple of nodes).
    """
    R, T, VertexC = (G.graph[k] for k in ('R', 'T', 'VertexC'))
    fnT = G.graph.get('fnT')
    roots = range(-R, 0)
    angle_rank__ = G.graph.get('angle_rank__', None)
    if angle_rank__ is None:
        angle__, angle_rank__, _ = angle_helpers(G)
    else:
        angle__ = G.graph['angle__']
    # TODO: There is a corner case here: for multiple roots, the gates are not
    #       being checked between different roots. Unlikely but possible case.
    # iterable of non-gate edges:
    Edge = nx.subgraph_view(G, filter_node=lambda n: n >= 0).edges()
    if hooks is None:
        all_nodes = np.arange(T)
        IGate = [all_nodes] * R
    else:
        IGate = hooks
    # it is important to consider touch as crossing
    # because if a gate goes precisely through a node
    # there will be nothing to prevent it from spliting
    # that node's subtree
    less = np.less_equal if touch_is_cross else np.less
    for u, v in Edge:
        if fnT is not None:
            u, v = fnT[u], fnT[v]
        uC = VertexC[u]
        vC = VertexC[v]
        for root, iGate in zip(roots, IGate):
            angle_ = angle__[:, root]
            rank_ = angle_rank__[:, root]
            rootC = VertexC[root]
            uvA = angle_[v] - angle_[u]
            swaped = (-np.pi < uvA) & (uvA < 0.0) | (np.pi < uvA)
            lo, hi = (v, u) if swaped else (u, v)
            loR, hiR = rank_[lo], rank_[hi]
            pR_ = rank_[iGate]
            W = loR > hiR  # wraps +-pi
            supL = less(loR, pR_)  # angle(low) <= angle(probe)
            infH = less(pR_, hiR)  # angle(probe) <= angle(high)
            is_rank_within = ~W & supL & infH | W & ~supL & infH | W & supL & ~infH
            for n in iGate[np.flatnonzero(is_rank_within)].tolist():
                # this test confirms the crossing because `is_rank_within`
                # established that root–n is on a line crossing u–v
                if n == u or n == v:
                    continue
                if not is_same_side(uC, vC, rootC, VertexC[n]):
                    u, v = (u, v) if u < v else (v, u)
                    yield (u, v), (root, n)


def validate_routeset(G: nx.Graph) -> list[tuple[int, int, int, int]]:
    """Validate G's tree topology and absence of crossings.

    Check if the routeset represented by G's edges is topologically sound,
    repects capacity and has no edge crossings nor branch splitting.

    Args:
      G: graph to evaluate

    Returns:
      list of crossings/splits, G is valid if an empty list is returned

    Example::

      Xings = validate_routeset(G)
        for u, v, s, t in Xings:
          if u != v:
            print(f'{u}–{v} crosses {s}–{t}')
          else:
            print(f'detour @ {u} splits {s}–{v}–{t}')

    """
    R, T, B = (G.graph[k] for k in 'RTB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    VertexC = G.graph['VertexC']
    if C > 0 or D > 0:
        fnT = G.graph['fnT']
    else:
        fnT = np.arange(T + R)
        fnT[-R:] = range(-R, 0)

    # TOPOLOGY check: is it a proper tree?
    calcload(G)

    # TOPOLOGY check: is load within capacity?
    max_load = G.graph['max_load']
    capacity = G.graph.get('capacity')
    if capacity is not None:
        assert max_load <= capacity, f'κ = {capacity}, max_load= {max_load}'
    else:
        capacity = G.graph['capacity'] = max_load

    # check edge×edge crossings
    #  Edge = np.array(tuple((fnT[u], fnT[v]) for u, v in G.edges))
    XTings = get_interferences_list(np.array(G.edges), VertexC, fnT)
    # parallel is considered no crossing
    # analyse cases of touch
    Xings = []
    for uvst, p in XTings:
        if p is None:
            Xings.append(uvst)
            continue
        if G.degree[p] == 1:
            # trivial case: no way to break a branch apart
            continue
        # make u be the touch-point within ⟨s, t⟩
        u = uvst[p]
        s, t = uvst[2:] if p < 2 else uvst[:2]

        u_, s_, t_ = fnT[(u, s, t),].tolist()
        bunch = [fnT[nb].item() for nb in G[u]]
        is_split, insideI, outsideI = is_bunch_split_by_corner(
            VertexC[bunch], *VertexC[[s_, u_, t_]]
        )
        if is_split:
            Xings.append((s_, t_, bunch[insideI[0]], bunch[outsideI[0]]))

    # check detour nodes for branch-splitting
    d_start = T + B + C
    for d, d_ in enumerate(fnT[d_start : d_start + D].tolist(), start=d_start):
        if d_ >= T or G.degree[d_] == 1:
            # either the detour node is over a border vertex or the node is a leaf:
            #   no branch splitting possible
            continue
        dA, dB = (fnT[nb] for nb in G[d])
        bunch = [fnT[nb].item() for nb in G[d_]]
        is_split, insideI, outsideI = is_bunch_split_by_corner(
            VertexC[bunch], *VertexC[[dA, d_, dB]]
        )
        if is_split:
            Xings.append((d_, d_, bunch[insideI[0]], bunch[outsideI[0]]))
        # assert not is_split, \
        #     f'Detour around node {d_} splits a branch; ' \
        #     f'inside: {[bunch[i] for i in insideI]}; ' \
        #     f'outside: {[bunch[i] for i in outsideI]}'
    return Xings


def _routeset_fnT(G: nx.Graph) -> np.ndarray:
    R, T, B = (G.graph[k] for k in 'RTB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    if C > 0 or D > 0:
        return G.graph['fnT']
    fnT = np.arange(T + B + R)
    fnT[-R:] = range(-R, 0)
    return fnT


def _prime_node_path(
    G: nx.Graph, path: tuple[int, ...], fnT: np.ndarray
) -> tuple[int, ...]:
    prime_path = tuple(int(fnT[n]) for n in path)
    R = G.graph['R']
    trimmed = prime_path
    while len(trimmed) > 1 and -R <= trimmed[0] < 0:
        trimmed = trimmed[1:]
    while len(trimmed) > 1 and -R <= trimmed[-1] < 0:
        trimmed = trimmed[:-1]
    if len(trimmed) > 1:
        prime_path = trimmed
    if len(prime_path) > 1 and prime_path[0] < prime_path[-1]:
        prime_path = prime_path[::-1]
    return prime_path


def _maximal_graph_paths(G: nx.Graph) -> list[tuple[int, ...]]:
    visited = set()
    paths = []

    def step(prev: int, node: int) -> list[int]:
        path = [node]
        while G.degree[node] == 2:
            candidates = [nb for nb in G[node] if nb != prev]
            if len(candidates) != 1:
                break
            prev, node = node, candidates[0]
            edge = (prev, node) if prev < node else (node, prev)
            if edge in visited:
                break
            path.append(node)
        return path

    for u, v in G.edges:
        edge = (u, v) if u < v else (v, u)
        if edge in visited:
            continue
        path = tuple(reversed(step(v, u))) + tuple(step(u, v))
        for a, b in zip(path[:-1], path[1:]):
            visited.add((a, b) if a < b else (b, a))
        paths.append(path)
    return paths


def _routeset_polylines(G: nx.Graph) -> list[_RoutePolyline]:
    R = G.graph['R']
    roots = set(range(-R, 0))
    if max(dict(G.degree()).values(), default=0) <= 2:
        return [
            _RoutePolyline(path, 'path')
            for path in _maximal_graph_paths(G)
        ]

    visited = set()
    polylines: list[_RoutePolyline] = []

    def edge_key(u: int, v: int) -> tuple[int, int]:
        return (u, v) if u < v else (v, u)

    def walk(prev: int, node: int) -> tuple[int, ...]:
        path = [prev, node]
        visited.add(edge_key(prev, node))
        while node not in roots and G.degree[node] == 2:
            candidates = [nb for nb in G[node] if nb != prev]
            if len(candidates) != 1:
                break
            nxt = candidates[0]
            key = edge_key(node, nxt)
            if key in visited:
                break
            prev, node = node, nxt
            path.append(node)
            visited.add(key)
        return tuple(path)

    for root in sorted(roots):
        for nb in G[root]:
            path = walk(root, nb)
            polylines.append(_RoutePolyline(path, 'feeder', path[-1]))

    starts = [n for n in G if n not in roots and G.degree[n] != 2]
    for start in starts:
        for nb in G[start]:
            if nb in roots:
                continue
            if edge_key(start, nb) in visited:
                continue
            path = walk(start, nb)
            polylines.append(_RoutePolyline(path, 'link'))

    return polylines


def _path_coords(
    VertexC: np.ndarray, fnT: np.ndarray, path: tuple[int, ...]
) -> list[tuple[float, float]]:
    coords = []
    for node in path:
        coord = tuple(float(c) for c in VertexC[int(fnT[node])])
        if not coords or coord != coords[-1]:
            coords.append(coord)
    return coords


def _common_subpath_span(
    path_a: tuple[int, ...], path_b: tuple[int, ...]
) -> tuple[int, int, int, int, tuple[int, ...]] | None:
    """Find a shared node run of at least one segment between two paths."""
    best: tuple[int, int, int, int, tuple[int, ...]] | None = None
    best_len = 1
    for candidate_b in (path_b, path_b[::-1]):
        for i, node_a in enumerate(path_a):
            for j, node_b in enumerate(candidate_b):
                if node_a != node_b:
                    continue
                length = 1
                while (
                    i + length < len(path_a)
                    and j + length < len(candidate_b)
                    and path_a[i + length] == candidate_b[j + length]
                ):
                    length += 1
                if length > best_len:
                    best_len = length
                    best = i, i + length, j, j + length, candidate_b
    return best


def _cross_sign(a: np.ndarray, b: np.ndarray, eps: float = 1e-15) -> int:
    cross = a[0] * b[1] - a[1] * b[0]
    if abs(cross) <= eps:
        return 0
    return 1 if cross > 0 else -1


def _shared_run_has_crossing_cones(
    path_a: tuple[int, ...], path_b: tuple[int, ...], VertexC: np.ndarray
) -> bool:
    """True when approach and separation cones have the same orientation."""
    span = _common_subpath_span(path_a, path_b)
    if span is None:
        return False
    start_a, end_a, start_b, end_b, oriented_b = span
    if not (
        0 < start_a
        and end_a < len(path_a)
        and 0 < start_b
        and end_b < len(oriented_b)
    ):
        return False

    shared_start = path_a[start_a]
    shared_end = path_a[end_a - 1]
    approach_a = VertexC[shared_start] - VertexC[path_a[start_a - 1]]
    approach_b = VertexC[shared_start] - VertexC[oriented_b[start_b - 1]]
    separation_a = VertexC[path_a[end_a]] - VertexC[shared_end]
    separation_b = VertexC[oriented_b[end_b]] - VertexC[shared_end]

    approach_sign = _cross_sign(approach_a, approach_b)
    separation_sign = _cross_sign(separation_a, separation_b)
    return approach_sign != 0 and approach_sign == separation_sign


def _point_segment_distance(pC: np.ndarray, aC: np.ndarray, bC: np.ndarray) -> float:
    ab = bC - aC
    denom = np.dot(ab, ab)
    if denom == 0.0:
        return np.hypot(*(pC - aC)).item()
    t = np.clip(np.dot(pC - aC, ab) / denom, 0.0, 1.0)
    closest = aC + t * ab
    return np.hypot(*(pC - closest)).item()


def _unique_rays(rays: list[np.ndarray], angle_tol: float) -> list[np.ndarray]:
    unique = []
    for ray in rays:
        norm = np.hypot(*ray).item()
        if norm == 0.0:
            continue
        unit = ray / norm
        if all(
            abs(unit[0] * other[1] - unit[1] * other[0]) > angle_tol
            or np.dot(unit, other) < 0
            for other in unique
        ):
            unique.append(unit)
    return unique


def _local_rays_at_point(
    coords: list[tuple[float, float]],
    point: tuple[float, float],
    *,
    endpoint_tol: float,
    angle_tol: float,
) -> list[np.ndarray]:
    pC = np.array(point)
    rays = []
    scale = max(
        (
            np.hypot(*(np.array(b) - np.array(a))).item()
            for a, b in zip(coords[:-1], coords[1:])
        ),
        default=1.0,
    )
    tol = endpoint_tol * max(scale, 1.0)
    for a, b in zip(coords[:-1], coords[1:]):
        aC = np.array(a)
        bC = np.array(b)
        if _point_segment_distance(pC, aC, bC) > tol:
            continue
        if np.hypot(*(aC - pC)).item() > tol:
            rays.append(aC - pC)
        if np.hypot(*(bC - pC)).item() > tol:
            rays.append(bC - pC)
    return _unique_rays(rays, angle_tol)


def _rays_alternate(
    rays_a: list[np.ndarray], rays_b: list[np.ndarray], angle_tol: float
) -> bool:
    if len(rays_a) < 2 or len(rays_b) < 2:
        return False
    for pair_a in combinations(rays_a, 2):
        if abs(pair_a[0][0] * pair_a[1][1] - pair_a[0][1] * pair_a[1][0]) <= angle_tol:
            continue
        for pair_b in combinations(rays_b, 2):
            if (
                abs(pair_b[0][0] * pair_b[1][1] - pair_b[0][1] * pair_b[1][0])
                <= angle_tol
            ):
                continue
            ordered = sorted(
                (
                    (math.atan2(ray[1], ray[0]) % (2 * math.pi), label)
                    for label, pair in (('a', pair_a), ('b', pair_b))
                    for ray in pair
                )
            )
            labels = [label for _, label in ordered]
            if labels in (['a', 'b', 'a', 'b'], ['b', 'a', 'b', 'a']):
                return True
    return False


def _polyline_crosses_at_point(
    coords_a: list[tuple[float, float]],
    coords_b: list[tuple[float, float]],
    point: tuple[float, float],
    *,
    angle_tol: float,
    endpoint_tol: float,
) -> bool:
    pC = np.array(point)
    scale = max(
        max(
            (
                np.hypot(*(np.array(b) - np.array(a))).item()
                for a, b in zip(coords[:-1], coords[1:])
            ),
            default=1.0,
        )
        for coords in (coords_a, coords_b)
    )
    tol = endpoint_tol * max(scale, 1.0)
    all_vertices = tuple(np.array(coord) for coord in (*coords_a, *coords_b))
    if any(np.hypot(*(pC - vertexC)).item() <= tol for vertexC in all_vertices):
        return False
    rays_a = _local_rays_at_point(
        coords_a, point, endpoint_tol=endpoint_tol, angle_tol=angle_tol
    )
    rays_b = _local_rays_at_point(
        coords_b, point, endpoint_tol=endpoint_tol, angle_tol=angle_tol
    )
    return _rays_alternate(rays_a, rays_b, angle_tol)


def _detour_node_set(G: nx.Graph) -> set[int]:
    T, B = (G.graph[k] for k in 'TB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    return set(range(T + B + C, T + B + C + D))


def _angle_to_bunch(aC: np.ndarray, oC: np.ndarray, bunchC: np.ndarray) -> np.ndarray:
    A = aC - oC
    B = bunchC - oC
    return np.arctan2(A[0] * B[:, 1] - A[1] * B[:, 0], B @ A)


def _bunch_split_by_corner(
    bunchC: np.ndarray,
    aC: np.ndarray,
    oC: np.ndarray,
    bC: np.ndarray,
    *,
    margin: float = 1e-3,
) -> tuple[bool, np.ndarray, np.ndarray]:
    angle_a = _angle_to_bunch(aC, oC, bunchC)
    angle_b = _angle_to_bunch(bC, oC, bunchC)
    keep = ~np.logical_or(
        np.isclose(angle_a, 0, atol=margin),
        np.isclose(angle_b, 0, atol=margin),
    )

    a = aC - oC
    b = bC - oC
    angle_ab = math.atan2(a[0] * b[1] - a[1] * b[0], np.dot(a, b))
    if angle_ab > 0:
        in_a = angle_a > 0
        in_b = angle_b < 0
    else:
        in_a = angle_a < 0
        in_b = angle_b > 0
    inside = np.logical_and(keep, np.logical_and(in_a, in_b))
    outside = np.logical_and(keep, np.logical_or(~in_a, ~in_b))
    return bool(any(inside) and any(outside)), np.flatnonzero(inside), np.flatnonzero(
        outside
    )


def _detour_split_at_node(
    G: nx.Graph,
    node: int,
    fnT: np.ndarray,
    VertexC: np.ndarray,
) -> tuple[int, tuple[int, int]] | None:
    prime = int(fnT[node])
    if prime not in G or G.degree[prime] == 1 or G.degree[node] != 2:
        return None
    dA, dB = (int(fnT[nb]) for nb in G[node])
    bunch = [int(fnT[nb]) for nb in G[prime]]
    is_split, insideI, outsideI = _bunch_split_by_corner(
        VertexC[bunch], *VertexC[[dA, prime, dB]]
    )
    if not is_split:
        return None
    return prime, (bunch[insideI[0]], bunch[outsideI[0]])


def _point_is_geometric_crossing(
    G: nx.Graph,
    path_nodes_a: tuple[int, ...],
    path_nodes_b: tuple[int, ...],
    coords_a: list[tuple[float, float]],
    coords_b: list[tuple[float, float]],
    point: tuple[float, float],
    fnT: np.ndarray,
    VertexC: np.ndarray,
    *,
    angle_tol: float,
    endpoint_tol: float,
) -> bool:
    if _point_is_at_shared_node(
        point,
        path_nodes_a,
        path_nodes_b,
        fnT,
        VertexC,
        endpoint_tol=endpoint_tol,
    ):
        return False

    if _point_is_detour_branch_split(
        G, path_nodes_a, point, fnT, VertexC, endpoint_tol=endpoint_tol
    ) or _point_is_detour_branch_split(
        G, path_nodes_b, point, fnT, VertexC, endpoint_tol=endpoint_tol
    ):
        return False

    if _point_is_common_graph_endpoint(
        point,
        path_nodes_a,
        path_nodes_b,
        fnT,
        VertexC,
        endpoint_tol=endpoint_tol,
    ):
        return False

    return _polyline_crosses_at_point(
        coords_a,
        coords_b,
        point,
        angle_tol=angle_tol,
        endpoint_tol=endpoint_tol,
    )


def _point_is_detour_branch_split(
    G: nx.Graph,
    path: tuple[int, ...],
    point: tuple[float, float],
    fnT: np.ndarray,
    VertexC: np.ndarray,
    *,
    endpoint_tol: float,
) -> bool:
    detour_nodes = _detour_node_set(G)
    pC = np.array(point)
    for node in path[1:-1]:
        if node not in detour_nodes:
            continue
        split = _detour_split_at_node(G, node, fnT, VertexC)
        if split is None:
            continue
        prime, _ = split
        if np.hypot(*(pC - VertexC[prime])).item() <= endpoint_tol:
            return True
    return False


def _detour_branch_split_findings(
    G: nx.Graph,
    polylines: list[_RoutePolyline],
    prime_paths: list[tuple[int, ...]],
    fnT: np.ndarray,
    VertexC: np.ndarray,
) -> list[dict[str, Any]]:
    detour_nodes = _detour_node_set(G)
    if not detour_nodes:
        return []

    findings: list[dict[str, Any]] = []
    T = G.graph['T']
    for polyline, prime_path in zip(polylines, prime_paths):
        path = polyline.nodes
        if not any(node in detour_nodes for node in path):
            continue
        non_root_end = (
            int(fnT[polyline.non_root_end])
            if polyline.non_root_end is not None
            else None
        )
        seen_primes = set()
        for node in path[1:-1]:
            if node not in detour_nodes:
                continue
            prime = int(fnT[node])
            if prime in seen_primes or not 0 <= prime < T or prime == non_root_end:
                continue
            split = _detour_split_at_node(G, node, fnT, VertexC)
            if split is None:
                continue
            prime, split_nodes = split
            seen_primes.add(prime)
            point = (float(VertexC[prime][0]), float(VertexC[prime][1]))
            findings.append(
                {
                    'kind': 'branch_split',
                    'path_nodes_a': path,
                    'path_nodes_b': split_nodes,
                    'edge_pairs': [(path, split_nodes)],
                    'path_a': prime_path,
                    'path_b': (prime, prime, *split_nodes),
                    'split_node': prime,
                    'geometry': shp.Point(point),
                }
            )
    return findings


def _intersection_is_only_at_shared_nodes(
    path_a: tuple[int, ...],
    path_b: tuple[int, ...],
    fnT: np.ndarray,
    VertexC: np.ndarray,
    intersection,
    *,
    endpoint_tol: float,
) -> bool:
    shared_nodes = set(path_a) & set(path_b)
    if not shared_nodes or intersection.length > 0:
        return False
    shared_coords = [VertexC[int(fnT[node])] for node in shared_nodes]
    points = list(_iter_geometry_points(intersection))
    if not points:
        return False
    for point in points:
        pC = np.array(point)
        if not any(
            np.hypot(*(pC - sharedC)).item() <= endpoint_tol
            for sharedC in shared_coords
        ):
            return False
    return True


def _point_is_at_shared_node(
    point: tuple[float, float],
    path_a: tuple[int, ...],
    path_b: tuple[int, ...],
    fnT: np.ndarray,
    VertexC: np.ndarray,
    *,
    endpoint_tol: float,
) -> bool:
    shared_nodes = set(path_a) & set(path_b)
    if not shared_nodes:
        return False
    pC = np.array(point)
    return any(
        np.hypot(*(pC - VertexC[int(fnT[node])])).item() <= endpoint_tol
        for node in shared_nodes
    )


def _point_is_common_graph_endpoint(
    point: tuple[float, float],
    path_nodes_a: tuple[int, ...],
    path_nodes_b: tuple[int, ...],
    fnT: np.ndarray,
    VertexC: np.ndarray,
    *,
    endpoint_tol: float,
) -> bool:
    endpoint_primes = {
        int(fnT[path_nodes_a[0]]),
        int(fnT[path_nodes_a[-1]]),
        int(fnT[path_nodes_b[0]]),
        int(fnT[path_nodes_b[-1]]),
    }
    pC = np.array(point)
    return any(
        np.hypot(*(pC - VertexC[prime])).item() <= endpoint_tol
        for prime in endpoint_primes
    )


def _iter_geometry_points(geometry) -> Iterator[tuple[float, float]]:
    if geometry.geom_type == 'Point':
        yield geometry.x, geometry.y
    elif geometry.geom_type == 'MultiPoint':
        for point in geometry.geoms:
            yield point.x, point.y
    elif geometry.geom_type in {'LineString', 'LinearRing'}:
        yield from geometry.coords
    elif geometry.geom_type.startswith('Multi') or geometry.geom_type == (
        'GeometryCollection'
    ):
        for part in geometry.geoms:
            yield from _iter_geometry_points(part)


def _iter_point_geometries(geometry) -> Iterator[tuple[float, float]]:
    if geometry.geom_type == 'Point':
        yield geometry.x, geometry.y
    elif geometry.geom_type == 'MultiPoint':
        for point in geometry.geoms:
            yield point.x, point.y
    elif geometry.geom_type == 'GeometryCollection':
        for part in geometry.geoms:
            yield from _iter_point_geometries(part)


def _points_geometry(points: list[tuple[float, float]]):
    if len(points) == 1:
        return shp.Point(points[0])
    return shp.MultiPoint(points)


def full_geometric_crossings(
    G: nx.Graph,
    *,
    include_touches: bool = False,
    length_tol: float = 1e-12,
    angle_tol: float = 1e-10,
    endpoint_tol: float = 1e-9,
) -> list[dict]:
    """Find route intersections using Shapely geometries.

    This is a heavier, geometry-first diagnostic complement to
    :func:`validate_routeset`. Route polylines are translated through ``fnT`` so
    contour and detour clones are tested at their prime coordinates.

    Args:
      G: routeset graph.
      include_touches: include point contacts that are not proper crossings.
      length_tol: minimum shared length for a collinear overlap.
      angle_tol: minimum sine-like ray separation for local crossing tests.
      endpoint_tol: relative distance tolerance for endpoint/common-node touches.

    Returns:
      List of dictionaries with keys ``kind``, ``path_nodes_a``,
      ``path_nodes_b``, ``path_a``, ``path_b`` and ``geometry``.
    """
    VertexC = G.graph['VertexC']
    fnT = _routeset_fnT(G)
    polylines = _routeset_polylines(G)
    paths = [polyline.nodes for polyline in polylines]
    prime_paths = [_prime_node_path(G, path, fnT) for path in paths]
    path_coords = [_path_coords(VertexC, fnT, path) for path in paths]
    lines = [shp.LineString(coords) for coords in path_coords]
    tree = shp.STRtree(lines)
    findings = _detour_branch_split_findings(
        G,
        polylines,
        prime_paths,
        fnT,
        VertexC,
    )
    seen = set()

    for i, line_a in enumerate(lines):
        for j in tree.query(line_a, predicate='intersects').tolist():
            if j <= i:
                continue
            key = (i, j)
            if key in seen:
                continue
            seen.add(key)
            path_nodes_a = paths[i]
            path_nodes_b = paths[j]
            line_b = lines[j]
            intersection = line_a.intersection(line_b)
            if intersection.is_empty:
                continue
            if _intersection_is_only_at_shared_nodes(
                path_nodes_a,
                path_nodes_b,
                fnT,
                VertexC,
                intersection,
                endpoint_tol=endpoint_tol,
            ):
                continue
            path_a = prime_paths[i]
            path_b = prime_paths[j]
            kind = None
            finding_geometry = intersection
            if intersection.length > length_tol:
                if _shared_run_has_crossing_cones(path_a, path_b, VertexC):
                    kind = 'overlap_cross'
            if kind is None:
                crossing_points = [
                    point
                    for point in _iter_point_geometries(intersection)
                    if _point_is_geometric_crossing(
                        G,
                        path_nodes_a,
                        path_nodes_b,
                        path_coords[i],
                        path_coords[j],
                        point,
                        fnT,
                        VertexC,
                        angle_tol=angle_tol,
                        endpoint_tol=endpoint_tol,
                    )
                ]
                if crossing_points:
                    kind = 'cross'
                    finding_geometry = _points_geometry(crossing_points)
                elif include_touches:
                    kind = 'touch'
                else:
                    continue
                if kind == 'touch':
                    if include_touches:
                        kind = 'touch'
                    else:
                        continue

            if path_b < path_a:
                path_nodes_a, path_nodes_b = path_nodes_b, path_nodes_a
                path_a, path_b = path_b, path_a
            findings.append(
                {
                    'kind': kind,
                    'path_nodes_a': path_nodes_a,
                    'path_nodes_b': path_nodes_b,
                    'edge_pairs': [(path_nodes_a, path_nodes_b)],
                    'path_a': path_a,
                    'path_b': path_b,
                    'geometry': finding_geometry,
                }
            )

    merged: dict[tuple[str, tuple[int, ...], tuple[int, ...]], dict[str, Any]] = {}
    for finding in findings:
        key = finding['kind'], finding['path_a'], finding['path_b']
        if key not in merged:
            merged[key] = finding
            continue
        merged[key]['edge_pairs'].extend(finding['edge_pairs'])
        merged[key]['geometry'] = shp.union_all(
            [merged[key]['geometry'], finding['geometry']]
        )

    return [
        {**finding, 'geometry': finding['geometry'].wkt}
        for finding in merged.values()
    ]


def list_edge_crossings(
    S: nx.Graph, A: nx.Graph
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """List edge×edge crossings for the network topology in S.

    `S` must only use extended Delaunay edges. It will not detect crossings
    of non-extDelaunay gates or detours.

    Args:
      S: solution topology
      A: available edges used in creating `S`

    Returns:
      list of 2-tuple (crossing) of 2-tuple (edge, ordered)
    """
    eeXings = []
    checked = set()
    diagonals = A.graph['diagonals']
    for u, v in S.edges:
        u, v = (u, v) if u < v else (v, u)
        st = diagonals.get((u, v))
        if st is not None:
            # ⟨u, v⟩ is a diagonal of Delanay edge ⟨s, t⟩
            if st in S.edges:
                # crossing with Delaunay edge ⟨s, t⟩
                eeXings.append((st, (u, v)))
            s, t = st
            # ⟨s, t⟩ may be part of up to two triangles, check their 4 sides
            sides = (
                ((w, y) if w < y else (y, w))
                for w, y in ((u, s), (s, v), (v, t), (t, u))
            )
            for side in sides:
                diag = diagonals.inv.get(side, False)
                if diag and diag in S.edges and diag not in checked:
                    checked.add((u, v))
                    eeXings.append((diag, (u, v)))
    return eeXings
