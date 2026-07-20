import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import shapely as shp
from bidict import bidict

from .geometric import (
    angle_helpers,
    is_bunch_split_by_corner,
    is_same_side,
    polylines_cross_at_point,
)
from .interarraylib import S_from_G, calcload, validate_topology


@dataclass(frozen=True)
class _RoutePolyline:
    nodes: tuple[int, ...]
    non_root_end: int | None = None


def get_interferences_list(
    Edge: np.ndarray, VertexC: np.ndarray, fnT: np.ndarray | None = None, EPSILON=1e-15
) -> list[tuple[tuple[int, int, int, int], int | None]]:
    """List all crossings between edges in the ``Edge`` (E×2) numpy array.

    Coordinates must be provided in the ``VertexC`` (V×2) array.

    ``Edge`` contains indices to VertexC. If ``Edge`` includes detour nodes
    (i.e. indices go beyond ``VertexC``'s length), ``fnT`` translation table
    must be provided.

    Should be used when edges are not limited to the expanded Delaunay set.

    Returns:
      list of interferences, where each interference is:
        ((4 vertices of the two edges involved), one of the vertices or ``None``)
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


def edge_conflicts(u: int, v: int, diagonals: bidict) -> Iterator[tuple[int, int]]:
    """Iterate over edges conflicting with ``(u, v)``.

    Args:
      u: node
      v: node
      diagonals: map of crossings Delaunay↔diagonals
    """
    u, v = (u, v) if u < v else (v, u)
    st = diagonals.get((u, v))
    if st is None:
        # ⟨u, v⟩ is a Delaunay edge
        st = diagonals.inv.get((u, v))
        if st is not None and st[0] >= 0:
            yield st
    else:
        # ⟨u, v⟩ is a diagonal of Delanay edge ⟨s, t⟩
        # crossing with Delaunay edge
        yield st

        s, t = st
        # two triangles may contain ⟨s, t⟩, each defined by their non-st vertex
        for hat in (u, v):
            for diag in (
                diagonals.inv.get((w, y) if w < y else (y, w))
                for w, y in ((s, hat), (hat, t))
            ):
                if diag is not None and diag[0] >= 0:
                    yield diag


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
    """Iterator over all edge crossings in an expanded Delaunay edge set ``A``.

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

    If ``hooks`` is ``None``, all nodes that are not a root neighbor are
    considered. Used in constraint generation for ILP model.

    Args:
      G: Routeset or edgeset (A) to examine.
      hooks: Nodes to check, grouped by root in subsequences from root ``-R``
        to ``-1``. If ``None``, all non-root nodes are checked using ``'root'``
        node attribute.
      touch_is_cross: If ``True``, count as crossing a gate going over a node.

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

    # TOPOLOGY check: does the routeset keep the shape it declares?
    violations = validate_topology(S_from_G(G), capacity)
    assert not violations, '; '.join(violations)

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
    """Identity translation table (clones → primes); synthesized when G has none."""
    R, T, B = (G.graph[k] for k in 'RTB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    if C > 0 or D > 0:
        return G.graph['fnT']
    fnT = np.arange(T + B + R)
    fnT[-R:] = range(-R, 0)
    return fnT


def _canonical_prime_path(
    G: nx.Graph, path: tuple[int, ...], fnT: np.ndarray
) -> tuple[int, ...]:
    """Translate a polyline to primes, drop bordering roots, canonicalize direction."""
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


def _routeset_polylines(G: nx.Graph) -> list[_RoutePolyline]:
    """Walk G into one polyline per feeder plus one per inter-junction link.

    A feeder runs from a root through a degree-2 chain to the first leaf or
    branching node. A link runs between two non-degree-2 nodes that are not
    roots. Together these cover every edge exactly once.
    """
    R = G.graph['R']
    roots = set(range(-R, 0))
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
            polylines.append(_RoutePolyline(path, non_root_end=path[-1]))

    starts = [n for n in G if n not in roots and G.degree[n] != 2]
    for start in starts:
        for nb in G[start]:
            if nb in roots:
                continue
            if edge_key(start, nb) in visited:
                continue
            path = walk(start, nb)
            polylines.append(_RoutePolyline(path))

    return polylines


def _polyline_coords(
    VertexC: np.ndarray, fnT: np.ndarray, path: tuple[int, ...]
) -> np.ndarray:
    """(N, 2) coords of a path's primes, with consecutive duplicates collapsed."""
    raw = VertexC[fnT[list(path)]]
    if len(raw) <= 1:
        return raw
    keep = np.empty(len(raw), dtype=bool)
    keep[0] = True
    keep[1:] = np.any(raw[1:] != raw[:-1], axis=1)
    return raw[keep]


def _shared_run_swaps_sides(coords_a: np.ndarray, coords_b: np.ndarray) -> bool:
    """``True`` iff two polylines share an interior run and exit on the same side
    at each end.

    When two polylines overlap on a shared sub-sequence of vertices, an actual *cross*
    requires the two paths to enter the overlap from opposite half-planes and exit to
    opposite half-planes — equivalently, the orientation (cross-product sign) of the
    two approach vectors equals that of the two separation vectors.

    Operates on raw polyline coords (with consecutive duplicates collapsed) rather
    than canonical prime paths, so root-leg context preserved in the geometry —
    but trimmed from canonical prime paths — remains available here.
    """
    Na, Nb = len(coords_a), len(coords_b)
    if Na < 2 or Nb < 2:
        return False

    best = None
    best_len = 1
    for cb in (coords_b, coords_b[::-1]):
        for i in range(Na):
            for j in range(Nb):
                if not np.array_equal(coords_a[i], cb[j]):
                    continue
                length = 1
                while (
                    i + length < Na
                    and j + length < Nb
                    and np.array_equal(coords_a[i + length], cb[j + length])
                ):
                    length += 1
                if length > best_len:
                    best_len = length
                    best = (i, i + length, j, j + length, cb)
    if best is None:
        return False

    start_a, end_a, start_b, end_b, oriented_b = best
    # require at least one segment of context on each side of the shared run
    if not (0 < start_a and end_a < Na and 0 < start_b and end_b < Nb):
        return False

    shared_start = coords_a[start_a]
    shared_end = coords_a[end_a - 1]
    approach_a = shared_start - coords_a[start_a - 1]
    approach_b = shared_start - oriented_b[start_b - 1]
    separation_a = coords_a[end_a] - shared_end
    separation_b = oriented_b[end_b] - shared_end

    EPS = 1e-15

    def _cross_sign(u, v):
        c = u[0] * v[1] - u[1] * v[0]
        return 0 if abs(c) <= EPS else (1 if c > 0 else -1)

    approach_sign = _cross_sign(approach_a, approach_b)
    separation_sign = _cross_sign(separation_a, separation_b)
    return approach_sign != 0 and approach_sign == separation_sign


def _detour_splits(
    G: nx.Graph, fnT: np.ndarray, VertexC: np.ndarray
) -> dict[int, tuple[int, tuple[int, int]]]:
    """Map each detour clone whose route splits its prime's branch.

    Returns ``{detour_node: (prime, (inside_neighbor, outside_neighbor))}``. A
    detour clone splits a branch when its incoming/outgoing rays put the prime's
    other neighbours on opposite sides of the corner the detour cuts.
    """
    T, B = (G.graph[k] for k in 'TB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    if D == 0:
        return {}
    splits: dict[int, tuple[int, tuple[int, int]]] = {}
    for d in range(T + B + C, T + B + C + D):
        prime = int(fnT[d])
        if prime not in G or not 0 <= prime < T:
            continue
        if G.degree[prime] == 1 or G.degree[d] != 2:
            continue
        dA, dB = (int(fnT[nb]) for nb in G[d])
        bunch = [int(fnT[nb]) for nb in G[prime]]
        is_split, insideI, outsideI = is_bunch_split_by_corner(
            VertexC[bunch], *VertexC[[dA, prime, dB]]
        )
        if is_split:
            splits[d] = (prime, (bunch[insideI[0]], bunch[outsideI[0]]))
    return splits


def _branch_split_findings(
    splits: dict[int, tuple[int, tuple[int, int]]],
    polylines: list[_RoutePolyline],
    prime_paths: list[tuple[int, ...]],
    fnT: np.ndarray,
    VertexC: np.ndarray,
) -> list[dict[str, Any]]:
    """Emit one finding per polyline that traverses a detour-split prime."""
    if not splits:
        return []
    findings: list[dict[str, Any]] = []
    for polyline, prime_path in zip(polylines, prime_paths):
        non_root_end = (
            int(fnT[polyline.non_root_end])
            if polyline.non_root_end is not None
            else None
        )
        seen_primes: set[int] = set()
        for node in polyline.nodes[1:-1]:
            split = splits.get(node)
            if split is None:
                continue
            prime, split_nodes = split
            if prime in seen_primes or prime == non_root_end:
                continue
            seen_primes.add(prime)
            findings.append(
                {
                    'kind': 'branch_split',
                    'path_nodes_a': polyline.nodes,
                    'path_nodes_b': split_nodes,
                    'path_a': prime_path,
                    'path_b': (prime, prime, *split_nodes),
                    'split_node': prime,
                    'geometry': shp.Point(VertexC[prime].tolist()),
                }
            )
    return findings


def _exclusion_coords(
    path_a: tuple[int, ...],
    path_b: tuple[int, ...],
    fnT: np.ndarray,
    VertexC: np.ndarray,
    splits: dict[int, tuple[int, tuple[int, int]]],
) -> np.ndarray:
    """Coordinates where intersections are not crossings: shared nodes,
    endpoints of either polyline, and detour-split primes that both paths visit."""
    primes: set[int] = {int(fnT[n]) for n in path_a} & {int(fnT[n]) for n in path_b}
    primes |= {
        int(fnT[path_a[0]]),
        int(fnT[path_a[-1]]),
        int(fnT[path_b[0]]),
        int(fnT[path_b[-1]]),
    }
    for path in (path_a, path_b):
        for node in path[1:-1]:
            split = splits.get(node)
            if split is not None:
                primes.add(split[0])
    return VertexC[sorted(primes)]


def _iter_points(geometry) -> Iterator[tuple[float, float]]:
    """Yield (x, y) for each Point inside ``geometry``; line parts are skipped."""
    if geometry.geom_type == 'Point':
        yield geometry.x, geometry.y
    elif geometry.geom_type == 'MultiPoint':
        for point in geometry.geoms:
            yield point.x, point.y
    elif geometry.geom_type == 'GeometryCollection':
        for part in geometry.geoms:
            yield from _iter_points(part)


def _intersection_only_at_excluded(
    intersection,
    excluded: np.ndarray,
    *,
    endpoint_tol: float,
) -> bool:
    """``True`` if every Point of a length-0 intersection lies at an excluded coord."""
    if intersection.length > 0:
        return False
    points = list(_iter_points(intersection))
    if not points:
        return False
    P = np.asarray(points)
    # distance from each intersection point to each excluded coord
    diffs = P[:, None, :] - excluded[None, :, :]
    dists = np.hypot(diffs[..., 0], diffs[..., 1])
    return bool(np.all(np.any(dists <= endpoint_tol, axis=1)))


def find_geometric_crossings(
    G: nx.Graph,
    *,
    include_touches: bool = False,
    length_tol: float = 1e-12,
    angle_tol: float = 1e-10,
    endpoint_tol: float = 1e-9,
) -> list[dict]:
    """Find route intersections in a routeset using Shapely geometries.

    Geometry-first diagnostic complement to :func:`validate_routeset` and
    :func:`list_edge_crossings`. Unlike :func:`list_edge_crossings`, which only
    detects crossings between extended-Delaunay edges (i.e. it requires a
    routeset built from ``A``, OptiWindNet's available-edges graph), this
    routine works on **any** routeset graph that exposes ``VertexC`` (and
    ``fnT`` if it carries contour or detour clones). It can therefore validate
    routes produced by external tools, hand-built test graphs, or post-edited
    OptiWindNet results — at the cost of a heavier geometry-based check.

    Polylines are extracted from ``G`` (one per feeder, plus one per
    junction-to-junction link) and translated through ``fnT`` so that contour
    and detour clones are tested at their prime coordinates.

    Args:
      G: routeset graph. Must have graph attributes ``'T'``, ``'R'``, ``'B'``, and
        ``'VertexC'``; ``'fnT'`` is required iff ``C > 0`` or ``D > 0``.
      include_touches: also report point contacts that are not proper crossings
        (otherwise touches are silently dropped).
      length_tol: collinear overlaps shorter than this are not classified.
      angle_tol: minimum cross-product magnitude used to deduplicate co-directional
        rays in the local crossing test.
      endpoint_tol: distance below which an intersection point is treated as
        coincident with a path endpoint, shared node, or detour-split prime.

    Returns:
      One dict per finding, with keys:

      - ``'kind'``: one of
          - ``'cross'``: two polylines cross at one or more isolated points;
          - ``'overlap_cross'``: two polylines share a sub-run and exit the
            overlap on opposite sides at both ends (a true cross expressed as
            a coincident segment);
          - ``'branch_split'``: a detour-clone whose prime is a real terminal
            cuts that terminal's subtree into pieces;
          - ``'touch'`` (only when ``include_touches=True``): point contact
            that is not classified as a cross (e.g. tangent kiss).
      - ``path_nodes_a``, ``path_nodes_b``: the raw polyline node sequences.
      - ``path_a``, ``path_b``: canonical prime-path tuples (sorted so that
        ``path_a < path_b`` lexicographically).
      - ``geometry``: WKT string of the offending Shapely geometry (Point,
        MultiPoint, LineString, MultiLineString, …).
    """
    VertexC = G.graph['VertexC']
    fnT = _routeset_fnT(G)
    polylines = _routeset_polylines(G)
    paths = [polyline.nodes for polyline in polylines]
    prime_paths = [_canonical_prime_path(G, path, fnT) for path in paths]
    path_coords = [_polyline_coords(VertexC, fnT, path) for path in paths]
    splits = _detour_splits(G, fnT, VertexC)

    findings = _branch_split_findings(splits, polylines, prime_paths, fnT, VertexC)

    lines = [shp.LineString(coords) for coords in path_coords]
    tree = shp.STRtree(lines)

    for i, line_a in enumerate(lines):
        for j in tree.query(line_a, predicate='intersects').tolist():
            if j <= i:
                continue
            intersection = line_a.intersection(lines[j])
            if intersection.is_empty:
                continue
            excluded = _exclusion_coords(paths[i], paths[j], fnT, VertexC, splits)
            if _intersection_only_at_excluded(
                intersection, excluded, endpoint_tol=endpoint_tol
            ):
                continue

            path_a, path_b = prime_paths[i], prime_paths[j]
            kind: str | None = None
            geometry = intersection

            if intersection.length > length_tol and _shared_run_swaps_sides(
                path_coords[i], path_coords[j]
            ):
                kind = 'overlap_cross'

            if kind is None:
                crossings = _filter_crossing_points(
                    intersection,
                    excluded,
                    path_coords[i],
                    path_coords[j],
                    angle_tol=angle_tol,
                    endpoint_tol=endpoint_tol,
                )
                if crossings:
                    kind = 'cross'
                    geometry = (
                        shp.Point(crossings[0])
                        if len(crossings) == 1
                        else shp.MultiPoint(crossings)
                    )
                elif include_touches:
                    kind = 'touch'
                else:
                    continue

            path_nodes_a, path_nodes_b = paths[i], paths[j]
            if path_b < path_a:
                path_nodes_a, path_nodes_b = path_nodes_b, path_nodes_a
                path_a, path_b = path_b, path_a
            findings.append(
                {
                    'kind': kind,
                    'path_nodes_a': path_nodes_a,
                    'path_nodes_b': path_nodes_b,
                    'path_a': path_a,
                    'path_b': path_b,
                    'geometry': geometry,
                }
            )

    return [{**finding, 'geometry': finding['geometry'].wkt} for finding in findings]


def _filter_crossing_points(
    intersection,
    excluded: np.ndarray,
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    *,
    angle_tol: float,
    endpoint_tol: float,
) -> list[np.ndarray]:
    """Return point-intersections that are genuine X-crossings.

    Drops points near any excluded coord (shared nodes, polyline endpoints,
    detour-split primes) and points where local rays don't alternate.
    """
    P = np.asarray(list(_iter_points(intersection)))
    if len(P) == 0:
        return []
    if len(excluded):
        diffs = P[:, None, :] - excluded[None, :, :]
        near_excluded = np.any(
            np.hypot(diffs[..., 0], diffs[..., 1]) <= endpoint_tol, axis=1
        )
    else:
        near_excluded = np.zeros(len(P), dtype=bool)
    # tolerance scales with the largest segment among either polyline
    scale = max(
        np.linalg.norm(np.diff(coords_a, axis=0), axis=1).max(initial=1.0),
        np.linalg.norm(np.diff(coords_b, axis=0), axis=1).max(initial=1.0),
    )
    tol = endpoint_tol * scale
    return [
        P[k]
        for k in range(len(P))
        if not near_excluded[k]
        and polylines_cross_at_point(
            coords_a,
            coords_b,
            P[k],
            tol=tol,
            angle_tol=angle_tol,
        )
    ]


def list_edge_crossings(
    S: nx.Graph, A: nx.Graph
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """List edge×edge crossings for the network topology in S.

    ``S`` must only use extended Delaunay edges. It will not detect crossings
    of non-extDelaunay gates or detours.

    Args:
      S: solution topology
      A: available edges used in creating ``S``

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
