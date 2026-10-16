# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Derived graph variants of the same kind.

Every ``as_*()`` here returns the kind it was given -- ``L`` to ``L``, ``S`` to
``S``, ``G`` to ``G`` -- with different properties: normalized, single-root,
obstacle-free, undetoured, rehooked. For a conversion between kinds, see
:mod:`optiwindnet.converting`.
"""

import math
from itertools import pairwise
from typing import cast

import networkx as nx
import numpy as np

from .converting import _chains
from .geometric import CoordPair
from .loads import _branch_loads, calcload

__all__ = (
    'as_hooked_to_head', 'as_hooked_to_nearest', 'as_normalized',
    'as_obstacle_free', 'as_rescaled', 'as_single_root',
    'as_stratified_vertices', 'as_undetoured',
)  # fmt: skip


def as_obstacle_free(Lʹ: nx.Graph) -> nx.Graph:
    """Make a shallow copy of an instance and remove its obstacles.

    The vertices that are used only by obstacles are also removed.
    To be used on locations (edge-less graphs).

    Args:
      Lʹ: input location

    Returns:
      location without obstacles.
    """
    L = Lʹ.copy()
    obstacles = Lʹ.graph.get('obstacles')
    if obstacles is None:
        # Lʹ has no obstacles to remove
        return L
    del L.graph['obstacles']
    T = L.graph['T']
    R = L.graph['R']
    borderʹ = Lʹ.graph.get('border')
    borderset = set(borderʹ[borderʹ >= T].tolist()) if borderʹ is not None else set()
    removable = set()
    for obstacle in obstacles:
        removable.update(set(obstacle[obstacle >= T].tolist()) - borderset)
    to_remove = sorted(removable)
    VertexCʹ = Lʹ.graph['VertexC']
    Bʹ = Lʹ.graph['B']
    VertexC = np.vstack(
        (
            VertexCʹ[:T],
            VertexCʹ[[i for i in range(T, T + Bʹ) if i not in to_remove]],
            VertexCʹ[-R:],
        )
    )
    B = Bʹ - len(to_remove)
    L.graph.update(
        B=B,
        VertexC=VertexC,
        name=Lʹ.graph.get('name', '') + '.solid',
        handle=Lʹ.graph.get('handle', '') + '_solid',
    )
    if borderʹ is not None:
        border = borderʹ.copy()
        for i, v in enumerate(to_remove):
            border[border >= (v - i)] -= 1
        L.graph['border'] = border
    return L


def as_single_root(Lʹ: nx.Graph) -> nx.Graph:
    """Make a shallow copy of an instance and reduce its roots to one.

    The output's root is the centroid of the input's roots.
    This may not work well for locations with obstacles, use
    ``as_obstacle_free()`` first.

    Args:
      Lʹ: input location

    Returns:
      location with a single root.
    """
    R, T, VertexCʹ = (Lʹ.graph[k] for k in ('R', 'T', 'VertexC'))
    L = Lʹ.copy()
    if R <= 1:
        return L
    to_transfer = {}
    Bʹ = Lʹ.graph['B']
    if 'border' in L.graph:
        borderʹ = L.graph['border']
        root_in_border = borderʹ < 0
        if root_in_border.any():
            border = borderʹ.copy()
            next_v = T + Bʹ
            for i in np.flatnonzero(root_in_border):
                v = borderʹ[i]
                if v in to_transfer:
                    border[i] = to_transfer[v]
                else:
                    to_transfer[v] = next_v
                    border[i] = next_v
                    next_v += 1
            B = Bʹ + len(to_transfer)
            L.graph['border'] = border
        else:
            B = Bʹ
    else:
        borderʹ = []
        root_in_border = slice(0, 0)
        B = Bʹ
    VertexC = np.vstack(
        (VertexCʹ[:-R], VertexCʹ[list(to_transfer.keys())], VertexCʹ[-R:].mean(axis=0))
    )
    L.remove_nodes_from(range(-R, -1))
    L.graph.update(VertexC=VertexC, R=1, B=B)
    L.graph['name'] += '.1_OSS'
    L.graph['handle'] += '_1'
    return L


def as_normalized(
    Aʹ: nx.Graph, *, offset: CoordPair | None = None, scale: float | None = None
) -> nx.Graph:
    """Make a copy of an instance and shift and scale its geometry.

    Coordinates are subtracted by graph attribute ``'norm_offset'``.
    All lengths and coordinates are multiplied by graph attribute ``'norm_scale'``.
    Graph attribute ``'is_normalized'`` is set to ``True``.
    Affected linear attributes: ``'VertexC'``, ``'d2roots'`` (graph);
    ``'length'`` (edge).

    Args:
        Aʹ: (or Gʹ) any instance that has inherited ``'scale'`` from an
            edgeset ``Aʹ``.
        offset: coordinates (2,) offset to override graph's ``'norm_offset'``
        scale: multiplicative scaling factor to override graph's ``'norm_scale'``

    Returns:
        A copy of the instance with changed coordinates and linear metrics.
    """
    A = Aʹ.copy()
    if offset is None:
        offset = Aʹ.graph['norm_offset']
    else:
        A.graph['norm_offset'] = offset
    if scale is None:
        scale = Aʹ.graph['norm_scale']
    else:
        A.graph['norm_scale'] = scale
    A.graph['is_normalized'] = True
    for _, _, eData in A.edges(data=True):
        eData['length'] *= scale
    A.graph['VertexC'] = scale * (Aʹ.graph['VertexC'] - offset)
    d2roots = Aʹ.graph.get('d2roots')
    if d2roots is not None:
        A.graph['d2roots'] = scale * d2roots
    return A


def as_rescaled(Gʹ: nx.Graph, L: nx.Graph) -> nx.Graph:
    """Revert normalization done by :func:`as_normalized`.

    Args:
      Gʹ: routeset to rescale to pre-normalization size.
      L: (or G or A) locations or routeset to get ``'VertexC'`` from (also
        ``'d2roots'``, if available).

    Returns:
      Routeset with coordinates and lengths at site scale.
    """
    if not Gʹ.graph.get('is_normalized', False):
        # Gʹ is not marked as normalized
        return Gʹ
    G = Gʹ.copy()
    # alternatively, we could do the math, but this safeguards the coord's hash
    G.graph['VertexC'] = L.graph['VertexC']
    denorm_factor = 1 / G.graph['norm_scale']
    for _, _, eData in G.edges(data=True):
        eData['length'] *= denorm_factor
    d2roots = L.graph.get('d2roots')
    if d2roots is not None:
        G.graph['d2roots'] = d2roots
    elif 'd2roots' in G.graph:
        del G.graph['d2roots']
    del G.graph['is_normalized']
    # this factor can be used later to scale metadata (such as 'objective')
    G.graph['denormalization'] = denorm_factor
    return G


def as_undetoured(Gʹ: nx.Graph) -> nx.Graph:
    """Create an undetoured version of Gʹ.

    Creates a shallow copy of ``Gʹ`` without detour nodes (and possibly *with*
    the resulting crossings). Changed links' ``'kind'`` become ``'tentative'``.

    This is to be applied to a routeset that already has detours. It serves to
    re-run PathFinder on a detoured routeset, but it is not the best solution
    to prepare a routeset to be used as warmstart (re-hooking is missing).
    """
    G = Gʹ.copy()
    T, B, R = (G.graph[k] for k in 'TBR')
    C = G.graph.get('C', 0)
    if not G.graph.get('D', 0):
        return G
    tentative = set(G.graph.get('tentative', ()))
    for source, sink, path in list(_chains(G)):
        if not any(n >= T + B + C for n in path):
            continue
        attrs = G[source][path[1]].copy()
        G.remove_nodes_from(path[1:-1])
        G.add_edge(
            source,
            sink,
            **{
                **attrs,
                'kind': 'tentative',
                'length': math.dist(
                    G.graph['VertexC'][source], G.graph['VertexC'][sink]
                ),
            },
        )
        tentative.add((min(source, sink), max(source, sink)))
    G.graph.pop('D', None)
    if C:
        fnT = G.graph['fnT']
        G.graph['fnT'] = np.hstack((fnT[: T + B + C], fnT[-R:]))
    else:
        G.graph.pop('fnT', None)
    G.graph['tentative'] = sorted(tentative)
    return G


def _as_rehooked(G: nx.DiGraph, d2roots: np.ndarray, *, heads_only: bool) -> nx.DiGraph:
    """Rehook tentative feeders, defaulting to all when the list is absent."""
    G = G.copy()
    if not G.graph.get('has_loads'):
        calcload(G)
    T, R = G.graph['T'], G.graph['R']
    feeders = G.graph.get('tentative')
    if feeders is None:
        feeders = [(r, n) for r in range(-R, 0) for n in G.predecessors(r)]
    tentative = []
    for root, old in feeders:
        subtree = cast(set[int], nx.ancestors(G, old)) | {old}
        candidates = [
            n
            for n in subtree
            if 0 <= n < T and (not heads_only or G.in_degree[n] == 0 or n == old)
        ]
        hook = min(candidates, key=lambda n: d2roots[n, root])
        if hook != old:
            for source, sink in pairwise(nx.shortest_path(G, hook, old)):
                attrs = G[source][sink].copy()
                G.remove_edge(source, sink)
                G.add_edge(sink, source, **attrs)
            attrs = G[old][root].copy()
            G.remove_edge(old, root)
            G.add_edge(hook, root, **{**attrs, 'length': d2roots[hook, root]})
            # only the rehooked branch changes: move its load back onto the root
            expected = G.nodes[root]['load']
            G.nodes[root]['load'] -= attrs['load']
            _branch_loads(G, hook, root, G.nodes[old]['subtree'])
            total = G.nodes[root]['load']
            assert total == expected, (
                f'rehook {old}–{hook}: load calculated ({total}) != '
                f'expected load ({expected})'
            )
        G[hook][root]['kind'] = 'tentative'
        tentative.append((root, hook))
    G.graph['tentative'] = tentative
    return G


def as_hooked_to_nearest(Gʹ: nx.DiGraph, d2roots: np.ndarray) -> nx.DiGraph:
    """Make tentative feeders link to the nearest-to-root node of each subtree.

    Output may be branched (use with care with path routesets). Meant for
    radial and branched solutions, not ringed ones.

    Moves each feeder listed in ``Gʹ.graph['tentative']`` -- every feeder when
    the list is absent -- to the terminal of its subtree that is closest to the
    feeder's root according to ``d2roots``. The arcs between the old and the
    new hook are reversed, so that current keeps flowing toward the root, and
    the loads of the rehooked branch are recalculated (all loads are calculated
    first if ``Gʹ`` carries none). The resulting feeders are marked
    ``'tentative'`` and replace the graph's ``'tentative'`` list.

    Should be called after :func:`as_undetoured` if the goal is to use G as a
    warm start for MILP models.

    Args:
      Gʹ: routeset or topology S; it is not modified.
      d2roots: distance from nodes to roots (e.g. ``A.graph['d2roots']``).

    Returns:
      Rehooked copy of ``Gʹ``.
    """
    return _as_rehooked(Gʹ, d2roots, heads_only=False)


def as_hooked_to_head(Sʹ: nx.DiGraph, d2roots: np.ndarray) -> nx.DiGraph:
    """Make tentative feeders link to the nearest-to-root end of each string.

    Only works with solutions where subtrees are paths (radial topology).

    Moves each feeder listed in ``Sʹ.graph['tentative']`` -- every feeder when
    the list is absent -- to whichever end of its string is closest to the
    feeder's root according to ``d2roots``. The arcs along the string are
    reversed when its other end becomes the hook, so that current keeps flowing
    toward the root, and the string's loads are recalculated (all loads are
    calculated first if ``Sʹ`` carries none). The resulting feeders are marked
    ``'tentative'`` and replace the graph's ``'tentative'`` list.

    Should be called after :func:`as_undetoured` if the goal is to use S as a
    warm start for MILP models.

    Args:
      Sʹ: solution topology; it is not modified.
      d2roots: distance from nodes to roots (e.g. ``A.graph['d2roots']``).

    Returns:
      Rehooked copy of ``Sʹ``.
    """
    return _as_rehooked(Sʹ, d2roots, heads_only=True)


def as_stratified_vertices(Lʹ: nx.Graph) -> nx.Graph:
    """Ensure border-vertices are all in the B-range of VertexC.

    Apply this to L when terminal or root coordinates are to be updated by writting to
    the array elements of VertexC. In order to keep the borders in place, they must not
    rely on vertices in the terminal or root sections (T-range, R-range). This function
    creates duplicates of any terminal-vertex or root-vertex used by borders/obstacles.

    Args:
      L: location geometry to be stratified
    Returns:
      New location geometry with stratified vertices
    """
    L = Lʹ.copy()
    R, T = (L.graph[k] for k in 'RT')
    border = L.graph.get('border', np.array(()))
    obstacles = L.graph.get('obstacles', [])
    if any(border < T) or any(any(obstacle < T) for obstacle in obstacles):
        # is not stratified
        VertexC = L.graph['VertexC']
        VertexC = np.vstack(
            (
                VertexC[:T],
                VertexC[border],
                *(VertexC[obstacle] for obstacle in obstacles),
                VertexC[-R:],
            )
        )
        border_sizes = np.array(
            [border.shape[0]] + [obstacle.shape[0] for obstacle in obstacles]
        )
        obstacle_idxs = np.cumsum(border_sizes) + T
        L.graph.update(
            VertexC=VertexC,
            B=border_sizes.sum().item(),
            border=np.arange(T, T + border.shape[0]),
            obstacles=[np.arange(a, b) for a, b in pairwise(obstacle_idxs)],
        )
    return L
