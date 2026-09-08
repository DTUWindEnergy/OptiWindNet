# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Derived graph variants of the same kind.

Every ``as_*()`` here returns the kind it was given -- ``L`` to ``L``, ``S`` to
``S``, ``G`` to ``G`` -- with different properties: normalized, single-root,
obstacle-free, undetoured, rehooked. For a conversion between kinds, see
:mod:`optiwindnet.converting`.
"""

from itertools import pairwise

import networkx as nx
import numpy as np

from .geometric import CoordPair
from .loads import bfs_subtree_loads

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
    """Make a shallow copy of an instance and shift and scale its geometry.

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
    C, D = (G.graph.get(k, 0) for k in 'CD')
    if not D:
        return G
    R, T, B = (G.graph[k] for k in 'RTB')
    VertexC = G.graph['VertexC']
    tentative = []
    for r in range(-R, 0):
        for n in [n for n in G.neighbors(r) if n >= T + B + C]:
            rev = r
            G.remove_edge(n, r)
            while n >= T:
                rev = n
                (n,) = G.neighbors(rev)
                G.remove_node(rev)
            G.add_edge(
                r,
                n,
                load=G.nodes[n]['load'],
                kind='tentative',
                reverse=False,
                length=np.hypot(*(VertexC[n] - VertexC[r])).item(),
            )
            tentative.append((r, n))
    del G.graph['D']
    if C:
        fnT = G.graph['fnT']
        G.graph['fnT'] = np.hstack((fnT[: T + B + C], fnT[-R:]))
    else:
        del G.graph['fnT']
    G.graph['tentative'] = tentative
    return G


def as_hooked_to_nearest(Gʹ: nx.Graph, d2roots: np.ndarray) -> nx.Graph:
    """Make tentative feeders link to the nearest-to-root node of each subtree.

    Output may be branched (use with care with path routesets).

    Sifts through all ``'tentative'`` gates' subtrees and choose the hook closest
    to the respective root according to ``d2roots``.

    Should be called after :func:`as_undetoured` if the goal is to use G as a
    warmstart for MILP models.

    Args:
      G: routeset or topology S
      d2roots: distance from nodes to roots (e.g. ``A.graph['d2roots']``)
    """
    assert Gʹ.graph.get('has_loads')
    G = Gʹ.copy()
    R, T = G.graph['R'], G.graph['T']
    # mappings to quickly obtain all nodes on a subtree
    num_subtree = sum(G.degree[r] for r in range(-R, 0))
    nodes_from_subtree_id = np.fromiter(
        ([] for _ in range(num_subtree)), count=num_subtree, dtype=object
    )
    subtree_from_node = np.empty((T,), dtype=object)
    # Only terminals are hook candidates, but clones (detour and contour nodes,
    # numbered from T up) carry a 'load' too, so clearing the subtree's loads
    # must reach them: a clone that kept a stale load would be read back by
    # bfs_subtree_loads() and counted twice.
    loaded_from_subtree_id = np.fromiter(
        ([] for _ in range(num_subtree)), count=num_subtree, dtype=object
    )
    for n, subtree_id in G.nodes(data='subtree'):
        if subtree_id is None:
            continue
        loaded_from_subtree_id[subtree_id].append(n)
        if 0 <= n < T:
            subtree = nodes_from_subtree_id[subtree_id]
            subtree.append(n)
            subtree_from_node[n] = subtree

    # do the actual rehooking
    # TODO: rehook should take into account the other roots
    #       see PathFinder.create_detours()
    tentative = []
    hook_getter = ((r, nb) for r in range(-R, 0) for nb in tuple(G.neighbors(r)))
    for r, hook in G.graph.pop('tentative', hook_getter):
        subtree = subtree_from_node[hook]
        new_hook = subtree[np.argmin(d2roots[subtree, r])]
        if new_hook != hook:
            subtree_load = G.nodes[hook]['load']
            G.remove_edge(r, hook)
            G.add_edge(
                r,
                new_hook,
                length=d2roots[new_hook, r],
                kind='tentative',
                load=subtree_load,
            )
            for node in loaded_from_subtree_id[G.nodes[hook]['subtree']]:
                G.nodes[node].pop('load', None)

            ref_load = G.nodes[r]['load']
            G.nodes[r]['load'] = ref_load - subtree_load
            total_parent_load = bfs_subtree_loads(
                G, r, [new_hook], G.nodes[new_hook]['subtree']
            )
            assert total_parent_load == ref_load, (
                f'parent ({total_parent_load}) != expected load ({ref_load})'
            )
        else:
            # only necessary if using hook_getter (e.g. Gʹ is a S)
            G[r][new_hook]['kind'] = 'tentative'
        tentative.append((r, new_hook))
    G.graph['tentative'] = tentative
    return G


def as_hooked_to_head(Sʹ: nx.Graph, d2roots: np.ndarray) -> nx.Graph:
    """Make tentative feeders link to the nearest-to-root end of each string.

    Only works with solutions where subtrees are paths (radial topology).

    Sifts through the subtrees of ``'tentative'`` feeders and re-hook the subtree via
    the end-node that is nearest to the respective root according to ``d2roots``.

    Should be called after :func:`as_undetoured` if the goal is to use S as a
    warmstart for MILP models.

    Args:
      S: solution topology
      d2roots: distance from nodes to roots (e.g. ``A.graph['d2roots']``)
    """
    assert Sʹ.graph.get('has_loads')
    S = Sʹ.copy()
    R, T = S.graph['R'], S.graph['T']
    # mappings to quickly obtain all nodes on a subtree
    S_T = nx.subgraph_view(Sʹ, filter_node=lambda n: n >= 0)
    num_subtree = sum(S.degree[r] for r in range(-R, 0))
    nodes_from_subtree_id = np.fromiter(
        ([] for _ in range(num_subtree)), count=num_subtree, dtype=object
    )
    subtree_from_node = np.empty((T,), dtype=object)
    headtail_from_subtree_id = np.fromiter(
        ([] for _ in range(num_subtree)), count=num_subtree, dtype=object
    )
    headtail_from_node = np.empty((T,), dtype=object)
    for n, subtree_id in S.nodes(data='subtree'):
        if 0 <= n < T:
            subtree = nodes_from_subtree_id[subtree_id]
            subtree.append(n)
            subtree_from_node[n] = subtree
            headtail = headtail_from_subtree_id[subtree_id]
            headtail_from_node[n] = headtail
            if S_T.degree[n] <= 1:
                headtail.append(n)

    # do the actual rehooking
    # TODO: rehook should take into account the other roots
    #       see PathFinder.create_detours()
    tentative = []
    hook_getter = ((r, nb) for r in range(-R, 0) for nb in tuple(S.neighbors(r)))
    for r, hook in S.graph.pop('tentative', hook_getter):
        headtail = headtail_from_node[hook]
        new_hook = headtail[np.argmin(d2roots[headtail, r])]
        if new_hook != hook:
            subtree_load = S.nodes[hook]['load']
            S.remove_edge(r, hook)
            S.add_edge(r, new_hook, kind='tentative', load=subtree_load)
            for node in subtree_from_node[hook]:
                del S.nodes[node]['load']

            ref_load = S.nodes[r]['load']
            S.nodes[r]['load'] = ref_load - subtree_load
            total_parent_load = bfs_subtree_loads(
                S, r, [new_hook], S.nodes[new_hook]['subtree']
            )
            assert total_parent_load == ref_load, (
                f'parent ({total_parent_load}) != expected load ({ref_load})'
            )
        else:
            # only necessary if using hook_getter (e.g. Gʹ is a S)
            S[r][new_hook]['kind'] = 'tentative'
        tentative.append((r, new_hook))
    S.graph['tentative'] = tentative
    return S


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
