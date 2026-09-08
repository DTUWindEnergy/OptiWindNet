# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Cable-load computation over solution topologies and routesets."""

import math

import networkx as nx

from .types import Topology

__all__ = (
    'add_ring_to_S', 'bfs_subtree_loads', 'calcload',
    'split_rings_and_calc_loads',
)  # fmt: skip


# A link's ``'reverse'`` flag orients it independently of the node order it
# happens to be stored in. Current flows from the terminal that sources it to the
# root that sinks it, and readers recover that direction with::
#
#     u, v = (u, v) if ((u < v) == edgeD['reverse']) else (v, u)
#
# which yields ``(source, sink)`` for either stored order. So every writer sets
# ``reverse = source < sink``. Beware of writing ``load[u] < load[v]`` instead: it
# only matches while ``u < v`` holds, and silently mis-orients links stored the
# other way round. Feeders are never reversed, the sink being a root and a root's
# id negative; links carrying ``load=0`` (a ring's zero-load link) have no current and
# so no direction to encode.
def _bfs_loads_walk(_adj, _node, T, visited, queue) -> None:
    """Descend the subtrees seeded in ``queue``, appending every node reached.

    Each ``queue`` entry is ``(node, parent, edgeD, parentD, subtree)``, with
    ``edgeD`` the ⟨parent, node⟩ link data and ``parentD`` the parent's node
    data. Queue (BFS) order places every node before its descendants, which is
    what :func:`_bfs_loads_unwind` relies on. Every node gets its ``'subtree'``
    and the base its descendants' loads are added to; the accumulation happens
    on the way back up.

    A node that keeps a stale ``'load'`` (callers that clear only part of the
    graph, e.g. :func:`as_hooked_to_nearest`) starts from it, unless it is a
    leaf of the traversal, which always restarts from its own contribution.
    """
    i = 0
    while i < len(queue):
        node, parent, _, _, subtree = queue[i]
        i += 1
        nodeD = _node[node]
        nodeD['subtree'] = subtree
        stop = len(queue)
        for nbr, edgeD in _adj[node].items():
            # a load=0 link is a ring zero-load link: never traverse across it
            if nbr == parent or edgeD.get('load') == 0:
                continue
            if nbr in visited:
                raise ValueError(f'node {nbr} reached twice: not a tree below {node}')
            visited.add(nbr)
            queue.append((nbr, node, edgeD, nodeD, subtree))
        default = 1 if node < T else 0  # load is 1 for wtg nodes
        nodeD['load'] = default if len(queue) == stop else nodeD.get('load', default)


def _bfs_loads_unwind(_node, queue) -> None:
    """Accumulate the loads of the traversal recorded in ``queue``.

    Reversed BFS order visits every node after all of its descendants, so each
    node's load is complete before it is added to its parent's.
    """
    for node, parent, edgeD, parentD, _ in reversed(queue):
        load = _node[node]['load']
        # the child sources the current, the parent sinks it (towards the root)
        edgeD['load'] = load
        edgeD['reverse'] = node < parent
        parentD['load'] += load


def bfs_subtree_loads(G, parent, children, subtree, visited=None):
    """Descend the subtree, updating edge and node attributes.

    Meant to be called by :func:`calcload`, but can be used independently (e.g.
    from PathFinder). Nodes must not have a ``'load'`` attribute.

    Args:
      G: graph to traverse.
      parent: node the traversal descends from.
      children: nodes of ``G`` to descend into.
      subtree: subtree id to assign to every node visited.
      visited: nodes already claimed by this traversal; pass one set across
        several calls to keep them from claiming a node twice. A fresh set is
        used when omitted.

    Returns:
      Total number of descendant nodes

    Raises:
      ValueError: a node is reached twice, so the traversal is not descending a
        tree -- ``G`` holds a cycle, or two roots reach the same node.
    """
    T = G.graph['T']
    if visited is None:
        visited = {parent}
    _adj, _node = G._adj, G._node
    nodeD = _node[parent]
    default = 1 if parent < T else 0  # load is 1 for wtg nodes
    if not children:
        nodeD['load'] = default
        return default
    nodeD['load'] = nodeD.get('load', default)
    adjP = _adj[parent]
    queue = []
    for child in children:
        if child in visited:
            raise ValueError(f'node {child} reached twice: not a tree below {parent}')
        visited.add(child)
        queue.append((child, parent, adjP[child], nodeD, subtree))
    _bfs_loads_walk(_adj, _node, T, visited, queue)
    _bfs_loads_unwind(_node, queue)
    return nodeD['load']


def split_rings_and_calc_loads(S: nx.Graph, A: nx.Graph) -> None:
    """Close path-form ring arms into canonical rings and compute their loads.

    Only the ringed builders (HGS, LKH and the ``method='ringed'`` constructor)
    call this, on a solution ``S`` that is still a set of simple
    ``root → … → root`` paths missing their zero-load links. Each path is walked
    and closed into a canonical ring (see :func:`add_ring_to_S`), using ``A`` to
    pick the longer zero-load link on odd-length rings; a tail already touching
    a root bridges two roots ``(r1, r2)``. Every ring receives exactly one
    zero-load link (``load=0``, no current flows through it), and each node's
    subtree id and load, the edges' loads, and the graph's ``max_load`` /
    ``has_loads`` / root loads are set.

    All ringed solvers must call this before returning a solution, so that every
    ringed ``S`` carries exactly one ``load=0`` link per ring.
    """
    # Ring construction: S is path-form (no zero-load links yet). Walk each root's
    # single-feeder path to its tail, then close it into a canonical ring; a tail
    # already touching a root bridges two roots (r1, r2).
    R = S.graph['R']
    paths: list[tuple[tuple[int, int], list[int]]] = []
    seen: set[int] = set()  # first terminal of each ring already walked
    for root in range(-R, 0):
        for gate in S[root]:
            if gate in seen:
                # bridging ring: already walked from its other subroot's root
                continue
            ordered = [gate]
            back, fwd = root, gate
            while True:
                nbrs = [n for n in S[fwd] if n != back]
                if not nbrs:
                    break
                (nxt,) = nbrs  # ValueError here means S has a branching subtree
                if nxt < 0:
                    break
                ordered.append(nxt)
                back, fwd = fwd, nxt
            seen.update(ordered)
            tn = ordered[-1]
            end_roots = [
                r
                for r in range(-R, 0)
                if r in S[tn] and (len(ordered) > 1 or r != root)
            ]
            end_root = end_roots[0] if end_roots else root
            paths.append(((root, end_root), ordered))
    S.remove_edges_from(list(S.edges))
    max_load = 0
    for subtree_id, (roots, ordered) in enumerate(paths):
        add_ring_to_S(S, roots, ordered, subtree_id, A)
        max_load = max(max_load, math.ceil(len(ordered) / 2))
    for root in range(-R, 0):
        # a load=0 feeder carries no current, so it adds nothing to its root
        # (the zero-load link of a bridging stub is a feeder, not an interior link)
        S.nodes[root]['load'] = sum(
            S.nodes[n]['load'] for n in S[root] if S[root][n]['load'] != 0
        )
    S.graph['max_load'] = max_load
    S.graph['has_loads'] = True


def calcload(G: nx.Graph) -> None:
    """Calculate link loads and update edge and node attributes of ``G``.

    ``G`` must already be in final form (a forest, or a ring-form graph whose
    ``load=0`` zero-load links are present). A breadth-first traversal of each root's
    subtree propagates the loads, treating ``load=0`` links (ring zero-load links) as
    breaks. Each node's subtree id and outgoing load land on its ``'subtree'`` /
    ``'load'`` attributes, the edges' ``'load'`` attributes are updated, and the
    graph's ``'max_load'``, ``'has_loads'`` and root loads are set.

    Ring construction — closing path-form arms into rings — lives in
    :func:`split_rings_and_calc_loads`, which the ringed builders call instead.
    """
    R, T = (G.graph[k] for k in 'RT')
    # the raw dicts: indexing G[u][v] and G.nodes[n] instead would rebuild a
    # view object on every access, which dominates the cost of this traversal
    # pyrefly: ignore[missing-attribute]
    _adj, _node = G._adj, G._node
    for data in _node.values():
        data.pop('load', None)

    # one set across every root: a node claimed by two roots is reported too
    visited = set(range(-R, 0))
    queue = []
    subroots = []
    subtree = 0
    for root in range(-R, 0):
        rootD = _node[root]
        rootD['load'] = 0
        for subroot, edgeD in _adj[root].items():
            # A load=0 feeder (degenerate multi-root ring zero-load link) carries
            # no load.
            if edgeD.get('load') == 0:
                continue
            if subroot in visited:
                raise ValueError(
                    f'node {subroot} reached twice: not a tree below {root}'
                )
            visited.add(subroot)
            queue.append((subroot, root, edgeD, rootD, subtree))
            subroots.append(subroot)
            subtree += 1
    _bfs_loads_walk(_adj, _node, T, visited, queue)
    _bfs_loads_unwind(_node, queue)

    max_load = max((_node[subroot]['load'] for subroot in subroots), default=0)
    total_load = sum(_node[root]['load'] for root in range(-R, 0))
    if len(_node) > T + R:
        # Clones inside a routed ring's open cable are separated from both arms by
        # load=0 segments. They intentionally carry no current and are therefore
        # not reached by the root traversals above.
        for node, nodeD in _node.items():
            if (
                node >= T
                and 'load' not in nodeD
                and all(edgeD.get('load') == 0 for edgeD in _adj[node].values())
            ):
                nodeD['load'] = 0
    if total_load != T:
        raise ValueError(f'root loads sum to {total_load}, expected T = {T}')
    G.graph['has_loads'] = True
    G.graph['max_load'] = max_load


def _ring_split_position(ordered: list[int], A: nx.Graph | None = None) -> int:
    """Choose the balanced position between a ring's two arms."""
    n = len(ordered)
    m, mod = divmod(n, 2)
    m += mod
    if mod and n > 1 and A is not None:
        rev, center, fwd = ordered[m - 2], ordered[m - 1], ordered[m]
        rev_len = A[rev][center]['length'] if A.has_edge(rev, center) else 0
        fwd_len = A[center][fwd]['length'] if A.has_edge(center, fwd) else 0
        if rev_len > fwd_len:
            m -= 1
    return m


def add_ring_to_S(
    S: nx.Graph,
    roots: tuple[int, int],
    ordered: list[int],
    subtree: int,
    A: nx.Graph | None = None,
) -> None:
    """Add a single ring to topology graph ``S`` in canonical form.

    A ring is the union of two radial arms, fed by ``r1`` and ``r2`` and joined
    at their tail ends; it bridges two substations when ``r1 != r2``. ``ordered``
    is the terminal sequence ``[t1, ..., tn]`` walked along the ring, so that
    ``t1`` and ``tn`` are the feeder-connected terminals. Both feeders
    ``(r1, t1)`` and ``(r2, tn)`` are real, load-bearing cables; the ring's single
    zero-load link is the edge at the load midpoint, marked by ``load=0`` (a real
    cable, no current flows through it).

    Arm 1 (the ``t1`` side) gets ``m = ceil(n / 2)`` terminals, so each arm holds at
    most ``ceil(n / 2)`` — i.e. half of the doubled ring capacity. When the ring has
    an even number of nodes (odd ``n``), the middle terminal has two candidate split
    edges yielding balanced arms; if ``A`` is provided, the longer of the two is
    chosen as the zero-load link.

    Node ``'load'``/``'subtree'`` and edge ``'load'``/``'reverse'`` are all set
    here; the caller is responsible for the root node's aggregate load.

    Args:
      S: topology graph to add the ring to (modified in place).
      roots: the pair ``(r1, r2)`` of (negative) root node ids, equal when both
        feeders share one root.
      ordered: terminal sequence ``[t1, ..., tn]`` along the ring.
      subtree: subtree id to assign to every node of the ring (both arms).
      A: optional available-links graph, used to pick the longer split edge on
        odd-node rings.
    """
    # the builder declares the shape it establishes
    S.graph['topology'] = Topology.RINGED
    r1, r2 = roots
    n = len(ordered)
    if n == 1:
        # Degenerate ring: a single terminal has feeder(s) on it.
        S.add_node(ordered[0], load=1, subtree=subtree)
        S.add_edge(r1, ordered[0], load=1, reverse=False)
        if r1 != r2:
            S.add_edge(r2, ordered[0], load=0, reverse=False)
        return
    m = _ring_split_position(ordered, A)
    # Node loads: arm 1 nodes ordered[0..m-1] carry m..1; arm 2 nodes
    # ordered[m..n-1] carry 1..(n - m) toward their own feeder.
    for i, t in enumerate(ordered):
        S.add_node(t, load=(m - i if i < m else i - m + 1), subtree=subtree)
    # Two feeders (both real cables) and the interior edges. A feeder sinks into
    # its root, whose id is negative, so it is never reversed.
    S.add_edge(r1, ordered[0], load=m, reverse=False)
    S.add_edge(r2, ordered[-1], load=n - m, reverse=False)
    for i in range(n - 1):
        u, v = ordered[i], ordered[i + 1]
        if i == m - 1:
            # zero-load link of the ring: real cable, no current (marked by load=0),
            # so it has no flow direction to encode
            S.add_edge(u, v, load=0, reverse=False)
        else:
            load = m - 1 - i if i < m else i - m + 1
            # current flows towards the arm's feeder, i.e. towards the heavier end
            u_lighter = S.nodes[u]['load'] < S.nodes[v]['load']
            source, sink = (u, v) if u_lighter else (v, u)
            S.add_edge(u, v, load=load, reverse=source < sink)
