# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import math
import pickle
import sys
from collections import defaultdict
from collections.abc import Iterator, Sequence
from hashlib import sha256
from itertools import chain, pairwise

import networkx as nx
import numba as nb
import numpy as np
from bitarray import bitarray

from .geometric import CoordPair, angle_helpers, rotate

_lggr = logging.getLogger(__name__)
debug, warn, error = _lggr.debug, _lggr.warning, _lggr.error

__all__ = (
    'assign_cables',
    'describe_G',
    'pathdist',
    'count_diagonals',
    'bfs_subtree_loads',
    'calcload',
    'split_rings_and_calc_loads',
    'add_ring_to_S',
    'rings_from_links',
    'directed_links',
    'site_fingerprint',
    'fun_fingerprint',
    'L_from_site',
    'G_from_S',
    'S_from_G',
    'L_from_G',
    'S_from_terse_links',
    'terse_links_from_S',
    'as_obstacle_free',
    'as_single_root',
    'as_normalized',
    'as_rescaled',
    'as_undetoured',
    'as_hooked_to_nearest',
    'as_hooked_to_head',
    'as_stratified_vertices',
    'add_terminal_closest_root',
    'add_link_blockmap',
    'add_link_cosines',
    'make_remap',
    'scaffolded',
)

_essential_graph_attrs = (
    'R',
    'T',
    'B',
    'VertexC',
    'name',
    'handle',
    'border',  # required
    'obstacles',
    'landscape_angle',  # optional
    'norm_scale',
    'norm_offset',  # optional
)


def assign_cables(
    G: nx.Graph, cables: list[tuple[int, float | int]], currency: str = '€'
):
    """Assign a cable type to each edge of ``G`` and update attribute ``'cost'``.

    Each edge is assigned the cheapest cable type that can carry its load. The
    edge attribute ``'cable'`` is the index in ``cables`` of the type chosen.

    Changes ``G`` in place.

    Args:
      G: networkx graph with edges having a ``'load'`` attribute (use ``calcload(G)``)
      cables: [(«capacity», «cost»), ...] in increasing capacity order (each
        cable entry must be a tuple)
      currency: symbol representing the unit of the cost
    """
    capacity = max(cables)[0]
    if G.graph['max_load'] > capacity:
        raise ValueError('Maximum cable capacity is smaller than maximum load in G.')
    run_len_ = (b[0] - a[0] for a, b in pairwise(chain(((0,),), cables)))
    kind = [k for k, run_len in enumerate(run_len_) for _ in range(run_len)]
    cost = [cables[k][1] for k in kind]
    has_cost = sum(cost) > 0
    for _, _, data in G.edges(data=True):
        if data['load'] == 0:
            # ring open point ('split'): a real cable with no current — assign the
            # thinnest cable type, but no current-carrying capacity is consumed.
            data['cable'] = 0
            if has_cost:
                data['cost'] = data['length'] * cost[0]
            continue
        k = data['load'] - 1
        data['cable'] = kind[k]
        if has_cost:
            data['cost'] = data['length'] * cost[k]
    G.graph['cables'] = cables
    if has_cost:
        G.graph['currency'] = currency
    if 'capacity' not in G.graph:
        G.graph['capacity'] = capacity


def describe_G(G: nx.Graph, significant_digits: int = 5) -> list[str]:
    """Create a 3-4 line summary of G's properties.

    ``significant_digits`` applies only to total length and is enforced only when the
    integer part has fewer significant digits than ``significant_digits``.

    Args:
      G: route set instance
      significant_digits: minimum number of significant digits used for total length
    Returns:
      Text lines: capacity and T, excess feeders and feeders per root, total length,
        total cost.
    """
    R = G.graph['R']
    T = G.graph['T']
    capacity = G.graph['capacity']
    roots = range(1, R + 1)
    RootL = {-r: G.nodes[-r].get('label', f'[{-r}]') for r in roots}
    desc = []
    desc.append(f'κ = {capacity}, T = {T}')
    feeder_info = [f'{rootL}: {G.degree[r]}' for r, rootL in RootL.items()]
    excess_feeders = sum(G.degree[-r] for r in roots) - math.ceil(T / capacity)
    desc.append(f'({excess_feeders:+d}) {", ".join(feeder_info)}')
    length = G.size(weight='length')
    if length > 0:
        intdigits = int(np.floor(np.log10(length))) + 1
        fracdigits = max(0, significant_digits - intdigits)
        desc.append(
            f'Σλ = {{:_.{fracdigits}f}}\u00a0m'.format(
                round(length, fracdigits)
            ).replace('_', '\u202f')
        )
    if 'currency' in G.graph:
        desc.append(
            f'{G.size(weight="cost"):_.0f}\u00a0'.replace('_', '\u202f')
            + G.graph['currency']
        )
    return desc


def update_lengths(G):
    """Adds missing edge lengths.

    Changes G in place.
    """
    VertexC = G.graph['VertexC']
    for u, v, dataE in G.edges(data=True):
        if 'length' not in dataE:
            dataE['length'] = np.hypot(*(VertexC[u] - VertexC[v]).T)


def pathdist(G, path):
    """Calculate the total length of a ``path`` of nodes in ``G``.

    Uses the nodes' coordinates (does not rely on edge attributes).
    """
    VertexC = G.graph['VertexC']
    dist = 0.0
    p = path[0]
    for n in path[1:]:
        dist += np.hypot(*(VertexC[p] - VertexC[n]).T).item()
        p = n
    return dist


def count_diagonals(S: nx.Graph, A: nx.Graph) -> int:
    """Count the number of Delaunay diagonals (extended edges) of ``A`` in ``S``.

    Args:
      S: solution topology
      A: available edges used in creating ``S``

    Returns:
      number of non-gate edges of ``S`` that are of kind ``'extended'`` or
        ``'contour_extended'`` (kind is read from ``A``).

    Raises:
      ValueError: if an edge of unknown kind is found.
    """
    delaunay = 0
    extended = 0
    gates = 0
    other = 0
    for u, v in S.edges:
        if u < 0 or v < 0:
            gates += 1
            continue
        kind = A[u][v]['kind']
        if kind is not None:
            if kind.endswith('delaunay'):
                delaunay += 1
            elif kind.endswith('extended'):
                extended += 1
            else:
                other += 1
                raise ValueError('Unknown edge kind: ' + kind)
    assert S.number_of_edges() == delaunay + extended + gates + other
    return extended


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
# id negative; links carrying ``load=0`` (a ring's open point) have no current and
# so no direction to encode.
def bfs_subtree_loads(G, parent, children, subtree):
    """Recurse down the subtree, updating edge and node attributes.

    Meant to be called by :func:`calcload`, but can be used independently (e.g.
    from PathFinder). Nodes must not have a ``'load'`` attribute.

    Returns:
      Total number of descendant nodes
    """
    T = G.graph['T']
    nodeD = G.nodes[parent]
    default = 1 if parent < T else 0  # load is 1 for wtg nodes
    if not children:
        nodeD['load'] = default
        return default
    load = nodeD.get('load', default)
    for child in children:
        G.nodes[child]['subtree'] = subtree
        # a load=0 link is a ring open point: never traverse across it
        grandchildren = {n for n in G[child] if G[child][n].get('load') != 0} - {parent}
        childload = bfs_subtree_loads(G, child, grandchildren, subtree)
        # the child sources the current, the parent sinks it (towards the root)
        G[parent][child].update(load=childload, reverse=child < parent)
        load += childload
    nodeD['load'] = load
    return load


def split_rings_and_calc_loads(S: nx.Graph, A: nx.Graph) -> None:
    """Close path-form ring arms into canonical rings and compute their loads.

    Only the ringed builders (HGS, LKH and the ``method='ringed'`` constructor)
    call this, on a solution ``S`` that is still a set of simple
    ``root → … → root`` paths missing their open points. Each path is walked and
    closed into a canonical ring (see :func:`add_ring_to_S`), using ``A`` to pick
    the longer open-point edge on odd-length rings; a tail already touching a root
    bridges two roots ``(r1, r2)``. Every ring receives exactly one open-point
    link (``load=0``, no current flows through it), and each node's subtree id and
    load, the edges' loads, and the graph's ``max_load`` / ``has_loads`` / root
    loads are set.

    All ringed solvers must call this before returning a solution, so that every
    ringed ``S`` carries exactly one ``load=0`` link per ring.
    """
    # Ring construction: S is path-form (no open points yet). Walk each root's
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
        # (the open point of a bridging stub is a feeder, not an interior link)
        S.nodes[root]['load'] = sum(
            S.nodes[n]['load'] for n in S[root] if S[root][n]['load'] != 0
        )
    S.graph['max_load'] = max_load
    S.graph['has_loads'] = True


def calcload(G: nx.Graph) -> None:
    """Calculate link loads and update edge and node attributes of ``G``.

    ``G`` must already be in final form (a forest, or a ring-form graph whose
    ``load=0`` open points are present). A breadth-first traversal of each root's
    subtree propagates the loads, treating ``load=0`` links (ring open points) as
    breaks. Each node's subtree id and outgoing load land on its ``'subtree'`` /
    ``'load'`` attributes, the edges' ``'load'`` attributes are updated, and the
    graph's ``'max_load'``, ``'has_loads'`` and root loads are set.

    Ring construction — closing path-form arms into rings — lives in
    :func:`split_rings_and_calc_loads`, which the ringed builders call instead.
    """
    R, T = (G.graph[k] for k in 'RT')
    for _, data in G.nodes(data=True):
        if 'load' in data:
            del data['load']

    subtree = 0
    total_load = 0
    max_load = 0
    for root in range(-R, 0):
        G.nodes[root]['load'] = 0
        for subroot in G[root]:
            # a load=0 feeder (degenerate multi-root ring open point) carries no load
            if G[root][subroot].get('load') == 0:
                continue
            _ = bfs_subtree_loads(G, root, [subroot], subtree)
            subtree += 1
            max_load = max(max_load, G.nodes[subroot]['load'])
        total_load += G.nodes[root]['load']
    assert total_load == T, f'counted ({total_load}) != nonrootnodes({T})'
    G.graph['has_loads'] = True
    G.graph['max_load'] = max_load


def add_ring_to_S(
    S: nx.Graph,
    roots: tuple[int, int],
    ordered: list[int],
    subtree: int,
    A: nx.Graph | None = None,
) -> None:
    """Add a single ring (closed loop) to topology graph ``S`` in canonical form.

    A ring is the union of two radial arms, fed by ``r1`` and ``r2`` and joined
    at their tail ends; it bridges two substations when ``r1 != r2``. ``ordered``
    is the terminal sequence ``[t1, ..., tn]`` walked along the ring, so that
    ``t1`` and ``tn`` are the feeder-connected terminals. Both feeders
    ``(r1, t1)`` and ``(r2, tn)`` are real, load-bearing cables; the single open
    point of the ring is the edge at the load midpoint, marked by ``load=0`` (a
    real cable, no current flows through it).

    Arm 1 (the ``t1`` side) gets ``m = ceil(n / 2)`` terminals, so each arm holds at
    most ``ceil(n / 2)`` — i.e. half of the doubled ring capacity. When the ring has
    an even number of nodes (odd ``n``), the middle terminal has two candidate split
    edges yielding balanced arms; if ``A`` is provided, the longer of the two is
    chosen as the open point.

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
    S.graph['topology'] = 'ringed'
    r1, r2 = roots
    n = len(ordered)
    if n == 1:
        # Degenerate ring: a single terminal has feeder(s) on it.
        S.add_node(ordered[0], load=1, subtree=subtree)
        S.add_edge(r1, ordered[0], load=1, reverse=False)
        if r1 != r2:
            S.add_edge(r2, ordered[0], load=0, reverse=False)
        return
    m, mod = divmod(n, 2)
    m += mod  # arm 1 (t1 side) gets ceil(n / 2); open point defaults to t2 side
    if mod and A is not None:
        # odd ring node count: the unique middle terminal has two valid split edges
        #   — keep the open point on the longer one when A is available.
        rev, center, fwd = ordered[m - 2], ordered[m - 1], ordered[m]
        rev_len = A[rev][center]['length'] if A.has_edge(rev, center) else 0
        fwd_len = A[center][fwd]['length'] if A.has_edge(center, fwd) else 0
        if rev_len > fwd_len:
            m -= 1
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
            # open point of the ring: real cable, no current (marked by load=0),
            # so it has no flow direction to encode
            S.add_edge(u, v, load=0, reverse=False)
        else:
            load = m - 1 - i if i < m else i - m + 1
            # current flows towards the arm's feeder, i.e. towards the heavier end
            u_lighter = S.nodes[u]['load'] < S.nodes[v]['load']
            source, sink = (u, v) if u_lighter else (v, u)
            S.add_edge(u, v, load=load, reverse=source < sink)


def rings_from_links(active_links, R: int) -> list[tuple[tuple[int, int], list[int]]]:
    """Recover ordered ring terminal sequences from a set of active links.

    ``active_links`` is any iterable of ``(u, v)`` node pairs that are active in a
    RINGED solution (terminal-terminal links plus the two feeders of each ring).
    Each ring is returned as ``((r1, r2), [t1, ..., tn])`` with ``t1`` and ``tn``
    the feeder-connected terminals, obtained by walking the terminal adjacency
    from the head subroot to the tail one; ``r1`` feeds ``t1`` and ``r2`` feeds
    ``tn``. The ring bridges two substations when ``r1 != r2``.

    Feeders are identified by having exactly one negative (root) endpoint; a ring
    with a single terminal (``n == 1``) has both feeders on that terminal.
    """
    term_adj: dict[int, set[int]] = defaultdict(set)
    subroots: dict[int, list[int]] = defaultdict(list)
    term_to_roots: dict[int, list[int]] = defaultdict(list)
    for u, v in active_links:
        if u >= 0 and v >= 0:
            term_adj[u].add(v)
            term_adj[v].add(u)
        else:
            r, t = (u, v) if u < 0 else (v, u)
            subroots[r].append(t)
            term_to_roots[t].append(r)
    rings: list[tuple[tuple[int, int], list[int]]] = []
    # `subroots` is consumed as the walk goes: each feeder is claimed once
    for r in range(-R, 0):
        while subroots[r]:
            t1 = subroots[r].pop(0)
            chain_ = [t1]
            prev, curr = None, t1
            while True:
                nxts = [x for x in term_adj[curr] if x != prev]
                if not nxts:
                    break
                prev, curr = curr, nxts[0]
                chain_.append(curr)
            tn = chain_[-1]
            # claim the tail feeder: a ring of n > 1 always has one, a lone
            # terminal only if it bridges two roots
            if tn in subroots[r]:
                r2 = r
            else:
                r2 = next(
                    (rc for rc in term_to_roots[tn] if rc != r and tn in subroots[rc]),
                    None,
                )
            if r2 is None:
                r2 = r
            else:
                subroots[r2].remove(tn)
            rings.append(((r, r2), chain_))
    return rings


def _validate_ringed(S: nx.Graph, capacity: int | None) -> list[str]:
    """Violations of the canonical RINGED shape (see :func:`add_ring_to_S`)."""
    violations = []
    R, T = S.graph['R'], S.graph['T']

    # topology-graph ring edges are pure geometry: they carry no edge 'kind'
    kinds = {d.get('kind') for _, _, d in S.edges(data=True)} - {None}
    if kinds:
        violations.append(
            f'ring edges must not carry a kind, got {sorted(map(str, kinds))}'
        )

    rings = rings_from_links(list(S.edges()), R)

    # the rings partition the terminal set: every terminal in exactly one ring
    covered = sorted(t for _, ordered in rings for t in ordered)
    if covered != list(range(T)):
        violations.append('rings must partition the terminals')

    # two feeders per ring, except a lone terminal hanging off a single root
    n_feeders = sum(1 for u, v in S.edges if u < 0 or v < 0)
    expected = sum(
        1 if len(ordered) == 1 and rs[0] == rs[1] else 2 for rs, ordered in rings
    )
    if n_feeders != expected:
        violations.append(f'expected {expected} feeders, got {n_feeders}')

    if not S.graph.get('has_loads'):
        return violations

    for roots, ordered in rings:
        n = len(ordered)
        arm = math.ceil(n / 2)
        # a ring holds up to 2*capacity terminals (two arms of ceil(n / 2))
        if capacity is not None and arm > capacity:
            violations.append(f'ring at {roots} needs arms of {arm} > κ = {capacity}')
        # the heaviest node of a ring carries a full arm: the arms are balanced
        heaviest = max(S.nodes[t]['load'] for t in ordered)
        if heaviest != arm:
            violations.append(
                f'ring at {roots} has unbalanced arms: heaviest node carries '
                f'{heaviest}, balanced arms would carry {arm}'
            )
        # the two arm-head (feeder) terminals carry the whole ring between them
        heads = S.nodes[ordered[0]]['load'] + (
            S.nodes[ordered[-1]]['load'] if n > 1 else 0
        )
        if heads != n:
            violations.append(
                f'ring at {roots} spans {n} terminals, but its arm heads carry {heads}'
            )
        # exactly one open point per ring: a real cable with no current through it
        opens = [
            (u, v)
            for u, v in zip(ordered, ordered[1:])
            if S.has_edge(u, v) and S[u][v]['load'] == 0
        ]
        if len(opens) != (1 if n > 1 else 0):
            violations.append(
                f'ring at {roots} has {len(opens)} open points, expected '
                f'{1 if n > 1 else 0}'
            )
    return violations


def validate_topology(S: nx.Graph, capacity: int | None = None) -> list[str]:
    """Check ``S`` against the invariants of the topology it declares.

    The canonical shape of a solution is a contract of the library, not of the
    test suite, so the invariants live next to the builders that establish them.

    Args:
      S: topology graph to check. ``S.graph['topology']`` is mandatory: it is
        one of ``'ringed'``, ``'radial'`` or ``'branched'``.
      capacity: cable capacity; defaults to ``S.graph['capacity']``. Capacity
        checks are skipped when neither is available.

    Returns:
      list of human-readable violations; ``S`` is valid if it is empty.

    Example::

      violations = validate_topology(S, capacity)
      if violations:
          print('\\n'.join(violations))

    """
    violations = []
    R, T = S.graph['R'], S.graph['T']
    if capacity is None:
        capacity = S.graph.get('capacity')
    topology = S.graph['topology']

    # --- universal invariants ------------------------------------------------
    if S.graph.get('has_loads'):
        edge_loads = [d['load'] for _, _, d in S.edges(data=True)]
        max_load = max(edge_loads, default=0)
        if capacity is not None and max_load > capacity:
            violations.append(f'κ = {capacity}, max_load = {max_load}')
        # all terminals are accounted for at the roots
        total = sum(S.nodes[r]['load'] for r in range(-R, 0))
        if total != T:
            violations.append(f'root loads sum to {total}, expected T = {T}')

    # --- topology shape ------------------------------------------------------
    if topology == 'ringed':
        violations += _validate_ringed(S, capacity)
    elif topology in ('radial', 'branched'):
        if not nx.is_forest(S):
            violations.append(f'{topology} topology must be a forest')
        if topology == 'radial':
            degrees = [S.degree(t) for t in range(T)]
            if max(degrees, default=0) > 2:
                violations.append('radial subtrees must be simple paths')
    else:
        raise ValueError(f'unknown topology: {topology!r}')
    return violations


def directed_links(S: nx.Graph) -> Iterator[tuple[int, int, int]]:
    """Yield ``(source, sink, flow)`` for every link of ``S``.

    Forest topologies read each link's orientation off its ``'reverse'`` flag
    (see the note above :func:`bfs_subtree_loads`), so ``flow`` is just the
    link's load.

    A RINGED ``S`` -- as declared by ``S.graph['topology']`` -- stores each ring
    split into two arms at a load-0 open point, which is not how a flow
    formulation sees it: there a ring is one directed chain of its ``n``
    terminals, fed by a flowless closing feeder at one end and draining through
    a feeder carrying the whole ring at the other. Such rings are *radialized*
    into that chain here (walking across the open point with
    :func:`rings_from_links`), so the open point becomes an ordinary
    flow-carrying link.

    A ring bridging two roots drains through the one feeding the head of the
    walk and closes on the other; which of the two drains is arbitrary, as it
    moves no cable.

    Args:
      S: solution topology.

    Yields:
      ``(source, sink, flow)`` per link, current flowing ``source`` -> ``sink``.
      ``flow`` is 0 for links carrying no current: a ring's closing feeder.
    """
    if S.graph['topology'] != 'ringed':
        for u, v, edgeD in S.edges(data=True):
            source, sink = (u, v) if ((u < v) == edgeD['reverse']) else (v, u)
            yield source, sink, edgeD['load']
        return
    for root, chain_ in rings_from_links(S.edges(), S.graph['R']):
        head_root, tail_root = root
        n = len(chain_)
        # the ring drains through chain_[0], whose feeder carries all of it, and
        # closes on the far feeder. A lone terminal needs no special case: it is
        # both head and tail, and the chain below is empty.
        yield chain_[0], head_root, n
        yield tail_root, chain_[-1], 0
        for j in range(n - 1, 0, -1):
            yield chain_[j], chain_[j - 1], n - j


def site_fingerprint(
    VertexC: np.ndarray, boundary: np.ndarray
) -> tuple[bytes, dict[str, bytes]]:
    #  VertexCpkl = pickle.dumps(np.round(VertexC, 2))
    #  boundarypkl = pickle.dumps(np.round(boundary, 2))
    VertexCpkl = pickle.dumps(VertexC)
    boundarypkl = pickle.dumps(boundary)
    return (
        sha256(VertexCpkl + boundarypkl).digest(),
        dict(VertexC=VertexCpkl, boundary=boundarypkl),
    )


def fun_fingerprint(fun=None) -> dict[str, bytes | str]:
    if fun is None:
        fcode = sys._getframe().f_back.f_code
    else:
        fcode = fun.__code__
    return dict(
        funhash=sha256(fcode.co_code).digest(),
        funfile=fcode.co_filename,
        funname=fcode.co_name,
    )


def L_from_site(
    *,
    VertexC: np.ndarray,
    T: int,
    R: int,
    B: int = 0,
    border: np.ndarray | None = None,
    obstacles: list[np.ndarray] | None = None,
    name: str = '',
    handle: str = 'L_from_site',
    landscape_angle: float | None = None,
) -> nx.Graph:
    """Create L from a location's attributes.

    Args:
      VertexC: numpy.ndarray (V, 2) with all (x, y) coordinates (V = R + T + B)
      T: int number of wtg
      R: int number of oss
      B: number of border and obstacle zones' vertices
      border: array (B,) of VertexC indices that define the border (ccw)
      obstacles: sequence of numpy.ndarray of VertexC indices
      name: site name
      handle: site identifier

    Returns:
      Graph containing ``N = R + T`` nodes and no edges (all args become graph
      attributes).
    """
    L = nx.Graph(T=T, R=R, B=B, VertexC=VertexC, name=name, handle=handle)
    if border is not None:
        L.graph['border'] = border
    if obstacles is not None:
        L.graph['obstacles'] = obstacles
    if landscape_angle is not None:
        L.graph['landscape_angle'] = landscape_angle
    L.add_nodes_from(range(T), kind='wtg')
    L.add_nodes_from(range(-R, 0), kind='oss')
    return L


def G_from_S(S: nx.Graph, A: nx.Graph) -> nx.Graph:
    """Create G from S and A.

    Graph ``S`` contains the topology of a routeset network (nodes only, no
    contours or detours). ``S`` must have been created from the available edges
    in ``A``, whose contour information is used to obtain a routeset ``G``
    (possibly with contours, but not with detours – use PathFinder afterward).
    """
    R, T, B = (A.graph[k] for k in 'RTB')
    VertexC, d2roots, diagonals = (
        A.graph[k] for k in ('VertexC', 'd2roots', 'diagonals')
    )
    G = nx.create_empty_copy(S)
    for k in (
        'B',
        'border',
        'obstacles',
        'name',
        'handle',
        'landscape_angle',
        'norm_scale',
        'norm_offset',
        'is_normalized',
    ):
        value = A.graph.get(k)
        if value is not None:
            G.graph[k] = value

    stunts_primes = A.graph.get('stunts_primes')
    if stunts_primes:
        num_stunts = len(stunts_primes)
        G.graph['B'] -= num_stunts
    else:
        num_stunts = 0
    # remove supertriangle and stunts coordinates from VertexC
    G.graph['VertexC'] = np.vstack((VertexC[: -R - 3 - num_stunts], VertexC[-R:]))

    nx.set_node_attributes(
        G,
        {n: label for n, label in A.nodes(data='label') if label is not None},
        'label',
    )
    nx.set_node_attributes(G, 'wtg', 'kind')
    for r in range(-R, 0):
        G.nodes[r]['kind'] = 'oss'
    if 'is_normalized' in A.graph:
        G.graph['is_normalized'] = True
    # non_A_edges are the far-reaching gates and ocasionally the result of
    # a poor solver (e.g. LKH-3)
    non_A_edges = S.edges - A.edges
    # TA_source, TA_target = np.array(S.edges - non_A_edges).T
    common_TA = S.edges - non_A_edges
    iC = T + B
    clone2prime = []
    tentative = []
    shortened_contours = {}
    num_diagonals = 0
    # add to G the S edges that are in A
    for edge in common_TA:
        s, t = edge if edge[0] < edge[1] else edge[::-1]
        is_split = S.get_edge_data(s, t, {}).get('load') == 0
        AedgeD = A[s][t]
        subtree_id = S.nodes[t]['subtree']
        # only count diagonals that are not gates
        num_diagonals += AedgeD['kind'] == 'extended' and s >= 0
        midpath = AedgeD.get('midpath')

        # split edges are ring open points (load=0: no current flows). The open
        # point keeps its geometry kind — it may follow a contour like any edge.
        if is_split:
            load = S[s][t]['load']
            st_reverse = False
            if midpath is None:
                G.add_edge(
                    s,
                    t,
                    length=AedgeD['length'],
                    load=load,
                    reverse=st_reverse,
                )
                continue
            # has a contour: fall through to contour expansion below
        else:
            # This block checks for gate×edge crossings, which may be unnecessary
            # depending on how S was generated. (e.g. creator == 'MILP...' and
            # gateXings_constraint == True).
            st_is_tentative = False
            if s < 0:
                # ⟨s, t⟩ is a gate
                if midpath is not None:
                    # While we do not have magic portals, make all contoured gate
                    # of kind tentative, so that we do not block access to root
                    # around a contour node.
                    st_is_tentative = True
                elif (s, t) in diagonals:
                    # ⟨s, t⟩ is a diagonal
                    u, v = diagonals[(s, t)]
                    if (u, v) in S.edges:
                        # ⟨s, t⟩'s Delaunay is in S -> Xing
                        st_is_tentative = True
                    else:
                        # check the other diagonals that cross ⟨s, t⟩ (in A)
                        for side in ((u, s), (s, v), (v, t), (t, u)):
                            side = side if side[0] < side[1] else side[::-1]
                            if side in diagonals.inv and diagonals.inv[side] in S.edges:
                                # side's diagonal is in S -> Xing
                                st_is_tentative = True
                                break
                elif (s, t) in diagonals.inv and diagonals.inv[(s, t)] in S.edges:
                    # ⟨s, t⟩ is a Delanay edge and its diagonal is in S -> Xing
                    st_is_tentative = True

            load = S[s][t]['load']
            # current flows towards the heavier end (the one nearer a root)
            st_source, st_sink = (
                (s, t) if S.nodes[s]['load'] < S.nodes[t]['load'] else (t, s)
            )
            st_reverse = st_source < st_sink
            if st_is_tentative:
                G.add_edge(
                    s,
                    t,
                    length=AedgeD['length'],
                    load=load,
                    reverse=st_reverse,
                    kind='tentative',
                )
                tentative.append((s, t))
                continue
            if midpath is None:
                # no contour in A's ⟨s, t⟩ -> straightforward
                G.add_edge(s, t, length=AedgeD['length'], load=load, reverse=st_reverse)
                continue

        # contour edge (reached for regular contour edges and split edges with
        # contour); split-ness rides on load=0, so the kind stays 'contour'
        edge_kind = 'contour'
        shortcuts = AedgeD.get('shortcuts')
        if shortcuts is not None:
            if len(shortcuts) == len(midpath):
                # contour is a glitch of make_planar_embedding's P_paths
                if s < 0:
                    # ⟨s, t⟩ is a gate -> make it tentative
                    # This is a hack. It will force PathFinder to check for
                    # crossings and the edge will be confirmed a non-A gate.
                    G.add_edge(
                        s,
                        t,
                        kind='tentative',
                        reverse=False,
                        load=load,
                        length=np.hypot(*(VertexC[s] - VertexC[t]).T),
                    ).item()
                    tentative.append((s, t))
                    continue
                G.add_edge(
                    s,
                    t,
                    reverse=st_reverse,
                    load=load,
                    length=AedgeD['length'],
                )
                shortened_contours[(s, t)] = midpath, []
                continue
            shortpath = midpath.copy()
            for short in shortcuts:
                shortpath.remove(short)
            shortened_contours[(s, t)] = midpath, shortpath
            midpath = shortpath
        path = [s] + midpath + [t]
        lengths = np.hypot(*(VertexC[path[1:]] - VertexC[path[:-1]]).T)
        u = s
        for prime, length in zip(path[1:-1], lengths):
            clone2prime.append(prime)
            v = iC
            iC += 1
            G.add_node(v, kind='contour', load=load, subtree=subtree_id)
            reverse = st_reverse == (u < v)
            G.add_edge(
                u,
                v,
                length=length.item(),
                load=load,
                kind=edge_kind,
                reverse=reverse,
                A_edge=(s, t),
            )
            u = v
        reverse = st_reverse == (u < t)
        G.add_edge(
            u,
            t,
            length=lengths[-1].item(),
            load=load,
            kind=edge_kind,
            reverse=reverse,
            A_edge=(s, t),
        )
    if shortened_contours:
        G.graph['shortened_contours'] = shortened_contours
    if clone2prime:
        if stunts_primes:
            # Contour clones may address stunt vertices, which were dropped from
            # the compacted VertexC above. Map them to their original primes so
            # the emitted fnT stays consistent with VertexC. (PathFinder later
            # closes the stunt-id gap in the clone *node* numbering and remaps
            # any detour clones it adds that trace through stunts.)
            first_stunt = T + G.graph['B']
            stunt2prime = {
                first_stunt + i: prime for i, prime in enumerate(stunts_primes)
            }
            clone2prime = [stunt2prime.get(prime, prime) for prime in clone2prime]
        fnT = np.arange(iC + R)
        fnT[T + B : -R] = clone2prime
        fnT[-R:] = range(-R, 0)
        G.graph.update(fnT=fnT, C=len(clone2prime))
    # add to G the S edges that are not in A
    rogue = []
    for s, t in non_A_edges:
        s, t = (s, t) if s < t else (t, s)
        if s < 0:
            # far-reaching gate (includes a ring's second feeder when not in A):
            # a real cable, same physical route as a regular feeder
            G.add_edge(
                s,
                t,
                length=d2roots[t, s].item(),
                kind='tentative',
                load=S.nodes[t]['load'],
                reverse=False,
            )
            tentative.append((s, t))
        else:
            # rogue edge (not supposed to be on the routeset, poor solver)
            st_reverse = S.edges[s, t]['reverse']
            load = S.nodes[s]['load'] if st_reverse else S.nodes[t]['load']
            G.add_edge(
                s,
                t,
                length=np.hypot(*(VertexC[s] - VertexC[t])).item(),
                kind='rogue',
                load=load,
                reverse=st_reverse,
            )
            rogue.append((s, t))
    if rogue:
        G.graph['rogue'] = rogue

    # Check on crossings between G's gates that are in A and G's edges
    diagonals = A.graph['diagonals']
    P = A.graph['planar']
    for r in range(-R, 0):
        for n in set(S.neighbors(r)) & set(A.neighbors(r)):
            #  TODO: if ⟨r, n⟩ is a contour in A, G[r][n] might fail. FIXIT
            st = diagonals.get((r, n))
            if st is not None:
                # st is a Delaunay edge
                if st in G.edges:
                    G[r][n]['kind'] = 'tentative'
                    tentative.append((r, n))
                    continue
                crossings = False
                s, t = st
                # ensure u–s–v–t is ccw
                u, v = (r, n) if (P[r][t]['cw'] == s and P[n][s]['cw'] == t) else (n, r)
                # examine the two triangles ⟨s, t⟩ belongs to
                for a, b, c in ((s, t, u), (t, s, v)):
                    # this is for diagonals crossing diagonals
                    cbD = P[c].get(b)
                    # was triangle edge removed (constraint Xing)? if yes, no diagonal
                    if cbD is not None:
                        d = cbD['ccw']
                        diag_da = (a, d) if a < d else (d, a)
                        if d == P[b][c]['cw'] and diag_da in G.edges:
                            crossings = True
                            break
                    acD = P[a].get(c)
                    # was triangle edge removed (constraint Xing)? if yes, no diagonal
                    if acD is not None:
                        e = acD['ccw']
                        diag_eb = (e, b) if e < b else (b, e)
                        if e == P[c][a]['cw'] and diag_eb in G.edges:
                            crossings = True
                            break
                if crossings:
                    G[r][n]['kind'] = 'tentative'
                    tentative.append((r, n))
                    continue
            else:
                uv = diagonals.inv.get((r, n))
                if uv is not None and uv in G.edges:
                    # uv is a Delaunay edge crossing ⟨r, n⟩
                    G[r][n]['kind'] = 'tentative'
                    tentative.append((r, n))
                    continue
    if tentative:
        G.graph['tentative'] = tentative

    G.graph.update(
        num_diagonals=num_diagonals,
    )
    return G


def S_from_G(G: nx.Graph) -> nx.Graph:
    """Get ``G``'s topology (contours, detours, lengths, coords are dropped).

    If using S to warm-start a MILP model, call after :func:`S_from_G`:
      * :func:`as_hooked_to_nearest`: if the model uses ``topology='branched'``
      * :func:`as_hooked_to_head`: if the model uses ``topology='radial'``

    This ensures that topology ``S`` is feasible (if radial) and not
    trivially suboptimal (if branched).

    RINGED routesets are supported: the rings' cycle-closing links are preserved
    (see the traversal note below), so ``S`` keeps the ring partition of ``G``.

    Args:
        G: must contain a feasible solution (tree, path or ring)

    Returns:
        Topology of ``G``
    """
    R, T = (G.graph[k] for k in 'RT')
    capacity = G.graph['capacity']
    has_loads = G.graph.get('has_loads', False)
    S = nx.Graph(
        T=T,
        R=R,
        capacity=capacity,
    )

    def is_real(n: int) -> bool:
        "Only roots and terminals survive in S (border vertices and clones do not)."
        return n < T

    for r in range(-R, 0):
        S.add_node(r, kind='oss', **({'load': G.nodes[r]['load']} if has_loads else {}))
    for t in sorted(n for n in G if 0 <= n < T):
        if has_loads:
            S.add_node(
                t, kind='wtg', load=G.nodes[t]['load'], subtree=G.nodes[t]['subtree']
            )
        else:
            S.add_node(t, kind='wtg')

    # Links already joining two real nodes carry over verbatim, keeping ``G``'s
    # own orientation: 'reverse' is relative to the stored node order, and the
    # RINGED builders (:func:`add_ring_to_S`) and the forest ones
    # (:func:`bfs_subtree_loads`) give it different meanings — copying sidesteps
    # having to pick one.
    for u, v, edgeD in G.edges(data=True):
        if is_real(u) and is_real(v):
            if has_loads:
                S.add_edge(u, v, load=edgeD['load'], reverse=edgeD['reverse'])
            else:
                S.add_edge(u, v)

    # Every remaining link runs through a chain of non-real nodes (border
    # vertices, contour and detour clones), which collapses to the single link
    # joining the real nodes at its ends. Walking outward from each real node --
    # rather than following a DFS *tree* -- is what keeps the cycle-closing links
    # of a RINGED topology: a ring's second feeder is a DFS back edge, so a tree
    # traversal drops it and silently merges the two rings it separates.
    for s in sorted(n for n in G if is_real(n)):
        for nbr in G[s]:
            if is_real(nbr):
                continue
            prev, node = s, nbr
            while not is_real(node):
                fwd = [x for x in G[node] if x != prev]
                if len(fwd) != 1:
                    break
                prev, node = node, fwd[0]
            if not is_real(node) or node == s or S.has_edge(s, node):
                continue
            if has_loads:
                # orient parent -> child (the parent carries the heavier load,
                # being the closer of the two to a root). A chain spanning a root
                # is a feeder, which both conventions above agree to leave
                # unreversed (a root's id is negative, so always the lower one).
                s_load, node_load = G.nodes[s]['load'], G.nodes[node]['load']
                u, v = (s, node) if s_load >= node_load else (node, s)
                S.add_edge(u, v, load=G.edges[s, nbr]['load'], reverse=u > v)
            else:
                S.add_edge(s, node)

    creator = G.graph.get('creator')
    if creator is not None:
        S.graph['creator'] = creator
    method_options = G.graph.get('method_options')
    if method_options is not None:
        S.graph['method_options'] = method_options
    S.graph['topology'] = G.graph['topology']
    if has_loads:
        S.graph['has_loads'] = True
    else:
        calcload(S)
    return S


def L_from_G(G: nx.Graph) -> nx.Graph:
    """Return new location with nodes and site attributes from G.

    The returned location graph ``L`` retains only roots, nodes and basic graph
    attributes. All edges and remaining attributes are not carried from ``G``.

    Args:
      G: routeset graph to extract site data from.

    Returns:
      Site graph (no edges) with lean attributes.
    """
    R, T = (G.graph[k] for k in 'RT')
    L = nx.Graph(**{k: G.graph[k] for k in _essential_graph_attrs if k in G.graph})

    # TODO: remove this entire legacy compatibility block after a couple of releases.
    # BEGIN: Legacy compatibility block for graphs whose VertexC/B still reflect stunts
    num_stunts = G.graph.get('num_stunts')
    if num_stunts:
        VertexC = G.graph['VertexC']
        base_B = G.graph['B'] - num_stunts
        L.graph['VertexC'] = np.vstack((VertexC[: T + base_B], VertexC[-R:]))
        L.graph['B'] = base_B
    stunts_primes = G.graph.get('stunts_primes')
    if stunts_primes:
        VertexC = L.graph['VertexC']
        L.graph['VertexC'] = np.vstack(
            (VertexC[: -R - len(stunts_primes)], VertexC[-R:])
        )
        L.graph['B'] -= len(stunts_primes)
    # END: Legacy compatibility block

    L.add_nodes_from(
        ((n, {'label': label}) for n, label in G.nodes(data='label') if 0 <= n < T),
        kind='wtg',
    )
    for r in range(-R, 0):
        L.add_node(r, label=G.nodes[r].get('label'), kind='oss')
    return L


def compress_ring_routes(routes: list[tuple[int, list[int], int]]) -> list[int]:
    """Flatten ring routes ``(open_root, walk, close_root)`` into a sequence.

    Each route is written as its ``open_root`` (only when it differs from the
    running root), its walk, then its ``close_root``. A shared boundary root
    (``close_i == open_{i+1}``) is therefore written once, so two consecutive
    negative numbers are always distinct; the last route's ``close`` is elided
    when it equals its ``open`` (a single-root ring, inferred at end-of-sequence).

    Every route contributes its walk and the first writes a marker, so the
    sequence always has more entries than walk nodes -- which is what
    distinguishes a ringed encoding from a positional forest one.
    """
    seq: list[int] = []
    pen = None
    last = len(routes) - 1
    for idx, (open_, walk, close) in enumerate(routes):
        if open_ != pen:
            seq.append(int(open_))
        seq.extend(int(x) for x in walk)
        if idx == last and close == open_:
            pen = close
            continue
        seq.append(int(close))
        pen = close
    return seq


def parse_ring_routes(seq: Sequence[int]) -> list[tuple[int, list[int], int]]:
    """Inverse of :func:`compress_ring_routes`: recover ``(open, walk, close)``."""
    routes: list[tuple[int, list[int], int]] = []
    seq = [int(x) for x in seq]
    i, length, pen = 0, len(seq), None
    while i < length:
        if seq[i] < 0:
            open_ = seq[i]
            i += 1
        else:
            open_ = pen
        walk: list[int] = []
        while i < length and seq[i] >= 0:
            walk.append(seq[i])
            i += 1
        if i >= length:
            close = open_  # last route, single-root: close inferred == open
        else:
            # a lone marker is the shared close/next-open; two consecutive markers
            # are this route's close then a distinct next open
            close = seq[i]
            i += 1
        pen = close
        routes.append((open_, walk, close))
    return routes


def _ring_routes_from_S(S: nx.Graph, R: int) -> list[tuple[int, list[int], int]]:
    """Recover the ring routes ``(open_root, walk, close_root)`` of a ringed ``S``.

    The walk direction of each ring is chosen so that the open point falls where
    :func:`add_ring_to_S` (called with ``A=None`` on decode) re-derives it: the
    load midpoint ``ceil(n / 2) - 1``. For a ring with an odd number of terminals
    the two balanced split edges map one-to-one onto the two walk directions, so
    this preserves the exact open point without storing it.

    A ring may open and close on different roots, so reversing a walk swaps the
    two roots along with the ends they feed.
    """
    split_pairs = {
        frozenset((u, v)) for u, v, d in S.edges(data=True) if d.get('load') == 0
    }
    routes: list[tuple[int, list[int], int]] = []
    for root, ordered in rings_from_links(list(S.edges()), R):
        open_, close = root
        n = len(ordered)
        if n > 1:
            # locate the open point in the walk and orient so it lands on the
            # decoder's default split edge (index ceil(n / 2) - 1)
            j = next(
                k
                for k in range(n - 1)
                if frozenset((ordered[k], ordered[k + 1])) in split_pairs
            )
            if j != math.ceil(n / 2) - 1:
                ordered = ordered[::-1]
                open_, close = close, open_
        routes.append((open_, ordered, close))
    return routes


def _ring_sequence_from_S(S: nx.Graph, R: int) -> list[int]:
    """Encode the rings of a RINGED topology ``S`` as a flat sequence.

    The rings are drawn as routes, each leaving a root and returning to a root
    -- not necessarily the same one. See :func:`S_from_terse_links` for the
    inverse, and :func:`compress_ring_routes` for the sequence layout.
    """
    return compress_ring_routes(_ring_routes_from_S(S, R))


def S_from_terse_links(terse_links, R=None, T=None, **kwargs):
    """Create a solution topology graph ``S`` from its ``terse_links`` encoding.

    Inverse function of :func:`terse_links_from_S`. Handles both encodings:

    * *forest* topologies (radial/branched) are stored positionally – the array
      has exactly ``T`` entries and ``(i, terse_links[i])`` is a directed link;
    * *ringed* topologies are stored as a sequence of routes (see
      :func:`_ring_sequence_from_S`), which has a number of entries different
      from ``T``.

    The two are told apart by comparing the number of entries with ``T`` (see
    :func:`terse_links_from_S`): equal ⇒ forest, otherwise ⇒ ringed. ``T``
    and ``R`` may be supplied explicitly; when omitted a forest encoding is
    assumed (``T = len(terse_links)``) and ``R`` is inferred from the array.

    Args:
      terse_links: topology encoded as a 1D array.
      R: number of roots of the problem (inferred when omitted).
      T: number of terminals of the problem (inferred when omitted).

    Returns:
      Solution topology S.
    """
    terse_links = np.asarray(terse_links)
    n = terse_links.shape[0]
    if R is None:
        R = abs(int(terse_links.min())) if n else 1
    if T is None:
        # No T given: assume the (unambiguous) forest encoding, where n == T.
        T = n
    if n == T:
        # forest encoding: (i, terse_links[i]) is a directed link
        S = nx.Graph(T=T, R=R, **kwargs)
        S.add_edges_from(tuple(zip(range(T), terse_links.tolist())))
        calcload(S)
        # the encoding does not distinguish radial from branched: claim the
        # weaker of the two unless the caller knows better
        S.graph.setdefault('topology', 'branched')
        if 'capacity' not in kwargs:
            S.graph['capacity'] = S.graph['max_load']
        return S
    # ringed encoding: rebuild each route as a canonical ring
    S = nx.Graph(T=T, R=R, **kwargs)
    S.add_nodes_from(range(-R, 0))
    routes = parse_ring_routes(terse_links.tolist())
    max_load = 0
    for subtree, (open_, ordered, close) in enumerate(routes):
        add_ring_to_S(S, (open_, close), ordered, subtree, A=None)
        max_load = max(max_load, math.ceil(len(ordered) / 2))
    for root in range(-R, 0):
        # a load=0 feeder carries no current, so it adds nothing to its root
        S.nodes[root]['load'] = sum(
            S.nodes[nbr]['load'] for nbr in S[root] if S[root][nbr]['load'] != 0
        )
    S.graph['max_load'] = max_load
    S.graph['has_loads'] = True
    if 'capacity' not in kwargs:
        S.graph['capacity'] = max_load
    return S


def terse_links_from_S(S):
    """Make a terse representation of the topology ``S`` as a 1D array.

    Inverse function of :func:`S_from_terse_links`. A *forest* topology (radial
    or branched) is stored positionally – ``(i, terse[i])`` are the links, one
    entry per terminal. A *ringed* topology needs two feeders per ring, which a
    single parent per node cannot capture, so it is stored instead as a sequence
    of routes (see :func:`_ring_sequence_from_S`): the terminals of each ring in
    walking order, with (negative) root numbers marking the roots each route
    opens and closes on. The two encodings are told apart by their length
    relative to ``T``.

    The encoding follows ``S.graph['topology']``.

    Args:
      S: solution topology (a forest, or a canonical ringed topology).

    Returns:
      1D array ``terse`` encoding ``S``.
    """
    T = S.graph['T']
    if S.graph['topology'] != 'ringed':
        terse_links = np.zeros((T,), dtype=np.int_)
        # convert the graph to array representing the tree (edges i->terse[i])
        for u, v, edgeD in S.edges(data=True):
            u, v = (u, v) if u < v else (v, u)
            i, target = (u, v) if edgeD['reverse'] else (v, u)
            terse_links[i] = target
        return terse_links
    # ringed topology: encode the routes sequentially. Every route contributes
    # its terminals and the first one always writes a root marker, so a ringed
    # encoding always has more than T entries and never collides with the
    # T-entry forest encoding.
    seq = _ring_sequence_from_S(S, S.graph['R'])
    return np.array(seq, dtype=np.int_)


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
        (list() for _ in range(num_subtree)), count=num_subtree, dtype=object
    )
    subtree_from_node = np.empty((T,), dtype=object)
    for n, subtree_id in G.nodes(data='subtree'):
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
            for node in subtree:
                del G.nodes[node]['load']

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
        (list() for _ in range(num_subtree)), count=num_subtree, dtype=object
    )
    subtree_from_node = np.empty((T,), dtype=object)
    headtail_from_subtree_id = np.fromiter(
        (list() for _ in range(num_subtree)), count=num_subtree, dtype=object
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


def make_remap(G, refG, H, refH):
    """Create a mapping between two representations of the same site.

    CAUTION: only WTG node remapping is implemented.

    If the nodes in ``G`` and in ``H`` represent the same site, but have different
    orientation, scale and node order, the mapping produced here can be used
    with ``NetworkX.relabel_nodes(G, remap)`` to translate a routeset in G to a
    routeset in H.

    Args:
      G: routeset with obsolete representation.
      refG: two nodes to used as references.
      H: routeset with valid representation.
      refH: two nodes corresponding to ``refG``
    """
    T = G.graph['T']
    VertexC = G.graph['VertexC'][:T]
    vecref = VertexC[refG[1]] - VertexC[refG[0]]
    angleG = np.arctan2(*vecref)
    scaleG = np.hypot(*vecref)
    GvertC = (VertexC - VertexC[refG[0]]) / scaleG
    VertexC = H.graph['VertexC'][:T]
    vecref = VertexC[refH[1]] - VertexC[refH[0]]
    angleH = np.arctan2(*vecref)
    scaleH = np.hypot(*vecref)
    HvertC = rotate(
        (VertexC - VertexC[refH[0]]) / scaleH, 180 * (angleH - angleG) / np.pi
    )
    remap = {}
    for i, coordH in enumerate(HvertC):
        j = np.argmin(np.hypot(*(GvertC - coordH).T))
        remap[j] = i
    return remap


def add_terminal_closest_root(A: nx.Graph) -> None:
    """Add attributes ``'root'`` to terminals and ``'rootmap__'`` to ``A``.

    Changes A in-place.

    * node attribute ``'root'`` is the index of the root closest to node.
    * graph attribute ``'rootmap__'`` is an R-long list of T-long bitarrays.

    Args:
      A: available-links graph
    """
    R = A.graph['R']
    T = A.graph['T']
    closest_root_ = np.argmin(A.graph['d2roots'], axis=1) - R
    nx.set_node_attributes(
        A, {n: r.item() for n, r in enumerate(closest_root_)}, 'root'
    )
    # while 'd2roots' includes border vertices, 'rootmap__' must not
    A.graph['rootmask__'] = [
        bitarray((closest_root_[:T] == r).tolist()) for r in range(-R, 0)
    ]


@nb.njit(cache=True)
def _blockmap_inner(u, v, angle__, angle_rank__, VertexC, R, T):
    """Compute blockage bitmap for edge (u, v) across all roots.

    Returns an ``(R, T)`` boolean array where ``True`` means turbine ``t`` is blocked
    by edge ``(u, v)`` with respect to root ``r``.
    """
    root_offset = VertexC.shape[0] - R
    blocked = np.zeros((R, T), dtype=np.bool_)
    uC = VertexC[u]
    vC = VertexC[v]
    vec_x = vC[0] - uC[0]
    vec_y = vC[1] - uC[1]
    for r in range(R):
        uR = angle_rank__[u, r]
        vR = angle_rank__[v, r]
        uv_angle = angle__[v, r] - angle__[u, r]
        if uv_angle < 0:
            uR, vR = vR, uR
        root_idx = root_offset + r
        rootC_x = VertexC[root_idx, 0]
        rootC_y = VertexC[root_idx, 1]
        root_cross = (rootC_x - uC[0]) * vec_y - (rootC_y - uC[1]) * vec_x
        is_root_sign_pos = root_cross > 0
        if abs(uv_angle) <= np.pi:
            for t in range(T):
                ar = angle_rank__[t, r]
                if ar <= uR or ar >= vR:
                    continue
                w_cross = (VertexC[t, 0] - uC[0]) * vec_y - (
                    VertexC[t, 1] - uC[1]
                ) * vec_x
                if is_root_sign_pos:
                    if w_cross <= 0:
                        blocked[r, t] = True
                else:
                    if w_cross >= 0:
                        blocked[r, t] = True
        else:
            for t in range(T):
                ar = angle_rank__[t, r]
                if ar >= uR and ar <= vR:
                    continue
                w_cross = (VertexC[t, 0] - uC[0]) * vec_y - (
                    VertexC[t, 1] - uC[1]
                ) * vec_x
                if is_root_sign_pos:
                    if w_cross <= 0:
                        blocked[r, t] = True
                else:
                    if w_cross >= 0:
                        blocked[r, t] = True
    return blocked


def add_link_blockmap(A: nx.Graph):
    """Add edge attributes ``'blocked__'``.

    Edges' attribute ``'blocked__'`` are R-long list of T-long bitarray maps.

    If an edge's ``blocked__[r][t] == 1``, then this edge crosses the line-of-sight t-r.

    Changes ``A`` in place. ``A`` should have no feeder edges.

    Note:
      * this function neglects borders and contours.
      * the space taken scales with ``R × T × num_edges(A)``
    """
    VertexC = A.graph['VertexC']
    R, T = A.graph['R'], A.graph['T']
    angle__, angle_rank__, dups_from_root_rank__ = angle_helpers(
        A, include_borders=False
    )
    # TODO: check if dups_from_root_rank__ has a role here
    A.graph['angle__'] = angle__
    A.graph['angle_rank__'] = angle_rank__
    A.graph['dups_from_root_rank__'] = dups_from_root_rank__
    for u, v, edgeD in A.edges(data=True):
        blocked = _blockmap_inner(u, v, angle__, angle_rank__, VertexC, R, T)
        blocked__ = []
        for r in range(R):
            ba = bitarray()
            ba.frombytes(np.packbits(blocked[r]).tobytes())
            del ba[T:]
            blocked__.append(ba)
        edgeD['blocked__'] = blocked__


def add_link_cosines(A: nx.Graph):
    """Add cosine of the angle wrt each root to all links of A as attribute ``'cos_'``.

    Changes A in-place. The cosine is of the acute angle between the link line and the
    line that contains the mid-point of the link and the root (for each root).
    """
    R = A.graph['R']
    VertexC = A.graph['VertexC']
    RootC = VertexC[-R:]

    edge_ = np.fromiter(
        chain.from_iterable(A.edges()),
        dtype=int,
        count=2 * A.number_of_edges(),
    ).reshape((-1, 2))
    edgeC = VertexC[edge_]
    uC = edgeC[:, 0, :]
    vC = edgeC[:, 1, :]
    edge_vec_ = vC - uC
    edge_len_ = np.hypot(*edge_vec_.T)
    mid_edge_ = 0.5 * (uC + vC)
    mid_vec_ = mid_edge_[:, None, :] - RootC
    mid_len_ = np.hypot(mid_vec_[..., 0], mid_vec_[..., 1])
    cos__ = abs(np.vecdot(edge_vec_[:, None, :], mid_vec_)) / (
        edge_len_[:, None] * mid_len_
    )
    nx.set_edge_attributes(
        A,
        {(edge[0], edge[1]): cos_.tolist() for edge, cos_ in zip(edge_, cos__)},
        name='cos_',
    )


def scaffolded(G: nx.Graph, P: nx.PlanarEmbedding) -> nx.Graph:
    """Create a new graph merging G and P.

    Useful for visualizing the funnels explored by :class:`.pathfinding.PathFinder`.
    ``G`` must have been created using ``P``.

    Args:
      G: network graph for location
      P: planar embedding of location

    Returns:
      Merged graph (pass to :func:`.plotting.gplot` or :func:`.svg.svgplot`).
    """
    scaff = P.to_undirected()
    scaff.graph.update(G.graph)
    for attr in 'fnT C'.split():
        if attr in scaff.graph:
            del scaff.graph[attr]
    R, T, B, C, D = (G.graph.get(k, 0) for k in 'R T B C D'.split())
    nx.set_edge_attributes(scaff, 'scaffold', name='kind')
    constraints = P.graph.get('constraint_edges', [])
    for edge in constraints:
        scaff.edges[edge]['kind'] = 'constraint'
    for n, d in scaff.nodes(data=True):
        if n not in G.nodes:
            continue
        d.update(G.nodes[n])
    if C > 0 or D > 0:
        fnT_G = G.graph['fnT']
    else:
        fnT_G = np.arange(R + T + B + C + D)
        fnT_G[-R:] = range(-R, 0)
    for u, v in G.edges:
        st = fnT_G[u], fnT_G[v]
        if st in scaff.edges and 'kind' in scaff.edges[st]:
            del scaff.edges[st]['kind']
    # a 'shortened_contours' entry collapses a fence onto fewer clones than
    # mesh hops (sharing clones across contours), so the loop above only
    # catches its two collapsed endpoints; walk the stored full midpath too.
    for (s, t), (midpath, _) in G.graph.get('shortened_contours', {}).items():
        for a, b in zip((s, *midpath), (*midpath, t)):
            st = (a, b) if a < b else (b, a)
            if st in scaff.edges and 'kind' in scaff.edges[st]:
                del scaff.edges[st]['kind']
    VertexC = G.graph['VertexC']
    supertriangleC = P.graph['supertriangleC']
    if G.graph.get('is_normalized'):
        supertriangleC = G.graph['norm_scale'] * (
            supertriangleC - G.graph['norm_offset']
        )
    VertexC = np.vstack((VertexC[:-R], supertriangleC, VertexC[-R:]))
    # scaff's own nodes are G's primes + P's supertriangle + roots (no
    # clones: G's clone ids alias P's supertriangle ids, so clones never
    # get added as scaff nodes above). This fnT must address that node
    # space (not G's, used only for the clone->prime remap loop above).
    fnT = np.arange(T + B + 3 + R)
    fnT[-R:] = range(-R, 0)
    scaff.graph.update(VertexC=VertexC, fnT=fnT)
    if 'capacity' in scaff.graph:
        # hack to prevent `gplot()` from showing infobox
        del scaff.graph['capacity']
    return scaff
