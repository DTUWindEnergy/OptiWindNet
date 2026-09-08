# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Conversions that cross graph kinds.

Every function here takes one kind and returns another: a site to a location
graph ``L``, ``L`` and ``A`` to a topology ``S``, ``S`` to a routeset ``G``, and ``S``
to or from its bit and terse encodings. For a variant of the same kind, see
:mod:`optiwindnet.transforming`.
"""

from itertools import chain

import networkx as nx
import numpy as np
from bitarray import bitarray, frozenbitarray

from .identity import _CANONICAL_TERMINAL_LINKS
from .loads import calcload
from .terse import TerseLinks
from .types import Topology

__all__ = (
    'G_from_S', 'L_from_G', 'L_from_site', 'S_from_G', 'S_from_linkbits',
    'S_from_terse_links', 'linkbits_from_S', 'terse_links_from_S',
)  # fmt: skip


def linkbits_from_S(A: nx.Graph, S: nx.Graph) -> frozenbitarray:
    """Encode topology ``S`` over ``A``'s canonical undirected available-links set.

    Terminal-terminal positions follow the lexicographic edge order cached by
    :func:`~optiwindnet.mesh.make_planar_embedding`. The ``R * T`` feeder
    positions follow in terminal-major order, with roots from ``-R`` to ``-1``.
    ``A`` must provide its ``'_canonical_terminal_links'`` graph attribute.
    ``S`` must be an undetoured topology containing only terminals and roots.
    """
    R, T = (A.graph[key] for key in 'RT')
    links = A.graph[_CANONICAL_TERMINAL_LINKS]
    feeder_start = len(links)

    ends = np.fromiter(chain.from_iterable(S.edges), dtype=np.int64).reshape(-1, 2)
    lo, hi = ends.min(axis=1), ends.max(axis=1)
    is_feeder = lo < 0
    offender = is_feeder & ((lo < -R) | (hi < 0) | (hi >= T))
    if offender.any():
        u, v = ends[offender][0].tolist()
        raise ValueError(f'S contains a non-canonical feeder link: {(u, v)}')

    # locate terminal links by their lexicographic key in the canonical order
    wanted = lo[~is_feeder] * T + hi[~is_feeder]
    if wanted.size:
        keys = links[:, 0].astype(np.int64) * T + links[:, 1]
        position = np.searchsorted(keys, wanted)
        found = position < feeder_start
        found[found] = keys[position[found]] == wanted[found]
        if not found.all():
            u, v = ends[~is_feeder][~found][0].tolist()
            edge = (u, v) if u < v else (v, u)
            raise ValueError(f'S contains a terminal link absent from A: {edge}')
    else:
        position = wanted
    positions = np.concatenate(
        (position, feeder_start + hi[is_feeder] * R + lo[is_feeder] + R)
    )

    nbits = feeder_start + R * T
    flat = np.zeros(nbits, dtype=np.uint8)
    flat[positions] = 1
    packed = bitarray(buffer=np.packbits(flat).tobytes(), endian='big')
    return frozenbitarray(packed[:nbits])


def S_from_linkbits(linkbits: bitarray, A: nx.Graph) -> nx.Graph:
    """Decode canonical ``linkbits`` into an undetoured topology ``S``.

    Uses the same terminal-link and feeder ordering as :func:`linkbits_from_S`.
    ``A`` must provide its ``'_canonical_terminal_links'`` graph attribute.
    The bit count must match ``A``'s available-links set; a mismatch raises
    ``ValueError``. Equal bit counts are not proof of a common origin: compare
    ``A``'s ``'_linkset_id'`` with the producer's to confirm that the bits
    and the graph describe the same available-links set.

    The result contains all ``T`` terminals and ``R`` roots, including isolated
    nodes, without node attributes. Only connectivity is recovered:
    loads, capacity, topology type and other solution metadata are not encoded.
    No feasibility validation or load calculation is performed.
    """
    R, T = (A.graph[key] for key in 'RT')
    links = A.graph[_CANONICAL_TERMINAL_LINKS]
    feeder_start = len(links)
    nbits = feeder_start + R * T
    if len(linkbits) != nbits:
        raise ValueError(f'Expected {nbits} link bits for A, got {len(linkbits)}')

    S = nx.Graph(R=R, T=T)
    S.add_nodes_from(range(-R, T))
    for position in linkbits.search(bitarray('1')):
        if position < feeder_start:
            u, v = map(int, links[position])
        else:
            u, root_index = divmod(position - feeder_start, R)
            v = root_index - R
        S.add_edge(u, v)
    return S


_essential_graph_attrs = (
    # required
    'R', 'T', 'B', 'VertexC', 'name', 'handle', 'border',
    # optional
    'obstacles', 'landscape_angle', 'norm_scale', 'norm_offset',
)  # fmt: skip


def _rings_from_S(S: nx.Graph) -> list[tuple[tuple[int, int], list[int]]]:
    """Recover ordered ring terminal sequences from a RINGED solution graph.

    Each ring is returned as ``((r1, r2), [t1, ..., tn])`` with ``t1`` and ``tn``
    the feeder-connected terminals, obtained by walking the terminal adjacency
    from the head subroot to the tail one; ``r1`` feeds ``t1`` and ``r2`` feeds
    ``tn``. The ring bridges two substations when ``r1 != r2``.

    Feeders are identified by having exactly one negative (root) endpoint; a ring
    with a single terminal (``n == 1``) has both feeders on that terminal.
    """
    R = S.graph['R']
    subroots = {r: [t for t in S[r] if t >= 0] if r in S else [] for r in range(-R, 0)}
    rings: list[tuple[tuple[int, int], list[int]]] = []
    # `subroots` is consumed as the walk goes: each feeder is claimed once
    for r in range(-R, 0):
        while subroots[r]:
            t1 = subroots[r].pop(0)
            chain_ = [t1]
            prev, curr = None, t1
            while True:
                nxts = [x for x in S[curr] if x >= 0 and x != prev]
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
                    (rc for rc in S[tn] if rc < 0 and rc != r and tn in subroots[rc]),
                    None,
                )
            if r2 is None:
                r2 = r
            else:
                subroots[r2].remove(tn)
            rings.append(((r, r2), chain_))
    return rings


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
    contours or detours). ``S`` must have been created from the available links
    in ``A``, whose contour information is used to obtain a routeset ``G``
    (possibly with contours, but not with detours – use PathFinder afterward).
    """
    R, T, B = (A.graph[k] for k in 'RTB')
    VertexC, d2roots, diagonals = (
        A.graph[k] for k in ('VertexC', 'd2roots', 'diagonals')
    )
    G = nx.create_empty_copy(S)
    carry_over = (
        'B', 'border', 'obstacles', 'name', 'handle',
        'landscape_angle', 'norm_scale', 'norm_offset', 'is_normalized',
    )  # fmt: skip
    for k in carry_over:
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
    # a scalar `values` is applied to every node; the stubs only cover mappings
    nx.set_node_attributes(G, 'wtg', 'kind')  # pyrefly: ignore[no-matching-overload]
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

        # Split edges are ring zero-load links (load=0: no current flows). The
        # link keeps its geometry kind — it may follow a contour like any edge.
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
    """Get ``G``'s topology (contours, detours, lengths and coords are dropped).

    If using ``S`` to warm-start a MILP model, call after :func:`S_from_G`:

    - :func:`as_hooked_to_nearest` for ``Topology.BRANCHED``;
    - :func:`as_hooked_to_head` for ``Topology.RADIAL``.

    This makes a radial ``S`` feasible and avoids a trivially suboptimal
    branched ``S``. For RINGED routesets, cycle-closing links are retained, so
    ``S`` preserves the ring partition of ``G``.

    Args:
      G: feasible routed solution with a tree, path or ring topology.

    Returns:
      Topology of ``G``.

    Raises:
      ValueError: a routed link does not form a chain between two real nodes.
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
            remaining = G.number_of_nodes()
            while not is_real(node):
                fwd = [x for x in G[node] if x != prev]
                if len(fwd) != 1:
                    raise ValueError(
                        f'route from {s} is not a chain at {node}: '
                        f'expected one forward link, got {len(fwd)}'
                    )
                prev, node = node, fwd[0]
                remaining -= 1
                if remaining < 0:
                    raise ValueError(f'route from {s} does not reach another real node')
            if node == s:
                raise ValueError(f'route from {s} returns to itself')
            if S.has_edge(s, node):
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
        S.graph['max_load'] = G.graph['max_load']
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


def S_from_terse_links(terse_links, R=None, T=None, topology=None, **kwargs):
    """Create topology ``S`` from a self-describing or legacy encoding."""
    if isinstance(terse_links, TerseLinks):
        encoding = terse_links
        if topology is not None and Topology(topology) is not encoding.topology:
            raise ValueError('topology disagrees with the TerseLinks value')
        if R is not None and R != encoding.R:
            raise ValueError('R disagrees with the TerseLinks value')
        if T is not None and T != encoding.T:
            raise ValueError('T disagrees with the TerseLinks value')
    else:
        array = np.asarray(terse_links)
        if array.ndim != 1:
            raise ValueError('terse links must be a 1D array')
        encoding = TerseLinks.from_array(array.tolist(), topology=topology, R=R, T=T)
    return encoding.to_topology(**kwargs)


def terse_links_from_S(S):
    """Return the self-describing compact representation of topology ``S``."""
    return TerseLinks.from_topology(S)
