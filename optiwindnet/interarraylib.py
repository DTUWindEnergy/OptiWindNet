# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import math
from collections.abc import Iterator
from itertools import chain, pairwise

import networkx as nx
import numba as nb
import numpy as np
from bitarray import bitarray

from .geometric import angle_helpers, rotate
from .loads import rings_from_S
from .types import Topology

_lggr = logging.getLogger(__name__)
debug, warn, error = _lggr.debug, _lggr.warning, _lggr.error

__all__ = (
    'add_link_blockmap', 'add_link_cosines', 'add_terminal_closest_root',
    'assign_cables', 'count_diagonals', 'describe_G', 'directed_links',
    'make_remap', 'pathdist', 'scaffolded', 'update_lengths',
)  # fmt: skip


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
            # ring zero-load link ('split'): a real cable with no current — assign the
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


def _format_length(length: float, significant_digits: int = 5) -> str:
    """Format ``length`` with '_' as thousands separator.

    ``significant_digits`` is a minimum, enforced through fraction digits only.
    """
    intdigits = int(np.floor(np.log10(length))) + 1
    fracdigits = max(0, significant_digits - intdigits)
    return f'{{:_.{fracdigits}f}}'.format(round(length, fracdigits))


def describe_G(G: nx.Graph, significant_digits: int = 5) -> list[str]:
    """Create a 3-4 line summary of G's properties.

    ``significant_digits`` applies only to total length and is enforced only when the
    integer part has fewer significant digits than ``significant_digits``.

    Args:
      G: routeset instance
      significant_digits: minimum number of significant digits used for total length

    Returns:
      Text lines with capacity and T, excess feeders and feeders per root, total
      length and total cost.
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
        desc.append(
            'Σλ = '
            + _format_length(length, significant_digits).replace('_', '\u202f')
            + '\u00a0m'
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


def directed_links(S: nx.Graph) -> Iterator[tuple[int, int, int]]:
    """Yield ``(source, sink, flow)`` for every link of ``S``.

    Forest topologies read each link's orientation off its ``'reverse'`` flag
    (see the note above :func:`bfs_subtree_loads`), so ``flow`` is just the
    link's load.

    A RINGED ``S`` -- as declared by ``S.graph['topology']`` -- stores each ring
    split into two arms at a zero-load link, which is not how a flow
    formulation sees it: there a ring is one directed chain of its ``n``
    terminals, fed by a flowless closing feeder at one end and draining through
    a feeder carrying the whole ring at the other. Such rings are *radialized*
    into that chain here (walking across the zero-load link with
    :func:`rings_from_S`), so the zero-load link becomes an ordinary
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
    if S.graph['topology'] is not Topology.RINGED:
        for u, v, edgeD in S.edges(data=True):
            source, sink = (u, v) if ((u < v) == edgeD['reverse']) else (v, u)
            yield source, sink, edgeD['load']
        return
    for root, chain_ in rings_from_S(S):
        head_root, tail_root = root
        n = len(chain_)
        # the ring drains through chain_[0], whose feeder carries all of it, and
        # closes on the far feeder. A lone terminal needs no special case: it is
        # both head and tail, and the chain below is empty.
        yield chain_[0], head_root, n
        yield tail_root, chain_[-1], 0
        for j in range(n - 1, 0, -1):
            yield chain_[j], chain_[j - 1], n - j


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
    for attr in ['fnT', 'C']:
        if attr in scaff.graph:
            del scaff.graph[attr]
    R, T, B, C, D = (G.graph.get(k, 0) for k in ['R', 'T', 'B', 'C', 'D'])
    # a scalar `values` is applied to every edge; the stubs only cover mappings
    # pyrefly: ignore[no-matching-overload]
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
