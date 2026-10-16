# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Conversions that cross graph kinds.

Every function here takes one kind and returns another: a site to a location
graph ``L``, ``L`` and ``A`` to a topology ``S``, ``S`` to a routeset ``G``, and ``S``
to or from its bit and terse encodings. For a variant of the same kind, see
:mod:`optiwindnet.transforming`.
"""

import logging
import math
from collections.abc import Iterator
from itertools import chain, pairwise

import networkx as nx
import numpy as np
from bitarray import bitarray, frozenbitarray

from .identity import _CANONICAL_TERMINAL_LINKS
from .loads import _orient_tree, calcload, nonunit_inflow, validate_terminal_power
from .terse import TerseLinks
from .types import Topology

__all__ = (
    'G_from_S', 'L_from_G', 'L_from_site', 'S_from_G', 'S_from_linkbits',
    'S_from_terse_links', 'linkbits_from_S', 'terse_links_from_S',
)  # fmt: skip

_lggr = logging.getLogger(__name__)


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


def S_from_linkbits(
    linkbits: bitarray, A: nx.Graph, *, topology: Topology
) -> nx.DiGraph:
    """Decode canonical ``linkbits`` into an undetoured topology ``S``.

    Uses the same terminal-link and feeder ordering as :func:`linkbits_from_S`.
    ``A`` must provide its ``'_canonical_terminal_links'`` graph attribute.
    The bit count must match ``A``'s available-links set; a mismatch raises
    ``ValueError``. Equal bit counts are not proof of a common origin: compare
    ``A``'s ``'_linkset_id'`` with the producer's to confirm that the bits
    and the graph describe the same available-links set.

    ``topology`` determines the electrical orientation and ring-arm split.
    The result is a directed solution with unit terminal inflow and calculated
    loads. A same-root singleton ring has opposite feeder arcs even though
    its geometric link occupies one bit. Capacity and producer metadata are
    supplied separately by the caller.
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
    if topology is Topology.RINGED:
        from .loads import split_rings_and_calc_loads

        return split_rings_and_calc_loads(S, A)
    S = _orient_tree(S)
    S.graph['topology'] = topology
    calcload(S)
    return S


_essential_graph_attrs = (
    # required
    'R', 'T', 'B', 'VertexC', 'name', 'handle', 'border',
    # optional
    'obstacles', 'landscape_angle', 'norm_scale', 'norm_offset',
    'power_unit', 'powers_set',
)  # fmt: skip


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


def G_from_S(S: nx.DiGraph, A: nx.Graph) -> nx.DiGraph:
    """Create G from S and A.

    Graph ``S`` contains the topology of a routeset network (nodes only, no
    contours or detours). ``S`` must have been created from the available links
    in ``A``, whose contour information is used to obtain a routeset ``G``
    (possibly with contours, but not with detours – use PathFinder afterward).

    Warns:
      UserWarning: if the topology contains links absent from the mesh.
    """
    from .crossings import edge_conflicts

    R, T, B = (A.graph[k] for k in 'RTB')
    VertexC, diagonals = A.graph['VertexC'], A.graph['diagonals']
    G = nx.create_empty_copy(S)
    for key in (
        'border',
        'obstacles',
        'name',
        'handle',
        'landscape_angle',
        'norm_scale',
        'norm_offset',
        'is_normalized',
        'power_unit',
        'powers_set',
    ):
        if key in A.graph:
            G.graph[key] = A.graph[key]
    # the quantization of the solve, when recorded on S, takes precedence over
    # the power the site declares (absent from S if the capacity was in inflow)
    if 'power_per_inflow' in A.graph:
        G.graph.setdefault('power_per_inflow', A.graph['power_per_inflow'])
    if 'powers_set' in A.graph:
        nx.set_node_attributes(G, {t: A.nodes[t]['power'] for t in range(T)}, 'power')
    validate_terminal_power(G)
    stunts = A.graph.get('stunts_primes', ())
    G.graph['B'] = B - len(stunts)
    G.graph['VertexC'] = np.vstack((VertexC[: -R - 3 - len(stunts)], VertexC[-R:]))
    for n in G:
        G.nodes[n]['kind'] = 'oss' if n < 0 else 'wtg'
        if n in A and 'label' in A.nodes[n]:
            G.nodes[n]['label'] = A.nodes[n]['label']

    selected = {tuple(sorted(edge)) for edge in S.edges}
    clone2prime, tentative, rogue = [], set(), set()
    shortened = {}
    num_diagonals = 0
    for source, sink, data in S.edges(data=True):
        s, t = sorted((source, sink))
        feeder = s < 0
        key = (s, t)
        attrs = data.copy()
        if not A.has_edge(s, t):
            attrs.update(
                length=float(A.graph['d2roots'][t, s])
                if feeder
                else math.dist(VertexC[s], VertexC[t]),
                kind='tentative' if feeder else 'rogue',
            )
            G.add_edge(source, sink, **attrs)
            (tentative if feeder else rogue).add(key)
            continue
        available = A[s][t]
        num_diagonals += not feeder and available['kind'] == 'extended'
        midpath = available.get('midpath')
        if feeder and (
            midpath is not None
            or any(edge in selected for edge in edge_conflicts(s, t, diagonals))
        ):
            G.add_edge(
                source,
                sink,
                **{
                    **attrs,
                    'length': available['length'],
                    'kind': 'tentative',
                },
            )
            tentative.add(key)
            continue
        if midpath is None:
            G.add_edge(source, sink, **{**attrs, 'length': available['length']})
            continue
        shortcuts = available.get('shortcuts', ())
        path_primes = (
            []
            if len(shortcuts) == len(midpath)
            else [n for n in midpath if n not in shortcuts]
        )
        if shortcuts:
            shortened[key] = midpath, path_primes
        if not path_primes:
            G.add_edge(source, sink, **{**attrs, 'length': available['length']})
            continue
        if source != s:
            path_primes = path_primes[::-1]
        path = [source]
        subtree = S.nodes[source if source >= 0 else sink]['subtree']
        for prime in path_primes:
            clone = T + B + len(clone2prime)
            clone2prime.append(prime)
            G.add_node(clone, kind='contour', load=data['load'], subtree=subtree)
            path.append(clone)
        path.append(sink)
        primes = [source, *path_primes, sink]
        lengths = np.linalg.norm(np.diff(VertexC[primes], axis=0), axis=1)
        for (u, v), length in zip(pairwise(path), lengths):
            G.add_edge(
                u,
                v,
                **{
                    **attrs,
                    'length': float(length),
                    'kind': 'contour',
                    'A_edge': key,
                },
            )
    if clone2prime:
        stunt2prime = {T + B - len(stunts) + i: n for i, n in enumerate(stunts)}
        fnT = np.arange(T + B + len(clone2prime) + R)
        fnT[T + B : -R] = [stunt2prime.get(n, n) for n in clone2prime]
        fnT[-R:] = range(-R, 0)
        G.graph.update(C=len(clone2prime), fnT=fnT)
    if tentative:
        G.graph['tentative'] = sorted(tentative)
    if rogue:
        G.graph['rogue'] = sorted(rogue)
        _lggr.warning(
            'Solution uses rogue links absent from the mesh: %s.', sorted(rogue)
        )
    if shortened:
        G.graph['shortened_contours'] = shortened
    G.graph['num_diagonals'] = num_diagonals
    return G


def _chains(G: nx.Graph) -> Iterator[tuple[int, int, list[int]]]:
    """Yield ``(s, t, path)`` for each chain of non-real nodes of routeset ``G``.

    Real nodes are terminals and roots (``n < T``); border vertices, contour and
    detour clones are not. A chain starts at a real node ``s``, runs through
    non-real nodes only and ends at the next real node ``t``: it is the route
    of one link bent around obstacles or detoured. ``path`` lists its nodes
    from ``s`` to ``t``. Each chain is yielded once
    in electrical flow order by following successors. Zero-load cable chains
    retain their deterministic arc orientation.

    Raises:
      ValueError: a chain branches, does not reach a real node or returns to
        its start.
    """
    T = G.graph['T']
    for source in sorted(n for n in G if n < T):
        for node in G[source]:
            if node < T:
                continue
            path = [source, node]
            while node >= T:
                onward = list(G[node])
                if len(onward) != 1 or onward[0] in path:
                    raise ValueError(f'route from {source} is not a directed chain')
                node = onward[0]
                path.append(node)
            yield source, node, path


def S_from_G(G: nx.DiGraph) -> nx.DiGraph:
    """Get ``G``'s topology (contours, detours, lengths and coords are dropped).

    Terminal ``'inflow'`` attributes are preserved so that recalculating loads
    on ``S`` uses the same terminal contributions as on ``G``, as are the graph
    attributes that record the quantization of the solve (``'power_per_inflow'``,
    ``'capacity_nominal'`` and ``'power_rtol'``). Declared nominal power belongs
    to the site and routed graphs, not the topology.

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
    S = nx.DiGraph(T=T, R=R, capacity=capacity)

    def is_real(n: int) -> bool:
        "Only roots and terminals survive in S (border vertices and clones do not)."
        return n < T

    for r in range(-R, 0):
        S.add_node(r, kind='oss', **({'load': G.nodes[r]['load']} if has_loads else {}))
    for t in sorted(n for n in G if 0 <= n < T):
        nodeD = G.nodes[t]
        # Preserve inflow for subsequent load calculations.
        inflow_attrs = {'inflow': nodeD['inflow']} if 'inflow' in nodeD else {}
        if has_loads:
            S.add_node(
                t,
                kind='wtg',
                load=nodeD['load'],
                subtree=nodeD['subtree'],
                **inflow_attrs,
            )
        else:
            S.add_node(t, kind='wtg', **inflow_attrs)

    S.add_edges_from(
        (u, v, {'load': data['load']} if has_loads else {})
        for u, v, data in G.edges(data=True)
        if is_real(u) and is_real(v)
    )
    for source, target, path in _chains(G):
        S.add_edge(
            source,
            target,
            **({'load': G[source][path[1]]['load']} if has_loads else {}),
        )

    creator = G.graph.get('creator')
    if creator is not None:
        S.graph['creator'] = creator
    method_options = G.graph.get('method_options')
    if method_options is not None:
        S.graph['method_options'] = method_options
    S.graph['topology'] = G.graph['topology']
    for k in ('power_per_inflow', 'capacity_nominal', 'power_rtol'):
        if k in G.graph:
            S.graph[k] = G.graph[k]
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
    Of the turbine power, only the declaration is kept (terminal ``'power'``
    under ``'powers_set'``, otherwise ``'power_per_inflow'`` and the inflow
    other than one), not the quantization of the solve.

    Args:
      G: routeset graph to extract site data from.

    Returns:
      Site graph (no edges) with lean attributes.
    """
    R, T = (G.graph[k] for k in 'RT')
    L = nx.Graph(**{k: G.graph[k] for k in _essential_graph_attrs if k in G.graph})
    if 'power_per_inflow' in G.graph and 'powers_set' not in G.graph:
        # uniform power: the power of the single inflow is the declared power
        L.graph['power_per_inflow'] = G.graph['power_per_inflow']

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
    if 'powers_set' in G.graph:
        nx.set_node_attributes(L, {t: G.nodes[t]['power'] for t in range(T)}, 'power')
    else:
        nx.set_node_attributes(L, nonunit_inflow(G), 'inflow')
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
