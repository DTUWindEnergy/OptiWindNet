# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import base64
import io
import json
import math
from collections.abc import Sequence
from functools import partial
from hashlib import sha256
from itertools import chain, pairwise
from socket import getfqdn, gethostname
from typing import Any, Mapping

import networkx as nx
import numpy as np

from ..interarraylib import calcload, compress_ring_routes, parse_ring_routes
from ..utils import make_handle
from .model import (
    Machine,
    Method,
    NodeSet,
    RouteSet,
)

__all__ = ()

PackType = Mapping[str, Any]

# Set of not-to-store keys commonly found in G routesets (they are either
# already stored in database fields or are cheap to regenerate or too big.
_misc_not = {
    'VertexC',
    'anglesYhp',
    'anglesXhp',
    'anglesRank',
    'angles',
    'd2rootsRank',
    'd2roots',
    'name',
    'boundary',
    'capacity',
    'B',
    'runtime',
    'runtime_unit',
    'edges_fun',
    'D',
    'DetourC',
    'fnT',
    'landscape_angle',
    'Root',
    'creation_options',
    'G_nodeset',
    'T',
    'non_A_gates',
    'funfile',
    'funhash',
    'funname',
    'diagonals',
    'planar',
    'has_loads',
    'R',
    'Subtree',
    'handle',
    'non_A_edges',
    'max_load',
    'fun_fingerprint',
    'hull',
    'solver_log',
    'length_mismatch_on_db_read',
    'gnT',
    'C',
    'border',
    'obstacles',
    'num_diagonals',
    'crossings_map',
    'tentative',
    'method_options',
    'is_normalized',
    'norm_scale',
    'norm_offset',
    'detextra',
    'rogue',
    'clone2prime',
    'valid',
    'path_in_P',
    'shortened_contours',
    'nonAedges',
    'method',
    'num_stunts',
    'crossings',
    'creator',
    'inter_terminal_clearance_min',
    'inter_terminal_clearance_safe',
    'stunts_primes',
}


def L_from_nodeset(nodeset: NodeSet, handle: str | None = None) -> nx.Graph:
    """Translate a NodeSet database entry to a location graph.

    Args:
      nodeset: an entry from the database NodeSet table.

    Returns:
      Graph L containing the positions and location metadata.
    """
    T = nodeset.T
    R = nodeset.R
    B = nodeset.B
    border = np.array(nodeset.constraint_vertices[: nodeset.constraint_groups[0]])
    name = nodeset.name
    if handle is None:
        handle = make_handle(name if name[0] != '!' else name[1 : name.index('!', 1)])
    L = nx.Graph(
        R=R,
        T=T,
        B=B,
        name=name,
        handle=handle,
        VertexC=np.lib.format.read_array(io.BytesIO(nodeset.VertexC)),
        landscape_angle=nodeset.landscape_angle,
    )
    if len(border) > 0:
        L.graph['border'] = border
    if len(nodeset.constraint_groups) > 1:
        obstacle_idx = np.cumsum(np.array(nodeset.constraint_groups))
        L.graph.update(
            obstacles=[
                np.array(nodeset.constraint_vertices[a:b])
                for a, b in pairwise(obstacle_idx)
            ]
        )
    L.add_nodes_from(((n, {'kind': 'wtg'}) for n in range(T)))
    L.add_nodes_from(((r, {'kind': 'oss'}) for r in range(-R, 0)))
    return L


def G_from_routeset(routeset: RouteSet) -> nx.Graph:
    """Translate a RouteSet database entry to a routeset graph.

    Args:
      routeset: an entry from the database RouteSet table.

    Returns:
      Graph G containing the routeset.
    """
    nodeset = routeset.nodes
    G = L_from_nodeset(nodeset)
    misc = routeset.misc if routeset.misc is not None else {}
    G.graph.update(
        C=routeset.C,
        D=routeset.D,
        handle=routeset.handle,
        capacity=routeset.capacity,
        creator=routeset.creator,
        method=dict(
            solver_name=routeset.method.solver_name,
            timestamp=routeset.method.timestamp,
            funname=routeset.method.funname,
            funfile=routeset.method.funfile,
            funhash=routeset.method.funhash,
        ),
        runtime=routeset.runtime,
        method_options=routeset.method.options,
        **misc,
    )

    if routeset.detextra is not None:
        G.graph['detextra'] = routeset.detextra

    untersify_to_G(G, terse=routeset.edges, clone2prime=routeset.clone2prime)
    calc_length = G.size(weight='length')
    if abs(calc_length / routeset.length - 1) > 1e-5:
        G.graph['length_mismatch_on_db_read'] = calc_length - routeset.length
    if routeset.rogue:
        for u, v in zip(routeset.rogue[::2], routeset.rogue[1::2]):
            G[u][v]['kind'] = 'rogue'
    if routeset.tentative:
        for r, n in zip(routeset.tentative[::2], routeset.tentative[1::2]):
            G[r][n]['kind'] = 'tentative'
    return G


def packnodes(G: nx.Graph) -> PackType:
    R, T, B = (G.graph[k] for k in 'RTB')
    VertexC = G.graph['VertexC']

    VertexC_npy_io = io.BytesIO()
    np.lib.format.write_array(VertexC_npy_io, VertexC, version=(3, 0))
    VertexC_npy = VertexC_npy_io.getvalue()
    digest = sha256(VertexC_npy).digest()

    if G.name[0] == '!':
        name = G.name + base64.b64encode(digest).decode('ascii')
    else:
        name = G.name
    constraint_vertices = list(
        chain((G.graph.get('border', ()),), G.graph.get('obstacles', ()))
    )
    pack = dict(
        T=T,
        R=R,
        B=B,
        name=name,
        VertexC=VertexC_npy,
        constraint_groups=[p.shape[0] for p in constraint_vertices],
        constraint_vertices=np.concatenate(
            constraint_vertices, dtype=int, casting='unsafe'
        ).tolist(),
        landscape_angle=G.graph.get('landscape_angle', 0.0),
        digest=digest,
    )
    return pack


def packmethod(method_options: dict) -> PackType:
    options = {
        k: method_options[k]
        for k in sorted(method_options)
        if k not in ('fun_fingerprint', 'solver_name')
    }
    ffprint = method_options['fun_fingerprint']
    digest = sha256(ffprint['funhash'] + json.dumps(options).encode()).digest()
    pack = dict(
        digest=digest,
        solver_name=method_options['solver_name'],
        options=options,
        **ffprint,
    )
    return pack


def add_if_absent(entity: type, pack: PackType) -> bytes:
    digest = pack['digest']
    if not entity.select().where(entity.digest == digest).exists():
        entity.create(**pack)
    return digest


def method_from_G(G: nx.Graph) -> bytes:
    """
    Returns:
        Primary key of the entry.
    """
    pack = packmethod(G.graph['method_options'])
    return add_if_absent(Method, pack)


def nodeset_from_G(G: nx.Graph) -> bytes:
    """Returns primary key of the entry."""
    pack = packnodes(G)
    return add_if_absent(NodeSet, pack)


def _walk_ring_in_G(G: nx.Graph, root: int, subroot: int) -> tuple[list[int], int]:
    """Walk one ring's cycle in ``G`` from a subroot until a root is reached.

    Every non-root node of a canonical ring has degree 2 (both terminals and the
    contour/detour clones spliced into its edges), so the walk is forced. Returns
    the ordered list of walked nodes (roots excluded) and the root it closes on
    (the same root for a single-terminal stub).
    """
    walk = [subroot]
    prev, curr = root, subroot
    while True:
        nbrs = [x for x in G[curr] if x != prev]
        if not nbrs:  # single-terminal stub (degree-1)
            return walk, root
        if len(nbrs) != 1:
            raise ValueError(f'ring node {curr} is not degree-2 (neighbours {nbrs})')
        (nxt,) = nbrs
        if nxt < 0:  # reached a root: the ring closes here
            return walk, nxt
        walk.append(nxt)
        prev, curr = curr, nxt


def _ring_routes_from_G(
    G: nx.Graph, R: int, T: int
) -> list[tuple[int, list[int], int]]:
    """Recover the ring routes ``(open_root, walk, close_root)`` of a ringed ``G``.

    ``walk`` lists a ring's nodes (terminals and clones) between its two
    subroots. Each walk is oriented so its open point (``load=0``) falls on the
    load midpoint that :func:`untersify_to_G` re-derives from the terminal count,
    so the open point round-trips without being stored. A ring may open and close
    on different roots (a routed ring can bridge two substations).
    """
    split_pairs = {
        frozenset((u, v)) for u, v, d in G.edges(data=True) if d.get('load') == 0
    }
    routes: list[tuple[int, list[int], int]] = []
    visited: set[tuple[int, int]] = set()
    for root in range(-R, 0):
        for subroot in sorted(G[root]):
            if (root, subroot) in visited:
                continue
            walk, close = _walk_ring_in_G(G, root, subroot)
            visited.add((root, walk[0]))
            visited.add((close, walk[-1]))
            open_ = root
            n = sum(1 for x in walk if x < T)
            if n > 1:
                seen, a = 0, None
                for i in range(len(walk) - 1):
                    if (
                        walk[i] < T
                        and walk[i + 1] < T
                        and frozenset((walk[i], walk[i + 1])) in split_pairs
                    ):
                        a = seen
                        break
                    if walk[i] < T:
                        seen += 1
                if a is None:
                    raise ValueError('ring open point not found among its terminals')
                if a != math.ceil(n / 2) - 1:
                    walk = walk[::-1]
                    open_, close = close, open_
            routes.append((open_, walk, close))
    return routes


def terse_pack_from_G(G: nx.Graph) -> PackType:
    """Convert ``G``'s edges to a format suitable for storing in the database.

    A *forest* routeset (radial/branched) is stored positionally: ``G`` is
    undirected, but the edge attribute ``'reverse'`` and node numbering encode
    the power-flow direction, so ``(i, edges[i])`` is a directed link and the
    array has one entry per non-root node (``T + C + D``).

    A *ringed* routeset needs two feeders per ring, which a single parent per
    node cannot capture, so it is stored as a sequence of routes instead (see
    :func:`compress_ring_routes`). It therefore has more than ``T + C + D``
    entries, which is how :func:`untersify_to_G` tells the two encodings apart.

    Returns:
        dict with keys:
            edges: positional links (forest) or a route sequence (ringed)
            clone2prime: mapping the above-T clones to below-T nodes
    """
    R, T, B = (G.graph[k] for k in 'RTB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    if not G.graph.get('has_loads'):
        calcload(G)
    # the recorded topology picks the encoding
    if G.graph['topology'] != 'ringed':
        terse = np.empty((T + C + D,), dtype=int)
        for u, v, edgeD in G.edges(data=True):
            reverse = edgeD.get('reverse')
            if reverse is None:
                raise ValueError('reverse must not be None')
            u, v = (u, v) if u < v else (v, u)
            i, target = (u, v) if reverse else (v, u)
            if i < T:
                terse[i] = target
            else:
                terse[i - B] = target
        edges = terse.tolist()
    else:
        edges = compress_ring_routes(_ring_routes_from_G(G, R, T))
    terse_pack = dict(edges=edges)
    if C > 0 or D > 0:
        terse_pack['clone2prime'] = G.graph['fnT'][T + B : -R].tolist()
    return terse_pack


def infer_topology(G: nx.Graph, terse: Sequence[int]) -> str:
    """Infer the topology of a record stored before ``topology`` was an attribute.

    Reads, in order of authority:

    * the ``terse`` encoding: a route sequence settles RINGED, being the only
      signal that comes from the routeset itself. The converse does not hold --
      a ring of one terminal is a stub, so an all-stub RINGED solution has no
      cycles and is stored positionally, like a forest;
    * ``method_options['topology']``, recorded by the MILP backends. A recorded
      RINGED is taken at face value even against a positional encoding: it is
      right for the all-stub case above, and where it is wrong
      :func:`~optiwindnet.interarraylib.validate_topology` says so, reading the
      whole graph rather than the length of its encoding;
    * ``creator``: HGS and LKH solve a CVRP, whose routes are paths, so their
      non-ringed output is RADIAL. The constructor names its method instead.

    Falls back to ``'branched'``, the weakest claim any forest satisfies, when a
    record carries none of these.

    Args:
      G: routeset graph, already carrying the record's metadata.
      terse: the record's ``edges`` sequence.

    Returns:
      one of ``'ringed'``, ``'radial'`` or ``'branched'``.
    """
    T = G.graph['T']
    C, D = (G.graph.get(k, 0) for k in 'CD')
    if len(terse) != T + C + D:
        return 'ringed'

    method_options = G.graph.get('method_options') or {}
    topology = method_options.get('topology')
    if topology in ('ringed', 'radial', 'branched'):
        return topology

    creator = G.graph.get('creator', '')
    if creator in ('baselines.hgs', 'baselines.lkh'):
        return 'radial'
    if creator == 'constructor':
        return 'radial' if method_options.get('method') == 'radial_EW' else 'branched'
    return 'branched'


def untersify_to_G(G: nx.Graph, terse: list, clone2prime: list) -> None:
    """Rebuild ``G``'s edges from a terse pack. Changes G in place!

    Forest routesets (``len(terse) == T + C + D``) are decoded positionally;
    ringed routesets (more entries) are decoded as a sequence of routes, each
    ring's open point re-derived at the load midpoint of its terminals.

    Sets ``G.graph['topology']`` from :func:`infer_topology` if the record did
    not carry one.
    """
    R, T, B = (G.graph[k] for k in 'RTB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    VertexC = G.graph['VertexC']
    terse = list(terse)
    if clone2prime:
        G.add_nodes_from(range(T + B, T + B + C), kind='contour')
        G.add_nodes_from(range(T + B + C, T + B + C + D), kind='detour')
        fnT = np.arange(R + T + B + C + D)
        fnT[T + B : T + B + C + D] = clone2prime
        fnT[-R:] = range(-R, 0)
        G.graph['fnT'] = fnT
    else:
        fnT = None

    # a stored topology is authoritative; older records get an inferred one
    if 'topology' not in G.graph:
        G.graph['topology'] = infer_topology(G, terse)

    if len(terse) == T + C + D:
        # positional forest encoding: (source, terse[source]) is a directed link
        terse_arr = np.asarray(terse)
        source = np.arange(len(terse_arr))
        if clone2prime:
            source[T:] += B
            Length = np.hypot(*(VertexC[fnT[terse_arr]] - VertexC[fnT[source]]).T)
        else:
            Length = np.hypot(*(VertexC[terse_arr] - VertexC[source]).T)
        G.add_weighted_edges_from(
            zip(source.tolist(), terse_arr.tolist(), Length.tolist()), weight='length'
        )
    else:
        _untersify_ring_seq(G, terse, T, VertexC, fnT)

    if clone2prime:
        for _, _, edgeD in G.edges(range(T + B, T + B + C), data=True):
            edgeD['kind'] = 'contour'
        for _, _, edgeD in G.edges(range(T + B + C, T + B + C + D), data=True):
            edgeD['kind'] = 'detour'
    calcload(G)


def _untersify_ring_seq(
    G: nx.Graph, seq: list[int], T: int, VertexC: np.ndarray, fnT: np.ndarray | None
) -> None:
    """Rebuild a ringed ``G``'s edges from its route sequence (in place)."""

    def length(a: int, b: int) -> float:
        pa, pb = (fnT[a], fnT[b]) if fnT is not None else (a, b)
        return float(np.hypot(*(VertexC[pa] - VertexC[pb])))

    for open_, walk, close in parse_ring_routes(seq):
        term_pos = [k for k, v in enumerate(walk) if v < T]
        n = len(term_pos)
        chain = [open_, *walk] + ([close] if n > 1 else [])
        for a, b in zip(chain, chain[1:]):
            G.add_edge(a, b, length=length(a, b))
        if n > 1:
            # the open point closes the ring at the load midpoint of its arms
            m = math.ceil(n / 2)
            p = term_pos[m - 1]
            if term_pos[m] != p + 1:
                raise ValueError('ring split terminals are not adjacent in the walk')
            u, v = walk[p], walk[p + 1]
            G[u][v].update(load=0, reverse=False)


def oddtypes_to_serializable(obj):
    if isinstance(obj, (list, tuple)):
        return type(obj)(oddtypes_to_serializable(item) for item in obj)
    elif isinstance(obj, dict):
        return {k: oddtypes_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    else:
        return obj


def pack_G(G: nx.Graph) -> dict[str, Any]:
    R, T, B = (G.graph[k] for k in 'RTB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    terse_pack = terse_pack_from_G(G)
    misc = {key: G.graph[key] for key in G.graph.keys() - _misc_not}
    for k, v in misc.items():
        misc[k] = oddtypes_to_serializable(v)
    if not misc:
        misc = {}
    length = G.size(weight='length')
    handle = G.graph.get('handle')
    if handle is None:
        handle = make_handle(G.graph['name'])
    packed_G = dict(
        R=R,
        T=T,
        C=C,
        D=D,
        handle=handle,
        capacity=G.graph['capacity'],
        length=length,
        creator=G.graph['creator'],
        runtime=G.graph['runtime'],
        feeders_per_root=[len(G[root]) for root in range(-R, 0)],
        misc=misc,
        **terse_pack,
    )
    # Optional fields
    if C + D > 0:
        packed_G['clone2prime'] = G.graph['fnT'][-C - D - R : -R].tolist()
    concatenate_tuples = partial(sum, start=())
    pack_if_given = (  # key, function to prepare data
        ('detextra', None),
        ('num_diagonals', None),
        ('tentative', concatenate_tuples),
        ('rogue', concatenate_tuples),
    )
    packed_G.update(
        {
            k: (fun(G.graph[k]) if fun else G.graph[k])
            for k, fun in pack_if_given
            if k in G.graph
        }
    )
    return packed_G


def store_G(G: nx.Graph) -> int:
    """Store ``G``'s data to a new :class:`RouteSet` record in the database.

    If the NodeSet or Method are not yet in the database, they will be added.

    Args:
        G: Graph with the routeset.

    Returns:
        Primary key of the newly created RouteSet record.
    """
    packed_G = pack_G(G)
    nodesetID = nodeset_from_G(G)
    methodID = method_from_G(G)
    machineID = get_machine_pk()
    packed_G.update(
        nodes=nodesetID,
        method=methodID,
        machine=machineID,
    )
    rs = RouteSet.create(**packed_G)
    return rs.id


def get_machine_pk() -> int:
    fqdn = getfqdn()
    hostname = gethostname()
    if fqdn == 'localhost':
        machine = hostname
    else:
        if hostname.startswith('n-'):
            machine = fqdn[len(hostname) :]
        else:
            machine = fqdn
    m, _ = Machine.get_or_create(name=machine)
    return m.id


def G_by_method(G: nx.Graph, method: Method) -> nx.Graph:
    """Fetch from the database a layout for ``G`` by ``method``.

    ``G`` must be a layout solution with the necessary info in the ``G.graph`` dict.
    ``method`` is a Method.
    """
    farmname = G.name
    c = G.graph['capacity']
    rs = (
        RouteSet.select()
        .join(NodeSet)
        .where(
            NodeSet.name == farmname,
            RouteSet.method == method.digest,
            RouteSet.capacity == c,
        )
        .get()
    )
    Gdb = G_from_routeset(rs)
    calcload(Gdb)
    return Gdb


def Gs_from_attrs(
    farm: object,
    methods: Method | Sequence[object],
    capacities: int | Sequence[int],
) -> list[tuple[nx.Graph]]:
    """Fetch from the database a list (one per capacity) of tuples (one per
    method) of layouts.

    ``farm`` must have the desired NodeSet name in the ``name`` attribute.
    ``methods`` is a (sequence of) Method instance(s).
    ``capacities`` is a (sequence of) int(s).
    """
    Gs = []
    if not isinstance(methods, Sequence):
        methods = (methods,)
    if not isinstance(capacities, Sequence):
        capacities = (capacities,)
    for c in capacities:
        Gtuple = tuple(
            G_from_routeset(
                RouteSet.select()
                .join(NodeSet)
                .where(
                    NodeSet.name == farm.name,
                    RouteSet.method == m.digest,
                    RouteSet.capacity == c,
                )
                .get()
            )
            for m in methods
        )
        for G in Gtuple:
            calcload(G)
        Gs.append(Gtuple)
    return Gs
