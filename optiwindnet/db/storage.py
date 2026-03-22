# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import base64
import io
import json
from collections.abc import Sequence
from functools import partial
from hashlib import sha256
from itertools import chain, pairwise
from socket import getfqdn, gethostname
from typing import Any, Mapping

import networkx as nx
import numpy as np

from .model import (
    Machine,
    Method,
    NodeSet,
    RouteSet,
)
from ..interarraylib import calcload
from ..utils import make_handle

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
    R = nodeset.R
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

    edges = routeset.edges
    clone2prime = routeset.clone2prime
    if routeset.stuntC:
        stuntC = np.lib.format.read_array(io.BytesIO(routeset.stuntC))
        stunt_count = len(stuntC)
        lo = nodeset.T + nodeset.B
        hi = lo + stunt_count
        edges = list(routeset.edges)
        for i, target in enumerate(edges):
            if lo <= target < hi:
                raise ValueError(
                    f'RouteSet {routeset.id} edge target points into stunt range: '
                    f'edges[{i}]={target}, stunt range=[{lo}, {hi})'
                )
            if target >= hi:
                edges[i] = target - stunt_count
        if clone2prime:
            clone2prime = list(clone2prime)
            VertexC = np.lib.format.read_array(io.BytesIO(nodeset.VertexC))
            nearest: list[int] = []
            for coord in stuntC:
                delta = VertexC - coord
                sqdist = np.einsum('ij,ij->i', delta, delta)
                nearest.append(int(np.argmin(sqdist)))
            for i, target in enumerate(clone2prime):
                if lo <= target < hi:
                    clone2prime[i] = nearest[target - lo]

    untersify_to_G(G, terse=edges, clone2prime=clone2prime)
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


def add_if_absent(entity: NodeSet | Method, pack: PackType) -> bytes:
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


def terse_pack_from_G(G: nx.Graph) -> PackType:
    """Convert `G`'s edges to a format suitable for storing in the database.

    Although graph `G` in undirected, the edge attribute `'reverse'` and its
    nodes' numbers encode the direction of power flow. The terse
    representation uses that and the fact that `G` is a tree.

    Returns:
        dict with keys:
            edges: where <i, edges[i]> is a directed edge of `G`
            clone2prime: mapping the above-T clones to below-T nodes
    """
    R, T, B = (G.graph[k] for k in 'RTB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    terse = np.empty((T + C + D,), dtype=int)
    if not G.graph.get('has_loads'):
        calcload(G)
    for u, v, reverse in G.edges(data='reverse'):
        if reverse is None:
            raise ValueError('reverse must not be None')
        u, v = (u, v) if u < v else (v, u)
        i, target = (u, v) if reverse else (v, u)
        if i < T:
            terse[i] = target
        else:
            terse[i - B] = target
    terse_pack = dict(edges=terse.tolist())
    if C > 0 or D > 0:
        terse_pack['clone2prime'] = G.graph['fnT'][T + B : -R].tolist()
    return terse_pack


def untersify_to_G(G: nx.Graph, terse: list, clone2prime: list) -> None:
    """
    Changes G in place!
    """
    R, T, B = (G.graph[k] for k in 'RTB')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    VertexC = G.graph['VertexC']
    terse = np.asarray(terse)
    source = np.arange(len(terse))
    if clone2prime:
        source[T:] += B
        contournodes = range(T + B, T + B + C)
        detournodes = range(T + B + C, T + B + C + D)
        G.add_nodes_from(contournodes, kind='contour')
        G.add_nodes_from(detournodes, kind='detour')
        fnT = np.arange(R + T + B + C + D)
        fnT[T + B : T + B + C + D] = clone2prime
        fnT[-R:] = range(-R, 0)
        G.graph['fnT'] = fnT
        Length = np.hypot(*(VertexC[fnT[terse]] - VertexC[fnT[source]]).T)
    else:
        Length = np.hypot(*(VertexC[terse] - VertexC[source]).T)
    G.add_weighted_edges_from(
        zip(source.tolist(), terse.tolist(), Length.tolist()), weight='length'
    )
    if clone2prime:
        for _, _, edgeD in G.edges(contournodes, data=True):
            edgeD['kind'] = 'contour'
        for _, _, edgeD in G.edges(detournodes, data=True):
            edgeD['kind'] = 'detour'
    calcload(G)


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
        is_normalized=G.graph.get('is_normalized', False),
        runtime=G.graph['runtime'],
        num_gates=[len(G[root]) for root in range(-R, 0)],
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
        ('valid', None),
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
    """Store `G`'s data to a new `RouteSet` record in the database.

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
    """Fetch from the database a layout for `G` by `method`.
    `G` must be a layout solution with the necessary info in the G.graph dict.
    `method` is a Method.
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
    """
    Fetch from the database a list (one per capacity) of tuples (one per
    method) of layouts.
    `farm` must have the desired NodeSet name in the `name` attribute.
    `methods` is a (sequence of) Method instance(s).
    `capacities` is a (sequence of) int(s).
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
