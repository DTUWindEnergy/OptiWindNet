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

from ..fingerprint import fingerprint_coordinates
from ..interarraylib import calcload
from ..terse import LinkScope, TerseLinks
from ..types import Topology
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

    if 'topology' not in G.graph:
        G.graph['topology'] = infer_topology(G, routeset.edges)
    encoding = TerseLinks(
        links=tuple(routeset.edges),
        topology=G.graph['topology'],
        scope=LinkScope.ROUTESET,
        T=G.graph['T'],
        R=G.graph['R'],
        B=G.graph['B'],
        C=G.graph['C'],
        D=G.graph['D'],
        clone2prime=tuple(routeset.clone2prime or ()),
        nodeset_digest=bytes(nodeset.digest),
    )
    G = encoding.to_routeset(G)
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
    digest, VertexC_npy = fingerprint_coordinates(G.graph['VertexC'])

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


def infer_topology(G: nx.Graph, terse: Sequence[int]) -> Topology:
    """Infer the topology of a record stored before ``topology`` was an attribute.

    Reads, in order of authority:

    * the ``terse`` encoding: a route sequence settles RINGED, being the only
      signal that comes from the routeset itself;
    * ``method_options['topology']``, recorded by the MILP backends;
    * ``creator``: HGS and LKH solve a CVRP, whose routes are paths, so their
      non-ringed output is RADIAL. The constructor names its method instead.

    Falls back to ``'branched'``, the weakest claim any forest satisfies, when a
    record carries none of these.

    Args:
      G: routeset graph, already carrying the record's metadata.
      terse: the record's ``edges`` sequence.

    Returns:
      Topology enum entry (``Topology.{RINGED, RADIAL, BRANCHED}``).
    """
    T = G.graph['T']
    C, D = (G.graph.get(k, 0) for k in 'CD')
    if len(terse) != T + C + D:
        return Topology.RINGED

    method_options = G.graph.get('method_options') or {}
    topology = method_options.get('topology')
    if topology in ('ringed', 'radial', 'branched'):
        return Topology(topology)

    creator = G.graph.get('creator', '')
    if creator in ('baselines.hgs', 'baselines.lkh'):
        return Topology.RADIAL
    if creator == 'constructor':
        if method_options.get('method') == 'radial_EW':
            return Topology.RADIAL
        return Topology.BRANCHED
    return Topology.BRANCHED


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
    R, T = (G.graph[k] for k in 'RT')
    C, D = (G.graph.get(k, 0) for k in 'CD')
    terse = TerseLinks.from_routeset(G)
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
        edges=terse.tolist(),
    )
    # Optional fields
    if C + D > 0:
        packed_G['clone2prime'] = list(terse.clone2prime)
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
