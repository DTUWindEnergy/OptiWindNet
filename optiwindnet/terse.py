# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Compact, self-describing encodings of solution and routeset links."""

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
from itertools import pairwise
from typing import Any

import networkx as nx
import numpy as np

from .fingerprint import fingerprint_coordinates
from .types import Topology

__all__ = ('LinkScope', 'TerseLinks')


class LinkScope(StrEnum):
    """Graph layer represented by :class:`TerseLinks`."""

    TOPOLOGY = auto()
    ROUTESET = auto()


@dataclass(frozen=True, slots=True)
class _EdgeSpec:
    u: int
    v: int
    is_open: bool = False


def _compress_ring_routes(
    routes: Sequence[tuple[int, Sequence[int], int]],
) -> tuple[int, ...]:
    """Flatten ``(open_root, walk, close_root)`` ring routes."""
    seq: list[int] = []
    previous_root = None
    last = len(routes) - 1
    for idx, (open_, walk, close) in enumerate(routes):
        if open_ != previous_root:
            seq.append(int(open_))
        seq.extend(int(x) for x in walk)
        if idx != last or close != open_:
            seq.append(int(close))
        previous_root = close
    return tuple(seq)


def _parse_ring_routes(seq: Sequence[int]) -> list[tuple[int, list[int], int]]:
    """Recover ``(open_root, walk, close_root)`` ring routes."""
    routes: list[tuple[int, list[int], int]] = []
    values = [int(x) for x in seq]
    i, length, previous_root = 0, len(values), None
    while i < length:
        if values[i] < 0:
            open_ = values[i]
            i += 1
        elif previous_root is not None:
            open_ = previous_root
        else:
            raise ValueError('ring route does not begin at a root')

        walk: list[int] = []
        while i < length and values[i] >= 0:
            walk.append(values[i])
            i += 1
        if not walk:
            raise ValueError('ring route contains no non-root nodes')
        if i >= length:
            close = open_
        else:
            close = values[i]
            i += 1
        previous_root = close
        routes.append((open_, walk, close))
    return routes


def _walk_ring(G: nx.Graph, root: int, subroot: int) -> tuple[list[int], int]:
    """Walk a ring from one root feeder to the other."""
    walk = [subroot]
    previous, current = root, subroot
    while True:
        neighbours = [node for node in G[current] if node != previous]
        if not neighbours:
            return walk, root
        if len(neighbours) != 1:
            raise ValueError(
                f'ring node {current} is not degree-2 (neighbours {neighbours})'
            )
        (next_,) = neighbours
        if next_ < 0:
            return walk, next_
        walk.append(next_)
        previous, current = current, next_


def _ring_routes_from_graph(
    G: nx.Graph, R: int, T: int
) -> list[tuple[int, list[int], int]]:
    """Extract canonical ring routes from either a topology or routeset graph."""
    routes: list[tuple[int, list[int], int]] = []
    visited_feeders: set[tuple[int, int]] = set()
    for root in range(-R, 0):
        for subroot in sorted(G[root]):
            if (root, subroot) in visited_feeders:
                continue
            walk, close = _walk_ring(G, root, subroot)
            visited_feeders.add((root, walk[0]))
            visited_feeders.add((close, walk[-1]))
            open_ = root

            terminal_positions = [i for i, node in enumerate(walk) if node < T]
            terminal_count = len(terminal_positions)
            if terminal_count == 0:
                raise ValueError('ring route contains no terminals')
            if terminal_count == 1 and close != open_:
                if G[open_][walk[0]].get('load') == 0:
                    open_, close = close, open_
                    walk.reverse()
            elif terminal_count > 1:
                open_boundaries = []
                for boundary, (left, right) in enumerate(pairwise(terminal_positions)):
                    segment = zip(walk[left:right], walk[left + 1 : right + 1])
                    if all(G[u][v].get('load') == 0 for u, v in segment):
                        open_boundaries.append(boundary)
                if len(open_boundaries) != 1:
                    raise ValueError(
                        'ring must have one open terminal-to-terminal link; '
                        f'found {len(open_boundaries)}'
                    )
                if open_boundaries[0] != math.ceil(terminal_count / 2) - 1:
                    walk.reverse()
                    open_, close = close, open_
            routes.append((open_, walk, close))
    return routes


def _parent_links_from_graph(G: nx.Graph, T: int, B: int, clone_count: int):
    targets: list[int | None] = [None] * (T + clone_count)
    for u, v, edge_data in G.edges(data=True):
        reverse = edge_data.get('reverse')
        if reverse is None:
            raise ValueError('reverse must not be None')
        u, v = (u, v) if u < v else (v, u)
        source, target = (u, v) if reverse else (v, u)
        slot = source if source < T else source - B
        if not 0 <= slot < len(targets):
            raise ValueError(f'parent source {source} is outside the encoded node set')
        if targets[slot] is not None:
            raise ValueError(f'node {source} has more than one parent')
        targets[slot] = int(target)
    missing = [i for i, target in enumerate(targets) if target is None]
    if missing:
        raise ValueError(f'nodes without an encoded parent: {missing}')
    return tuple(target for target in targets if target is not None)


@dataclass(frozen=True, slots=True)
class TerseLinks(Sequence[int]):
    """Compact links plus the metadata needed to reconstruct their graph.

    ``RADIAL`` and ``BRANCHED`` values use one parent target per non-root node.
    ``RINGED`` values use the flattened route representation. A topology value
    represents ``S``; a routeset value additionally carries the clone mapping
    needed to reproduce ``G`` against its location graph.
    """

    links: tuple[int, ...]
    topology: Topology
    scope: LinkScope
    T: int
    R: int
    B: int = 0
    C: int = 0
    D: int = 0
    clone2prime: tuple[int, ...] = ()
    nodeset_digest: bytes | None = None

    def __post_init__(self):
        object.__setattr__(self, 'links', tuple(int(x) for x in self.links))
        object.__setattr__(self, 'topology', Topology(self.topology))
        object.__setattr__(self, 'scope', LinkScope(self.scope))
        object.__setattr__(self, 'clone2prime', tuple(int(x) for x in self.clone2prime))
        if min(self.T, self.R, self.B, self.C, self.D) < 0 or self.R == 0:
            raise ValueError('T, R, B, C and D must be non-negative, with R > 0')
        if len(self.clone2prime) != self.C + self.D:
            raise ValueError('clone2prime must have exactly C + D entries')
        if self.scope is LinkScope.TOPOLOGY and (
            self.B
            or self.C
            or self.D
            or self.clone2prime
            or self.nodeset_digest is not None
        ):
            raise ValueError('a topology encoding cannot carry routed nodes')
        if self.topology is not Topology.RINGED:
            expected = self.T + self.C + self.D
            if len(self.links) != expected:
                raise ValueError(
                    f'{self.topology} parent encoding requires {expected} links; '
                    f'received {len(self.links)}'
                )
            valid_nodes = set(range(self.T))
            valid_nodes.update(
                range(self.T + self.B, self.T + self.B + self.C + self.D)
            )
            valid_nodes.update(range(-self.R, 0))
            invalid = sorted(set(self.links) - valid_nodes)
            if invalid:
                raise ValueError(f'parent targets contain invalid nodes: {invalid}')
        else:
            routes = _parse_ring_routes(self.links)
            roots = [root for open_, _, close in routes for root in (open_, close)]
            invalid_roots = sorted({root for root in roots if not -self.R <= root < 0})
            if invalid_roots:
                raise ValueError(f'ring routes contain invalid roots: {invalid_roots}')
            walked = [node for _, walk, _ in routes for node in walk]
            expected = list(range(self.T))
            expected.extend(range(self.T + self.B, self.T + self.B + self.C + self.D))
            if sorted(walked) != expected:
                raise ValueError(
                    'ring routes must contain every terminal and clone exactly once'
                )

    def __len__(self) -> int:
        return len(self.links)

    def __iter__(self) -> Iterator[int]:
        return iter(self.links)

    def __getitem__(self, index):
        return self.links[index]

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        array = np.asarray(self.links, dtype=dtype)
        if copy:
            array = array.copy()
        return array

    def __repr__(self) -> str:
        parts = [
            f'scope={self.scope.value!r}',
            f'topology={self.topology.value!r}',
            f'T={self.T}',
            f'R={self.R}',
        ]
        if self.scope is LinkScope.ROUTESET:
            parts.extend((f'B={self.B}', f'C={self.C}', f'D={self.D}'))
        parts.append(f'links={len(self.links)}')
        return f'TerseLinks({", ".join(parts)})'

    def tolist(self) -> list[int]:
        """Return the encoded links as a plain list."""
        return list(self.links)

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned representation suitable for JSON serialization."""
        return {
            'version': 1,
            'links': self.tolist(),
            'topology': self.topology.value,
            'scope': self.scope.value,
            'T': self.T,
            'R': self.R,
            'B': self.B,
            'C': self.C,
            'D': self.D,
            'clone2prime': list(self.clone2prime),
            'nodeset_digest': (
                self.nodeset_digest.hex() if self.nodeset_digest is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'TerseLinks':
        """Restore a value created by :meth:`to_dict`."""
        version = data.get('version')
        if version != 1:
            raise ValueError(f'unsupported TerseLinks version: {version!r}')
        nodeset_digest = data.get('nodeset_digest')
        return cls(
            links=tuple(data['links']),
            topology=data['topology'],
            scope=data['scope'],
            T=data['T'],
            R=data['R'],
            B=data.get('B', 0),
            C=data.get('C', 0),
            D=data.get('D', 0),
            clone2prime=tuple(data.get('clone2prime', ())),
            nodeset_digest=(
                bytes.fromhex(nodeset_digest) if nodeset_digest is not None else None
            ),
        )

    @classmethod
    def from_topology(cls, S: nx.Graph) -> 'TerseLinks':
        """Encode solution topology ``S``."""
        topology = Topology(S.graph['topology'])
        R, T = (S.graph[key] for key in 'RT')
        if topology is Topology.RINGED:
            links = _compress_ring_routes(_ring_routes_from_graph(S, R, T))
        else:
            links = _parent_links_from_graph(S, T, B=0, clone_count=0)
        return cls(
            links=links,
            topology=topology,
            scope=LinkScope.TOPOLOGY,
            T=T,
            R=R,
        )

    @classmethod
    def from_routeset(cls, G: nx.Graph) -> 'TerseLinks':
        """Encode routed solution ``G``, including its clone mapping."""
        if not G.graph.get('has_loads'):
            G = G.copy()
            from .interarraylib import calcload

            calcload(G)
        topology = Topology(G.graph['topology'])
        R, T, B = (G.graph[key] for key in 'RTB')
        C, D = (G.graph.get(key, 0) or 0 for key in 'CD')
        if topology is Topology.RINGED:
            links = _compress_ring_routes(_ring_routes_from_graph(G, R, T))
        else:
            links = _parent_links_from_graph(G, T, B, C + D)
        clone2prime = tuple(int(x) for x in G.graph['fnT'][T + B : -R]) if C + D else ()
        return cls(
            links=links,
            topology=topology,
            scope=LinkScope.ROUTESET,
            T=T,
            R=R,
            B=B,
            C=C,
            D=D,
            clone2prime=clone2prime,
            nodeset_digest=fingerprint_coordinates(G.graph['VertexC'])[0],
        )

    @classmethod
    def from_array(
        cls,
        links: Sequence[int],
        *,
        topology: Topology | str | None = None,
        R: int | None = None,
        T: int | None = None,
    ) -> 'TerseLinks':
        """Adapt a legacy topology array to a self-describing value."""
        values = tuple(int(x) for x in links)
        if topology is not None:
            topology = Topology(topology)
        if T is None:
            T = (
                sum(value >= 0 for value in values)
                if topology is Topology.RINGED
                else len(values)
            )
        if topology is None:
            topology = Topology.RINGED if len(values) != T else Topology.BRANCHED
        if R is None:
            R = abs(min(values, default=-1))
        return cls(
            links=values,
            topology=topology,
            scope=LinkScope.TOPOLOGY,
            T=T,
            R=R,
        )

    def _edge_specs(self) -> Iterator[_EdgeSpec]:
        if self.topology is not Topology.RINGED:
            for slot, target in enumerate(self.links):
                source = slot if slot < self.T else slot + self.B
                yield _EdgeSpec(source, target)
            return

        for open_, walk, close in _parse_ring_routes(self.links):
            terminal_positions = [i for i, node in enumerate(walk) if node < self.T]
            terminal_count = len(terminal_positions)
            chain = [open_, *walk]
            if terminal_count > 1 or close != open_:
                chain.append(close)
            open_edges: set[frozenset[int]] = set()
            if terminal_count == 1 and close != open_:
                open_edges.add(frozenset((walk[-1], close)))
            elif terminal_count > 1:
                midpoint = math.ceil(terminal_count / 2)
                left = terminal_positions[midpoint - 1]
                right = terminal_positions[midpoint]
                open_edges.update(
                    frozenset(edge) for edge in pairwise(walk[left : right + 1])
                )
            for u, v in pairwise(chain):
                yield _EdgeSpec(
                    u,
                    v,
                    is_open=frozenset((u, v)) in open_edges,
                )

    def to_topology(self, **graph_attrs) -> nx.Graph:
        """Reconstruct topology ``S``."""
        if self.scope is not LinkScope.TOPOLOGY:
            raise ValueError('a routeset encoding must be reconstructed with a site')
        S = nx.Graph(T=self.T, R=self.R, topology=self.topology, **graph_attrs)
        S.add_nodes_from(range(-self.R, 0), kind='oss')
        S.add_nodes_from(range(self.T), kind='wtg')
        for edge in self._edge_specs():
            attrs = {'load': 0, 'reverse': False} if edge.is_open else {}
            S.add_edge(edge.u, edge.v, **attrs)

        from .interarraylib import calcload

        calcload(S)
        if 'capacity' not in graph_attrs:
            S.graph['capacity'] = S.graph['max_load']
        return S

    def to_routeset(self, L: nx.Graph, **graph_attrs) -> nx.Graph:
        """Reconstruct routeset ``G`` against location graph ``L``."""
        if self.scope is not LinkScope.ROUTESET:
            raise ValueError('a topology encoding must be routed before creating G')
        for key, expected in (('R', self.R), ('T', self.T), ('B', self.B)):
            actual = L.graph[key]
            if actual != expected:
                raise ValueError(
                    f'location {key}={actual} does not match encoding {key}={expected}'
                )
        if (
            self.nodeset_digest is not None
            and fingerprint_coordinates(L.graph['VertexC'])[0] != self.nodeset_digest
        ):
            raise ValueError('location geometry does not match the routed encoding')
        G = nx.create_empty_copy(L, with_data=True)
        G.graph.update(
            topology=self.topology,
            C=self.C,
            D=self.D,
            **graph_attrs,
        )
        clone_start = self.T + self.B
        contour_nodes = range(clone_start, clone_start + self.C)
        detour_nodes = range(clone_start + self.C, clone_start + self.C + self.D)
        G.add_nodes_from(contour_nodes, kind='contour')
        G.add_nodes_from(detour_nodes, kind='detour')
        if self.C + self.D:
            fnT = np.arange(self.R + self.T + self.B + self.C + self.D)
            fnT[clone_start : clone_start + self.C + self.D] = self.clone2prime
            fnT[-self.R :] = range(-self.R, 0)
            G.graph['fnT'] = fnT
        else:
            fnT = None

        VertexC = G.graph['VertexC']

        def coordinate_index(node: int) -> int:
            return int(fnT[node]) if fnT is not None else node

        for edge in self._edge_specs():
            u_coord, v_coord = coordinate_index(edge.u), coordinate_index(edge.v)
            attrs = {
                'length': float(np.hypot(*(VertexC[u_coord] - VertexC[v_coord]))),
            }
            if edge.is_open:
                attrs.update(load=0, reverse=False)
            G.add_edge(edge.u, edge.v, **attrs)

        for _, _, edge_data in G.edges(contour_nodes, data=True):
            edge_data['kind'] = 'contour'
        for _, _, edge_data in G.edges(detour_nodes, data=True):
            edge_data['kind'] = 'detour'

        from .interarraylib import calcload

        calcload(G)
        return G
