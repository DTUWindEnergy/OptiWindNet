# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import heapq
import logging
import math
from bisect import bisect_left
from collections import defaultdict, namedtuple
from itertools import chain

import networkx as nx
import numpy as np
from bitarray import bitarray
from scipy.stats import rankdata

from .crossings import gateXing_iter
from .geometric import rotation_checkers_factory
from .interarraylib import bfs_subtree_loads, scaffolded
from .mesh import planar_flipped_by_routeset

__all__ = ('PathFinder',)

_lggr = logging.getLogger(__name__)
debug, info, warn, error = _lggr.debug, _lggr.info, _lggr.warning, _lggr.error

NULL = np.iinfo(int).min
PseudoNode = namedtuple('PseudoNode', 'prime sector parent dist d_hop cum_turn'.split())
# Terminology used by PathFinder internals:
#   wall: one non-traversable mesh segment; route walls are contour edges of
#     the route, constraint walls are planar constraint edges (borders and
#     obstacles).
#   fence: a sequence of walls forming a polygonal line. Two flavors share
#     this concept: route fences (a cable's run that includes >= 1 constraint
#     edge) and constraint fences (the planar constraint chain itself).
#   chain: the overlap of two fences; chain walking handles these explicitly.
#   portal: a traversable mesh edge between adjacent triangles.
#   channel: the triangle corridor explored through portals.
#   funnel: shortest-path state maintained while advancing along a channel.
#   fan: the full cyclic neighborhood around a vertex, especially a root.
#   cone: one angular region in a fan, bounded by two wall-neighbor vertices.
#     The two bounding walls belong to *distinct* fences — a cone is
#     precisely the wedge that separates two fences at a shared vertex.
#     Two adjacent walls of the same fence (e.g. the two endpoints of a
#     touching fence at one vertex, or the two constraint walls at a
#     non-chain-end vertex) do not form a cone in this sense; they bound
#     the fence's own inside/outside, which is not traversable.
#   prime: a geometry vertex id; pn_id: a pseudonode id in the PathNodes tree.
# Funnel mechanics:
#   portal advance: one `traverser.send((portal, side))` step that feeds a new
#     vertex `_new = portal[side]` into the funnel. It is the unit of channel
#     progress; whether the funnel narrows or the apex moves is decided
#     downstream, inside `_traverse_channel`, from the geometry of `_new`.
#   apex (lagging): the funnel's convergence vertex (`_apex` / `apex`). It
#     stays fixed across "inside" portal advances and only moves when an
#     ultrafar or infranear step forces a wapex walk-back. Comments separate
#     the lagging `_apex` from the per-step `_apex_eff` (effective apex used
#     for the new pseudonode's parent and distance), which can differ on
#     infranear steps where `_apex` itself does not update.
# A chain-end is a constraint vertex where one or more route fences transition
# on/off the constraint (= the start or end of a route fence's on-constraint
# segment). The fences at the chain-end stack cyclically; n distinct fences
# (the constraint fence + the route fences that transition here) define n-1
# chains, one per cyclic-adjacent pair of fences.
#
# A Fence (in this module) records one route fence — i.e. one cable's run
# that includes >= 1 constraint edge:
#   endpoints: (s, t) — the off-constraint primes (= the A-edge endpoints
#     for regular contours, or the shortened-contour edge endpoints).
#   primes_on_constraint: ordered prime-vertex list (>= 1 vertex) along the
#     on-constraint segment.
#   subtree: the cable's subtree id; used as the `sector` label for chain-
#     walk pseudonodes at on-constraint primes so overlapping chains keep
#     distinct (prime, sector) buckets.
# Constraint fences are not represented as Fence instances — their walls
# come from the planar embedding's constraint edges.
Fence = namedtuple('Fence', 'endpoints primes_on_constraint subtree')
# AccessCone: one angular wedge at a chain-end through which a chain can be
# entered or exited. The wedge is bounded by two wall-neighbor primes
# (`left`, `right`) of `vertex` in CW order around `vertex`, and may contain
# zero or more `spokes` (non-bound cyclic neighbors interior to the wedge,
# in CW order from `left` toward `right`). Only chain-interior wedges are
# represented as AccessCones — non-chain regions (the void on the far side
# of the constraint, and any navigable wedge that doesn't separate two
# fences) are not registered, since the chain mechanism has no business
# there.
AccessCone = namedtuple('AccessCone', 'vertex left right spokes')
# Chain: one overlap between two fences. Each chain owns exactly two access
# cones, one at each of its chain-end vertices (or both at the same vertex
# for single-vertex/touching chains). Either cone may serve as entry, with
# the other as exit — what's forbidden is entering and exiting through the
# same cone.
#   subtree: route-fence subtree id; doubles as the chain identifier (each
#     route fence labels exactly one chain) and as the `sector` label for
#     chain-walk pseudonodes so overlapping chains keep distinct
#     (prime, sector) buckets.
#   cones: 2-tuple of AccessCone.
#   walks: 2-tuple of prime sequences. walks[i] steps from cones[i].vertex
#     (exclusive) to cones[1 - i].vertex (inclusive). Empty for single-vertex
#     chains where the two cones share `vertex`.
Chain = namedtuple('Chain', 'subtree cones walks')


def _sorted3(a: int, b: int, c: int) -> tuple[int, int, int]:
    """Return three integers sorted ascending without allocating a list."""
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    return a, b, c


def _node_dist(VertexC: np.ndarray, u: int, v: int) -> float:
    """Euclidean distance between two indexed coordinate rows."""
    ux, uy = VertexC[u]
    vx, vy = VertexC[v]
    return math.hypot(ux - vx, uy - vy)


def _expand_P_paths_edge(
    s: int, t: int, shortcuts: dict[tuple[int, int], list[int]]
) -> list[int]:
    """Recursively expand a P_paths shortcut hop into the full P-edge sequence.

    `shortcuts` maps a normalized (u_lo, v_hi) pair to the list of vertices
    along the underlying P-path. Returns [s, t] verbatim when (s, t) is not
    a shortcut.
    """
    key = (s, t) if s < t else (t, s)
    path = shortcuts.get(key)
    if path is None:
        return [s, t]
    if path[0] != s:
        path = path[::-1]
    expanded = [path[0]]
    for u, v in zip(path[:-1], path[1:]):
        expanded.extend(_expand_P_paths_edge(u, v, shortcuts)[1:])
    return expanded


def _expand_P_paths_path(
    path: list[int], shortcuts: dict[tuple[int, int], list[int]]
) -> list[int]:
    """Expand every shortcut hop along `path` into its underlying P-edges."""
    expanded = [path[0]]
    for s, t in zip(path[:-1], path[1:]):
        expanded.extend(_expand_P_paths_edge(s, t, shortcuts)[1:])
    return expanded


def _undo_P_paths_expansion(G: nx.Graph, A: nx.Graph) -> None:
    """Reverse `G_from_S`'s expansion of P_paths shortcuts in contour chains.

    Each contour A-edge chain in G follows the fully P-edge-expanded shortpath.
    For the final routeset every prime sub-sequence that matches an entry of
    `P_paths_shortcuts` collapses back into a single shortcut edge.
    """
    P = A.graph.get('P_paths_shortcuts')
    if not P or G.graph.get('C', 0) == 0:
        return
    fnT, VertexC = G.graph['fnT'], G.graph['VertexC']
    T, B, R = (G.graph[k] for k in 'TBR')
    chains: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for u, v, d in G.edges(data=True):
        if d.get('kind') == 'contour' and 'A_edge' in d:
            chains[d['A_edge']].append((u, v))
    aux_nodes, drop, add = set(), [], []
    for (s, t), edges in chains.items():
        adj: dict[int, list[int]] = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        nodes, prev, curr = [s], None, s
        while curr != t:
            nxt = next(n for n in adj[curr] if n != prev)
            nodes.append(nxt)
            prev, curr = curr, nxt
        primes = [int(fnT[n]) for n in nodes]
        keep, i = [0], 0
        while i < len(primes) - 1:
            j = i + 1
            for k in range(len(primes) - 1, i + 1, -1):
                a, b = primes[i], primes[k]
                key = (a, b) if a < b else (b, a)
                sc = P.get(key)
                if (
                    sc is not None
                    and (list(sc) if a < b else list(sc[::-1])) == primes[i : k + 1]
                ):
                    j = k
                    break
            keep.append(j)
            i = j
        if len(keep) == len(nodes):
            continue
        drop.extend(edges)
        load = G[edges[0][0]][edges[0][1]].get('load')
        keep_set = set(keep)
        for k in range(len(nodes)):
            if k not in keep_set:
                aux_nodes.add(nodes[k])
        for ki, kj in zip(keep[:-1], keep[1:]):
            length = float(np.hypot(*(VertexC[primes[ki]] - VertexC[primes[kj]])))
            add.append(
                (
                    nodes[ki],
                    nodes[kj],
                    dict(
                        length=length,
                        load=load,
                        kind='contour',
                        A_edge=(s, t),
                        reverse=nodes[ki] < nodes[kj],
                    ),
                )
            )
    if not aux_nodes:
        return
    G.remove_edges_from(drop)
    G.remove_nodes_from(aux_nodes)
    for u, v, attrs in add:
        G.add_edge(u, v, **attrs)
    contour = sorted(n for n, d in G.nodes(data=True) if d.get('kind') == 'contour')
    detour = sorted(n for n, d in G.nodes(data=True) if d.get('kind') == 'detour')
    Cn, Dn = len(contour), len(detour)
    mapping, nl = {}, T + B
    for n in contour + detour:
        if n != nl:
            mapping[n] = nl
        nl += 1
    inv = {v: k for k, v in mapping.items()}
    new_fnT = np.arange(R + T + B + Cn + Dn)
    new_fnT[-R:] = range(-R, 0)
    for nl in range(T + B, T + B + Cn + Dn):
        new_fnT[nl] = int(fnT[inv.get(nl, nl)])
    if mapping:
        nx.relabel_nodes(G, mapping, copy=False)
    G.graph.update(fnT=new_fnT, C=Cn, D=Dn)


class PathNodes(dict):
    """Tree of pseudonodes for shortest-path candidates.

    A prime is a geometry vertex id. A pseudonode id (`pn_id`) identifies one
    occurrence of a prime in the path tree, since the same prime can be reached
    from different sectors or parents.
    """

    count: int
    prime_from_pn: dict
    pn_ids_from_prime_sector: defaultdict
    last_added_pn: int

    def __init__(self):
        super().__init__()
        self.count = 0
        self.prime_from_pn = {}
        self.pn_ids_from_prime_sector = defaultdict(list)
        self.last_added_pn = NULL

    def add(
        self,
        prime: int,
        sector: int,
        parent_pn: int,
        dist: float,
        d_hop: float,
        cum_turn: float = 0.0,
    ) -> int:
        if parent_pn not in self:
            error(
                'attempted to add an edge in `PathNodes` to nonexistent parent (%d)',
                parent_pn,
            )
        parent_prime = self.prime_from_pn[parent_pn]
        for prev_pn_id in self.pn_ids_from_prime_sector[prime, sector]:
            if self[prev_pn_id].parent == parent_pn:
                self.last_added_pn = prev_pn_id
                return prev_pn_id
        pn_id = self.count
        self.count += 1
        self[pn_id] = PseudoNode(prime, sector, parent_pn, dist, d_hop, cum_turn)
        self.pn_ids_from_prime_sector[prime, sector].append(pn_id)
        self.prime_from_pn[pn_id] = prime
        debug('pseudoedge «%d->%d» added', prime, parent_prime)
        self.last_added_pn = pn_id
        return pn_id


class PathFinder:
    """Router for feeders that would cross other routes if laid in a straight line.

    PathFinder finds the shortest segmented (or detoured) routes for tentative feeders
    (i.e. those that were created without a check for crossings of other routes). The
    path-finding is performed when the instance is initialized, but a route set is
    returned only with a call to method `.create_detours()`.

    Only edges in graph attribute 'tentative' or, lacking that, edges with the
    attribute 'kind' == 'tentative' are checked for crossings.

    Args:
      G: the route set without detours
      P: the planar embedding associated with A
      A: the available links graph
      branched: if True, any terminal can be linked to root, else only subtrees'
        heads/tails
      iterations_limit: maximum number of steps in the path-finding process
      traversals_limit: maximum number of times a single portal may be traversed
      bad_streak_limit: limit on how many steps in a row without finding an improved
        path the traverser is allowed to take

    Example::

      P, A = make_planar_embedding(L)  # L represents the geometry of the location
      S = some_solver(A, ...)  # S is a topology
      G_tentative = G_from_S(S, A)  # G_tentative is almost a route set
      G = PathFinder(G_tentative, planar=P, A=A).create_detours()

    Note:
      On ``capacity=2`` instances the defaults may not suffice to find all
      shortest feeders. If `validate_routeset(G)` reports any crossings, retry
      with ``traversals_limit=10`` and ``iterations_limit=50000``.

    """

    def __init__(
        self,
        Gʹ: nx.Graph,
        planar: nx.PlanarEmbedding,
        A: nx.Graph,
        *,
        branched: bool = True,
        iterations_limit: int = 15000,
        traversals_limit: int = 3,
        bad_streak_limit: int = 5,
        turn_limit: float | None = None,
    ) -> None:
        self.iterations_limit = iterations_limit
        self.traversals_limit = traversals_limit
        self.bad_streak_limit = bad_streak_limit
        # Path-cumulative turn limit (advancers whose path winding exceeds
        # this are dropped) scales (sub-)logarithmically with cable capacity
        # Q: f(Q) = 3π/4 + (5π/4) * ln(Q/2) / ln(6), giving f(2) = 3π/4 and
        # f(12) = 2π. Lower-capacity routes have simpler geometry, so excess
        # winding is more likely circling; higher-capacity routes legitimately
        # need more wrap. Pass an explicit value to override.
        if turn_limit is None:
            Q = Gʹ.graph.get('capacity')
            if Q is None or Q < 2:
                turn_limit = 2.0 * math.pi
            else:
                turn_limit = (3 * math.pi / 4) + (
                    (5 * math.pi / 4) * math.log(Q / 2) / math.log(6)
                )
        self.turn_limit = turn_limit
        self.iterations = 0
        G = Gʹ.copy()
        R, T, B = (A.graph[k] for k in 'RTB')
        C = G.graph.get('C', 0)
        assert not G.graph.get('D'), 'Gʹ has already has detours.'
        self.ST = T + B

        debug(
            '>PathFinder: "%s" (T = %d)',
            G.graph.get('name') or G.graph.get('handle') or 'unnamed',
            T,
        )

        # tentative will be copied later, by initializing a set from it.
        tentative = G.graph.get('tentative')
        if tentative is None:
            tentative = []
            hooks_by_root = []
            for r in range(-R, 0):
                feeders = set(
                    n for n in G.neighbors(r) if G[r][n].get('kind') == 'tentative'
                )
                tentative.extend((r, n) for n in feeders)
                hooks_by_root.append(
                    np.fromiter(feeders, count=len(feeders), dtype=int)
                )
        else:
            hooks_by_root = [set() for _ in range(R)]
            for r, n in tentative:
                hooks_by_root[r].add(n)
            hooks_by_root = [
                np.fromiter(hooks, count=len(hooks), dtype=int)
                for hooks in hooks_by_root
            ]

        Xings = [feeder for _, feeder in gateXing_iter(G, hooks=hooks_by_root)]
        # Add also feeders whose straight line crosses constraint geometry.
        Xings.extend(
            (r, n)
            for r in range(-R, 0)
            for n in G.neighbors(r)
            if 'los_d2root' in A.nodes[n] and r in A.nodes[n]['los_d2root']
        )

        self.G, self.Xings, self.tentative, self.A = G, Xings, set(tentative), A
        if not Xings:
            # no crossings, there is no point in pathfinding
            return

        # clone2prime must be a copy of the one from Gʹ
        if C > 0:
            fnT = G.graph['fnT']
            clone2prime = fnT[T + B : -R].tolist()
        else:
            fnT = np.arange(R + T + B)
            fnT[-R:] = range(-R, 0)
            clone2prime = []
        self.fnT = fnT
        VertexC = A.graph['VertexC']
        d2roots = A.graph['d2roots']
        Rank = A.graph.get('d2rootsRank')
        diagonals = A.graph['diagonals']
        self.saved_shortened_contours = saved_shortened_contours = []
        shortened_contours = G.graph.get('shortened_contours')
        if shortened_contours is not None:
            # G has edges that shortcut some longer paths along P edges.
            # We need to put these paths back in G to flip some of P's edges.
            # The changes made here are undone in `create_detours()`.
            P_paths_shortcuts = A.graph.get('P_paths_shortcuts', {})

            edges_to_remove = []
            clone_offset = T + B
            # New clones added here are auxiliary: they are not appended to
            # `self.clone2prime` nor counted in `self.C`, so `create_detours`
            # sees the original C when it computes `clone_idx = T + B + C`.
            # They are removed from G in `create_detours` along with the
            # helper edges, before any detour clone is allocated.
            aux_clone_idx = T + B + C
            aux_clone2prime = []
            for (s, t), (midpath, shortpath) in shortened_contours.items():
                # G follows shortpath, but we want it to follow midpath
                subtree_id = G.nodes[t]['subtree']
                stored_edges = []
                u = s
                # G's contour clones follow the expanded shortpath (see
                # G_from_S), so expand here before locating clones by prime id.
                expanded_shortpath = (
                    _expand_P_paths_path([s] + shortpath + [t], P_paths_shortcuts)[1:-1]
                    if shortpath
                    else []
                )
                if expanded_shortpath:
                    # there may be more than one edge cloning the same constraint vertex
                    choices = [
                        v
                        for v in G[u]
                        if v >= clone_offset and fnT[v] == expanded_shortpath[0]
                    ]
                    if len(choices) > 1:
                        # checks just one more hop -> bizarre cases may lead to error
                        nb = (
                            t if len(expanded_shortpath) <= 1 else expanded_shortpath[1]
                        )
                        for v in choices:
                            if (G._adj[v].keys() - {u}).pop() == nb:
                                break
                    else:
                        v = choices[0]
                else:
                    v = t
                midpath = _expand_P_paths_path([s] + midpath + [t], P_paths_shortcuts)[
                    1:-1
                ]
                while v != t:
                    stored_edges.append((u, v, G[u][v]))
                    edges_to_remove.append((u, v))
                    u, v = v, (G._adj[v].keys() - {u}).pop()
                stored_edges.append((u, v, G[u][v]))
                edges_to_remove.append((u, v))
                # load is shared along the original (pre-shortening) chain
                load = stored_edges[0][2]['load']
                helper_edges = []
                helper_nodes = []
                u, u_prime = s, s
                for prime in midpath:
                    clone = aux_clone_idx
                    aux_clone_idx += 1
                    aux_clone2prime.append(prime)
                    G.add_node(clone, kind='contour', load=load, subtree=subtree_id)
                    length = float(np.hypot(*(VertexC[u_prime] - VertexC[prime])))
                    G.add_edge(
                        u,
                        clone,
                        length=length,
                        load=load,
                        kind='contour',
                        A_edge=(s, t),
                    )
                    helper_edges.append((u, clone))
                    helper_nodes.append(clone)
                    u, u_prime = clone, prime
                length = float(np.hypot(*(VertexC[u_prime] - VertexC[t])))
                G.add_edge(
                    u,
                    t,
                    length=length,
                    load=load,
                    kind='contour',
                    A_edge=(s, t),
                )
                helper_edges.append((u, t))
                saved_shortened_contours.append(
                    (stored_edges, helper_edges, helper_nodes)
                )
            G.remove_edges_from(edges_to_remove)
            # Extend fnT so `_find_paths` can map the new clone ids back to
            # their primes. `self.C` and `self.clone2prime` are intentionally
            # NOT bumped — the extension is reverted in `create_detours` by
            # removing the aux clone nodes (G.graph['fnT'] is rebuilt there).
            if aux_clone2prime:
                extended_fnT = np.empty(
                    R + T + B + C + len(aux_clone2prime), dtype=fnT.dtype
                )
                extended_fnT[: T + B + C] = fnT[: T + B + C]
                extended_fnT[T + B + C : -R] = aux_clone2prime
                extended_fnT[-R:] = range(-R, 0)
                fnT = extended_fnT
                G.graph['fnT'] = fnT
                self.fnT = fnT
        P = planar_flipped_by_routeset(
            G, planar=planar, VertexC=VertexC, diagonals=diagonals
        )
        self.d2roots = d2roots
        self.d2rootsRank = (
            Rank if Rank is not None else rankdata(d2roots, method='dense', axis=0)
        )
        self.predetour_length = Gʹ.size(weight='length')
        self.branched = branched
        self.R, self.T, self.B, self.C = R, T, B, C
        self.P, self.A, self.VertexC, self.clone2prime = P, A, VertexC, clone2prime
        self.stunts_primes = A.graph.get('stunts_primes')
        self.adv_counter = 0
        self._find_paths()

    def _trace_path(self, start_prime: int, pn_id: int):
        """Return the path and hop distances from `start_prime` to a root."""
        paths = self.paths
        path = [start_prime]
        dists = []
        pn = paths[pn_id]
        while pn_id >= 0:
            dists.append(pn.d_hop)
            pn_id = pn.parent
            path.append(paths.prime_from_pn[pn_id])
            pn = paths[pn_id]
        return path, dists

    def get_best_path(self, n: int):
        """
        `_.get_best_path(«node»)` produces a `tuple(path, dists)`.
        `path` contains a sequence of nodes from the original
        networx.Graph `G`, from «node» to the closest root.
        `dists` contains the lengths of the segments defined by `paths`.
        """
        paths = self.paths
        best_pn_by_pair_id = self.best_pn_by_pair_id
        pair_ids_by_prime = self.pair_ids_by_prime
        try:
            _, pn_id = min(
                (paths[pn_id].dist, pn_id)
                for pair_id in pair_ids_by_prime.get(n, ())
                if (pn_id := best_pn_by_pair_id[pair_id]) is not None
            )
        except ValueError:
            info('Path not found for «%d»', n)
            return [], []
        return self._trace_path(n, pn_id)

    def _scan_sector_from_opposite(self, prime: int, opposite: int) -> int:
        """Uncached sector scan for one `(prime, opposite)` pair."""
        T = self.T
        G = self.G
        P = self.P
        tentative = self.tentative
        if prime >= T:
            # `prime` is on a constraint wall or is a supertriangle vertex,
            # hence it is only reachable from one side -> arbitrary sector id
            return NULL
        if opposite in G._adj.get(prime, {}):
            # special case: visiting a DEAD-END
            return opposite
        prime_adj = G._adj.get(prime, {})
        nbr = P[prime][opposite]['ccw']
        for _ in range(len(P._adj[prime])):
            if nbr < T and nbr in prime_adj:
                if nbr >= 0 or (nbr, prime) not in tentative:
                    return nbr
            nbr = P[prime][nbr]['ccw']
        # could not find a non-tentative G edge around prime
        return NULL

    def _get_sector_from_opposite(self, prime: int, opposite: int) -> int:
        """Return the cached sector for reaching `prime` from `opposite`."""
        if prime >= self.T:
            return NULL
        try:
            return self.sector_by_prime_opposite[prime][opposite]
        except AttributeError:
            return self._scan_sector_from_opposite(prime, opposite)
        except KeyError:
            return self._scan_sector_from_opposite(prime, opposite)

    def _ensure_pair_id(self, prime: int, sector: int) -> int:
        """Return a dense id for a `(prime, sector)` best-path bucket."""
        pair = (prime, sector)
        pair_id = self.pair_id_by_prime_sector.get(pair)
        if pair_id is None:
            pair_id = len(self.best_pn_by_pair_id)
            self.pair_id_by_prime_sector[pair] = pair_id
            self.pair_ids_by_prime[prime].append(pair_id)
            self.best_pn_by_pair_id.append(None)
        return pair_id

    def _precompute_sector_lookup(self, fences: list[Fence]) -> None:
        """Precompute sector and dense `(prime, sector)` ids for pathfinding."""
        P = self.P
        T = self.T
        G = self.G
        tentative = self.tentative

        sector_by_prime_opposite: dict[int, dict[int, int]] = {}
        pair_id_by_prime_sector: dict[tuple[int, int], int] = {}
        pair_ids_by_prime: defaultdict[int, list[int]] = defaultdict(list)

        def add_pair(prime: int, sector: int) -> None:
            pair = (prime, sector)
            if pair not in pair_id_by_prime_sector:
                pair_id = len(pair_id_by_prime_sector)
                pair_id_by_prime_sector[pair] = pair_id
                pair_ids_by_prime[prime].append(pair_id)

        for prime in P:
            if prime < 0:
                add_pair(prime, prime)
            elif prime >= T:
                add_pair(prime, NULL)

        for prime in range(T):
            if prime not in P:
                add_pair(prime, NULL)
                continue
            cw_nbrs = list(P.neighbors_cw_order(prime))
            valid_sector = {
                nbr
                for nbr in cw_nbrs
                if (
                    nbr < T
                    and nbr in G._adj.get(prime, {})
                    and (nbr >= 0 or (nbr, prime) not in tentative)
                )
            }
            by_opposite: dict[int, int] = {}
            for opposite in cw_nbrs:
                if opposite in G._adj.get(prime, {}):
                    sector = opposite
                else:
                    nbr = P[prime][opposite]['ccw']
                    for _ in range(len(cw_nbrs)):
                        if nbr in valid_sector:
                            sector = nbr
                            break
                        nbr = P[prime][nbr]['ccw']
                    else:
                        sector = NULL
                by_opposite[opposite] = sector
                add_pair(prime, sector)
            add_pair(prime, NULL)
            sector_by_prime_opposite[prime] = by_opposite

        # Route-fence pseudonode buckets: each on-constraint prime visited by
        # a route fence gets a (prime, subtree_id) bucket. Chain walks add
        # pseudonodes here, keeping overlapping chains' descents in distinct
        # `best_pn_by_pair_id` slots.
        for fence in fences:
            for prime in fence.primes_on_constraint:
                add_pair(prime, fence.subtree)

        self.sector_by_prime_opposite = sector_by_prime_opposite
        self.pair_id_by_prime_sector = pair_id_by_prime_sector
        self.pair_ids_by_prime = pair_ids_by_prime

    def _advance_portal(
        self,
        adv_id: int,
        portal: tuple[int, int],
        funnel_state: tuple,
        is_triangle_seen: bitarray,
        side: int | None = None,
    ):
        P = self.P
        T = self.T
        prioqueue = self.prioqueue
        portal_set = self.portal_set
        chain_end_set = self.chain_end_set
        chain_access = self.chain_access
        traversals_limit = self.traversals_limit
        num_traversals = self.num_traversals
        triangles = P.graph['triangles']
        traverser = self._traverse_channel(adv_id, *funnel_state)
        next(traverser)
        if side is not None:
            prio, is_promising = traverser.send((portal, side))
            yield prio, portal, is_promising
            next(traverser)
            # NOTE: do NOT fire portal-side-trigger here — this branch only runs for
            # sub-advancers freshly spawned by `_spawn_exit_cone` after a
            # chain walk. The advancer's first pseudonode is parented under
            # pn_w_id (the chain-end we just exited); engaging again from
            # this single hop would re-enter the same chain stack.
        while True:
            # look for children portals
            left, right = portal
            n = P[left][right]['ccw']
            if n not in P[right] or P[left][n]['ccw'] == right or n < 0:
                debug('{%d} advancer reached DEAD-END (root or mesh edge)', adv_id)
                return
            triangle_idx = bisect_left(triangles, _sorted3(left, right, n))
            if is_triangle_seen[triangle_idx]:
                debug('{%d} advancer revisited triangle', adv_id)
                return
            is_triangle_seen[triangle_idx] = 1
            # check whether the other two sides of the triangle are portals
            portal_left = (left, n)
            portal_right = (n, right)
            has_left_portal = portal_left in portal_set
            has_right_portal = portal_right in portal_set
            if has_left_portal and has_right_portal:
                # channel bifurcation, spawn new advancer
                #  trace('{%d} advancer asking for funnel_state', adv_id)
                # get traverser state
                funnel_state = next(traverser)
                prio = funnel_state[0]
                heapq.heappush(
                    prioqueue,
                    (
                        prio,
                        self.adv_counter,
                        self._advance_portal(
                            self.adv_counter,
                            portal_right,
                            funnel_state,
                            is_triangle_seen.copy(),
                            0,
                        ),
                    ),
                )
                self.adv_counter += 1
                next(traverser)
            elif not has_left_portal and not has_right_portal:
                # DEAD-END: both triangle sides are not portals.
                # triangle-trigger: if (left, right) is a chain-entry pair
                # at chain-end `n` (i.e. the cone-at-`n` it bounds is a
                # chain interior) and the per-vertex budget hasn't been
                # consumed, engage the chain. The two phantom portal sends
                # force the funnel apex onto pn_n via a standard portal
                # advance, mirroring the apex motion an exit-side advancer
                # will need on the partner side.
                # Non-chain wedges (the void across the constraint, and
                # navigable-but-not-chain "outer" wedges) are not in
                # chain_access, so the trigger silently no-ops here and
                # leaves the budget intact for an advancer that does
                # arrive via a chain entry.
                access = (
                    chain_access.get((n, left, right)) if n in chain_end_set else None
                )
                if access is not None and num_traversals[(n, n)] < traversals_limit:
                    chain, c_side = access
                    traverser.send(((left, n), 1))
                    next(traverser)
                    traverser.send(((n, right), 0))
                    next(traverser)
                    num_traversals[(n, n)] += 1
                    self._walk_chain(
                        n,
                        chain,
                        c_side,
                        self.paths.last_added_pn,
                        is_triangle_seen,
                    )
                elif 0 <= n < T:
                    prio, is_promising = traverser.send(((left, n), 1))
                    next(traverser)
                debug('{%d} advancer reached DEAD-END (not portals)', adv_id)
                return
            # process  portal
            if has_left_portal:
                portal, side = portal_left, 1
            else:
                portal, side = portal_right, 0
            prio, is_promising = traverser.send((portal, side))
            yield prio, portal, is_promising
            next(traverser)
            # portal-side-trigger: the portal-advance next step y=n is a chain-end.
            # Engage the chain whose cone-at-y is bounded by the parent
            # portal's pair (left, right). Reuse the parent's portal-advance
            # pseudonode for y as the chain entry (an additional send would
            # produce a same-prime self-link). chain_access miss == "this
            # parent-portal pair is not a chain entry," so the trigger
            # no-ops and the budget is preserved.
            y = portal[side]
            access = chain_access.get((y, left, right)) if y in chain_end_set else None
            if access is not None and num_traversals[(y, y)] < traversals_limit:
                chain, c_side = access
                num_traversals[(y, y)] += 1
                self._walk_chain(
                    y, chain, c_side, self.paths.last_added_pn, is_triangle_seen
                )

    def _walk_chain(
        self,
        y_entry: int,
        chain: Chain,
        side: int,
        entry_pn: int,
        is_triangle_seen: bitarray,
    ) -> None:
        """Walk `chain` from its `cones[side]` access cone (entered via
        the trigger that called us) to its `cones[1 - side]` cone,
        creating chain-walk pseudonodes along the way and spawning exit
        advancers from the partner cone.

        `entry_pn` is the pseudonode for `y_entry` (the chain-end the
        funnel just landed on); for spanning chains `y_entry ==
        chain.cones[side].vertex`, for single-vertex chains both cones
        share that vertex.
        """
        walk = chain.walks[side]
        exit_cone = chain.cones[1 - side]

        paths = self.paths
        VertexC = self.VertexC
        pair_id_by_prime_sector = self.pair_id_by_prime_sector
        best_pn_by_pair_id = self.best_pn_by_pair_id

        cur = y_entry
        parent_pn = entry_pn
        for c_next in walk:
            d_hop = _node_dist(VertexC, cur, c_next)
            pn_parent = paths[parent_pn]
            d_total = pn_parent.dist + d_hop
            parent_pn = paths.add(
                c_next, chain.subtree, parent_pn, d_total, d_hop, pn_parent.cum_turn
            )
            pair_id = pair_id_by_prime_sector[(c_next, chain.subtree)]
            best_pn_id = best_pn_by_pair_id[pair_id]
            if best_pn_id is None or d_total < paths[best_pn_id].dist:
                best_pn_by_pair_id[pair_id] = parent_pn
            cur = c_next

        self._spawn_exit_cone(exit_cone, parent_pn, is_triangle_seen)

    def _extract_fences(self) -> list[Fence]:
        """Collect route fences used in G, one per unique A-edge contour.

        Each route fence corresponds to an A-edge whose `midpath` attribute
        lists the constraint vertices it routes through. Shortened contours
        hold their (longer) midpath on `G.graph['shortened_contours']` keyed
        by the same A-edge; the regular case reads `midpath` from
        `self.A[s][t]`. Only fences with a non-empty on-constraint segment
        are emitted.

        Both midpath sources are stored in P_paths shortcut form (the
        bidirectional_dijkstra path on `P_paths`). They are expanded here so
        consecutive entries of `[s, *primes_on_constraint, t]` are P-edges,
        matching the expansion already applied in `G_from_S` (regular
        contours) and earlier in this `__init__` (shortened contours) — the
        expanded edges are precisely what drove `planar_flipped_by_routeset`,
        so the result is consistent with `self.P`.
        """
        G = self.G
        A = self.A
        shortcuts = A.graph.get('P_paths_shortcuts', {})
        fences: list[Fence] = []
        shortened = G.graph.get('shortened_contours') or {}

        # Fence endpoints (s, t) are tree members of S — they're the endpoints
        # of a cable edge — so both carry the 'subtree' attribute. Root-
        # endpoint A-edges with midpath are routed to kind='tentative' by
        # G_from_S and never reach this loop.
        # Shortened contours (PathFinder restored their midpaths into G).
        for endpoints, (midpath, _) in shortened.items():
            if midpath:
                expanded = _expand_P_paths_path(
                    [endpoints[0], *midpath, endpoints[1]], shortcuts
                )[1:-1]
                fences.append(
                    Fence(endpoints, expanded, G.nodes[endpoints[1]]['subtree'])
                )

        # Clone-based contours: dedup by A_edge attribute on contour edges.
        seen: set[tuple[int, int]] = set(shortened)
        for _, _, d in G.edges(data=True):
            if d.get('kind') != 'contour':
                continue
            ae = d.get('A_edge')
            if ae is None or ae in seen:
                continue
            seen.add(ae)
            midpath = A[ae[0]][ae[1]].get('midpath') or []
            if midpath:
                expanded = _expand_P_paths_path([ae[0], *midpath, ae[1]], shortcuts)[
                    1:-1
                ]
                fences.append(Fence(ae, expanded, G.nodes[ae[1]]['subtree']))
        return fences

    def _partition_into_cones(
        self, c: int, cone_bounds: set[int], rotated: list[int]
    ) -> list[tuple[int, int, list[int], list[tuple[int, int]]]]:
        """Partition `c`'s cyclic neighbors `rotated` into wedges between
        consecutive members of `cone_bounds`. For each wedge, return
        `(left, right, spokes, pair_keys)` where `pair_keys` lists the
        `(a, b)` pairs of consecutive cyclic-neighbors of `c` that fall
        inside the wedge — these are the `(left, right)` lookup keys an
        advancer crossing into `c` from a parent triangle on this wedge's
        side will present.
        """
        n_cw = len(rotated)
        bound_positions = [i for i, nb in enumerate(rotated) if nb in cone_bounds]
        n_cones = len(bound_positions)
        out: list[tuple[int, int, list[int], list[tuple[int, int]]]] = []
        for k in range(n_cones):
            bpos = bound_positions[k]
            nbpos = bound_positions[(k + 1) % n_cones]
            left, right = rotated[bpos], rotated[nbpos]
            spokes: list[int] = []
            pair_keys: list[tuple[int, int]] = []
            prev_nb = left
            cur = (bpos + 1) % n_cw
            while cur != nbpos:
                v = rotated[cur]
                if v >= 0 and v not in cone_bounds:
                    spokes.append(v)
                pair_keys.append((prev_nb, v))
                prev_nb = v
                cur = (cur + 1) % n_cw
            pair_keys.append((prev_nb, right))
            out.append((left, right, spokes, pair_keys))
        return out

    def _build_touching_chains_at(
        self, v: int, touching: list[Fence]
    ) -> list[tuple[Chain, list[tuple[int, int]], list[tuple[int, int]]]] | None:
        """Build the chains hosted at single-vertex chain-end `v`. Each
        touching route fence merely passes through `v`, contributing two
        cone bounds (one per fence-wall incident to `v`); with the
        constraint's two cone bounds, this gives 2 + 2*n_fences cone
        bounds total. The fence stack at `v` produces n_fences chains
        (constraint paired with innermost, plus each cyclically-adjacent
        fence pair in the stack); each chain owns two access cones at
        `v`, one on each cyclic side of the fence stack. Empty walks —
        the chain is single-vertex.

        Returns: list of (Chain, pair_keys_a, pair_keys_b) per chain at
        `v`. The caller registers `chain_access` for both cones using
        these pair keys.
        """
        constraint_bounds = self.constraint_bounds.get(v, set())
        if len(constraint_bounds) != 2:
            error(
                'expected 2 constraint bounds at single-vertex chain %d, got %d',
                v,
                len(constraint_bounds),
            )
            return None
        # Map each fence wall to its owner subtree. Reject if two fences
        # share a wall at `v` — the cyclic-stack notion of nesting would
        # not be well-defined.
        wall_to_subtree: dict[int, int] = {}
        for fence in touching:
            for endpoint in fence.endpoints:
                if endpoint in wall_to_subtree:
                    error(
                        'touching fences share wall %d at chain-end %d',
                        endpoint,
                        v,
                    )
                    return None
                wall_to_subtree[endpoint] = fence.subtree
        cone_bounds = set(wall_to_subtree) | constraint_bounds
        cw_nbrs = list(self.P.neighbors_cw_order(v))
        cones = self._partition_into_cones(v, cone_bounds, cw_nbrs)
        # Sentinel id distinct from any subtree (subtrees are >= 0).
        C_ID = -1
        bound_id: dict[int, int] = {cb: C_ID for cb in constraint_bounds}
        bound_id.update(wall_to_subtree)
        # Collect chain-interior cones — those whose two bounds belong to
        # distinct fences (where the constraint counts as a fence). Wedges
        # bounded by two walls of the same fence (the "outer" wedge of the
        # stack, navigable but not chain-related) and wedges bounded by
        # two constraint walls (the void on the far side of the constraint
        # border) are skipped: the chain mechanism has no business there.
        chain_cones: dict[tuple[int, int], list[int]] = defaultdict(list)
        for k, (lb, rb, _, _) in enumerate(cones):
            a, b = bound_id[lb], bound_id[rb]
            if a == b:
                continue
            chain_cones[(a, b) if a < b else (b, a)].append(k)
        # Walk the navigable arc between the two constraint walls (the arc
        # that contains every fence wall) to recover the fence stack from
        # innermost (closest to constraint) to outermost.
        nbr_pos = {nb: i for i, nb in enumerate(cw_nbrs)}
        cb_a, cb_b = constraint_bounds
        n_cw = len(cw_nbrs)

        def fence_walls_in_arc(direction: int) -> list[int]:
            walls: list[int] = []
            i = (nbr_pos[cb_a] + direction) % n_cw
            while i != nbr_pos[cb_b]:
                nb = cw_nbrs[i]
                if nb in wall_to_subtree:
                    walls.append(nb)
                i = (i + direction) % n_cw
            return walls

        walls_cw = fence_walls_in_arc(+1)
        walls_ccw = fence_walls_in_arc(-1)
        nav_walls = walls_cw if len(walls_cw) >= len(walls_ccw) else walls_ccw
        fence_seq = [wall_to_subtree[w] for w in nav_walls]
        if len(fence_seq) != 2 * len(touching):
            error(
                'fence walls (%d) on navigable arc at %d do not match touching fences (%d)',
                len(fence_seq),
                v,
                len(touching),
            )
            return None
        half = len(fence_seq) // 2
        if fence_seq[half:] != fence_seq[:half][::-1]:
            error('touching fences not nested at chain-end %d: %s', v, fence_seq)
            return None
        stack = fence_seq[:half]
        if len(set(stack)) != len(stack):
            error('repeated subtree id in fence stack at %d: %s', v, stack)
            return None
        # Expected chains: constraint paired with the innermost fence, plus
        # each cyclically-adjacent fence pair in the stack.
        stack_pos = {sub: i for i, sub in enumerate(stack)}
        expected_chains: set[tuple[int, int]] = set()
        innermost = stack[0]
        expected_chains.add(
            (C_ID, innermost) if C_ID < innermost else (innermost, C_ID)
        )
        for inner, outer in zip(stack, stack[1:]):
            expected_chains.add((inner, outer) if inner < outer else (outer, inner))
        if set(chain_cones) != expected_chains:
            error(
                'chain cones mismatch at %d: have %s, expected %s',
                v,
                set(chain_cones),
                expected_chains,
            )
            return None
        # Build a Chain per chain-interior pair. Owner subtree convention:
        # the route fence (when one bound belongs to the constraint), or —
        # for chains between two route fences — the one farther from the
        # constraint in the stack (matches the spanning multi-fence
        # convention so each route fence labels exactly one chain).
        out: list[tuple[Chain, list[tuple[int, int]], list[tuple[int, int]]]] = []
        for chain_key, cone_indices in chain_cones.items():
            if len(cone_indices) != 2:
                error(
                    'chain %s at %d has %d cones (expected 2)',
                    chain_key,
                    v,
                    len(cone_indices),
                )
                return None
            a, b = chain_key
            if a == C_ID:
                owner = b
            elif b == C_ID:
                owner = a
            else:
                owner = a if stack_pos[a] > stack_pos[b] else b
            ki, kj = cone_indices
            li, ri, si, ki_pairs = cones[ki]
            lj, rj, sj, kj_pairs = cones[kj]
            cone_i = AccessCone(v, li, ri, si)
            cone_j = AccessCone(v, lj, rj, sj)
            chain = Chain(owner, (cone_i, cone_j), ([], []))
            out.append((chain, ki_pairs, kj_pairs))
        return out

    def _spanning_access_cones_at(
        self, c: int, endings: list[tuple[Fence, str]]
    ) -> list[tuple[int, AccessCone, list[tuple[int, int]], list[int]]] | None:
        """Build access cones at chain-end `c` for one or more spanning
        fences. Each fence at `c` owns one chain (one cone here, paired
        across-the-mp by the caller); the constraint fence sits as the
        innermost member of the cyclic stack. With n route fences in
        the stack there are n_fences = n + 1 wedge bounds and n chains
        — each between two cyclically-adjacent stack members.

        Returns: list of `(subtree, AccessCone, pair_keys, fence_mp)`,
        one per chain owned by a route fence at `c`. `fence_mp` is the
        owning fence's full `primes_on_constraint`, used by the caller
        to assemble walks.
        """
        P = self.P
        constraint_bounds = self.constraint_bounds

        f0, side0 = endings[0]
        chain_step_nbr = (
            f0.primes_on_constraint[1]
            if side0 == 'start'
            else f0.primes_on_constraint[-2]
        )
        for fence, side in endings[1:]:
            other = (
                fence.primes_on_constraint[1]
                if side == 'start'
                else fence.primes_on_constraint[-2]
            )
            if other != chain_step_nbr:
                error(
                    'fences disagree on chain step at %d: %d vs %d',
                    c,
                    chain_step_nbr,
                    other,
                )

        cw_nbrs = list(P.neighbors_cw_order(c))
        n_cw = len(cw_nbrs)
        nbr_pos = {nb: i for i, nb in enumerate(cw_nbrs)}
        chain_step_pos = nbr_pos[chain_step_nbr]

        candidates = constraint_bounds.get(c, set()) - {chain_step_nbr}
        if len(candidates) != 1:
            error('expected exactly 1 constraint bound at %d, got %s', c, candidates)
            return None
        constraint_bound = next(iter(candidates))

        # Pick cyclic direction so constraint_bound is closest to chain_step_nbr.
        # This keeps the fence stack's "outer" direction consistent with the
        # partner end — needed when multi-fence stacks (mirrored chains, e.g.
        # RID 26654) want to agree on inner-vs-outer at both ends.
        cw_offset = (nbr_pos[constraint_bound] - chain_step_pos) % n_cw
        ccw_offset = (chain_step_pos - nbr_pos[constraint_bound]) % n_cw
        cw_direction = cw_offset <= ccw_offset

        if cw_direction:
            rotated = [cw_nbrs[(chain_step_pos + i) % n_cw] for i in range(n_cw)]
        else:
            rotated = [chain_step_nbr] + [
                cw_nbrs[(chain_step_pos - i) % n_cw] for i in range(1, n_cw)
            ]
        rotated_pos = {nb: i for i, nb in enumerate(rotated)}

        # Per-fence anchor wall at c, in stack order. The constraint fence is
        # represented by `constraint_bound` (chain_step_nbr is the rotation
        # anchor and is appended to cone_bounds separately). Each route fence
        # is represented by its off-constraint endpoint at `c`. Sorted by
        # position in `rotated`, the list runs from innermost to outermost.
        outer_bounds_by_fence: list[int] = [constraint_bound]
        for fence, side in endings:
            bound = fence.endpoints[0] if side == 'start' else fence.endpoints[1]
            outer_bounds_by_fence.append(bound)
        outer_bounds_by_fence.sort(key=rotated_pos.__getitem__)

        cones = self._partition_into_cones(
            c, {chain_step_nbr, *outer_bounds_by_fence}, rotated
        )

        n_fences = len(outer_bounds_by_fence)
        if n_fences < 2:
            return []

        # Cones in cyclic order from chain_step_nbr (in `rotated` direction):
        #   cones[0]: void wedge between chain_step_nbr and constraint_bound
        #     — across the constraint border, not navigable. Skipped.
        #   cones[1..n_fences-1]: chain interiors, one per stacked fence pair.
        #   cones[n_fences]: navigable-but-non-chain wedge between the
        #     outermost fence wall and chain_step_nbr (the cyclic-other side).
        #     Skipped — an advancer entering it has no chain to walk.
        out: list[tuple[int, AccessCone, list[tuple[int, int]], list[int]]] = []
        for k in range(n_fences - 1):
            # Chain at cones[k+1] is bounded by outer_bounds_by_fence[k]
            # (inner) and outer_bounds_by_fence[k+1] (outer). Owner is
            # the route fence whose endpoint at `c` matches the outer
            # bound — by the cw_direction selection above, constraint_bound
            # always sorts to outer_bounds_by_fence[0], so the outer bound
            # of any chain (k+1 >= 1) is a route-fence endpoint.
            ref_bound = outer_bounds_by_fence[k + 1]
            ref_ending = next(
                (
                    (fence, side)
                    for fence, side in endings
                    if (fence.endpoints[0] if side == 'start' else fence.endpoints[1])
                    == ref_bound
                ),
                None,
            )
            if ref_ending is None:
                error('no fence matches cone bound %d at %d', ref_bound, c)
                continue
            ref_fence, _ = ref_ending

            left, right, spokes, pair_keys = cones[k + 1]
            if not cw_direction:
                # Canonicalize bounds to CW around `c`. Consumers
                # (`_spawn_exit_cone`, `_advance_portal`) assume CW: the
                # navigable interior of an exit triangle is on the CCW
                # side of (left, right), and `P[left][right]['ccw']` gives
                # the next advance vertex. When `rotated` traverses CCW,
                # the partition emits bounds + spokes in CCW order; flip
                # them here so the AccessCone records the canonical CW
                # ordering.
                left, right = right, left
                spokes = spokes[::-1]
            cone = AccessCone(c, left, right, spokes)
            out.append(
                (
                    ref_fence.subtree,
                    cone,
                    pair_keys,
                    list(ref_fence.primes_on_constraint),
                )
            )
        return out

    def _precompute_chains(
        self, fences: list[Fence]
    ) -> tuple[dict[tuple[int, int, int], tuple[Chain, int]], set[int]]:
        """Build the chain topology from the route fences.

        Spanning fences (on-constraint segment length >= 2) contribute one
        access cone at each end; the two are paired (matched by subtree)
        into one Chain. Touching fences (length 1) contribute two access
        cones at a single chain-end (one per cyclic side of the fence
        stack), already paired locally into a Chain by
        `_build_touching_chains_at`.

        When a spanning fence has only one end with the next-mp vertex on
        a constraint edge, the other end is demoted to touching and gets
        a synthetic single-vertex chain. The full-mp spanning chain is
        wired separately to one of the demoted end's touching cones (the
        first by index, matching the prior `cone_subtree.index(subtree)`
        behavior). The touching chain at the demoted end "owns" entries
        at its cones — we register the spanning chain's far-side cone
        first and let the touching chain overwrite, so triggers entering
        the demoted-end cones dispatch to the local touching chain rather
        than walking back across the full mp.

        Returns:
          chain_access: dict[(vertex, left, right) -> (Chain, side)]
            Both pair orientations registered; lookup miss == "not a
            chain entry" — the trigger then does nothing and consumes no
            traversal budget.
          chain_end_set: set[int]  # vertices hosting any access cone.
        """
        constraint_bounds = self.constraint_bounds

        spanning_at: dict[int, list[tuple[Fence, str]]] = defaultdict(list)
        touching_at: dict[int, list[Fence]] = defaultdict(list)
        for fence in fences:
            mp = fence.primes_on_constraint
            if len(mp) >= 2:
                # The spanning model assumes the fence walks contiguously
                # through `mp` along constraint edges at each end, so each
                # chain-step neighbor must be a constraint neighbor of its
                # chain-end. When P_paths' shortest path leaves the
                # constraint via a non-constraint mesh edge (e.g. a
                # triangulation diagonal cutting across an obstacle), the
                # affected end cannot host spanning topology. Demote that
                # end alone to a touching encounter with synthesized walls
                # = the off-constraint endpoint and the next mp vertex.
                s, t = fence.endpoints
                start_ok = mp[1] in constraint_bounds.get(mp[0], set())
                end_ok = mp[-2] in constraint_bounds.get(mp[-1], set())
                if start_ok:
                    spanning_at[mp[0]].append((fence, 'start'))
                else:
                    touching_at[mp[0]].append(Fence((s, mp[1]), [mp[0]], fence.subtree))
                if end_ok:
                    spanning_at[mp[-1]].append((fence, 'end'))
                else:
                    touching_at[mp[-1]].append(
                        Fence((mp[-2], t), [mp[-1]], fence.subtree)
                    )
            else:
                touching_at[mp[0]].append(fence)

        chain_access: dict[tuple[int, int, int], tuple[Chain, int]] = {}
        chain_end_set: set[int] = set()

        def register(
            v: int, pair_keys: list[tuple[int, int]], chain: Chain, side: int
        ) -> None:
            for pa, pb in pair_keys:
                chain_access[(v, pa, pb)] = (chain, side)
                chain_access[(v, pb, pa)] = (chain, side)

        # Build spanning access cones per chain-end vertex.
        spanning_by_subtree: dict[
            int, list[tuple[int, AccessCone, list[tuple[int, int]], list[int]]]
        ] = defaultdict(list)
        for c, endings in spanning_at.items():
            result = self._spanning_access_cones_at(c, endings)
            if result is None:
                continue
            chain_end_set.add(c)
            for subtree, cone, pair_keys, mp in result:
                spanning_by_subtree[subtree].append((c, cone, pair_keys, mp))

        # Pair fully-spanning chains (both ends OK -> 2 entries per subtree).
        # Demoted-one-end fences contribute only 1 entry and are deferred.
        deferred_spanning: dict[
            int, tuple[int, AccessCone, list[tuple[int, int]], list[int]]
        ] = {}
        for subtree, entries in spanning_by_subtree.items():
            if len(entries) == 2:
                (c0, cone0, keys0, mp), (c1, cone1, keys1, _) = entries
                if cone0.vertex == mp[0] and cone1.vertex == mp[-1]:
                    walk_0, walk_1 = list(mp[1:]), list(mp[-2::-1])
                elif cone0.vertex == mp[-1] and cone1.vertex == mp[0]:
                    walk_0, walk_1 = list(mp[-2::-1]), list(mp[1:])
                else:
                    error(
                        'spanning chain %d: cones at %d, %d do not match mp ends',
                        subtree,
                        cone0.vertex,
                        cone1.vertex,
                    )
                    continue
                chain = Chain(subtree, (cone0, cone1), (walk_0, walk_1))
                register(c0, keys0, chain, 0)
                register(c1, keys1, chain, 1)
            elif len(entries) == 1:
                deferred_spanning[subtree] = entries[0]
            else:
                error(
                    'spanning chain subtree %d has %d access cones (expected 1 or 2)',
                    subtree,
                    len(entries),
                )

        # Build touching chains. Track each (vertex, subtree) -> [Chain, ...]
        # for wiring deferred spanning chains.
        touching_chains_by_vertex_subtree: dict[tuple[int, int], list[Chain]] = (
            defaultdict(list)
        )
        for v, touching in touching_at.items():
            if v in spanning_at:
                error('mixed spanning + touching fences at %d not supported', v)
                continue
            result = self._build_touching_chains_at(v, touching)
            if result is None:
                continue
            chain_end_set.add(v)
            for chain, keys_a, keys_b in result:
                touching_chains_by_vertex_subtree[(v, chain.subtree)].append(chain)
                # Touching chains register AFTER deferred spanning chains
                # below, so their entries take precedence at any cone shared
                # with a deferred chain (the demoted-end corner case).
                register(v, keys_a, chain, 0)
                register(v, keys_b, chain, 1)

        # Wire deferred (one-end-demoted) spanning chains to the demoted
        # end's touching counterpart.
        for subtree, (c, cone, pair_keys, mp) in deferred_spanning.items():
            demoted_end = mp[-1] if cone.vertex == mp[0] else mp[0]
            partners = touching_chains_by_vertex_subtree.get((demoted_end, subtree), [])
            if len(partners) != 1:
                error(
                    'demoted spanning chain %d at %d: expected 1 touching '
                    'partner at %d, found %d',
                    subtree,
                    cone.vertex,
                    demoted_end,
                    len(partners),
                )
                continue
            # Pair with the touching chain's first cone — preserves the
            # historic `cone_subtree.index(subtree)` choice. The other
            # touching cone reaches the spanning side only via this one.
            partner_cone = partners[0].cones[0]
            if cone.vertex == mp[0]:
                walk_0, walk_1 = list(mp[1:]), list(mp[-2::-1])
            else:
                walk_0, walk_1 = list(mp[-2::-1]), list(mp[1:])
            chain = Chain(subtree, (cone, partner_cone), (walk_0, walk_1))
            # Register only the spanning-side cone for entry. The
            # demoted-end (partner) cone's entries already belong to the
            # local touching chain (registered above) — overwriting them
            # would route incoming touching-cone advancers back across the
            # full mp, contradicting the touching semantics there.
            register(c, pair_keys, chain, 0)

        return chain_access, chain_end_set

    def _spawn_exit_cone(
        self,
        cone: AccessCone,
        pn_w_id: int,
        is_triangle_seen: bitarray,
    ) -> None:
        """Spawn end-spoke and intermediate-pair advancers covering exit
        through `cone`. `cone.left` and `cone.right` are the wall-neighbor
        primes delimiting the wedge in CW order around `cone.vertex`;
        `cone.spokes` are the non-bound spokes inside.
        """
        P = self.P
        paths = self.paths
        prioqueue = self.prioqueue
        portal_set = self.portal_set
        VertexC = self.VertexC
        best_pn_by_pair_id = self.best_pn_by_pair_id
        pair_id_by_prime_sector = self.pair_id_by_prime_sector
        ensure_pair_id = self._ensure_pair_id
        w = cone.vertex
        pn_w = paths[pn_w_id]
        cum_turn_w = pn_w.cum_turn

        def _add_cone_exit_pn(v: int) -> tuple[int, float]:
            """Pseudonode at `v` parented by pn_w; returns (pn_id, d_hop)."""
            if v == w:
                return pn_w_id, 0.0
            d_hop = _node_dist(VertexC, w, v)
            d_total = pn_w.dist + d_hop
            sec_v = self._get_sector_from_opposite(v, w) if v >= 0 else NULL
            pn_v = paths.add(v, sec_v, pn_w_id, d_total, d_hop, cum_turn_w)
            pair_id = pair_id_by_prime_sector.get((v, sec_v))
            if pair_id is None:
                pair_id = ensure_pair_id(v, sec_v)
            best_pn_id = best_pn_by_pair_id[pair_id]
            if best_pn_id is None or d_total < paths[best_pn_id].dist:
                best_pn_by_pair_id[pair_id] = pn_v
            return pn_v, d_hop

        def _launch(left: int, right: int, side_init: int) -> None:
            wl, d_hop_left = _add_cone_exit_pn(left)
            wr, d_hop_right = _add_cone_exit_pn(right)
            hops = [h for h in (d_hop_left, d_hop_right) if h > 0]
            d_hop_min = min(hops) if hops else 0.0
            sub_prio = (pn_w.dist + d_hop_min, 0.0, 1.0)
            funnel_state = (sub_prio, w, pn_w_id, [left, right], [wl, wr], 0)
            sub_advancer = self._advance_portal(
                self.adv_counter,
                (left, right),
                funnel_state,
                is_triangle_seen.copy(),
                side_init,
            )
            heapq.heappush(prioqueue, (sub_prio, self.adv_counter, sub_advancer))
            self.adv_counter += 1

        spokes = cone.spokes
        if spokes:
            x_1, x_k = spokes[0], spokes[-1]
            _launch(w, x_1, 1)
            _launch(x_k, w, 0)
            for xi, xj in zip(spokes, spokes[1:]):
                if (xi, xj) in portal_set:
                    _launch(xi, xj, 1)
        else:
            # Single-triangle exit: only the connecting portal between the
            # two cone-bounding wall-neighbors. Skip if it would re-engage the
            # same chain-end (third vertex of the new triangle is w).
            if (
                (cone.left, cone.right) in portal_set
                and cone.right in P[cone.left]
                and P[cone.left][cone.right].get('ccw') != w
            ):
                _launch(cone.left, cone.right, 1)

    def _chain_end_sector(self, y: int, opposite: int) -> int:
        """Sector for a portal-side-trigger narrowing onto chain-end `y`
        across portal `(y, opposite)`. The cone at `y` the funnel just
        left is the wedge adjacent to `opposite` on the parent-triangle
        side; we resolve it by checking the two cones cyclically adjacent
        to `opposite` at `y` and returning the chain's subtree if exactly
        one of them is a chain interior. Returns NULL for non-chain cones
        or when both adjacent cones are chain interiors of different
        chains (overlapping fences: `opposite` alone can't disambiguate).
        """
        if y not in self.chain_end_set:
            return NULL
        P_y = self.P[y]
        if opposite not in P_y:
            return NULL
        edge = P_y[opposite]
        chain_access = self.chain_access
        access_cw = chain_access.get((y, opposite, edge['cw']))
        access_ccw = chain_access.get((y, opposite, edge['ccw']))
        sub_cw = access_cw[0].subtree if access_cw is not None else None
        sub_ccw = access_ccw[0].subtree if access_ccw is not None else None
        if sub_cw == sub_ccw:
            return sub_cw if sub_cw is not None else NULL
        if sub_cw is None:
            return sub_ccw
        if sub_ccw is None:
            return sub_cw
        # overlapping fences: the two adjacent cones host different chains and
        # `opposite` alone cannot disambiguate
        return NULL

    def _traverse_channel(
        self,
        adv_id,
        prio: tuple,
        _apex: int,
        apex: int,
        _funnel: list[int],
        wedge_end: list[int],
        bad_streak: int = 0,
    ):
        # variable naming notation:
        # for variables that represent a node, they may occur in two versions:
        #     - _node: the index it contains maps to a coordinate in VertexC
        #     - pn_id: pseudonode index in self.paths
        #             translation: _node = paths.prime_from_pn[pn_id]
        cw, ccw, cross = rotation_checkers_factory(self.VertexC)
        # Tolerance for treating a numerically-zero cross product as collinear:
        # apex/wall/_new line-of-sight should not flip funnel branches due to
        # float-arithmetic noise.
        EPS_COLLINEAR = 1e-17

        paths = self.paths
        best_pn_by_pair_id = self.best_pn_by_pair_id
        pair_id_by_prime_sector = self.pair_id_by_prime_sector
        sector_by_prime_opposite = self.sector_by_prime_opposite
        ensure_pair_id = self._ensure_pair_id
        scan_sector = self._scan_sector_from_opposite
        chain_end_set = self.chain_end_set
        chain_end_sector = self._chain_end_sector
        ST = self.ST
        T = self.T
        num_traversals = self.num_traversals
        bad_streak_limit = self.bad_streak_limit
        turn_limit = self.turn_limit

        # for next_left, next_right, new_portal_iter in portal_iter:
        while True:
            #  trace('<%d> traverser before first yield', adv_id)
            portal_step = yield
            if portal_step is None:
                #  trace('<%d> new traverser sent for evaluation', adv_id)
                yield (
                    prio,
                    _apex,
                    apex,
                    _funnel.copy(),
                    wedge_end.copy(),
                    bad_streak,
                )
                continue
            portal, side = portal_step
            #  trace('<%d> got (portal, side)', adv_id)

            _new = portal[side]
            opposite = portal[1 - side]
            if 0 <= _new < T:
                try:
                    sector_new = sector_by_prime_opposite[_new][opposite]
                except KeyError:
                    sector_new = scan_sector(_new, opposite)
            elif _new in chain_end_set:
                sector_new = chain_end_sector(_new, opposite)
            else:
                sector_new = NULL
            pair_id = pair_id_by_prime_sector.get((_new, sector_new))
            if pair_id is None:
                pair_id = ensure_pair_id(_new, sector_new)
            _nearside = _funnel[side]
            _farside = _funnel[not side]
            test = ccw if side else cw
            # Sign that turns "cross < 0" (cw) into the test for this side.
            # side==0: test=cw  → orient = cross
            # side==1: test=ccw → orient = -cross
            # so orient < 0 ⇔ test passes; |orient| < ε ⇔ collinear.
            orient_sign = -1.0 if side else 1.0

            #  if _nearside == _apex:  # debug info
            #      print(f"{'RIGHT' if side else 'LEFT '} "
            #            f'nearside({_nearside}) == apex({_apex})')
            debug(
                '<%d> %s _new(%d) _nearside(%d) _farside(%d) _apex(%d), _wedge_end: %d %d, _funnel: %s',
                adv_id,
                'RIGHT' if side else 'LEFT ',
                _new,
                _nearside,
                _farside,
                _apex,
                paths.prime_from_pn[wedge_end[0]],
                paths.prime_from_pn[wedge_end[1]],
                _funnel,
            )

            # One signed cross per wall; ε folds collinearity into the same
            # comparison: "test or collinear" ⇔ orient < ε,
            # "test and not collinear" ⇔ orient < -ε.
            orient_near = orient_sign * cross(_nearside, _new, _apex)
            orient_far = orient_sign * cross(_farside, _new, _apex)

            if _nearside == _apex or orient_near < EPS_COLLINEAR:
                # not infranear (collinear with apex→nearside is treated as
                # line-of-sight: _new lies on the wall, apex stays put)
                if orient_far < -EPS_COLLINEAR:
                    # ultrafar (⟨new, apex⟩ strictly cuts farside; collinear
                    # with apex→farside is line-of-sight, apex stays put)
                    debug('<%d> ultrafar', adv_id)
                    current_wapex = wedge_end[not side]
                    _current_wapex = paths.prime_from_pn[current_wapex]
                    _funnel[not side] = _current_wapex
                    contender_wapex = paths[current_wapex].parent
                    _contender_wapex = paths.prime_from_pn[contender_wapex]
                    # Walk the wapex toward the farside wall while the test
                    # predicate selects the contender. The `== _new` clause
                    # forces one more step whenever the wapex sits on a
                    # prime equal to `_new` (chain-anchor case):
                    # cross(_new, _new, contender) = 0 makes `test` false,
                    # so without the override the loop would exit with the
                    # wapex coincident with `_new`; paths.add would then
                    # parent the new pseudonode for `_new` under another
                    # pseudonode for the same prime — a self-link.
                    while (
                        _current_wapex != _farside
                        and _contender_wapex >= 0
                        and (
                            _current_wapex == _new
                            or test(_new, _current_wapex, _contender_wapex)
                        )
                    ):
                        _funnel[not side] = _current_wapex
                        current_wapex = contender_wapex
                        _current_wapex = _contender_wapex
                        contender_wapex = paths[current_wapex].parent
                        _contender_wapex = paths.prime_from_pn[contender_wapex]
                    _apex = _current_wapex
                    apex = current_wapex
                else:
                    # not ultrafar nor infranear (⟨new, apex⟩ in line-of-sight)
                    debug('<%d> inside', adv_id)
                _apex_eff, apex_eff = _apex, apex
                _funnel[side] = _new
            else:
                # infranear (⟨new, apex⟩ cuts nearside)
                debug('<%d> infranear', adv_id)
                current_wapex = wedge_end[side]
                _current_wapex = paths.prime_from_pn[current_wapex]
                contender_wapex = paths[current_wapex].parent
                _contender_wapex = paths.prime_from_pn[contender_wapex]
                # See ULTRAFAR loop: `== _new` forces one more step past a
                # chain-anchor where the wapex would otherwise sit
                # coincident with `_new`.
                while (
                    _current_wapex != _nearside
                    and _contender_wapex >= 0
                    and (
                        _current_wapex == _new
                        or test(_current_wapex, _new, _contender_wapex)
                    )
                ):
                    current_wapex = contender_wapex
                    _current_wapex = _contender_wapex
                    contender_wapex = paths[current_wapex].parent
                    _contender_wapex = paths.prime_from_pn[contender_wapex]
                _apex_eff, apex_eff = _current_wapex, current_wapex

            # rate, wait, add
            d_hop = _node_dist(self.VertexC, _apex_eff, _new)
            apex_pn = paths[apex_eff]
            d_new = apex_pn.dist + d_hop
            best_pn_id = best_pn_by_pair_id[pair_id]
            unseen = best_pn_id is None
            # signed turn at apex_eff: angle from (grandparent -> apex_eff)
            # segment to (apex_eff -> _new) segment.
            gp_pn_id = apex_pn.parent
            if gp_pn_id is None:
                step_turn = 0.0
            else:
                _gp = paths.prime_from_pn[gp_pn_id]
                ax = self.VertexC[_apex_eff]
                gp = self.VertexC[_gp]
                nv = self.VertexC[_new]
                v1x, v1y = ax[0] - gp[0], ax[1] - gp[1]
                v2x, v2y = nv[0] - ax[0], nv[1] - ax[1]
                step_turn = math.atan2(v1x * v2y - v1y * v2x, v1x * v2x + v1y * v2y)
            cum_turn = apex_pn.cum_turn + step_turn
            d_prio = d_new if _new < ST else prio[0]
            score_0 = d_prio
            score_1 = bad_streak + 0.5 if unseen else bad_streak
            score_2 = 1.0 if unseen else (d_new / paths[best_pn_id].dist)
            # Path-cumulative turn cap: total winding from path root to the
            # candidate pseudonode beyond the threshold marks the advancer
            # as unpromising. bad_streak <= 1 waives the drop — a recently-
            # active advancer gets through.
            is_promising = bad_streak < bad_streak_limit and (
                abs(cum_turn) <= turn_limit or bad_streak <= 1
            )
            prio = (score_0, score_1, score_2)
            yield prio, is_promising
            #  trace('<%d> traverser after second yield', adv_id)
            new_pn_id = self.paths.add(
                _new, sector_new, apex_eff, d_new, d_hop, cum_turn
            )
            wedge_end[side] = new_pn_id
            num_traversals[portal] += 1
            # get best_pn_id again, as the situation may have changed
            best_pn_id = best_pn_by_pair_id[pair_id]
            if best_pn_id is None or d_new < paths[best_pn_id].dist:
                best_pn_by_pair_id[pair_id] = new_pn_id
                debug(
                    '<%d> new best pn for (%d, %d) via %d: d_path = %.2f',
                    adv_id,
                    _new,
                    sector_new,
                    _apex_eff,
                    d_new,
                )
                # first arrival at (_new, sector_new) discounts the bad_streak
                #   but finding a new best_pn_id resets the bad_streak
                bad_streak = max(0, bad_streak - 1) if best_pn_id is None else 0
            elif not math.isclose(d_new, paths[best_pn_id].dist):
                bad_streak += 1

    def _find_paths(self):
        #  print('[exp] starting _explore()')
        G, P, R = self.G, self.P, self.R
        d2roots, d2rootsRank = self.d2roots, self.d2rootsRank
        ST = self.ST
        iterations_limit = self.iterations_limit
        self.prioqueue = prioqueue = []
        num_traversals = defaultdict(lambda: 0)
        self.num_traversals = num_traversals
        traversals_limit = self.traversals_limit
        paths = self.paths = PathNodes()
        triangles = P.graph['triangles']

        # set of portals (i.e. edges of P that are not used in G)
        fnT = G.graph.get('fnT')
        if fnT is not None:
            edges_G_primed = {
                ((u, v) if u < v else (v, u))
                for u, v in (fnT[edge,] for edge in G.edges)
            }
        else:
            edges_G_primed = {((u, v) if u < v else (v, u)) for u, v in G.edges}
        edges_P = {
            ((u, v) if u < v else (v, u)) for u, v in P.edges if u < ST or v < ST
        }
        constraint_edges = P.graph['constraint_edges']
        portal_set = (edges_P - edges_G_primed) - constraint_edges
        self.portal_set = portal_set | {(v, u) for u, v in portal_set}

        # Route fences are needed before sector lookup so each on-constraint
        # prime can be registered in `pair_id_by_prime_sector` under its
        # fence's subtree id.
        fences = self._extract_fences()
        self._precompute_sector_lookup(fences)
        self.best_pn_by_pair_id = [None] * len(self.pair_id_by_prime_sector)

        # constraint_bounds[c] = the cone-bounding neighbors of c that come
        # from constraint walls (= the other endpoints of constraint walls
        # incident to c). Used by `_precompute_chains` to identify each
        # chain-end's constraint cone bounds.
        constraint_bounds: dict[int, set[int]] = defaultdict(set)
        for u, v in constraint_edges:
            constraint_bounds[u].add(v)
            constraint_bounds[v].add(u)
        self.constraint_bounds = constraint_bounds

        # Build the chain topology: one Chain per route fence (or two for
        # the demoted-one-end corner case), with chain_access mapping
        # (chain-end vertex, parent-portal-pair) -> (Chain, side). The
        # trigger sites in `_advance_portal` consult this to decide
        # whether to engage a chain — non-chain wedges (the void on the
        # far side of the constraint, and navigable wedges that don't
        # separate two fences) are not registered, so the trigger
        # silently no-ops there and the per-vertex traversal budget is
        # spent only on actual chain walks.
        self.chain_access, self.chain_end_set = self._precompute_chains(fences)

        # launch channel traversers around the roots to the prioqueue
        best_pn_by_pair_id = self.best_pn_by_pair_id
        ensure_pair_id = self._ensure_pair_id
        for r in range(-R, 0):
            paths[r] = PseudoNode(r, r, None, 0.0, 0.0, 0.0)
            paths.prime_from_pn[r] = r
            paths.pn_ids_from_prime_sector[r, r] = [r]
            for left in P.neighbors(r):
                right = P[r][left]['cw']
                portal = (left, right)
                portal_sorted = (right, left) if right < left else portal

                # Chain-ends adjacent to root in the fan are stepped over by
                # the regular init advancer (triangle/portal-side-trigger fires on the far
                # vertex `n`, never on `left`/`right`), so engage the chain
                # directly here. The path arrives at `left` from the triangle
                # (r, left, right), so the cone-at-`left` bounded by (r, right)
                # picks the chain to engage. Done BEFORE the portal-validity
                # `continue` because a chain-end may be boxed in by walls
                # (no valid fan portal touches it), which would otherwise
                # leave it unengaged. Each chain-end neighbor of `r` becomes
                # `left` exactly once over the fan iteration.
                if left in self.chain_end_set:
                    access = self.chain_access.get((left, r, right))
                    if access is not None:
                        chain, c_side = access
                        d_c = d2roots[left, r].item()
                        pn_c = paths.add(left, chain.subtree, r, d_c, d_c)
                        pair_id = self.pair_id_by_prime_sector.get(
                            (left, chain.subtree)
                        )
                        if pair_id is None:
                            pair_id = self._ensure_pair_id(left, chain.subtree)
                        if (
                            best_pn_by_pair_id[pair_id] is None
                            or d_c < paths[best_pn_by_pair_id[pair_id]].dist
                        ):
                            best_pn_by_pair_id[pair_id] = pn_c
                        num_traversals[(left, left)] = traversals_limit
                        self._walk_chain(
                            left, chain, c_side, pn_c, bitarray(len(triangles))
                        )

                if right not in P[r] or portal_sorted not in portal_set:
                    # (left, right, root) not a triangle
                    # or (left, right) is not a portal
                    continue
                # flag initial portal as visited
                num_traversals[right, left] = traversals_limit

                if left >= ST or (left in G.nodes and len(G._adj[left]) == 0):
                    sec_left = NULL
                else:
                    sec_left = right
                    for _ in P[left]:
                        sec_left = P[left][sec_left]['ccw']
                        incr_edge = (
                            (sec_left, left) if sec_left < left else (left, sec_left)
                        )
                        if incr_edge in edges_G_primed or incr_edge in constraint_edges:
                            break
                    else:
                        # G is inconsistent, unable to identify sec_left
                        sec_left = NULL

                if right >= ST or (right in G.nodes and len(G._adj[right]) == 0):
                    sec_right = NULL
                else:
                    sec_right = r
                d_left = d2roots[left, r].item()
                d_right = d2roots[right, r].item()
                # add the first pseudo-nodes to paths
                wedge_end = [
                    paths.add(left, sec_left, r, d_left, d_left),
                    paths.add(right, sec_right, r, d_right, d_right),
                ]

                # shortest paths for roots' P.neighbors is a straight line
                best_pn_by_pair_id[ensure_pair_id(left, sec_left)] = wedge_end[0]
                best_pn_by_pair_id[ensure_pair_id(right, sec_right)] = wedge_end[1]

                # prioritize by distance to the closest node of the portal
                d_closest = (
                    d_left if d2rootsRank[left, r] <= d2rootsRank[right, r] else d_right
                )
                prio = (d_closest, 0.0, 1.0)
                funnel_state = (prio, r, r, [left, right], wedge_end, 0)
                advancer = self._advance_portal(
                    self.adv_counter,
                    (left, right),
                    funnel_state,
                    bitarray(len(triangles)),
                )
                heapq.heappush(prioqueue, (prio, self.adv_counter, advancer))
                self.adv_counter += 1
        # process edges in the prioqueue
        #  print(f'[exp] starting main loop, |prioqueue| = {len(prioqueue)}')
        _, adv_id, advancer = heapq.heappop(prioqueue)
        iter = 0
        while iter < iterations_limit:
            iter += 1
            debug('_find_paths[%d]: advancer id <%d>', iter, adv_id)
            try:
                # advance one portal
                prio, portal, is_promising = next(advancer)
            except StopIteration:
                # advancer decided to stop, get a new one
                if not prioqueue:
                    break
                _, adv_id, advancer = heapq.heappop(prioqueue)
            else:
                if is_promising or num_traversals[portal] < traversals_limit:
                    # advancer is still promising, push it back to queue and get top one
                    _, adv_id, advancer = heapq.heappushpop(
                        prioqueue, (prio, adv_id, advancer)
                    )
                else:
                    # forget advancer and get a new one
                    if not prioqueue:
                        break
                    _, adv_id, advancer = heapq.heappop(prioqueue)

        if iter == iterations_limit:
            warn('PathFinder loop aborted after iterations_limit reached: %d', iter)
        debug('PathFinder: loops performed: %d', iter)
        self.iterations = iter

    def _apply_all_best_paths(self, G: nx.Graph):
        """
        Update G with the paths found by `_find_paths()`.
        """
        get_best_path = self.get_best_path
        for n in range(self.T):
            path, dists = get_best_path(n)
            nx.add_path(G, path, kind='virtual')

    def best_paths_overlay(self) -> nx.Graph:
        """Merges the shortest paths for all nodes with `G`.

        The output includes `G`'s edges, excluding its feeders.

        Returns:
          Merged graph (pass to `plotting.gplot()` or 'svg.svgplot()`).
        """
        J = nx.Graph()
        J.add_nodes_from(self.G.nodes)
        self._apply_all_best_paths(J)
        K = self.G.copy()
        K.graph['overlay'] = J
        if 'capacity' in K.graph:
            # hack to prevent `gplot()` from showing infobox
            del K.graph['capacity']
        return nx.subgraph_view(K, filter_edge=lambda u, v: u >= 0 and v >= 0)

    def scaffolded(self) -> nx.Graph:
        """Wrapper for `interarraylib.scaffolded`."""
        return scaffolded(self.G, P=self.P)

    def create_detours(self) -> nx.Graph:
        """Reroute all feeder edges in G with crossings using detour paths.

        Returns:
            New networkx.Graph (shallow copy of G, with detours).
        """
        # TODO: create_detours() cannot be called twice. Enforce that!
        G, Xings, tentative = self.G.copy(), self.Xings, self.tentative.copy()

        if not Xings:
            for r, n in tentative:
                # remove the 'tentative' kind
                if 'kind' in G[r][n]:
                    del G[r][n]['kind']
            if 'tentative' in G.graph:
                del G.graph['tentative']
            _undo_P_paths_expansion(G, self.A)
            debug('<PathFinder: no crossings, detagged all tentative edges.')
            return G

        if self.saved_shortened_contours is not None:
            # Restore shortcut contours as they were before finding paths.
            # Removing `helper_nodes` strips the auxiliary contour clones
            # added during unshortening, returning G to the state expected
            # by detour-clone allocation (which uses `clone_idx = T+B+C`).
            for (
                stored_edges,
                helper_edges,
                helper_nodes,
            ) in self.saved_shortened_contours:
                G.remove_edges_from(helper_edges)
                G.remove_nodes_from(helper_nodes)
                G.add_edges_from(stored_edges)

        R, T, B, C = self.R, self.T, self.B, self.C
        clone2prime = self.clone2prime.copy()
        paths = self.paths
        best_pn_by_pair_id = self.best_pn_by_pair_id
        pair_ids_by_prime = self.pair_ids_by_prime
        clone_idx = T + B + C
        failed_detours = []

        subtree_from_subtree_id = defaultdict(list)
        subtree_id_from_n = {}
        for n in chain(range(T), range(T + B, clone_idx)):
            subtree_id = G.nodes[n]['subtree']
            subtree_from_subtree_id[subtree_id].append(n)
            subtree_id_from_n[n] = subtree_id

        for r, n in set(Xings):
            tentative.remove((r, n))
            subtree_id = subtree_id_from_n[n]
            subtree = subtree_from_subtree_id[subtree_id]
            subtree_load = G.nodes[n]['load']
            # set of nodes to examine is different depending on `branched`
            hook_candidates = (
                [n for n in subtree if n < T]
                if self.branched
                else [n, next(h for h in subtree if len(G._adj[h]) == 1)]
            )
            debug('hook_candidates: %s', hook_candidates)

            try:
                dist, pn_id, hook = min(
                    (paths[pn_id].dist, pn_id, hook)
                    for hook in hook_candidates
                    for pair_id in pair_ids_by_prime.get(hook, ())
                    if (pn_id := best_pn_by_pair_id[pair_id]) is not None
                )
            except ValueError:
                error(
                    'subtree of node %d has no non-crossing paths to '
                    'any root: leaving feeder as-is',
                    n,
                )
                # unable to fix this crossing
                failed_detours.append((r, n))
                continue
            debug('best: hook = %d, dist = %.2f', hook, dist)

            path, dists = self._trace_path(hook, pn_id)
            if not math.isclose(sum(dists), dist):
                error(
                    'distance sum (%.1f) != best distance (%.1f), hook = %d, path: %s',
                    sum(dists),
                    dist,
                    hook,
                    path,
                )

            debug('path: %s', path)
            if len(path) < 2:
                error('no path found for %d-%d', r, n)
                continue
            added_clones = len(path) - 2
            Clone = list(range(clone_idx, clone_idx + added_clones))
            clone_idx += added_clones
            clone2prime.extend(path[1:-1])
            G.add_nodes_from(
                (
                    (
                        c,
                        {
                            'label': str(c),
                            'kind': 'detour',
                            'subtree': subtree_id,
                            'load': subtree_load,
                        },
                    )
                    for c in Clone
                )
            )
            if [n, r] != path:
                # TODO: adapt this for contoured feeders
                #       maybe that's the place to prune contour clones
                G.remove_edge(r, n)
                if r != path[-1]:
                    debug(
                        'root changed from %d to %d for subtree of feeder %d, '
                        'now hooked to %d',
                        r,
                        path[-1],
                        n,
                        path[0],
                    )
                    subtree_load = G.nodes[n]['load']
                    G.nodes[r]['load'] -= subtree_load
                    G.nodes[path[-1]]['load'] += subtree_load
                G.add_weighted_edges_from(
                    zip(path[:1] + Clone, Clone + path[-1:], dists),
                    weight='length',
                    load=subtree_load,
                )
                for _, _, edgeD in G.edges(Clone, data=True):
                    edgeD.update(kind='detour', reverse=True)
                if added_clones > 0:
                    # an edge reaching root always has target < source
                    G[Clone[-1]][path[-1]]['reverse'] = False
            else:
                del G[n][r]['kind']
                debug(
                    'feeder %d–%d touches a node (touched node does not become'
                    ' a detour).',
                    n,
                    r,
                )
            if n != path[0]:
                # the hook changed: update 'load' attributes of edges/nodes
                debug('hook changed from %d to %d: recalculating loads', n, path[0])

                for node in subtree:
                    del G.nodes[node]['load']

                if Clone:
                    parent = Clone[0]
                    ref_load = subtree_load
                    G.nodes[parent]['load'] = 0
                else:
                    parent = path[-1]
                    ref_load = G.nodes[parent]['load']
                    G.nodes[parent]['load'] = ref_load - subtree_load
                total_parent_load = bfs_subtree_loads(G, parent, [path[0]], subtree_id)
                assert total_parent_load == ref_load, (
                    f'detour {n}–{path[0]}: load calculated '
                    f'({total_parent_load}) != expected load ({ref_load})'
                )

        # former tentative feeders that were not in Xings cease to be tentative
        for r, n in tentative:
            del G[r][n]['kind']

        if failed_detours:
            warn('Failed: %s', failed_detours)
            G.graph['tentative'] = failed_detours
        else:
            del G.graph['tentative']

        D = clone_idx - T - B - C
        detextra = G.size(weight='length') / self.predetour_length - 1
        if self.stunts_primes is not None:
            num_stunts = len(self.stunts_primes)
            G = nx.relabel_nodes(
                G,
                {clone: clone - num_stunts for clone in range(T + B, clone_idx)},
                copy=False,
            )
            clone_idx -= num_stunts
            B -= num_stunts
            if clone2prime:
                for stunt, prime in enumerate(self.stunts_primes, start=T + B):
                    try:
                        while True:
                            i = clone2prime.index(stunt)
                            clone2prime[i] = prime
                    except ValueError:
                        continue

        fnT = np.arange(R + clone_idx)
        fnT[T + B : clone_idx] = clone2prime
        fnT[-R:] = range(-R, 0)
        G.graph.update(
            B=B,
            D=D,
            fnT=fnT,
            detextra=detextra,
            iterations_pfinder=self.iterations,
        )
        _undo_P_paths_expansion(G, self.A)
        debug(
            '<PathFinder: created %d detour vertices, total length changed by %.2f%%',
            G.graph['D'],
            100 * detextra,
        )
        # TODO: there might be some lost contour clones that could be prunned
        return G
