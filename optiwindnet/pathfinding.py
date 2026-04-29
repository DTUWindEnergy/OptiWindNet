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
#   wall: one non-traversable mesh segment; route_walls come from contour route
#     edges, constraint_walls come from planar constraint edges (borders and
#     obstacles).
#   fence: a sequence of connected walls.
#   chain: the overlap of two fences; chain walking handles these explicitly.
#   portal: a traversable mesh edge between adjacent triangles.
#   channel: the triangle corridor explored through portals.
#   funnel: shortest-path state maintained while advancing along a channel.
#   fan: the full cyclic neighborhood around a vertex, especially a root.
#   cone: one angular region in a fan, bounded by wall-neighbor vertices.
#   prime: a geometry vertex id; pn_id: a pseudonode id in the PathNodes tree.
# Static per-chain-end topology; built once and looked up during exploration.
#   walk: list of (vertex, sector) — forward chain steps; last is the sister end.
#   cones: list of (cone_left_wall, cone_right_wall, cone_spokes) — cyclic in cw order.
#   always_skip: cone indices to never spawn into (outside or same-link).
#   entry_cones: dict v → frozenset of cone indices to skip given entry through v.
#   same_link_neighbors: cw-neighbors v whose entry cones are all on the chain's
#     own terminal-terminal link (path doesn't need to detour through this end).
ChainTopo = namedtuple(
    'ChainTopo', 'walk cones always_skip entry_cones same_link_neighbors'.split()
)


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

        self.G, self.Xings, self.tentative = G, Xings, set(tentative)
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

            def expand_P_paths_edge(s, t):
                key = (s, t) if s < t else (t, s)
                path = P_paths_shortcuts.get(key)
                if path is None:
                    return [s, t]
                if path[0] != s:
                    path = path[::-1]
                expanded = [path[0]]
                for u, v in zip(path[:-1], path[1:]):
                    expanded.extend(expand_P_paths_edge(u, v)[1:])
                return expanded

            def expand_P_paths_path(path):
                expanded = [path[0]]
                for s, t in zip(path[:-1], path[1:]):
                    expanded.extend(expand_P_paths_edge(s, t)[1:])
                return expanded

            edges_to_remove = []
            edges_to_add = []
            clone_offset = T + B
            for (s, t), (midpath, shortpath) in shortened_contours.items():
                # G follows shortpath, but we want it to follow midpath
                subtree_id = G.nodes[t]['subtree']
                stored_edges = []
                u = s
                # G's contour clones follow the expanded shortpath (see
                # G_from_S), so expand here before locating clones by prime id.
                expanded_shortpath = (
                    expand_P_paths_path([s] + shortpath + [t])[1:-1]
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
                midpath = expand_P_paths_path([s] + midpath + [t])[1:-1]
                while v != t:
                    stored_edges.append((u, v, G[u][v]))
                    edges_to_remove.append((u, v))
                    u, v = v, (G._adj[v].keys() - {u}).pop()
                stored_edges.append((u, v, G[u][v]))
                edges_to_remove.append((u, v))
                helper_edges = []
                u = s
                for v in midpath:
                    # this will use constraint vertices
                    G.add_node(v)
                    helper_edges.append((u, v))
                    edges_to_add.append((u, v))
                    G.nodes[v]['subtree'] = subtree_id
                    u = v
                helper_edges.append((u, t))
                edges_to_add.append((u, t))
                saved_shortened_contours.append((stored_edges, helper_edges))
            G.remove_edges_from(edges_to_remove)
            G.add_edges_from(edges_to_add, kind='contour')
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
        self.P, self.VertexC, self.clone2prime = P, VertexC, clone2prime
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

    def _precompute_sector_lookup(self) -> None:
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
        traversals_limit = self.traversals_limit
        num_traversals = self.num_traversals
        triangles = P.graph['triangles']
        traverser = self._traverse_channel(adv_id, *funnel_state)
        next(traverser)
        if side is not None:
            prio, is_promising = traverser.send((portal, side))
            yield prio, portal, is_promising
            next(traverser)
            # NOTE: do NOT fire Trigger B here — this branch only runs for
            # advancers spawned by the chain's exit logic (regular advancers
            # don't pass an initial side). Their initial portal already
            # touches the chain-end they're trying to step away from, so
            # firing chain engagement again here would just recurse.
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
                # If `n` is a chain-end with budget, engage the chain by
                # anchoring the funnel from both chain boundary walls, then walk the
                # chain forward; otherwise update the (n, sector) best pn.
                if n in chain_end_set and num_traversals[(n, n)] < traversals_limit:
                    # Two phantom-portal sends place y=n in the funnel from
                    # both chain boundary walls; the second send anchors pn_n on the
                    # contour apex via standard funnel narrowing.
                    traverser.send(((left, n), 1))
                    next(traverser)
                    traverser.send(((n, right), 0))
                    next(traverser)
                    num_traversals[(n, n)] += 1
                    self._walk_chain(
                        n, left, self.paths.last_added_pn, is_triangle_seen
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
            # Trigger B: just-processed endpoint is a chain-end and the
            # cone we entered through is NOT bounded entirely by route walls on
            # the chain's own terminal-terminal link (otherwise the path
            # doesn't need to detour through the chain).
            y = portal[side]
            x = portal[1 - side]
            chain_topo = self.chain_topo
            if (
                y in chain_end_set
                and num_traversals[(y, y)] < traversals_limit
                and x not in chain_topo[y].same_link_neighbors
            ):
                # Reuse the parent's just-narrowed pseudonode for y as the
                # chain entry (an additional send would produce a same-prime
                # self-link).
                num_traversals[(y, y)] += 1
                self._walk_chain(y, x, self.paths.last_added_pn, is_triangle_seen)

    def _walk_chain(
        self,
        y_entry: int,
        x_entry: int,
        entry_pn: int,
        is_triangle_seen: bitarray,
    ) -> None:
        """Walk the chain forward from `y_entry` along the precomputed sequence,
        registering a pseudonode at each visited chain vertex (parented by the
        previous one). At the exit chain-end, spawn an advancer into each
        cone that is neither entry-side, outside, nor on the chain's own link.

        `entry_pn` is the pseudonode for `y_entry` (created by the caller's
        funnel narrowing). `x_entry` is the path's entry-side neighbor at
        `y_entry`, used as the entry vertex in the zero-walk case.
        """
        paths = self.paths
        VertexC = self.VertexC

        prev, cur = x_entry, y_entry
        parent_pn = entry_pn
        for c_next, sector in self.chain_topo[y_entry].walk:
            d_hop = _node_dist(VertexC, cur, c_next)
            pn_parent = paths[parent_pn]
            parent_pn = paths.add(
                c_next,
                sector,
                parent_pn,
                pn_parent.dist + d_hop,
                d_hop,
                pn_parent.cum_turn,
            )
            prev, cur = cur, c_next

        topo = self.chain_topo.get(cur)
        if topo is None:
            debug('chain: exit %d has no usable cone topology', cur)
            return
        skip = topo.always_skip | topo.entry_cones.get(prev, frozenset())
        for k, (left_wall, right_wall, cone_spokes) in enumerate(topo.cones):
            if k in skip:
                continue
            self._spawn_exit_cone(
                cur,
                left_wall,
                right_wall,
                cone_spokes,
                parent_pn,
                is_triangle_seen,
            )

    def _precompute_chain_topology(
        self, chain_end_set: set[int]
    ) -> dict[int, ChainTopo]:
        """Build the static cone-and-walk registry for every chain-end with
        at least two incident walls. Skipped chain-ends (degenerate) are absent from
        the result and won't trigger chain engagement.
        """
        P = self.P
        route_walls = self.route_walls
        constraint_walls = self.constraint_walls
        link_of_prime = self.link_of_prime
        topo: dict[int, ChainTopo] = {}

        for c in chain_end_set:
            cw_nbrs = list(P.neighbors_cw_order(c))
            n = len(cw_nbrs)
            wall_neighbors = route_walls.get(c, set()) | constraint_walls.get(c, set())
            wall_positions = [i for i, v in enumerate(cw_nbrs) if v in wall_neighbors]
            if len(wall_positions) < 2:
                continue
            n_cones = len(wall_positions)
            chain_link = link_of_prime.get(c, frozenset())

            cones: list[tuple[int, int, list[int]]] = []
            always_skip: set[int] = set()
            same_link_cones: set[int] = set()
            cone_of_idx: dict[int, int] = {}  # cw-index → cone idx (interior only)
            for k, wall_pos in enumerate(wall_positions):
                next_wall_pos = wall_positions[(k + 1) % n_cones]
                cone_spokes: list[int] = []
                cur = (wall_pos + 1) % n
                while cur != next_wall_pos:
                    v = cw_nbrs[cur]
                    cone_of_idx[cur] = k
                    if v not in wall_neighbors and v >= 0:
                        cone_spokes.append(v)
                    cur = (cur + 1) % n
                left_wall, right_wall = cw_nbrs[wall_pos], cw_nbrs[next_wall_pos]
                cones.append((left_wall, right_wall, cone_spokes))
                link_a = link_of_prime.get(left_wall, frozenset())
                link_b = link_of_prime.get(right_wall, frozenset())
                if not link_a and not link_b:
                    always_skip.add(k)
                if link_a == chain_link and link_b == chain_link:
                    always_skip.add(k)
                    same_link_cones.add(k)

            entry_cones: dict[int, frozenset[int]] = {}
            for i, v in enumerate(cw_nbrs):
                if v in wall_neighbors:
                    k_at = wall_positions.index(i)
                    entry_cones[v] = frozenset({k_at, (k_at - 1) % n_cones})
                else:
                    entry_cones[v] = frozenset({cone_of_idx[i]})

            same_link_neighbors = (
                {v for v, ec in entry_cones.items() if ec.issubset(same_link_cones)}
                if same_link_cones
                else set()
            )

            # Forward walk from c through the chain, precomputing each step's
            # sector. Stops when forward wall overlap disagrees or branches.
            walk: list[tuple[int, int]] = []
            c_prev, c_cur = None, c
            visited = {c}
            while True:
                forward = (
                    route_walls.get(c_cur, set()) & constraint_walls.get(c_cur, set())
                ) - {c_prev}
                if len(forward) != 1:
                    break
                c_next = next(iter(forward))
                if c_next in visited:
                    break
                visited.add(c_next)
                cand_fwd = (
                    route_walls.get(c_next, set()) | constraint_walls.get(c_next, set())
                ) - {c_cur}
                sector = min(cand_fwd | {c_cur})
                walk.append((c_next, sector))
                c_prev, c_cur = c_cur, c_next

            topo[c] = ChainTopo(
                walk=walk,
                cones=cones,
                always_skip=frozenset(always_skip),
                entry_cones=entry_cones,
                same_link_neighbors=same_link_neighbors,
            )
        return topo

    def _spawn_exit_cone(
        self,
        w: int,
        cone_left_wall: int,
        cone_right_wall: int,
        cone_spokes: list[int],
        pn_w_id: int,
        is_triangle_seen: bitarray,
    ) -> None:
        """Spawn end-spoke and intermediate-pair advancers covering one exit
        cone at chain-end `w`. `cone_left_wall` and `cone_right_wall` are the
        cone's bounding wall-neighbor vertices in cw order; `cone_spokes` are
        non-wall spokes inside.
        """
        P = self.P
        paths = self.paths
        prioqueue = self.prioqueue
        portal_set = self.portal_set
        VertexC = self.VertexC
        best_pn_by_pair_id = self.best_pn_by_pair_id
        pair_id_by_prime_sector = self.pair_id_by_prime_sector
        ensure_pair_id = self._ensure_pair_id
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

        if cone_spokes:
            x_1, x_k = cone_spokes[0], cone_spokes[-1]
            _launch(w, x_1, 1)
            _launch(x_k, w, 0)
            for xi, xj in zip(cone_spokes, cone_spokes[1:]):
                if (xi, xj) in portal_set:
                    _launch(xi, xj, 1)
        else:
            # Single-triangle exit: only the connecting portal between the
            # two bounding wall-neighbor vertices. Skip if it would re-engage the same
            # chain-end (third vertex of the new triangle is w).
            if (
                (cone_left_wall, cone_right_wall) in portal_set
                and cone_right_wall in P[cone_left_wall]
                and P[cone_left_wall][cone_right_wall].get('ccw') != w
            ):
                _launch(cone_left_wall, cone_right_wall, 1)

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
            else:
                portal, side = portal_step
            #  trace('<%d> got (portal, side)', adv_id)

            _new = portal[side]
            if 0 <= _new < T:
                opposite = portal[1 - side]
                try:
                    sector_new = sector_by_prime_opposite[_new][opposite]
                except KeyError:
                    sector_new = scan_sector(_new, opposite)
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
                    # Loop continues while the current wapex doesn't yet hit
                    # the farside wall and the test predicate passes. The
                    # `== _new` clause forces one more step whenever the
                    # narrowing parks at a pseudonode whose prime matches
                    # `_new` (chain-anchor case: cross(new, new, contender)=0
                    # would otherwise exit the loop and leave a degenerate
                    # apex that paths.add would turn into a self-link).
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
                # chain-anchor degenerate apex.
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
        self._precompute_sector_lookup()
        self.best_pn_by_pair_id = [None] * len(self.pair_id_by_prime_sector)

        # Chain pre-processing: chain-end primes are constraint-wall primes touched
        # by a kind='contour' G-edge AND on a constraint wall.
        # Walking the chain in prime space uses route_walls together with
        # constraint_walls.
        T_, B_ = self.T, self.B
        clone_offset = T_ + B_

        def _prime(n: int) -> int:
            if n < 0:
                return n
            if fnT is not None and n >= clone_offset:
                return int(fnT[n])
            return n

        route_walls: dict[int, set[int]] = defaultdict(set)
        for u, v, d in G.edges(data=True):
            if d.get('kind') != 'contour':
                continue
            pu, pv = _prime(u), _prime(v)
            if pu == pv:
                continue
            route_walls[pu].add(pv)
            route_walls[pv].add(pu)
        constraint_walls: dict[int, set[int]] = defaultdict(set)
        for u, v in constraint_edges:
            constraint_walls[u].add(v)
            constraint_walls[v].add(u)
        chain_end_set = {
            p for p in route_walls if T_ <= p < clone_offset and p in constraint_walls
        }
        self.route_walls = route_walls
        self.constraint_walls = constraint_walls

        # For each prime on a kind='contour' G-edge polyline, determine its
        # terminal-terminal link = the frozenset of terminals at the polyline's
        # endpoints. Used by chain-engagement gating (Trigger B) and exit-cone
        # spawnability classification.
        link_of_prime: dict[int, frozenset] = {}
        seen: set[int] = set()
        for start in route_walls:
            if start in seen:
                continue
            comp: set[int] = set()
            stack = [start]
            while stack:
                u = stack.pop()
                if u in comp:
                    continue
                comp.add(u)
                stack.extend(v for v in route_walls[u] if v not in comp)
            terminals = frozenset(p for p in comp if 0 <= p < T_)
            for p in comp:
                link_of_prime[p] = terminals
            seen.update(comp)
        self.link_of_prime = link_of_prime

        # Per-chain-end static topology: cone partition, entry-cone lookup,
        # forward chain-walk sequence. Built once, queried by Triggers A/B
        # and `_walk_chain` instead of recomputing degree-bounded scans on
        # every chain engagement.
        self.chain_topo: dict[int, ChainTopo] = self._precompute_chain_topology(
            chain_end_set
        )
        # Set of chain-ends with usable topology (used as Trigger guard).
        self.chain_end_set = set(self.chain_topo)

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
                # the regular init advancer (Trigger A/B fires on the far
                # vertex `n`, never on `left`/`right`), so engage the chain
                # directly here. A chain entrance from root is a single
                # straight hop — one distance, and sector = NULL (chain-ends are
                # constraint-wall vertices, for which sector lookup always returns
                # NULL, so this matches the entry pn that Triggers A/B produce via
                # funnel narrowing and lets `paths.add` dedupe against them). Done
                # BEFORE the portal-validity `continue`
                # because a chain-end may be boxed in by walls (no valid
                # fan portal touches it), which would otherwise leave it
                # unengaged. Each chain-end neighbor of `r` becomes `left`
                # exactly once over the fan iteration, so checking `left`
                # alone covers every chain-end in the fan.
                if (
                    left in self.chain_end_set
                    and r not in self.chain_topo[left].same_link_neighbors
                ):
                    d_c = d2roots[left, r].item()
                    pn_c = paths.add(left, NULL, r, d_c, d_c)
                    num_traversals[(left, left)] = traversals_limit
                    self._walk_chain(left, r, pn_c, bitarray(len(triangles)))

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
            debug('<PathFinder: no crossings, detagged all tentative edges.')
            return G

        if self.saved_shortened_contours is not None:
            # Restore shortcut contours as they were before finding paths.
            for stored_edges, helper_edges in self.saved_shortened_contours:
                G.remove_edges_from(helper_edges)
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
        debug(
            '<PathFinder: created %d detour vertices, total length changed by %.2f%%',
            D,
            100 * detextra,
        )
        # TODO: there might be some lost contour clones that could be prunned
        return G
