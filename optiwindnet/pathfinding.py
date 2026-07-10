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
#   subtree: route-fence subtree id, used as the `sector` label for chain-
#     walk pseudonodes so overlapping chains keep distinct (prime, sector)
#     buckets. A route fence whose mp has interior non-constraint hops is
#     split into one sub-fence per contiguous-walls segment; sub-segments
#     share `subtree`, and chain pairing uses `(subtree, mp[0], mp[-1])`
#     to keep the per-segment chains separate.
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
    """Recursively expand a ``P_paths`` shortcut hop into the full P-edge sequence.

    ``shortcuts`` maps a normalized ``(u_lo, v_hi)`` pair to the list of vertices
    along the underlying P-path. Returns ``[s, t]`` verbatim when ``(s, t)`` is not
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
    """Expand every shortcut hop along ``path`` into its underlying P-edges."""
    expanded = [path[0]]
    for s, t in zip(path[:-1], path[1:]):
        expanded.extend(_expand_P_paths_edge(s, t, shortcuts)[1:])
    return expanded


class PathNodes(dict):
    """Tree of pseudonodes for shortest-path candidates.

    A prime is a geometry vertex id. A pseudonode id (``pn_id``) identifies one
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
    returned only with a call to method :meth:`create_detours`.

    Only edges in graph attribute ``'tentative'`` or, lacking that, edges with the
    attribute ``'kind'`` with value ``'tentative'`` are checked for crossings.

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
      On instances with ``capacity=2``, the default values may not enable finding all
      the shortest feeders. If :func:`.crossings.find_geometric_crossings` reports any
      crossings, retry with ``traversals_limit=10`` and ``iterations_limit=50000``.

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
            # no crossings, there is no point in pathfinding; still set P and
            # fences so that scaffolded() works even when create_detours() is
            # not called. fences stays empty here: even if G has contour
            # clones from G_from_S, they go unconverted by scaffolded() on
            # this path, since fences are only built below.
            self.P = planar
            self.fences = []
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

        # Single pass over G.edges: non-contour edges contribute their
        # prime pair to `edges_G_primes` directly; contour edges register
        # their A-edge for later fence emission. G's contour clones may
        # follow a synthetic (shortcut) prime sequence, so the fence-side
        # loop below substitutes the fully P-edge-expanded chain for what
        # those clones would naively project to.
        shortened = G.graph.get('shortened_contours') or {}
        contour_A_edges: dict[tuple[int, int], int] = {
            ae: G.nodes[ae[1]]['subtree'] for ae in shortened
        }
        edges_G_primes: set[tuple[int, int]] = set()
        for u, v, d in G.edges(data=True):
            if d.get('kind') == 'contour':
                ae = d.get('A_edge')
                if ae is not None and ae not in contour_A_edges:
                    contour_A_edges[ae] = G.nodes[ae[1]]['subtree']
                continue
            pu, pv = int(fnT[u]), int(fnT[v])
            edges_G_primes.add((pu, pv) if pu < pv else (pv, pu))

        # Build fences from the discovered contour A-edges. The midpath
        # source is `shortened` for shortened contours and `A[s][t]['midpath']`
        # otherwise; both store the bidirectional_dijkstra path on `P_paths`,
        # which we expand to a real P-edge sequence. Fence endpoints (s, t)
        # are tree members of S — root-endpoint A-edges with midpath are
        # routed to kind='tentative' by G_from_S and never appear here.
        # Interior non-constraint hops in the expanded mp (P_paths chose a
        # diagonal cutting between disjoint constraint chains) split the
        # fence into one sub-fence per contiguous-walls segment, each with
        # synthesized endpoints at the break primes; sub-fences share the
        # original `subtree`. `edges_G_primes` records the full chain_seq
        # union as barriers regardless of the split.
        #
        # constraint_bounds[c] = the constraint-wall neighbors of c (the other
        # endpoints of constraint edges incident to c). Built from `planar`,
        # but valid for the flipped `P` too: `planar_flipped_by_routeset` only
        # flips non-constraint edges, so it leaves `constraint_edges` (and
        # hence this adjacency) untouched. Used by the fence-split below and by
        # the chain-topology helpers (`_precompute_chains` and friends).
        constraint_bounds: dict[int, set[int]] = defaultdict(set)
        for u, v in planar.graph['constraint_edges']:
            constraint_bounds[u].add(v)
            constraint_bounds[v].add(u)
        self.constraint_bounds = constraint_bounds
        shortcuts = A.graph.get('P_paths_shortcuts', {})
        # `edges_G_primes` already holds the routeset edges; contour hops are
        # added per (re)build below. A contour hop absent from the base
        # embedding is a P-diagonal the flip must realize; when two requested
        # diagonals' flips interfere the flip silently leaves one unrealized.
        # Rather than predict crossings, use the flip as the oracle: realize,
        # find any contour diagonal it dropped, de-shortcut that hop (or its
        # blocking partner) onto the constraint mesh, and re-flip. Routeset
        # diagonals are never touched — an unrealized route edge is a tolerated
        # discrepancy (gates, etc.), not a fence break.
        edges_routeset = set(edges_G_primes)

        # Expand every contour's midpath to its interior P-edge / P-diagonal hop
        # sequence (mutated in place by the de-shortcut resolver).
        contour_mps: dict[tuple[int, int], tuple[int, list[int]]] = {}
        for ae, subtree in contour_A_edges.items():
            midpath = (
                shortened[ae][0] if ae in shortened else A[ae[0]][ae[1]].get('midpath')
            )
            if not midpath:
                continue
            expanded = _expand_P_paths_path([ae[0], *midpath, ae[1]], shortcuts)[1:-1]
            contour_mps[ae] = (subtree, list(expanded))

        def build_fences() -> tuple[
            set[tuple[int, int]],
            list[Fence],
            dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
        ]:
            # (Re)build the G-prime edge set and the route fences from the
            # current contour midpaths; also map each contour P-diagonal hop to
            # the fence location(s) that walk through it.
            egp = set(edges_routeset)
            flist: list[Fence] = []
            diag_locs: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = (
                defaultdict(list)
            )
            for ae, (subtree, mp) in contour_mps.items():
                chain_seq = (ae[0], *mp, ae[1])
                for pos, (a, b) in enumerate(zip(chain_seq[:-1], chain_seq[1:])):
                    egp.add((a, b) if a < b else (b, a))
                    if not planar.has_edge(a, b):
                        diag_locs[(a, b) if a < b else (b, a)].append((ae, pos))
                breaks = [
                    i
                    for i in range(len(mp) - 1)
                    if mp[i + 1] not in constraint_bounds.get(mp[i], ())
                ]
                if not breaks:
                    flist.append(Fence(ae, mp, subtree))
                    continue
                s, t = ae
                segments: list[tuple[int, int]] = []
                start = 0
                for b in breaks:
                    segments.append((start, b + 1))
                    start = b + 1
                segments.append((start, len(mp)))
                for k, (lo, hi) in enumerate(segments):
                    sub_s = s if k == 0 else mp[lo - 1]
                    sub_t = t if k == len(segments) - 1 else mp[hi]
                    flist.append(Fence((sub_s, sub_t), list(mp[lo:hi]), subtree))
            return egp, flist, diag_locs

        def flip(egp: set[tuple[int, int]]) -> nx.PlanarEmbedding:
            return planar_flipped_by_routeset(
                egp, planar=planar, VertexC=VertexC, ST=self.ST, diagonals=diagonals
            )

        # Identify conflicting contour diagonals and resolve them directly
        # in a single pass
        # 1. Find all contour diagonals (hops that are not in the base planar mesh)
        contour_diags = set()
        for ae, (subtree, mp) in contour_mps.items():
            chain_seq = (ae[0], *mp, ae[1])
            for a, b in zip(chain_seq[:-1], chain_seq[1:]):
                if not planar.has_edge(a, b):
                    contour_diags.add((a, b) if a < b else (b, a))

        # 2. Precompute base edge and quad sides for each diagonal in the
        # base planar embedding
        base_edge = {}
        quad_sides = {}
        for d in contour_diags:
            u, v = d
            common = [w for w in planar.neighbors(u) if planar.has_edge(v, w)]
            found = False
            for s in common:
                for c in common:
                    if s < c and planar.has_edge(s, c):
                        # Verify s-c is the base edge currently present in
                        # the triangulation
                        if {planar[s][c]['ccw'], planar[c][s]['ccw']} == {u, v}:
                            base_edge[d] = (s, c)
                            quad_sides[d] = {
                                (u, s) if u < s else (s, u),
                                (s, v) if s < v else (v, s),
                                (v, c) if v < c else (c, v),
                                (c, u) if c < u else (u, c),
                            }
                            found = True
                            break
                if found:
                    break

        # Helper to get the most-anchored constraint-mesh endpoint of an edge
        def get_mesh_endpoint(edge: tuple[int, int]) -> int | None:
            s, t = edge
            s_in = s in constraint_bounds
            t_in = t in constraint_bounds
            if s_in and t_in:
                return (
                    s if len(constraint_bounds[s]) >= len(constraint_bounds[t]) else t
                )
            return s if s_in else (t if t_in else None)

        # 3. Detect conflicting diagonal flips
        conflicts = set()
        for d1 in contour_diags:
            for d2 in contour_diags:
                if d1 < d2:
                    b1, b2 = base_edge.get(d1), base_edge.get(d2)
                    if b1 is not None and b2 is not None:
                        if (
                            b1 == b2
                            or b1 in quad_sides.get(d2, ())
                            or b2 in quad_sides.get(d1, ())
                        ):
                            conflicts.add((d1, d2))

        # 4. Resolve conflicts directly by scheduling de-shortcuts
        to_deshortcut = {}
        self.unrealized_contours_resolved = False
        for d1, d2 in conflicts:
            if d1 in to_deshortcut or d2 in to_deshortcut:
                continue
            b1 = base_edge[d1]
            b2 = base_edge[d2]
            w1 = get_mesh_endpoint(b1)
            w2 = get_mesh_endpoint(b2)
            if w1 is not None:
                to_deshortcut[d1] = w1
            elif w2 is not None:
                to_deshortcut[d2] = w2

        # 5. Apply de-shortcuts directly to the midpaths
        if to_deshortcut:
            self.unrealized_contours_resolved = True
            # Map each scheduled diagonal to its occurrence positions in the midpaths
            contour_diag_locs = defaultdict(list)
            for ae, (subtree, mp) in contour_mps.items():
                chain_seq = (ae[0], *mp, ae[1])
                for pos, (a, b) in enumerate(zip(chain_seq[:-1], chain_seq[1:])):
                    k = (a, b) if a < b else (b, a)
                    if k in to_deshortcut:
                        contour_diag_locs[k].append((ae, pos))

            inserts = defaultdict(set)
            for d, w in to_deshortcut.items():
                for ae, pos in contour_diag_locs[d]:
                    inserts[ae].add((pos, w))
            for ae, items in inserts.items():
                mp = contour_mps[ae][1]
                for pos, w in sorted(items, reverse=True):
                    mp.insert(pos, w)

        # 6. Rebuild and perform planar flips exactly once
        edges_G_primes, fences, _ = build_fences()
        P = flip(edges_G_primes)

        self.fences = fences
        self.edges_G_primes = edges_G_primes
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

        # Safety net: every fence hop must be an edge of the flipped navigation
        # mesh. A contour diagonal the resolver could not place surfaces here
        # clearly, rather than cryptically deep in the chain builder.
        for fence in fences:
            seq = (fence.endpoints[0], *fence.primes_on_constraint, fence.endpoints[1])
            for a, b in zip(seq[:-1], seq[1:]):
                if not P.has_edge(a, b):
                    raise ValueError(
                        'PathFinder: fence for subtree %d (A-edge %s) hop %d-%d '
                        'is absent from the flipped navigation mesh — an '
                        'unresolved contour-diagonal conflict.'
                        % (fence.subtree, fence.endpoints, a, b)
                    )

        # Precompute everything that depends only on (P, edges_G_primes,
        # fences). `_find_paths` then runs the fan-init / main loop with
        # plain dict / set lookups.
        ST = self.ST
        constraint_edges = P.graph['constraint_edges']
        edges_P = {
            ((u, v) if u < v else (v, u)) for u, v in P.edges if u < ST or v < ST
        }
        portal_set = (edges_P - edges_G_primes) - constraint_edges
        self.portal_set = portal_set | {(v, u) for u, v in portal_set}

        self._precompute_sector_lookup(fences)
        self.best_pn_by_pair_id = [None] * len(self.pair_id_by_prime_sector)

        # Build the chain topology: one Chain per route fence, with
        # chain_access mapping
        # (chain-end vertex, parent-portal-pair) -> (Chain, side). The
        # trigger sites in `_advance_portal` consult this to decide
        # whether to engage a chain — non-chain wedges (the void on the
        # far side of the constraint, and navigable wedges that don't
        # separate two fences) are not registered, so the trigger
        # silently no-ops there and the per-vertex traversal budget is
        # spent only on actual chain walks.
        self.chain_access, self.chain_end_set = self._precompute_chains(fences)

        self._find_paths()

    def _find_quad_and_sides(
        self, u: int, v: int, planar: nx.PlanarEmbedding
    ) -> tuple[tuple[int, int], set[tuple[int, int]]] | None:
        """Find the base edge and the four sides of the quad for diagonal (u, v)."""
        common = [w for w in planar.neighbors(u) if planar.has_edge(v, w)]
        for s in common:
            for c in common:
                if s < c and planar.has_edge(s, c):
                    # Verify s-c is the base edge currently present in the triangulation
                    if {planar[s][c]['ccw'], planar[c][s]['ccw']} == {u, v}:
                        return (s, c), {
                            (u, s) if u < s else (s, u),
                            (s, v) if s < v else (v, s),
                            (v, c) if v < c else (c, v),
                            (c, u) if c < u else (u, c),
                        }
        return None

    def _get_mesh_endpoint(
        self, base: tuple[int, int], constraint_bounds: dict[int, set[int]]
    ) -> int | None:
        """Return the most-anchored constraint-mesh endpoint of the base edge."""
        u, v = base
        u_in = u in constraint_bounds
        v_in = v in constraint_bounds
        if u_in and v_in:
            u_len = len(constraint_bounds[u])
            v_len = len(constraint_bounds[v])
            if u_len > v_len:
                return u
            elif v_len > u_len:
                return v
            return max(u, v)
        return u if u_in else (v if v_in else None)

    def _deshortcut_unrealized_contours(
        self,
        unrealized: list[tuple[int, int]],
        contour_diag_locs: dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
        contour_mps: dict[tuple[int, int], tuple[int, list[int]]],
        planar: nx.PlanarEmbedding,
        constraint_bounds: dict[int, set[int]],
    ) -> bool:
        """De-shortcut contour diagonals the flip failed to realize.

        ``unrealized`` lists contour P-diagonals ``(a, b)`` absent from the
        just-flipped mesh — each breaks the fence that walks through it. The flip
        realizes a diagonal by removing the base edge it crosses and locking the
        four sides of its quad; two such diagonals interfere when the base of one
        is a side of the other's quad, so only one is realized. Base edge and quad
        sides are read from the embedding's rotation system (``planar.neighbors``
        and the ``cw``/``ccw`` half-edge links); no coordinates are consulted.

        A dropped diagonal is de-shortcut by inserting, between ``a`` and ``b``,
        the endpoint of its crossed base edge that lies on the constraint mesh
        (so the chain builder can anchor it); the two new sides are base-P edges,
        needing no flip, and :func:`create_detours` reapplies the shortcut. When the
        dropped diagonal has no constraint-mesh endpoint (its base spans two
        terminals), the *realized* diagonal whose flip locked its base is
        de-shortcut instead, freeing the dropped one for the next flip. Returns
        ``True`` if any midpath changed.
        """
        # Precompute quads for all contour diagonals to avoid redundant searches
        quad_cache = {}
        for d in contour_diag_locs:
            quad_cache[d] = self._find_quad_and_sides(d[0], d[1], planar)

        # Choose, per conflict, the contour diagonal to de-shortcut and its
        # constraint-mesh vertex.
        targets: dict[tuple[int, int], int] = {}
        for d in unrealized:
            q = quad_cache.get(d)
            if q is None:
                continue
            base, _ = q
            w = self._get_mesh_endpoint(base, constraint_bounds)
            if w is not None:
                targets[d] = w
                continue
            # No own detour: de-shortcut the realized contour diagonal whose flip
            # locked d's base edge (base(d) is a side of its quad).
            for e in contour_diag_locs:
                if e == d or e in unrealized:
                    continue
                qe = quad_cache.get(e)
                if qe is None:
                    continue
                base_e, sides_e = qe
                if base in sides_e:
                    we = self._get_mesh_endpoint(base_e, constraint_bounds)
                    if we is not None:
                        targets[e] = we
                        break

        if not targets:
            return False

        # Insert the chosen vertex between each hop's endpoints (descending
        # position so earlier insertions don't shift later ones).
        inserts = defaultdict(set)
        for d, w in targets.items():
            for ae, pos in contour_diag_locs[d]:
                inserts[ae].add((pos, w))
        for ae, items in inserts.items():
            mp = contour_mps[ae][1]
            for pos, w in sorted(items, reverse=True):
                mp.insert(pos, w)
        return True

    def _trace_path(self, start_prime: int, pn_id: int):
        """Return the path and hop distances from ``start_prime`` to a root."""
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
        ``_.get_best_path(«node»)`` produces a ``tuple(path, dists)``.
        ``path`` contains a sequence of nodes from the original
        ``networkx.Graph`` ``G``, from ``«node»`` to the closest root.
        ``dists`` contains the lengths of the segments defined by ``paths``.
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
        """Uncached sector scan for one ``(prime, opposite)`` pair."""
        T = self.T
        G = self.G
        P = self.P
        tentative = self.tentative
        if prime >= T:
            # `prime` is on a constraint wall or is a supertriangle vertex,
            # hence it is only reachable from one side -> arbitrary sector id
            return NULL
        if opposite in G._adj.get(prime, {}):  # type: ignore
            # special case: visiting a DEAD-END
            return opposite
        prime_adj = G._adj.get(prime, {})  # type: ignore
        nbr = P[prime][opposite]['ccw']
        for _ in range(len(P._adj[prime])):  # type: ignore
            if nbr < T and nbr in prime_adj:
                if nbr >= 0 or (nbr, prime) not in tentative:
                    return nbr
            nbr = P[prime][nbr]['ccw']
        # could not find a non-tentative G edge around prime
        return NULL

    def _get_sector_from_opposite(self, prime: int, opposite: int) -> int:
        """Return the cached sector for reaching ``prime`` from ``opposite``."""
        if prime >= self.T:
            return NULL
        try:
            return self.sector_by_prime_opposite[prime][opposite]
        except AttributeError:
            return self._scan_sector_from_opposite(prime, opposite)
        except KeyError:
            return self._scan_sector_from_opposite(prime, opposite)

    def _precompute_sector_lookup(self, fences: list[Fence]) -> None:
        """Precompute sector and dense ``(prime, sector)`` ids for pathfinding."""
        P = self.P
        T = self.T
        ST = self.ST
        R = self.R
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
                # Roots get `(r, r)` (canonical root pseudonode anchor) and
                # `(r, NULL)` (path arriving at a root from an advance, or a
                # root appearing as a cone exit prime).
                add_pair(prime, prime)
                add_pair(prime, NULL)
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
                    and nbr in G._adj.get(prime, {})  # type: ignore
                    and (nbr >= 0 or (nbr, prime) not in tentative)
                )
            }
            by_opposite: dict[int, int] = {}
            for opposite in cw_nbrs:
                if opposite in G._adj.get(prime, {}):  # type: ignore
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

        # Fan-init pseudonode buckets: at the start of `_find_paths`, each
        # root's planar fan picks a (prime, sector) where `sector` is the
        # first cyclic neighbor of `prime` (CCW from the parent triangle's
        # opposite vertex) reached via a barrier — a G-edge prime-pair or a
        # constraint edge. The sector can be a constraint vertex / root /
        # supertriangle vertex, none of which the per-terminal scan above
        # would record. We register them here so `_find_paths` can do a
        # plain dict lookup.
        # Only valid portals matter: `_find_paths` skips `(r, left)` when
        # `(left, right)` is not in `portal_set`, and `_fan_init_sector`'s
        # walk requires `right` to be a P-neighbor of `left` (true for
        # P-edges, but `(left, right)` need not be a P-edge in general).
        portal_set = self.portal_set
        fan_sectors: dict[tuple[int, int], tuple[int, int]] = {}
        for r in range(-R, 0):
            if r not in P:
                continue
            for left in P.neighbors(r):
                right = P[r][left]['cw']
                if (left, right) not in portal_set:
                    continue
                sec_left = self._fan_init_sector(left, right) if left < ST else NULL
                if right >= ST or (right in G.nodes and len(G._adj[right]) == 0):  # type: ignore
                    sec_right = NULL
                else:
                    sec_right = r
                fan_sectors[(r, left)] = (sec_left, sec_right)
                add_pair(left, sec_left)
                add_pair(right, sec_right)

        self.sector_by_prime_opposite = sector_by_prime_opposite
        self.pair_id_by_prime_sector = pair_id_by_prime_sector
        self.pair_ids_by_prime = pair_ids_by_prime
        self.fan_sectors = fan_sectors

    def _fan_init_sector(self, prime: int, opposite: int) -> int:
        """Sector for a fan-init pseudonode at ``prime`` reached from ``opposite``.

        Walks ``prime``'s P-cyclic neighbors CCW from ``opposite`` and returns
        the first one whose edge from ``prime`` is a barrier (a G-edge in
        prime form, or a constraint edge). Falls back to NULL when the
        barrier-incident neighbor cannot be identified (boxed-in or
        inconsistent G).
        """
        P = self.P
        G = self.G
        edges_G_primes = self.edges_G_primes
        constraint_edges = P.graph['constraint_edges']
        if prime in G.nodes and len(G._adj[prime]) == 0:  # type: ignore
            return NULL
        sector = opposite
        for _ in P[prime]:
            sector = P[prime][sector]['ccw']
            incr_edge = (sector, prime) if sector < prime else (prime, sector)
            if incr_edge in edges_G_primes or incr_edge in constraint_edges:
                return sector
        return NULL

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
        """Walk ``chain`` from its ``cones[side]`` access cone (entered via
        the trigger that called us) to its ``cones[1 - side]`` cone,
        creating chain-walk pseudonodes along the way and spawning exit
        advancers from the partner cone.

        ``entry_pn`` is the pseudonode for ``y_entry`` (the chain-end the
        funnel just landed on); for spanning chains, ``y_entry`` is
        ``chain.cones[side].vertex``; for single-vertex chains, both cones share
        that vertex.
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

    def _partition_into_cones(
        self, c: int, cone_bounds: set[int], rotated: list[int]
    ) -> list[tuple[int, int, list[int], list[tuple[int, int]]]]:
        """Partition ``c``'s cyclic neighbors ``rotated`` into wedges between
        consecutive members of ``cone_bounds``. For each wedge, return
        ``(left, right, spokes, pair_keys)`` where ``pair_keys`` lists the
        ``(a, b)`` pairs of consecutive cyclic-neighbors of ``c`` that fall
        inside the wedge — these are the ``(left, right)`` lookup keys an
        advancer crossing into ``c`` from a parent triangle on this wedge's
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

    def _build_chains_at(
        self,
        v: int,
        spanning_endings: list[tuple[Fence, str]],
        touching: list[Fence],
    ) -> (
        tuple[
            list[tuple[int, AccessCone, list[tuple[int, int]], list[int]]],
            list[tuple[Chain, list[tuple[int, int]], list[tuple[int, int]]]],
        ]
        | None
    ):
        """Build chains at chain-end vertex ``v`` for any mix of spanning and
        touching fences (including pure spanning, pure touching, and mixed).
        Single unified cone-partition + label-pair classification.

        Returns ``(spanning_entries, local_chains)`` where:
          spanning_entries: ``(subtree, cone, pair_keys, mp)`` per cross-mp
            chain at ``v``, in the format consumed by the caller's
            ``spanning_by_chain`` pairing pass.
          local_chains: ``(Chain, pair_keys_a, pair_keys_b)`` per locally-
            paired chain at ``v``, in the format used by the caller for
            registration.

        Model: a spanning fence at ``v`` contributes one wall (its off-
        constraint endpoint at ``v``); its "other wall" is the chain-step
        constraint edge. A touching fence contributes two walls (both
        endpoints). Each ray is labelled with a subtree id (``C_ID`` for
        constraint bounds); chains are read off pairs of consecutive
        bound-rays in cw order. The chain-step ray (when any spanning
        fence is present at ``v``) has a dual face: ``C_ID`` on the void
        side, innermost spanning subtree on the navigable arc side —
        this lets the cone immediately inside the chain-step collapse
        against the spanning subtree when no touching is interposed,
        and to host the (touching ↔ spanning) cone when one is. With
        no spanning at ``v``, both constraint bounds carry the plain
        ``C_ID`` label and the chain partition reproduces the touching-
        only stack.
        """
        P = self.P
        constraint_bounds_v = self.constraint_bounds.get(v, set())
        if len(constraint_bounds_v) != 2:
            error(
                'expected 2 constraint bounds at chain-end %d, got %d',
                v,
                len(constraint_bounds_v),
            )
            return None

        # Identify chain_step direction (only meaningful when any spanning
        # fence ends at `v`). All spanning fences at `v` must agree on it.
        if spanning_endings:
            f0, side0 = spanning_endings[0]
            chain_step_nbr: int | None = (
                f0.primes_on_constraint[1]
                if side0 == 'start'
                else f0.primes_on_constraint[-2]
            )
            for fence, side in spanning_endings[1:]:
                other = (
                    fence.primes_on_constraint[1]
                    if side == 'start'
                    else fence.primes_on_constraint[-2]
                )
                if other != chain_step_nbr:
                    error(
                        'fences disagree on chain step at %d: %d vs %d',
                        v,
                        chain_step_nbr,
                        other,
                    )
                    return None
            if chain_step_nbr not in constraint_bounds_v:
                error(
                    'chain_step %d not in constraint_bounds at %d',
                    chain_step_nbr,
                    v,
                )
                return None
            cb_others = constraint_bounds_v - {chain_step_nbr}
            cb_other = next(iter(cb_others))
        else:
            # Pure touching at v: pick either cb as "chain_step" — both
            # constraint bounds behave symmetrically as plain C_ID labels;
            # this choice only affects which cone is cones[0] (the void
            # wedge), which is skipped regardless.
            chain_step_nbr = next(iter(constraint_bounds_v))
            cb_other = next(iter(constraint_bounds_v - {chain_step_nbr}))

        # cw direction so chain_step is adjacent to cb_other in `rotated`.
        cw_nbrs = list(P.neighbors_cw_order(v))
        n_cw = len(cw_nbrs)
        nbr_pos = {nb: i for i, nb in enumerate(cw_nbrs)}
        chain_step_pos = nbr_pos[chain_step_nbr]
        cw_offset = (nbr_pos[cb_other] - chain_step_pos) % n_cw
        ccw_offset = (chain_step_pos - nbr_pos[cb_other]) % n_cw
        cw_direction = cw_offset <= ccw_offset
        if cw_direction:
            rotated = [cw_nbrs[(chain_step_pos + i) % n_cw] for i in range(n_cw)]
        else:
            rotated = [chain_step_nbr] + [
                cw_nbrs[(chain_step_pos - i) % n_cw] for i in range(1, n_cw)
            ]
        rotated_pos = {nb: i for i, nb in enumerate(rotated)}

        # Wall labels (ray -> subtree). Spanning contributes one wall (anchor)
        # per fence; touching contributes two. A wall shared between a
        # spanning anchor and a touching endpoint is allowed iff their subtrees
        # match (same physical contour). Different subtrees sharing a wall is
        # geometrically ambiguous — reject.
        wall_label: dict[int, int] = {}
        spanning_by_anchor: dict[int, tuple[int, Fence, str]] = {}
        for fence, side in spanning_endings:
            anchor = fence.endpoints[0] if side == 'start' else fence.endpoints[1]
            if anchor in (chain_step_nbr, cb_other):
                error(
                    'spanning anchor %d at %d coincides with a constraint bound',
                    anchor,
                    v,
                )
                return None
            if anchor in spanning_by_anchor:
                error('two spanning fences share anchor %d at %d', anchor, v)
                return None
            spanning_by_anchor[anchor] = (fence.subtree, fence, side)
            existing = wall_label.get(anchor)
            if existing is not None and existing != fence.subtree:
                error(
                    'wall collision at %d, ray %d: subtrees %d vs %d',
                    v,
                    anchor,
                    existing,
                    fence.subtree,
                )
                return None
            wall_label[anchor] = fence.subtree
        for fence in touching:
            for endpoint in fence.endpoints:
                if endpoint in (chain_step_nbr, cb_other):
                    error(
                        'touching wall %d at %d coincides with a constraint bound',
                        endpoint,
                        v,
                    )
                    return None
                existing = wall_label.get(endpoint)
                if existing is not None and existing != fence.subtree:
                    error(
                        'wall collision at %d, ray %d: subtrees %d vs %d',
                        v,
                        endpoint,
                        existing,
                        fence.subtree,
                    )
                    return None
                wall_label[endpoint] = fence.subtree

        # Innermost spanning subtree = highest rotated_pos among anchors
        # (closest to chain_step on the navigable arc side). This subtree
        # labels the chain-step ray's navigable face — i.e., it occupies
        # the angular slot right beside chain_step on the nav arc. With
        # no spanning at `v`, this label is unused.
        C_ID = -1
        if spanning_by_anchor:
            innermost_anchor = max(
                spanning_by_anchor.keys(), key=rotated_pos.__getitem__
            )
            innermost_span_subtree = spanning_by_anchor[innermost_anchor][0]
        else:
            innermost_span_subtree = C_ID

        cone_bounds = set(wall_label.keys()) | {chain_step_nbr, cb_other}
        cones = self._partition_into_cones(v, cone_bounds, rotated)
        # By construction cones[0] is the wedge from chain_step (rotated[0])
        # to cb_other (rotated[1]) — the void wedge across the constraint.
        if not cones or cones[0][0] != chain_step_nbr or cones[0][1] != cb_other:
            error('cone partition misalignment at chain-end %d', v)
            return None

        def label_at(ray: int, on_void_side: bool) -> int:
            if ray == chain_step_nbr:
                return C_ID if on_void_side else innermost_span_subtree
            if ray == cb_other:
                return C_ID
            return wall_label[ray]

        # Group cones by label-pair (chain id at v).
        chain_cones_by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)
        for k, (left, right, _spokes, _keys) in enumerate(cones):
            on_void = k == 0
            ll = label_at(left, on_void)
            rl = label_at(right, on_void)
            if ll == rl:
                continue  # interior of a fence stack OR void
            if ll == C_ID and rl == C_ID:
                continue
            key = (ll, rl) if ll < rl else (rl, ll)
            chain_cones_by_pair[key].append(k)

        def make_cone(k: int) -> tuple[AccessCone, list[tuple[int, int]]]:
            left, right, spokes, keys = cones[k]
            if not cw_direction:
                left, right = right, left
                spokes = spokes[::-1]
            return AccessCone(v, left, right, spokes), keys

        spanning_entries: list[
            tuple[int, AccessCone, list[tuple[int, int]], list[int]]
        ] = []
        local_chains: list[
            tuple[Chain, list[tuple[int, int]], list[tuple[int, int]]]
        ] = []

        for (a, b), cone_indices in chain_cones_by_pair.items():
            span_anchors_in_pair = [
                (anchor, sub)
                for anchor, (sub, _, _) in spanning_by_anchor.items()
                if sub in (a, b)
            ]
            if len(cone_indices) == 1:
                # Cross-mp chain. Must involve a spanning fence at `v` that
                # has its chain step on the missing side. Owner convention:
                # the inner-of-the-pair spanning fence — the one whose anchor is
                # closer to chain_step on the navigable arc (higher
                # `rotated_pos`). For a (cb_b, span) chain there's only one
                # spanning fence in the pair; for (span_outer, span_inner)
                # we pick the inner.
                if not span_anchors_in_pair:
                    error(
                        'singleton chain (%d, %d) at %d with no spanning to cross mp',
                        a,
                        b,
                        v,
                    )
                    return None
                anchor, sub = max(span_anchors_in_pair, key=lambda c: rotated_pos[c[0]])
                _, fence, _ = spanning_by_anchor[anchor]
                cone_obj, keys = make_cone(cone_indices[0])
                spanning_entries.append(
                    (sub, cone_obj, keys, list(fence.primes_on_constraint))
                )
            elif len(cone_indices) == 2:
                # Locally paired chain (touching-style).
                # Owner subtree: route fence farther from chain_step in stack
                # (= smaller rotated_pos among its walls — closer to cb_other).
                if a == C_ID or b == C_ID:
                    owner = b if a == C_ID else a
                else:
                    walls_a = [w for w, s in wall_label.items() if s == a]
                    walls_b = [w for w, s in wall_label.items() if s == b]
                    min_pos_a = min(rotated_pos[w] for w in walls_a)
                    min_pos_b = min(rotated_pos[w] for w in walls_b)
                    owner = a if min_pos_a < min_pos_b else b
                cone_i, keys_i = make_cone(cone_indices[0])
                cone_j, keys_j = make_cone(cone_indices[1])
                chain = Chain(owner, (cone_i, cone_j), ([], []))
                local_chains.append((chain, keys_i, keys_j))
            else:
                error(
                    'chain (%d, %d) at %d has %d cones (expected 1 or 2)',
                    a,
                    b,
                    v,
                    len(cone_indices),
                )
                return None

        return spanning_entries, local_chains

    def _precompute_chains(
        self, fences: list[Fence]
    ) -> tuple[dict[tuple[int, int, int], tuple[Chain, int]], set[int]]:
        """Build the chain topology from the route fences.

        For every chain-end vertex (any vertex on the on-constraint segment
        of any fence), :meth:`_build_chains_at` partitions the cyclic-neighbor fan
        into cones, labels each ray by subtree (with the chain-step ray
        double-faced when any spanning fence ends there), and emits each
        chain as either a cross-mp spanning entry (one cone at this end,
        paired with the cone at the other mp end below) or a locally-paired
        chain (two cones at this end, paired in place).

        The fence-split in :meth:`__init__` guarantees every spanning fence walks
        contiguously along constraint edges through ``mp`` (any non-constraint
        hop, including one at either end, breaks the fence into separate
        sub-fences). So both ends of a spanning fence always host spanning
        topology — there is no one-end "demotion" case to handle here.

        Returns:
          chain_access: dict[(vertex, left, right) → (Chain, side)]
            Both pair orientations registered; lookup miss == "not a
            chain entry" — the trigger then does nothing and consumes no
            traversal budget.
          chain_end_set: set[int]  # vertices hosting any access cone.
        """
        spanning_at: dict[int, list[tuple[Fence, str]]] = defaultdict(list)
        touching_at: dict[int, list[Fence]] = defaultdict(list)
        for fence in fences:
            mp = fence.primes_on_constraint
            if len(mp) >= 2:
                # The split invariant (see docstring) makes both chain-step
                # neighbors constraint neighbors of their chain-ends, so the
                # fence spans at both ends.
                spanning_at[mp[0]].append((fence, 'start'))
                spanning_at[mp[-1]].append((fence, 'end'))
            else:
                touching_at[mp[0]].append(fence)

        # Detect dead-end spanning chains. When two spanning fences of the
        # *same* subtree meet at one chain-end vertex, both walls of the
        # corridor there belong to that subtree, so the corridor leads back
        # into the same tree — a dead-end pocket with no useful through-route,
        # not worth routing. (If the two off-constraint walls coincide on a
        # single node-vertex the corridor pinches to a point; if they are
        # distinct the inner wall still shadows the outer, so only one fence
        # could ever own the shared access cone. Both are treated the same: a
        # genuine through-chain in this configuration would need an extremely
        # contrived instance.) Mark such chains by key so they are dropped at
        # *both* mp-ends — dropping only one end would leave the other with a
        # lone cone that fails the 2-cone pairing below. Spanning fences of
        # *different* subtrees sharing a wall remain a genuine ambiguity,
        # handled (rejected) in `_build_chains_at`; chains of other subtrees
        # along the same border are untouched.
        dead_chain_keys: set[tuple[int, int, int]] = set()
        for endings in spanning_at.values():
            by_subtree: dict[int, list[Fence]] = defaultdict(list)
            for fence, _side in endings:
                by_subtree[fence.subtree].append(fence)
            for shared in by_subtree.values():
                if len(shared) >= 2:
                    for fence in shared:
                        mp = fence.primes_on_constraint
                        dead_chain_keys.add((fence.subtree, mp[0], mp[-1]))

        def is_dead(fence: Fence) -> bool:
            mp = fence.primes_on_constraint
            return (fence.subtree, mp[0], mp[-1]) in dead_chain_keys

        chain_access: dict[tuple[int, int, int], tuple[Chain, int]] = {}
        chain_end_set: set[int] = set()

        def register(
            v: int, pair_keys: list[tuple[int, int]], chain: Chain, side: int
        ) -> None:
            for pa, pb in pair_keys:
                chain_access[(v, pa, pb)] = (chain, side)
                chain_access[(v, pb, pa)] = (chain, side)

        # One unified builder for every chain-end vertex. It emits:
        #   - cross-mp spanning entries (paired across mp by the loop below)
        #   - locally-paired chains (registered immediately)
        # The grouping key (subtree, mp[0], mp[-1]) separates split sub-fences
        # sharing a subtree — `mp` is identical across both end-entries of a
        # given fence, and sub-fences from one A-edge split have disjoint
        # mp-end pairs by construction.
        spanning_by_chain: dict[
            tuple[int, int, int],
            list[tuple[int, AccessCone, list[tuple[int, int]], list[int]]],
        ] = defaultdict(list)
        chain_end_vertices = set(spanning_at) | set(touching_at)
        for v in chain_end_vertices:
            # Drop dead-end chains' fences at every vertex they touch, so no
            # half-chain survives. Other fences at `v` still build normally.
            v_spanning = [
                (f, side) for f, side in spanning_at.get(v, []) if not is_dead(f)
            ]
            v_touching = [f for f in touching_at.get(v, []) if not is_dead(f)]
            if not v_spanning and not v_touching:
                continue
            result = self._build_chains_at(v, v_spanning, v_touching)
            if result is None:
                continue
            chain_end_set.add(v)
            v_spanning_entries, v_local_chains = result
            for subtree, cone, pair_keys, mp in v_spanning_entries:
                spanning_by_chain[(subtree, mp[0], mp[-1])].append(
                    (v, cone, pair_keys, mp)
                )
            for ch, keys_a, keys_b in v_local_chains:
                register(v, keys_a, ch, 0)
                register(v, keys_b, ch, 1)

        # Pair each spanning chain's two end-cones (one per fence end) into a
        # Chain. By the split invariant every chain_key has exactly 2 entries.
        for chain_key, entries in spanning_by_chain.items():
            subtree = chain_key[0]
            if len(entries) != 2:
                error(
                    'spanning chain %s has %d access cones (expected 2)',
                    chain_key,
                    len(entries),
                )
                continue
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

        return chain_access, chain_end_set

    def _spawn_exit_cone(
        self,
        cone: AccessCone,
        pn_w_id: int,
        is_triangle_seen: bitarray,
    ) -> None:
        """Spawn end-spoke and intermediate-pair advancers covering exit
        through ``cone``. ``cone.left`` and ``cone.right`` are the wall-neighbor
        primes delimiting the wedge in CW order around ``cone.vertex``;
        ``cone.spokes`` are the non-bound spokes inside.
        """
        P = self.P
        paths = self.paths
        prioqueue = self.prioqueue
        portal_set = self.portal_set
        VertexC = self.VertexC
        best_pn_by_pair_id = self.best_pn_by_pair_id
        pair_id_by_prime_sector = self.pair_id_by_prime_sector
        w = cone.vertex
        pn_w = paths[pn_w_id]
        cum_turn_w = pn_w.cum_turn

        def _add_cone_exit_pn(v: int) -> tuple[int, float]:
            """Pseudonode at ``v`` parented by ``pn_w``; returns ``(pn_id, d_hop)``."""
            if v == w:
                return pn_w_id, 0.0
            d_hop = _node_dist(VertexC, w, v)
            d_total = pn_w.dist + d_hop
            sec_v = self._get_sector_from_opposite(v, w) if v >= 0 else NULL
            pn_v = paths.add(v, sec_v, pn_w_id, d_total, d_hop, cum_turn_w)
            pair_id = pair_id_by_prime_sector[(v, sec_v)]
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
        """Sector for a portal-side-trigger narrowing onto chain-end ``y``
        across portal ``(y, opposite)``. The cone at ``y`` the funnel just
        left is the wedge adjacent to ``opposite`` on the parent-triangle
        side; we resolve it by checking the two cones cyclically adjacent
        to ``opposite`` at ``y`` and returning the chain's subtree if exactly
        one of them is a chain interior. Returns NULL for non-chain cones
        or when both adjacent cones are chain interiors of different
        chains (overlapping fences: ``opposite`` alone can't disambiguate).
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
            pair_id = pair_id_by_prime_sector[(_new, sector_new)]
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
                '<%d> %s _new(%d) _nearside(%d) _farside(%d) _apex(%d),'
                ' _wedge_end: %d %d, _funnel: %s',
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
        P, R = self.P, self.R
        d2roots, d2rootsRank = self.d2roots, self.d2rootsRank
        iterations_limit = self.iterations_limit
        self.prioqueue = prioqueue = []
        num_traversals = defaultdict(lambda: 0)
        self.num_traversals = num_traversals
        traversals_limit = self.traversals_limit
        paths = self.paths = PathNodes()
        triangles = P.graph['triangles']
        portal_set = self.portal_set

        # launch channel traversers around the roots to the prioqueue
        best_pn_by_pair_id = self.best_pn_by_pair_id
        pair_id_by_prime_sector = self.pair_id_by_prime_sector
        fan_sectors = self.fan_sectors
        for r in range(-R, 0):
            paths[r] = PseudoNode(r, r, None, 0.0, 0.0, 0.0)
            paths.prime_from_pn[r] = r
            paths.pn_ids_from_prime_sector[r, r] = [r]
            for left in P.neighbors(r):
                right = P[r][left]['cw']
                portal = (left, right)
                portal_sorted = (right, left) if right < left else portal

                # Chain-ends adjacent to root in the fan are stepped over by
                # the regular init advancer (triangle/portal-side-trigger
                # fires on the far
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
                        # `(left, chain.subtree)` is always pre-registered by
                        # `_precompute_sector_lookup` (left is a chain-end =
                        # member of fence.primes_on_constraint with the same
                        # subtree id).
                        pair_id = pair_id_by_prime_sector[(left, chain.subtree)]
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

                # `_precompute_sector_lookup` already resolved & registered
                # the fan sectors for (r, left); both pairs always exist.
                sec_left, sec_right = fan_sectors[(r, left)]
                d_left = d2roots[left, r].item()
                d_right = d2roots[right, r].item()
                # add the first pseudo-nodes to paths
                wedge_end = [
                    paths.add(left, sec_left, r, d_left, d_left),
                    paths.add(right, sec_right, r, d_right, d_right),
                ]

                # shortest paths for roots' P.neighbors is a straight line
                best_pn_by_pair_id[pair_id_by_prime_sector[(left, sec_left)]] = (
                    wedge_end[0]
                )
                best_pn_by_pair_id[pair_id_by_prime_sector[(right, sec_right)]] = (
                    wedge_end[1]
                )

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
        Update G with the paths found by :meth:`_find_paths`.
        """
        get_best_path = self.get_best_path
        for n in range(self.T):
            path, dists = get_best_path(n)
            nx.add_path(G, path, kind='virtual')

    def best_paths_overlay(self) -> nx.Graph:
        """Merges the shortest paths for all nodes with ``G``.

        The output includes ``G``'s edges, excluding its feeders.

        Returns:
          Merged graph (pass to :func:`.plotting.gplot` or :func:`.svg.svgplot`).
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
        """Wrapper for :func:`.interarraylib.scaffolded`.

        Also unmarks (as real route, not background mesh) every navigation-mesh
        hop crossed by a route fence. :func:`.interarraylib.scaffolded` cannot do
        this on its own: a contour edge in ``G`` may be a ``shortened_contours``
        shortcut spanning several mesh hops through one shared clone, and only
        ``self.fences`` (built from ``A``'s midpaths during path-finding) has the
        fully expanded, conflict-resolved hop sequence.
        """
        scaff = scaffolded(self.G, P=self.P)
        for endpoints, primes_on_constraint, _ in self.fences:
            chain = (endpoints[0], *primes_on_constraint, endpoints[1])
            for a, b in zip(chain[:-1], chain[1:]):
                st = (a, b) if a < b else (b, a)
                if st in scaff.edges and 'kind' in scaff.edges[st]:
                    del scaff.edges[st]['kind']
        return scaff

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
                else [n, next(h for h in subtree if len(G._adj[h]) == 1)]  # type: ignore
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
                assert math.isclose(
                    total_parent_load, ref_load, rel_tol=1e-9, abs_tol=1e-9
                ), (
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
        # TODO: there might be some lost contour clones that could be pruned
        return G
