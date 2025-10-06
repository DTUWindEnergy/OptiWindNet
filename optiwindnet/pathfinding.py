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
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from .crossings import gateXing_iter
from .geometric import rotation_checkers_factory
from .interarraylib import bfs_subtree_loads, scaffolded
from .mesh import planar_flipped_by_routeset

__all__ = ('PathFinder',)

_lggr = logging.getLogger(__name__)
debug, info, warn, error = _lggr.debug, _lggr.info, _lggr.warning, _lggr.error

NULL = np.iinfo(int).min
PseudoNode = namedtuple('PseudoNode', 'node sector parent dist d_hop'.split())


class PathNodes(dict):
    """Helper class to build a tree that uses clones of prime nodes
    (i.e. where the same prime node can appear as more than one node)."""

    count: int
    prime_from_id: dict
    ids_from_prime_sector: defaultdict
    last_added: int

    def __init__(self):
        super().__init__()
        self.count = 0
        self.prime_from_id = {}
        self.ids_from_prime_sector = defaultdict(list)
        self.last_added = NULL

    def add(
        self, _source: int, sector: int, parent: int, dist: float, d_hop: float
    ) -> int:
        if parent not in self:
            error(
                'attempted to add an edge in `PathNodes` to nonexistent parent (%d)',
                parent,
            )
        _parent = self.prime_from_id[parent]
        for prev_id in self.ids_from_prime_sector[_source, sector]:
            if self[prev_id].parent == parent:
                self.last_added = prev_id
                return prev_id
        id = self.count
        self.count += 1
        self[id] = PseudoNode(_source, sector, parent, dist, d_hop)
        self.ids_from_prime_sector[_source, sector].append(id)
        self.prime_from_id[id] = _source
        debug('pseudoedge «%d->%d» added', _source, _parent)
        self.last_added = id
        return id


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
      promissing_margin: fraction in excess of the best path that is still considered
        promissing, so that the traverser is allowed to proceed
      bad_streak_limit: limit on how many steps in a row without finding an improved
        path the traverser is allowed to take

    Example::

      P, A = make_planar_embedding(L)  # L represents the geometry of the location
      S = some_solver(A, ...)  # S is a topology
      G_tentative = G_from_S(S, A)  # G_tentative is almost a route set
      G = PathFinder(G_tentative, planar=P, A=A).create_detours()

    """

    def __init__(
        self,
        Gʹ: nx.Graph,
        planar: nx.PlanarEmbedding,
        A: nx.Graph | None = None,
        *,
        branched: bool = True,
        iterations_limit: int = 15000,
        traversals_limit: int = 2,
        promising_margin: float = 0.1,
        bad_streak_limit: int = 9,
    ) -> None:
        self.iterations_limit = iterations_limit
        self.traversals_limit = traversals_limit
        self.promising_margin = promising_margin
        self.bad_streak_limit = bad_streak_limit
        self.iterations = 0
        G = Gʹ.copy()
        R, T, B = (G.graph[k] for k in 'RTB')
        C = G.graph.get('C', 0)
        assert not G.graph.get('D'), 'Gʹ has already has detours.'
        self.ST = T + B

        # Block for facilitating the printing of debug messages.
        allnodes = np.arange(T + R + B + 3)
        allnodes[-R:] = range(-R, 0)
        #  self.n2s = NodeStr(allnodes, T + B + 3)

        debug(
            '>PathFinder: "%s" (T = %d)',
            G.graph.get('name') or G.graph.get('handle') or 'unnamed',
            T,
        )

        # tentative will be copied later, by initializing a set from it.
        tentative = G.graph.get('tentative')
        hooks2check = []
        if tentative is None:
            # TODO: this case should be removed ('tentative' attr mandatory)
            tentative = []
            for r in range(-R, 0):
                gates = set(
                    n for n in G.neighbors(r) if G[r][n].get('kind') == 'tentative'
                )
                tentative.extend((r, n) for n in gates)
                hooks2check.append(gates)
        else:
            hooks2check.extend(set() for _ in range(R))
            for r, n in tentative:
                hooks2check[r].add(n)

        Xings = list(
            gateXing_iter(
                G,
                hooks=[
                    np.fromiter(h2c, count=len(h2c), dtype=int) for h2c in hooks2check
                ],
                borders=planar.graph.get('constraint_edges'),
            )
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
        #  clone2prime = list(G.graph.get('clone2prime', ()))
        # TODO: work around PathFinder getting metrics for the supertriangle
        #       nodes -> do away with A metrics, eliminate A from args
        if A is None:
            VertexC = G.graph['VertexC']
            supertriangleC = planar.graph['supertriangleC']
            if G.graph.get('is_normalized'):
                supertriangleC = G.graph['norm_scale'] * (
                    supertriangleC - G.graph['norm_offset']
                )
            VertexC = np.vstack((VertexC[: T + B], supertriangleC, VertexC[-R:]))
            d2roots = cdist(VertexC[:-R], VertexC[-R:])
            Rank = None
            diagonals = None
        else:
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
            edges_to_remove = []
            edges_to_add = []
            clone_offset = T + B
            for (s, t), (midpath, shortpath) in shortened_contours.items():
                # G follows shortpath, but we want it to follow midpath
                subtree_id = G.nodes[t]['subtree']
                stored_edges = []
                u = s
                if shortpath:
                    # there may be more than one edge cloning same border vertex
                    choices = [
                        v for v in G[u] if v >= clone_offset and fnT[v] == shortpath[0]
                    ]
                    if len(choices) > 1:
                        # checks just one more hop -> bizarre cases may lead to error
                        nb = t if len(shortpath) <= 1 else shortpath[1]
                        for v in choices:
                            if (G._adj[v].keys() - {u}).pop() == nb:
                                break
                    else:
                        v = choices[0]
                else:
                    v = t
                while v != t:
                    stored_edges.append((u, v, G[u][v]))
                    edges_to_remove.append((u, v))
                    u, v = v, (G._adj[v].keys() - {u}).pop()
                stored_edges.append((u, v, G[u][v]))
                edges_to_remove.append((u, v))
                helper_edges = []
                u = s
                for v in midpath:
                    # this will use border nodes, watchout!
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
        self.hooks2check = hooks2check
        self.num_revisits = 0
        self.adv_counter = 0
        self._find_paths()

    def get_best_path(self, n: int):
        """
        `_.get_best_path(«node»)` produces a `tuple(path, dists)`.
        `path` contains a sequence of nodes from the original
        networx.Graph `G`, from «node» to the closest root.
        `dists` contains the lengths of the segments defined by `paths`.
        """
        paths = self.paths
        paths_available = tuple((paths[id].dist, id) for id in self.I_path[n].values())
        if paths_available:
            _, id = min(paths_available)
            path = [n]
            dists = []
            pseudonode = paths[id]
            while id >= 0:
                dists.append(pseudonode.d_hop)
                id = pseudonode.parent
                path.append(paths.prime_from_id[id])
                pseudonode = paths[id]
            return path, dists
        else:
            info('Path not found for «%d»', n)
            return [], []

    def _get_sector(self, _node: int, portal: tuple[int, int]):
        """
        Given a `_node` and a `portal` to which `_node` belongs, visit the
        neighbors of `_node` starting from from the opposite node in `portal`
        and rotating in the counter-clockwise direction.
        The first neighbor that forms one of G's edges with `_node` is the
        sector. The sector is a way of identifying from which side of a
        non-traversable barrier the path is reaching `_node`.
        """
        T = self.T
        G = self.G
        P = self.P
        tentative = self.tentative
        if _node >= T:
            # _node is in a constraint edge, is a contour or is in the supertriangle,
            # hence it is only reachable from one side -> arbitrary sector id
            return NULL
        _opposite = portal[0] if _node == portal[1] else portal[1]
        if _opposite in G._adj[_node]:
            # special case: visiting a DEAD-END
            return _opposite
        _nbr = P[_node][_opposite]['ccw']
        for _ in range(len(P._adj[_node])):
            if _nbr < T and _nbr in G[_node]:
                if _nbr >= 0 or (_nbr, _node) not in tentative:
                    return _nbr
            _nbr = P[_node][_nbr]['ccw']
        # could not find a non-tentative G edge around _node
        return NULL

    def _advance_portal(
        self,
        adv_id: int,
        portal: tuple[int, int],
        traverser_args: tuple,
        is_triangle_seen: bitarray,
        side: int | None = None,
    ):
        P = self.P
        T = self.T
        prioqueue = self.prioqueue
        portal_set = self.portal_set
        triangles = P.graph['triangles']
        traverser = self._traverse_channel(adv_id, *traverser_args)
        next(traverser)
        if side is not None:
            d_ref, is_promising = traverser.send((portal, side))
            yield d_ref, portal, is_promising
            next(traverser)
        while True:
            # look for children portals
            left, right = portal
            n = P[left][right]['ccw']
            if n not in P[right] or P[left][n]['ccw'] == right or n < 0:
                debug('{%d} advancer reached DEAD-END (root or mesh edge)', adv_id)
                return
            triangle_idx = bisect_left(triangles, tuple(sorted([left, right, n])))
            if is_triangle_seen[triangle_idx]:
                debug('{%d} advancer revisited triangle', adv_id)
                return
            is_triangle_seen[triangle_idx] = 1
            # check whether the other two sides of the triangle are portals
            portals = [
                (portal, side)
                for portal, side in (((left, n), 1), ((n, right), 0))
                if portal in portal_set
            ]
            if len(portals) == 2:
                portal_bif, side_bif = portals[1]
                # channel bifurcation, spawn new advancer
                #  trace('{%d} advancer asking for traverser_args', adv_id)
                # get traverser state
                traverser_args = next(traverser)
                d_ref = traverser_args[0]
                heapq.heappush(
                    prioqueue,
                    (
                        d_ref,
                        self.adv_counter,
                        self._advance_portal(
                            self.adv_counter,
                            portal_bif,
                            traverser_args,
                            is_triangle_seen.copy(),
                            side_bif,
                        ),
                    ),
                )
                self.adv_counter += 1
                next(traverser)
            elif not portals:
                # DEAD-END: both triangle sides are not portals
                if 0 <= n <= T:
                    # there is a (node, sector) to update inside the dead-end
                    d_ref, is_promising = traverser.send(((left, n), 1))
                    # no need to yield, but make sure the last path pnode is added
                    next(traverser)
                debug('{%d} advancer reached DEAD-END (not portals)', adv_id)
                return
            # process  portal
            portal, side = portals[0]
            #  trace('{%d} advancer sending (portal, side)', adv_id)
            d_ref, is_promising = traverser.send((portal, side))
            yield d_ref, portal, is_promising
            next(traverser)

    def _traverse_channel(
        self,
        trav_id,
        d_ref: float,
        _apex: int,
        apex: int,
        _funnel: list[int],
        wedge_end: list[int],
        bad_streak: int = 0,
    ):
        # variable naming notation:
        # for variables that represent a node, they may occur in two versions:
        #     - _node: the index it contains maps to a coordinate in VertexC
        #     - node: contains a pseudonode index (i.e. an index in self.paths)
        #             translation: _node = paths.prime_from_id[node]
        cw, ccw = rotation_checkers_factory(self.VertexC)
        paths = self.paths
        I_path = self.I_path
        ST = self.ST
        num_traversals = self.num_traversals
        promising_bar = 1.0 + self.promising_margin
        bad_streak_limit = self.bad_streak_limit

        # for next_left, next_right, new_portal_iter in portal_iter:
        while True:
            #  trace('<%d> traverser before first yield', trav_id)
            adv_msg = yield
            if adv_msg is None:
                #  trace('<%d> new traverser sent for evaluation', trav_id)
                yield (
                    d_ref,
                    _apex,
                    apex,
                    _funnel.copy(),
                    wedge_end.copy(),
                    bad_streak,
                )
                continue
            else:
                portal, side = adv_msg
            #  trace('<%d> got (portal, side)', trav_id)

            _new = portal[side]
            sector_new = self._get_sector(_new, portal)
            _nearside = _funnel[side]
            _farside = _funnel[not side]
            test = ccw if side else cw

            #  if _nearside == _apex:  # debug info
            #      print(f"{'RIGHT' if side else 'LEFT '} "
            #            f'nearside({F[_nearside]}) == apex({F[_apex]})')
            debug(
                '<%d> %s _new(%d) _nearside(%d) _farside(%d) _apex(%d), _wedge_end: %d %d, _funnel: %s',
                trav_id,
                'RIGHT' if side else 'LEFT ',
                _new,
                _nearside,
                _farside,
                _apex,
                paths.prime_from_id[wedge_end[0]],
                paths.prime_from_id[wedge_end[1]],
                _funnel,
            )

            if _nearside == _apex or test(_nearside, _new, _apex):
                # not infranear
                if test(_farside, _new, _apex):
                    # ultrafar (⟨new, apex⟩ cuts farside)
                    debug('<%d> ultrafar', trav_id)
                    current_wapex = wedge_end[not side]
                    _current_wapex = paths.prime_from_id[current_wapex]
                    _funnel[not side] = _current_wapex
                    contender_wapex = paths[current_wapex].parent
                    _contender_wapex = paths.prime_from_id[contender_wapex]
                    #  print(f"{'RIGHT' if side else 'LEFT '} "
                    #        f'current_wapex({F[_current_wapex]}) '
                    #        f'contender_wapex({F[_contender_wapex]})')
                    while (
                        _current_wapex != _farside
                        and _contender_wapex >= 0
                        and test(_new, _current_wapex, _contender_wapex)
                    ):
                        _funnel[not side] = _current_wapex
                        #  wedge_end[not side] = current_wapex
                        current_wapex = contender_wapex
                        _current_wapex = _contender_wapex
                        contender_wapex = paths[current_wapex].parent
                        _contender_wapex = paths.prime_from_id[contender_wapex]
                        #  print(f"{'RIGHT' if side else 'LEFT '} "
                        #        f'current_wapex({F[_current_wapex]}) '
                        #        f'contender_wapex({F[_contender_wapex]})')
                    _apex = _current_wapex
                    apex = current_wapex
                else:
                    # not ultrafar nor infranear (⟨new, apex⟩ in line-of-sight)
                    debug('<%d> inside', trav_id)
                _apex_eff, apex_eff = _apex, apex
                _funnel[side] = _new
            else:
                # infranear (⟨new, apex⟩ cuts nearside)
                debug('<%d> infranear', trav_id)
                current_wapex = wedge_end[side]
                _current_wapex = paths.prime_from_id[current_wapex]
                #  print(f'{F[_current_wapex]}')
                contender_wapex = paths[current_wapex].parent
                _contender_wapex = paths.prime_from_id[contender_wapex]
                while (
                    _current_wapex != _nearside
                    and _contender_wapex >= 0
                    and test(_current_wapex, _new, _contender_wapex)
                ):
                    current_wapex = contender_wapex
                    _current_wapex = _contender_wapex
                    #  print(f'{F[current_wapex]}')
                    contender_wapex = paths[current_wapex].parent
                    _contender_wapex = paths.prime_from_id[contender_wapex]
                _apex_eff, apex_eff = _current_wapex, current_wapex

            # rate, wait, add
            d_hop = np.hypot(*(self.VertexC[_apex_eff] - self.VertexC[_new]).T).item()
            pseudoapex = paths[apex_eff]
            d_new = pseudoapex.dist + d_hop
            keeper = I_path[_new].get(sector_new)
            is_promising = bad_streak < bad_streak_limit and (
                keeper is None or d_new < promising_bar * paths[keeper].dist
            )
            # for supertriangle vertices, do not update the d_ref used for prioritizing
            # (it would be sent to the bottom of heapq beacause of the big distances)
            if _new < ST:
                d_ref = d_new
            yield d_ref, is_promising
            #  trace('<%d> traverser after second yield', trav_id)
            new = self.paths.add(_new, sector_new, apex_eff, d_new, d_hop)
            num_traversals[portal] += 1
            # get keeper again, as the situation may have changed
            keeper = I_path[_new].get(sector_new)
            if keeper is None or d_new < paths[keeper].dist:
                self.I_path[_new][sector_new] = new
                debug(
                    '<%d> new keeper for (%d, %d) via %d: d_path = %.2f',
                    trav_id,
                    _new,
                    sector_new,
                    _apex_eff,
                    d_new,
                )
                bad_streak = 0
            elif not math.isclose(d_new, paths[keeper].dist):
                bad_streak += 1

            wedge_end[side] = paths.last_added

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
        self.bifurcation = None
        I_path = defaultdict(dict)
        self.I_path = I_path

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

        # launch channel traversers around the roots to the prioqueue
        for r in range(-R, 0):
            paths[r] = PseudoNode(r, r, None, 0.0, 0.0)
            paths.prime_from_id[r] = r
            paths.ids_from_prime_sector[r, r] = [r]
            for left in P.neighbors(r):
                right = P[r][left]['cw']
                portal = (left, right)
                portal_sorted = (right, left) if right < left else portal
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
                    while True:
                        sec_left = P[left][sec_left]['ccw']
                        incr_edge = (
                            (sec_left, left) if sec_left < left else (left, sec_left)
                        )
                        if incr_edge in edges_G_primed or incr_edge in constraint_edges:
                            break

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
                I_path[left][sec_left], I_path[right][sec_right] = wedge_end

                # prioritize by distance to the closest node of the portal
                d_closest = (
                    d_left if d2rootsRank[left, r] <= d2rootsRank[right, r] else d_right
                )
                traverser_pack = (
                    d_closest,
                    r,
                    r,
                    [left, right],
                    wedge_end,
                    0,
                )
                advancer = self._advance_portal(
                    self.adv_counter,
                    (left, right),
                    traverser_pack,
                    bitarray(len(triangles)),
                )
                heapq.heappush(prioqueue, (d_closest, self.adv_counter, advancer))
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
                d_ref, portal, is_promising = next(advancer)
            except StopIteration:
                # advancer decided to stop, get a new one
                if not prioqueue:
                    break
                _, adv_id, advancer = heapq.heappop(prioqueue)
            else:
                if is_promising or num_traversals[portal] < traversals_limit:
                    # advancer is still promising, push it back to queue and get top one
                    _, adv_id, advancer = heapq.heappushpop(
                        prioqueue, (d_ref, adv_id, advancer)
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
            for id in self.I_path[n].values():
                if id < 0:
                    # n is a root's neighbor
                    continue
            path, dists = get_best_path(n)
            nx.add_path(G, path, kind='virtual')

    def best_paths_overlay(self) -> nx.Graph:
        """Merges the shortest paths for all nodes with `G`.

        The output includes `G`'s edges, excluding its gates.

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
        """Reroute all gate edges in G with crossings using detour paths.

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
        paths, I_path = self.paths, self.I_path
        clone_idx = T + B + C
        failed_detours = []

        subtree_from_subtree_id = defaultdict(list)
        subtree_id_from_n = {}
        for n in chain(range(T), range(T + B, clone_idx)):
            subtree_id = G.nodes[n]['subtree']
            subtree_from_subtree_id[subtree_id].append(n)
            subtree_id_from_n[n] = subtree_id

        for r, n in set(gate for _, gate in Xings):
            tentative.remove((r, n))
            subtree_id = subtree_id_from_n[n]
            subtree = subtree_from_subtree_id[subtree_id]
            subtree_load = G.nodes[n]['load']
            # set of nodes to examine is different depending on `branched`
            hookchoices = (
                [n for n in subtree if n < T]
                if self.branched
                else [n, next(h for h in subtree if len(G._adj[h]) == 1)]
            )
            debug('hookchoices: %s', hookchoices)

            path_options = list(
                chain.from_iterable(
                    (
                        (paths[id].dist, id, hook, sec)
                        for sec, id in I_path[hook].items()
                    )
                    for hook in hookchoices
                )
            )
            if not path_options:
                error(
                    'subtree of node %d has no non-crossing paths to '
                    'any root: leaving gate as-is',
                    n,
                )
                # unable to fix this crossing
                failed_detours.append((r, n))
                continue
            dist, id, hook, sect = min(path_options)
            debug('best: hook = %d, sector = %d, dist = %.2f', hook, sect, dist)

            path = [hook]
            dists = []
            pseudonode = paths[id]
            while id >= 0:
                dists.append(pseudonode.d_hop)
                id = pseudonode.parent
                path.append(paths.prime_from_id[id])
                pseudonode = paths[id]
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
                # TODO: adapt this for contoured gates
                #       maybe that's the place to prune contour clones
                G.remove_edge(r, n)
                if r != path[-1]:
                    debug(
                        'root changed from %d to %d for subtree of gate %d, '
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
                    'gate %d–%d touches a node (touched node does not become'
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

        # former tentative gates that were not in Xings cease to be tentative
        for r, n in tentative:
            del G[r][n]['kind']

        if failed_detours:
            warn('Failed: %s', failed_detours)
            G.graph['tentative'] = failed_detours
        else:
            del G.graph['tentative']

        D = clone_idx - T - B - C
        detextra = G.size(weight='length') / self.predetour_length - 1
        stunts_primes = G.graph.pop('stunts_primes', False)
        if stunts_primes:
            num_stunts = len(stunts_primes)
            G = nx.relabel_nodes(
                G,
                {clone: clone - num_stunts for clone in range(T + B, clone_idx)},
                copy=False,
            )
            clone_idx -= num_stunts
            B -= num_stunts
            VertexC = G.graph['VertexC']
            G.graph['VertexC'] = np.vstack((VertexC[: T + B], VertexC[-R:]))
            if clone2prime:
                for stunt, prime in enumerate(stunts_primes, start=T + B):
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
