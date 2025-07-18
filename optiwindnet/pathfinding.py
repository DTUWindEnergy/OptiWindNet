# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import heapq
import logging
import math
from collections import defaultdict, namedtuple
from collections.abc import Iterable
from itertools import chain

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from .crossings import gateXing_iter
from .geometric import rotation_checkers_factory
from .interarraylib import bfs_subtree_loads, scaffolded
from .mesh import planar_flipped_by_routeset
from .utils import F

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
    """
    Router for gates that don't belong to the PlanarEmbedding of the graph.
    Initialize it with a detour-free routeset `G` and it will find paths from
    all nodes to the nearest root without crossing any used edges.

    Only edges in graph attribute 'tentative' or, lacking that, edges with the
    attribute 'kind' == 'tentative' are checked for crossings.

    These paths can be used to replace the existing gates that cross other
    edges by gate paths with detours.

    Example:
    ========

    H = PathFinder(G, planar=P, A=A).create_detours()
    """

    def __init__(
        self,
        Gʹ: nx.Graph,
        planar: nx.PlanarEmbedding,
        A: nx.Graph | None = None,
        branched: bool = True,
        iterations_limit: int = 10000,
    ) -> None:
        G = Gʹ.copy()
        R, T, B = (G.graph[k] for k in 'RTB')
        C = G.graph.get('C', 0)
        assert not G.graph.get('D'), 'Gʹ has already has detours.'

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
        clone2prime = list(G.graph.get('clone2prime', ()))
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
            clone_offset = T + B
            for (s, t), (midpath, shortpath) in shortened_contours.items():
                # G follows shortpath, but we want it to follow midpath
                subtree_id = G.nodes[t]['subtree']
                stored_edges = []
                path = [s] + shortpath + [t]
                for u_, v_ in zip(path[:-1], path[1:]):
                    u = u_ if u_ < T else (clone_offset + clone2prime.index(u_))
                    v = v_ if v_ < T else (clone_offset + clone2prime.index(v_))
                    stored_edges.append((u, v, G[u][v]))
                    # the nodes are left for later reuse
                    G.remove_edge(u, v)
                helper_edges = []
                u = s
                for v in midpath:
                    # this will use border nodes, watchout!
                    helper_edges.append((u, v))
                    G.add_edge(u, v, kind='contour')
                    G.nodes[v]['subtree'] = subtree_id
                    u = v
                helper_edges.append((u, t))
                G.add_edge(u, t, kind='contour')
                saved_shortened_contours.append((stored_edges, helper_edges))
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
        self.hooks2check, self.iterations_limit = hooks2check, iterations_limit
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
        if _node >= self.T:
            # _node is in a border or is in the supertriangle, which means it
            # is only reachable from one side, hence an arbitrary sector id.
            # _node may also be the prime of a contour node, which is ok, since
            # the other side is a pinched portal and has a distinct sector.
            return NULL
        _opposite = portal[0] if _node == portal[1] else portal[1]
        _nbr = self.P[_node][_opposite]['ccw']
        for _ in range(len(self.P._adj[_node])):
            if (_node, _nbr) in self.G.edges:
                return _nbr
            _nbr = self.P[_node][_nbr]['ccw']
        # could not find a non-tentative G edge around _node
        return NULL

    def _advance_portal(self, left: int, right: int):
        P = self.P
        while True:
            # look for children portals
            n = P[left][right]['ccw']
            if n not in P[right] or P[left][n]['ccw'] == right or n < 0:
                # (left, right, n) is not a triangle or n is a root
                return
            # examine the other two sides of the triangle
            next_portals = []
            for s, t, side in ((left, n, 1), (n, right, 0)):
                st_sorted = (s, t) if s < t else (t, s)
                if st_sorted not in self.portal_set:
                    continue
                next_portals.append(((s, t), side))
            try:
                # this `pop()` will raise IndexError if we are at a dead-end
                first, fside = next_portals.pop()
                # use this instead of the if-else-block when done debugging
                #  yield left, right, (
                #          self._portal_iter(*next_portals[0])
                #          if next_portals
                #          else None)
                if next_portals:
                    second, sside = next_portals[0]
                    debug('advance on (%d, %d) and branch on (%d, %d)', *first, *second)
                    yield (
                        first,
                        fside,
                        chain(((second, sside, None),), self._advance_portal(*second)),
                    )
                else:
                    debug('advance on (%d, %d)', *first)
                    yield first, fside, None
            except IndexError:
                # dead-end reached
                debug('dead-end: (%d, %d)', left, right)
                return
            left, right = first

    def _traverse_channel(
        self,
        d_ref: float,
        _apex: int,
        apex: int,
        _funnel: list[int],
        wedge_end: list[int],
        portal_iter: Iterable,
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
        
        # for next_left, next_right, new_portal_iter in portal_iter:
        for portal, side, new_portal_iter in portal_iter:
            #  print('[tra]')

            if new_portal_iter is not None:
                # spawn a branched traverser
                #  print(f'new channel {self.n2s(_apex, *_funnel)} -> '
                #        f"{F[_new]} {'RIGHT' if side else 'LEFT '}")
                branched_traverser = self._traverse_channel(
                    d_ref, _apex, apex, _funnel.copy(), wedge_end.copy(), new_portal_iter
                )
                self.bifurcation = d_ref, branched_traverser

            _new = portal[side]
            _nearside = _funnel[side]
            _farside = _funnel[not side]
            test = ccw if side else cw

            #  if _nearside == _apex:  # debug info
            #      print(f"{'RIGHT' if side else 'LEFT '} "
            #            f'nearside({F[_nearside]}) == apex({F[_apex]})')
            debug(
                    '%s _new(%d) _nearside(%d) _farside(%d) _apex(%d), _wedge_end: %d %d, _funnel: %s',
                'RIGHT' if side else 'LEFT ',
                _new,
                _nearside,
                _farside,
                _apex,
                paths.prime_from_id[wedge_end[0]],
                paths.prime_from_id[wedge_end[1]],
                _funnel
            )
            
            if _nearside == _apex or test(_nearside, _new, _apex):
                # not infranear
                if test(_farside, _new, _apex):
                    # ultrafar (⟨new, apex⟩ cuts farside)
                    debug('ultrafar')
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
                    debug('inside')
                _apex_eff, apex_eff = _apex, apex
                _funnel[side] = _new
            else:
                # infranear (⟨new, apex⟩ cuts nearside)
                debug('infranear')
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
            new_sector = self._get_sector(_new, portal)
            incumbent = I_path[_new].get(new_sector)
            is_better = incumbent is None or d_new < paths[incumbent].dist
            yield d_new, portal, (_new, _apex_eff), is_better
            new = self.paths.add(_new, new_sector, apex_eff, d_new, d_hop)
            self.uncharted[portal] = max(self.uncharted[portal] - 1, 0)
            # get incumbent again, as the situation may have changed
            incumbent = I_path[_new].get(new_sector)
            if incumbent is None or d_new < paths[incumbent].dist:
                self.I_path[_new][new_sector] = new
                debug('(%d, %d) added with d_path = %.2f', _new, _apex_eff, d_new)

            wedge_end[side] = paths.last_added
            d_ref = d_new

    def _find_paths(self):
        #  print('[exp] starting _explore()')
        G, P, R, T, B = self.G, self.P, self.R, self.T, self.B
        d2roots, d2rootsRank = self.d2roots, self.d2rootsRank
        iterations_limit = self.iterations_limit
        prioqueue = []
        # `uncharted` records whether portals have been traversed
        # (it is orientation-sensitive – two permutations)
        uncharted = defaultdict(lambda: 3)
        paths = self.paths = PathNodes()
        self.uncharted = uncharted
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
        ST = T + B
        self.ST = ST
        edges_P = {
            ((u, v) if u < v else (v, u)) for u, v in P.edges if u < ST or v < ST
        }
        constraint_edges = P.graph['constraint_edges']
        portal_set = (edges_P - edges_G_primed) - constraint_edges
        self.portal_set = portal_set

        counter = 0
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
                # flag initial portals as visited
                self.uncharted[portal] = 0
                self.uncharted[right, left] = 0

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
                closest, d_closest = (
                    (left, d_left)
                    if d2rootsRank[left, r] <= d2rootsRank[right, r]
                    else (right, d_right)
                )
                traverser = self._traverse_channel(
                    d_closest, r, r, [left, right], wedge_end, self._advance_portal(left, right),
                )
                heapq.heappush(prioqueue, (d_closest, counter, traverser))
                counter += 1
        # process edges in the prioqueue
        #  print(f'[exp] starting main loop, |prioqueue| = {len(prioqueue)}')
        _, _, traverser = heapq.heappop(prioqueue)
        iter = 0
        while len(prioqueue) > 0 and iter < iterations_limit:
            #  print(f'[exp] {iter}')
            iter += 1
            try:
                # make the traverser advance one portal
                d_contender, portal, hop, is_better = next(traverser)
            except StopIteration:
                #  print(f'[exp]_traverser {self.n2s(*hop)} was '
                #        'dropped (dead-end).')
                _, _, traverser = heapq.heappop(prioqueue)
            else:
                if is_better or uncharted[portal]:
                    #  print(f'[exp]_pushing dist = {d_contender:.0f}, '
                    #        f'{self.n2s(*hop)} ')
                    if self.bifurcation is not None:
                        d_contender_bif, traverser_bif = self.bifurcation
                        heapq.heappush(
                            prioqueue, (d_contender_bif, counter, traverser_bif)
                        )
                        counter += 1
                        self.bifurcation = None
                    _, _, traverser = heapq.heappushpop(
                        prioqueue, (d_contender, counter, traverser)
                    )
                    counter += 1
                else:
                    self.bifurcation = None
                    # traverser yields were bad, do not re-add to queue: get a new one
                    _, _, traverser = heapq.heappop(prioqueue)
                    
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
        )
        debug(
            '<PathFinder: created %d detour vertices, total length changed by %.2f%%',
            D,
            100 * detextra,
        )
        # TODO: there might be some lost contour clones that could be prunned
        return G
