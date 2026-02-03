# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import time

import numpy as np
import networkx as nx
from scipy.stats import rankdata
from bitarray import bitarray
from bitarray.util import ones, zeros

from ..crossings import edge_conflicts
from ..geometric import (
    angle_oracles_factory,
)
from ..interarraylib import (
    fun_fingerprint,
    calcload,
    add_link_blockmap,
    add_terminal_closest_root,
)
from .priorityqueue import PriorityQueue

__all__ = ()

_lggr = logging.getLogger(__name__)
debug, info, warn, error = _lggr.debug, _lggr.info, _lggr.warning, _lggr.error

_ONE = bitarray('1')

# empirically obtained coefficients
_empirical_rootlust = {
    2: (0.199963337895445, 0.08426213176671275),
    3: (0.242814965204555, 0.039061695424637846),
    4: (0.18000159349690245, 0.08300298311928281),
    5: (0.14867313705016125, 0.0801649122840704),
    6: (0.06253058300839699, 0.09461742952868712),
    7: (0.15377762467951342, 0.049933595137178356),
    8: (0.17138512281655421, 0.034559581686989825),
    9: (0.16983803760158625, 0.03466468487805137),
    10: (0.17219919344831916, 0.04040308699369742),
    11: (0.13499063146609241, 0.0566116601450132),
    12: (0.15550714746208458, 0.02539994948567122),
}


def constructor(
    Aʹ: nx.Graph,
    capacity: int,
    method: str = 'esau_williams',
    rootlust_: tuple[float, float] = (),
    maxiter: int = 10000,
    keep_log: bool = False,
    blockage_link_cos_lim: float = 0.85,  # 30°
    blockage_link_feeder_lim: float = 2.0,
    blockage_subtree_feeder_lim: float = 2.5,
) -> nx.Graph:
    """Create a network using a constructive greedy heuristic.

    Methods:
    - 'esau_williams': Esau-Williams C-MST heuristic modified to avoid crossings.
    - 'modified_EW': EW further modified with a small bias towards moving radially.
    - 'rootlust': EW further modified so that the lust towards root increases as capacity decreases.

    Args:
      Aʹ: available links graph
      capacity: max number of terminals in a subtree
      method: choice of method (see list above)
      maxiter: fail-safe to avoid locking in an infinite loop

    Returns:
      Solution topology S.
    """

    start_time = time.perf_counter()
    R, T = (Aʹ.graph[k] for k in 'RT')
    _T = range(T)
    VertexC = Aʹ.graph['VertexC']
    diagonals = Aʹ.graph['diagonals']
    d2roots = Aʹ.graph['d2roots']
    S = nx.Graph(R=R, T=T)
    A = Aʹ.copy()
    P_A = A.graph['planar']

    roots = range(-R, 0)

    add_terminal_closest_root(A)
    rootmask__ = A.graph['rootmask__']
    d2rootsRank = rankdata(d2roots, method='dense', axis=0)
    if not rootlust_:
        rootlust_ = _empirical_rootlust[capacity]
    else:
        rootlust_ = rootlust_[0], rootlust_[1] / (capacity - 1)

    # removing root nodes from A to speedup enqueue_best_union
    # this may be done because G already starts with feeders
    A.remove_nodes_from(roots)
    # END: prepare auxiliary graph with all allowed edges and metrics

    # ensure roots are added, even if the star graph uses a subset of them
    S.add_nodes_from(roots)
    # BEGIN: create initial star graph
    #  S.add_edges_from(((n, r) for n, r in A.nodes(data='root')))
    # END: create initial star graph

    # BEGIN: helper data structures

    # mappings from nodes
    # <subtree_>: maps nodes to the list of nodes in their subtree
    subtree_ = [zeros(T) for _ in _T]
    for t, subtree in zip(_T, subtree_):
        subtree[t] = True
    # <subroot_>: maps terminals to their subroots
    subroot_ = list(_T)
    last_hop_of = [t for t in _T]
    hooks_of = [[t] for t in _T]
    # <is_stale_>: mask of stale subroots (in need of target refresh)
    is_stale_ = ones(T)
    # <is_not_full_>: mask of subroots with spare capacity
    is_not_full_ = ones(T)
    # memory allocation for temporary constructs
    # <to_retarget_>: mask of subroots that are stale and not full
    to_retarget_ = bitarray(T)

    # mappings from components (indexed by their subroots)
    # <subtree_span__>: pairs (most_CW, most_CCW) of extreme nodes of each
    #                  subtree
    subtree_span__ = [[(t, t) for _ in roots] for t in _T]
    # <subtree_blocked__>: sets of blocked terminals from other components wrt each root
    subtree_blocked__ = [[bitarray(T) for _ in _T] for _ in roots]

    # mappings from components (identified by their subroots)
    # <who_targets_>: maps component to set of components queued to merge in
    who_targets_ = [set() for _ in _T]

    # other structures
    # <last_hops_count_>: number of times a particular node is used as the hop closest
    #                   to root in the feeder link
    last_hops_count_ = [1 for _ in _T]
    # <pq>: the smaller the payload, the higher the priority
    pq = PriorityQueue()
    # enqueue_best_union()
    # <i>: iteration counter
    i = 0
    # <prevented_crossing>: counter for edges discarded due to crossings
    prevented_crossings = 0
    # END: helper data structures

    # BEGIN: alternative methods of selecting the best edge to expand components
    def find_union_esau_williams_tradeoff(subroot):
        # Esau, L. R., and K. C. Williams.
        # “On Teleprocessing System Design, Part II: A Method for Approximating the Optimal Network.”
        # IBM Systems Journal 5, no. 3 (1966): 142–47. https://doi.org/10.1147/sj.53.0142.
        subtree = subtree_[subroot]
        capacity_left = capacity - subtree.count()
        choices = []
        edges2discard = []
        for u in subtree.search(_ONE):
            for v in A[u]:
                if (
                    subroot_[v] == subroot
                    or subtree_[subroot_[v]].count() > capacity_left
                ):
                    # useless edges
                    edges2discard.append((u, v))
                else:
                    tradeoff = (
                        A[u][v]['length'] - d2roots[subroot, A.nodes[subroot]['root']]
                    )
                    if tradeoff <= 0:
                        # useful edges
                        # v's proximity to root is used as tie-breaker
                        choices.append(
                            (tradeoff, d2rootsRank[v, A.nodes[v]['root']], u, v)
                        )
        return (min(choices) if choices else ()), edges2discard

    margin = 1.02

    def find_union_mod_esau_williams_tradeoff(subroot):
        subtree = subtree_[subroot]
        capacity_left = capacity - subtree.count()
        sr_d2root = d2roots[subroot, A.nodes[subroot]['root']]
        choices = []
        edges2discard = []
        for u in subtree.search(_ONE):
            for v in A[u]:
                if (
                    subroot_[v] == subroot
                    or subtree_[subroot_[v]].count() > capacity_left
                ):
                    # useless edges
                    edges2discard.append((u, v))
                else:
                    extent = A[u][v]['length']
                    if extent <= sr_d2root:
                        # useful edges
                        # v's proximity to root is used as tie-breaker
                        choices.append(
                            (extent, d2rootsRank[v, A.nodes[v]['root']], u, v)
                        )
        if choices:
            choices.sort()
            best_extent, best_rank, *best_edge = choices[0]
            for extent, rank, *edge in choices[1:]:
                if extent > margin * best_extent:
                    # no more edges within margin
                    break
                if rank < best_rank:
                    best_extent, best_rank, best_edge = extent, rank, edge
            return (best_extent - sr_d2root, best_rank, *best_edge), edges2discard
        else:
            return (), edges2discard

    def find_union_rootlust(subroot):
        # gather all the edges leaving the subtree of subroot
        subtree = subtree_[subroot]
        root = A.nodes[subroot]['root']
        d2root = d2roots[subroot, root]
        capacity_used = subtree.count()
        capacity_left = capacity - capacity_used
        rootlust = rootlust_[0] + rootlust_[1] * capacity_used
        choices = []
        edges2discard = []
        for u in subtree.search(_ONE):
            for v in A[u]:
                sr_v = subroot_[v]
                if sr_v == subroot or subtree_[sr_v].count() > capacity_left:
                    # useless edges
                    edges2discard.append((u, v))
                else:
                    W = A[u][v]['length']
                    if W <= d2root:
                        # useful edges
                        root_v = A.nodes[v]['root']
                        d2rGain = d2root - d2roots[sr_v, root_v]
                        tiebreaker = d2rootsRank[v, root_v]
                        choices.append(
                            (W - d2root - d2rGain * rootlust, tiebreaker, u, v)
                        )
        return (min(choices) if choices else ()), edges2discard

    # END: alternative methods of selecting the best edge to expand components

    def savings_outweight_detours(sr_dropped, u, v):
        """Note: the detour_increase calculated here is an estimate."""
        sr_kept = subroot_[v]
        blocked__ = A[u][v]['blocked__']
        detour_increase = 0.0
        clones = []
        is_last_hop_ = bitarray(count > 0 for count in last_hops_count_)
        union_ = subtree_[sr_dropped] | subtree_[sr_kept]
        for r, rootmask_, blocked_ in zip(roots, rootmask__, blocked__):
            # subtree_span__[sr_kept] gives the span of the union because it was
            #   updated just before calling this function
            lo, hi = subtree_span__[sr_kept][r]
            for hop in (rootmask_ & is_last_hop_ & (blocked_ | union_)).search(_ONE):
                count = last_hops_count_[hop] - (hop == sr_dropped) - (hop == sr_kept)
                if hop == lo or hop == hi or count == 0:
                    # no increase if the union span is not extending beyond hop's angle
                    continue
                extent_lo = d2roots[lo, r] + np.hypot(*(VertexC[lo] - VertexC[hop]))
                extent_hi = d2roots[hi, r] + np.hypot(*(VertexC[hi] - VertexC[hop]))
                extent, clone = (
                    (extent_lo, lo) if extent_lo <= extent_hi else (extent_hi, hi)
                )
                detour_increase += count * (extent - d2roots[hop, r]).item()
                clones.append((hop, clone, count))
        savings = d2roots[sr_dropped].min().item() - A[u][v]['length']
        if clones:
            debug(
                'savings of %.3f vs. detour increase of %.3f for rerouting %s',
                savings,
                detour_increase,
                clones,
            )
        return savings, detour_increase, clones

    if method == 'esau_williams':
        find_union = find_union_esau_williams_tradeoff
    elif method == 'modified_EW':
        find_union = find_union_mod_esau_williams_tradeoff
    elif method == 'rootlust':
        add_link_blockmap(A)
        find_union = find_union_rootlust
        angle__, angle_rank__ = A.graph['angle__'], A.graph['angle_rank__']
        union_limits, angle_ccw = angle_oracles_factory(angle__, angle_rank__)

    def enqueue_best_union(subroot):
        debug('<enqueue_best_union> starting... subroot = <%d>', subroot)
        best_choice, edges2discard = find_union(subroot)
        A.remove_edges_from(edges2discard)
        if best_choice:
            priority, _, u, v = best_choice
            pq.add(priority, subroot, (u, v))
            who_targets_[subroot_[v]].add(subroot)
            debug(
                '<pushed> sr_u <%d>, «%d~%d», priority = %.3f', subroot, u, v, priority
            )
        else:
            # no viable edge is shorter than subroot for this subtree
            debug('<cancelling> %d', subroot)
            if subroot in pq.tags:
                pq.cancel(subroot)

    # initialize pq
    for n in _T:
        enqueue_best_union(n)

    # BEGIN: main loop
    while True:
        i += 1
        if i > maxiter:
            error('maxiter reached (%d)', i)
            break
        debug('[%d]', i)
        to_retarget_[:] = is_stale_ & is_not_full_
        if to_retarget_.any():
            stale_subtrees = tuple(to_retarget_.search(_ONE))
            debug('stale_subtrees: %s', stale_subtrees)
            for subtree in stale_subtrees:
                enqueue_best_union(subtree)
            is_stale_.setall(0)
        if not pq:
            # finished
            break
        sr_dropped, (u, v) = pq.top()
        debug('<popped> «%d~%d», sr_dropped: <%d>', u, v, sr_dropped)

        # check if the union still makes sense
        if (u, v) not in A.edges:
            debug('<discard> «%d~%d» not in A anymore', u, v)
            continue

        sr_kept = subroot_[v]
        root = A.nodes[sr_kept]['root']

        if method == 'rootlust':
            # assess the union's angle span
            union_span_ = [
                union_limits(
                    r, u, *subtree_span__[sr_dropped][r], v, *subtree_span__[sr_kept][r]
                )
                for r in roots
            ]
            debug('<angle_span> //%s//', union_span_[root])
            # update the component's angle span
            subtree_span__[sr_kept], sr_kept_old_span = (
                union_span_,
                subtree_span__[sr_kept],
            )

            savings, detour_increase, clones = savings_outweight_detours(
                sr_dropped, u, v
            )
            if savings < detour_increase:
                debug('<discard> «%d~%d» too long detours', u, v)
                subtree_span__[sr_kept] = sr_kept_old_span
                continue

            # update the last hops
            last_hop_dropped = last_hop_of[sr_dropped]
            last_hops_count_[last_hop_dropped] -= 1
            hooks_of[last_hop_dropped].remove(sr_dropped)
            last_hop_of[sr_dropped] = None
            # reroute
            for old_last_hop, new_last_hop, count in clones:
                last_hops_count_[old_last_hop] -= count
                last_hops_count_[new_last_hop] += count
                for hook in hooks_of[old_last_hop]:
                    last_hop_of[hook] = new_last_hop
                hooks_of[new_last_hop].extend(hooks_of[old_last_hop])
                hooks_of[old_last_hop] = []

            # update the component's blocked set
            for r, subtree_blocked_ in zip(roots, subtree_blocked__):
                subtree_blocked_[sr_kept] |= (
                    subtree_blocked_[sr_dropped] | A[u][v]['blocked__'][r]
                )
                subtree_blocked_[sr_kept] &= ~subtree_[sr_kept]

        # add edge to effect union of subtree of u to subtree of v (via subroot of v)
        subtree = subtree_[sr_kept]
        subtree_dropped = subtree_[sr_dropped]
        subtree |= subtree_dropped
        capacity_left = capacity - subtree.count()

        sr_v_entry = pq.tags.get(sr_kept)
        if sr_v_entry is not None:
            _, _, _, (_, t) = sr_v_entry
            sr_kept_target = subroot_[t]
            if sr_kept_target != sr_dropped:
                who_targets_[sr_kept_target].remove(sr_kept)
        # TODO: think about why a discard was needed
        who_targets_[sr_kept].remove(sr_dropped)

        # assign root, subroot and subtree to the newly added nodes
        root_u = A.nodes[u]['root']
        if root_u != root:
            rootmask__[root_u] &= ~subtree_dropped
            rootmask__[root] |= subtree_dropped
        for t in subtree_dropped.search(_ONE):
            A.nodes[t]['root'] = root
            subroot_[t] = sr_kept
        A.graph['rootmask__'][root] |= subtree_dropped
        debug('<add edge> «%d~%d» subroot <%d>', u, v, sr_kept)
        if _lggr.isEnabledFor(logging.DEBUG) and pq:
            debug('heap top: <%d>, «%s» %.3f', pq[0][-2], pq[0][-1], pq[0][0])
        else:
            debug('heap EMPTY')
        S.add_edge(u, v)
        A.remove_edge(u, v)
        A.remove_edges_from(edge_conflicts(u, v, diagonals))

        # finished adding the edge, now check the consequences
        who_targets_[sr_dropped].discard(sr_kept)

        if method == 'rootlust' and ((u, v) if u < v else (v, u)) not in diagonals:
            # this fixes unions that result in 2 sides of a triangle being used but
            #   where the unused side is not the longest one (this fix make it so)
            for rot in ('cw', 'ccw'):
                s = P_A[v][u][rot]
                if (
                    subtree[s]
                    and s in S[v]
                    and P_A[s][v][rot] == u
                    and Aʹ[u][s]['length'] < Aʹ[v][s]['length']
                ):
                    S.remove_edge(v, s)
                    S.add_edge(u, s)
                    A.remove_edge(u, s)

        if capacity_left > 0:
            if method == 'rootlust':
                # rootlust needs a more aggressive retargetting
                is_stale_[list(who_targets_[sr_dropped] | who_targets_[sr_kept])] = True
                who_targets_[sr_kept].clear()
            else:
                for subroot in tuple(who_targets_[sr_kept]):
                    if subtree_[subroot].count() > capacity_left:
                        who_targets_[sr_kept].remove(subroot)
                        is_stale_[subroot] = True
                for subroot in who_targets_[sr_dropped]:
                    if subtree_[subroot].count() > capacity_left:
                        is_stale_[subroot] = True
                    else:
                        who_targets_[sr_kept].add(subroot)
            is_stale_[sr_kept] = True
        else:
            # max capacity reached: subtree full
            is_not_full_[sr_kept] = False
            if sr_kept in pq.tags:
                # this is required because of i=0 feeders
                pq.cancel(sr_kept)
            subtree_nodes = tuple(subtree.search(_ONE))
            # by removing nodes, all the useless edges are discarded
            A.remove_nodes_from(subtree_nodes)
            debug('subtree complete <%d>: %s', sr_kept, subtree_nodes)
            is_stale_[list(who_targets_[sr_dropped] | who_targets_[sr_kept])] = True
            who_targets_[sr_kept] = None
        is_not_full_[sr_dropped] = False
        subtree_[sr_dropped] = None
        who_targets_[sr_dropped] = None
    # END: main loop

    # add feeders
    is_subroot_ = bitarray(subtree is not None for subtree in subtree_)
    for r, rootmask_ in zip(roots, rootmask__):
        for sr in (rootmask_ & is_subroot_).search(_ONE):
            S.add_edge(r, sr)

    calcload(S)
    # algorithm finished, store some info in the graph object
    S.graph.update(
        runtime=time.perf_counter() - start_time,
        capacity=capacity,
        creator='EW_presolver',
        iterations=i,
        prevented_crossings=prevented_crossings,
        method_options=dict(
            fun_fingerprint=_constructor_fun_fingerprint,
        ),
    )
    #  if keep_log:
    #      S.graph['method_log'] = log
    return S


_constructor_fun_fingerprint = fun_fingerprint(constructor)
