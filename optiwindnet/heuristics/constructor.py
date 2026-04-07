# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import time
from itertools import chain, tee

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
_DEFAULT_BIAS_MARGIN = 0.02

# empirically obtained coefficients
_rootlust_coefs = (
    # rootlust_0 = [0] + [1]/capacity
    (0.08927350087510766, 0.3660630101515834),
    # rootlust_1 = [0] + [1]*rootlust_0
    (0.6105314984368293, -2.0350588552021245),
)


def constructor(
    Aʹ: nx.Graph,
    capacity: int,
    method: str = 'rootlust',
    *,
    rootlust_: tuple[float, float] = (),
    maxiter: int = 10000,
    bias_margin: float = _DEFAULT_BIAS_MARGIN,
    keep_log: bool = False,
    blockage_link_cos_lim: float = 0.85,  # 30°
    blockage_link_feeder_lim: float = 2.0,
    blockage_subtree_feeder_lim: float = 2.5,
) -> nx.Graph:
    """Create a network using a constructive greedy heuristic.

    The overall structure of the constructive algorithm is based on:

      Esau, L. R., and K. C. Williams. "On Teleprocessing System Design, Part II: A Method
        for Approximating the Optimal Network." IBM Systems Journal 5, no. 3 (1966):
        142–47. https://doi.org/10.1147/sj.53.0142.

    However, this implementation uses the extended Delaunay triangulation (given in A) as
    the base connectivity, and implements terminal-terminal crossing prevention. This
    means that even the method named 'esau_williams' does not match exactly the paper's
    description, but the similarities are still substantial.

    Note that constructor cannot be constrained in the number of feeders and that only
    method 'radial_EW' is constrained to producing radial topologies (i.e. subtrees are
    always simple paths) as opposed to the branched topologies produced by the others.

    Methods:
    - 'esau_williams': Esau-Williams C-MST heuristic modified to avoid crossings (EW).
    - 'biased_EW': EW with a bias towards moving radially (root-ward) on quasi-ties.
    - 'rootlust': EW with a tunable root-ward bias that increases as capacity decreases.
    - 'radial_EW': EW variant that produces radial subtrees (simple paths from root).

    Args:
      Aʹ: available links graph
      capacity: max number of terminals in a subtree
      method: choice of method (see Methods)
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
        # closed form approximations for providing the best rootlust for the capacity
        rootlust_0 = _rootlust_coefs[0][0] + _rootlust_coefs[0][1] / capacity
        rootlust_1 = _rootlust_coefs[1][0] + _rootlust_coefs[1][1] * rootlust_0
        # pre-scale rootlust_1 to avoid the division inside the loop
        rootlust_ = rootlust_0, rootlust_1 / (capacity - 1)
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
    subtree_: list[bitarray | None] = [zeros(T) for _ in _T]
    for t, subtree in zip(_T, subtree_):
        subtree[t] = True
    # <subroot_>: maps terminals to their subroots
    subroot_ = list(_T)
    last_hop_of: list[int | None] = [t for t in _T]
    hooks_of = [[t] for t in _T]
    # <is_stale_>: mask of stale subroots (in need of target refresh)
    is_stale_ = zeros(T)
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
    who_targets_: list[set[int] | None] = [set() for _ in _T]

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
    if method == 'radial_EW':
        # <tail_>: the endpoint of path subtrees (radial_EW)
        tail_ = [t for t in _T]
        num_insertions = 0
    # END: helper data structures

    # BEGIN: alternative methods of selecting the best edge to expand components
    def find_union_esau_williams_tradeoff(subroot):
        """Straightforward implementation of the Esau-Williams trade-off.

        Included for educational purposes, since both 'biased_EW' and 'rootlust'
        produce better topologies on average.
        """
        # Esau, L. R., and K. C. Williams.
        # "On Teleprocessing System Design, Part II: A Method for Approximating the Optimal Network."
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

    # relative limit to consider extents equivalent in 'biased_EW'
    extent_threshold = 1.0 + bias_margin

    def find_union_biased_EW_tradeoff(subroot):
        subtree = subtree_[subroot]
        capacity_left = capacity - subtree.count()
        d2root = d2roots[subroot, A.nodes[subroot]['root']]
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
                    if extent <= d2root:
                        # useful edges
                        # v's proximity to root is used as tie-breaker
                        choices.append(
                            (extent, d2rootsRank[v, A.nodes[v]['root']], u, v)
                        )
        if choices:
            choices.sort()
            best_extent, best_rank, *best_edge = choices[0]
            for extent, rank, *edge in choices[1:]:
                if extent > extent_threshold * best_extent:
                    # no more edges within margin
                    break
                if rank < best_rank:
                    best_extent, best_rank, best_edge = extent, rank, edge
            return (best_extent - d2root, best_rank, *best_edge), edges2discard
        else:
            return (), edges2discard

    def find_union_rootlust_tradeoff(subroot):
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

    def find_union_radial_EW_tradeoff(subroot):
        subtree = subtree_[subroot]
        subtree_count = subtree.count()
        capacity_left = capacity - subtree_count
        choices = []
        edges2discard = []
        d2root = d2roots[subroot, A.nodes[subroot]['root']]
        if subtree_count == 1:
            # insertion is only assessed when subtree has a single node
            if i != 0:
                # search for insertions only after the initial queue-filling run
                candidates = []
                source, target = tee(P_A.neighbors_cw_order(subroot))
                for u, v in zip(source, chain(target, (next(target),))):
                    # assess path insertion options
                    nb_ = A[subroot]
                    if u in nb_ and v in nb_ and (u, v) in S.edges:
                        candidates.append((u, v))
                    elif diag := diagonals.inv.get((u, v) if u < v else (v, u)):
                        n = diag[0] if diag[1] == subroot else diag[1]
                        # check the triangles (u, subroot, n) and (n, subroot, v)
                        if u in nb_ and n in nb_ and (u, v) in S.edges:
                            candidates.append((u, n))
                        if n in nb_ and v in nb_ and (n, v) in S.edges:
                            candidates.append((n, v))
                for u, v in candidates:
                    extent = (
                        A[subroot][u]['length']
                        + A[subroot][v]['length']
                        - Aʹ[u][v]['length']
                    )
                    if extent <= d2root:
                        tiebreaker = d2roots[subroot_[u], A.nodes[u]['root']]
                        choices.append((extent, tiebreaker, u, v))
            # this is for finding extension options
            endpoints = ((subroot, tail_[subroot]),)
        else:
            endpoints = ((subroot, tail_[subroot]), (tail_[subroot], subroot))

        for u, tail_u in endpoints:
            for v in A[u]:
                sr_v = subroot_[v]
                subtree_v = subtree_[sr_v]
                tail_v = tail_[sr_v]
                if sr_v == subroot or subtree_v.count() > capacity_left:
                    edges2discard.append((u, v))
                    continue
                if v != sr_v and v != tail_v:
                    continue
                extent = A[u][v]['length']
                if extent <= d2root:
                    d2root_sr_v = d2roots[sr_v, A.nodes[sr_v]['root']]
                    if v == sr_v and v != tail_v:
                        # subroot must change to the tail of subroot or of sr_v
                        union_feeder = min(
                            d2roots[tail_u].min(), d2roots[tail_[sr_v]].min()
                        )
                        extent += union_feeder - d2root_sr_v
                        if extent > d2root:
                            continue
                    choices.append((extent, d2root_sr_v, u, v))
        if choices:
            choices.sort()
            best_extent, best_feeder_length, *best_edge = choices[0]
            for extent, feeder_length, *edge in choices[1:]:
                if extent > extent_threshold * best_extent:
                    # no more edges within margin
                    break
                if feeder_length < best_feeder_length:
                    best_extent, best_feeder_length, best_edge = extent, feeder_length, edge
            return (best_extent - d2root, best_feeder_length, *best_edge), edges2discard
        else:
            return (), edges2discard

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
    elif method == 'biased_EW':
        find_union = find_union_biased_EW_tradeoff
    elif method == 'rootlust':
        add_link_blockmap(A)
        find_union = find_union_rootlust_tradeoff
        angle__, angle_rank__ = A.graph['angle__'], A.graph['angle_rank__']
        union_limits, angle_ccw = angle_oracles_factory(angle__, angle_rank__)
    elif method == 'radial_EW':
        find_union = find_union_radial_EW_tradeoff
    else:
        raise ValueError(f'Unsupported constructor method: {method!r}')

    def cancel_if_queued(subroot):
        if subroot in pq.tags:
            pq.cancel(subroot)

    def enqueue_best_union(subroot):
        debug('<enqueue_best_union> starting... subroot = <%d>', subroot)
        best_choice, edges2discard = find_union(subroot)
        A.remove_edges_from(edges2discard)
        if best_choice:
            priority, _, u, v = best_choice
            pq.add(priority, subroot, (u, v))
            if who_targets_ is not None:
                who_targets_[subroot_[v]].add(subroot)
            debug(
                '<pushed> sr_u <%d>, «%d~%d», priority = %.3f', subroot, u, v, priority
            )
        else:
            debug('<cancelling> %d', subroot)
            cancel_if_queued(subroot)

    def reassign_subroot(subroot_from, subroot_to, root_to):
        """Change the subroot of a subtree to another node of that subtree.

        This is only relevant to the 'radial_EW' method. Any unions that need a
        subroot that is different from the sr_kept one may need this reassignment.

        Subroots are used in multiple data structures, a call to this function must
        effect the change across all of them. One additional change is the possible
        root reassignment if `subroot_to` is closer to a different root than that of
        `subroot_from`.
        """
        debug(
            'reassigning subroot %d to %d via root %d',
            subroot_from,
            subroot_to,
            root_to,
        )
        for n in subtree_[subroot_from].search(_ONE):
            subroot_[n] = subroot_to
            A.nodes[n]['root'] = root_to
        subtree_[subroot_to], subtree_[subroot_from] = (
            subtree_[subroot_from],
            subtree_[subroot_to],
        )
        tail_[subroot_to] = subroot_from
        who_targets_[subroot_to] = who_targets_[subroot_from]
        who_targets_[subroot_from] = None
        for who in who_targets_:
            if who is not None and subroot_from in who:
                who.remove(subroot_from)
                who.add(subroot_to)

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

        # REFRESH entries of stale subtrees

        to_retarget_[:] = is_stale_ & is_not_full_
        if to_retarget_.any():
            stale_subtrees = tuple(to_retarget_.search(_ONE))
            debug('stale_subtrees: %s', stale_subtrees)
            for subroot in stale_subtrees:
                enqueue_best_union(subroot)
            is_stale_.setall(0)
        if not pq:
            break

        # GET prioritary union (changes only the queue)

        sr_u, (u, v) = pq.top()

        # ASSESS prioritary union (no change in state)

        sr_kept = subroot_[v]
        sr_dropped = sr_u
        debug('<popped> «%d~%d», sr_dropped: <%d>', u, v, sr_dropped)
        if (u, v) not in A.edges and subroot_[u] == sr_u:
            debug('<discard> «%d~%d» not in A anymore', u, v)
            continue

        if method == 'radial_EW':
            #  if subroot_[u] == subroot_[v]:
            if subroot_[u] != sr_u:
                # this is an insertion
                if (u, sr_u) not in A.edges or (v, sr_u) not in A.edges:
                    debug('<discard> «%d~%d~%d» not in A anymore', u, sr_u, v)
                    continue
                debug('INSERTION of %d between %d and %d', sr_u, u, v)
                num_insertions += 1
                # start by opening the receiver path
                S.remove_edge(u, v)
                # add only one of the edges of the insertion here
                S.add_edge(u, sr_u)
                A.remove_edge(u, sr_u)
                A.remove_edges_from(edge_conflicts(u, sr_u, diagonals))
                # set it up so that the common machinery adds the other edge
                u = sr_u
            elif v == sr_kept and subtree_[v].count() > 1:
                # this is an extension and the union feeder must be other than sr_kept
                debug('EXTENSION with sr_kept (%d) change', sr_kept)
                # find the free endpoint of the dropped subroot
                if u == sr_u:
                    alt_sr_u = tail_[sr_u]
                    root_u = d2roots[alt_sr_u].argmin() - R
                else:
                    alt_sr_u = sr_u
                    root_u = A.nodes[sr_u]['root']
                tail_v = tail_[sr_kept]
                alt_root_v = d2roots[tail_v].argmin() - R
                if d2roots[tail_v, alt_root_v] <= d2roots[alt_sr_u, root_u]:
                    # union subroot is the tail of kept subroot
                    reassign_subroot(sr_kept, tail_v, alt_root_v)
                    debug('SUBROOT %d -> %d', sr_kept, tail_v)
                    sr_kept = tail_v
                else:
                    # union subroot is in the dropped subroot: reverse direction
                    if alt_sr_u != sr_u:
                        # subroot_[u] must change
                        reassign_subroot(sr_u, alt_sr_u, root_u)
                        debug('SUBROOT %d -> %d', sr_u, alt_sr_u)
                    u, v, sr_dropped, sr_kept = v, u, sr_kept, alt_sr_u
                    debug('DIRECTION (%d, %d) -> (%d, %d)', v, u, u, v)
                # set the tail of the union outcome
                tail_[sr_kept] = tail_[sr_dropped]
            elif u == tail_[sr_dropped]:
                # set the tail of the union outcome
                tail_[sr_kept] = sr_u
            else:
                # set the tail of the union outcome
                tail_[sr_kept] = tail_[sr_dropped]

        root = A.nodes[sr_kept]['root']

        if method == 'rootlust':
            # assess the union's angle span
            union_span_ = [
                union_limits(
                    r,
                    u,
                    *subtree_span__[sr_dropped][r],
                    v,
                    *subtree_span__[sr_kept][r],
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

            # EFFECT prioritary union

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
        who_targets_[sr_kept].discard(sr_dropped)
        who_targets_[sr_dropped].discard(sr_kept)

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
        debug('TAIL of %d: %d', sr_kept, tail_[sr_kept])
        if _lggr.isEnabledFor(logging.DEBUG) and pq:
            debug('heap top: <%d>, «%s» %.3f', pq[0][-2], pq[0][-1], pq[0][0])
        else:
            debug('heap EMPTY')
        S.add_edge(u, v)
        A.remove_edge(u, v)
        A.remove_edges_from(edge_conflicts(u, v, diagonals))

        # finished adding the edge, now check the consequences

        if method == 'rootlust' and ((u, v) if u < v else (v, u)) not in diagonals:
            # this fixes unions that result in 2 sides of a triangle being used but
            #   where the unused side is not the longest one (this fix make it so)
            for rot in ('cw', 'ccw'):
                s = P_A[v][u][rot]
                if P_A[s][v][rot] != u:
                    # uvs is not a triangle
                    continue
                # TODO: redundant `and`: is `subtree[s]` way faster than `s in S[v]`?
                if subtree[s] and s in S[v]:
                    Aʹs = Aʹ[s]
                    if u in Aʹs and Aʹs[u]['length'] < Aʹs[v]['length']:
                        S.remove_edge(v, s)
                        S.add_edge(u, s)
                        A.remove_edge(u, s)
                        continue
                diagonal = diagonals.inv.get((s, v) if s < v else (v, s))
                if diagonal is not None:
                    w, x = diagonal
                    t = w if x == v else x
                    if subtree[t] and t in S[v]:
                        Aʹt = Aʹ[t]
                        if u in Aʹt and Aʹt[u]['length'] < Aʹt[v]['length']:
                            S.remove_edge(v, t)
                            S.add_edge(u, t)
                            A.remove_edge(u, t)

        if capacity_left > 0:
            if method in ('rootlust', 'radial_EW'):
                # some methods need aggressive retargetting
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
            method=method,
            fun_fingerprint=_constructor_fun_fingerprint,
        ),
    )
    if method == 'radial_EW':
        S.graph['num_insertions'] = num_insertions
    #  if keep_log:
    #      S.graph['method_log'] = log
    return S


_constructor_fun_fingerprint = fun_fingerprint(constructor)
