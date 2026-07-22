# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import time
from collections import defaultdict
from itertools import chain, tee

import networkx as nx
import numpy as np
from bitarray import bitarray
from bitarray.util import ones, zeros
from scipy.stats import rankdata

from ..crossings import edge_conflicts
from ..fingerprint import fingerprint_function
from ..geometric import (
    angle_oracles_factory,
)
from ..interarraylib import (
    add_link_blockmap,
    add_terminal_closest_root,
    calcload,
    split_rings_and_calc_loads,
)
from ..types import Topology
from .priorityqueue import PriorityQueue

__all__ = ()

_lggr = logging.getLogger(__name__)
_debug, _info, _warn, _error = _lggr.debug, _lggr.info, _lggr.warning, _lggr.error

_ONE = bitarray('1')
_DEFAULT_BIAS_MARGIN = 0.02

# ringed root-ward bias (r0, r1), tuned on the R>1 repository locations; nearly
# constant in capacity, so unlike _rootlust_coefs it is not a closed form
_RINGED_ROOTLUST = (0.7, 0.3)

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
    rootlust_: tuple[float, float] | None = None,
    maxiter: int = 10000,
    bias_margin: float | None = None,
    weigh_detours: bool = True,
    straight_feeder_route: bool = False,
    keep_log: bool = False,
    blockage_link_cos_lim: float = 0.85,  # 30°
    blockage_link_feeder_lim: float = 2.0,
    blockage_subtree_feeder_lim: float = 2.5,
) -> nx.Graph:
    """Create a network using a constructive greedy heuristic.

    The overall structure of the constructive algorithm is based on:

      Esau, L. R., and K. C. Williams. "On Teleprocessing System Design,
        Part II: A Method for Approximating the Optimal Network."
        IBM Systems Journal 5, no. 3 (1966):
        142–47. https://doi.org/10.1147/sj.53.0142.

    However, this implementation uses the extended Delaunay triangulation (given in A)
    as the base connectivity, and implements terminal-terminal crossing prevention.
    This means that even the method named ``'esau_williams'`` does not match exactly
    the paper's description, but the similarities are still substantial.

    Note that constructor cannot be constrained in the number of feeders and that only
    method ``'radial_EW'`` is constrained to producing radial topologies (i.e. subtrees
    are always simple paths) as opposed to the branched topologies produced by the
    others.

    Available Methods:
      ``'esau_williams'``
        Esau-Williams C-MST heuristic modified to avoid crossings (EW).
      ``'biased_EW'``
        EW with a bias towards moving radially (root-ward) on quasi-ties.
      ``'rootlust'``
        EW with a tunable root-ward bias that increases as capacity decreases.
      ``'radial_EW'``
        EW variant that produces radial subtrees (simple paths from root).
      ``'ringed'``
        Grows simple-path subtrees that are closed into rings at finalization:
        each endpoint connects to its nearest root (two feeders, which may bridge
        two roots), joined at an open point (``load=0``, no current).
        ``capacity`` is the per-arm limit, so a ring holds up to ``2 * capacity``
        terminals. Unions are ranked by their total saving — the feeders shed at
        the two joined endpoints minus the connecting edge's length (Clarke-Wright
        style) — with a ``bias_margin`` window favoring the more root-ward union
        on quasi-ties.

    Args:
      Aʹ: available links graph
      capacity: max number of terminals in a subtree
      method: choice of method (see Available Methods)
      bias_margin: (biased_EW | radial_EW | ringed) fractional margin within
        which candidates are equivalent, resolving the quasi-tie root-ward. For
        ``'ringed'`` the margin is a fraction of the best union ``saving`` (not
        the edge ``extent``, since the ringed saving also depends on the peer
        feeders). Defaults to 0.02.
      weigh_detours: (!= esau_williams) only add edges whose tradeoff is not
        outweighted by detours
      straight_feeder_route: prevent crossings of feeders
        (incompatible with ``weigh_detours=True``)
      maxiter: fail-safe to avoid locking in an infinite loop

    Returns:
      Solution topology S.
    """

    start_time = time.perf_counter()
    if straight_feeder_route and weigh_detours:
        _warn(
            'Setting `weigh_detours` to False because `straight_feeder_route=True` '
            'was requested. Set `weigh_detours=False` to suppress this message.'
        )
        weigh_detours = False
    R, T = (Aʹ.graph[k] for k in 'RT')
    _T = range(T)
    ringed = method == 'ringed'
    # methods that grow subtrees as simple paths (endpoints tracked by <tail_>)
    radial_like = method in ('radial_EW', 'ringed')
    if bias_margin is None:
        bias_margin = _DEFAULT_BIAS_MARGIN
    capacity_report = capacity
    if ringed:
        # A ring is two radial arms sharing one root, joined at their tails. Build
        # a simple path of up to 2*capacity terminals (each arm holds at most
        # `capacity`); the path is closed into a ring at finalization, with the
        # open point placed at the load midpoint so neither arm exceeds capacity.
        capacity *= 2
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
    if rootlust_:
        r0, r1 = rootlust_
    elif ringed:
        # ringed's root-ward bias is near-constant in capacity (tuned on the
        # R>1 repository locations); the branched closed form does not apply
        r0, r1 = _RINGED_ROOTLUST
    else:
        # closed form approximations for the best rootlust for the capacity
        r0 = _rootlust_coefs[0][0] + _rootlust_coefs[0][1] / capacity
        r1 = _rootlust_coefs[1][0] + _rootlust_coefs[1][1] * r0
    # pre-scale the slope to avoid the division inside the loop
    rootlust_ = r0, r1 / max(capacity - 1, 1)

    # removing root nodes from A to speedup enqueue_best_union
    # this may be done because G already starts with feeders
    A.remove_nodes_from(roots)
    # END: prepare auxiliary graph with all allowed edges and metrics

    # ensure roots are added, even if the star graph uses a subset of them
    S.add_nodes_from(roots)

    # BEGIN: helper data structures

    # mappings from nodes
    # <subtree_>: maps nodes to the list of nodes in their subtree
    subtree_: list[bitarray | None] = []
    for t in _T:
        subtree = zeros(T)
        subtree[t] = True
        subtree_.append(subtree)
    # <subroot_>: maps terminals to their subroots
    subroot_ = list(_T)
    #  last_hop_of: list[int | None] = [t for t in _T]
    #  hooks_of = [[t] for t in _T]
    # <is_stale_>: mask of stale subroots (in need of target refresh)
    is_stale_ = zeros(T)
    # <is_extendable_>: mask of subroots with spare capacity
    is_extendable_ = ones(T)
    # <is_root_nb__>: mask of node coordinates that are the last hop of a full
    #   feeder route (weigh_detours)
    is_root_nb__ = tuple(zeros(T) for _ in roots)
    # <is_corner_>: mask of node coordinates that are detour corners (weigh_detours)
    is_corner_ = zeros(T)
    # memory allocation for temporary constructs
    # <to_retarget_>: mask of subroots that are stale and not full
    to_retarget_ = bitarray(T)

    # mappings from components (indexed by their subroots)
    # <subtree_span__>: pairs (most_CW, most_CCW) of extreme nodes of each subtree
    subtree_span__ = [[(t, t) for _ in roots] for t in _T]
    # <subtree_blocked__>: sets of blocked terminals from other components wrt each root
    #  subtree_blocked__ = [[bitarray(T) for _ in _T] for _ in roots]
    # <detours_via_prime_>: holds the detour segment upstream from corner
    #   (weigh_detours)
    detours_via_prime_ = defaultdict(list)

    # mappings from components (identified by their subroots)
    # <who_targets_>: maps component to set of components queued to merge in
    who_targets_: list[set[int] | None] = [set() for _ in _T]

    # other structures
    # <pq>: the smaller the entry, the higher the priority
    pq = PriorityQueue()
    # <i>: iteration counter
    i = 0
    # <tail_>: the endpoint of path subtrees (only read if radial_like)
    tail_ = list(_T)
    num_insertions = 0
    # END: helper data structures

    # relative limit to consider two extents equivalent
    extent_threshold = 1.0 + bias_margin

    def biased_chooser(choices, d2root):
        # Choose the union candidate with a bias towards extending root-ward
        if choices:
            choices.sort()
            best_extent, best_feeder_length, *best_edge = choices[0]
            extent_lim = best_extent * extent_threshold
            for extent, feeder_length, *edge in choices[1:]:
                if extent > extent_lim:
                    # no more edges within bias_margin
                    break
                # the runner-up edge is "equivalent" to the best one, compare feeders
                if feeder_length < best_feeder_length:
                    best_extent, best_feeder_length = extent, feeder_length
                    best_edge = edge
            return (best_extent - d2root, best_feeder_length, *best_edge)
        else:
            return ()

    def ringed_chooser(choices):
        # Among candidates `(-saving, feeder_pair, u, v)` within `bias_margin` of
        # the best `saving`, pick the smallest resulting feeder pair (root-ward).
        # The window acts on `saving` (not `extent` as in biased_chooser) since
        # the ringed saving also depends on the peer feeders; bias_margin == 0
        # reduces to exact-saving ties broken by feeder_pair.
        if not choices:
            return ()
        choices.sort()
        best_neg_saving, best_feeders, *best_edge = choices[0]
        # window lower bound on saving; with bias_margin == 0 this is exactly
        # `best_saving`, so only exact-saving ties remain (equivalent to min())
        saving_lim = -best_neg_saving * (1.0 - bias_margin)
        for neg_saving, feeders, *edge in choices[1:]:
            if -neg_saving < saving_lim:
                # no more candidates within bias_margin of the best saving
                break
            # the candidate is "equivalent" to the best one; prefer root-ward
            if feeders < best_feeders:
                best_neg_saving, best_feeders = neg_saving, feeders
                best_edge = edge
        return (best_neg_saving, best_feeders, *best_edge)

    # BEGIN: alternative methods of selecting the best edge to expand components
    def find_union_esau_williams_tradeoff(subroot):
        """Straightforward implementation of the Esau-Williams trade-off.

        Included for educational purposes, since both ``'biased_EW'`` and ``'rootlust'``
        produce better topologies on average.
        """
        # Esau, L. R., and K. C. Williams.
        # "On Teleprocessing System Design, Part II: A Method for Approximating
        # the Optimal Network."
        # IBM Systems Journal 5, no. 3 (1966): 142–47.
        # https://doi.org/10.1147/sj.53.0142.
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
        return biased_chooser(choices, d2root), edges2discard

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
            # insertion is only considered when subtree has a single node
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

        for u_head, u_tail in endpoints:
            for v in A[u_head]:
                sr_v = subroot_[v]
                subtree_v = subtree_[sr_v]
                tail_v = tail_[sr_v]
                if sr_v == subroot or subtree_v.count() > capacity_left:
                    edges2discard.append((u_head, v))
                    continue
                if v != sr_v and v != tail_v:
                    continue
                extent = A[u_head][v]['length']
                if extent <= d2root:
                    d2root_sr_v = d2roots[sr_v, A.nodes[sr_v]['root']]
                    if v == sr_v and v != tail_v:
                        # subroot must change either to u_tail or to the tail of sr_v
                        union_feeder = min(
                            d2roots[u_tail].min(), d2roots[tail_[sr_v]].min()
                        )
                        # add the increase in feeder length to the edge extent
                        extent += union_feeder - d2root_sr_v
                        if extent > d2root:
                            continue
                        tiebreaker = union_feeder
                    else:
                        tiebreaker = d2root_sr_v
                    choices.append((extent, tiebreaker, u_head, v))
        return biased_chooser(choices, d2root), edges2discard

    def find_union_ringed_tradeoff(subroot):
        """Rank unions that grow simple paths (closed into rings at finalization).

        A ring costs its path edges plus two feeders, one at each endpoint; each
        subroot attaches to its own nearest root, so the feeder pair is
        ``d2roots[subroot].min() + d2roots[tail].min()`` (subroots nearest
        different roots make the ring bridge them). Joining this component's endpoint
        ``u_head`` to a peer's endpoint ``v`` sheds the two joined feeders and
        keeps the two far ones::

            saving = feeder_pair + feeder_pair_sr_v - extent - union_feeders

        This is the exact change in total cost (the Clarke-Wright / Esau-Williams
        tradeoff; https://doi.org/10.1287/opre.12.4.568). Only ``saving >= 0``
        unions qualify, priority ``-saving``; ties within ``bias_margin`` of the
        best saving are broken root-ward (see :func:`ringed_chooser`). Inserting
        a lone node between two adjacent path nodes instead sheds its feeder pair
        and leaves the receiver unchanged: ``tradeoff = extent - feeder_pair``.
        """
        subtree = subtree_[subroot]
        subtree_count = subtree.count()
        capacity_left = capacity - subtree_count
        tail = tail_[subroot]
        # length of this component's two feeders; each end attaches to its own
        # nearest root, so a ring may bridge two roots (r1, r2)
        feeder_pair = d2roots[subroot].min() + d2roots[tail].min()
        # rootlust-like root-ward pull, growing as the ring fills (see
        # find_union_rootlust_tradeoff); for an extension at u_head,
        # feeder_pair - union_feeders is the shed feeder minus the adopted far
        # feeder, the ringed analog of rootlust's d2rGain
        rootlust = rootlust_[0] + rootlust_[1] * subtree_count
        choices = []
        edges2discard = []
        if subtree_count == 1:
            # insertion is only considered when subtree has a single node
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
                    # inserting the lone node sheds its feeder pair (here
                    # feeder_pair == 2*d2root); the receiver's feeders are
                    # unchanged
                    tradeoff = extent - feeder_pair
                    if tradeoff <= 0:
                        sr_u = subroot_[u]
                        if subtree_[sr_u].count() + 1 > capacity:
                            continue
                        # the receiving component's feeder pair as tie-breaker
                        tiebreaker = (d2roots[sr_u] + d2roots[tail_[sr_u]]).min()
                        choices.append((tradeoff, tiebreaker, u, v))
            # this is for finding extension options
            endpoints = ((subroot, tail),)
        else:
            endpoints = ((subroot, tail), (tail, subroot))

        for u_head, u_tail in endpoints:
            # a union at u_head sheds u_head's feeder; u_tail keeps its feeder
            for v in A[u_head]:
                sr_v = subroot_[v]
                subtree_v = subtree_[sr_v]
                tail_v = tail_[sr_v]
                if sr_v == subroot or subtree_v.count() > capacity_left:
                    edges2discard.append((u_head, v))
                    continue
                if v != sr_v and v != tail_v:
                    continue
                extent = A[u_head][v]['length']
                # far_v: the peer's far endpoint, which keeps its feeder
                far_v = tail_v if v == sr_v else sr_v
                feeder_pair_sr_v = d2roots[sr_v].min() + d2roots[tail_v].min()
                union_feeders = d2roots[u_tail].min() + d2roots[far_v].min()
                saving = feeder_pair + feeder_pair_sr_v - extent - union_feeders
                if saving >= 0:
                    biased = saving + rootlust * (feeder_pair - union_feeders)
                    choices.append((-biased, union_feeders, u_head, v))
        return ringed_chooser(choices), edges2discard

    # END: alternative methods of selecting the best edge to expand components

    def blocked_feeders(u, v, sr_dropped, sr_kept):
        blocked__ = A[u][v]['blocked__']
        union = subtree_[sr_dropped] | subtree_[sr_kept]
        feeders = []
        for r, blocked_, is_root_nb_ in zip(roots, blocked__, is_root_nb__):
            feeders.extend(
                (r, n) for n in (is_root_nb_ & blocked_ & ~union).search(_ONE)
            )
        return feeders

    def estimate_detours(u, v, sr_dropped, sr_kept):
        """Note: the ``detour_increase`` calculated here is an estimate."""
        # assess the union's angle span
        span_dropped_ = subtree_span__[sr_dropped]
        span_kept_ = subtree_span__[sr_kept]
        union_span_ = []
        for r in roots:
            LO, HI = span_dropped_[r]
            lo, hi = span_kept_[r]
            union_span_.append(union_limits(r, u, LO, HI, v, lo, hi))
        blocked__ = A[u][v]['blocked__']
        detour_increase = 0.0
        changes = []
        #  is_last_hop_ = bitarray(count > 0 for count in last_hops_count_)
        union = subtree_[sr_dropped] | subtree_[sr_kept]
        for r, blocked_, is_root_nb_ in zip(roots, blocked__, is_root_nb__):
            lo, hi = union_span_[r]
            hops = []
            for prime in (is_root_nb_ & blocked_ & ~union).search(_ONE):
                # feeder blocked by (u, v) was not previously detoured by union
                former_extent = d2roots[prime, r]
                if prime in detours_via_prime_:
                    hops.extend(
                        (prime, former_extent, None) for _ in detours_via_prime_[prime]
                    )
                if subtree_[prime] is not None:
                    hops.append((prime, former_extent, None))
            moved_by_uv_ = is_root_nb_ & is_corner_ & union
            # the extremes (lo and hi) of union are not affected by (u, v)
            moved_by_uv_[lo] = moved_by_uv_[hi] = False
            for prime in moved_by_uv_.search(_ONE):
                # edge (u, v) changes an existing feeder detour
                # move to the previous coordinate in the detour
                for hop in detours_via_prime_[prime]:
                    #  former_extent = d2roots[prime, r] + extent
                    former_extent = d2roots[prime, r] + np.hypot(
                        *(VertexC[hop] - VertexC[prime])
                    )
                    hops.append((hop, former_extent, prime))
            for hop, former_extent, dropped in hops:
                extent_lo = d2roots[lo, r] + np.hypot(*(VertexC[lo] - VertexC[hop]))
                extent_hi = d2roots[hi, r] + np.hypot(*(VertexC[hi] - VertexC[hop]))
                extent, corner = (
                    (extent_lo, lo) if extent_lo <= extent_hi else (extent_hi, hi)
                )
                detour_increase += (extent - former_extent).item()
                changes.append((hop, corner, r, dropped))
        if changes:
            _debug('detour increase of %.3f for rerouting %s', detour_increase, changes)
        return detour_increase, union_span_, changes

    try:
        find_union = dict(
            esau_williams=find_union_esau_williams_tradeoff,
            biased_EW=find_union_biased_EW_tradeoff,
            rootlust=find_union_rootlust_tradeoff,
            radial_EW=find_union_radial_EW_tradeoff,
            ringed=find_union_ringed_tradeoff,
        )[method]
    except KeyError:
        raise ValueError(f'Unsupported constructor method: {method!r}')
    #  use_blockage = weigh_detours and method in ('rootlust', 'radial_EW')
    use_blockage = weigh_detours and method != 'esau_williams'

    if use_blockage or straight_feeder_route:
        add_link_blockmap(A)
        angle__, angle_rank__ = A.graph['angle__'], A.graph['angle_rank__']
        union_limits, angle_ccw = angle_oracles_factory(angle__, angle_rank__)

    def drop_target(subroot, payload):
        """Drop ``subroot`` from the ``who_targets_`` set of the peer it targets in
        ``payload``.

        ``payload`` is the queue entry's ``(u, v)``; the targeted component is the one
        holding ``v``. Keeps ``who_targets_`` consistent with the queue, so it never
        retains a subroot whose subtree has already been consumed (set to ``None``).
        """
        targeted = who_targets_[subroot_[payload[1]]]
        if targeted is not None:
            targeted.discard(subroot)

    def enqueue_best_union(subroot):
        _debug('<enqueue_best_union> starting... subroot = <%d>', subroot)
        # invariant upkeep: clear the previous-target membership before retargeting
        prev_entry = pq.tags.get(subroot)
        if prev_entry is not None:
            drop_target(subroot, prev_entry[-1])
        best_choice, edges2discard = find_union(subroot)
        A.remove_edges_from(edges2discard)
        if best_choice:
            priority, _, u, v = best_choice
            pq.add(priority, subroot, (u, v))
            who_targets_[subroot_[v]].add(subroot)
            _debug(
                '<pushed> sr_u <%d>, «%d~%d», priority = %.3f', subroot, u, v, priority
            )
        else:
            is_root_nb__[A.nodes[subroot]['root']][subroot] = True
            _debug('<cancelling> %d', subroot)
            pq.cancel(subroot)

    def reassign_subroot(subroot_from, subroot_to, root_to):
        """Change the subroot of a subtree to another node of that subtree.

        This is only relevant to the ``'radial_EW'`` method. Any unions that need a
        subroot that is different from the ``sr_kept`` one may need this reassignment.

        Subroots are used in multiple data structures, a call to this function must
        effect the change across all of them. One additional change is the possible
        root reassignment if ``subroot_to`` is closer to a different root than that of
        ``subroot_from``.
        """
        _debug(
            'reassigning subroot %d to %d via root %d',
            subroot_from,
            subroot_to,
            root_to,
        )
        pq.cancel(subroot_from)
        is_extendable_[subroot_from] = False
        is_extendable_[subroot_to] = True
        # not necessary to reassign: is_stale_
        for n in subtree_[subroot_from].search(_ONE):
            subroot_[n] = subroot_to
            A.nodes[n]['root'] = root_to
        subtree_[subroot_to] = subtree_[subroot_from]
        subtree_[subroot_from] = None
        tail_[subroot_to] = subroot_from
        who_targets_[subroot_to] = who_targets_[subroot_from]
        who_targets_[subroot_from] = None
        for who in who_targets_:
            if who is not None and subroot_from in who:
                who.remove(subroot_from)
                who.add(subroot_to)
        if use_blockage:
            subtree_span__[subroot_to] = subtree_span__[subroot_from]
            # update the component's blocked set
            #  for subtree_blocked_ in subtree_blocked__:
            #      subtree_blocked_[subroot_to] = subtree_blocked_[subroot_from]

    # initialize pq
    for n in _T:
        enqueue_best_union(n)

    # only read if use_blockage, in which case estimate_detours() assigns them
    union_span_: list[tuple[int, int]] = []
    changes: list[tuple[int, int, int, int | None]] = []

    # BEGIN: main loop
    while True:
        i += 1
        if i > maxiter:
            _error('maxiter reached (%d)', i)
            break
        _debug('[%d]', i)

        # REFRESH entries of stale subtrees

        to_retarget_[:] = is_stale_ & is_extendable_
        if to_retarget_.any():
            stale_subtrees = tuple(to_retarget_.search(_ONE))
            _debug('stale_subtrees: %s', stale_subtrees)
            for subroot in stale_subtrees:
                enqueue_best_union(subroot)
            is_stale_.setall(0)
        if not pq:
            break

        # GET union candidate (changes only the queue)

        prio, sr_u, (u, v) = pq.top()
        # sr_u left the queue: keep who_targets_ consistent (covers both the
        # union-effected path and the discard-after-pop paths below)
        drop_target(sr_u, (u, v))
        tradeoff = -prio

        # ASSESS union (no change in state)

        sr_kept = subroot_[v]
        # HACK: queue entries encode an insertion if sr_kept has edge (u, v) in S
        is_insertion = subroot_[u] != sr_u and (u, v) in S.edges
        _debug('<pop> «%d~%d», sr_dropped: <%d>, ins: %s', u, v, sr_u, is_insertion)
        # re-validate the popped entry: state may have changed since it was
        # queued. A failed check refreshes the subtree (is_stale_) and retries.
        if not is_insertion:
            if subroot_[u] == subroot_[v]:
                _debug('<discard> «%d~%d» same subroot', u, v)
                is_stale_[sr_u] = True
                continue
            if radial_like and (
                (u != sr_u and u != tail_[sr_u])
                or (v != sr_kept and v != tail_[sr_kept])
            ):
                _debug('<discard> «%d~%d» nodes are no longer endpoints', u, v)
                is_stale_[sr_u] = True
                continue
            if subtree_[subroot_[u]].count() + subtree_[subroot_[v]].count() > capacity:
                _debug('<discard> «%d~%d» combined capacity exceeded', u, v)
                is_stale_[sr_u] = True
                continue
        elif subtree_[subroot_[u]].count() + 1 > capacity:
            _debug('<discard> «%d~%d» insertion capacity exceeded', u, v)
            is_stale_[sr_u] = True
            continue
        if (u, v) not in A.edges:
            if not is_insertion:
                _debug('<discard> «%d~%d» not in A anymore', u, v)
                is_stale_[sr_u] = True
                continue
            elif (u, sr_u) not in A.edges or (v, sr_u) not in A.edges:
                _debug('<discard> «%d~%d~%d» not in A anymore', u, sr_u, v)
                is_stale_[sr_u] = True
                continue

        if use_blockage:
            # only proceed if tradeoff is greater of equal to the growth in detours
            detours_growth, union_span_, changes = estimate_detours(
                *((sr_u, v, sr_u, sr_kept) if is_insertion else (u, v, sr_u, sr_kept))
            )

            if tradeoff < detours_growth:
                A.remove_edge((sr_u if is_insertion else u), v)
                is_stale_[sr_u] = True
                _debug(
                    '<discard> «%d~%d»: tradeoff (%.3f) smaller than'
                    ' growth in detours (%.3f)',
                    u,
                    v,
                    tradeoff,
                    detours_growth,
                )
                continue

        if straight_feeder_route:
            blocked = blocked_feeders(
                *((sr_u, v, sr_u, sr_kept) if is_insertion else (u, v, sr_u, sr_kept))
            )
            if blocked:
                A.remove_edge((sr_u if is_insertion else u), v)
                is_stale_[sr_u] = True
                _debug('<discard> «%d~%d»: would cross feeders %s', u, v, blocked)
                continue

        # EFFECT union

        if radial_like:
            if is_insertion:
                # this is an insertion
                _debug('INSERTION of %d between %d and %d', sr_u, u, v)
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
                _debug('EXTENSION with sr_kept (%d) change', sr_kept)
                if ringed:
                    # a ring has feeders at both endpoints, so which endpoint
                    # carries the subroot label is cost-neutral: move it to the
                    # kept tail
                    union_tail = tail_[sr_u] if u == sr_u else sr_u
                    tail_v = tail_[sr_kept]
                    reassign_subroot(sr_kept, tail_v, A.nodes[sr_kept]['root'])
                    _debug('SUBROOT %d -> %d', sr_kept, tail_v)
                    sr_kept = tail_v
                    # set the tail of the union outcome
                    tail_[sr_kept] = union_tail
                else:
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
                        _debug('SUBROOT %d -> %d', sr_kept, tail_v)
                        sr_kept = tail_v
                    else:
                        # union subroot is in the dropped subtree: reverse union
                        # direction
                        if alt_sr_u != sr_u:
                            # subroot_[u] must change
                            reassign_subroot(sr_u, alt_sr_u, root_u)
                            _debug('SUBROOT %d -> %d', sr_u, alt_sr_u)
                        u, v, sr_u, sr_kept = v, u, sr_kept, alt_sr_u
                        pq.cancel(sr_u)
                        _debug('DIRECTION (%d, %d) -> (%d, %d)', v, u, u, v)
                    # set the tail of the union outcome
                    tail_[sr_kept] = tail_[sr_u]
            elif u == tail_[sr_u]:
                # set the tail of the union outcome
                tail_[sr_kept] = sr_u
            else:
                # set the tail of the union outcome
                tail_[sr_kept] = tail_[sr_u]

        sr_dropped = sr_u
        root = A.nodes[sr_kept]['root']

        if use_blockage:
            _debug('<angle_span> //%s//', union_span_[root])
            subtree_span__[sr_kept] = union_span_

            for hop, corner, r, dropped in changes:
                if dropped is not None:
                    # detour corner swap
                    # hop->dropped->r changes to hop->corner->r
                    is_corner_[dropped] = False
                    if dropped in detours_via_prime_:
                        del detours_via_prime_[dropped]
                    is_root_nb__[r][dropped] = False
                else:
                    # detour segment creation (hop->corner->r)
                    is_root_nb__[r][hop] = False
                detours_via_prime_[corner].append(hop)
                is_corner_[corner] = True
                is_root_nb__[r][corner] = True

            # update the component's blocked set
            #  for r, subtree_blocked_ in zip(roots, subtree_blocked__):
            #      subtree_blocked_[sr_kept] |= (
            #          subtree_blocked_[sr_dropped] | A[u][v]['blocked__'][r]
            #      )
            #      subtree_blocked_[sr_kept] &= ~subtree_[sr_kept]

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
                who_targets_[sr_kept_target].discard(sr_kept)
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
        _debug('<add edge> «%d~%d» subroot <%d>', u, v, sr_kept)
        #  _debug('TAIL of %d: %d', sr_kept, tail_[sr_kept])
        if _lggr.isEnabledFor(logging.DEBUG) and pq:
            _debug('heap top: <%d>, «%s» %.3f', pq[0][-2], pq[0][-1], pq[0][0])
        else:
            _debug('heap EMPTY')
        S.add_edge(u, v)
        A.remove_edge(u, v)
        A.remove_edges_from(edge_conflicts(u, v, diagonals))

        # finished adding the edge, now check the consequences

        if method == 'rootlust' and ((u, v) if u < v else (v, u)) not in diagonals:
            # this fixes unions that result in 2 sides of a triangle being used but
            #   where the unused side is not the longest one (this fix makes it so)
            to_swap = ()
            for rot in ('cw', 'ccw'):
                s = P_A[v][u][rot]
                if P_A[s][v][rot] != u:
                    # uvs is not a triangle
                    continue
                # TODO: redundant `and`: is `subtree[s]` way faster than `s in S[v]`?
                if subtree[s] and s in S[v]:
                    Aʹs = Aʹ[s]
                    if u in Aʹs and Aʹs[u]['length'] < Aʹs[v]['length']:
                        to_swap = (s, v, u)
                        break
                diagonal = diagonals.inv.get((s, v) if s < v else (v, s))
                if diagonal is not None:
                    w, x = diagonal
                    t = w if x == v else x
                    if subtree[t] and t in S[v]:
                        Aʹt = Aʹ[t]
                        if u in Aʹt and Aʹt[u]['length'] < Aʹt[v]['length']:
                            to_swap = (t, v, u)
                            break
            if to_swap:
                pivot, v, u = to_swap
                S.remove_edge(v, pivot)
                S.add_edge(u, pivot)
                if pivot in A[u]:
                    A.remove_edge(u, pivot)

        if capacity_left > 0:
            #  if method in ('rootlust', 'radial_EW'):
            if method != 'esau_williams':
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
            is_extendable_[sr_kept] = False
            is_root_nb__[root][sr_kept] = True
            if sr_kept in pq.tags:
                # this is required because of i=0 feeders
                pq.cancel(sr_kept)
            subtree_nodes = tuple(subtree.search(_ONE))
            # by removing nodes, all the useless edges are discarded
            A.remove_nodes_from(subtree_nodes)
            _debug('subtree complete <%d>: %s', sr_kept, subtree_nodes)
            is_stale_[list(who_targets_[sr_dropped] | who_targets_[sr_kept])] = True
            who_targets_[sr_kept] = None
        is_extendable_[sr_dropped] = False
        subtree_[sr_dropped] = None
        who_targets_[sr_dropped] = None
    # END: main loop

    # add feeders (the 'ringed' method closes each path subtree into a ring)
    is_subroot_ = bitarray(subtree is not None for subtree in subtree_)
    if ringed and R > 1:
        # each ring subroot attaches to its own nearest root; when the two
        # prefer different roots the ring bridges them (calcload reads the pair
        # of feeders back as a (r1, r2) ring)
        for sr in is_subroot_.search(_ONE):
            S.add_edge(int(d2roots[sr].argmin()) - R, sr)
            far = tail_[sr]
            if far != sr:
                S.add_edge(int(d2roots[far].argmin()) - R, far)
    else:
        for r, rootmask_ in zip(roots, rootmask__):
            for sr in (rootmask_ & is_subroot_).search(_ONE):
                S.add_edge(r, sr)
    S.graph['topology'] = (
        Topology.RINGED
        if ringed
        else Topology.RADIAL
        if method == 'radial_EW'
        else Topology.BRANCHED
    )
    # ringed: close each path subtree into a ring (adds open points); else set loads
    if ringed:
        split_rings_and_calc_loads(S, Aʹ)
    else:
        calcload(S)
    # algorithm finished, store some info in the graph object
    S.graph.update(
        runtime=time.perf_counter() - start_time,
        capacity=capacity_report,
        creator='constructor',
        iterations=i,
        method_options=dict(
            method=method,
            fun_fingerprint=_constructor_fun_fingerprint,
        ),
    )
    if radial_like:
        S.graph['num_insertions'] = num_insertions
    #  if keep_log:
    #      S.graph['method_log'] = log
    return S


_constructor_fun_fingerprint = fingerprint_function(constructor)
