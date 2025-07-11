# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import time
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from ..crossings import edge_crossings
from ..geometric import (
    angle,
    angle_helpers,
    angle_oracles_factory,
    apply_edge_exemptions,
    assign_root,
    is_bunch_split_by_corner,
    is_crossing,
    is_same_side,
)
from ..interarraylib import L_from_G, fun_fingerprint
from ..mesh import make_planar_embedding
from ..utils import Alerter, F, NodeStr
from .priorityqueue import PriorityQueue

__all__ = ()

_lggr = logging.getLogger(__name__)
debug, info, warn, error = _lggr.debug, _lggr.info, _lggr.warning, _lggr.error


def OBEW(
    L,
    capacity=8,
    rootlust=None,
    maxiter=10000,
    maxDepth=4,
    MARGIN=1e-4,
    warnwhere=None,
    weightfun=None,
):
    """Obstacle Bypassing Esau-Williams heuristic for C-MST.

    Recommended `rootlust`: '0.6*cur_capacity/capacity'

    Args:
        L: networkx.Graph
        capacity: max number of terminals in a subtree
        rootlust: expression to use for biasing weights
        warnwhere: print debug info based on utils.Alerter

    Returns:
        G_cmst: networkx.Graph
    """

    start_time = time.perf_counter()

    if warnwhere is not None:
        Awarn = Alerter(warnwhere, 'i')
    else:  # could check if debug is True and make warn = print

        def Awarn(*args, **kwargs):
            pass

    # rootlust_formula = '0.7*(cur_capacity/(capacity - 1))**2'
    # rootlust_formula = f'0.6*cur_capacity/capacity'
    # rootlust_formula = f'0.6*(cur_capacity + 1)/capacity'
    # rootlust_formula = f'0.6*cur_capacity/(capacity - 1)'

    if rootlust is None:

        def rootlustfun(_):
            return 0.0
    else:
        rootlustfun = eval('lambda cur_capacity: ' + rootlust, locals())
    # rootlust = lambda cur_capacity: 0.7*(cur_capacity/(capacity - 1))**2

    # save relevant options to store in the graph later
    options = dict(MARGIN=MARGIN, variant='C', rootlust=rootlust)

    R, T, B = (L.graph[k] for k in 'RTB')
    _T = range(T)
    roots = range(-R, 0)

    # list of variables indexed by vertex id:
    #     d2roots, d2rootsRank, angles, anglesRank
    #     subtree_, VertexC
    # list of variables indexed by subtree id:
    #     CompoIn, CompoLolim, CompoHilim
    # dicts keyed by subtree id
    #     DetourHop, detourLoNotHi
    # sets of subtree ids:
    #     commited_
    #
    # need to have a table of vertex -> subroot node

    # TODO: do away with pre-calculated crossings
    Xings = L.graph.get('crossings')

    # crossings = L.graph['crossings']
    # BEGIN: prepare auxiliary graph with all allowed edges and metrics
    _, A = make_planar_embedding(L)
    assign_root(A)
    P = A.graph['planar']
    diagonals = A.graph['diagonals']
    d2roots = A.graph['d2roots']
    d2rootsRank = rankdata(d2roots, method='dense', axis=0)
    angles, anglesRank = angle_helpers(A)
    union_limits, angle_ccw = angle_oracles_factory(angles, anglesRank)

    # apply weightfun on all delaunay edges
    if weightfun is not None:
        # TODO: fix `apply_edge_exemptions()` for the
        #       `delaunay()` without triangles
        apply_edge_exemptions(A)
        options['weightfun'] = weightfun.__name__
        options['weight_attr'] = 'length'
        for _, _, data in A.edges(data=True):
            data['length'] = weightfun(data)
    # removing root nodes from A to speedup enqueue_best_union
    # this may be done because G already starts with feeders
    A.remove_nodes_from(roots)
    # END: prepare auxiliary graph with all allowed edges and metrics

    # BEGIN: create initial star graph
    G = L_from_G(L) if L.number_of_edges() > 0 else L.copy()
    G.add_weighted_edges_from(
        ((n, r, d2roots[n, r]) for n, r in A.nodes(data='root')), weight='length'
    )
    nx.set_node_attributes(G, {n: r for n, r in A.nodes(data='root')}, 'root')
    # END: create initial star graph

    # BEGIN: helper data structures

    # upper estimate number of Detour nodes:
    # Dmax = round(2*T/3/capacity)
    Dmax = T

    # mappings from nodes
    # <subtree_>: maps nodes to the list of nodes in their subtree
    subtree_ = [[t] for t in _T] + (B + Dmax) * [None]
    # TODO: fnT might be better named Pof (Prime of)
    # <fnT>: farm node translation table
    #        to be used when indexing: VertexC, d2roots, angles, etc
    #        fnT[-R..(T+Dmax)] -> -R..T
    fnT = np.arange(T + B + Dmax + R)
    fnT[-R:] = roots

    # <Stale>: list of detour nodes that were discarded
    Stale = []

    # this is to make fnT available for plot animation
    # a new, trimmed, array be assigned after the heuristic is done
    G.graph['fnT'] = fnT

    n2s = NodeStr(fnT, T)
    # <subroot_>: maps vertices (terminals and clones) to subroot terminals
    subroot_ = list(_T) + (B + Dmax) * [None]

    # mappings from components (identified by their subroots)
    # <ComponIn>: maps component to set of components queued to merge in
    ComponIn = [set() for _ in _T]
    # <subtree_span_>: pairs (most_CW, most_CCW) of extreme nodes of each
    #                  subtree, indexed by subroot (former subroot)
    subtree_span__ = [[(t, t) for t in _T] for _ in roots]

    # mappings from roots
    # <commited_>: set of proximals of finished components (one set per root).
    #              proximals are the nodes (either terminals or detours) that
    #              are neighbors to root
    commited_ = [set() for _ in roots]

    # other structures
    # <pq>: queue prioritized by lowest tradeoff length
    pq = PriorityQueue()
    # enqueue_best_union()
    # <stale_subtrees>: deque for components that need to go through
    # stale_subtrees = deque()
    stale_subtrees = set()
    # <edges2ban>: deque for edges that should not be considered anymore
    # edges2ban = deque()
    # TODO: this is not being used, decide what to do about it
    edges2ban = set()
    VertexC = L.graph['VertexC']
    # number of Detour nodes added
    D = 0
    # <DetourHop>: maps subroot nodes to a list of nodes of the Detour path
    #              (root is not on the list)
    DetourHop = defaultdict(list)
    detourLoNotHi = dict()
    # detour = defaultdict(list)
    # detouroverlaps = {}

    # <i>: iteration counter
    i = 0
    # <prevented_crossing>: counter for edges discarded due to crossings
    prevented_crossings = 0
    # END: helper data structures

    def is_crossing_feeder(root, subroot, u, v, touch_is_cross=False):
        less = np.less_equal if touch_is_cross else np.less
        # get the primes of all nodes
        _subroot, _u, _v = fnT[[subroot, u, v]]
        uvA = angles[_v, root] - angles[_u, root]
        swaped = (-np.pi < uvA) & (uvA < 0.0) | (np.pi < uvA)
        lo, hi = (_v, _u) if swaped else (_u, _v)
        loR, hiR, srR = anglesRank[(lo, hi, _subroot), root]
        W = loR > hiR  # wraps +-pi
        supL = less(loR, srR)  # angle(low) <= angle(probe)
        infH = less(srR, hiR)  # angle(probe) <= angle(high)
        if ~W & supL & infH | W & ~supL & infH | W & supL & ~infH:
            if not is_same_side(*VertexC[[_u, _v, root, _subroot]]):
                # crossing subroot
                debug(
                    '<crossing> discarding «%s–%s»: would cross subroot <%s>',
                    F[u],
                    F[v],
                    F[subroot],
                )
                return True
        return False

    def commit_subroot(root, sr_v):
        if sr_v not in commited_[root]:
            commited_[root].add(sr_v)
            log.append((i, 'commit', (sr_v, root)))
            debug('<commit_subroot> feeder [%s–%s] added', F[sr_v], F[root])

    def get_union_choices_plain(subroot, forbidden=None):
        # gather all the edges leaving the subtree of subroot
        if forbidden is None:
            forbidden = set()
        forbidden.add(subroot)
        d2root = d2roots[fnT[subroot], G.nodes[subroot]['root']]
        capacity_left = capacity - len(subtree_[subroot])
        weighted_edges = []
        edges2discard = []
        for u in subtree_[subroot]:
            for v in A[u]:
                if subroot_[v] in forbidden or len(subtree_[v]) > capacity_left:
                    # useless edges
                    edges2discard.append((u, v))
                else:
                    # newd2root = d2roots[fnT[subroot_[v]], G.nodes[fnT[v]]['root']]
                    W = A[u][v]['length']
                    # if W <= d2root:  # TODO: what if I use <= instead of <?
                    if W < d2root:
                        # useful edges
                        #  tiebreaker = d2rootsRank[fnT[v], A[u][v]['root']]
                        tiebreaker = d2rootsRank[fnT[v], A.nodes[v]['root']]
                        weighted_edges.append((W, tiebreaker, u, v))
                        #  weighted_edges.append((W-(d2root - newd2root)/3,
                        #                           tiebreaker, u, v))
        return weighted_edges, edges2discard

    def get_union_choices(subroot, forbidden=None):
        # gather all the edges leaving the subtree of subroot
        if forbidden is None:
            forbidden = set()
        forbidden.add(subroot)
        root = G.nodes[subroot]['root']
        d2root = d2roots[subroot, root]
        capacity_left = capacity - len(subtree_[subroot])
        root_lust = rootlustfun(len(subtree_[subroot]))
        weighted_edges = []
        edges2discard = []
        for u in subtree_[subroot]:
            for v in A[u]:
                if subroot_[v] in forbidden or len(subtree_[v]) > capacity_left:
                    # useless edges
                    edges2discard.append((u, v))
                else:
                    d2rGain = d2root - d2roots[subroot_[v], G.nodes[fnT[v]]['root']]
                    W = A[u][v]['length']
                    # if W <= d2root:  # TODO: what if I use <= instead of <?
                    if W < d2root:
                        # useful edges
                        #  tiebreaker = d2rootsRank[fnT[v], A[u][v]['root']]
                        tiebreaker = d2rootsRank[fnT[v], A.nodes[v]['root']]
                        # weighted_edges.append((W, tiebreaker, u, v))
                        weighted_edges.append(
                            (W - d2rGain * root_lust, tiebreaker, u, v)
                        )
        return weighted_edges, edges2discard

    def sort_union_choices(weighted_edges):
        # this function could be outside esauwilliams()
        unordchoices = np.array(
            weighted_edges,
            dtype=[
                ('weight', np.float64),
                ('vd2rootR', np.int_),
                ('u', np.int_),
                ('v', np.int_),
            ],
        )
        # result = np.argsort(unordchoices, order=['weight'])
        # unordchoices  = unordchoices[result]

        # DEVIATION FROM Esau-Williams
        # rounding of weight to make ties more likely
        # tie-breaking by proximity of 'v' node to root
        # purpose is to favor radial alignment of components
        tempchoices = unordchoices.copy()
        tempchoices['weight'] /= tempchoices['weight'].min()
        tempchoices['weight'] = (20 * tempchoices['weight']).round()  # 5%

        result = np.argsort(tempchoices, order=['weight', 'vd2rootR'])
        choices = unordchoices[result]
        return choices

    def enqueue_best_union(subroot):
        debug('<enqueue_best_union> starting... subroot = <%s>', F[subroot])
        if edges2ban:
            debug('<<<<<<<edges2ban>>>>>>>>>>> _%d_', len(edges2ban))
        while edges2ban:
            # edge2ban = edges2ban.popleft()
            edge2ban = edges2ban.pop()
            ban_queued_union(*edge2ban)
        # () get component expansion edges with weight
        weighted_edges, edges2discard = get_union_choices(subroot)
        # discard useless edges
        A.remove_edges_from(edges2discard)
        # () sort choices
        choices = sort_union_choices(weighted_edges) if weighted_edges else []
        if len(choices) > 0:
            choice = choices[0]
            # merging is better than subroot, submit entry to pq
            # weight, u, v = choice
            weight, _, u, v = choice
            # tradeoff calculation
            tradeoff = weight - d2roots[fnT[subroot], A.nodes[subroot]['root']]
            pq.add(tradeoff, subroot, (u, v))
            ComponIn[subroot_[v]].add(subroot)
            debug(
                '<pushed> sr_u <%s>, «%s–%s», tradeoff = %.3f',
                F[subroot],
                F[u],
                F[v],
                -tradeoff,
            )

        else:
            # no viable edge is better than subroot for this node
            # this becomes a final subroot
            if i:  # run only if not at i = 0
                # commited feeders at iteration 0 do not cross any other edges
                # they are not included in commited_ because the algorithm
                # considers the feeders extending to infinity (not really)
                root = A.nodes[subroot]['root']
                commit_subroot(root, subroot)
                # check_heap4crossings(root, subroot)
            debug('<cancelling> %s', F[subroot])
            if subroot in pq.tags:
                # i=0 feeders and check_heap4crossings reverse_entry
                # may leave accepting subtrees out of pq
                pq.cancel(subroot)

    def ban_queued_union(sr_u, u, v):
        if (u, v) in A.edges:
            A.remove_edge(u, v)
        else:
            debug('<<< UNLIKELY <ban_queued_union()> «%s–%s» not in A >>>', F[u], F[v])
        sr_v = subroot_[v]
        # TODO: think about why a discard was needed
        ComponIn[sr_v].discard(sr_u)
        # stale_subtrees.appendleft(sr_u)
        stale_subtrees.add(sr_u)
        # enqueue_best_union(sr_u)

        # BEGIN: block to be simplified
        is_reverse = False
        componin = sr_v in ComponIn[sr_u]
        reverse_entry = pq.tags.get(sr_v)
        if reverse_entry is not None:
            _, _, _, (s, t) = reverse_entry
            if (t, s) == (u, v):
                # TODO: think about why a discard was needed
                ComponIn[sr_u].discard(sr_v)
                # this is assymetric on purpose (i.e. not calling
                # pq.cancel(sr_u), because enqueue_best_union will do)
                pq.cancel(sr_v)
                enqueue_best_union(sr_v)
                is_reverse = True

        if componin != is_reverse:
            # TODO: Why did I expect always False here? It is sometimes True.
            debug(
                '«%s–%s», sr_u <%s>, sr_v <%s> componin: %s, is_reverse: %s',
                F[u],
                F[v],
                F[sr_u],
                F[sr_v],
                componin,
                is_reverse,
            )

        # END: block to be simplified

    # TODO: check if this function is necessary (not used)
    def abort_edge_addition(sr_u, u, v):
        if (u, v) in A.edges:
            A.remove_edge(u, v)
        else:
            print(
                f'<<<< UNLIKELY <abort_edge_addition()> {n2s(u, v)} not in A.edges >>>>'
            )
        ComponIn[subroot_[v]].remove(sr_u)
        enqueue_best_union(sr_u)

    # TODO: check if this function is necessary (not used)
    def get_subtrees_closest_node(subroot, origin):
        componodes = subtree_[subroot]
        if len(componodes) > 1:
            dist = np.squeeze(
                cdist(VertexC[fnT[componodes]], VertexC[np.newaxis, fnT[origin]])
            )
        else:
            dist = np.array(
                [
                    np.hypot(
                        *(
                            VertexC[fnT[componodes[0]]]
                            - VertexC[np.newaxis, fnT[origin]]
                        ).T
                    )
                ]
            )
        idx = np.argmin(dist)
        closest = componodes[idx]
        return closest

    def get_crossings(s, t, detour_waiver=False):
        """generic crossings checker
        common node is not crossing"""
        s_, t_ = fnT[[s, t]]
        st = (s_, t_) if s_ < t_ else (t_, s_)
        if st in P.edges or st in diagonals:
            # <(s_, t_) is in the expanded Delaunay edge set>
            Xlist = edge_crossings(s_, t_, G, diagonals)
            # crossings with expanded Delaunay already checked
            # just detour edges missing
            nbunch = list(range(T, T + D))
        else:
            # <(s, t) is not in the expanded Delaunay edge set>
            Xlist = []
            nbunch = None  # None means all nodes
        sC, tC = VertexC[[s_, t_]]
        # st_is_detour = s >= T or t >= T
        for w_x in G.edges(nbunch):
            w, x = w_x
            w_, x_ = fnT[[w, x]]
            # both_detours = st_is_detour and (w >= T or x >= T)
            skip = detour_waiver and (w >= T or x >= T)
            if skip or s_ == w_ or t_ == w_ or s_ == x_ or t_ == x_:
                # <edges have a common node>
                continue
            if is_crossing(sC, tC, *VertexC[[w_, x_]], touch_is_cross=True):
                Xlist.append(w_x)
        return Xlist

    def deprecated_get_crossings(s, t):
        # TODO: THIS RELIES ON precalculated crossings
        sC, tC = VertexC[fnT[[s, t]]]
        rootC = VertexC[-R:]
        if np.logical_or.reduce(((sC - rootC) * (tC - rootC)).sum(axis=1) < 0):
            # pre-calculation pruned edges with more than 90° angle
            # so the output will be equivalent to finding a crossings
            return (None,)
        Xlist = []
        for w, x in Xings[frozenset(fnT[[s, t]])]:
            if G.has_edge(w, x):
                # debug and print(f'{n2s(w, x)} crosses {n2s(s, t)}')
                Xlist.append((w, x))
        return Xlist

    def plan_detour(
        root, blocked, goal_, u, v, barrierLo, barrierHi, savings, depth=0, remove=set()
    ):
        """
        (blocked, goal_) is the detour segment
        (u, v) is an edge crossing it
        barrierLo/Hi are the extremes of the subtree of (u, v) wrt root
        savings = <benefit of the edge addition> - <previous detours>
        """
        # print(f'[{i}] <plan_detour_recursive[{depth}]> {n2s(u, v)} '
        #       f'blocking {n2s(blocked, goal_)}')
        goalC = VertexC[goal_]
        subroot = subroot_[blocked]
        detourHop = DetourHop[subroot]
        blocked_ = fnT[blocked]
        Awarn(
            f'({depth}) '
            + ('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n' if depth == 0 else '')
            + f'{n2s(u, v)} blocks {n2s(blocked)}, '
            f'subroot: {n2s(subroot)}'
        )

        # <refL>: length of the edge crossed by (u, v) – reference of cost
        if goal_ < 0:  # goal_ is a root
            refL = d2roots[blocked_, goal_]
        else:
            refL = np.hypot(*(goalC - VertexC[blocked_]).T)
        Awarn(f'refL: {refL:.0f}')

        is_blocked_a_clone = blocked >= T
        if is_blocked_a_clone:
            blockedHopI = detourHop.index(blocked)
            Awarn(f'detourHop: {n2s(*detourHop)}')

        # TODO: this would be a good place to handle this special case
        #       but it requires major refactoring of the code
        # if blocked_ in (u, v) and goal_ < 0 and detourHop[-2] >= T:
        #    # <(u, v) edge is actually pushing blocked to one of the limits of
        #    #  Barrier, this means the actual blocked hop is further away>
        #    blockedHopI -= 1
        #    actual_blocked = detourHop[blockedHopI]
        #    remove = remove | {blocked}
        #    refL += np.hypot(*(VertexC[fnT[actual_blocked]]
        #                       - VertexC[blocked_]))
        #    blocked = actual_blocked
        #    blocked_ = fnT[blocked]
        #    is_blocked_a_clone = blocked >= T

        not2hook = remove.copy()

        Barrier = subtree_[u] + subtree_[v]

        store = []
        # look for detours on the Lo and Hi sides of barrier
        for corner_, loNotHi, sidelabel in (
            (barrierLo, True, 'Lo'),
            (barrierHi, False, 'Hi'),
        ):
            Awarn(f'({depth}|{sidelabel}) BEGIN: {n2s(corner_)}')

            # block for possible future change (does nothing)
            nearest_root = -R + np.argmin(d2roots[corner_])
            if nearest_root != root:
                debug(
                    '[%d] corner: %s is closest to %s than to %s',
                    i,
                    corner_,
                    nearest_root,
                    root,
                )

            # block for finding the best hook
            cornerC = VertexC[corner_]
            Blocked = subtree_[subroot].copy()
            for rem in remove:
                Blocked.remove(rem)
            if is_blocked_a_clone:
                for j, (hop2check, prevhop) in enumerate(
                    zip(detourHop[blockedHopI::-1], detourHop[blockedHopI - 1 :: -1])
                ):
                    if j == 2:
                        debug(
                            '[%d] (2nd iter) depth: %d, blocked: %s, subroot: %s',
                            i,
                            depth,
                            blocked,
                            subroot,
                        )
                        # break
                    hop2check_ = fnT[hop2check]
                    is_hop_a_barriers_clone = hop2check_ in Barrier
                    prevhop_ = fnT[prevhop]
                    prevhopC = VertexC[prevhop_]
                    hop2checkC = VertexC[hop2check_]
                    discrim = angle(prevhopC, hop2checkC, cornerC) > 0
                    is_concave_at_hop2check = (
                        not is_hop_a_barriers_clone
                        and (discrim != detourLoNotHi[hop2check])
                    ) or (is_hop_a_barriers_clone and (discrim != loNotHi))
                    Awarn(
                        f'concavity check at {n2s(hop2check)}: '
                        f'{is_concave_at_hop2check}'
                    )
                    if is_concave_at_hop2check:
                        # Awarn(f'CONCAVE at {n2s(hop2check)}')
                        #       f'at {n2s(hop2check)}: {is_concave_at_hop2check}, '
                        #       f'remove: {", ".join([F[r] for r in remove])}')
                        if hop2check not in remove:
                            if prevhop >= T and hop2check not in Barrier:
                                prevprevhop = detourHop[blockedHopI - j - 2]
                                prevprevhopC = VertexC[fnT[prevprevhop]]
                                prevhopSubTreeC = VertexC[
                                    [h for h in subtree_[prevhop_] if h < T]
                                ]
                                # TODO: the best thing would be to use here the
                                #       same split algorithm used later
                                #       (barrsplit)
                                if is_bunch_split_by_corner(
                                    prevhopSubTreeC, cornerC, prevhopC, prevprevhopC
                                )[0]:
                                    break
                            # get_crossings(corner_, prevhop_)):
                            # <detour can actually bypass the previous one>
                            Blocked.remove(hop2check)
                            not2hook |= {hop2check}
                    else:
                        break
            # get the distance from every node in Blocked to corner
            D2corner = np.squeeze(
                cdist(VertexC[fnT[Blocked]], VertexC[np.newaxis, corner_])
            )
            if not D2corner.shape:
                D2corner = [float(D2corner)]
            hookI = np.argmin(D2corner)
            hook = Blocked[hookI]
            hookC = VertexC[fnT[hook]]

            # block for calculating the length of the path to replace
            prevL = refL
            shift = hook != blocked
            if shift and is_blocked_a_clone:
                prevhop_ = blocked_
                for hop in detourHop[blockedHopI - 1 :: -1]:
                    hop_ = fnT[hop]
                    prevL += np.hypot(*(VertexC[hop_] - VertexC[prevhop_]))
                    Awarn(f'adding {n2s(hop_, prevhop_)}')
                    prevhop_ = hop_
                    if hop == hook or hop < T:
                        break
            Awarn(f'prevL: {prevL:.0f}')

            # check if the bend at corner is necessary
            discrim = angle(hookC, cornerC, goalC) > 0
            dropcorner = discrim != loNotHi
            # if hook < T and dropcorner:
            # if dropcorner and False:  # TODO: conclude this test
            if dropcorner and fnT[hook] != corner_:
                Awarn(f'DROPCORNER {sidelabel}')
                # <bend unnecessary>
                detourL = np.hypot(*(goalC - hookC))
                addedL = prevL - detourL
                # debug and print(f'[{i}] CONCAVE: '
                # print(f'[{i}] CONCAVE: '
                #       f'{n2s(hook, corner_, goal_)}')
                detourX = get_crossings(goal_, hook, detour_waiver=True)
                if not detourX:
                    path = (hook,)
                    LoNotHi = tuple()
                    direct = True
                    store.append((addedL, path, LoNotHi, direct, shift))
                    continue
            Awarn(f'hook: {n2s(hook)}')
            nearL = (
                d2roots[corner_, goal_] if goal_ < 0 else np.hypot(*(goalC - cornerC))
            )
            farL = D2corner[hookI]
            addedL = farL + nearL - prevL
            Awarn(f'{n2s(hook, corner_, goal_)} addedL: {addedL:.0f}')
            if addedL > savings:
                # <detour is more costly than the savings from (u, v)>
                store.append((np.inf, (hook, corner_)))
                continue

            # TODO: the line below is risky. it disconsiders detour nodes
            #       when checking if a subtree is split
            BarrierPrime = np.array([b for b in Barrier if b < T])
            BarrierC = VertexC[fnT[BarrierPrime]]

            is_barrier_split, insideI, outsideI = is_bunch_split_by_corner(
                BarrierC, hookC, cornerC, goalC
            )

            # TODO: think if this subroot edge waiver is correct
            FarX = [
                farX
                for farX in get_crossings(hook, corner_, detour_waiver=True)
                if farX[0] >= 0 and farX[1] >= 0
            ]
            # this will condense multiple edges from the same subtree into one
            FarXsubtree = {subroot_[s]: (s, t) for s, t in FarX}

            # BEGIN barrHack block
            Nin, Nout = len(insideI), len(outsideI)
            if is_barrier_split and (Nin == 1 or Nout == 1):
                # <possibility of doing the barrHack>
                barrHackI = outsideI[0] if Nout <= Nin else insideI[0]
                barrierX_ = BarrierPrime[barrHackI]
                # TODO: these criteria are too ad hoc, improve it
                if subroot_[barrierX_] in FarXsubtree:
                    del FarXsubtree[subroot_[barrierX_]]
                elif (
                    barrierX_ not in G[corner_]
                    and d2roots[barrierX_, root] > 1.1 * d2roots[fnT[hook], root]
                ):
                    # <this is a spurious barrier split>
                    # ignore this barrier split
                    is_barrier_split = False
                    Awarn('spurious barrier split detected')
            else:
                barrHackI = None
            # END barrHack block

            # possible check to include: (something like this)
            # if (anglesRank[corner_, root] >
            #     anglesRank[ComponHiLim[subroot_[blocked], root]]
            Awarn(
                f'barrsplit: {is_barrier_split}, inside: {len(insideI)}, '
                f'outside: {len(outsideI)}, total: {len(BarrierC)}'
            )
            # if is_barrier_split or get_crossings(corner_, goal_):
            barrAddedL = 0
            nearX = get_crossings(corner_, goal_, detour_waiver=True)
            if nearX:
                Awarn(f'nearX: {", ".join(n2s(*X) for X in nearX)}')
            if nearX or (is_barrier_split and barrHackI is None):
                # <barrier very split or closer segment crosses some edge>
                store.append((np.inf, (hook, corner_)))
                continue
            # elif (is_barrier_split and len(outsideI) == 1 and
            #       len(insideI) > 3):
            elif is_barrier_split:
                # <barrier slightly split>
                # go around small interferences with the barrier itself
                Awarn(
                    f'SPLIT: {n2s(hook, corner_)} leaves '
                    f'{n2s(BarrierPrime[barrHackI])} isolated'
                )
                # will the FarX code handle this case?
                barrierXC = BarrierC[barrHackI]
                # barrpath = (hook, barrierX, corner_)
                # two cases: barrHop before or after corner_
                corner1st = d2rootsRank[barrierX_, root] < d2rootsRank[corner_, root]
                if corner1st:
                    barrAddedL = (
                        np.hypot(*(goalC - barrierXC))
                        + np.hypot(*(cornerC - barrierXC))
                        - nearL
                    )
                    barrhop = (barrierX_,)
                else:
                    barrAddedL = (
                        np.hypot(*(hookC - barrierXC))
                        + np.hypot(*(cornerC - barrierXC))
                        - farL
                    )
                    barrhop = (corner_,)
                    corner_ = barrierX_
                Awarn(f'barrAddedL: {barrAddedL:.0f}')
                addedL += barrAddedL
                barrLoNotHi = (loNotHi,)
            else:
                barrhop = tuple()
                barrLoNotHi = tuple()

            if len(FarXsubtree) > 1:
                Awarn(
                    f'NOT IMPLEMENTED: many ({len(FarXsubtree)}) '
                    f'crossings of {n2s(hook, corner_)} ('
                    f'{", ".join([n2s(a, b) for a, b in FarXsubtree.values()])})'
                )
                store.append((np.inf, (hook, corner_)))
                continue
            elif FarXsubtree:  # there is one crossing
                subbarrier, farX = FarXsubtree.popitem()
                # print('farX:', n2s(*farX))
                if depth > maxDepth:
                    warn('<plan_detour[%d]> max depth (%s) exceeded.', depth, maxDepth)
                    store.append((np.inf, (hook, corner_)))
                    continue
                else:
                    new_barrierLo, new_barrierHi = subtree_span__[root][subbarrier]
                    remaining_savings = savings - addedL
                    subdetour = plan_detour(
                        root,
                        hook,
                        corner_,
                        *farX,
                        new_barrierLo,
                        new_barrierHi,
                        remaining_savings,
                        depth + 1,
                        remove=not2hook,
                    )
                    if subdetour is None:
                        store.append((np.inf, (hook, corner_)))
                        continue
                    subpath, subaddedL, subLoNotHi, subshift = subdetour

                    subcorner = subpath[-1]
                    # TODO: investigate why plan_detour is suggesting
                    #       hops that are not primes
                    subcorner_ = fnT[subcorner]
                    subcornerC = VertexC[subcorner_]
                    # check if the bend at corner is necessary
                    nexthopC = fnT[barrhop[0]] if barrhop else goalC
                    discrim = angle(subcornerC, cornerC, nexthopC) > 0
                    dropcorner = discrim != loNotHi
                    # TODO: revisit and fix this 'if' block
                    #  if dropcorner:
                    #      subcornerC = VertexC[subcorner_]
                    #      dropL = np.hypot(*(nexthopC - subcornerC))
                    #      dc_addedL = dropL - prevL + barrAddedL
                    #      direct = len(subpath) == 1
                    #      if not direct:
                    #          subfarL = np.hypot(*(cornerC - subcornerC))
                    #          subnearL = subaddedL - subfarL + nearL
                    #          dc_addedL += subnearL
                    #      # debug and print(f'[{i}] CONCAVE: '
                    #      # print(f'[{i}] CONCAVE: '
                    #      #       f'{n2s(hook, corner_, goal_)}')
                    #      dcX = get_crossings(subcorner_, corner_,
                    #                          detour_waiver=True)
                    #      if not dcX:
                    #          print(f'[{i}, {depth}] dropped corner '
                    #                f'{n2s(corner_)}')
                    #          path = (*subpath, *barrhop)
                    #          LoNotHi = (*subLoNotHi, *barrLoNotHi)
                    #          store.append((dc_addedL, path, LoNotHi,
                    #                        direct, shift))
                    #          continue

                    # combine the nested detours
                    path = (*subpath, corner_, *barrhop)
                    LoNotHi = (*subLoNotHi, *barrLoNotHi)
                    addedL += subaddedL
                    shift = subshift
            else:  # there are no crossings
                path = (hook, corner_, *barrhop)
                LoNotHi = (*barrLoNotHi,)
            Awarn(f'{sidelabel} STORE: {n2s(*path)} addedL: {addedL:.0f}')
            # TODO: properly check for direct connection
            # TODO: if shift: test if there is a direct path
            #       from hook to root
            direct = False
            # TODO: deprecate shift?
            store.append((addedL, path, LoNotHi, direct, shift))

        # choose between the low or high corners
        if store[0][0] < savings or store[1][0] < savings:
            loNotHi = store[0][0] < store[1][0]
            cost, path, LoNotHi, direct, shift = store[not loNotHi]
            Awarn(
                f'({depth}) '
                f'take: {n2s(*store[not loNotHi][1], goal_)} (@{cost:.0f}), '
                f'drop: {n2s(*store[loNotHi][1], goal_)} '
                f'(@{store[loNotHi][0]:.0f})'
            )
            debug(
                f'<plan_detour[{depth}]>: {n2s(u, v)} crosses '
                f'{n2s(blocked, goal_)} but {n2s(*path, goal_)} '
                'may be used instead.'
            )
            return (path, cost, LoNotHi + (loNotHi,), shift)
        return None

    def add_corner(hook, corner_, subroot, loNotHi):
        nonlocal D
        D += 1

        if D > Dmax:
            # TODO: extend VertexC, fnT and subroot_
            warn('@@@@@@@@@@@@@@ Dmax REACHED @@@@@@@@@@@@@@')
        corner = T + B + D - 1

        # update coordinates mapping fnT
        fnT[corner] = corner_

        # subtree being rerouted
        subroot_[corner] = subroot
        subtree_[subroot].append(corner)
        subtree_[corner] = subtree_[subroot]

        # update DetourHop
        DetourHop[subroot].append(corner)
        # update detourLoNotHi
        detourLoNotHi[corner] = loNotHi
        # add Detour node
        G.add_node(corner, kind='detour', root=G.nodes[hook]['root'])
        log.append((i, 'addDN', (corner_, corner)))
        # add detour edges
        length = np.hypot(*(VertexC[fnT[hook]] - VertexC[corner_]).T)
        G.add_edge(
            hook, corner, length=length, kind='detour', color='yellow', style='dashed'
        )
        log.append((i, 'addDE', (hook, corner, fnT[hook], corner_)))
        return corner

    def move_corner(corner, hook, corner_, subroot, loNotHi):
        # update translation tables
        fnT[corner] = corner_

        # update DetourHop
        DetourHop[subroot].append(corner)
        # update detourLoNotHi
        detourLoNotHi[corner] = loNotHi
        # update edges lengths
        farL = np.hypot(*(VertexC[fnT[hook]] - VertexC[corner_]).T)
        # print(f'[{i}] updating {n2s(hook, corner)}')
        G[hook][corner].update(length=farL)
        log.append((i, 'movDN', (hook, corner, fnT[hook], corner_)))
        return corner

    def make_detour(blocked, path, LoNotHi, shift):
        hook, *Corner_ = path

        subroot = subroot_[blocked]
        root = G.nodes[subroot]['root']
        # if Corner_[0] is None:
        if not Corner_:
            # <a direct feeder replacing previous feeder>
            # TODO: this case is very outdated, probably wrong
            debug('[%d] <make_detour> direct subroot «%s–%s»', i, hook, root)
            # remove previous proximal
            commited_[root].remove(blocked)
            subtree_[subroot].remove(blocked)
            G.remove_edge(blocked, root)
            log.append((i, 'remE', (blocked, root)))
            # make a new direct feeder
            length = d2roots[fnT[hook], root]
            G.add_edge(
                hook, root, length=length, kind='detour', color='yellow', style='dashed'
            )
            log.append((i, 'addDE', (hook, root, fnT[hook], root)))
            commited_[root].add(hook)
        else:
            detourHop = DetourHop[subroot]
            if blocked < T or hook == blocked:
                # <detour only affects one feeder segment: blocked-root>

                # remove the blocked proximal edge
                commited_[root].remove(blocked)
                G.remove_edge(blocked, root)
                log.append((i, 'remE', (blocked, root)))

                # create new corner nodes
                if hook < T:
                    # add the first entry in DetourHop (always prime)
                    detourHop.append(hook)
                for corner_, loNotHi in zip(Corner_, LoNotHi):
                    corner = add_corner(hook, corner_, subroot, loNotHi)
                    hook = corner
                # add the last feeder segment: last corner node to root
                length = d2roots[corner_, root]
                G.add_edge(
                    corner,
                    root,
                    length=length,
                    kind='detour',
                    color='yellow',
                    style='dashed',
                )
                log.append((i, 'addDE', (corner, root, corner_, root)))
                commited_[root].add(corner)
            else:
                # <detour affects edges further from blocked node>

                assert blocked == detourHop[-1]
                # stales = iter(detourHop[-2:0:-1])

                # number of new corners needed
                newN = len(Corner_) - len(detourHop) + 1

                try:
                    j = detourHop.index(hook)
                except ValueError:
                    # <the path is starting from a new prime>
                    j = 0
                    stales = iter(detourHop[1:])
                    k = abs(newN) if newN < 0 else 0
                    new2stale_cut = detourHop[k : k + 2]
                    detourHop.clear()
                    detourHop.append(hook)
                else:
                    stales = iter(detourHop[j + 1 :])
                    newN += j
                    k = j + (abs(newN) if newN < 0 else 0)
                    new2stale_cut = detourHop[k : k + 2]
                    del detourHop[j + 1 :]
                # print(f'[{i}] <make_detour> removing {n2s(*new2stale_cut)}, '
                #       f'{new2stale_cut in G.edges}')
                # newN += j
                # TODO: this is not handling the case of more stale hops than
                #       necessary for the detour path (must at least cleanup G)
                if newN < 0:
                    debug(
                        '[%d] WARNING <make_detour> more stales than needed: %d',
                        i,
                        abs(newN),
                    )
                    while newN < 0:
                        stale = next(stales)
                        G.remove_node(stale)
                        log.append((i, 'remN', stale))
                        subtree_[subroot].remove(stale)
                        Stale.append(stale)
                        newN += 1
                else:
                    G.remove_edge(*new2stale_cut)
                    log.append((i, 'remE', new2stale_cut))
                for j, (corner_, loNotHi) in enumerate(zip(Corner_, LoNotHi)):
                    if j < newN:
                        # create new corner nodes
                        corner = add_corner(hook, corner_, subroot, loNotHi)
                    else:
                        stale = next(stales)
                        if j == newN:
                            # add new2stale_cut edge
                            # print(f'[{i}] adding {n2s(hook, stale)}')
                            G.add_edge(
                                hook,
                                stale,
                                kind='detour',
                                color='yellow',
                                style='dashed',
                            )
                            log.append((i, 'addDE', (hook, stale, fnT[hook], corner_)))
                        # move the stale corners to their new places
                        corner = move_corner(stale, hook, corner_, subroot, loNotHi)
                    hook = corner
                # update the subroot edge length
                nearL = d2roots[corner_, root]
                G[corner][root].update(length=nearL)
                log.append((i, 'movDN', (corner, root, corner_, root)))

    def check_feeder_crossings(u, v, sr_v, sr_u):
        nonlocal tradeoff

        union = subtree_[u] + subtree_[v]
        r2keep = G.nodes[sr_v]['root']
        r2drop = G.nodes[sr_u]['root']

        if r2keep == r2drop:
            roots2check = (r2keep,)
        else:
            roots2check = (r2keep, r2drop)

        # assess the union's angle span
        unionHi = np.empty((len(roots),), dtype=int)
        unionLo = np.empty((len(roots),), dtype=int)
        for root, subtree_span_ in zip(roots, subtree_span__):
            keepLo, keepHi = subtree_span_[sr_v]
            dropLo, dropHi = subtree_span_[sr_u]
            unionLo[root], unionHi[root] = union_limits(
                root, u, dropLo, dropHi, v, keepLo, keepHi
            )
            debug(
                '<angle_span> root %s: //%s:%s// ',
                F[root],
                F[unionLo[root]],
                F[unionHi[root]],
            )

        abort = False
        Detour = {}

        for root in roots2check:
            for proximal in commited_[root] - {v}:
                if is_crossing_feeder(root, proximal, u, v, touch_is_cross=True) or (
                    proximal >= T
                    and fnT[proximal] in (u, v)
                    and (
                        is_bunch_split_by_corner(
                            VertexC[fnT[union]],
                            *VertexC[
                                fnT[[DetourHop[subroot_[proximal]][-2], proximal, root]]
                            ],
                        )[0]
                    )
                ):
                    # print('getting detour')
                    # detour = plan_detour(root, proximal,
                    #                     u, v, unionLo[root],
                    #                     unionHi[root], -tradeoff)
                    # TODO: it would be worth checking if changing roots is the
                    #       shortest path to avoid the (u, v) block
                    detour = plan_detour(
                        root,
                        proximal,
                        root,
                        u,
                        v,
                        unionLo[root],
                        unionHi[root],
                        -tradeoff,
                    )
                    if detour is not None:
                        Detour[proximal] = detour
                    else:
                        debug(
                            '<check_feeder_crossings> discarding «%s–%s»: '
                            'would block subroot %s',
                            F[u],
                            F[v],
                            proximal,
                        )
                        abort = True
                        break
            if abort:
                break

        if not abort and Detour:
            debug(
                '<check_feeder_crossings> detour options: %s',
                tuple(path for path, _, _, _ in Detour.values()),
            )
            # <crossing detected but detours are possible>
            detoursCost = sum((cost for _, cost, _, _ in Detour.values()))
            if detoursCost < -tradeoff:
                # add detours to G
                detdesc = [
                    f'blocked {n2s(blocked)}, '
                    f'subroot {n2s(subroot_[blocked])}, '
                    f'{n2s(*path)} '
                    f'@{cost:.0f}'
                    for blocked, (path, cost, loNotHi, shift) in Detour.items()
                ]
                Awarn('\n' + '\n'.join(detdesc))
                for blocked, (path, _, LoNotHi, shift) in Detour.items():
                    make_detour(blocked, path, LoNotHi, shift)
            else:
                debug(
                    'Multiple Detour cancelled for «%s–%s» (tradeoff = %.0f)'
                    ' × (cost = %.0f):',
                    F[u],
                    F[v],
                    -tradeoff,
                    detoursCost,
                )
                abort = True
        return abort, unionLo, unionHi

    # initialize pq
    for n in _T:
        enqueue_best_union(n)
    # create a global tradeoff variable
    tradeoff = 0

    # BEGIN: main loop
    def loop():
        """Takes a step in the iterative tree building process.
        Return value [bool]: not done."""
        nonlocal i, prevented_crossings, tradeoff
        while True:
            i += 1
            if i > maxiter:
                error('maxiter reached (%d)', i)
                return
            if stale_subtrees:
                debug('<loop> stale_subtrees: ', tuple(F[sr] for sr in stale_subtrees))
            while stale_subtrees:
                enqueue_best_union(stale_subtrees.pop())
            if not pq:
                # finished
                return
            tradeoff = pq[0][0]
            debug('[%d] -tradeoff = %.0f', i, -tradeoff)
            sr_u, (u, v) = pq.top()
            debug('<loop> POPPED «%s–%s», sr_u: <%s>', F[u], F[v], F[sr_u])
            capacity_left = capacity - len(subtree_[u]) - len(subtree_[v])

            if capacity_left < 0:
                warn('@@@@@ Does this ever happen?? @@@@@')
                ban_queued_union(sr_u, u, v)
                yield (u, v), False
                continue

            # BEGIN edge crossing check
            # check if (u, v) crosses an existing edge
            # look for crossing edges within the neighborhood of (u, v)
            # only works if using the expanded delaunay edges
            #  eX = edge_crossings(u, v, G, triangles, triangles_exp)
            eX = edge_crossings(u, v, G, diagonals)
            # Detour edges need to be checked separately
            if not eX and D:
                uC, vC = VertexC[fnT[[u, v]]]
                eXtmp = []
                eXnodes = set()
                nodes2check = set()
                BarrierC = VertexC[fnT[subtree_[u] + subtree_[v]]]
                for s, t in G.edges(range(T, T + D)):
                    skip = False
                    if s < 0 or t < 0:
                        # skip feeders (will be checked later)
                        continue
                    s_, t_ = fnT[[s, t]]
                    Corner = []
                    # below are the 2 cases in which a new edge
                    # will join two subtrees across a detour edge
                    if (
                        s >= T
                        and (s_ == u or s_ == v)
                        and s != DetourHop[subroot_[s]][-1]
                    ):
                        Corner.append(s)
                    if (
                        t >= T
                        and (t_ == u or t_ == v)
                        and t != DetourHop[subroot_[t]][-1]
                    ):
                        Corner.append(t)
                    for corner in Corner:
                        a, b = G[corner]
                        if is_bunch_split_by_corner(
                            BarrierC, *VertexC[fnT[[a, corner, b]]]
                        )[0]:
                            debug(
                                '[%d] «%s–%s» would cross %s–%s–%s',
                                i,
                                F[u],
                                F[v],
                                F[a],
                                F[corner],
                                F[b],
                            )
                            eX.append((a, corner, b))
                            skip = True
                    if skip:
                        continue
                    if is_crossing(uC, vC, *VertexC[fnT[[s, t]]], touch_is_cross=False):
                        eXtmp.append((s, t))
                        if s in eXnodes:
                            nodes2check.add(s)
                        if t in eXnodes:
                            nodes2check.add(t)
                        eXnodes.add(s)
                        eXnodes.add(t)
                for s, t in eXtmp:
                    if s in nodes2check:
                        for w in G[s]:
                            if w != t and not is_same_side(
                                uC, vC, *VertexC[fnT[[w, t]]]
                            ):
                                eX.append((s, t))
                    elif t in nodes2check:
                        for w in G[t]:
                            if w != s and not is_same_side(
                                uC, vC, *VertexC[fnT[[w, s]]]
                            ):
                                eX.append((s, t))
                    else:
                        eX.append((s, t))

            if eX:
                debug(
                    '<edge_crossing> discarding «%s–%s»: would cross %s',
                    F[u],
                    F[v],
                    tuple((F[s], F[t]) for s, t in eX),
                )
                # abort_edge_addition(sr_u, u, v)
                prevented_crossings += 1
                ban_queued_union(sr_u, u, v)
                yield (u, v), None
                continue
            # END edge crossing check

            # BEGIN subroot crossing check
            # check if (u, v) crosses an existing subroot
            sr_v = subroot_[v]
            root = G.nodes[sr_v]['root']
            r2drop = G.nodes[sr_u]['root']
            if root != r2drop:
                debug(
                    f'<distinct_roots>: [{F[u]}] is bound to '
                    f'[{F[r2drop]}], while'
                    f'[{F[v]}] is to {F[root]}.'
                )

            abort, unionLo, unionHi = check_feeder_crossings(u, v, sr_v, sr_u)
            if abort:
                prevented_crossings += 1
                ban_queued_union(sr_u, u, v)
                yield (u, v), None
                continue
            # END subroot crossing check

            # (u, v) edge addition starts here
            subtree = subtree_[v]
            subtree.extend(subtree_[u])
            G.remove_edge(A.nodes[u]['root'], sr_u)
            log.append((i, 'remE', (A.nodes[u]['root'], sr_u)))

            sr_v_entry = pq.tags.get(sr_v)
            if sr_v_entry is not None:
                _, _, _, (_, t) = sr_v_entry
                # print('node', F[t], 'subroot', F[subroot_[t]])
                ComponIn[subroot_[t]].remove(sr_v)
            # TODO: think about why a discard was needed
            ComponIn[sr_v].discard(sr_u)

            # update the component's angle span
            for lo, hi, subtree_span_ in zip(unionLo, unionHi, subtree_span__):
                subtree_span_[sr_v] = lo, hi

            # assign root, subroot and subtree to the newly added nodes
            for n in subtree_[u]:
                A.nodes[n]['root'] = root
                G.nodes[n]['root'] = root
                subroot_[n] = sr_v
                subtree_[n] = subtree
            debug('<loop> NEW EDGE «%s–%s», sr_v <%s>', F[u], F[v], F[sr_v])
            if not pq:
                debug('EMPTY heap')
            #  G.add_edge(u, v, **A.edges[u, v])
            G.add_edge(u, v, length=A[u][v]['length'])
            log.append((i, 'addE', (u, v)))
            # remove from consideration edges internal to subtree_
            A.remove_edge(u, v)

            # finished adding the edge, now check the consequences
            if capacity_left > 0:
                for subroot in list(ComponIn[sr_v]):
                    if len(subtree_[subroot]) > capacity_left:
                        # <this subtree got too big for subtree_[subroot] to join>
                        ComponIn[sr_v].discard(subroot)
                        stale_subtrees.add(subroot)
                # for subroot in ComponIn[sr_u]:
                for subroot in ComponIn[sr_u] - ComponIn[sr_v]:
                    if len(subtree_[subroot]) > capacity_left:
                        stale_subtrees.add(subroot)
                    else:
                        ComponIn[sr_v].add(subroot)
                stale_subtrees.add(sr_v)
            else:
                # max capacity reached: subtree full
                if sr_v in pq.tags:  # required because of i=0 feeders
                    pq.cancel(sr_v)
                commit_subroot(root, sr_v)
                # don't consider connecting to this full subtree anymore
                A.remove_nodes_from(subtree)
                # for subroot in ((ComponIn[sr_u] | ComponIn[sr_v]) - {sr_u,
                # sr_v}):
                for subroot in ComponIn[sr_u] | ComponIn[sr_v]:
                    stale_subtrees.add(subroot)

    # END: main loop

    log = []
    G.graph['log'] = log
    for _ in loop():
        pass

    if Stale:
        debug('Stale nodes (%d): %s', len(Stale), tuple(F[n] for n in Stale))
        old2new = np.arange(T + B, T + B + D)
        mask = np.ones(D, dtype=bool)
        for s in Stale:
            old2new[s - T - B + 1 :] -= 1
            mask[s - T - B] = False
        mapping = dict(zip(range(T + B, T + B + D), old2new))
        for k in Stale:
            mapping.pop(k)
        nx.relabel_nodes(G, mapping, copy=False)
        fnT[T + B : T + B + D - len(Stale)] = fnT[T + B : T + B + D][mask]
        D -= len(Stale)

    debug('FINISHED – Detour nodes added: %d', D)

    if _lggr.isEnabledFor(logging.DEBUG):
        not_marked = []
        for root in roots:
            for proximal in G[root]:
                if proximal not in commited_[root]:
                    not_marked.append(proximal)
        if not_marked:
            debug(
                '@@@@ WARNING: proximals %s were not commited @@@@',
                tuple(F[subroot] for subroot in not_marked),
            )

    # algorithm finished, store some info in the graph object
    G.graph.update(
        creator='OBEW',
        capacity=capacity,
        runtime=time.perf_counter() - start_time,
        d2roots=d2roots,
        method_options=options
        | dict(
            fun_fingerprint=fun_fingerprint(),
        ),
        solver_details=dict(
            iterations=i,
            prevented_crossings=prevented_crossings,
        ),
    )
    if D > 0:
        G.graph['D'] = D
        G.graph['fnT'] = np.concatenate((fnT[: T + B + D], fnT[-R:]))

    return G
