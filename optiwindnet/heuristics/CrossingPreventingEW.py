# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import time

import networkx as nx
import numpy as np
from scipy.stats import rankdata

from ..crossings import edge_crossings
from ..geometric import (
    angle_helpers,
    angle_oracles_factory,
    apply_edge_exemptions,
    assign_root,
    complete_graph,
    is_crossing,
    is_same_side,
)
from ..mesh import delaunay
from ..utils import F
from .priorityqueue import PriorityQueue

__all__ = ()

_lggr = logging.getLogger(__name__)
debug, info, warn, error = _lggr.debug, _lggr.info, _lggr.warning, _lggr.error


def CPEW(
    G_base,
    capacity=8,
    delaunay_based=True,
    maxiter=10000,
    weightfun=None,
    weight_attr='length',
):
    """Crossing Preventing Esau-Williams heuristic for C-MST

    Args:
      G_base: networkx.Graph
      c: capacity
    Returns:
      G_cmst: networkx.Graph
    """

    start_time = time.perf_counter()
    # grab relevant options to store in the graph later
    options = dict(delaunay_based=delaunay_based)

    R = G_base.graph['R']
    T = G_base.graph['T']
    _T = range(T)
    roots = range(-R, 0)
    VertexC = G_base.graph['VertexC']

    # BEGIN: prepare auxiliary graph with all allowed edges and metrics
    if delaunay_based:
        A = delaunay(G_base, bind2root=True)
        diagonals = A.graph['diagonals']
        # apply weightfun on all delaunay edges
        if weightfun is not None:
            # TODO: fix `apply_edge_exemptions()` for the
            #       `delaunay()` without triangles
            apply_edge_exemptions(A)
        # TODO: decide whether to keep this 'else' (to get edge arcs)
        # else:
        # apply_edge_exemptions(A)
    else:
        A = complete_graph(G_base)

    assign_root(A)
    d2roots = A.graph['d2roots']
    d2rootsRank = rankdata(d2roots, method='dense', axis=0)
    angles, anglesRank = angle_helpers(G_base)
    union_limits, angle_ccw = angle_oracles_factory(angles, anglesRank)

    if weightfun is not None:
        options['weightfun'] = weightfun.__name__
        options['weight_attr'] = weight_attr
        for u, v, data in A.edges(data=True):
            data[weight_attr] = weightfun(data)
    # removing root nodes from A to speedup enqueue_best_union
    # this may be done because G already starts with feeders
    A.remove_nodes_from(roots)
    # END: prepare auxiliary graph with all allowed edges and metrics

    # BEGIN: create initial star graph
    G = nx.create_empty_copy(G_base)
    G.add_weighted_edges_from(
        ((n, r, d2roots[n, r]) for n, r in A.nodes(data='root') if n >= 0),
        weight=weight_attr,
    )
    # END: create initial star graph

    # BEGIN: helper data structures

    # mappings from nodes
    # <subtree_>: maps nodes to the list of nodes in their subtree
    subtree_ = [[t] for t in _T]
    # <subroot_>: maps terminals to their subroots
    subroot_ = list(_T)

    # mappings from components (identified by their subroots)
    # <ComponIn>: maps component to set of components queued to merge in
    ComponIn = [set() for _ in _T]
    # <subtree_span_>: pairs (most_CW, most_CCW) of extreme nodes of each
    #                  subtree, indexed by subroot (former subroot)
    subtree_span_ = [(t, t) for t in _T]

    # mappings from roots
    # <commited_>: set of subroots of finished components (one set per root)
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
    # TODO: edges{T,C,V} could be used to vectorize the edge crossing detection
    # <edgesN>: array of nodes of the edges of G (T×2)
    # <edgesC>: array of node coordinates for edges of G (T×2×2)
    # <edgesV>: array of vectors of the edges of G (T×2)
    # <i>: iteration counter
    i = 0
    # <prevented_crossing>: counter for edges discarded due to crossings
    prevented_crossings = 0
    # END: helper data structures

    def is_crossing_feeder(root, subroot, u, v, touch_is_cross=False):
        less = np.less_equal if touch_is_cross else np.less
        uvA = angles[v, root] - angles[u, root]
        swaped = (-np.pi < uvA) & (uvA < 0.0) | (np.pi < uvA)
        lo, hi = (v, u) if swaped else (u, v)
        loR, hiR, srR = anglesRank[(lo, hi, subroot), root]
        W = loR > hiR  # wraps +-pi
        supL = less(loR, srR)  # angle(low) <= angle(probe)
        infH = less(srR, hiR)  # angle(probe) <= angle(high)
        if ~W & supL & infH | W & ~supL & infH | W & supL & ~infH:
            if not is_same_side(*VertexC[[u, v, root, subroot]]):
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
        commited_[root].add(sr_v)
        log.append((i, 'finalG', (sr_v, root)))
        debug('<final> subroot [%s] added', F[sr_v])

    def get_union_choices(subroot, forbidden=None):
        # gather all the edges leaving the subtree of subroot
        if forbidden is None:
            forbidden = set()
        forbidden.add(subroot)
        d2root = d2roots[subroot, A.nodes[subroot]['root']]
        capacity_left = capacity - len(subtree_[subroot])
        weighted_edges = []
        edges2discard = []
        for u in subtree_[subroot]:
            for v in A[u]:
                if subroot_[v] in forbidden or len(subtree_[v]) > capacity_left:
                    # useless edges
                    edges2discard.append((u, v))
                else:
                    W = A[u][v][weight_attr]
                    # if W <= d2root:  # TODO: what if I use <= instead of <?
                    if W < d2root:
                        # useful edges
                        tiebreaker = d2rootsRank[v, A[u][v]['root']]
                        weighted_edges.append((W, tiebreaker, u, v))
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

    def first_non_crossing(choices, subroot):
        """go through choices and return the first that does not cross a final
        subroot"""
        # TODO: remove subroot from the parameters
        nonlocal prevented_crossings
        found = False
        # BEGIN: for loop that picks an edge
        for weight, tiebreaker, u, v in choices:
            found = True
            root = A[u][v]['root']

            # check if a subroot is crossing the edge (u, v)
            # TODO: this only looks at the feeders connecting to the edges'
            # closest root , is it relevant to look at all roots?
            # PendingG = set()

            for commited in commited_[root]:
                # TODO: test a subroot exactly overlapping with a node
                # Elaborating: angleRank will take care of this corner case.
                # the subroot will fall within one of the edges around the node
                if is_crossing_feeder(root, commited, u, v):
                    # crossing subroot, discard edge
                    prevented_crossings += 1
                    # TODO: call ban_queued_union (problem: these edges are not
                    # queued)
                    if (u, v) in A.edges:
                        A.remove_edge(u, v)
                    else:
                        debug(
                            '<<< UNLIKELY.A first_non_crossing(): (%s, %s)not in A >>>',
                            F[u],
                            F[v],
                        )
                    if subroot_[v] in ComponIn[subroot]:
                        # this means the target component was in line to
                        # connect to the current component
                        debug(
                            '<<< UNLIKELY.B first_non_crossing(): subroot_'
                            '[%s] in ComponIn[%s] >>>',
                            F[v],
                            F[subroot],
                        )
                        _, _, _, (s, t) = pq.tags.get(subroot_[v])
                        if t == u:
                            ComponIn[subroot].remove(subroot_[v])
                            stale_subtrees.add(subroot_[v])

                    found = False
                    break
            # for pending in PendingG:
            #  print(f'<pending> processing '
            #        f'pending [{F[pending]}]')
            # enqueue_best_union(pending)
            if found:
                break
        # END: for loop that picks an edge
        return (weight, u, v) if found else ()

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
        # () check subroot crossings
        choice = first_non_crossing(choices, subroot)
        if choice:
            # merging is better than subroot, submit entry to pq
            weight, u, v = choice
            # tradeoff calculation
            tradeoff = weight - d2roots[subroot, A.nodes[subroot]['root']]
            pq.add(tradeoff, subroot, (u, v))
            ComponIn[subroot_[v]].add(subroot)
            debug(
                '<pushed> sr_u <%s>, «%s–%s», tradeoff = %.3f',
                F[subroot],
                F[u],
                F[v],
                tradeoff,
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
                check_heap4crossings(root, subroot)
            debug('<cancelling> %s', F[subroot])
            if subroot in pq.tags:
                # i=0 feeders and check_heap4crossings reverse_entry
                # may leave accepting subtrees out of pq
                pq.cancel(subroot)

    def check_heap4crossings(root, commited):
        """search the heap for edges that cross the subroot 'commited'.
        calls enqueue_best_union for each of the subtrees involved"""
        for tradeoff, _, sr_u, uv in pq:
            # if uv is None or uv not in A.edges:
            if uv is None:
                continue
            u, v = uv
            if is_crossing_feeder(root, commited, u, v):
                nonlocal prevented_crossings
                # crossing subroot, discard edge
                prevented_crossings += 1
                ban_queued_union(sr_u, u, v)

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
                '<<<< UNLIKELY <abort_edge_addition()> '
                f'({F[u]}, {F[v]}) not in A.edges >>>>'
            )
        ComponIn[subroot_[v]].remove(sr_u)
        enqueue_best_union(sr_u)

    # initialize pq
    for n in _T:
        enqueue_best_union(n)

    log = []
    G.graph['log'] = log
    loop = True
    # BEGIN: main loop
    while loop:
        i += 1
        if i > maxiter:
            error('maxiter reached (%d)', i)
            break
        debug('[%d]', i)
        # debug(f'[{i}] bj–bm root: {A.edges[(F.bj, F.bm)]["root"]}')
        if stale_subtrees:
            debug('stale_subtrees: %s', tuple(F[subroot] for subroot in stale_subtrees))
        while stale_subtrees:
            # enqueue_best_union(stale_subtrees.popleft())
            enqueue_best_union(stale_subtrees.pop())
        if not pq:
            # finished
            break
        sr_u, (u, v) = pq.top()
        debug('<popped> «%s–%s», sr_u: <%s>', F[u], F[v], F[sr_u])

        # TODO: main loop should do only
        # - pop from pq
        # - check if adding edge would block some component
        # - add edge
        # - call enqueue_best_union for everyone affected

        # check if (u, v) crosses an existing edge
        if delaunay_based:
            # look for crossing edges within the neighborhood of (u, v)
            # faster, but only works if using the expanded delaunay edges
            eX = edge_crossings(u, v, G, diagonals)
        else:
            # when using the edges of a complete graph
            # alternate way - slower
            eX = []
            eXtmp = []
            eXnodes = set()
            nodes2check = set()
            uC, vC = VertexC[[u, v]]
            for s, t in G.edges:
                if s == u or t == u or s == v or t == v or s < 0 or t < 0:
                    # skip if the edges have a common node or (s, t) is a subroot
                    continue
                if is_crossing(uC, vC, *VertexC[[s, t]], touch_is_cross=True):
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
                        if w != t and not is_same_side(uC, vC, *VertexC[[w, t]]):
                            eX.append((s, t))
                elif t in nodes2check:
                    for w in G[t]:
                        if w != s and not is_same_side(uC, vC, *VertexC[[w, s]]):
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
            continue

        sr_v = subroot_[v]
        root = A.nodes[sr_v]['root']

        capacity_left = capacity - len(subtree_[u]) - len(subtree_[v])

        # assess the union's angle span
        keepLo, keepHi = subtree_span_[sr_v]
        dropLo, dropHi = subtree_span_[sr_u]
        unionLo, unionHi = union_limits(root, u, dropLo, dropHi, v, keepLo, keepHi)
        debug('<angle_span> //%s:%s//', F[unionLo], F[unionHi])

        # check which feeders are within the union's angle span
        lR, hR = anglesRank[(unionLo, unionHi), root]
        anglesWrap = lR > hR
        abort = False
        # the more conservative check would be using sr_v instead of
        # sr_u in the line below (but then the filter needs changing)
        distanceThreshold = d2rootsRank[sr_u, root]
        for subroot in [g for g in G[root] if d2rootsRank[g, root] > distanceThreshold]:
            sr_rank = anglesRank[subroot, root]
            if (
                not anglesWrap
                and (lR < sr_rank < hR)
                or (anglesWrap and (sr_rank > lR or sr_rank < hR))
            ):
                # possible occlusion of subtree[subroot] by union subtree
                debug(
                    '<check_occlusion> «%s-%s» might cross subroot <%s>',
                    F[u],
                    F[v],
                    F[subroot],
                )
                if subroot in commited_[root]:
                    if is_crossing_feeder(root, subroot, u, v, touch_is_cross=True):
                        abort = True
                        break
                elif subroot in ComponIn[sr_u] or subroot in ComponIn[sr_v]:
                    if len(subtree_[subroot]) > capacity_left:
                        # check crossing with subroot
                        if is_crossing_feeder(root, subroot, u, v, touch_is_cross=True):
                            # find_option for subroot, but forbidding sr_u, sr_v
                            abort = True
                            break
                    else:
                        debug(
                            '$$$ UNLIKELY: subroot <%s> could merge with '
                            'subtree <%s> $$$',
                            F[subroot],
                            F[sr_v],
                        )
                else:
                    # check crossing with next union for subroot
                    entry = pq.tags.get(subroot)
                    if entry is not None:
                        _, _, _, (s, t) = entry
                        if is_crossing(*VertexC[[u, v, s, t]], touch_is_cross=False):
                            abort = True
                            break

        if abort:
            debug('### «%s-%s» would block subroot %s ###', F[u], F[v], F[subroot])
            prevented_crossings += 1
            ban_queued_union(sr_u, u, v)
            continue

        # edge addition starts here

        # update the component's angle span
        subtree_span_[sr_v] = unionLo, unionHi

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

        # assign root, subroot and subtree to the newly added nodes
        for n in subtree_[u]:
            A.nodes[n]['root'] = root
            subroot_[n] = sr_v
            subtree_[n] = subtree
        debug('<add edge> «%s–%s» subroot <%s>', F[u], F[v], F[sr_v])
        if _lggr.isEnabledFor(logging.DEBUG) and pq:
            debug(
                'heap top: <%s>, «%s» %.3f',
                F[pq[0][-2]],
                tuple(F[x] for x in pq[0][-1]),
                pq[0][0],
            )
        else:
            debug('heap EMPTY')
        #  G.add_edge(u, v, **A.edges[u, v])
        G.add_edge(u, v, **{weight_attr: A[u][v][weight_attr]})
        log.append((i, 'addE', (u, v)))
        # remove from consideration edges internal to subtrees
        A.remove_edge(u, v)

        # finished adding the edge, now check the consequences
        if capacity_left > 0:
            for subroot in list(ComponIn[sr_v]):
                if len(subtree_[subroot]) > capacity_left:
                    # TODO: think about why a discard was needed
                    # ComponIn[sr_v].remove(subroot)
                    ComponIn[sr_v].discard(subroot)
                    # enqueue_best_union(subroot)
                    # stale_subtrees.append(subroot)
                    stale_subtrees.add(subroot)
            for subroot in ComponIn[sr_u] - ComponIn[sr_v]:
                if len(subtree_[subroot]) > capacity_left:
                    # enqueue_best_union(subroot)
                    # stale_subtrees.append(subroot)
                    stale_subtrees.add(subroot)
                else:
                    ComponIn[sr_v].add(subroot)
            # ComponIn[sr_u] = None
            # enqueue_best_union(sr_v)
            # stale_subtrees.append(sr_v)
            stale_subtrees.add(sr_v)
        else:
            # max capacity reached: subtree full
            if sr_v in pq.tags:  # if required because of i=0 feeders
                pq.cancel(sr_v)
            commit_subroot(root, sr_v)
            # don't consider connecting to this full subtree nodes anymore
            A.remove_nodes_from(subtree)
            for subroot in ComponIn[sr_u] | ComponIn[sr_v]:
                # enqueue_best_union(subroot)
                # stale_subtrees.append(subroot)
                stale_subtrees.add(subroot)
            # ComponIn[sr_u] = None
            # ComponIn[sr_v] = None
            check_heap4crossings(root, sr_v)
    # END: main loop

    if _lggr.isEnabledFor(logging.DEBUG):
        not_marked = []
        for root in roots:
            for subroot in G[root]:
                if subroot not in commited_[root]:
                    not_marked.append(subroot)
        if not_marked:
            debug(
                '@@@@ WARNING: subroots %s were not commited @@@@',
                tuple(F[subroot] for subroot in not_marked),
            )

    # algorithm finished, store some info in the graph object
    G.graph['iterations'] = i
    G.graph['prevented_crossings'] = prevented_crossings
    G.graph['capacity'] = capacity
    G.graph['creator'] = 'CPEW'
    G.graph['edges_fun'] = CPEW
    G.graph['creation_options'] = options
    G.graph['runtime_unit'] = 's'
    G.graph['runtime'] = time.perf_counter() - start_time
    return G
