# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import time

import networkx as nx
from scipy.stats import rankdata

from ..crossings import edge_crossings
from ..geometric import assign_root
from ..interarraylib import calcload
from ..utils import F
from .priorityqueue import PriorityQueue

__all__ = ()

_lggr = logging.getLogger(__name__)
debug, info, warn, error = _lggr.debug, _lggr.info, _lggr.warning, _lggr.error


def EW_presolver(Aʹ: nx.Graph, capacity: int, maxiter=10000) -> nx.Graph:
    """Modified Esau-Williams heuristic for C-MST with limited crossings

    Args:
        A: available edges graph
        capacity: maximum link capacity
        maxiter: fail-safe to avoid locking in an infinite loop

    Returns:
        Solution topology.
    """

    start_time = time.perf_counter()
    R, T = (Aʹ.graph[k] for k in 'RT')
    _T = range(T)
    diagonals = Aʹ.graph['diagonals']
    d2roots = Aʹ.graph['d2roots']
    S = nx.Graph(R=R, T=T)
    A = Aʹ.copy()

    roots = range(-R, 0)

    assign_root(A)
    d2rootsRank = rankdata(d2roots, method='dense', axis=0)

    # removing root nodes from A to speedup enqueue_best_union
    # this may be done because G already starts with feeders
    A.remove_nodes_from(roots)
    # END: prepare auxiliary graph with all allowed edges and metrics

    # BEGIN: create initial star graph
    S.add_edges_from(((n, r) for n, r in A.nodes(data='root')))
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
    log = []
    # END: helper data structures

    def component_merging_edge(subroot, forbidden=None, margin=1.02):
        # gather all the edges leaving the subtree of subroot
        if forbidden is None:
            forbidden = set()
        forbidden.add(subroot)
        capacity_left = capacity - len(subtree_[subroot])
        choices = []
        sr_d2root = d2roots[subroot, A.nodes[subroot]['root']]
        edges2discard = []
        for u in subtree_[subroot]:
            for v in A[u]:
                if subroot_[v] in forbidden or len(subtree_[v]) > capacity_left:
                    # useless edges
                    edges2discard.append((u, v))
                else:
                    W = A[u][v]['length']
                    if W <= sr_d2root:
                        # useful edges
                        # v's proximity to root is used as tie-breaker
                        choices.append((W, d2rootsRank[v, A.nodes[v]['root']], u, v))
        if not choices:
            return None, 0.0, edges2discard
        choices.sort()
        best_W, best_rank, *best_edge = choices[0]
        for W, rank, *edge in choices[1:]:
            if W > margin * best_W:
                # no more edges within margin
                break
            if rank < best_rank:
                best_W, best_rank, best_edge = W, rank, edge
        tradeoff = best_W - sr_d2root
        return best_edge, tradeoff, edges2discard

    def enqueue_best_union(subroot):
        debug('<enqueue_best_union> starting... subroot = <%s>', F[subroot])
        if edges2ban:
            debug('<<<<<<<edges2ban>>>>>>>>>>> _%d_', len(edges2ban))
        while edges2ban:
            # edge2ban = edges2ban.popleft()
            edge2ban = edges2ban.pop()
            ban_queued_union(*edge2ban)
        # () get component expansion edge with weight
        edge, tradeoff, edges2discard = component_merging_edge(subroot)
        # discard useless edges
        A.remove_edges_from(edges2discard)
        if edge is not None:
            # merging is better than subroot, submit entry to pq
            # tradeoff calculation
            pq.add(tradeoff, subroot, edge)
            ComponIn[subroot_[edge[1]]].add(subroot)
            debug(
                '<pushed> sr_u <%s>, «%s–%s», tradeoff = %.3f',
                F[subroot],
                F[edge[0]],
                F[edge[1]],
                tradeoff,
            )
        else:
            # no viable edge is better than subroot for this node
            debug('<cancelling> %s', F[subroot])
            if subroot in pq.tags:
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

    # initialize pq
    for n in _T:
        enqueue_best_union(n)

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
        # look for crossing edges within the neighborhood of (u, v)
        # this works for expanded delaunay edges (see CPEW for all edges)
        # TODO: Remove the crossing diagonal/delaunay when adding each edge,
        #       but not sure if this would be completely unnecessary.
        #       Can an edge be banned from the queue without knowing its subroot?
        eX = edge_crossings(u, v, S, diagonals)

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

        # edge addition starts here

        subtree = subtree_[v]
        subtree.extend(subtree_[u])
        S.remove_edge(A.nodes[u]['root'], sr_u)
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
        S.add_edge(u, v)
        log.append((i, 'addE', (u, v)))
        # remove from consideration edges internal to subtrees
        A.remove_edge(u, v)
        # TODO: Remove the crossing diagonal/delaunay when adding each edge,

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
            # don't consider connecting to this full subtree nodes anymore
            A.remove_nodes_from(subtree)
            for subroot in ComponIn[sr_u] | ComponIn[sr_v]:
                # enqueue_best_union(subroot)
                # stale_subtrees.append(subroot)
                stale_subtrees.add(subroot)
            # ComponIn[sr_u] = None
            # ComponIn[sr_v] = None
    # END: main loop

    calcload(S)
    # algorithm finished, store some info in the graph object
    S.graph.update(
        runtime=time.perf_counter() - start_time,
        capacity=capacity,
        creator='EW_presolver',
        solver_details=dict(
            log=log,
            iterations=i,
            prevented_crossings=prevented_crossings,
        ),
    )
    return S
