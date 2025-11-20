# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import time
from typing import Callable

import networkx as nx
from scipy.stats import rankdata

from ..geometric import apply_edge_exemptions, assign_root, complete_graph
from ..mesh import delaunay
from .priorityqueue import PriorityQueue

__all__ = ()

_lggr = logging.getLogger(__name__)
debug, info, warn, error = _lggr.debug, _lggr.info, _lggr.warning, _lggr.error


def ClassicEW(
    L: nx.Graph,
    capacity: int,
    delaunay_based: bool = False,
    maxiter: int = 10000,
    weightfun: Callable | None = None,
    weight_attr: str = 'length',
) -> nx.Graph:
    """Classic Esau-Williams heuristic for C-MST.

    Args:
      L: location graph
      capacity: max number of terminals in a subtree
      maxiter: fail-safe to avoid locking in an infinite loop
    Returns:
      Routeset graph G
    """

    start_time = time.perf_counter()
    # grab relevant options to store in the graph later
    options = dict(delaunay_based=delaunay_based)

    R = L.graph['R']
    T = L.graph['T']
    _T = range(T)
    roots = range(-R, 0)

    # BEGIN: prepare auxiliary graph with all allowed edges and metrics
    if delaunay_based:
        A = delaunay(L, bind2root=True)
        # apply weightfun on all delaunay edges
        if weightfun is not None:
            apply_edge_exemptions(A)
        # TODO: decide whether to keep this 'else' (to get edge arcs)
        # else:
        # apply_edge_exemptions(A)
    else:
        A = complete_graph(L)

    assign_root(A)
    d2roots = A.graph['d2roots']
    d2rootsRank = rankdata(d2roots, method='dense', axis=0)

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
    G = nx.create_empty_copy(L)
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

    # mappings from roots
    # <commited_>: set of subroots of finished components (one set per root)
    commited_ = [set() for _ in roots]

    # other structures
    # <pq>: queue prioritized by lowest tradeoff length
    pq = PriorityQueue()
    # enqueue_best_union()
    # <stale_subtrees>: deque for components that need to go through
    stale_subtrees = set()
    # <i>: iteration counter
    i = 0
    # END: helper data structures

    def commit_subroot(root, sr_v):
        commited_[root].add(sr_v)
        log.append((i, 'finalG', (sr_v, root)))
        debug('<final> subroot [%d] added', sr_v)

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

    def enqueue_best_union(subroot):
        debug('<enqueue_best_union> starting... subroot = <%d>', subroot)
        # () get component expansion edges with weight
        weighted_edges, edges2discard = get_union_choices(subroot)
        # discard useless edges
        A.remove_edges_from(edges2discard)
        if weighted_edges:
            # () sort choices
            weight, _, u, v = min(weighted_edges)
            choice = (weight, u, v)
        else:
            choice = False
        if choice:
            # merging is better than subroot, submit entry to pq
            weight, u, v = choice
            # tradeoff calculation
            tradeoff = weight - d2roots[subroot, A.nodes[subroot]['root']]
            pq.add(tradeoff, subroot, (u, v))
            ComponIn[subroot_[v]].add(subroot)
            debug(
                '<pushed> sr_u <%d>, «%d~%d», tradeoff = %.3f', subroot, u, v, tradeoff
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
            debug('<cancelling> %d', subroot)
            if subroot in pq.tags:
                # i=0 feeders and check_heap4crossings reverse_entry
                # may leave accepting subtrees out of pq
                pq.cancel(subroot)

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
        if stale_subtrees:
            debug('stale_subtrees: %s', stale_subtrees)
        while stale_subtrees:
            # enqueue_best_union(stale_subtrees.popleft())
            enqueue_best_union(stale_subtrees.pop())
        if not pq:
            # finished
            break
        sr_u, (u, v) = pq.top()
        debug('<popped> «%d~%d», sr_u: <%d>', u, v, sr_u)

        sr_v = subroot_[v]
        root = A.nodes[sr_v]['root']

        capacity_left = capacity - len(subtree_[u]) - len(subtree_[v])

        # edge addition starts here

        subtree = subtree_[v]
        subtree.extend(subtree_[u])
        G.remove_edge(A.nodes[u]['root'], sr_u)
        log.append((i, 'remE', (A.nodes[u]['root'], sr_u)))

        sr_v_entry = pq.tags.get(sr_v)
        if sr_v_entry is not None:
            _, _, _, (_, t) = sr_v_entry
            ComponIn[subroot_[t]].remove(sr_v)
        # TODO: think about why a discard was needed
        ComponIn[sr_v].discard(sr_u)

        # assign root, subroot and subtree to the newly added nodes
        for n in subtree_[u]:
            A.nodes[n]['root'] = root
            subroot_[n] = sr_v
            subtree_[n] = subtree
        debug('<add edge> «%d~%d» subroot <%d>', u, v, sr_v)
        if _lggr.isEnabledFor(logging.DEBUG) and pq:
            debug('heap top: <%d>, «%s» %.3f', pq[0][-2], pq[0][-1], pq[0][0])
        else:
            debug('heap EMPTY')
        G.add_edge(u, v, **{weight_attr: A[u][v][weight_attr]})
        log.append((i, 'addE', (u, v)))
        # remove from consideration edges internal to subtrees
        A.remove_edge(u, v)

        # finished adding the edge, now check the consequences
        if capacity_left > 0:
            for subroot in list(ComponIn[sr_v]):
                if len(subtree_[subroot]) > capacity_left:
                    ComponIn[sr_v].discard(subroot)
                    stale_subtrees.add(subroot)
            for subroot in ComponIn[sr_u] - ComponIn[sr_v]:
                if len(subtree_[subroot]) > capacity_left:
                    stale_subtrees.add(subroot)
                else:
                    ComponIn[sr_v].add(subroot)
            stale_subtrees.add(sr_v)
        else:
            # max capacity reached: subtree full
            if sr_v in pq.tags:  # if required because of i=0 feeders
                pq.cancel(sr_v)
            commit_subroot(root, sr_v)
            # don't consider connecting to this full subtree nodes anymore
            A.remove_nodes_from(subtree)
            for subroot in ComponIn[sr_u] | ComponIn[sr_v]:
                stale_subtrees.add(subroot)
    # END: main loop

    if _lggr.isEnabledFor(logging.DEBUG):
        not_marked = []
        for root in roots:
            for subroot in G[root]:
                if subroot not in commited_[root]:
                    not_marked.append(subroot)
        if not_marked:
            debug('@@@@ WARNING: subroots %s were not commited @@@@', not_marked)

    # algorithm finished, store some info in the graph object
    G.graph['iterations'] = i
    G.graph['capacity'] = capacity
    G.graph['creator'] = 'ClassicEW'
    G.graph['edges_fun'] = ClassicEW
    G.graph['creation_options'] = options
    G.graph['runtime_unit'] = 's'
    G.graph['runtime'] = time.perf_counter() - start_time
    return G
