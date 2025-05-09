# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import operator
import time
import logging

import numpy as np
import networkx as nx
from scipy.stats import rankdata

from ..mesh import delaunay
from ..geometric import (angle, apply_edge_exemptions, assign_root,
                         complete_graph, angle_helpers)
from ..utils import NodeTagger
from .priorityqueue import PriorityQueue


lggr = logging.getLogger(__name__)
debug, info, warn, error = lggr.debug, lggr.info, lggr.warning, lggr.error

F = NodeTagger()


def ClassicEW(G_base, capacity=8, delaunay_based=False, maxiter=10000,
              weightfun=None, weight_attr='length'):
    '''Classic Esau-Williams heuristic for C-MST
    inputs:
    G_base: networkx.Graph
    c: capacity
    returns G_cmst: networkx.Graph'''

    start_time = time.perf_counter()
    # grab relevant options to store in the graph later
    options = dict(delaunay_based=delaunay_based)

    R = G_base.graph['R']
    T = G_base.graph['T']
    _T = range(T)
    roots = range(-R, 0)
    VertexC = G_base.graph['VertexC']
    d2roots = G_base.graph.get('d2roots')

    # BEGIN: prepare auxiliary graph with all allowed edges and metrics
    if delaunay_based:
        A = delaunay(G_base, bind2root=True)
        # apply weightfun on all delaunay edges
        if weightfun is not None:
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

    if weightfun is not None:
        options['weightfun'] = weightfun.__name__
        options['weight_attr'] = weight_attr
        for u, v, data in A.edges(data=True):
            data[weight_attr] = weightfun(data)
    # removing root nodes from A to speedup find_option4gate
    # this may be done because G already starts with gates
    A.remove_nodes_from(roots)
    # END: prepare auxiliary graph with all allowed edges and metrics

    # BEGIN: create initial star graph
    G = nx.create_empty_copy(G_base)
    G.add_weighted_edges_from(
        ((n, r, d2roots[n, r]) for n, r in A.nodes(data='root') if n >= 0),
        weight=weight_attr)
    # END: create initial star graph

    # BEGIN: helper data structures

    # mappings from nodes
    # <subtree_>: maps nodes to the list of nodes in their subtree
    subtree_ = [[t] for t in _T]
    # <subroot_>: maps nodes to their gates
    subroot_ = list(_T)

    # mappings from components (identified by their gates)
    # <ComponIn>: maps component to set of components queued to merge in
    ComponIn = [set() for _ in _T]
    # <subtree_span_>: pairs (most_CW, most_CCW) of extreme nodes of each
    #                  subtree, indexed by subroot (former subroot)
    subtree_span_ = [(t, t) for t in _T]

    # mappings from roots
    # <final_sr_>: set of gates of finished components (one set per root)
    final_sr_ = [set() for _ in roots]

    # other structures
    # <pq>: queue prioritized by lowest tradeoff length
    pq = PriorityQueue()
    # find_option4gate()
    # <gates2upd8>: deque for components that need to go through
    gates2upd8 = set()
    # <i>: iteration counter
    i = 0
    # END: helper data structures

    def make_gate_final(root, sr_v):
        final_sr_[root].add(sr_v)
        log.append((i, 'finalG', (sr_v, root)))
        debug('<final> subroot [%s] added', F[sr_v])

    def component_merging_choices(subroot, forbidden=None):
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
                if (subroot_[v] in forbidden or
                        len(subtree_[v]) > capacity_left):
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
            dtype=[('weight', np.float64),
                   ('vd2rootR', np.int_),
                   ('u', np.int_),
                   ('v', np.int_)])
        # result = np.argsort(unordchoices, order=['weight'])
        # unordchoices  = unordchoices[result]

        # DEVIATION FROM Esau-Williams
        # rounding of weight to make ties more likely
        # tie-breaking by proximity of 'v' node to root
        # purpose is to favor radial alignment of components
        tempchoices = unordchoices.copy()
        # tempchoices['weight'] /= tempchoices['weight'].min()
        # tempchoices['weight'] = (20*tempchoices['weight']).round()  # 5%

        result = np.argsort(tempchoices, order=['weight', 'vd2rootR'])
        choices = unordchoices[result]
        return choices

    def find_option4gate(subroot):
        debug('<find_option4gate> starting... subroot = <%s>', F[subroot])
        # () get component expansion edges with weight
        weighted_edges, edges2discard = component_merging_choices(subroot)
        # discard useless edges
        A.remove_edges_from(edges2discard)
        # () sort choices
        choices = sort_union_choices(weighted_edges) if weighted_edges else []
        # () check subroot crossings
        # choice = first_non_crossing(choices, subroot)
        if len(choices) > 0:
            weight, _, u, v = choices[0]
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
            debug('<pushed> sr_u <%s>, «%s–%s», tradeoff = %.3f',
                  F[subroot], F[u], F[v], tradeoff)
        else:
            # no viable edge is better than subroot for this node
            # this becomes a final subroot
            if i:  # run only if not at i = 0
                # definitive gates at iteration 0 do not cross any other edges
                # they are not included in final_sr_ because the algorithm
                # considers the gates extending to infinity (not really)
                root = A.nodes[subroot]['root']
                make_gate_final(root, subroot)
                # check_heap4crossings(root, subroot)
            debug('<cancelling> %s', F[subroot])
            if subroot in pq.tags:
                # i=0 gates and check_heap4crossings reverse_entry
                # may leave accepting gates out of pq
                pq.cancel(subroot)

    def ban_queued_edge(sr_u, u, v):
        if (u, v) in A.edges:
            A.remove_edge(u, v)
        else:
            debug('<<< UNLIKELY <ban_queued_edge()> «%s–%s» not in A >>>',
                  F[u], F[v])
        sr_v = subroot_[v]
        # TODO: think about why a discard was needed
        ComponIn[sr_v].discard(sr_u)
        # gates2upd8.appendleft(sr_u)
        gates2upd8.add(sr_u)
        # find_option4gate(sr_u)

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
                # pq.cancel(sr_u), because find_option4gate will do)
                pq.cancel(sr_v)
                find_option4gate(sr_v)
                is_reverse = True

        if componin != is_reverse:
            # TODO: Why did I expect always False here? It is sometimes True.
            debug('«%s–%s», sr_u <%s>, sr_v <%s> componin: %s, is_reverse: %s',
                  F[u], F[v], F[sr_u], F[sr_v], componin, is_reverse)

        # END: block to be simplified

    # TODO: check if this function is necessary (not used)
    def abort_edge_addition(sr_u, u, v):
        if (u, v) in A.edges:
            A.remove_edge(u, v)
        else:
            print('<<<< UNLIKELY <abort_edge_addition()> '
                  f'({F[u]}, {F[v]}) not in A.edges >>>>')
        ComponIn[subroot_[v]].remove(sr_u)
        find_option4gate(sr_u)

    # initialize pq
    for n in _T:
        find_option4gate(n)

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
        if gates2upd8:
            debug('gates2upd8: %s', tuple(F[subroot] for subroot in gates2upd8))
        while gates2upd8:
            # find_option4gate(gates2upd8.popleft())
            find_option4gate(gates2upd8.pop())
        if not pq:
            # finished
            break
        sr_u, (u, v) = pq.top()
        debug('<popped> «%s–%s», sr_u: <%s>', F[u], F[v], F[sr_u])

        sr_v = subroot_[v]
        root = A.nodes[sr_v]['root']

        capacity_left = capacity - len(subtree_[u]) - len(subtree_[v])

        # assess the union's angle span
        keepLo, keepHi = subtree_span_[sr_v]
        dropLo, dropHi = subtree_span_[sr_u]
        newHi = dropHi if angle(*VertexC[[keepHi, root, dropHi]]) > 0 else keepHi
        newLo = dropLo if angle(*VertexC[[dropLo, root, keepLo]]) > 0 else keepLo
        debug(f'<angle_span> //%s:%s//', F[newLo], F[newHi])


        # edge addition starts here

        # update the component's angle span
        subtree_span_[sr_v] = newLo, newHi

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
        if lggr.isEnabledFor(logging.DEBUG) and pq:
            debug('heap top: <%s>, «%s» %.3f', F[pq[0][-2]],
                  tuple(F[x] for x in pq[0][-1]), pq[0][0])
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
                    gates2upd8.add(subroot)
            for subroot in ComponIn[sr_u] - ComponIn[sr_v]:
                if len(subtree_[subroot]) > capacity_left:
                    gates2upd8.add(subroot)
                else:
                    ComponIn[sr_v].add(subroot)
            gates2upd8.add(sr_v)
        else:
            # max capacity reached: subtree full
            if sr_v in pq.tags:  # if required because of i=0 gates
                pq.cancel(sr_v)
            make_gate_final(root, sr_v)
            # don't consider connecting to this full subtree nodes anymore
            A.remove_nodes_from(subtree)
            for subroot in ComponIn[sr_u] | ComponIn[sr_v]:
                gates2upd8.add(subroot)
    # END: main loop

    if lggr.isEnabledFor(logging.DEBUG):
        not_marked = []
        for root in roots:
            for subroot in G[root]:
                if subroot not in final_sr_[root]:
                    not_marked.append(subroot)
        if not_marked:
            debug('@@@@ WARNING: gates %s were not marked as final @@@@',
                  tuple(F[subroot] for subroot in not_marked))

    # algorithm finished, store some info in the graph object
    G.graph['iterations'] = i
    G.graph['capacity'] = capacity
    G.graph['creator'] = 'ClassicEW'
    G.graph['edges_fun'] = ClassicEW
    G.graph['creation_options'] = options
    G.graph['runtime_unit'] = 's'
    G.graph['runtime'] = time.perf_counter() - start_time
    return G
