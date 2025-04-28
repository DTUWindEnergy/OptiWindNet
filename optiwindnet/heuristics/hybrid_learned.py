# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import time
import logging
import math

import numpy as np
import networkx as nx

from ..geometric import angle, assign_root
from ..crossings import edge_crossings
from ..utils import NodeTagger
from .priorityqueue import PriorityQueue
from ..interarraylib import as_normalized, calcload

F = NodeTagger()

lggr = logging.getLogger(__name__)
debug, info, warn, error = lggr.debug, lggr.info, lggr.warning, lggr.error


class Appraiser():

    def __init__(self, A: nx.Graph, capacity: int):
        # problem features
        self.capacity = capacity
        self.T = A.graph['T']
        # scale is the median of distances of terminals to their nearest root
        d2roots = A.graph['d2roots']
        closest_root = np.argmin(d2roots, axis=1)
        self.scale = np.median(d2roots[:, closest_root])
        print(self.scale, d2roots.min(), d2roots.max())

    def appraise(
        self, 
        num_edge_alt,
        num_subtrees,
        # edge features
        u_is_gate,
        v_is_gate,
        load,
        target_load,
        extent,
        gate_d2root,
        target_d2root,
        cos,
    ):
        pass
        

def hybrid_learned(Aʹ: nx.Graph, capacity: int, maxiter=10000) -> nx.Graph:
    '''Hybrid machine-learning and Esau-Williams heuristic for C-MST
    
    Args:
        A: available edges graph
        capacity: maximum link capacity
        maxiter: fail-safe to avoid locking in an infinite loop

    Returns:
        Solution topology.
    '''

    start_time = time.perf_counter()
    R, T = Aʹ.graph['R'], Aʹ.graph['T']
    diagonals = Aʹ.graph['diagonals']
    d2roots = Aʹ.graph['d2roots']
    S = nx.Graph(R=R, T=T)
    appraiser = Appraiser(Aʹ, capacity)
    A = as_normalized(Aʹ, scale=appraiser.scale)

    roots = range(-R, 0)
    VertexC = A.graph['VertexC']

    assign_root(A)

    # removing root nodes from A to speedup find_option4gate
    # this may be done because G already starts with gates
    A.remove_nodes_from(roots)
    # END: prepare auxiliary graph with all allowed edges and metrics

    # BEGIN: component accounting
    num_components = T
    min_components = math.ceil(T/capacity)
    # <is_subroot_>: flags if terminal should be linked to a root
    is_subroot_ = np.full((T,), True, dtype=bool)
    # END: component accounting

    # BEGIN: helper data structures

    # mappings from nodes
    # <subtrees>: maps nodes to the list of nodes in their subtree
    subtree_ = [[t] for t in range(T)]
    # <feeder_of>: maps terminals to their feeders
    feeder_of = [t for t in range(T)]

    # mappings from components (identified by their gates)
    # <ComponIn>: maps component to set of components queued to merge in
    ComponIn = [set() for _ in range(T)]
    ComponLoLim = np.arange(T)  # most CW node
    ComponHiLim = np.arange(T)  # most CCW node

    # other structures
    # <pq>: queue prioritized by lowest tradeoff length
    pq = PriorityQueue()
    # find_option4gate()
    # <gates2upd8>: deque for components that need to go through
    # gates2upd8 = deque()
    gates2upd8 = set()
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

    def component_merging_edge(gate, forbidden=None):
        # gather all the edges leaving the subtree of gate
        if forbidden is None:
            forbidden = set()
        forbidden.add(gate)
        choices = []
        gate_d2root = d2roots[gate, A.nodes[gate]['root']]
        load = len(subtree_[gate])
        load_left = capacity - load
        edges2discard = []
        edges2examine = []
        subtree_choices = set()
        for u in subtree_[gate]:
            u_is_gate=(u == gate),
            for v in A[u]:
                target_load = len(subtree_[v])
                target_gate = feeder_of[v]
                if (target_gate in forbidden or target_load > load_left):
                    edges2discard.append((u, v))
                else:
                    subtree_choices.add(target_gate)
                    edges2examine.append((u, v, u_is_gate, target_load,
                                          target_gate))
        # discard useless edges
        A.remove_edges_from(edges2discard)
        #  rel_component_excess = num_components/min_components - 1
        num_edge_alt = len(edges2examine)
        num_subtrees = len(subtree_choices)
        for u, v, u_is_gate, target_load, target_gate in edges2examine:
            target_root = A.nodes[target_gate]['root']
            extent = A[u][v]['length']
            vec_uv = VertexC[v] - VertexC[u]
            vec_ur = VertexC[root] - VertexC[u]
            cos_uv_ur = (np.dot(vec_uv, vec_ur)
                         /np.hypot(*vec_uv)/np.hypot(*vec_ur))
            choices.append((appraiser.appraise(
                # subtree-wide features
                num_edge_alt=num_edge_alt,
                num_subtrees=num_subtrees,
                # edge features
                u_is_gate=u_is_gate,
                v_is_gate=(v == target_gate),
                load=load,
                target_load=target_load,
                extent=extent,
                gate_d2root=gate_d2root,
                target_d2root=d2roots[target_gate, target_root],
                cos=cos_uv_ur,
                ), u, v)
            )
        if not choices:
            return None, 0., edges2discard
        choices.sort()
        tradeoff, *edge = choices[0]
        return edge, tradeoff

    def find_option4gate(gate):
        debug('<find_option4gate> starting... gate = <%s>', F[gate])
        if edges2ban:
            debug('<<<<<<<edges2ban>>>>>>>>>>> _%d_', len(edges2ban))
        while edges2ban:
            # edge2ban = edges2ban.popleft()
            edge2ban = edges2ban.pop()
            ban_queued_edge(*edge2ban)
        # () get component expansion edge with weight
        edge, tradeoff = component_merging_edge(gate)
        if edge is not None:
            pq.add(tradeoff, gate, edge)
            ComponIn[feeder_of[edge[1]]].add(gate)
            debug('<pushed> g2drop <%s>, «%s–%s», tradeoff = %.3f',
                  F[gate], F[edge[0]], F[edge[1]], tradeoff)
        else:
            # no viable edge is better than gate for this node
            debug('<cancelling> %s', F[gate])
            if gate in pq.tags:
                pq.cancel(gate)

    def ban_queued_edge(g2drop, u, v):
        if (u, v) in A.edges:
            A.remove_edge(u, v)
        else:
            warn('<< UNLIKELY <ban_queued_edge()> «%s–%s» not in A.edges >>',
                  F[u], F[v])
        g2keep = feeder_of[v]
        # TODO: think about why a discard was needed
        ComponIn[g2keep].discard(g2drop)
        # gates2upd8.appendleft(g2drop)
        gates2upd8.add(g2drop)
        # find_option4gate(g2drop)

        # BEGIN: block to be simplified
        is_reverse = False
        componin = g2keep in ComponIn[g2drop]
        reverse_entry = pq.tags.get(g2keep)
        if reverse_entry is not None:
            _, _, _, (s, t) = reverse_entry
            if (t, s) == (u, v):
                # TODO: think about why a discard was needed
                ComponIn[g2drop].discard(g2keep)
                # this is assymetric on purpose (i.e. not calling
                # pq.cancel(g2drop), because find_option4gate will do)
                pq.cancel(g2keep)
                find_option4gate(g2keep)
                is_reverse = True

        # if this if is not visited, replace the above with ComponIn check
        # this means that if g2keep is to also merge with g2drop, then the
        # edge of the merging must be (v, u)
        if componin != is_reverse:
            print(f'«{F[u]}–{F[v]}», '
                  f'g2drop <{F[g2drop]}>, g2keep <{F[g2keep]}> '
                  f'componin: {componin}, is_reverse: {is_reverse}')

        # END: block to be simplified

    # initialize pq
    for t in range(T):
        find_option4gate(t)

    loop = True
    # BEGIN: main loop
    while loop:
        i += 1
        if i > maxiter:
            error('ERROR: maxiter reached (%d)', i)
            break
        debug('[%d]', i)
        # debug(f'[{i}] bj–bm root: {A.edges[(F.bj, F.bm)]["root"]}')
        if gates2upd8:
            debug('gates2upd8: %s', tuple(F[gate] for gate in gates2upd8))
        for gate2upd8 in gates2upd8:
            # find_option4gate(gates2upd8.popleft())
            find_option4gate(gate2upd8)
        gates2upd8.clear()
        if not pq:
            # finished
            break
        g2drop, (u, v) = pq.top()
        debug('<popped> «%s–%s», g2drop: <%s>', F[u], F[v], F[g2drop])

        # TODO: main loop should do only
        # - pop from pq
        # - check if adding edge would block some component
        # - add edge
        # - call find_option4gate for everyone affected

        # check if (u, v) crosses an existing edge
        # look for crossing edges within the neighborhood of (u, v)
        # this works for expanded delaunay edges (see CPEW for all edges)
        eX = edge_crossings(u, v, S, diagonals)

        if eX:
            debug('<edge_crossing> discarding «%s–%s»: would cross %s',
                  F[u], F[v], tuple((F[s], F[t]) for s, t in eX))
            # abort_edge_addition(g2drop, u, v)
            prevented_crossings += 1
            ban_queued_edge(g2drop, u, v)
            continue

        g2keep = feeder_of[v]
        root = A.nodes[g2keep]['root']
        subtree = subtree_[v]

        # assess the union's angle span
        keepLo, keepHi = ComponLoLim[g2keep], ComponHiLim[g2keep]
        dropLo, dropHi = ComponLoLim[g2drop], ComponHiLim[g2drop]
        newHi = dropHi if angle(*VertexC[[keepHi, root, dropHi]]) > 0 else keepHi
        newLo = dropLo if angle(*VertexC[[dropLo, root, keepLo]]) > 0 else keepLo
        debug(f'<angle_span> //%s:%s//', F[newLo], F[newHi])

        # edge addition starts here
        ComponLoLim[g2keep], ComponHiLim[g2keep] = newLo, newHi
        subtree.extend(subtree_[u])
        S.remove_edge(A.nodes[u]['root'], g2drop)
        log.append((i, 'remE', (A.nodes[u]['root'], g2drop)))

        g2keep_entry = pq.tags.get(g2keep)
        if g2keep_entry is not None:
            _, _, _, (_, t) = g2keep_entry
            # print('node', F[t], 'gate', F[feeder_of[t]])
            ComponIn[feeder_of[t]].remove(g2keep)
        # TODO: think about why a discard was needed
        ComponIn[g2keep].discard(g2drop)

        # assign root, gate and subtree to the newly added terminals
        for t in subtree_[u]:
            A.nodes[t]['root'] = root
            feeder_of[t] = g2keep
            subtree_[t] = subtree
        debug('<add edge> «%s–%s» gate <%s>', F[u], F[v], F[g2keep])
        if pq:
            debug('heap top: <%s>, «%s» %.3f', F[pq[0][-2]],
                  tuple(F[x] for x in pq[0][-1]), pq[0][0])
        else:
            debug('heap EMPTY')
        #  G.add_edge(u, v, **A.edges[u, v])
        S.add_edge(u, v)
        is_subroot_[g2drop] = False
        num_components -= 1
        log.append((i, 'addE', (u, v)))
        # remove from consideration edges internal to subtrees
        A.remove_edge(u, v)

        # finished adding the edge, now check the consequences
        capacity_left = capacity - len(subtree_[u]) - len(subtree_[v])
        if capacity_left > 0:
            for gate in list(ComponIn[g2keep]):
                if len(subtree_[gate]) > capacity_left:
                    # TODO: think about why a discard was needed
                    # ComponIn[g2keep].remove(gate)
                    ComponIn[g2keep].discard(gate)
                    # find_option4gate(gate)
                    # gates2upd8.append(gate)
                    gates2upd8.add(gate)
            for gate in ComponIn[g2drop] - ComponIn[g2keep]:
                if len(subtree_[gate]) > capacity_left:
                    # find_option4gate(gate)
                    # gates2upd8.append(gate)
                    gates2upd8.add(gate)
                else:
                    ComponIn[g2keep].add(gate)
            # ComponIn[g2drop] = None
            # find_option4gate(g2keep)
            # gates2upd8.append(g2keep)
            gates2upd8.add(g2keep)
        else:
            # max capacity reached: subtree full
            S.add_edge(g2keep, root)
            is_subroot_[g2keep] = False
            if g2keep in pq.tags:  # "if" required because of i=0 gates
                pq.cancel(g2keep)
            # don't consider connecting to this full subtree nodes anymore
            A.remove_nodes_from(subtree)
            for gate in ComponIn[g2drop] | ComponIn[g2keep]:
                # find_option4gate(gate)
                # gates2upd8.append(gate)
                gates2upd8.add(gate)
            # ComponIn[g2drop] = None
            # ComponIn[g2keep] = None
    # END: main loop

    # add the feeders for sub-capacity components
    for sr in np.flatnonzero(is_subroot_):
        S.add_edge(sr, A.nodes[sr]['root'])

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
