# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import time
import logging
import math
import pickle
from enum import IntEnum
from typing import Self

import numpy as np
import networkx as nx
import torch
import torch_geometric as pyg

from ..geometric import angle, assign_root
from ..crossings import edge_crossings
from ..utils import NodeTagger
from .priorityqueue import PriorityQueue
from ..interarraylib import as_normalized, calcload

F = NodeTagger()

lggr = logging.getLogger(__name__)
debug, info, warn, error = lggr.debug, lggr.info, lggr.warning, lggr.error


class LinkCount(IntEnum):
    ONE_OR_TWO = 0
    THREE_OR_FOUR = 1
    FIVE_OR_SIX = 2
    SEVEN_OR_EIGHT = 3
    NINE_OR_MORE = 4

    @classmethod
    def encode(cls, count: int) -> Self:
        return cls(min(4, (count - 1)//2))


class UnionCount(IntEnum):
    ONE = 0
    TWO = 1
    THREE = 2
    FOUR = 3
    FIVE = 4
    SIX_OR_MORE = 5

    @classmethod
    def encode(cls, count: int) -> Self:
        return cls(min(5, count - 1))


class Appraiser():
    features_changed_: list[bool]
    num_feas_joins_: list[int | None]
    num_feas_links_: list[int | None]
    extent_min_: list[float | None]
    dependencies_: list[list[int]]

    def __init__(self, A: nx.Graph, capacity: int):
        # problem features
        self.capacity_12 = capacity/12
        T = A.graph['T']
        _T = range(T)
        # scale is the median of distances of terminals to their nearest root
        d2roots = A.graph['d2roots']
        self.log_rel_root_dist_ = np.log(
            d2roots/A.graph['inter_terminal_clearance_safe']
        )
        self.num_feas_joins_ = [None for _ in _T]
        self.num_feas_links_ = [None for _ in _T]
        self.extent_min_ = [None for _ in _T]
        self.features_changed_ = [True for _ in _T]
        self.dependencies_ = [[] for _ in _T]
        closest_root = np.argmin(d2roots, axis=1)
        self.scale = np.median(d2roots[:, closest_root])
        print(self.scale, d2roots.min(), d2roots.max())
        # load pytorch model
        model_path = '../../../../affairs/edge/hybrid/'
        model_def = pickle.load(open(model_path + 'MLP_auc_957_θ3329.pkl', 'rb'))
        model = pyg.nn.MLP(**model_def['config'])
        model.load_state_dict(model_def['state'])
        self.model = model

    def set_component_features(self,):
        pass

    def appraise(self, features_):
        # rules for making an appraisal stale:
        # - direct_neighbors: every subtree that has feas_links to one of the joined subtrees
        # - indirect_neighbors: only if there is change in the target's (min_extent, feas_links_cat, feas_unions_cat)

        #  't_is_subroot', stable
        #  't_load_12', stable
        #  't_span', stable
        #  't_log_rel_extent', ? depends... 
        #  't_log_rel_root_dist', stable
        #  't_feas_links_cat_0..4', needs checking
        #  't_feas_unions_cat_0..5', needs checking
        #  features: (sr_u, sr_v, u_is_subroot, v_is_subroot, load, target_load, extent, cos_uv_ur)
        #  ['s_is_subroot',
        #  't_is_subroot',
        #  'capacity_12',
        #  's_load_12',
        #  't_load_12',
        #  's_span',
        #  't_span',
        #  's_log_rel_extent',
        #  't_log_rel_extent',
        #  's_log_rel_root_dist',
        #  't_log_rel_root_dist',
        #  'union_span',
        #  'union_rel_load',
        #  'cos',
        #  's_feas_links_cat_0..4',
        #  't_feas_links_cat_0..4',
        #  's_feas_unions_cat_0..5',
        #  't_feas_unions_cat_0..5']
        #  features: (sr_u, sr_v, u_is_subroot, v_is_subroot, load, target_load, extent, cos_uv_ur)
        features = torch.tensor(((
            s_is_subroot,
            t_is_subroot,
            self.capacity_12,
            s_load/12,  # s_load_12
            t_load/12,  # t_load_12
            self.span[u],
            self.span[v],
            np.log(extent/self.s_extent_min),  # s_rel_extent
            np.log(extent/self.t_extent_min),  # t_rel_extent
            np.log(s_subroot_d2root/self.inter_terminal_clearance_safe),  # s_rel_root_dist
            np.log(t_subroot_d2root/self.inter_terminal_clearance_safe),  # t_rel_root_dist
            union_span,
            union_load/self.capacity,  # union_rel_load
            cos,
            LinkCount.encode(self.s_num_feas_links),  # s_num_feas_links_cat
            LinkCount.encode(self.t_num_feas_links),  # t_num_feas_links_cat
            LinkCount.encode(self.s_num_feas_unions),  # s_num_feas_unions_cat
            LinkCount.encode(self.t_num_feas_unions),  # t_num_feas_unions_cat
        ),))
        appraisal = model(features)

        

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
    _T = range(T)
    diagonals = Aʹ.graph['diagonals']
    d2roots = Aʹ.graph['d2roots']
    S = nx.Graph(R=R, T=T)
    appraiser = Appraiser(Aʹ, capacity)
    A = as_normalized(Aʹ, scale=appraiser.scale)

    roots = range(-R, 0)
    VertexC = A.graph['VertexC']

    assign_root(A)

    # removing root nodes from A to speedup enqueue_best_union
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
    # <subtree_>: maps nodes to the list of nodes in their subtree
    subtree_ = [[t] for t in _T]
    # <subroot_>: maps terminals to their subroots
    subroot_ = list(_T)

    # mappings from components (identified by their subroots)
    # <ComponIn>: maps component to set of components queued to merge in
    ComponIn = [set() for _ in _T]
    # <subtree_span_>: pairs (most_CW, most_CCW) of extreme nodes of each
    #                  subtree, indexed by subroot
    subtree_span_ = [(t, t) for t in _T]

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

    appraiser = Appraiser(A, capacity)
    # END: helper data structures

    def get_feas_links(subroot, forbidden=None):
        # gather all the edges leaving the subtree of subroot
        if forbidden is None:
            forbidden = set()
        forbidden.add(subroot)
        choices = []
        root = A.nodes[subroot]['root']
        sr_d2root = d2roots[subroot, root]
        rC = VertexC[root]
        load = len(subtree_[subroot])
        load_left = capacity - load
        unfeas_links = []
        feas_links = []
        feas_features_ = []
        feas_unions = set()
        extent_min = float('inf')
        for u in (t for t in subtree_[subroot] if A[t]):
            u_is_subroot=(u == subroot)
            uC = VertexC[u]
            vec_ur = rC - uC
            d_ur = np.hypot(*vec_ur)
            for v, uvD in A[u].items():
                target_load = len(subtree_[v])
                sr_v = subroot_[v]
                if (sr_v in forbidden or target_load > load_left):
                    unfeas_links.append((u, v))
                else:
                    feas_unions.add(sr_v)
                    extent = uvD['length']
                    vec_uv = VertexC[v] - uC
                    cos_uv_ur = (np.dot(vec_uv, vec_ur)/d_ur/np.hypot(*vec_uv))
                    feas_links.append((u, v))
                    feas_features_.append((
                        sr_u, sr_v,
                        u_is_subroot, (v == sr_v),
                        load, target_load,
                        extent, cos_uv_ur
                    ))
                    extent_min = min(extent_min, extent)
        # discard useless edges
        A.remove_edges_from(unfeas_links)
        #  rel_component_excess = num_components/min_components - 1
        stale_subtrees |= appraiser.dependencies_[subroot]
        dropped_dependencies = appraiser.dependencies_[subroot] - feas_unions
        for sr in dropped_dependencies:
            appraiser.dependencies_[sr].remove(subroot)
            stale_subtrees.add(sr)
        cat_feas_unions = UnionCount.encode(len(feas_unions))
        appraiser.dependencies_[subroot] = feas_unions
        for sr in feas_unions:
            appraiser.dependencies_[sr].add(subroot)
        cat_feas_links = LinkCount.encode(len(feas_links))
        changed = (
            (cat_feas_links != appraiser.cat_feas_links_[subroot])
            or (cat_feas_unions != appraiser.cat_feas_unions_[subroot])
            or (extent_min != appraiser.extent_min_[subroot])
        )
        if changed:
            stale_subtrees |= appraiser.dependencies_[subroot]
            appraiser.changed_[subroot] = changed
            appraiser.cat_feas_unions_[subroot] = cat_feas_unions
            appraiser.cat_feas_links_[subroot] = cat_feas_links
            appraiser.extent_min_[subroot] = extent_min
        return feas_links, feas_features_

    def propagate_join_(sr_u, sr_v):
        direct_stale = appraiser.dependencies_[sr_u]|appraiser.dependencies_[sr_v]


    def appraise_feasible(feas_links, feas_features_):
        appraisals = appraiser.appraise(feas_features_)
        # TODO: the appraiser calls often operate on a subset of feasible links,
        # so we need to join the still-good appraisals with the newly updated ones
        best_idx = np.argmin(appraisals)
        return feas_links[best_idx], appraisals[best_idx]

    def enqueue_best_union(subroot):
        debug('<enqueue_best_union> starting... subroot = <%s>', F[subroot])
        if edges2ban:
            debug('<<<<<<<edges2ban>>>>>>>>>>> _%d_', len(edges2ban))
        while edges2ban:
            # edge2ban = edges2ban.popleft()
            edge2ban = edges2ban.pop()
            ban_queued_union(*edge2ban)
        # () get component expansion edge with weight
        feas_links, feas_features = get_feas_links(subroot)
        if edge is not None:
            pq.add(tradeoff, subroot, edge)
            ComponIn[subroot_[edge[1]]].add(subroot)
            debug('<pushed> sr_u <%s>, «%s–%s», tradeoff = %.3f',
                  F[subroot], F[edge[0]], F[edge[1]], tradeoff)
        else:
            # no viable edge is better than subroot for this node
            debug('<cancelling> %s', F[subroot])
            if subroot in pq.tags:
                pq.cancel(subroot)

    def ban_queued_union(sr_u, u, v):
        if (u, v) in A.edges:
            A.remove_edge(u, v)
        else:
            warn('<< UNLIKELY <ban_queued_union()> «%s–%s» not in A.edges >>',
                  F[u], F[v])
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

        # if this if is not visited, replace the above with ComponIn check
        # this means that if sr_v is to also merge with sr_u, then the
        # edge of the merging must be (v, u)
        if componin != is_reverse:
            print(f'«{F[u]}–{F[v]}», '
                  f'sr_u <{F[sr_u]}>, sr_v <{F[sr_v]}> '
                  f'componin: {componin}, is_reverse: {is_reverse}')

        # END: block to be simplified

    # initialize pq
    for t in _T:
        enqueue_best_union(t)

    loop = True
    # BEGIN: main loop
    while loop:
        i += 1
        if i > maxiter:
            error('ERROR: maxiter reached (%d)', i)
            break
        debug('[%d]', i)
        # debug(f'[{i}] bj–bm root: {A.edges[(F.bj, F.bm)]["root"]}')
        if stale_subtrees:
            debug('stale_subtrees: %s', tuple(F[sr] for sr in stale_subtrees))
        for stale_subtree in stale_subtrees:
            # enqueue_best_union(stale_subtrees.popleft())
            enqueue_best_union(stale_subtree)
        stale_subtrees.clear()
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
        eX = edge_crossings(u, v, S, diagonals)

        if eX:
            debug('<edge_crossing> discarding «%s–%s»: would cross %s',
                  F[u], F[v], tuple((F[s], F[t]) for s, t in eX))
            # abort_edge_addition(sr_u, u, v)
            prevented_crossings += 1
            ban_queued_union(sr_u, u, v)
            continue

        sr_v = subroot_[v]
        root = A.nodes[sr_v]['root']
        subtree = subtree_[v]

        # assess the union's angle span
        keepLo, keepHi = subtree_span_[sr_v]
        dropLo, dropHi = subtree_span_[sr_u]
        newHi = dropHi if angle(*VertexC[[keepHi, root, dropHi]]) > 0 else keepHi
        newLo = dropLo if angle(*VertexC[[dropLo, root, keepLo]]) > 0 else keepLo
        # update the component's angle span
        subtree_span_[sr_v] = newLo, newHi
        debug(f'<angle_span> //%s:%s//', F[newLo], F[newHi])

        # edge addition starts here
        subtree.extend(subtree_[u])

        # this is wrong, the feeder removed should be the longest one
        S.remove_edge(A.nodes[u]['root'], sr_u)

        log.append((i, 'remE', (A.nodes[u]['root'], sr_u)))

        sr_v_entry = pq.tags.get(sr_v)
        if sr_v_entry is not None:
            _, _, _, (_, t) = sr_v_entry
            # print('node', F[t], 'subroot', F[subroot_[t]])
            ComponIn[subroot_[t]].remove(sr_v)
        # TODO: think about why a discard was needed
        ComponIn[sr_v].discard(sr_u)

        # assign root, subroot and subtree to the newly added terminals
        for t in subtree_[u]:
            A.nodes[t]['root'] = root
            subroot_[t] = sr_v
            subtree_[t] = subtree
        debug('<add edge> «%s–%s» subroot <%s>', F[u], F[v], F[sr_v])
        if pq:
            debug('heap top: <%s>, «%s» %.3f', F[pq[0][-2]],
                  tuple(F[x] for x in pq[0][-1]), pq[0][0])
        else:
            debug('heap EMPTY')
        #  G.add_edge(u, v, **A.edges[u, v])
        S.add_edge(u, v)
        is_subroot_[sr_u] = False
        num_components -= 1
        log.append((i, 'addE', (u, v)))
        # remove from consideration edges internal to subtrees
        A.remove_edge(u, v)

        # finished adding the edge, now check the consequences
        capacity_left = capacity - len(subtree_[u]) - len(subtree_[v])
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
            S.add_edge(sr_v, root)
            is_subroot_[sr_v] = False
            if sr_v in pq.tags:  # "if" required because of i=0 commits
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
