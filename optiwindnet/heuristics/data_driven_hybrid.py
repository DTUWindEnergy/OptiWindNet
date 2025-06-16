# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import time
import logging
import math
import pickle
import gzip
from enum import IntEnum
from typing import Self, Callable

import numpy as np
import networkx as nx
import torch
from mlhelpers.modelbuilders import FFMultilayerUniformModel

from ..geometric import assign_root, angle_helpers, angle_oracles_factory
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

    @classmethod
    def onehot(cls, value) -> list[int]:
        out = [0]*5
        out[value] = 1
        return out

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

    @classmethod
    def onehot(cls, value) -> list[int]:
        out = [0]*6
        out[value] = 1
        return out


class Appraiser():
    features_changed_: list[bool]

    def __init__(self, A: nx.Graph, capacity: int,
                 model_path: str,
                 subtree_span_: list[tuple[int, int]],
                 cat_feas_links_: list[int],
                 cat_feas_unions_: list[int],
                 extent_min_: list[float],
                 angle_ccw: Callable,
                 ):
        self.capacity = capacity
        self.subtree_span_  = subtree_span_
        self.cat_feas_links_ = cat_feas_links_
        self.cat_feas_unions_ = cat_feas_unions_
        self.extent_min_ = extent_min_
        self.angle_ccw = angle_ccw
        self.A = A
        # problem features
        self.capacity_12 = capacity/12
        self.inter_terminal_clearance_safe = A.graph['inter_terminal_clearance_safe']
        # scale is the median of distances of terminals to their nearest root
        d2roots = A.graph['d2roots']
        self.d2roots = d2roots
        self.log_rel_root_dist_ = np.log(
            d2roots/A.graph['inter_terminal_clearance_safe']
        )
        # load pytorch model
        #  model_path = '../../../../affairs/edge/hybrid/'
        #  model_def = pickle.load(open(model_path + 'MLP_auc_957_θ3329.pkl', 'rb'))
        with gzip.open(model_path, 'rb') as model_file:
            model_def = pickle.loads(model_file.read())
        #  model = pyg.nn.MLP(**model_def['config'])
        model = FFMultilayerUniformModel.from_suggestions(**model_def['config'])
        model.load_state_dict(model_def['state'])
        model.eval()
        self.model = model

    def appraise(self, partial_features_):
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
        #  partial_features: (sr_u, sr_v, u_is_subroot, v_is_subroot, load, target_load, extent, cos_uv_ur)
        features_list = []
        for (sr_s, sr_t, s_is_subroot, t_is_subroot, s_load, t_load, extent,
                cos_uv_ur, union_span) in partial_features_:
            s_root = self.A.nodes[sr_s]['root']
            t_root = self.A.nodes[sr_t]['root']
            # as of 2025-06: Angle span calculation is not implemented when
            #   uniting components connected to different roots.
            # TODO: subtree_span should be indexed by subroot and root;
            #       the relevant span is wrt the target's root
            s_span_lo, s_span_hi = self.subtree_span_[sr_s]
            t_span_lo, t_span_hi = self.subtree_span_[sr_t]
            union_lo, union_hi = union_span
            features_list.append((
                s_is_subroot,
                t_is_subroot,
                self.capacity_12,
                s_load/12,  # s_load_12
                t_load/12,  # t_load_12
                self.angle_ccw(s_span_lo, s_root, s_span_hi),
                self.angle_ccw(t_span_lo, t_root, t_span_hi),
                np.log(extent/self.extent_min_[sr_s]),  # s_rel_extent
                np.log(extent/self.extent_min_[sr_t]),  # t_rel_extent
                np.log(self.d2roots[sr_s, t_root]
                       /self.inter_terminal_clearance_safe),  # s_rel_root_dist
                np.log(self.d2roots[sr_t, t_root]
                       /self.inter_terminal_clearance_safe),  # t_rel_root_dist
                self.angle_ccw(union_lo, t_root, union_hi),
                (s_load + t_load)/self.capacity,  # union_rel_load
                cos_uv_ur,
                *LinkCount.onehot(self.cat_feas_links_[sr_s]),  # s_num_feas_links_cat
                *LinkCount.onehot(self.cat_feas_links_[sr_t]),  # t_num_feas_links_cat
                *UnionCount.onehot(self.cat_feas_unions_[sr_s]),  # s_num_feas_unions_cat
                *UnionCount.onehot(self.cat_feas_unions_[sr_t]),  # t_num_feas_unions_cat
            ))

        features = torch.tensor(features_list, dtype=torch.float32)
        with torch.no_grad():
            appraisals = self.model(features)
        return torch.squeeze(appraisals)
        

def data_driven_hybrid(Aʹ: nx.Graph, capacity: int, model_path: str, maxiter=10000) -> nx.Graph:
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

    closest_root = np.argmin(d2roots, axis=1)
    scale = np.median(d2roots[:, closest_root])
    #  print(self.scale, d2roots.min(), d2roots.max())
    A = as_normalized(Aʹ, scale=scale)

    roots = range(-R, 0)
    VertexC = A.graph['VertexC']

    assign_root(A)
    angles, anglesRank = angle_helpers(A)
    union_limits, angle_ccw = angle_oracles_factory(angles, anglesRank)

    # removing root nodes from A to speedup union searches
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
    # <subtree_span_>: pairs (most_CW, most_CCW) of extreme nodes of each
    #                  subtree, indexed by subroot
    subtree_span_ = [(t, t) for t in _T]

    # other structures
    # <pq>: queue prioritized by lowest negative appraisal
    pq = PriorityQueue()
    # enqueue_best_union()
    # <stale_subtrees>: deque for components that need to go through
    # stale_subtrees = deque()
    stale_subtrees = set(_T)
    fresh_subtrees = set()
    whoneeds_ = [set() for _ in _T]
    cat_feas_links_ = [-1]*T
    cat_feas_unions_ = [-1]*T
    extent_min_ = [-1.]*T
    # <i>: iteration counter
    i = 0
    steps_log = []

    appraiser = Appraiser(A, capacity, model_path, subtree_span_,
                          cat_feas_links_, cat_feas_unions_, extent_min_,
                          angle_ccw)
    # END: helper data structures

    def refresh_subtree(subroot, forbidden=None):
        '''
        - examine all the links incident on the subtree of subroot;
        - group them according to feasibility: feas/unfeas;
        - update features of subtree[subroot];
        - mark those that depend on subtree[subroot] as stale;
        '''
        if forbidden is None:
            forbidden = set()
        forbidden.add(subroot)
        root = A.nodes[subroot]['root']
        rC = VertexC[root]
        load = len(subtree_[subroot])
        load_left = capacity - load
        unfeas_links = []
        feas_links = []
        feas_features_ = []
        feas_unions = set()
        extent_min = float('inf')
        union_span_ = {}
        u_span = subtree_span_[subroot]
        for u in (t for t in subtree_[subroot] if A[t]):
            u_is_subroot=(u == subroot)
            uC = VertexC[u]
            vec_ur = rC - uC
            d_ur = np.hypot(*vec_ur)
            for v, uvD in A[u].items():
                sr_v = subroot_[v]
                target_load = len(subtree_[sr_v])
                if (sr_v in forbidden or target_load > load_left):
                    unfeas_links.append((u, v))
                else:
                    feas_unions.add(sr_v)
                    extent = uvD['length']
                    vec_uv = VertexC[v] - uC
                    # TODO: it is tempting to do cos_uv_ur only once, globally,
                    #       but that would need to account for multiple roots
                    cos_uv_ur = (np.dot(vec_uv, vec_ur)/d_ur/np.hypot(*vec_uv))
                    union_span = union_span_.get(sr_v)
                    if union_span is None:
                        # assess the union's angle span
                        v_span = subtree_span_[sr_v]
                        union_span = union_limits(root, u, *u_span, v, *v_span)
                        union_span_[sr_v] = union_span
                    feas_links.append((u, v))
                    feas_features_.append((
                        subroot, sr_v,
                        u_is_subroot, (v == sr_v),
                        load, target_load,
                        extent, cos_uv_ur,
                        union_span
                    ))
                    extent_min = min(extent_min, extent)
        link_caused_staleness = set()
        for s, t in unfeas_links:
            if (s, t) in pq.tags:
                pq.cancel((s, t))
            if (t, s) in pq.tags:
                pq.cancel((t, s))
            link_caused_staleness.add(subroot_[t])
        #  print(f'[{i}] {link_caused_staleness}\n{whoneeds_[subroot]}\n{feas_unions}')
        #  assert link_caused_staleness == whoneeds_[subroot] - feas_unions, 'set mismatch'
        #  assert len(link_caused_staleness & fresh_subtrees) == 0, 'size mismatch'
        #  rel_component_excess = num_components/min_components - 1
        #  dropped_dependencies = whoneeds_[subroot] - feas_unions
        # all that can no longer link to subtree_[subroot] are stale

        stale_subtrees.update(link_caused_staleness - fresh_subtrees)
        if not feas_unions:
            # this handles subtrees that reached capacity or became isolated
            S.add_edge(subroot, root)
            A.remove_nodes_from(subtree_[subroot])
            steps_log.append((subroot, root))
            is_subroot_[subroot] = False
            return [], []
        # discard useless edges
        A.remove_edges_from(unfeas_links)
        fresh_subtrees.add(subroot)
        whoneeds_[subroot] = feas_unions
        for sr in feas_unions:
            # TODO: this loop is probably unnecessary, confirming it with assert:
            # there might be a change in the subroot of a feas_union...
            #  assert subroot in whoneeds_[sr], 'subroot not in whoneeds_[feas_union]'
            # THIS LOOP IS ACTUALLY NECESSARY
            whoneeds_[sr].add(subroot)
        cat_feas_unions = UnionCount.encode(len(feas_unions))
        cat_feas_links = LinkCount.encode(len(feas_links))
        if ((cat_feas_links != cat_feas_links_[subroot])
                or (cat_feas_unions != cat_feas_unions_[subroot])
                or (extent_min != extent_min_[subroot])):
            # since the current subtree had features changes, all that depend
            # on it must be marked as stale
            stale_subtrees.update(whoneeds_[subroot] - fresh_subtrees)
            cat_feas_unions_[subroot] = cat_feas_unions
            cat_feas_links_[subroot] = cat_feas_links
            extent_min_[subroot] = extent_min
        return feas_links, feas_features_

    loop = True
    # BEGIN: main loop
    while loop:
        i += 1
        if i > maxiter:
            error('ERROR: maxiter reached (%d)', i)
            break
        debug('[%d]', i)
        if stale_subtrees:
            debug('stale_subtrees (%d): %s', len(stale_subtrees), tuple(F[sr] for sr in stale_subtrees))
        links_to_upd = []
        links_features = []
        while stale_subtrees:
            subroot = stale_subtrees.pop()
            feas_links, feas_features = refresh_subtree(subroot)
            links_to_upd.extend(feas_links)
            links_features.extend(feas_features)
        
        # appraise and enqueue links
        if links_features:
            appraisals = appraiser.appraise(links_features)
            for link, (sr_u, *_), appraisal in zip(links_to_upd, links_features, appraisals):
                pq.add(-appraisal, link, sr_u)

        if not pq:
            # finished
            break

        # get best link
        (u, v), sr_u = pq.top()
        debug('<popped> «%s–%s», sr_u: <%s>', F[u], F[v], F[sr_u])
        # TODO: reassess this hack
        if (u, v) not in A.edges:
            debug('>>> top-edge is not in A anymore <<<')
            continue

        sr_v = subroot_[v]
        if (d2roots[u, A.nodes[sr_u]['root']]
                < d2roots[v, A.nodes[sr_v]['root']]):
            sr_kept, sr_dropped = sr_u, sr_v
        else:
            sr_kept, sr_dropped = sr_v, sr_u

        root = A.nodes[sr_kept]['root']
        subtree = subtree_[sr_kept]

        # assess the union's angle span
        union_span = union_limits(root, u, *subtree_span_[sr_u],
                                        v, *subtree_span_[sr_v])
        debug(f'<angle_span> //%s//', union_span)

        # edge addition starts here
        debug('<add edge> «%s–%s» subroot <%s>', F[u], F[v], F[sr_v])
        S.add_edge(u, v)
        steps_log.append((u, v))

        # update the component's angle span
        subtree_span_[sr_kept] = union_span

        subtree.extend(subtree_[sr_dropped])
        stale_subtrees.clear()
        stale_subtrees.update(whoneeds_[sr_kept], whoneeds_[sr_dropped])
        # TODO: rethink why not: stale_subtrees.remove(sr_dropped)
        stale_subtrees.discard(sr_dropped)
        if len(subtree) == capacity:
            stale_subtrees.discard(sr_kept)
            S.add_edge(sr_kept, root)
            steps_log.append((sr_kept, root))
            A.remove_nodes_from(subtree)
        else:
            A.remove_edge(u, v)

        # this block might be unnecessary if whoneeds is not for dropped dependencies
        # TODO: rethink why not: whoneeds_[sr_dropped].remove(sr_kept)
        whoneeds_[sr_dropped].discard(sr_kept)
        for sr in whoneeds_[sr_dropped]:
            # TODO: rethink why not: whoneeds_[sr].remove(sr_dropped)
            whoneeds_[sr].discard(sr_dropped)
            whoneeds_[sr].add(sr_kept)

        # update terminal->subroot mapping for sr_dropped's terminals
        for t in subtree_[sr_dropped]:
            subroot_[t] = sr_kept
        if pq:
            debug('heap top: <%s>, «%s» %.3f', F[pq[0][-1]],
                  tuple(F[x] for x in pq[0][-2]), pq[0][0])
        else:
            debug('heap EMPTY')
        is_subroot_[sr_dropped] = False
        num_components -= 1
        # remove from A and pq the edges that cross ⟨u, v⟩
        for s, t in edge_crossings(u, v, A, diagonals):
            A.remove_edge(s, t)
            pq.cancel((s, t))
            pq.cancel((t, s))
            stale_subtrees.add(subroot_[s])
            stale_subtrees.add(subroot_[t])
    # END: main loop

    # add the feeders for sub-capacity components
    #  for sr in np.flatnonzero(is_subroot_):
    #      debug('Adding sub-capacity subtree: subroot %d', sr)
    #      S.add_edge(sr, A.nodes[sr]['root'])

    debug('Final number of components: %d', num_components)
    calcload(S)
    # algorithm finished, store some info in the graph object
    S.graph.update(
        runtime=time.perf_counter() - start_time,
        capacity=capacity,
        creator='data_driven_hybrid',
        iterations=i,
        solver_details=dict(
            steps_log=steps_log,
            iterations=i,
        ),
    )
    return S
