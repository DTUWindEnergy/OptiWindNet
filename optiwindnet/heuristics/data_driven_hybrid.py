# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import abc
import logging
import time
import math
import heapq
from collections import defaultdict
from enum import IntEnum
from typing import Callable, Self, Sequence

from bitarray import bitarray
import networkx as nx
import numpy as np
from scipy.stats import rankdata

from ..crossings import edge_crossings
from ..geometric import (
    angle_oracles_factory,
    minimum_spanning_forest,
)
from ..interarraylib import (
    as_normalized,
    calcload,
    add_link_cosines,
    add_link_blockmap,
    add_terminal_closest_root,
)

lggr = logging.getLogger(__name__)
debug, info, warn, error = lggr.debug, lggr.info, lggr.warning, lggr.error


__version = 'v10'

_ONE = bitarray('1')


class LinkCount(IntEnum):
    ONE_OR_TWO = 0
    THREE_OR_FOUR = 1
    FIVE_OR_SIX = 2
    SEVEN_OR_EIGHT = 3
    NINE_OR_MORE = 4

    @classmethod
    def encode(cls, count: int) -> Self:
        return cls(min(4, (count - 1) // 2))

    @classmethod
    def onehot(cls, value) -> list[int]:
        out = [0] * 5
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
        out = [0] * 6
        out[value] = 1
        return out


class Appraiser(abc.ABC):
    # DDHv10 - arbitrary constants (must match data-wrangling values)
    CAPACITY_FLOOR = 4
    T_SCALE = 0.02
    LOG_FLOOR = np.e**-6

    @abc.abstractmethod
    def appraise(self, partial_features_: list[tuple]) -> Sequence:
        pass

    def encode_categorical(self, *args):
        """No encoding by default."""
        return args

    def update_problem_variables(
        self,
        A: nx.Graph,
        Aʹ: nx.Graph,
        capacity: int,
        subtree_span_: list[list[tuple[int, int]]],
        cat_feas_links_: list[int],
        cat_feas_unions_: list[int],
        extent_min_: list[float],
        angle_ccw: Callable,
    ):
        self.A = A
        self.capacity = capacity
        self.capacity_cat = capacity - self.CAPACITY_FLOOR
        self.subtree_span_ = subtree_span_
        self.cat_feas_links_ = cat_feas_links_
        self.cat_feas_unions_ = cat_feas_unions_
        self.extent_min_ = extent_min_
        self.angle_ccw = angle_ccw
        self.T = T = A.graph['T']
        self.T_scaled = math.log(T * self.T_SCALE)
        self.max_steps = T - math.ceil(T / capacity)
        self.d2roots = A.graph['d2roots'][:T]
        self.log_d2roots = np.log(A.graph['d2roots'][:T])

        # calculate the ratio (second moment of area)/(s.m.a of homogeneous circle)
        # use unit area for the moment; only wrt the first root (the hard-coded -1)
        norm_T_d2root = Aʹ.graph['norm_scale'] * Aʹ.graph['d2roots'][:T, -1]
        # the unit area circle with root at the center and T homogeneously distributed
        # terminals would have a moment of inertia T/2/π - use it to normalize (range 1..7)
        self.log_rel_moment = math.log((norm_T_d2root**2).sum().item() * 2 * np.pi / T)

    def full_from_partial_features(self, partial_features_: list) -> list:
        """DDHv10 features"""
        full_features = []
        for (
            s,
            t,
            num_steps,
            sr_s,
            sr_t,
            s_is_subroot,
            t_is_subroot,
            s_load,
            t_load,
            is_delaunay,
            extent,
            cos,
            num_blocked,
            union_span,
        ) in partial_features_:
            s_root = self.A.nodes[sr_s]['root']
            t_root = self.A.nodes[sr_t]['root']
            # as of 2025-06: Angle span calculation is not implemented when
            #   uniting components connected to different roots.
            s_span_lo, s_span_hi = self.subtree_span_[sr_s][t_root]
            t_span_lo, t_span_hi = self.subtree_span_[sr_t][t_root]
            union_lo, union_hi = union_span
            union_load = s_load + t_load
            link_angle = self.angle_ccw(s, t_root, t)
            full_features.append(
                (
                    s_is_subroot,  # s_is_subroot,
                    t_is_subroot,  # t_is_subroot,
                    is_delaunay,  # is_delaunay
                    self.T_scaled,  # T_scaled
                    # NOTE: step_completeness is tricky, because it would trigger
                    #       a full appraisal refresh after each step. This is not
                    #       implemented and probably should not be. As a result,
                    #       the appraisals reflect the num_steps when they were last
                    #       updated. Hopefully the update won't lag too many steps.
                    num_steps / self.max_steps,  # step_completeness
                    self.log_rel_moment,  # log_rel_moment
                    s_load / self.capacity,  # s_rel_load
                    t_load / self.capacity,  # t_rel_load
                    self.log_d2roots[sr_s, s_root].item(),  # log_s_subroot_d2root
                    self.log_d2roots[sr_t, t_root].item(),  # log_t_subroot_d2root
                    self.angle_ccw(s_span_lo, t_root, s_span_hi),  # s_span
                    self.angle_ccw(t_span_lo, t_root, t_span_hi),  # t_span
                    self.extent_min_[sr_s],  # s_extent_min
                    self.extent_min_[sr_t],  # t_extent_min
                    np.log(
                        np.maximum(
                            self.d2roots[sr_s, t_root] - self.d2roots[sr_t, t_root],
                            self.LOG_FLOOR,
                        )
                    ).item(),  # log_radial_gain
                    np.log(
                        np.maximum(self.d2roots[sr_s, s_root] - extent, self.LOG_FLOOR)
                    ).item(),  # saving
                    self.angle_ccw(union_lo, t_root, union_hi),  # union_span
                    union_load / self.capacity,  # union_rel_load
                    math.log(extent),  # extent
                    cos,  # cos
                    (
                        link_angle
                        if link_angle <= math.pi
                        else (2 * math.pi - link_angle)
                    ),  # span
                    num_blocked / self.capacity,  # blocked_subtree_equiv
                    *self.encode_categorical(
                        self.cat_feas_links_[sr_s],  # s_num_feas_links_cat
                        self.cat_feas_links_[sr_t],  # t_num_feas_links_cat
                        self.cat_feas_unions_[sr_s],  # s_num_feas_unions_cat
                        self.cat_feas_unions_[sr_t],  # t_num_feas_unions_cat
                        self.capacity_cat,
                    ),
                )
            )
        return full_features


class AppraiserSGDClassifier(Appraiser):
    def __init__(self, model_data: dict):
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler

        self.name = model_data['name']
        model_params = np.load(model_data['path'])

        self.feature_subset_ = model_params['feature_subset_']

        self.model = model = SGDClassifier(loss='modified_huber')
        model.coef_ = model_params['coef_']
        model.intercept_ = model_params['intercept_']
        model.classes_ = model_params['classes_']
        model.n_features_in_ = model_params['n_features_in'].item()

        self.scaler = scaler = StandardScaler()
        scaler.mean_ = model_params['mean_']
        scaler.scale_ = model_params['scale_']
        scaler.var_ = scaler.scale_**2
        scaler.n_features_in_ = model_params['n_features_in'].item()
        scaler.n_samples_seen_ = model_params['n_samples_seen'].item()

    def appraise(self, partial_features_: list[tuple]) -> Sequence:
        features = np.array(
            self.full_from_partial_features(partial_features_),
            dtype=np.float32,
        )
        z_features = self.scaler.transform(features[:, self.feature_subset_])
        appraisals = self.model.predict_proba(z_features)[:, 1]
        return appraisals


class AppraiserXGBoost(Appraiser):
    def __init__(self, model_data: dict):
        import tl2cgen

        self.model = tl2cgen.Predictor(model_data['path'])
        self.name = model_data['name']
        self.DMatrix = tl2cgen.DMatrix

    def appraise(self, partial_features_: list[tuple]) -> Sequence:
        features = np.array(
            self.full_from_partial_features(partial_features_),
            dtype=np.float32,
        )
        dmat = self.DMatrix(features)
        appraisals = self.model.predict(dmat).squeeze(axis=(-2, -1))
        return appraisals


class AppraiserTorch(Appraiser):
    def __init__(self, model_data: dict):
        import torch

        self.torch = torch
        from mlhelpers import modelbuilders

        # load pytorch model
        self.model = getattr(modelbuilders, model_data['cls']).from_suggestions(
            **model_data['config']
        )
        self.model.load_state_dict(model_data['state'])
        self.model.eval()
        self.name = model_data['name']

    def update_problem_variables(
        self,
        A: nx.Graph,
        Aʹ: nx.Graph,
        capacity: int,
        subtree_span_: list[list[tuple[int, int]]],
        cat_feas_links_: list[int],
        cat_feas_unions_: list[int],
        extent_min_: list[float],
        angle_ccw: Callable,
    ):
        super().update_problem_variables(
            A,
            Aʹ,
            capacity,
            subtree_span_,
            cat_feas_links_,
            cat_feas_unions_,
            extent_min_,
            angle_ccw,
        )
        self.capacity_onehot = {
            4: (1, 0, 0, 0, 0, 0),
            5: (0, 1, 0, 0, 0, 0),
            6: (0, 0, 1, 0, 0, 0),
            7: (0, 0, 0, 1, 0, 0),
            8: (0, 0, 0, 0, 1, 0),
            9: (0, 0, 0, 0, 0, 1),
        }[capacity]

    def encode_categorical(
        self,
        s_num_feas_links_cat,
        t_num_feas_links_cat,
        s_num_feas_unions_cat,
        t_num_feas_unions_cat,
        capacity_cat,
    ):
        return (
            *LinkCount.onehot(s_num_feas_links_cat),
            *LinkCount.onehot(t_num_feas_links_cat),
            *UnionCount.onehot(s_num_feas_unions_cat),
            *UnionCount.onehot(t_num_feas_unions_cat),
            *self.capacity_onehot,
        )

    def appraise(self, partial_features_: list[tuple]) -> Sequence:
        features = self.torch.tensor(
            self.full_from_partial_features(partial_features_),
            dtype=self.torch.float32,
        )
        with self.torch.no_grad():
            appraisals = self.model(features)
        return appraisals.squeeze(1)


def data_driven_hybrid(
    Aʹ: nx.Graph,
    capacity: int,
    appraiser: Appraiser,
    maxiter=10000,
    threshold: float = 0.0,
    blockage_subtree_feeder_lim: float = 5.5,
) -> nx.Graph:
    """Hybrid machine-learning and Esau-Williams heuristic for C-MST

    Args:
        A: available edges graph
        capacity: maximum link capacity
        maxiter: fail-safe to avoid locking in an infinite loop

    Returns:
        Solution topology.
    """

    start_time = time.perf_counter()
    R, T = Aʹ.graph['R'], Aʹ.graph['T']
    _T = range(T)
    diagonals = Aʹ.graph['diagonals']
    d2rootsʹ = Aʹ.graph['d2roots']
    P_A = Aʹ.graph['planar']
    d2roots_rank_ = rankdata(d2rootsʹ, method='dense').reshape(d2rootsʹ.shape)

    # calculate the reference extent
    MSF = minimum_spanning_forest(Aʹ)
    # normalize all distances by the average edge extent of the MST forest
    A = as_normalized(Aʹ, scale=T / MSF.size(weight='length'))
    A.graph['d2roots_rank_'] = d2roots_rank_
    d2roots = A.graph['d2roots']
    roots = range(-R, 0)

    add_terminal_closest_root(A)
    # removing root nodes from A to speedup union searches
    A.remove_nodes_from(roots)
    add_link_blockmap(A)
    add_link_cosines(A)

    # remove links that have negative savings both ways from the start
    to_remove = []
    for u, v, edgeD in A.edges(data=True):
        extent = edgeD['length']
        root = A.nodes[v]['root']
        if (
            extent > d2roots[u, A.nodes[u]['root']]
            and extent > d2roots[v, A.nodes[v]['root']]
        ):
            # negative savings -> useless link
            to_remove.append((u, v))
    debug('links removed in pre-processing: %s', to_remove)
    A.remove_edges_from(to_remove)
    del to_remove
    # BEGIN: time-saving pre-calculations
    angle__, angle_rank__ = A.graph['angle__'], A.graph['angle_rank__']
    union_limits, angle_ccw = angle_oracles_factory(angle__, angle_rank__)
    is_delaunay_ = {}
    for u, v, kind in A.edges(data='kind'):
        uv_uniq = (u, v) if u < v else (v, u)
        is_delaunay_[uv_uniq] = kind.endswith('delaunay')
    # END: time-saving pre-calculations

    # BEGIN: component accounting
    # <is_feederless_>: flags subroots that still need a feeder
    is_feederless_ = np.full((T,), True, dtype=bool)
    # END: component accounting

    # BEGIN: helper data structures

    # mappings from nodes
    # <subtree_>: maps nodes to the list of nodes in their subtree
    subtree_ = [bitarray(T) for _ in _T]
    for t, subtree in zip(_T, subtree_):
        subtree[t] = 1
    # <subroot_>: maps terminals to their subroots
    subroot_ = list(_T)

    # mappings from components (indexed by their subroots)
    # <subtree_span_>: pairs (most_CW, most_CCW) of extreme nodes of each
    #                  subtree
    subtree_span_ = [[(t, t) for _ in roots] for t in _T]
    # <subtree_blocked_>: sets of blocked terminals from other components
    subtree_blocked_ = [bitarray(T) for _ in _T]
    max_blockable = blockage_subtree_feeder_lim * capacity

    # other structures
    # <pq>: queue prioritized by lowest negative appraisal
    #  pq = PriorityQueue()
    # enqueue_best_union()
    # <stale_subtrees>: deque for components that need to go through
    # stale_subtrees = deque()
    stale_subtrees = set(_T)
    fresh_subtrees = set()
    whoneeds_ = [set() for _ in _T]
    cat_feas_links_ = [-1] * T
    cat_feas_unions_ = [-1] * T
    extent_min_ = [-1.0] * T
    # indexed by the cat_feas_unions of the subroots:
    prio_tier_ = tuple(set() for _ in range(len(UnionCount)))
    _NULL_UNION = (-float('inf'), -1, -1, -1, -1)
    top_outbound_ = [_NULL_UNION] * T
    # <iteration>: iteration counter
    iteration = 0
    num_steps = 0

    # END: helper data structures

    # BEGIN: output data containers
    S = nx.Graph(R=R, T=T)
    steps_log = defaultdict(list)
    appraisal_log = {}
    purge_log = defaultdict(list)
    stale_log = {}
    # END: output data containers
    appraiser.update_problem_variables(
        A,
        Aʹ,
        capacity,
        subtree_span_,
        cat_feas_links_,
        cat_feas_unions_,
        extent_min_,
        angle_ccw,
    )

    def refresh_subtree(subroot):
        """
        - examine all the links incident on the subtree of subroot;
        - group them according to feasibility: feas/unfeas;
        - update features of subtree[subroot];
        - mark those that depend on subtree[subroot] as stale;
        """
        root = A.nodes[subroot]['root']
        load_self = subtree_[subroot].count()
        load_left = capacity - load_self
        unfeas_links = []
        # feasible (feas) means union load <= capacity
        num_feas_links = 0
        feas_unions = set()
        # proper means feasible and subtree of u has a longer feeder than of v
        proper_links = []
        proper_features_ = []
        extent_min = float('inf')
        union_span_cache = {}
        link_caused_staleness = set()
        for u in (t for t in subtree_[subroot].search(_ONE) if A[t]):
            u_is_subroot = u == subroot
            for v, uvD in A[u].items():
                uv_uniq = (u, v) if u < v else (v, u)
                sr_v = subroot_[v]
                root_v = A.nodes[sr_v]['root']
                load_other = subtree_[sr_v].count()
                extent = uvD['length']
                if sr_v == subroot:
                    # link internal to subtree only add once
                    if u < v:
                        unfeas_links.append(uv_uniq)
                    continue
                elif (
                    load_other > load_left
                    or (
                        (
                            subtree_blocked_[subroot]
                            | subtree_blocked_[sr_v]
                            | uvD['blocked__'][root_v]
                        )
                        & ~(subtree_[sr_v] | subtree_[subroot])
                    ).count()
                    > max_blockable
                ):
                    link_caused_staleness.add(sr_v)
                    unfeas_links.append(uv_uniq)
                    continue
                elif (d2roots_rank_[subroot, root] > d2roots_rank_[sr_v, root_v]) or (
                    (u > v)
                    and (d2roots_rank_[subroot, root] == d2roots_rank_[sr_v, root_v])
                ):
                    # uv links subtree with longer feeder to shorter: proper
                    # check if using uv reduces total length
                    if extent > d2roots[subroot, root]:
                        # negative savings -> useless
                        link_caused_staleness.add(sr_v)
                        unfeas_links.append(uv_uniq)
                        # sr_v needs to be reprocessed
                        fresh_subtrees.discard(sr_v)
                        continue
                    proper_links.append((uv_uniq, subroot, sr_v))
                    u_span = subtree_span_[subroot][root_v]
                    union_span = union_span_cache.get(sr_v)
                    if union_span is None:
                        # assess the union's angle span
                        v_span = subtree_span_[sr_v][root_v]
                        # TODO: for multi-root, spans should be wrt to root_v
                        union_span = union_limits(root_v, u, *u_span, v, *v_span)
                        union_span_cache[sr_v] = union_span
                    proper_features_.append(
                        (
                            u,
                            v,
                            num_steps,
                            subroot,
                            sr_v,
                            u_is_subroot,
                            (v == sr_v),
                            load_self,
                            load_other,
                            is_delaunay_[uv_uniq],
                            extent,
                            uvD['cos_'][root_v],
                            uvD['blocked__'][root_v].count(),
                            union_span,
                        )
                    )
                # next two lines ensure that incoming links are counted
                num_feas_links += 1
                feas_unions.add(sr_v)
                extent_min = min(extent_min, extent)
        #  for uv_uniq in unfeas_links:
        #      pq.cancel(uv_uniq)
        #  if uv_uniq in pq.tags:
        #      pq.cancel(uv_uniq)
        #  else:
        #      print('attempt to cancel non-existent', F[uv_uniq[0]], F[uv_uniq[1]])
        #  print(f'[{i}] {link_caused_staleness}\n{whoneeds_[subroot]}\n{feas_unions}')
        #  assert link_caused_staleness == whoneeds_[subroot] - feas_unions, 'set mismatch'
        #  assert len(link_caused_staleness & fresh_subtrees) == 0, 'size mismatch'
        #  rel_component_excess = num_components/min_components - 1
        #  dropped_dependencies = whoneeds_[subroot] - feas_unions
        # all that can no longer link to subtree_[subroot] are stale

        stale_subtrees.update(link_caused_staleness - fresh_subtrees)
        fresh_subtrees.add(subroot)
        prev_cat_feas_unions = cat_feas_unions_[subroot]
        if not feas_unions:
            # this handles subtrees that became isolated
            S.add_edge(subroot, root)
            subtree_nodes = tuple(subtree_[subroot].search(_ONE))
            A.remove_nodes_from(subtree_nodes)
            debug('<refresh> subroot <%d> finalized (isolated)', subroot)
            is_feederless_[subroot] = False
            purge_log[iteration].append(subtree_nodes)
            steps_log[iteration].append((subroot, root))
            stale_subtrees.update(whoneeds_[subroot] - fresh_subtrees)
            if prev_cat_feas_unions >= 0:
                #  prio_tier_[prev_cat_feas_unions].remove(subroot)
                prio_tier_[prev_cat_feas_unions].discard(subroot)
            cat_feas_unions_[subroot] = -1
            return [], []
        # discard useless edges
        if unfeas_links:
            A.remove_edges_from(unfeas_links)
            purge_log[iteration].append(tuple(unfeas_links))
        whoneeds_[subroot] = feas_unions
        cat_feas_unions = UnionCount.encode(len(feas_unions))
        cat_feas_links = LinkCount.encode(num_feas_links)
        update_stales = False
        if cat_feas_unions != prev_cat_feas_unions:
            if prev_cat_feas_unions >= 0:
                #  prio_tier_[prev_cat_feas_unions].remove(subroot)
                prio_tier_[prev_cat_feas_unions].discard(subroot)
            prio_tier_[cat_feas_unions].add(subroot)
            cat_feas_unions_[subroot] = cat_feas_unions
            update_stales = True
        if cat_feas_links != cat_feas_links_[subroot]:
            cat_feas_links_[subroot] = cat_feas_links
            update_stales = True
        if extent_min != extent_min_[subroot]:
            extent_min_[subroot] = extent_min
            update_stales = True
        if update_stales:
            # since the current subtree had features changes, all that depend
            # on it must be marked as stale
            stale_subtrees.update(whoneeds_[subroot] - fresh_subtrees)
        return proper_links, proper_features_

    loop = True
    inbound_unions_ = [[] for _ in _T]
    links_to_appraise = []
    links_features = []
    link_groups = []
    # BEGIN: main loop
    while loop:
        debug('[%d]', iteration)
        if stale_subtrees:
            debug(
                'stale_subtrees (%d): %s',
                len(stale_subtrees),
                stale_subtrees,
            )
        links_to_appraise.clear()
        links_features.clear()
        link_groups.clear()
        while stale_subtrees:
            subroot = stale_subtrees.pop()
            proper_links, proper_features = refresh_subtree(subroot)
            #  print(subroot, proper_links)
            if proper_links:
                link_groups.append((subroot, len(proper_links)))
                links_to_appraise.extend(proper_links)
                links_features.extend(proper_features)
            else:
                top_outbound_[subroot] = _NULL_UNION

        #  print('LINK_GROUPS\n', link_groups)
        #  print('prio_tier\n', prio_tier_)
        # appraise and enqueue links
        if links_to_appraise:
            fresh_appraisals = appraiser.appraise(links_features)
            appraisal_log[iteration] = tuple(links_to_appraise), fresh_appraisals
            j = 0
            for sr_u, num_appraisals in link_groups:
                i, j = j, j + num_appraisals
                for appraisal, (uv_uniq, _, sr_v) in zip(
                    fresh_appraisals[i:j].tolist(), links_to_appraise[i:j]
                ):
                    heapq.heappush(
                        inbound_unions_[sr_v], (-appraisal, uv_uniq, sr_u, sr_v)
                    )
                # get best-appraised outbound link for sr_u
                argmax = fresh_appraisals[i:j].argmax() + i
                top_outbound_[sr_u] = (
                    fresh_appraisals[argmax].item(),
                    *links_to_appraise[argmax],
                )

        prio_union = _NULL_UNION
        for prio_tier in prio_tier_:
            if prio_tier:
                # get the best-appraised link from the highest-priority non-empty tier
                for sr in prio_tier:
                    if top_outbound_[sr] > prio_union:
                        prio_union = top_outbound_[sr]
                        #  if subroot_[prio_union[1][0]] not in prio_union[2:4] or subroot_[prio_union[1][1]] not in prio_union[2:4]:
                        #      print(f'ERROR: top_outbound subroot <{sr}> prio_tier: {prio_tier}')
                    inbound_heap = inbound_unions_[sr]
                    while inbound_heap and -inbound_heap[0][0] > prio_union[0]:
                        negappraisal, (u, v), sr_dropped, sr_kept = inbound_heap[0]
                        sr_u, sr_v = subroot_[u], subroot_[v]
                        # if sr_dropped or sr_kept is out of sync with subroot_, the entry is stale
                        if (u, v) in A.edges and (
                            (sr_u == sr_dropped and sr_v == sr_kept)
                            or (sr_v == sr_dropped and sr_u == sr_kept)
                        ):
                            prio_union = -negappraisal, (u, v), sr_dropped, sr_kept
                            break
                        # remove stale entry (u, v) no longer in A
                        heapq.heappop(inbound_heap)
                if prio_union[0] > threshold:
                    # tier's best union qualifies
                    break
        else:
            # no union appraisal was higher than threshold
            if prio_union[1] == -1:
                # finished
                break
        appraisal, uv_uniq, sr_dropped, sr_kept = prio_union
        prio_tier_[cat_feas_unions_[sr_dropped]].remove(sr_dropped)
        inbound_unions_[sr_dropped].clear()
        inbound_unions_[sr_kept].clear()
        #  debug('heap top loop-top: <%d>, «%s» %.3f', pq[0][-1], pq[0][-2], -pq[0][0])

        # TODO: reassess this hack
        if uv_uniq not in A.edges:
            stale_log[iteration] = uv_uniq
            debug('>>> popped link ⟨%s⟩ is not in A anymore <<<', uv_uniq)
            print(f'>>> link ⟨{uv_uniq}⟩ is not in A anymore <<<')
            #  prio_tier_[tier_id].discard(sr_dropped)
            continue
        # convert uv_uniq back to ⟨source, target⟩
        #  print(subroot_[uv_uniq[0]], sr_dropped, sr_kept)
        u, v = uv_uniq if subroot_[uv_uniq[0]] == sr_dropped else uv_uniq[::-1]
        debug(
            '<got> !%s! «%d~%d», dropped: <%d> kept: <%d>, appraisal: %.3f',
            uv_uniq,
            u,
            v,
            sr_dropped,
            sr_kept,
            appraisal,
        )

        root = A.nodes[sr_kept]['root']
        subtree = subtree_[sr_kept]

        # assess the union's angle span
        union_span_ = [
            union_limits(
                r, u, *subtree_span_[sr_dropped][r], v, *subtree_span_[sr_kept][r]
            )
            for r in roots
        ]
        debug(
            '<angle_span> //%s// dropped: %s; kept: %s',
            union_span_[root],
            subtree_span_[sr_dropped][root],
            subtree_span_[sr_kept][root],
        )

        # edge addition starts here
        debug('<add edge> «%d~%d» subroot <%d>', u, v, sr_kept)
        S.add_edge(u, v)
        num_steps += 1
        steps_log[iteration].append((u, v))

        if ((u, v) if u < v else (v, u)) not in diagonals:
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
                        if (u, s) in A:
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
                            if (u, t) in A:
                                A.remove_edge(u, t)

        is_feederless_[sr_dropped] = False
        # update the component's angle span
        subtree_span_[sr_kept] = union_span_
        # update the component's blocked set
        # TODO: handle multiple roots
        subtree_blocked_[sr_kept] |= (
            subtree_blocked_[sr_dropped] | A[u][v]['blocked__'][root]
        )
        subtree |= subtree_[sr_dropped]
        subtree_blocked_[sr_kept] &= ~subtree
        # update terminal->subroot mapping for sr_dropped's terminals
        for t in subtree_[sr_dropped].search(_ONE):
            subroot_[t] = sr_kept

        stale_subtrees.clear()
        # TODO: figure out why .discard() instead of .remove()
        #  whoneeds_[sr_kept].remove(sr_dropped)
        whoneeds_[sr_kept].discard(sr_dropped)
        stale_subtrees.update(whoneeds_[sr_kept], whoneeds_[sr_dropped])
        A.remove_edge(u, v)
        if subtree.count() == capacity:
            stale_subtrees.discard(sr_kept)
            S.add_edge(sr_kept, root)
            debug('subroot <%d> finalized (full load)', sr_kept)
            is_feederless_[sr_kept] = False
            steps_log[iteration].append((sr_kept, root))
            #  prio_tier_[cat_feas_unions_[sr_kept]].remove(sr_kept)
            prio_tier_[cat_feas_unions_[sr_kept]].discard(sr_kept)
            subtree_nodes = tuple(subtree.search(_ONE))
            A.remove_nodes_from(subtree_nodes)
            purge_log[iteration].append(subtree_nodes)
            for sr in whoneeds_[sr_dropped]:
                # TODO: rethink why not: whoneeds_[sr].remove(sr_dropped)
                whoneeds_[sr].discard(sr_dropped)
            whoneeds_[sr_dropped].clear()
            for sr in whoneeds_[sr_kept]:
                # TODO: rethink why not: whoneeds_[sr].remove(sr_dropped)
                whoneeds_[sr].discard(sr_kept)
            whoneeds_[sr_kept].clear()
            cat_feas_unions_[sr_kept] = -1
        else:
            purge_log[iteration].append(((u, v),))
            # this block might be unnecessary if whoneeds is not for dropped dependencies
            # TODO: rethink why not: whoneeds_[sr_dropped].remove(sr_kept)
            whoneeds_[sr_dropped].discard(sr_kept)
            whoneeds_[sr_kept].update(whoneeds_[sr_dropped])
            for sr in whoneeds_[sr_dropped]:
                # TODO: rethink why not: whoneeds_[sr].remove(sr_dropped)
                whoneeds_[sr].discard(sr_dropped)
                whoneeds_[sr].add(sr_kept)

        cat_feas_unions_[sr_dropped] = -1
        # remove from A and pq the edges that cross ⟨u, v⟩
        for s, t in edge_crossings(u, v, A, diagonals):
            A.remove_edge(s, t)
            purge_log[iteration].append(((s, t),))
            sr_s, sr_t = subroot_[s], subroot_[t]
            if cat_feas_unions_[sr_s] >= 0:
                stale_subtrees.add(sr_s)
            if cat_feas_unions_[sr_t] >= 0:
                stale_subtrees.add(sr_t)
        #  print('TOP_LINK\n', top_link_)

        #  if pq:
        #      debug('heap top loop-end: <%d>, «%s» %.3f', pq[0][-1], pq[0][-2], -pq[0][0])
        #  else:
        #      debug('heap EMPTY')
        iteration += 1
        #  print(iteration, end='.')
        if iteration == maxiter:
            error(
                'ERROR[data_driven_hybrid]: reached maximum number of iterations (%d)',
                iteration,
            )
            break
    # END: main loop

    # add missing feeders (possibly sub-capacity components)
    for sr in np.flatnonzero(is_feederless_):
        # TODO: check if the is_feederless mechanism is needed
        #       it is likely that any isolated sub-capacity subtree will be
        #       refreshed before the main loop exits
        debug('Adding sub-capacity subtree: subroot %d', sr)
        S.add_edge(sr, A.nodes[sr]['root'])

    debug('Final number of components: %d', sum(S.degree[r] for r in roots))
    calcload(S)
    # algorithm finished, store some info in the graph object
    S.graph.update(
        runtime=time.perf_counter() - start_time,
        capacity=capacity,
        creator='data_driven_hybrid',
        iterations=iteration,
        solver_details=dict(
            steps_log=steps_log,
            purge_log=purge_log,
            appraisal_log=appraisal_log,
            stale_log=stale_log,
        ),
    )
    return S
