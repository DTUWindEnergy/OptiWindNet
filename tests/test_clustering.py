"""Tests for :func:`optiwindnet.clustering.clusterize`.

The guarantee that must never break is that clustering costs no extra feeder::

    Σ_c ceil(T_c / capacity) == ceil(T / capacity)

Which terminals land where is then a trade-off against total terminal-to-root
distance, a proxy for cable length.
"""

import math
from itertools import product

import networkx as nx
import numpy as np
import pytest

import optiwindnet.clustering as clustering_mod
from optiwindnet.clustering import clusterize
from optiwindnet.mesh import make_planar_embedding

CAPACITIES = range(3, 31)


def _make_A(d2roots: np.ndarray) -> nx.Graph:
    """Bare graph carrying only what ``clusterize`` reads: R, T and d2roots."""
    T, R = d2roots.shape[0], d2roots.shape[1]
    return nx.Graph(R=R, T=T, d2roots=d2roots)


def _feeders(cluster_, capacity: int) -> int:
    return sum(math.ceil(len(cluster) / capacity) for cluster in cluster_)


def _check_partition(cluster_, A: nx.Graph, capacity: int) -> None:
    """Every terminal placed exactly once, and no feeder wasted."""
    R, T = A.graph['R'], A.graph['T']
    assert len(cluster_) == R
    assigned = [n for cluster in cluster_ for n in cluster]
    assert sorted(assigned) == list(range(T)), 'terminals lost or duplicated'
    assert _feeders(cluster_, capacity) == math.ceil(T / capacity), (
        f'clustering wasted feeders: {[len(c) for c in cluster_]} needs '
        f'{_feeders(cluster_, capacity)}, location needs {math.ceil(T / capacity)}'
    )


# --------------------------------------------------------------------------
# the whole bundled repository
# --------------------------------------------------------------------------
@pytest.fixture(scope='module')
def multiroot_instances(locations) -> dict[str, nx.Graph]:
    """Mesh every bundled location that has more than one root."""
    return {
        L.graph['handle']: make_planar_embedding(L)[1]
        for L in locations
        if L.graph['R'] > 1
    }


def test_repository_has_multiroot_instances(multiroot_instances):
    assert len(multiroot_instances) > 10


def test_feeder_invariant_over_repository(multiroot_instances):
    """The invariant must hold for every bundled multi-root location."""
    for handle, A in multiroot_instances.items():
        for capacity in CAPACITIES:
            cluster_ = clusterize(A, capacity)
            try:
                _check_partition(cluster_, A, capacity)
            except AssertionError as exc:
                raise AssertionError(f'{handle} at capacity={capacity}: {exc}') from exc


def test_no_root_drained_across_the_repository(multiroot_instances):
    """No bundled instance drains a root that terminals are closest to.

    The old rule did (moraywest at capacity=30 gave [60, 0]). A regression guard, not
    a general guarantee - see ``test_root_may_be_drained_when_feeders_are_scarce``.
    It holds here because the bundled locations all have enough feeders to go around.
    """
    for handle, A in multiroot_instances.items():
        R, T = A.graph['R'], A.graph['T']
        closest_root = A.graph['d2roots'][:T].argmin(axis=1)
        wanted = {c for c in range(R) if (closest_root == c).any()}
        for capacity in CAPACITIES:
            assert math.ceil(T / capacity) >= R, 'premise of this test broken'
            cluster_ = clusterize(A, capacity)
            drained = {c for c in wanted if not cluster_[c]}
            assert not drained, (
                f'{handle} at capacity={capacity}: root(s) {drained} were drained, '
                f'though terminals are closest to them; sizes='
                f'{[len(c) for c in cluster_]}'
            )


def test_root_may_be_drained_when_feeders_are_scarce():
    """With fewer feeders than roots, a root is emptied however close it is.

    ``T <= capacity`` leaves one feeder for two roots, so one goes without, even
    though terminal 0 is closest to root 0. The feeder count binds, not the geometry.
    """
    d2roots = np.array([[1.0, 2.0]] + [[9.0, 1.0]] * 9)
    A = _make_A(d2roots)
    assert d2roots.argmin(axis=1)[0] == 0, 'terminal 0 is closest to root 0'

    cluster_ = clusterize(A, capacity=10)

    _check_partition(cluster_, A, capacity=10)
    assert cluster_[0] == set()
    assert cluster_[1] == set(range(10))


# --------------------------------------------------------------------------
# the pathology that motivated the rewrite
# --------------------------------------------------------------------------
def test_two_balanced_roots_are_not_collapsed():
    """Two roots with 9 tightly-grouped terminals each, capacity 10 -> [9, 9].

    Both clusters carry a partly filled feeder, which is right: the 18 remainder
    terminals need ``ceil(18 / 10) == 2`` feeders and the clusters hold exactly 2.
    Confining the slack to a single cluster (the rule this replaced) forbids this
    partition and drags terminals to the far root, 100x more distant here.
    """
    near, far = 1.0, 100.0
    d2roots = np.array([[near, far]] * 9 + [[far, near]] * 9)
    A = _make_A(d2roots)

    cluster_ = clusterize(A, capacity=10)

    _check_partition(cluster_, A, capacity=10)
    assert [len(c) for c in cluster_] == [9, 9]
    assert cluster_[0] == set(range(9))
    assert cluster_[1] == set(range(9, 18))


def test_empty_cluster_when_voronoi_cell_is_empty():
    """A root no terminal is closest to gets nothing, legitimately.

    The yunlin case: its second substation is farther from every terminal than the
    first, so no partition populates it for free. The callers must tolerate that.
    """
    d2roots = np.array([[50.0, 1.0]] * 8)
    A = _make_A(d2roots)

    cluster_ = clusterize(A, capacity=4)

    _check_partition(cluster_, A, capacity=4)
    assert cluster_[0] == set()
    assert cluster_[1] == set(range(8))


def test_voronoi_kept_when_it_wastes_no_feeder():
    """A closest-root partition that wastes no feeder is returned as is.

    It is then the unconstrained optimum of the proxy, so nothing can improve it.
    """
    rng = np.random.default_rng(0)
    d2roots = rng.uniform(1.0, 10.0, size=(12, 2))
    A = _make_A(d2roots)
    closest_root = d2roots.argmin(axis=1)
    voronoi = [set(np.flatnonzero(closest_root == c).tolist()) for c in range(2)]
    capacity = 2  # 12 terminals, 6 feeders: any split of an even count is round
    assume = _feeders(voronoi, capacity) == math.ceil(12 / capacity)

    cluster_ = clusterize(A, capacity)

    if assume:
        assert cluster_ == voronoi


# --------------------------------------------------------------------------
# optimality of the proxy
# --------------------------------------------------------------------------
def _brute_force_optimum(d2roots: np.ndarray, capacity: int) -> float:
    """Cheapest sum(d2roots) over every feeder-minimal partition, by exhaustion."""
    T, R = d2roots.shape
    feeders_min = math.ceil(T / capacity)
    best = math.inf
    for assignment in product(range(R), repeat=T):
        sizes = [assignment.count(c) for c in range(R)]
        if sum(math.ceil(s / capacity) for s in sizes) != feeders_min:
            continue
        best = min(best, sum(d2roots[n, c] for n, c in enumerate(assignment)))
    return best


@pytest.mark.parametrize('seed', range(8))
@pytest.mark.parametrize('capacity', (3, 4, 5))
def test_matches_brute_force_optimum(seed, capacity):
    """On instances small enough to exhaust, clusterize hits the proxy optimum.

    Hence an exact min-cost-flow or MILP formulation would only spend far more
    compute to reach the same number.
    """
    rng = np.random.default_rng(seed)
    d2roots = rng.uniform(1.0, 20.0, size=(10, 2))
    A = _make_A(d2roots)

    cluster_ = clusterize(A, capacity)

    _check_partition(cluster_, A, capacity)
    ours = sum(d2roots[n, c] for c, cluster in enumerate(cluster_) for n in cluster)
    assert ours == pytest.approx(_brute_force_optimum(d2roots, capacity))


# --------------------------------------------------------------------------
# the many-roots fallback
# --------------------------------------------------------------------------
def test_invariant_holds_on_the_many_roots_fallback(monkeypatch):
    """With too many budgets to enumerate, the one-at-a-time path still holds up."""
    monkeypatch.setattr(clustering_mod, '_MAX_BUDGETS', 0)
    rng = np.random.default_rng(7)
    d2roots = rng.uniform(1.0, 30.0, size=(47, 4))
    A = _make_A(d2roots)

    for capacity in CAPACITIES:
        cluster_ = clusterize(A, capacity)
        _check_partition(cluster_, A, capacity)
