"""Tests for the unified constructive heuristic `optiwindnet.heuristics.constructor`.

`constructor` replaced several near-duplicate heuristic modules. It exposes four
methods ('esau_williams', 'biased_EW', 'rootlust', 'radial_EW') and the flags
`weigh_detours` and `straight_feeder_route`. The production routers only ever
reach `biased_EW`, so the other methods/flags need direct coverage here.

Strategy (per the agreed plan): structural/capacity *invariants* on the raw
solution topology `S` for every method, plus a few end-to-end checks running the
realistic `constructor -> G_from_S -> PathFinder` pipeline and asserting
`validate_routeset(G) == []`. Invariants are preferred over golden snapshots so
the suite survives ongoing heuristic retuning.
"""

import logging

import networkx as nx
import pytest

from optiwindnet.heuristics import constructor
from optiwindnet.interarraylib import rings_from_S
from optiwindnet.types import Topology

from .cases import (
    CONSTRUCTOR_CASES,
    case_node_id,
    expected_topology,
    topology_golden_key,
)
from .helpers import terminal_terminal_crossings
from .sitecache import get_bundle
from .solver_topologies import assert_matches_golden, load_solver_topologies
from .topology_assertions import assert_topology

METHODS = ('esau_williams', 'biased_EW', 'rootlust', 'radial_EW')
BRANCHED_METHODS = ('esau_williams', 'biased_EW', 'rootlust')
CAPACITIES = (3, 5, 8)

# Locations drawn from the shipped optiwindnet/data set: one single-root and one
# multi-root case (the latter exercises the per-root loops over `roots`,
# `rootmask__` and `is_root_nb__`). Turbine count is irrelevant: the heuristic
# is fast even at T~100.
_LOCATIONS = {
    'cazzaro_2022_1ss': 'cazzaro_2022',
    'morayeast_3ss': 'morayeast',
}
_SOLVER_GOLDENS = load_solver_topologies()


@pytest.fixture(params=list(_LOCATIONS.values()), ids=list(_LOCATIONS), scope='session')
def mesh(request):
    """Planar embedding + available-links graph (P, A) for a data-set location.

    Built once per location per session (the mesh is the only slow part).
    """
    bundle = get_bundle(request.param)
    return bundle.P, bundle.A


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _each_terminal_reaches_exactly_one_root(S):
    R, T = S.graph['R'], S.graph['T']
    roots = range(-R, 0)
    for t in range(T):
        if sum(nx.has_path(S, t, r) for r in roots) != 1:
            return False
    return True


# --------------------------------------------------------------------------- #
# A. structural / capacity invariants on raw S
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('case', CONSTRUCTOR_CASES, ids=case_node_id)
def test_constructor_topology_cases(case):
    """The broad constructor matrix validates undetoured topology output."""
    A = get_bundle(case.site).A
    S = constructor(
        A,
        capacity=case.capacity,
        method=case.method,
        bias_margin=case.bias_margin,
        weigh_detours=case.feeder_route.value == 'segmented',
        straight_feeder_route=case.feeder_route.value == 'straight',
    )
    assert_topology(S, expected_topology(case), case.capacity)
    assert terminal_terminal_crossings(S, A.graph['VertexC']) == []
    if case.exact_golden:
        assert_matches_golden(S, _SOLVER_GOLDENS[topology_golden_key(case)])


@pytest.mark.parametrize('capacity', (3, 5, 8))
def test_constructor_ringed_capacity_sweep(capacity):
    A = get_bundle('albatros').A
    S = constructor(A, capacity=capacity, method='ringed')
    assert_topology(S, Topology.RINGED, capacity)
    assert all(len(terminals) <= 2 * capacity for _, terminals in rings_from_S(S))


def test_constructor_ringed_multi_root_uses_every_root():
    A = get_bundle('neart').A
    S = constructor(A, capacity=5, method='ringed')
    assert_topology(S, Topology.RINGED, 5)
    roots_used = {root for roots, _ in rings_from_S(S) for root in roots}
    assert roots_used == set(range(-A.graph['R'], 0))


@pytest.mark.parametrize('bias_margin', (0.0, 0.1, 0.5))
def test_constructor_ringed_bias_margin(bias_margin):
    A = get_bundle('albatros').A
    S = constructor(A, capacity=5, method='ringed', bias_margin=bias_margin)
    assert_topology(S, Topology.RINGED, 5)


def test_constructor_ringed_exact_double_capacity_boundary():
    A = get_bundle('example_location').A
    S = constructor(A, capacity=6, method='ringed')
    assert_topology(S, Topology.RINGED, 6)
    assert max(map(lambda ring: len(ring[1]), rings_from_S(S))) <= 12


def test_constructor_ringed_odd_site_exercises_single_terminal_ring():
    A = get_bundle('london').A
    S = constructor(A, capacity=1, method='ringed')
    assert_topology(S, Topology.RINGED, 1)
    assert any(len(terminals) == 1 for _, terminals in rings_from_S(S))


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('capacity', CAPACITIES)
def test_output_is_capacitated_forest(mesh, method, capacity):
    _, A = mesh
    T = A.graph['T']
    S = constructor(A, capacity=capacity, method=method)

    # spanning forest rooted at the substations: one parent edge per terminal
    assert nx.is_forest(S)
    assert S.number_of_edges() == T
    assert _each_terminal_reaches_exactly_one_root(S)

    # capacity respected (constructor calls calcload internally)
    assert 'max_load' in S.graph
    assert S.graph['max_load'] <= capacity


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('capacity', CAPACITIES)
def test_no_terminal_terminal_crossings(mesh, method, capacity):
    _, A = mesh
    S = constructor(A, capacity=capacity, method=method)
    crossings = terminal_terminal_crossings(S, A.graph['VertexC'])
    assert crossings == [], (
        f'{method} produced terminal-terminal crossings: {crossings}'
    )


@pytest.mark.parametrize('method', METHODS)
def test_graph_metadata(mesh, method):
    _, A = mesh
    S = constructor(A, capacity=5, method=method)
    assert S.graph['creator'] == 'constructor'
    assert S.graph['capacity'] == 5
    assert S.graph['iterations'] >= 1
    assert S.graph['method_options']['method'] == method


# --------------------------------------------------------------------------- #
# C. method-specific behaviour
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('capacity', (3, 4, 5, 6, 7))
def test_radial_produces_simple_paths(mesh, capacity):
    """radial_EW subtrees are simple paths: every terminal has degree <= 2.

    Sweeping capacities here also reliably exercises the radial insertion and
    subroot-reassignment machinery (path extension that flips the subroot).
    """
    _, A = mesh
    T = A.graph['T']
    S = constructor(A, capacity=capacity, method='radial_EW')
    assert all(S.degree(t) <= 2 for t in range(T))
    assert 'num_insertions' in S.graph


@pytest.mark.parametrize('method', BRANCHED_METHODS)
def test_branched_methods_omit_num_insertions(mesh, method):
    _, A = mesh
    S = constructor(A, capacity=5, method=method)
    assert 'num_insertions' not in S.graph


@pytest.mark.parametrize('capacity', CAPACITIES)
def test_rootlust_capacity_sweep_is_valid(mesh, capacity):
    """rootlust over several capacities exercises its triangle-swap fix path."""
    _, A = mesh
    S = constructor(A, capacity=capacity, method='rootlust')
    assert_topology(S, Topology.BRANCHED, capacity)


def test_custom_rootlust_param(mesh):
    """An explicit `rootlust_` tuple is accepted and yields a valid topology."""
    _, A = mesh
    S = constructor(A, capacity=5, method='rootlust', rootlust_=(0.1, 0.5))
    assert nx.is_forest(S)
    assert S.graph['max_load'] <= 5


# --------------------------------------------------------------------------- #
# D. flags and error handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('method', ('biased_EW', 'rootlust'))
def test_straight_feeder_route_is_valid(mesh, method):
    _, A = mesh
    S = constructor(
        A, capacity=5, method=method, weigh_detours=False, straight_feeder_route=True
    )
    assert_topology(S, Topology.BRANCHED, 5)


def test_straight_feeder_route_with_weigh_detours_warns(mesh, caplog):
    _, A = mesh
    with caplog.at_level(logging.WARNING, logger='optiwindnet.heuristics.constructor'):
        constructor(
            A,
            capacity=5,
            method='biased_EW',
            weigh_detours=True,
            straight_feeder_route=True,
        )
    assert any('weigh_detours' in r.getMessage() for r in caplog.records)


def test_unsupported_method_raises(mesh):
    _, A = mesh
    with pytest.raises(ValueError, match='Unsupported constructor method'):
        constructor(A, capacity=5, method='not_a_method')


def test_maxiter_failsafe_breaks_and_logs(mesh, caplog):
    """The `maxiter` fail-safe stops the loop, logs an error and still returns
    a (partially built) forest rather than hanging."""
    _, A = mesh
    with caplog.at_level(logging.ERROR, logger='optiwindnet.heuristics.constructor'):
        S = constructor(A, capacity=5, method='biased_EW', maxiter=1)
    assert any('maxiter' in r.getMessage() for r in caplog.records)
    assert nx.is_forest(S)
    assert S.number_of_edges() == A.graph['T']


def test_debug_logging_reports_heap_state(caplog):
    A = get_bundle('example_location').A
    with caplog.at_level(logging.DEBUG, logger='optiwindnet.heuristics.constructor'):
        constructor(A, capacity=3, method='rootlust')
    assert 'heap' in caplog.text
