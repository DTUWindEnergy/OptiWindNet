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

from optiwindnet.crossings import validate_routeset
from optiwindnet.heuristics import constructor
from optiwindnet.importer import L_from_yaml
from optiwindnet.interarraylib import G_from_S
from optiwindnet.mesh import make_planar_embedding
from optiwindnet.pathfinding import PathFinder

from . import paths
from .helpers import terminal_terminal_crossings

METHODS = ('esau_williams', 'biased_EW', 'rootlust', 'radial_EW')
BRANCHED_METHODS = ('esau_williams', 'biased_EW', 'rootlust')
CAPACITIES = (3, 5, 8)

# Locations drawn from the shipped optiwindnet/data set: one single-root and one
# multi-root case (the latter exercises the per-root loops over `roots`,
# `rootmask__` and `is_root_nb__`). Turbine count is irrelevant: the heuristic
# is fast even at T~100.
_LOC_FILES = {
    'cazzaro_1ss': 'Cazzaro-2022.yaml',  # R=1, T=50
    'moray_3ss': 'Moray East.yaml',  # R=3, T=100
}


@pytest.fixture(params=list(_LOC_FILES.values()), ids=list(_LOC_FILES), scope='session')
def mesh(request):
    """Planar embedding + available-links graph (P, A) for a data-set location.

    Built once per location per session (the mesh is the only slow part).
    """
    L = L_from_yaml(paths.DATA_DIR / request.param)
    return make_planar_embedding(L)


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


def _route_end_to_end(S, P, A):
    """Mirror EWRouter.route: turn S into a detoured routeset G.

    Feeder rerouting must respect the topology S declares, or PathFinder can
    hand back a routeset that no longer has the shape it solved for.
    """
    topology = S.graph['topology']
    return PathFinder(
        G_from_S(S, A),
        planar=P,
        A=A,
        branched=topology == 'branched',
        ringed=topology == 'ringed',
    ).create_detours()


# --------------------------------------------------------------------------- #
# A. structural / capacity invariants on raw S
# --------------------------------------------------------------------------- #
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
# B. end-to-end validity (constructor -> PathFinder -> validate_routeset)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('method', METHODS)
def test_end_to_end_routeset_is_valid(mesh, method):
    P, A = mesh
    S = constructor(A, capacity=5, method=method)
    G = _route_end_to_end(S, P, A)
    assert validate_routeset(G) == []


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
    P, A = mesh
    S = constructor(A, capacity=capacity, method='rootlust')
    assert nx.is_forest(S)
    assert S.graph['max_load'] <= capacity
    G = _route_end_to_end(S, P, A)
    assert validate_routeset(G) == []


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
    P, A = mesh
    S = constructor(
        A, capacity=5, method=method, weigh_detours=False, straight_feeder_route=True
    )
    G = _route_end_to_end(S, P, A)
    assert validate_routeset(G) == []


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
