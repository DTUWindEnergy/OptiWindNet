# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Consistency checks for solution topologies and routed solutions."""

import networkx as nx
import pytest

from optiwindnet.converting import (
    S_from_terse_links,
    terse_links_from_S,
)
from optiwindnet.loads import (
    calcload,
)
from optiwindnet.MILP import Topology
from optiwindnet.validating import validate_topology

from .helpers import ringed_S, tiny_wfn


# --------------------------------------------------------------------------- #
# validate_topology
# --------------------------------------------------------------------------- #
def _radial_S(T, links):
    """A radial S over ``links``, with loads and orientation from calcload."""
    S = nx.Graph(R=1, T=T, topology=Topology.RADIAL)
    S.add_edges_from(links)
    calcload(S)
    return S


def test_validate_topology_accepts_valid_topologies():
    assert validate_topology(tiny_wfn().S) == []
    assert validate_topology(ringed_S(1, [(-1, [0, 1, 2, 3]), (-1, [4, 5])])) == []


def test_validate_topology_accepts_string_declaration():
    S = ringed_S(1, [(-1, [0, 1, 2])])
    S.graph['topology'] = 'ringed'

    assert validate_topology(S) == []


def test_validate_topology_requires_loads():
    """Every producer sets loads, so a topology without them is unfinished.

    The shape checks read the loads, so tolerating their absence would silently
    skip most of them rather than validate a leaner graph.
    """
    S = _radial_S(3, [(-1, 0), (0, 1), (1, 2)])
    S.graph['has_loads'] = False

    assert validate_topology(S) == ['topology carries no loads']


def test_validate_topology_checks_shape_without_loads():
    S = nx.Graph(R=1, T=3, topology=Topology.RADIAL, has_loads=False)
    S.add_edges_from([(-1, 0), (0, 1), (1, 2), (0, 2)])

    violations = validate_topology(S)

    assert 'topology carries no loads' in violations
    assert 'radial topology must be a forest' in violations


def test_validate_topology_rejects_forest_without_reverse():
    """The positional forest encoding reads link orientation off ``'reverse'``."""
    S = _radial_S(3, [(-1, 0), (0, 1), (1, 2)])
    for _, _, edgeD in S.edges(data=True):
        del edgeD['reverse']

    violations = validate_topology(S)
    assert len(violations) == 1
    assert 'missing the "reverse" flag' in violations[0]


def test_validate_topology_rejects_stranded_terminal():
    """A terminal reaching no root is a broken solution in its own right.

    A stranded terminal sits in its own component without growing a cycle, so
    neither the forest check nor the simple-path one sees it.
    """
    S = nx.Graph(R=1, T=4, topology=Topology.RADIAL, has_loads=True, max_load=3)
    S.add_edge(-1, 0, load=3, reverse=False)
    S.add_edge(0, 1, load=2, reverse=False)
    S.add_edge(1, 2, load=1, reverse=False)
    S.add_node(3)
    S.nodes[-1]['load'] = 3
    for terminal, load in ((0, 3), (1, 2), (2, 1)):
        S.nodes[terminal]['load'] = load
    assert nx.is_forest(S)

    assert 'terminals not connected to any root: [3]' in validate_topology(S)


def test_validate_topology_rejects_terminal_absent_from_S():
    """A terminal missing from ``S`` altogether is stranded just the same.

    Asking for its degree used to raise instead of reporting.
    """
    S = nx.Graph(R=1, T=4, topology=Topology.RADIAL, has_loads=True, max_load=3)
    S.add_edge(-1, 0, load=3, reverse=False)
    S.add_edge(0, 1, load=2, reverse=False)
    S.add_edge(1, 2, load=1, reverse=False)
    S.nodes[-1]['load'] = 3
    for terminal, load in ((0, 3), (1, 2), (2, 1)):
        S.nodes[terminal]['load'] = load

    assert 'terminals not connected to any root: [3]' in validate_topology(S)


def test_validate_topology_rejects_misplaced_zero_load_link():
    """A ring's zero-load link must sit where its node loads split the arms.

    Moving it while leaving every node load alone keeps the zero-load-link count,
    the arm balance and the arm-head totals all intact, so only comparing the
    link loads against the node loads catches it.
    """
    S = ringed_S(1, [(-1, [0, 1, 2, 3])])
    assert S[1][2]['load'] == 0
    S[0][1]['load'], S[1][2]['load'] = 0, 1

    violations = validate_topology(S)
    assert any('opens between 0 and 1' in violation for violation in violations)
    assert any('states load' in violation for violation in violations)


def test_validate_topology_rejects_a_component_with_two_roots():
    S = nx.Graph(R=2, T=2, topology=Topology.BRANCHED, has_loads=False)
    S.add_edges_from([(-2, 0), (0, 1), (1, -1)])

    assert 'component contains multiple roots: [-2, -1]' in validate_topology(S)


def test_validate_topology_reports_every_shape_violation():
    S = nx.Graph(R=1, T=3, topology=Topology.RADIAL, has_loads=True, max_load=3)
    S.add_edges_from([(-1, 0), (0, 1), (0, 2), (1, 2)], load=1, reverse=False)
    S.nodes[-1]['load'] = 3
    for terminal in range(3):
        S.nodes[terminal]['load'] = 1

    violations = validate_topology(S)
    assert 'radial topology must be a forest' in violations
    assert 'radial subtrees must be simple paths' in violations


# --------------------------------------------------------------------------- #
# A topology that validates is representable: it survives its own encoding.
# The validity rules live in validate_topology; this confirms they are sufficient.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    'build',
    [
        pytest.param(lambda: tiny_wfn().S, id='radial'),
        pytest.param(lambda: ringed_S(1, [(-1, [0, 1, 2, 3])]), id='ring-even'),
        pytest.param(lambda: ringed_S(1, [(-1, [0, 1, 2])]), id='ring-odd'),
        pytest.param(lambda: ringed_S(1, [(-1, [0, 1]), (-1, [2, 3])]), id='rings'),
        pytest.param(
            lambda: ringed_S(2, [(-1, [0, 1, 2]), (-2, [3, 4])]), id='multi-root'
        ),
        pytest.param(lambda: ringed_S(1, [(-1, [0, 1, 2]), (-1, [3])]), id='stub'),
    ],
)
def test_validated_topology_round_trips_through_terse_links(build):
    S = build()
    assert validate_topology(S) == []

    terse = terse_links_from_S(S)
    S_rt = S_from_terse_links(terse, R=S.graph['R'], T=S.graph['T'])

    assert {frozenset(link) for link in S_rt.edges()} == {
        frozenset(link) for link in S.edges()
    }
    assert all(S_rt[u][v]['load'] == d['load'] for u, v, d in S.edges(data=True))
