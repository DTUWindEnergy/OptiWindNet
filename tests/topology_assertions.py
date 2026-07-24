"""Shared assertions for undetoured producer topology output."""

import networkx as nx

from optiwindnet.interarraylib import rings_from_S, validate_topology
from optiwindnet.types import Topology


def assert_topology(S: nx.Graph, expected: Topology, capacity: int) -> None:
    """Validate the common topology contract without invoking PathFinder."""
    expected = Topology(expected)
    R, T = (S.graph[key] for key in 'RT')
    assert Topology(S.graph['topology']) is expected
    assert S.graph['capacity'] == capacity
    assert S.graph['has_loads']
    assert validate_topology(S, capacity) == []
    assert set(range(T)) <= S.nodes
    assert set(range(-R, 0)) <= S.nodes
    assert sum(S.nodes[root]['load'] for root in range(-R, 0)) == T
    assert S.graph['max_load'] <= capacity
    assert {'creator', 'R', 'T', 'capacity', 'topology', 'has_loads'} <= S.graph.keys()

    if expected is Topology.RADIAL:
        assert nx.is_forest(S)
        assert all(S.degree(terminal) <= 2 for terminal in range(T))
    elif expected is Topology.BRANCHED:
        assert nx.is_forest(S)
    else:
        rings = rings_from_S(S)
        assert rings
        roots_used = {root for roots, _ in rings for root in roots}
        assert roots_used <= set(range(-R, 0))
