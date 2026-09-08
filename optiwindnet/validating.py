# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Consistency checks for solution topologies and routed solutions."""

import math
from itertools import pairwise

import networkx as nx

from .converting import S_from_G, _rings_from_S
from .loads import calcload
from .types import Topology

__all__ = ('validate_routeset', 'validate_topology')


def _validate_ringed(
    S: nx.Graph, capacity: int | None, *, check_loads: bool
) -> list[str]:
    """Return RINGED shape, capacity and load violations in ``S``."""
    violations = []
    T = S.graph['T']

    # topology-graph ring edges are pure geometry: they carry no edge 'kind'
    kinds = {d.get('kind') for _, _, d in S.edges(data=True)} - {None}
    if kinds:
        violations.append(
            f'ring edges must not carry a kind, got {sorted(map(str, kinds))}'
        )

    rings = _rings_from_S(S)

    # the rings partition the terminal set: every terminal in exactly one ring
    covered = sorted(t for _, ordered in rings for t in ordered)
    if covered != list(range(T)):
        violations.append('rings must partition the terminals')

    # two feeders per ring, except a lone terminal hanging off a single root
    n_feeders = sum(1 for u, v in S.edges if u < 0 or v < 0)
    expected = sum(
        1 if len(ordered) == 1 and rs[0] == rs[1] else 2 for rs, ordered in rings
    )
    if n_feeders != expected:
        violations.append(f'expected {expected} feeders, got {n_feeders}')

    for roots, ordered in rings:
        n = len(ordered)
        arm = math.ceil(n / 2)
        # a ring holds up to 2*capacity terminals (two arms of ceil(n / 2))
        if capacity is not None and arm > capacity:
            violations.append(f'ring at {roots} needs arms of {arm} > κ = {capacity}')
        if not check_loads:
            continue
        # the heaviest node of a ring carries a full arm: the arms are balanced
        heaviest = max(S.nodes[t]['load'] for t in ordered)
        if heaviest != arm:
            violations.append(
                f'ring at {roots} has unbalanced arms: heaviest node carries '
                f'{heaviest}, balanced arms would carry {arm}'
            )
        # the two arm-head (feeder) terminals carry the whole ring between them
        heads = S.nodes[ordered[0]]['load'] + (
            S.nodes[ordered[-1]]['load'] if n > 1 else 0
        )
        if heads != n:
            violations.append(
                f'ring at {roots} spans {n} terminals, but its arm heads carry {heads}'
            )
        # exactly one zero-load link per ring: a real cable with no current through it
        opens = [
            i
            for i, (u, v) in enumerate(pairwise(ordered))
            if S.has_edge(u, v) and S[u][v]['load'] == 0
        ]
        if len(opens) != (1 if n > 1 else 0):
            violations.append(
                f'ring at {roots} has {len(opens)} zero-load links, expected '
                f'{1 if n > 1 else 0}'
            )
        elif opens:
            # the zero-load link splits the ring where the node loads say it does:
            # arm 1 takes `arm` terminals, and an odd-terminal ring has a second
            # balanced split one link earlier (see :func:`add_ring_to_S`). Both
            # walk directions are covered: the two indices map onto each other.
            balanced = {arm - 1} if n % 2 == 0 else {arm - 1, arm - 2}
            if opens[0] not in balanced:
                u, v = ordered[opens[0]], ordered[opens[0] + 1]
                violations.append(
                    f'ring at {roots} opens between {u} and {v}, which is not '
                    f'where its node loads split the arms'
                )
    return violations


def validate_topology(S: nx.Graph, capacity: int | None = None) -> list[str]:
    """Check the nodes, links, loads and declared topology of ``S``.

    The declared topology determines the shape checks: RINGED topologies are
    checked as terminal rings, RADIAL topologies as simple rooted paths, and
    BRANCHED topologies as rooted trees. All topology types are checked for the
    expected roots and terminals, complete load and orientation attributes,
    consistent calculated loads and cable capacity.

    Args:
      S: topology graph to check. ``S.graph['topology']`` is mandatory: it is a
        :class:`~optiwindnet.types.Topology` member (or its ``str`` value). Loads are
        mandatory too -- ``S`` without them is reported as a violation, while
        structural checks that do not need them still run.
      capacity: cable capacity; defaults to ``S.graph['capacity']``. Capacity
        checks are skipped when neither is available.

    Returns:
      list of human-readable violations; ``S`` is valid if it is empty.

    Example::

      violations = validate_topology(S, capacity)
      if violations:
          print('\\n'.join(violations))

    """
    violations = []
    R, T = S.graph['R'], S.graph['T']
    if capacity is None:
        capacity = S.graph.get('capacity')
    try:
        topology = Topology(S.graph['topology'])
    except (TypeError, ValueError):
        return [f'unknown topology: {S.graph["topology"]!r}']

    # --- checks required for every topology ----------------------------------
    roots = set(range(-R, 0))
    terminals = set(range(T))
    expected_nodes = roots | terminals
    missing = sorted(expected_nodes - S.nodes)
    unexpected = sorted(S.nodes - expected_nodes)
    if missing:
        violations.append(f'topology nodes missing: {missing}')
    if unexpected:
        violations.append(f'unexpected topology nodes: {unexpected}')

    has_loads = bool(S.graph.get('has_loads'))
    loads_present = all('load' in data for _, data in S.nodes(data=True)) and all(
        'load' in data for _, _, data in S.edges(data=True)
    )
    unoriented = sorted(
        (u, v) for u, v, data in S.edges(data=True) if 'reverse' not in data
    )
    loads_complete = has_loads and not missing and loads_present and not unoriented
    if not has_loads:
        violations.append('topology carries no loads')
    elif not loads_present:
        violations.append('topology has incomplete load attributes')
    if unoriented:
        violations.append(f'links missing the "reverse" flag: {unoriented}')

    if loads_complete:
        reference = S.copy()
        try:
            calcload(reference)
        except ValueError as exc:
            violations.append(str(exc))
        else:
            violations += _load_mismatches(S, reference)
            if capacity is not None and reference.graph['max_load'] > capacity:
                violations.append(
                    f'κ = {capacity}, max_load = {reference.graph["max_load"]}'
                )

    # --- topology shape ------------------------------------------------------
    if topology is Topology.RINGED:
        bad_degrees = sorted(
            (t, S.degree(t))
            for t in terminals & S.nodes
            if S.degree(t) != 2 and not (S.degree(t) == 1 and next(iter(S[t])) in roots)
        )
        if bad_degrees:
            violations.append(f'invalid ring terminal degrees: {bad_degrees}')
        else:
            violations += _validate_ringed(S, capacity, check_loads=loads_complete)
    elif topology in (Topology.RADIAL, Topology.BRANCHED):
        if not nx.is_forest(S):
            violations.append(f'{topology} topology must be a forest')
        # every terminal is served by some root: a forest may leave a terminal
        # stranded in a component of its own without ever growing a cycle
        served = set()
        for component in nx.connected_components(S):
            component_roots = roots.intersection(component)
            if component_roots:
                served |= component
            if len(component_roots) > 1:
                violations.append(
                    f'component contains multiple roots: {sorted(component_roots)}'
                )
        # the subtraction also covers terminals absent from S altogether
        stranded = sorted(terminals - served)
        if stranded:
            violations.append(f'terminals not connected to any root: {stranded}')
        if topology is Topology.RADIAL and any(
            S.degree[t] > 2 for t in terminals if t in S
        ):
            violations.append('radial subtrees must be simple paths')

    return violations


def _load_mismatches(G: nx.Graph, Gʹ: nx.Graph) -> list[str]:
    """Return load and orientation attributes in ``G`` that differ from ``Gʹ``."""
    violations = []
    if G.graph.get('max_load') != Gʹ.graph['max_load']:
        violations.append(
            f'max_load is {G.graph.get("max_load")}, links carry {Gʹ.graph["max_load"]}'
        )
    for u, v, edgeD in G.edges(data=True):
        refD = Gʹ[u][v]
        if edgeD.get('load') != refD['load']:
            violations.append(
                f'link {u}–{v} states load {edgeD.get("load")}, carries {refD["load"]}'
            )
        # 'reverse' orients the link for terse_links and the flow formulations
        if edgeD.get('reverse') != refD['reverse']:
            violations.append(
                f'link {u}–{v} states reverse={edgeD.get("reverse")}, flows '
                f'reverse={refD["reverse"]}'
            )
    for node, nodeD in G.nodes(data=True):
        refD = Gʹ.nodes[node]
        if nodeD.get('load') != refD.get('load'):
            violations.append(
                f'node {node} states load {nodeD.get("load")}, carries '
                f'{refD.get("load")}'
            )
    return violations


def validate_routeset(G: nx.Graph) -> list[str]:
    """Check the electrical, topological and geometric validity of routeset ``G``.

    Stored loads and orientations are compared with values calculated on a
    copy. The routeset is reduced to a solution topology and passed to
    :func:`validate_topology`. Complete route polylines are checked for
    crossings, overlaps, branch splits, self-intersections and degenerate
    geometry. Geometric checks run independently of load and topology errors,
    and ``G`` is not modified.

    Args:
      G: routeset graph to evaluate.

    Returns:
      list of human-readable violations; ``G`` is valid if it is empty.

    Example::

      violations = validate_routeset(G)
      if violations:
          print('\\n'.join(violations))

    """
    # deferred: this orchestrator is the only part of interarraylib that needs
    # the crossings machinery, which the other importers of this module do not
    from .crossings import find_geometric_crossings

    violations = []
    reference = G.copy()
    if not G.graph.get('has_loads'):
        violations.append('routeset carries no loads')
    try:
        calcload(reference)
    except ValueError as exc:
        violations.append(str(exc))
        topology_source = G
    else:
        topology_source = reference
        if G.graph.get('has_loads'):
            violations += _load_mismatches(G, reference)

    try:
        S = S_from_G(topology_source)
    except (KeyError, ValueError) as exc:
        violations.append(f'routes cannot be reduced to a topology: {exc}')
    else:
        violations += validate_topology(S, G.graph.get('capacity'))

    for finding in find_geometric_crossings(G):
        if finding['kind'] == 'branch_split':
            violations.append(
                f'route {finding["path_a"]} splits the branch '
                f'{finding["path_b"]} at {finding["geometry"]}'
            )
        elif finding['kind'] == 'degenerate':
            violations.append(
                f'route {finding["path_a"]} has degenerate geometry '
                f'at {finding["geometry"]}'
            )
        else:
            violations.append(
                f'route {finding["path_a"]} crosses route {finding["path_b"]} '
                f'({finding["kind"]}) at {finding["geometry"]}'
            )
    return violations
