"""Validation tests for the RINGED topology.

The RINGED topology is now available across every solver family:

* the shared ring builders :func:`add_ring_to_S`, :func:`ringify_S` and
  :func:`rings_from_links` (canonical two-arm-with-split shape) that every
  backend reconstructs its solution through;
* the constructive heuristic ``method='ringed'`` (and its public
  :class:`~optiwindnet.api.EWRouter` wrapper);
* the MILP RINGED path (``ModelOptions(topology='ringed')``), both at the
  low level and through :class:`~optiwindnet.api.MILPRouter`;
* the HGS-CVRP and LKH-3 closed-CVRP paths (``ringed=True``).

Because every backend canonicalises its output through :func:`add_ring_to_S`
(the MILP path in ``MILP/_core.py:get_solution``, HGS/LKH/constructor through
:func:`ringify_S`), the same structural invariants hold for all of them. They
are collected in :func:`_assert_canonical_ringed` and reused throughout so the
suite pins the *shared contract* of a ringed solution rather than each
backend's incidental output.

As in ``test_MILP.py``, ``ortools.math_opt`` bundles copies of HiGHS/SCIP that
collide with the standalone packages if loaded into the same process, so every
ortools solve is dispatched to the shared ``ortools_worker`` subprocess. This
module must therefore not import ortools (directly or transitively) at module
scope -- keep such imports inside the individual test/worker functions.
"""

import math

import networkx as nx
import pytest

from optiwindnet.crossings import validate_routeset
from optiwindnet.geometric import is_crossing
from optiwindnet.heuristics import constructor
from optiwindnet.interarraylib import (
    G_from_S,
    add_ring_to_S,
    assign_cables,
    ringify_S,
    rings_from_links,
)
from optiwindnet.mesh import make_planar_embedding
from optiwindnet.pathfinding import PathFinder


# --------------------------------------------------------------------------- #
# Shared invariants of a canonical RINGED topology (backend-agnostic)
# --------------------------------------------------------------------------- #
def _rings(S):
    """Recover the ``(root, [t1, ..., tn])`` rings of a ringed topology ``S``."""
    return rings_from_links(list(S.edges()), S.graph['R'])


def _assert_canonical_ringed(S, capacity):
    """Assert the invariants shared by every canonical RINGED solution.

    A canonical ringed ``S`` is a set of rings sharing the ``R`` roots. Each
    ring of ``n >= 2`` terminals is two radial arms joined at their tails, with
    two real (load-bearing) feeders and exactly one load-0 ``'split'`` open
    point; a lone terminal (``n == 1``) collapses to a single radial stub (one
    feeder, no split). Returns the recovered rings for further per-test checks.
    """
    R, T = S.graph['R'], S.graph['T']

    # only the ring open point may carry an edge 'kind'
    kinds = {d.get('kind') for _, _, d in S.edges(data=True)}
    assert kinds <= {None, 'split'}, kinds

    # every cable segment (arm) stays within the per-cable capacity
    assert max(d['load'] for _, _, d in S.edges(data=True)) <= capacity

    # all terminals are accounted for at the roots
    assert sum(S.nodes[r]['load'] for r in range(-R, 0)) == T

    rings = _rings(S)
    # the rings partition the terminal set: every terminal in exactly one ring
    covered = sorted(t for _, ordered in rings for t in ordered)
    assert covered == list(range(T)), 'rings must partition the terminals'

    split_edges = [
        (u, v) for u, v, d in S.edges(data=True) if d.get('kind') == 'split'
    ]
    feeders = [(u, v) for u, v, d in S.edges(data=True) if u < 0 or v < 0]
    non_stub = [ordered for _, ordered in rings if len(ordered) > 1]
    stub = [ordered for _, ordered in rings if len(ordered) == 1]

    # exactly one open point per non-stub ring, none for single-terminal stubs
    assert len(split_edges) == len(non_stub)
    # no current flows through an open point
    assert all(S[u][v]['load'] == 0 for u, v in split_edges)
    # two feeders per non-stub ring, one per stub
    assert len(feeders) == 2 * len(non_stub) + len(stub)

    for _, ordered in rings:
        n = len(ordered)
        # a ring holds up to 2*capacity terminals (two arms of ceil(n / 2))
        assert math.ceil(n / 2) <= capacity
        # the heaviest cable/node of a ring carries a full arm: ceil(n / 2)
        assert max(S.nodes[t]['load'] for t in ordered) == math.ceil(n / 2)
        # the two arm-head (feeder) terminals carry the whole ring between them
        if n > 1:
            assert S.nodes[ordered[0]]['load'] + S.nodes[ordered[-1]]['load'] == n
        else:
            assert S.nodes[ordered[0]]['load'] == 1

    return rings


def _terminal_terminal_crossings(S, VertexC):
    """Crossings among terminal-terminal edges (feeders excluded).

    The heuristic guarantees no terminal-terminal crossings; straight feeders
    may still cross and are resolved later by PathFinder, so they are excluded.
    """
    edges = [(u, v) for u, v in S.edges if u >= 0 and v >= 0]
    crossings = []
    for i, (u, v) in enumerate(edges):
        for s, t in edges[i + 1 :]:
            if len({u, v, s, t}) < 4:
                continue
            if is_crossing(
                VertexC[u], VertexC[v], VertexC[s], VertexC[t], touch_is_cross=False
            ):
                crossings.append(((u, v), (s, t)))
    return crossings


# --------------------------------------------------------------------------- #
# Unit invariants of the shared ring builders (no solver involved)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('n', range(1, 13))
def test_add_ring_to_S_canonical_shape(n):
    """A ring built from an ordered terminal list is in canonical form."""
    S = nx.Graph(R=1, T=n)
    S.add_node(-1)
    add_ring_to_S(S, -1, list(range(n)), subtree=0, A=None)

    kinds = {d.get('kind') for _, _, d in S.edges(data=True)}
    assert kinds <= {None, 'split'}, 'only the open point may carry a kind'

    feeders = [d['load'] for u, v, d in S.edges(data=True) if u < 0 or v < 0]
    splits = [(u, v) for u, v, d in S.edges(data=True) if d.get('kind') == 'split']
    max_load = max(d['load'] for _, _, d in S.edges(data=True))

    # every arm holds at most half of the doubled ring capacity
    m = math.ceil(n / 2)
    assert max_load == m

    # the two feeders (both real cables) carry the two arm loads; a lone
    # terminal collapses to a single radial-stub feeder
    if n == 1:
        assert feeders == [1] and not splits
    else:
        assert sorted(feeders) == [n - m, m]
        assert len(splits) == 1, 'exactly one open point per ring'
        (su, sv) = splits[0]
        assert S[su][sv]['load'] == 0, 'no current flows through the open point'

    # node loads of the two feeder terminals sum to the ring size
    assert sum(S.nodes[t]['load'] for t in S.neighbors(-1)) == n
    # all nodes of the ring share one subtree id
    assert {S.nodes[t]['subtree'] for t in range(n)} == {0}


@pytest.mark.parametrize('n', range(1, 13))
def test_rings_from_links_roundtrip(n):
    """rings_from_links recovers the terminal set of a built ring."""
    S = nx.Graph(R=1, T=n)
    S.add_node(-1)
    add_ring_to_S(S, -1, list(range(n)), subtree=0, A=None)
    rings = rings_from_links(list(S.edges()), R=1)
    assert len(rings) == 1
    root, ordered = rings[0]
    assert root == -1
    assert set(ordered) == set(range(n))


def test_rings_from_links_multiple_rings_and_roots():
    """Two roots, several rings: each ring is recovered with its own root."""
    S = nx.Graph(R=2, T=9)
    S.add_nodes_from([-2, -1])
    add_ring_to_S(S, -1, [0, 1, 2, 3], subtree=0, A=None)
    add_ring_to_S(S, -1, [4, 5], subtree=1, A=None)
    add_ring_to_S(S, -2, [6, 7, 8], subtree=2, A=None)
    rings = rings_from_links(list(S.edges()), R=2)
    recovered = {(r, frozenset(ordered)) for r, ordered in rings}
    assert recovered == {
        (-1, frozenset({0, 1, 2, 3})),
        (-1, frozenset({4, 5})),
        (-2, frozenset({6, 7, 8})),
    }


@pytest.mark.parametrize('n', range(2, 12, 2))
def test_add_ring_to_S_even_nodes_default_split_is_balanced(n):
    """Without ``A``, an even-node ring (odd n) splits at the exact midpoint.

    ``m = ceil(n / 2)`` puts arm 1 at ceil and arm 2 at floor, so the two arm
    loads differ by exactly one for the balanced (no-``A``) placement.
    """
    S = nx.Graph(R=1, T=n)
    S.add_node(-1)
    add_ring_to_S(S, -1, list(range(n)), subtree=0, A=None)
    feeder_loads = sorted(d['load'] for u, v, d in S.edges(data=True) if u < 0 or v < 0)
    assert feeder_loads == [n // 2, math.ceil(n / 2)]


def test_add_ring_to_S_odd_n_uses_longer_split_edge_when_A_given():
    """On an even-node ring the open point is the longer of the two mid edges.

    n = 3 has a unique middle terminal (index 1) with two candidate split
    edges (0-1 and 1-2); with ``A`` supplied the *longer* one is opened.
    """
    for longer, expected in ((0, 1), (1, 2)):
        S = nx.Graph(R=1, T=3)
        S.add_node(-1)
        A = nx.Graph()
        A.add_edge(0, 1, length=10.0 if longer == 0 else 1.0)
        A.add_edge(1, 2, length=10.0 if longer == 1 else 1.0)
        add_ring_to_S(S, -1, [0, 1, 2], subtree=0, A=A)
        (su, sv) = [
            (u, v) for u, v, d in S.edges(data=True) if d.get('kind') == 'split'
        ][0]
        assert {su, sv} == {longer, expected}


# --------------------------------------------------------------------------- #
# ringify_S: path-form -> canonical ring conversion (used by HGS/LKH/constructor)
# --------------------------------------------------------------------------- #
def _path_form_S(R, paths):
    """Build a path-form topology: one non-branching path per (root, [terminals])."""
    T = sum(len(p) for _, p in paths)
    S = nx.Graph(R=R, T=T)
    S.add_nodes_from(range(-R, 0))
    for root, ordered in paths:
        S.add_edge(root, ordered[0])
        for a, b in zip(ordered, ordered[1:]):
            S.add_edge(a, b)
    return S


def test_ringify_S_single_root_matches_canonical_form():
    """ringify_S turns non-branching paths into canonical rings within capacity."""
    S = _path_form_S(1, [(-1, [0, 1, 2]), (-1, [3, 4])])
    ringify_S(S, None)

    rings = _assert_canonical_ringed(S, capacity=2)
    assert {frozenset(ordered) for _, ordered in rings} == {
        frozenset({0, 1, 2}),
        frozenset({3, 4}),
    }
    # graph-level bookkeeping is recomputed
    assert S.graph['has_loads'] is True
    assert S.graph['max_load'] == 2
    assert S.nodes[-1]['load'] == 5


def test_ringify_S_multi_root_keeps_rings_with_their_root():
    """Each root's paths become rings still attached to that root."""
    S = _path_form_S(2, [(-2, [0, 1]), (-1, [2, 3, 4, 5]), (-1, [6])])
    ringify_S(S, None)
    _assert_canonical_ringed(S, capacity=3)
    rings = _rings(S)
    by_root = {r: {frozenset(o) for rr, o in rings if rr == r} for r in (-2, -1)}
    assert by_root[-2] == {frozenset({0, 1})}
    assert by_root[-1] == {frozenset({2, 3, 4, 5}), frozenset({6})}


def test_ringify_S_renumbers_subtrees_uniquely_per_ring():
    """After ringify each ring is a single, distinct subtree id."""
    S = _path_form_S(1, [(-1, [0, 1]), (-1, [2, 3]), (-1, [4, 5])])
    ringify_S(S, None)
    ring_subtrees = [
        {S.nodes[t]['subtree'] for t in ordered} for _, ordered in _rings(S)
    ]
    # each ring shares one subtree id, and the ids are distinct across rings
    assert all(len(ids) == 1 for ids in ring_subtrees)
    flat = [next(iter(ids)) for ids in ring_subtrees]
    assert len(set(flat)) == len(flat)


def test_ringify_S_rejects_branching_subtree():
    """A branching (non-path) subtree cannot be ringified."""
    S = nx.Graph(R=1, T=3)
    S.add_node(-1)
    S.add_edge(-1, 0)
    S.add_edge(0, 1)
    S.add_edge(0, 2)  # branch at terminal 0
    with pytest.raises(ValueError):
        ringify_S(S, None)


# --------------------------------------------------------------------------- #
# Constructive heuristic RINGED (method='ringed') -- runs in-process
# --------------------------------------------------------------------------- #
_RINGED_MESHES = {
    'albatros_1ss': 'albatros',  # R=1, T=16
    'neart_2ss': 'neart',  # R=2, T=54 (exercises per-root ring loops)
}


@pytest.fixture(
    scope='session', params=list(_RINGED_MESHES.values()), ids=list(_RINGED_MESHES)
)
def ringed_mesh(request):
    """(P, A) for a repository location, built once per location per session."""
    from optiwindnet.api import load_repository

    L = getattr(load_repository(), request.param)
    return make_planar_embedding(L)


@pytest.mark.parametrize('capacity', (3, 5, 8))
def test_constructor_ringed_is_canonical(ringed_mesh, capacity):
    """The 'ringed' constructor output satisfies the shared ringed contract."""
    _, A = ringed_mesh
    S = constructor(A, capacity=capacity, method='ringed')
    rings = _assert_canonical_ringed(S, capacity)

    # a genuine ring (not a radial forest): at least one ring closes a loop,
    # so S carries a cycle and has more than T edges.
    assert not nx.is_forest(S)
    assert any(len(ordered) > 1 for _, ordered in rings)


@pytest.mark.parametrize('capacity', (3, 5, 8))
def test_constructor_ringed_no_terminal_terminal_crossings(ringed_mesh, capacity):
    _, A = ringed_mesh
    S = constructor(A, capacity=capacity, method='ringed')
    crossings = _terminal_terminal_crossings(S, A.graph['VertexC'])
    assert crossings == [], f'ringed produced terminal-terminal crossings: {crossings}'


def test_constructor_ringed_graph_metadata(ringed_mesh):
    _, A = ringed_mesh
    S = constructor(A, capacity=5, method='ringed')
    assert S.graph['creator'] == 'constructor'
    assert S.graph['capacity'] == 5
    assert S.graph['method_options']['method'] == 'ringed'
    # 'ringed' grows simple-path subtrees, so it tracks insertions like radial_EW
    assert 'num_insertions' in S.graph


def test_constructor_ringed_multi_root_covers_every_root(ringed_mesh):
    """On a multi-substation site every root carries at least one ring."""
    _, A = ringed_mesh
    R = A.graph['R']
    if R < 2:
        pytest.skip('single-root mesh')
    S = constructor(A, capacity=5, method='ringed')
    _assert_canonical_ringed(S, capacity=5)
    roots_used = {r for r, _ in _rings(S)}
    assert roots_used == set(range(-R, 0))


@pytest.mark.parametrize('bias_margin', (0.0, 0.1, 0.5))
def test_constructor_ringed_bias_margin_accepted(ringed_mesh, bias_margin):
    """The ringed-specific ``bias_margin`` window yields a valid ringed S."""
    _, A = ringed_mesh
    S = constructor(A, capacity=5, method='ringed', bias_margin=bias_margin)
    _assert_canonical_ringed(S, capacity=5)


@pytest.mark.parametrize('feeder_route', ('segmented', 'straight'))
def test_constructor_ringed_end_to_end_is_valid(ringed_mesh, feeder_route):
    """constructor(ringed) -> PathFinder(ringed=True) -> validate_routeset == []."""
    P, A = ringed_mesh
    if feeder_route == 'straight':
        kwargs = dict(weigh_detours=False, straight_feeder_route=True)
    else:
        kwargs = dict(weigh_detours=True, straight_feeder_route=False)
    S = constructor(A, capacity=5, method='ringed', **kwargs)
    G = PathFinder(G_from_S(S, A), planar=P, A=A, ringed=True).create_detours()
    assign_cables(G, [(5, 1.0)])
    assert validate_routeset(G) == []
    # the routed graph keeps the ring open points and stays within capacity
    assert sum(d.get('kind') == 'split' for _, _, d in G.edges(data=True)) >= 1
    assert G.graph['max_load'] <= 5
    assert G.size(weight='length') > 0.0


# --------------------------------------------------------------------------- #
# EWRouter(method='ringed') through the public WindFarmNetwork API
# --------------------------------------------------------------------------- #
def test_ewrouter_ringed_end_to_end():
    """WindFarmNetwork.optimize(EWRouter(method='ringed')) yields a ringed route."""
    from optiwindnet.api import EWRouter, WindFarmNetwork, load_repository

    capacity = 3
    wfn = WindFarmNetwork(cables=capacity, L=load_repository().albatros)
    wfn.optimize(router=EWRouter(method='ringed'))
    S, G = wfn.S, wfn.G

    _assert_canonical_ringed(S, capacity)
    assert sum(d.get('kind') == 'split' for _, _, d in G.edges(data=True)) >= 1
    assert G.graph['max_load'] <= capacity
    assert G.size(weight='length') > 0.0


# --------------------------------------------------------------------------- #
# RINGED warmstart: a ringed solution maps onto the model's single-chain flow
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('n', range(2, 11))
def test_ringed_warmstart_values_flow_conservation(n):
    """ringed_warmstart_values yields a flow-feasible single-chain assignment."""
    from types import SimpleNamespace

    from optiwindnet.MILP._core import ringed_warmstart_values

    terminals = list(range(n))
    root = -1
    E = [(u, v) for u in terminals for v in terminals if u < v]
    Ep = [(v, u) for u, v in E]
    stars = [(t, root) for t in terminals]
    starsp = [(root, t) for t in terminals]
    metadata = SimpleNamespace(
        R=1,
        link_={k: None for k in E + Ep + stars + starsp},
        flow_={k: None for k in E + Ep + stars},
    )

    S = nx.Graph(R=1, T=n)
    S.add_node(root)
    add_ring_to_S(S, root, terminals, subtree=0, A=None)

    link_vals, flow_vals = ringed_warmstart_values(metadata, S)

    # exactly one flow feeder (t → r) carries the whole ring (n)
    active_feeders = [(t, root) for t in terminals if link_vals[(t, root)]]
    assert len(active_feeders) == 1
    assert flow_vals[active_feeders[0]] == n
    # exactly one flowless closing feeder (r → t)
    assert sum(link_vals[(root, t)] for t in terminals) == 1

    # in-/out-degree of every terminal is exactly 1, and flow is conserved
    for t in terminals:
        out_links = [k for k in link_vals if k[0] == t and link_vals[k]]
        in_links = [k for k in link_vals if k[1] == t and link_vals[k]]
        assert len(out_links) == 1 and len(in_links) == 1
        out_flow = sum(flow_vals.get(k, 0) for k in out_links)
        in_flow = sum(flow_vals.get(k, 0) for k in in_links)
        assert out_flow - in_flow == 1  # each terminal sinks one unit


def test_ringed_warmstart_roundtrip_scip():
    """A ringed solution warm-starts a fresh ringed scip model (accepted)."""
    from optiwindnet.api import WindFarmNetwork, load_repository
    from optiwindnet.MILP import ModelOptions, solver_factory

    opts = ModelOptions(topology='ringed')
    wfn = WindFarmNetwork(cables=3, L=load_repository().albatros)
    try:
        seed = solver_factory('scip')
        seed.set_problem(P=wfn.P, A=wfn.A, capacity=3, model_options=opts)
        seed.solve(time_limit=10, mip_gap=0.05)
        S_warm, _ = seed.get_solution()
    except (FileNotFoundError, ModuleNotFoundError):
        pytest.skip('scip solver unavailable')

    solver = solver_factory('scip')
    # set_problem calls warmup_model; a rejected hint raises OWNWarmupFailed
    solver.set_problem(
        P=wfn.P, A=wfn.A, capacity=3, model_options=opts, warmstart=S_warm
    )
    assert solver.metadata.warmed_by == S_warm.graph['creator']


# --------------------------------------------------------------------------- #
# MILP RINGED on the notebook instances (dispatched to the ortools worker)
# --------------------------------------------------------------------------- #
def _solve_milp_ringed(name, capacity, time_limit, mip_gap):
    """Worker job: solve one repository instance with ortools.cp_sat RINGED.

    Runs in the ortools subprocess and returns only picklable primitives.
    """
    from optiwindnet.api import WindFarmNetwork, load_repository
    from optiwindnet.interarraylib import assign_cables
    from optiwindnet.MILP import ModelOptions, solver_factory

    L = getattr(load_repository(), name)
    wfn = WindFarmNetwork(cables=capacity, L=L)
    solver = solver_factory('ortools.cp_sat')
    solver.set_problem(
        P=wfn.P,
        A=wfn.A,
        capacity=capacity,
        model_options=ModelOptions(topology='ringed'),
    )
    info = solver.solve(time_limit=time_limit, mip_gap=mip_gap)
    S, G = solver.get_solution()
    assign_cables(G, [(capacity, 1.0)])  # exercises the load-0 split-edge path

    R, T = S.graph['R'], S.graph['T']
    return dict(
        termination=info.termination,
        length=G.size(weight='length'),
        kinds=sorted({str(d.get('kind')) for _, _, d in S.edges(data=True)}),
        max_edge_load=max(d['load'] for _, _, d in S.edges(data=True)),
        sum_root_load=sum(S.nodes[r]['load'] for r in range(-R, 0)),
        num_splits=sum(
            d.get('kind') == 'split' for _, _, d in S.edges(data=True)
        ),
        T=T,
        R=R,
        g_edges=G.number_of_edges(),
    )


# (name, capacity, time_limit, mip_gap, expected_optimum_or_None)
_MILP_CASES = [
    ('albatros', 3, 60, 0.01, 23819.8874),
    ('riffgat', 5, 60, 0.01, 20559.3133),
    ('cazzaro_2022', 7, 120, 0.01, None),  # only reaches FEASIBLE in the budget
]


@pytest.mark.parametrize(
    'name, capacity, time_limit, mip_gap, expected', _MILP_CASES
)
def test_milp_ringed_notebook_instances(
    name, capacity, time_limit, mip_gap, expected, ortools_worker
):
    res = ortools_worker.run(
        _solve_milp_ringed,
        (name, capacity, time_limit, mip_gap),
        time_limit + 60,
    )
    if isinstance(res, BaseException):
        raise res

    assert res['termination'] in ('OPTIMAL', 'FEASIBLE')
    # canonical representation: no legacy 'ring_back', only plain + split edges
    assert set(res['kinds']) <= {'None', 'split'}
    # every cable segment (arm) stays within the per-cable capacity
    assert res['max_edge_load'] <= capacity
    # all terminals are connected
    assert res['sum_root_load'] == res['T']
    # a ringed solution has at least one open point
    assert res['num_splits'] >= 1
    # routing produced a non-trivial network
    assert res['g_edges'] > res['T']
    assert res['length'] > 0.0
    if expected is not None:
        # these instances solve to proven optimality: length is deterministic
        assert res['length'] == pytest.approx(expected, rel=1e-3)


# --------------------------------------------------------------------------- #
# MILP RINGED on the in-process solvers (highs, scip) -- no ortools involved
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('solver_name', ['highs', 'scip'])
def test_milp_ringed_inprocess_solvers_are_canonical(solver_name):
    """highs / scip RINGED produce a canonical ringed S (skip if unavailable).

    These solvers never touch OR-Tools' native libraries, so they run directly
    in-process. The solve is loose (short budget, wide gap) because we assert
    the *structure* of a ringed solution, not its optimal length.
    """
    from optiwindnet.api import WindFarmNetwork, load_repository
    from optiwindnet.MILP import ModelOptions, solver_factory

    capacity = 3
    wfn = WindFarmNetwork(cables=capacity, L=load_repository().albatros)
    try:
        solver = solver_factory(solver_name)
        solver.set_problem(
            P=wfn.P,
            A=wfn.A,
            capacity=capacity,
            model_options=ModelOptions(topology='ringed'),
        )
        info = solver.solve(time_limit=30, mip_gap=0.2)
    except (FileNotFoundError, ModuleNotFoundError):
        pytest.skip(f'{solver_name} solver unavailable')

    assert info.termination.lower() in ('optimal', 'feasible')
    S, G = solver.get_solution()
    _assert_canonical_ringed(S, capacity)
    assign_cables(G, [(capacity, 1.0)])
    assert G.size(weight='length') > 0.0


def test_milprouter_ringed_end_to_end():
    """MILPRouter(topology='ringed') via WindFarmNetwork yields a ringed route.

    Exercises the high-level warmstart+solve path (api.py: MILPRouter.route),
    including the ringed-solution warmstart, not just the low-level solver.
    """
    from optiwindnet.api import MILPRouter, WindFarmNetwork, load_repository
    from optiwindnet.MILP import ModelOptions

    capacity = 3
    wfn = WindFarmNetwork(cables=capacity, L=load_repository().albatros)
    try:
        wfn.optimize(
            router=MILPRouter(
                solver_name='scip',
                model_options=ModelOptions(topology='ringed'),
                time_limit=30,
                mip_gap=0.2,
            )
        )
    except (FileNotFoundError, ModuleNotFoundError):
        pytest.skip('scip solver unavailable')

    _assert_canonical_ringed(wfn.S, capacity)
    G = wfn.G
    assert sum(d.get('kind') == 'split' for _, _, d in G.edges(data=True)) >= 1
    assert G.graph['max_load'] <= capacity
    assert G.size(weight='length') > 0.0


# --------------------------------------------------------------------------- #
# HGS-CVRP RINGED (closed CVRP) -- runs in-process (no ortools involved)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('capacity', (2, 3, 5))
def test_hgs_ringed_albatros(capacity):
    pytest.importorskip('hybgensea')
    from optiwindnet.api import WindFarmNetwork, as_normalized, load_repository
    from optiwindnet.baselines.hgs import hgs_cvrp

    L = load_repository().albatros
    wfn = WindFarmNetwork(cables=capacity, L=L)
    # structure-only test: a short budget still yields a feasible ringed
    # solution (HGS spends its whole time_limit, so keep it small)
    S = hgs_cvrp(
        as_normalized(wfn.A),
        capacity=capacity,
        time_limit=0.5,
        ringed=True,
        seed=0,
    )

    _assert_canonical_ringed(S, capacity)

    G = PathFinder(
        G_from_S(S, wfn.A), planar=wfn.P, A=wfn.A, branched=False, ringed=True
    ).create_detours()
    assign_cables(G, [(capacity, 1.0)])
    assert validate_routeset(G) == []
    assert G.size(weight='length') > 0.0


def test_lkh_ringed_albatros():
    """LKH-3 ringed (closed CVRP) produces a valid ring set covering all terminals."""
    import shutil

    if shutil.which('LKH') is None:
        pytest.skip('LKH executable not on PATH')
    from optiwindnet.api import WindFarmNetwork, as_normalized, load_repository
    from optiwindnet.baselines.lkh import lkh3

    capacity = 3
    L = load_repository().albatros
    wfn = WindFarmNetwork(cables=capacity, L=L)
    S = lkh3(
        as_normalized(wfn.A),
        capacity=capacity,
        time_limit=5,
        ringed=True,
        seed=0,
    )

    _assert_canonical_ringed(S, capacity)

    G = PathFinder(
        G_from_S(S, wfn.A), planar=wfn.P, A=wfn.A, branched=False, ringed=True
    ).create_detours()
    assign_cables(G, [(capacity, 1.0)])
    assert validate_routeset(G) == []
    assert G.size(weight='length') > 0.0


def test_hgs_router_ringed_end_to_end():
    """HGSRouter(ringed=True) produces a valid ringed routeset via WindFarmNetwork."""
    pytest.importorskip('hybgensea')
    from optiwindnet.api import HGSRouter, WindFarmNetwork, load_repository

    capacity = 3
    L = load_repository().albatros
    wfn = WindFarmNetwork(cables=capacity, L=L)
    wfn.optimize(router=HGSRouter(time_limit=0.5, ringed=True, seed=0))
    S, G = wfn.S, wfn.G

    _assert_canonical_ringed(S, capacity)
    # the routed graph carries the ring open points and stays within capacity
    assert sum(d.get('kind') == 'split' for _, _, d in G.edges(data=True)) >= 1
    assert G.graph['max_load'] <= capacity
    assert G.size(weight='length') > 0.0
