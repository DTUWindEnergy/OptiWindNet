"""Validation tests for the RINGED topology.

Covers three layers:

* unit invariants of the shared ring builders :func:`add_ring_to_S` and
  :func:`rings_from_links` (canonical two-arm-with-split shape);
* the MILP RINGED path on the three instances from
  ``docs/notebooks/ringed_topology_demo.ipynb`` (albatros/3, riffgat/5,
  cazzaro_2022/7), solved with ``ortools.cp_sat`` exactly as the notebook does;
* the HGS-CVRP RINGED path (closed CVRP) end to end.

As in ``test_MILP.py``, ``ortools.math_opt`` bundles copies of HiGHS/SCIP that
collide with the standalone packages if loaded into the same process, so every
ortools solve is dispatched to the shared ``ortools_worker`` subprocess. This
module must therefore not import ortools (directly or transitively) at module
scope.
"""

import math

import networkx as nx
import pytest

from optiwindnet.interarraylib import add_ring_to_S, rings_from_links


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
# HGS-CVRP RINGED (closed CVRP) — runs in-process (no ortools involved)
# --------------------------------------------------------------------------- #
def test_hgs_ringed_albatros():
    pytest.importorskip('hybgensea')
    from optiwindnet.api import WindFarmNetwork, as_normalized, load_repository
    from optiwindnet.baselines.hgs import hgs_cvrp
    from optiwindnet.interarraylib import G_from_S, assign_cables
    from optiwindnet.pathfinding import PathFinder

    capacity = 3
    L = load_repository().albatros
    wfn = WindFarmNetwork(cables=capacity, L=L)
    S = hgs_cvrp(
        as_normalized(wfn.A),
        capacity=capacity,
        time_limit=3,
        ringed=True,
        seed=0,
    )

    kinds = {d.get('kind') for _, _, d in S.edges(data=True)}
    assert kinds <= {None, 'split'}
    assert max(d['load'] for _, _, d in S.edges(data=True)) <= capacity
    assert sum(S.nodes[r]['load'] for r in range(-S.graph['R'], 0)) == S.graph['T']
    assert sum(d.get('kind') == 'split' for _, _, d in S.edges(data=True)) >= 1

    G = PathFinder(
        G_from_S(S, wfn.A), planar=wfn.P, A=wfn.A, branched=False, ringed=True
    ).create_detours()
    assign_cables(G, [(capacity, 1.0)])
    assert G.size(weight='length') > 0.0


def test_lkh_ringed_albatros():
    """LKH-3 ringed (closed CVRP) produces a valid ring set covering all terminals."""
    import shutil

    if shutil.which('LKH') is None:
        pytest.skip('LKH executable not on PATH')
    from optiwindnet.api import WindFarmNetwork, as_normalized, load_repository
    from optiwindnet.baselines.lkh import lkh3
    from optiwindnet.interarraylib import G_from_S, assign_cables
    from optiwindnet.pathfinding import PathFinder

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

    kinds = {d.get('kind') for _, _, d in S.edges(data=True)}
    assert kinds <= {None, 'split'}
    assert max(d['load'] for _, _, d in S.edges(data=True)) <= capacity
    # every terminal is covered by exactly one ring
    assert sum(S.nodes[r]['load'] for r in range(-S.graph['R'], 0)) == S.graph['T']
    assert sum(d.get('kind') == 'split' for _, _, d in S.edges(data=True)) >= 1

    G = PathFinder(
        G_from_S(S, wfn.A), planar=wfn.P, A=wfn.A, branched=False, ringed=True
    ).create_detours()
    assign_cables(G, [(capacity, 1.0)])
    assert G.size(weight='length') > 0.0


def test_hgs_router_ringed_end_to_end():
    """HGSRouter(ringed=True) produces a valid ringed routeset via WindFarmNetwork."""
    pytest.importorskip('hybgensea')
    from optiwindnet.api import HGSRouter, WindFarmNetwork, load_repository

    capacity = 3
    L = load_repository().albatros
    wfn = WindFarmNetwork(cables=capacity, L=L)
    wfn.optimize(router=HGSRouter(time_limit=3, ringed=True, seed=0))
    G = wfn.G

    # the routed graph carries the ring open points and stays within capacity
    assert sum(d.get('kind') == 'split' for _, _, d in G.edges(data=True)) >= 1
    assert G.graph['max_load'] <= capacity
    assert G.size(weight='length') > 0.0
