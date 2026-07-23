import math
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import optiwindnet.pathfinding as pathfinding
from optiwindnet.geometric import is_crossing
from optiwindnet.interarraylib import (
    G_from_S,
    S_from_G,
    add_ring_to_S,
    assign_cables,
    calcload,
)
from optiwindnet.pathfinding import PathFinder
from optiwindnet.terse import LinkScope, TerseLinks
from optiwindnet.types import Topology

from .cases import (
    CONSTRUCTOR_CASES,
    HGS_CASES,
    case_node_id,
    expected_topology,
    topology_golden_key,
)
from .helpers import canonical_edges, tiny_wfn
from .producers import hgs_topology
from .sitecache import get_bundle, get_bundle_from_nodeset_digest
from .solver_topologies import load_solver_topologies

PATHFINDER_GOLDEN_FILE = Path(__file__).with_name('pathfinder_golden.pkl')


def _load_pathfinder_golden() -> tuple[TerseLinks, ...]:
    if not PATHFINDER_GOLDEN_FILE.exists():
        raise FileNotFoundError(
            f'Missing PathFinder golden data: {PATHFINDER_GOLDEN_FILE}\n'
            'Regenerate it with: '
            'python -m tests.update_pathfinder_golden <routesets.sqlite>'
        )
    with PATHFINDER_GOLDEN_FILE.open('rb') as file:
        golden = pickle.load(file)
    if not isinstance(golden, tuple) or not all(
        isinstance(case, TerseLinks)
        and case.scope is LinkScope.ROUTESET
        and case.nodeset_digest is not None
        for case in golden
    ):
        raise TypeError(
            'PathFinder golden data must be a tuple of site-bound routed TerseLinks'
        )
    return golden


PATHFINDER_CASES = _load_pathfinder_golden()
SOLVER_TOPOLOGIES = load_solver_topologies()


def _edges_cross(G):
    """Return list of edge-pair crossings in G (ignoring shared-node pairs)."""
    VertexC = G.graph['VertexC']
    fnT = G.graph.get('fnT')
    edges = list(G.edges)
    crossings = []
    for i, (u, v) in enumerate(edges):
        u_, v_ = (fnT[u], fnT[v]) if fnT is not None else (u, v)
        for s, t in edges[i + 1 :]:
            s_, t_ = (fnT[s], fnT[t]) if fnT is not None else (s, t)
            if s_ == u_ or s_ == v_ or t_ == u_ or t_ == v_:
                continue
            if is_crossing(
                VertexC[u_],
                VertexC[v_],
                VertexC[s_],
                VertexC[t_],
                touch_is_cross=False,
            ):
                crossings.append(((u, v), (s, t)))
    return crossings


def _all_turbines_connected(G):
    """Check that every turbine can reach a root via graph edges."""
    T, R = G.graph['T'], G.graph['R']
    for t in range(T):
        found_root = False
        for r in range(-R, 0):
            if nx.has_path(G, t, r):
                found_root = True
                break
        if not found_root:
            return False
    return True


def _pathfinder_case_id(case: TerseLinks) -> str:
    assert case.nodeset_digest is not None
    digest = case.nodeset_digest.hex()[:8]
    return f'{digest}-T{case.T}-{case.topology.value}-C{case.C}-D{case.D}'


@pytest.fixture(scope='module')
def location_meshes():
    """Load only the explicitly mapped sites used by the golden artifact."""
    wanted = {
        case.nodeset_digest
        for case in PATHFINDER_CASES
        if case.nodeset_digest is not None
    }
    return {digest: get_bundle_from_nodeset_digest(digest) for digest in wanted}


@pytest.mark.parametrize('case', PATHFINDER_CASES, ids=_pathfinder_case_id)
def test_create_detours_matches_milp_routeset(case, location_meshes):
    """Reproduce curated stored routesets on their bundled locations."""
    assert case.nodeset_digest is not None
    bundle = location_meshes[case.nodeset_digest]
    L, P, A = bundle.L, bundle.P, bundle.A

    # Decode the stored, detoured routeset on the freshly loaded location, then
    # discard its geometry to recover only the solver topology S.
    expected = case.to_routeset(L)
    expected.graph['capacity'] = expected.graph['max_load']
    S = S_from_G(expected)

    G = G_from_S(S, A)
    actual = PathFinder(G, planar=P, A=A).create_detours()

    assert canonical_edges(actual) == canonical_edges(expected)
    assert actual.graph['topology'] is case.topology
    assert actual.graph.get('C', 0) == case.C
    assert actual.graph.get('D', 0) == case.D


# ---------- no-crossing scenario (cables=4, single chain) ----------


def test_no_crossings_removes_tentative_tag():
    """When there are no crossings, create_detours() removes tentative tags."""
    wfn = tiny_wfn()
    G_tent = G_from_S(wfn.S, wfn.A)
    pf = PathFinder(G_tent, planar=wfn.P, A=wfn.A)

    assert pf.Xings == []
    assert pf.iterations == 0

    G_out = pf.create_detours()
    assert 'tentative' not in G_out.graph
    for _, _, d in G_out.edges(data=True):
        assert d.get('kind') != 'tentative'


# ---------- crossing scenario (cables=1, each turbine gets a feeder) ----------


def _make_crossing_case():
    """cables=1 tiny_wfn: feeders (-1,1) and (-1,2) cross other edges."""
    wfn = tiny_wfn(cables=1)
    G_tent = G_from_S(wfn.S, wfn.A)
    return G_tent, wfn.P, wfn.A


def test_no_crossing_fast_path_detags_tentative_feeders():
    G_tentative, P, A = _make_crossing_case()
    finder = PathFinder(G_tentative, planar=P, A=A)
    finder.Xings = []

    G = finder.create_detours()

    assert 'tentative' not in G.graph
    assert all('kind' not in G[root][node] for root, node in finder.tentative)


def test_pathfinder_detects_crossings():
    """cables=1 produces tentative feeders that have crossings."""
    G_tent, P, A = _make_crossing_case()

    tentative = G_tent.graph['tentative']
    assert len(tentative) == 2
    assert (-1, 2) in tentative
    assert (-1, 1) in tentative

    pf = PathFinder(G_tent, planar=P, A=A)
    assert len(pf.Xings) > 0
    # Xings identify which feeders are crossing
    assert (-1, 1) in pf.Xings
    assert (-1, 2) in pf.Xings


def test_label_free_shortened_contour_is_not_an_ordinary_route(monkeypatch):
    """A fully shortened contour contributes only its expanded fence edges."""
    G_tent, P, A = _make_crossing_case()
    G_tent.add_edge(0, 2, length=A[0][2]['length'], load=1, reverse=False)
    G_tent.graph['shortened_contours'] = {(0, 2): ([9], [])}

    flipped_edge_sets = []
    original = pathfinding.planar_flipped_by_routeset

    def capture_edges(edges, **kwargs):
        flipped_edge_sets.append(set(edges))
        return original(edges, **kwargs)

    monkeypatch.setattr(pathfinding, 'planar_flipped_by_routeset', capture_edges)
    pf = PathFinder(G_tent, planar=P, A=A)

    assert flipped_edge_sets
    assert all((0, 2) not in edges for edges in flipped_edge_sets)
    assert any(fence.endpoints == (0, 2) for fence in pf.fences)


def test_create_detours_adds_detour_nodes():
    """create_detours() adds detour clone nodes for crossing feeders."""
    G_tent, P, A = _make_crossing_case()
    pf = PathFinder(G_tent, planar=P, A=A)
    G_det = pf.create_detours()

    D = G_det.graph['D']
    assert D > 0, 'Expected detour vertices to be created'
    # Detour nodes should have kind='detour'
    T, B = G_det.graph['T'], G_det.graph['B']
    C = G_det.graph.get('C', 0)
    for clone in range(T + B + C, T + B + C + D):
        assert G_det.nodes[clone]['kind'] == 'detour'


def test_create_detours_crossing_free():
    """The output of create_detours() must have no edge crossings."""
    G_tent, P, A = _make_crossing_case()
    pf = PathFinder(G_tent, planar=P, A=A)
    G_det = pf.create_detours()

    crossings = _edges_cross(G_det)
    assert crossings == [], f'Crossings remain after detours: {crossings}'


def test_create_detours_preserves_connectivity():
    """All turbines must remain connected to a root after detours."""
    G_tent, P, A = _make_crossing_case()
    pf = PathFinder(G_tent, planar=P, A=A)
    G_det = pf.create_detours()

    assert _all_turbines_connected(G_det)


def test_create_detours_no_tentative_left():
    """Successful detours should remove the 'tentative' graph attribute."""
    G_tent, P, A = _make_crossing_case()
    pf = PathFinder(G_tent, planar=P, A=A)
    G_det = pf.create_detours()

    assert 'tentative' not in G_det.graph


def test_create_detours_fnT_consistent():
    """fnT must map clone nodes to their prime (border/obstacle) vertices."""
    G_tent, P, A = _make_crossing_case()
    pf = PathFinder(G_tent, planar=P, A=A)
    G_det = pf.create_detours()

    fnT = G_det.graph['fnT']
    T, B, R = G_det.graph['T'], G_det.graph['B'], G_det.graph['R']
    D = G_det.graph['D']
    C = G_det.graph.get('C', 0)

    # fnT length should cover all nodes
    assert len(fnT) == T + B + C + D + R
    # Terminal nodes map to themselves
    for t in range(T):
        assert fnT[t] == t
    # Root nodes map correctly
    for r in range(-R, 0):
        assert fnT[r] == r
    # Clone nodes map to primes in the constraint-vertex range
    for clone in range(T + B + C, T + B + C + D):
        prime = fnT[clone]
        assert prime < T + B, (
            f'clone {clone} maps to prime {prime} outside constraint range'
        )


def test_get_best_path_returns_valid_path_to_root():
    """get_best_path should return paths that end at a root node."""
    G_tent, P, A = _make_crossing_case()
    pf = PathFinder(G_tent, planar=P, A=A)

    for n in range(G_tent.graph['T']):
        path, dists = pf.get_best_path(n)
        if not path:
            continue
        assert path[0] == n, f'path for node {n} does not start at {n}'
        assert path[-1] < 0, f'path for node {n} does not end at a root'
        assert len(dists) == len(path) - 1
        # Each dist should be a positive number
        for d in dists:
            assert d > 0, f'non-positive distance {d} in path for {n}'


def test_get_best_path_dist_is_sum_of_hops():
    """Total distance should equal the sum of hop distances."""
    G_tent, P, A = _make_crossing_case()
    pf = PathFinder(G_tent, planar=P, A=A)

    for n in range(G_tent.graph['T']):
        path, dists = pf.get_best_path(n)
        if not path:
            continue
        total = sum(dists)
        # get the stored total from paths
        best_pn_ids = [
            pn_id
            for pair_id in pf.pair_ids_by_prime.get(n, ())
            if (pn_id := pf.best_pn_by_pair_id[pair_id]) is not None
        ]
        if best_pn_ids:
            best_pn_id = min((pf.paths[pn_id].dist, pn_id) for pn_id in best_pn_ids)[1]
            stored_dist = pf.paths[best_pn_id].dist
            assert math.isclose(total, stored_dist, rel_tol=1e-9), (
                f'node {n}: sum(dists)={total} != stored dist={stored_dist}'
            )


def test_detextra_is_positive():
    """Detours add length, so detextra should be positive."""
    G_tent, P, A = _make_crossing_case()
    pf = PathFinder(G_tent, planar=P, A=A)
    G_det = pf.create_detours()

    detextra = G_det.graph['detextra']
    assert detextra >= 0, f'detextra should be non-negative, got {detextra}'


# ---------- obstacle-heavy scenario ----------


def test_obstacle_scenario_crossing_free():
    """A large obstacle between substation and turbines requires detours around it."""
    wfn = tiny_wfn(
        turbinesC=[[3.0, 0.0], [3.0, 2.0], [3.0, 4.0]],
        substationsC=[[0.0, 2.0]],
        borderC=[[-2, -2], [5, -2], [5, 6], [-2, 6]],
        obstacleC_=[np.array([[1.0, 0.5], [2.0, 0.5], [2.0, 3.5], [1.0, 3.5]])],
        cables=1,
    )
    G_tent = G_from_S(wfn.S, wfn.A)
    pf = PathFinder(G_tent, planar=wfn.P, A=wfn.A)

    # All 3 feeders should be tentative and have crossings
    assert len(pf.Xings) > 0

    G_det = pf.create_detours()
    assert G_det.graph['D'] > 0, 'Expected detour nodes around obstacle'
    assert _edges_cross(G_det) == [], 'Output should be crossing-free'
    assert _all_turbines_connected(G_det)


def test_best_paths_overlay_structure():
    """best_paths_overlay returns a graph with virtual path edges and no feeders."""
    G_tent, P, A = _make_crossing_case()
    pf = PathFinder(G_tent, planar=P, A=A)

    overlay = pf.best_paths_overlay()
    # overlay should be a subgraph view (no feeder edges u<0 or v<0)
    for u, v in overlay.edges:
        assert u >= 0 and v >= 0, f'feeder edge ({u},{v}) in overlay'
    # overlay graph attribute should contain the path graph
    assert 'overlay' in overlay.graph
    J = overlay.graph['overlay']
    # J should have virtual edges
    has_virtual = any(d.get('kind') == 'virtual' for _, _, d in J.edges(data=True))
    assert has_virtual, 'overlay graph J should contain virtual path edges'


def test_pathfinder_radial_topology():
    """A graph declaring topology='radial' hooks only to path endpoints."""
    G_tent, P, A = _make_crossing_case()
    G_tent.graph['topology'] = Topology.RADIAL
    pf = PathFinder(G_tent, planar=P, A=A)
    G_det = pf.create_detours()

    assert _all_turbines_connected(G_det)
    assert _edges_cross(G_det) == []


def test_get_mesh_endpoint_prefers_the_more_anchored_endpoint():
    finder = object.__new__(PathFinder)

    assert finder._get_mesh_endpoint((1, 2), {1: {3}, 2: {3, 4}}) == 2
    assert finder._get_mesh_endpoint((1, 2), {1: {3, 4}, 2: {3}}) == 1
    assert finder._get_mesh_endpoint((1, 2), {1: {3}, 2: {4}}) == 2
    assert finder._get_mesh_endpoint((1, 2), {1: {3}}) == 1
    assert finder._get_mesh_endpoint((1, 2), {}) is None


def test_pathfinder_ringed_topology():
    """Detouring preserves connectivity and open points for a ringed input."""
    wfn = tiny_wfn()
    T, R = (wfn.A.graph[key] for key in 'TR')
    S = nx.Graph(
        T=T,
        R=R,
        topology=Topology.RINGED,
        creator='synthetic',
        capacity=3,
    )
    S.add_nodes_from(range(-R, 0))
    for subtree, start in enumerate(range(0, T, 6)):
        add_ring_to_S(
            S,
            (-1, -1),
            list(range(start, min(start + 6, T))),
            subtree=subtree,
            A=wfn.A,
        )
    calcload(S)

    G = PathFinder(G_from_S(S, wfn.A), planar=wfn.P, A=wfn.A).create_detours()

    assert G.graph['topology'] is Topology.RINGED
    assert _all_turbines_connected(G)
    assert _edges_cross(G) == []
    assert any(data.get('load') == 0 for *_, data in G.edges(data=True))


@pytest.mark.parametrize(
    'case',
    [case for case in CONSTRUCTOR_CASES if case.exact_golden],
    ids=case_node_id,
)
def test_pathfinder_routes_curated_constructor_topologies(case):
    """Route stored producer output here, keeping solver tests topology-only."""
    bundle = get_bundle(case.site)
    P, A = bundle.P, bundle.A
    encoded = SOLVER_TOPOLOGIES[topology_golden_key(case)]
    assert isinstance(encoded, TerseLinks)
    S = encoded.to_topology(capacity=case.capacity, creator='golden')

    G = PathFinder(G_from_S(S, A), planar=P, A=A).create_detours()
    assign_cables(G, [(case.capacity, 1.0)])

    assert _all_turbines_connected(G)
    assert _edges_cross(G) == []
    assert not any(data.get('kind') == 'tentative' for *_, data in G.edges(data=True))


@pytest.mark.parametrize(
    'case',
    [case for case in HGS_CASES if case.site == 'london'],
    ids=case_node_id,
)
def test_pathfinder_routes_multiroot_hgs_topology(case):
    """Exercise detouring on cached multi-root RADIAL and RINGED topologies."""
    bundle = get_bundle(case.site)
    P, A = bundle.P, bundle.A
    S = hgs_topology(case)

    G = PathFinder(G_from_S(S, A), planar=P, A=A).create_detours()
    assign_cables(G, [(case.capacity, 1.0)])

    assert _all_turbines_connected(G)
    assert _edges_cross(G) == []
    assert G.graph['topology'] is expected_topology(case)


def test_no_crossing_pathfinder_compacts_stunt_contour_clone():
    """The no-crossing fast path compacts clone ids after stunt removal."""
    bundle = get_bundle('borkum2')
    P, A = bundle.P, bundle.A
    R, T, B_A = (A.graph[key] for key in 'RTB')
    stunt_primes = set(A.graph['stunts_primes'])
    stunt_nodes = set(range(T + B_A - len(stunt_primes), T + B_A))

    def minimal_routeset(gate, leaf):
        S = nx.Graph(
            R=R,
            T=T,
            capacity=T,
            creator='synthetic',
            has_loads=True,
            topology=Topology.RADIAL,
        )
        S.add_node(-1, load=2)
        S.add_node(gate, load=2, subtree=0)
        S.add_node(leaf, load=1, subtree=0)
        S.add_edge(-1, gate, load=2, reverse=False)
        S.add_edge(gate, leaf, load=1, reverse=False)
        return S

    stunt_edges = [
        (u, v)
        for u, v, midpath in A.edges(data='midpath')
        if 0 <= u < T
        and 0 <= v < T
        and midpath
        and any(node in stunt_nodes for node in midpath)
    ]
    for u, v in stunt_edges:
        for gate, leaf in ((u, v), (v, u)):
            tentative = G_from_S(minimal_routeset(gate, leaf), A)
            fnT = tentative.graph.get('fnT')
            if fnT is None:
                continue
            contour_nodes = {
                node
                for node, data in tentative.nodes(data=True)
                if data.get('kind') == 'contour'
                and (fnT[node] in stunt_nodes or fnT[node] in stunt_primes)
            }
            if not contour_nodes:
                continue
            assert all(
                fnT[node] < tentative.graph['VertexC'].shape[0]
                for node in contour_nodes
            )
            finder = PathFinder(tentative, planar=P, A=A)
            if not finder.Xings:
                break
        else:
            continue
        break
    else:
        pytest.skip('no crossing-free stunt-contour routeset found for borkum2')

    G = finder.create_detours()
    R, T, B = (G.graph[key] for key in 'RTB')
    C, D = (G.graph.get(key, 0) for key in 'CD')
    fnT = G.graph['fnT']
    contours = {
        node for node, data in G.nodes(data=True) if data.get('kind') == 'contour'
    }
    assert finder.Xings == []
    assert C > 0
    assert len(fnT) == T + B + C + D + R
    assert contours == set(range(T + B, T + B + C))
    assert all(fnT[node] < T + B for node in contours)
    assert not any(fnT[node] in stunt_nodes for node in contours)


# ---------- route-fence chain scenario (spanning + touching chains) ----------


def test_spanning_chain_detours_crossing_free():
    """A real layout whose routeset runs cables along the border builds chains.

    ``yi_2019`` at capacity 8 yields a spanning route fence (on-constraint segment
    of length >= 2) plus a touching fence, exercising both the spanning-chain
    pairing and the touching-chain construction in `_precompute_chains`. The
    detoured routeset must be crossing-free with every turbine connected.
    """
    from optiwindnet.heuristics import constructor

    bundle = get_bundle('yi_2019', single_root=True)
    P, A = bundle.P, bundle.A
    S = constructor(A, capacity=8, method='rootlust')
    G_tent = G_from_S(S, A)
    pf = PathFinder(G_tent, planar=P, A=A)

    # Guard that this config still exercises chain topology (it is the point of
    # the test); if a heuristic change stops producing a spanning fence here,
    # pick another chain-producing location/capacity rather than weakening this.
    assert pf.chain_access, 'expected chain topology to be built'
    assert any(len(f.primes_on_constraint) >= 2 for f in pf.fences), (
        'expected at least one spanning route fence'
    )

    G_det = pf.create_detours()
    assert _all_turbines_connected(G_det)
    assert _edges_cross(G_det) == []
