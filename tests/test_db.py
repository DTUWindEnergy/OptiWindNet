import networkx as nx
import numpy as np
import pytest

from optiwindnet.api import HGSRouter, WindFarmNetwork
from optiwindnet.db import (
    G_from_routeset,
    L_from_nodeset,
    NodeSet,
    RouteSet,
    database_connection,
    open_database,
    store_G,
)
from optiwindnet.db.storage import (
    add_if_absent,
    get_machine_pk,
    infer_topology,
    packnodes,
)
from optiwindnet.fingerprint import fingerprint_coordinates
from optiwindnet.interarraylib import S_from_G, validate_topology
from optiwindnet.terse import LinkScope, TerseLinks
from optiwindnet.types import Topology

from .helpers import assert_graph_equal, tiny_wfn

# ---------------------------
# Test model
# ---------------------------


def test_open_database(tmp_path):
    """ """
    dbfile = tmp_path / 'db_test.sqlite'

    # ensure file is not present
    if dbfile.exists():
        dbfile.unlink()

    # Expect OSError when trying to open a non-existent DB without create flag
    with pytest.raises(OSError):
        with database_connection(str(dbfile), create_db=False):
            pass

    # create the DB
    try:
        db = open_database(str(dbfile), create_db=True)
        # Verify tables were created
        assert db is not None
    finally:
        db.close()


def test_database_connection_closes_db(tmp_path):
    dbfile = tmp_path / 'db_test.sqlite'

    with database_connection(str(dbfile), create_db=True) as db:
        assert db is not None
        assert not db.is_closed()

    assert db.is_closed()


def test_database_connection_supports_db_usage(tmp_path):
    dbfile = tmp_path / 'db_test.sqlite'

    wfn = tiny_wfn()
    L = wfn.L
    L.name = 'Test'

    pack = packnodes(L)
    with database_connection(dbfile, create_db=True):
        digest = add_if_absent(NodeSet, pack)
        ns = NodeSet.get_by_id(digest)

    L2 = L_from_nodeset(ns)
    assert L2.graph['T'] == L.graph['T']
    assert L2.graph['R'] == L.graph['R']


# ---------------------------
# Test storage
# ---------------------------


def test_packnodes_uses_canonical_vertexc_fingerprint():
    L = tiny_wfn().L
    L.graph['VertexC'] = np.array(L.graph['VertexC'], order='F')

    pack = packnodes(L)
    digest, VertexC_npy = fingerprint_coordinates(L.graph['VertexC'])

    assert pack['digest'] == digest
    assert pack['VertexC'] == VertexC_npy
    assert b"'fortran_order': False" in VertexC_npy[:256]


def test_L_from_nodeset(tmp_path):
    dbfile = tmp_path / 'db_test.sqlite'

    # open and create db if not there
    with database_connection(dbfile, create_db=True):
        wfn = tiny_wfn()
        L = wfn.L
        L.name = 'Test'

        pack = packnodes(L)
        digest = add_if_absent(NodeSet, pack)
        ns = NodeSet.get_by_id(digest)

        L2 = L_from_nodeset(ns)
        assert L2.graph['T'] == L.graph['T']
        assert L2.graph['R'] == L.graph['R']
        assert np.allclose(L2.graph['VertexC'], L.graph['VertexC'])


def test_G_from_routeset(tmp_path):
    dbfile = tmp_path / 'db_test.sqlite'

    # open and create db if not there
    with database_connection(dbfile, create_db=True):
        get_machine_pk()

        wfn = tiny_wfn(router=HGSRouter(time_limit=0.1))
        G = wfn.G

        id = store_G(G)
        assert id == 1

        rs = RouteSet.get_by_id(id)
        assert rs.feeders_per_root == [len(G[root]) for root in range(-G.graph['R'], 0)]
    G_rs = G_from_routeset(rs)

    ignored_keys = {
        'bound',
        'method_options',
        'relgap',
        'solver_details',
        'D',
        'landscape_angle',
        'method',
        'norm_offset',
        'norm_scale',
        'num_diagonals',
    }
    assert_graph_equal(G_rs, G, ignored_graph_keys=ignored_keys, verbose=False)


def test_G_from_routeset_ringed(tmp_path, locations):
    """A RINGED routeset (a graph with cycles) round-trips through the database.

    The rings are stored as a sequence of routes in ``edges`` (roots interleaved
    with each ring's node walk), so it is longer than the one-entry-per-non-root
    -node forest encoding; the open points are re-derived at the load midpoint on
    read, not stored. The derived ``subtree`` node attribute is recomputed by
    ``calcload`` with a per-arm convention, so it is excluded from the
    comparison, as are the contour/detour counters that are ``None`` before
    storage and ``0`` after.
    """
    pytest.importorskip('hybgensea')
    dbfile = tmp_path / 'db_test.sqlite'

    with database_connection(dbfile, create_db=True):
        get_machine_pk()

        # capacity 5 yields both rings (split open points) and detour clones, so
        # this exercises the ring route-sequence together with the clone nodes.
        capacity = 5
        wfn = WindFarmNetwork(cables=capacity, L=locations.albatros)
        wfn.optimize(router=HGSRouter(time_limit=0.5, ringed=True, seed=0))
        G = wfn.G
        T = G.graph['T']
        C, D = (G.graph.get(k) or 0 for k in 'CD')
        assert C + D > 0, 'this fixture must have clone nodes to be meaningful'

        id = store_G(G)
        rs = RouteSet.get_by_id(id)
        # a ringed routeset is stored as a route sequence: longer than the
        # forest encoding (one entry per non-root node, clones included)
        assert len(rs.edges) > T + C + D
    G_rs = G_from_routeset(rs)

    ignored_keys = {
        'bound',
        'method_options',
        'relgap',
        'solver_details',
        'C',
        'D',
        'landscape_angle',
        'method',
        'norm_offset',
        'norm_scale',
        'num_diagonals',
    }
    assert_graph_equal(
        G_rs,
        G,
        ignored_graph_keys=ignored_keys,
        ignored_node_keys={'subtree'},
        verbose=False,
    )
    # the reloaded routeset is a genuine ring set, not a forest
    split_edges = [(u, v) for u, v, d in G_rs.edges(data=True) if d.get('load') == 0]
    assert split_edges, 'split open points must be restored on read'
    assert all(G_rs[u][v]['load'] == 0 for u, v in split_edges)


# tests when G has detours
def test_L_from_nodeset_detours(tmp_path):
    dbfile = tmp_path / 'db_test.sqlite'
    # open and create db if not there
    with database_connection(dbfile, create_db=True):
        wfn = tiny_wfn(cables=1)
        L = wfn.L
        L.name = 'Test'

        pack = packnodes(L)
        digest = add_if_absent(NodeSet, pack)
        ns = NodeSet.get_by_id(digest)

        L2 = L_from_nodeset(ns)
        assert L2.graph['T'] == L.graph['T']
        assert L2.graph['R'] == L.graph['R']
        assert np.allclose(L2.graph['VertexC'], L.graph['VertexC'])


def test_G_from_routeset_detours(tmp_path):
    dbfile = tmp_path / 'db_test.sqlite'

    # open and create db if not there
    with database_connection(dbfile, create_db=True):
        get_machine_pk()

        wfn = tiny_wfn(router=HGSRouter(time_limit=0.1))
        G = wfn.G

        id = store_G(G)
        assert id == 1

        rs = RouteSet.get_by_id(id)
        assert rs.feeders_per_root == [len(G[root]) for root in range(-G.graph['R'], 0)]
    G_rs = G_from_routeset(rs)

    ignored_keys = {
        'bound',
        'method_options',
        'relgap',
        'solver_details',
        'D',
        'landscape_angle',
        'method',
        'norm_offset',
        'norm_scale',
        'num_diagonals',
    }
    assert_graph_equal(G_rs, G, ignored_graph_keys=ignored_keys, verbose=False)


def _bare_G(**graph_attrs):
    """A 3-terminal, 1-root routeset graph carrying only record metadata."""
    return nx.Graph(
        R=1,
        T=3,
        B=0,
        capacity=3,
        VertexC=np.array([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0)]),
        **graph_attrs,
    )


_FOREST = [-1, 0, 1]  # positional: a chain -1--0--1--2
_RINGED = [-1, 0, 1, 2]  # route sequence: any length other than the positional one


@pytest.mark.parametrize(
    'terse, attrs, expected',
    (
        # the encoding settles RINGED on its own, whatever the metadata says
        (_RINGED, {}, Topology.RINGED),
        (_RINGED, {'creator': 'baselines.hgs'}, Topology.RINGED),
        # MILP records name their topology
        (_FOREST, {'method_options': {'topology': Topology.RADIAL}}, Topology.RADIAL),
        (
            _FOREST,
            {'method_options': {'topology': Topology.BRANCHED}},
            Topology.BRANCHED,
        ),
        # HGS/LKH solve a CVRP: their routes are paths
        (_FOREST, {'creator': 'baselines.hgs'}, Topology.RADIAL),
        (_FOREST, {'creator': 'baselines.lkh'}, Topology.RADIAL),
        # the constructor names its method instead
        (
            _FOREST,
            {'creator': 'constructor', 'method_options': {'method': 'radial_EW'}},
            Topology.RADIAL,
        ),
        (
            _FOREST,
            {'creator': 'constructor', 'method_options': {'method': 'esau_williams'}},
            Topology.BRANCHED,
        ),
        # nothing to go on: the weakest claim any forest satisfies
        (_FOREST, {}, Topology.BRANCHED),
    ),
)
def test_legacy_record_topology_is_inferred_before_decoding(terse, attrs, expected):
    """Records predating 'topology' get one inferred from encoding and metadata.

    ``validate_topology`` requires the attribute, so a graph read back from an
    older row must declare a shape, and must satisfy the one it declares.
    """
    G = _bare_G(**attrs)
    topology = infer_topology(G, terse)
    encoding = TerseLinks(
        links=tuple(terse),
        topology=topology,
        scope=LinkScope.ROUTESET,
        T=G.graph['T'],
        R=G.graph['R'],
        B=G.graph['B'],
    )
    G = encoding.to_routeset(G, capacity=G.graph['capacity'])

    assert G.graph['topology'] == expected
    assert validate_topology(S_from_G(G)) == []


def test_infer_topology_reports_a_recorded_claim_when_links_disagree():
    """Legacy inference reports metadata; graph validation is a later concern."""
    G = _bare_G(method_options={'topology': Topology.RINGED})
    assert infer_topology(G, _FOREST) is Topology.RINGED


def test_inference_recovers_the_topology_of_a_stripped_record(tmp_path):
    """A real record with 'topology' removed reads back with the shape it had.

    Pins that ``G_from_routeset`` populates the metadata inference reads
    (``creator``, ``method_options``) before it decodes the edges.
    """
    dbfile = tmp_path / 'db_test.sqlite'

    with database_connection(dbfile, create_db=True):
        get_machine_pk()
        G = tiny_wfn(router=HGSRouter(time_limit=0.1)).G
        rs = RouteSet.get_by_id(store_G(G))
        assert 'topology' in rs.misc, 'current records store it; older ones did not'

        misc = dict(rs.misc)
        del misc['topology']
        rs.misc = misc
        rs.save()

        G_rs = G_from_routeset(rs)

    assert G_rs.graph['creator'] == 'baselines.hgs'
    assert G_rs.graph['topology'] == G.graph['topology']
