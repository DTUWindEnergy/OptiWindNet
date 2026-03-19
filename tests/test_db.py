import pytest
import numpy as np

from optiwindnet.api import HGSRouter
from optiwindnet.db import (
    open_database,
    database_connection,
    G_from_routeset,
    L_from_nodeset,
    NodeSet,
    RouteSet,
    store_G,
)
from optiwindnet.db.storage import (
    packnodes,
    add_if_absent,
    get_machine_pk,
)
from .helpers import tiny_wfn, assert_graph_equal


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
        open_database(str(dbfile), create_db=False)
        assert dbfile.exists(), 'database file should have been created'

    # create the DB
    db = open_database(str(dbfile), create_db=True)

    # Verify tables were created
    assert db is not None


def test_database_connection_closes_db(tmp_path):
    dbfile = tmp_path / 'db_test.sqlite'

    with database_connection(str(dbfile), create_db=True) as db:
        assert db is not None
        assert not db.is_closed()

    assert db.is_closed()


def test_database_connection_supports_db_usage(tmp_path):
    dbfile = tmp_path / 'db_test.sqlite'

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


# ---------------------------
# Test storage
# ---------------------------


def test_L_from_nodeset(tmp_path):
    dbfile = tmp_path / 'db_test.sqlite'

    # open and create db if not there
    open_database(dbfile, create_db=True)

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
    open_database(dbfile, create_db=True)
    get_machine_pk()

    wfn = tiny_wfn(router=HGSRouter(time_limit=0.1))
    G = wfn.G

    id = store_G(G)
    assert id == 1

    rs = RouteSet.get_by_id(id)
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


# tests when G has detours
def test_L_from_nodeset_detours(tmp_path):
    dbfile = tmp_path / 'db_test.sqlite'
    # open and create db if not there
    open_database(dbfile, create_db=True)

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
    open_database(dbfile, create_db=True)
    get_machine_pk()

    wfn = tiny_wfn(router=HGSRouter(time_limit=0.1))
    G = wfn.G

    id = store_G(G)
    assert id == 1

    rs = RouteSet.get_by_id(id)
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
