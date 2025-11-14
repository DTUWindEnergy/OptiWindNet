from pathlib import Path
import pytest
import numpy as np
from pony.orm import db_session

from optiwindnet.api import HGSRouter
from optiwindnet.db import open_database, G_from_routeset, L_from_nodeset
from optiwindnet.db.storagev2 import (
    store_G,
    packnodes,
    add_if_absent,
    get_machine_pk,
)
from .helpers import tiny_wfn, assert_graph_equal

tmp_path = Path(__file__).parent / 'temp'

# ---------------------------
# Test modelv2
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

    expected_attrs = ['Entity', 'Machine', 'Method', 'NodeSet', 'RouteSet']

    for name in expected_attrs:
        assert hasattr(db, name), f"db should expose attribute '{name}'"


# ---------------------------
# Test storagev2
# ---------------------------


def test_L_from_nodeset(tmp_path):
    dbfile = tmp_path / 'db_test.sqlite'

    # open and create db if not there
    db = open_database(dbfile, create_db=True)

    wfn = tiny_wfn()
    L = wfn.L
    L.name = 'Test'

    pack = packnodes(L)
    with db_session:
        NodeSet = db.entities['NodeSet']
        digest = add_if_absent(NodeSet, pack)
        ns = NodeSet[digest]

    L2 = L_from_nodeset(ns)
    assert L2.graph['T'] == L.graph['T']
    assert L2.graph['R'] == L.graph['R']
    assert np.allclose(L2.graph['VertexC'], L.graph['VertexC'])


def test_G_from_routeset(tmp_path):
    dbfile = tmp_path / 'db_test.sqlite'

    # open and create db if not there
    db = open_database(dbfile, create_db=True)
    get_machine_pk(db)

    wfn = tiny_wfn(router=HGSRouter(time_limit=0.1))
    G = wfn.G

    id = store_G(G, db=db)
    assert id == 1

    with db_session:
        rs = db.RouteSet[id]
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
    db = open_database(dbfile, create_db=True)

    wfn = tiny_wfn(cables=1)
    L = wfn.L
    L.name = 'Test'

    pack = packnodes(L)
    with db_session:
        NodeSet = db.entities['NodeSet']
        digest = add_if_absent(NodeSet, pack)
        ns = NodeSet[digest]

    L2 = L_from_nodeset(ns)
    assert L2.graph['T'] == L.graph['T']
    assert L2.graph['R'] == L.graph['R']
    assert np.allclose(L2.graph['VertexC'], L.graph['VertexC'])


def test_G_from_routeset_detours(tmp_path):
    dbfile = tmp_path / 'db_test.sqlite'

    # open and create db if not there
    db = open_database(dbfile, create_db=True)
    get_machine_pk(db)

    wfn = tiny_wfn(router=HGSRouter(time_limit=0.1))
    G = wfn.G

    id = store_G(G, db=db)
    assert id == 1

    with db_session:
        rs = db.RouteSet[id]
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
