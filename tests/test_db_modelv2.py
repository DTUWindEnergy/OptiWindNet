from pathlib import Path
import pytest
from optiwindnet.db.modelv2 import open_database

tmp_path = Path(__file__).parent / "temp"

def test_open_database(tmp_path):
    """
    1) Guarantee DB file does not exist and assert open_database(..., create_db=False) raises OSError
    2) Create DB with create_db=True
    3) Assert file exists and DB exposes expected attributes
    """
    dbfile = tmp_path / "db_test.sqlite"

    # ensure file is not present
    if dbfile.exists():
        dbfile.unlink()

    # Expect OSError when trying to open a non-existent DB without create flag
    with pytest.raises(OSError):
        open_database(str(dbfile), create_db=False)
        assert dbfile.exists(), "database file should have been created"

    # create the DB
    db = open_database(str(dbfile), create_db=True)

    expected_attrs = ["Entity", "Machine", "Method", "NodeSet", "RouteSet"]

    for name in expected_attrs:
        assert hasattr(db, name), f"db should expose attribute '{name}'"


from optiwindnet.db.storagev2 import (
    G_from_routeset,
    L_from_nodeset,
    store_G,
    packnodes,
    add_if_absent,
)

import numpy as np
import pytest
from pony.orm import db_session, flush
from pony.orm import db_session
from .helpers import tiny_wfn



# ---------------------------
# Tests
# ---------------------------

def test_L_from_nodeset():
    dbfile = tmp_path / "db_test.sqlite"

    # create the DB
    db = open_database(str(dbfile), create_db=True)
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


# def test_store_G(tmp_path, monkeypatch):
#     dbfile = tmp_path / "db_test.sqlite"
#     # create the DB
#     db = open_database(str(dbfile), create_db=True)
#     wfn = tiny_wfn()
#     G = wfn.G
#     G.name = "Test_store_G"
#     G.method_options = {
#         'fun_fingerprint': {
#             'funfile': r'C:\code\OptiWindNet\optiwindnet\heuristics\EW_presolver.py',
#             'funhash': b'\x06...',  # whatever hash
#             'funname': 'EW_presolver'
#         },
#         'solver_name': 'my_solver'   # <--- Add this key
#         }

#     with db_session:
#         RouteSet = db.entities['RouteSet']

#         # Explicitly store the graph, then normalize to an entity
#         rs_or_pk = store_G(G, db=db)
#         rs = rs_or_pk if isinstance(rs_or_pk, RouteSet) else RouteSet[rs_or_pk]

#         assert rs.capacity == 3
#         assert rs.method.solver_name == 'solverX'

#         # call inside the session
#         G3 = G_from_routeset(rs)

#     # now it's just a plain NetworkX graph; safe outside session
#     assert list(G3.edges()) == {(-1, 0), (0, 1)}
