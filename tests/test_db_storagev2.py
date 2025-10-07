# SPDX-License-Identifier: MIT
# tests/test_db_storagev2.py

import datetime as dt
from typing import Iterable

import networkx as nx
import numpy as np
import pytest
from pony.orm import db_session, flush

from optiwindnet.db.modelv2 import open_database
from optiwindnet.db.storagev2 import (
    G_by_method,
    G_from_routeset,
    Gs_from_attrs,
    L_from_nodeset,
    add_if_absent,
    get_machine_pk,
    oddtypes_to_serializable,
    pack_G,
    packmethod,
    packnodes,
    store_G,
    terse_pack_from_G,
    untersify_to_G,
)

# ---------------------------
# Helpers
# ---------------------------


def _coords(pairs):
    return np.array(pairs, dtype=float)


def _edges_undirected(edges: Iterable[tuple[int, int]]):
    return {tuple(sorted(map(int, e))) for e in map(tuple, edges)}


def _make_L(T=2, R=1, name='Farm'):
    verts = _coords([(0.0, 0.0), (1.0, 0.0), (0.0, -1.0)])  # T + R
    L = nx.Graph(R=R, T=T, B=0, name=name, handle=name.lower(), VertexC=verts)
    # IMPORTANT: storage.packnodes expects these keys present and typed
    L.graph['border'] = np.array([], dtype=int)
    L.graph['obstacles'] = []
    L.add_nodes_from(range(T), kind='wtg')
    L.add_nodes_from(range(-R, 0), kind='oss')
    return L


def _make_method_options():
    return dict(
        solver_name='solverX',
        fun_fingerprint=dict(
            funhash=b'F' * 32,
            funfile='/tmp/solverX.py',
            funname='solve',
        ),
        opt_level=2,
    )


def _make_G_for_storage():
    """
    Tiny routeset G; topology: -1 -- 0 -- 1, power flows 1 -> 0 -> -1.
    Reverse flags set so terse encoding creates exactly one parent per terminal:
      terse[1] = 0  (edge 0-1 with reverse=False)
      terse[0] = -1 (edge -1-0 with reverse=False)
    """
    T, R = 2, 1
    verts = _coords([(0.0, 0.0), (1.0, 0.0), (0.0, -1.0)])
    G = nx.Graph(
        R=R,
        T=T,
        B=0,
        VertexC=verts,
        name='My Farm',
        handle='my_farm',
        capacity=3,
        creator='unit-test',
        runtime=0.01,
        method_options=_make_method_options(),
    )
    G.graph['border'] = np.array([], dtype=int)
    G.graph['obstacles'] = []

    for n in (-1, 0, 1):
        G.add_node(n, kind=('oss' if n == -1 else 'wtg'))

    # IMPORTANT: both reverse=False so that terse[1]=0 and terse[0]=-1
    G.add_edge(0, 1, length=1.0, reverse=False, load=1)  # interpreted as 1 -> 0
    G.add_edge(-1, 0, length=1.0, reverse=False, load=1)  # interpreted as 0 -> -1

    # Loads consistent with path 1->0->-1 (root load left as 0)
    G.nodes[1]['load'] = 1
    G.nodes[0]['load'] = 2
    G.nodes[-1]['load'] = 0
    G.graph['has_loads'] = True
    return G


# ---------------------------
# Tests
# ---------------------------


def test_packnodes_and_L_from_nodeset_roundtrip(tmp_path):
    db = open_database(str(tmp_path / 'db.sqlite'), create_db=True)
    L = _make_L(T=2, R=1, name='RoundTrip')

    pack = packnodes(L)
    with db_session:
        NodeSet = db.entities['NodeSet']
        digest = add_if_absent(NodeSet, pack)
        ns = NodeSet[digest]

    L2 = L_from_nodeset(ns)
    assert L2.graph['T'] == L.graph['T']
    assert L2.graph['R'] == L.graph['R']
    assert np.allclose(L2.graph['VertexC'], L.graph['VertexC'])


def test_packmethod_and_add_if_absent(tmp_path):
    db = open_database(str(tmp_path / 'db.sqlite'), create_db=True)
    mpack = packmethod(_make_method_options())

    with db_session:
        Method = db.entities['Method']
        dig1 = add_if_absent(Method, mpack)
        dig2 = add_if_absent(Method, mpack)
        assert dig1 == dig2  # idempotent


def test_terse_pack_and_untersify_to_G(tmp_path):
    G = _make_G_for_storage()
    terse_pack = terse_pack_from_G(G)
    terse = terse_pack['edges']

    L = _make_L(T=2, R=1)
    G2 = L.copy()
    untersify_to_G(G2, terse=terse, clone2prime=None)

    # Compare undirected
    assert _edges_undirected(G2.edges()) == {(-1, 0), (0, 1)}
    # Lengths should be positive
    assert G2.size(weight='length') > 0


def test_oddtypes_to_serializable_and_pack_G_misc(tmp_path):
    G = _make_G_for_storage()
    G.graph['custom_arr'] = np.array([1, 2, 3], dtype=np.int64)
    G.graph['custom_tuple'] = (np.int64(7), 8)
    G.graph['user_notes'] = {'when': dt.datetime(2024, 1, 1).isoformat()}

    packed = pack_G(G)
    misc = packed['misc']

    assert misc['custom_arr'] == [1, 2, 3]
    # tuples are preserved as tuples; numpy ints converted to py ints
    assert misc['custom_tuple'] == (7, 8)
    assert packed['capacity'] == 3
    assert packed['handle'] == 'my_farm'
    assert packed['R'] == 1 and packed['T'] == 2


from pony.orm import db_session


def test_store_G_and_read_back_with_G_from_routeset(tmp_path, monkeypatch):
    db = open_database(str(tmp_path / 'db.sqlite'), create_db=True)
    _ = packmethod(_make_method_options())
    G = _make_G_for_storage()
    with db_session:
        RouteSet = db.entities['RouteSet']

        # Ensure Machine row exists (first call may return None, second should not)
        _ = get_machine_pk(db)
        _ = get_machine_pk(db)

        # Explicitly store the graph, then normalize to an entity
        rs_or_pk = store_G(G, db=db)
        rs = rs_or_pk if isinstance(rs_or_pk, RouteSet) else RouteSet[rs_or_pk]

        assert rs.capacity == 3
        assert rs.method.solver_name == 'solverX'

        # call inside the session
        G3 = G_from_routeset(rs)

    # now it's just a plain NetworkX graph; safe outside session
    assert _edges_undirected(G3.edges()) == {(-1, 0), (0, 1)}


def test_get_machine_pk_is_idempotent(tmp_path, monkeypatch):
    db = open_database(str(tmp_path / 'db.sqlite'), create_db=True)
    monkeypatch.setattr('optiwindnet.db.storagev2.getfqdn', lambda: 'localhost')
    monkeypatch.setattr('optiwindnet.db.storagev2.gethostname', lambda: 'myhost')

    # The first call can return None (not flushed yet). Call twice and compare non-null PKs.
    pk1 = get_machine_pk(db)
    pk2 = get_machine_pk(db)
    assert pk2 is not None
    assert pk1 in (None, pk2)


def test_G_by_method_and_Gs_from_attrs(tmp_path, monkeypatch):
    db = open_database(str(tmp_path / 'db.sqlite'), create_db=True)
    _ = packmethod(_make_method_options())
    G = _make_G_for_storage()
    with db_session:
        Method = db.entities['Method']
        NodeSet = db.entities['NodeSet']

        # Ensure Machine row exists before storing the routeset
        _ = get_machine_pk(db)
        _ = get_machine_pk(db)

        # First, ensure the routeset actually exists in DB
        rs_or_pk = store_G(G, db=db)
        RouteSet = db.entities['RouteSet']
        _ = rs_or_pk if isinstance(rs_or_pk, RouteSet) else RouteSet[rs_or_pk]

        # Fetch helpers
        meth = Method.get(funname='solve')
        farm = NodeSet.get(name='My Farm')

        # run DB-backed helpers inside session
        Gdb = G_by_method(G, meth, db)
        tuples = Gs_from_attrs(farm, methods=(meth,), capacities=(3,), db=db)

    # Assertions outside the session are fine
    assert _edges_undirected(Gdb.edges()) == {(-1, 0), (0, 1)}
    assert len(tuples) == 1 and len(tuples[0]) == 1
    Gt = tuples[0][0]
    assert _edges_undirected(Gt.edges()) == {(-1, 0), (0, 1)}
