# tests/test_db_modelv2.py
# SPDX-License-Identifier: MIT

import datetime
import os
from pathlib import Path

import pytest
from pony.orm import db_session, flush, TransactionIntegrityError

from optiwindnet.db.modelv2 import open_database


# ----------------------------
# Helpers: sample payloads
# ----------------------------
def _sample_nodeset_kwargs():
    # B must equal sum(constraint_groups); with no border/obstacles, use 0.
    return dict(
        name="site-1",
        T=2,
        R=1,
        B=0,
        VertexC=b"V" * 8,                 # arbitrary bytes
        constraint_groups=[0],            # one group of size 0 (the border)
        constraint_vertices=[],           # concatenation of groups' vertices
        landscape_angle=0.0,
        digest=b"x" * 32,                 # PrimaryKey(bytes)
    )


def _sample_method_kwargs():
    return dict(
        solver_name="ortools",
        funname="route_solve",
        options={"time_limit": 1, "seed": 0},
        funfile="solver_impl.py",
        funhash=b"h" * 32,
        digest=b"m" * 32,                 # PrimaryKey(bytes)
    )


def _sample_routeset_kwargs():
    # Minimal, consistent values. num_gates length typically equals R.
    return dict(
        handle="run-001",
        valid=True,
        T=2,
        R=1,
        capacity=3,
        length=123.45,
        is_normalized=False,
        runtime=0.12,
        num_gates=[1],                    # one root → one gate count
        C=0,
        D=0,
        creator="unit-test",
        detextra=0.0,
        num_diagonals=0,
        tentative=[],                     # flattened int array is ok as empty
        rogue=[],
        misc={"note": "test"},
        stuntC=b"",                       # optional bytes
        clone2prime=[],                   # optional int array
        edges=[0, 1, 1, 2],               # arbitrary flattened edges encoding
    )


# ----------------------------
# Tests
# ----------------------------

def test_open_database_creates_file(tmp_path: Path):
    dbfile = tmp_path / "db.sqlite"
    assert not dbfile.exists()
    db = open_database(str(dbfile), create_db=True)
    # Accessing entities ensures mapping was generated
    assert "NodeSet" in db.entities
    assert dbfile.exists()


def test_nodeset_insert_and_uniqueness(tmp_path: Path):
    db = open_database(str(tmp_path / "db.sqlite"), create_db=True)
    NodeSet = db.entities["NodeSet"]

    # Insert one: check PK and unique 'name'
    with db_session:
        ns = NodeSet(**_sample_nodeset_kwargs())
        assert ns.digest == b"x" * 32

    # Duplicate name with different digest -> uniqueness error (raised on flush/commit)
    with pytest.raises(TransactionIntegrityError):
        with db_session:
            kw = _sample_nodeset_kwargs()
            kw["digest"] = b"y" * 32  # different PK, but SAME unique name -> boom
            NodeSet(**kw)
            flush()


def test_method_digest_uniqueness(tmp_path: Path):
    db = open_database(str(tmp_path / "db.sqlite"), create_db=True)
    Method = db.entities["Method"]

    with db_session:
        m1 = Method(**_sample_method_kwargs())
        assert m1.digest == b"m" * 32

    # Same digest again → PK/unique conflict on flush/commit
    with pytest.raises(TransactionIntegrityError):
        with db_session:
            Method(**_sample_method_kwargs())
            flush()


def test_machine_uniqueness_and_json(tmp_path: Path):
    db = open_database(str(tmp_path / "db.sqlite"), create_db=True)
    Machine = db.entities["Machine"]

    # Create and ensure auto id is materialized after flush
    with db_session:
        mach = Machine(name="runner-01", attrs={"cpu": "x86", "cores": 8})
        flush()
        mid = mach.id
        assert isinstance(mid, int)

    # Load in a new session and verify JSON contents
    with db_session:
        got = Machine.get(name="runner-01")
        assert got is not None
        assert got.attrs["cores"] == 8

    # Unique name violation: use valid attrs (None would raise ValueError instead)
    with pytest.raises(TransactionIntegrityError):
        with db_session:
            Machine(name="runner-01", attrs={"dup": True})
            flush()


def test_routeset_crud_and_relations(tmp_path: Path):
    db = open_database(str(tmp_path / "db.sqlite"), create_db=True)
    NodeSet = db.entities["NodeSet"]
    RouteSet = db.entities["RouteSet"]
    Method = db.entities["Method"]
    Machine = db.entities["Machine"]

    with db_session:
        ns = NodeSet(**_sample_nodeset_kwargs())
        meth = Method(**_sample_method_kwargs())
        mach = Machine(name="runner-02", attrs={"os": "linux"})
        rs = RouteSet(nodes=ns, method=meth, machine=mach, **_sample_routeset_kwargs())
        flush()
        rid = rs.id
        assert isinstance(rid, int)

    # Read back and check relations
    with db_session:
        got = RouteSet[rid]
        assert got.nodes.name == "site-1"
        assert got.method.solver_name == "ortools"
        assert got.machine.name == "runner-02"
        # update a field
        got.valid = False

    with db_session:
        assert RouteSet[rid].valid is False


def test_required_field_enforcement(tmp_path: Path):
    db = open_database(str(tmp_path / "db.sqlite"), create_db=True)
    NodeSet = db.entities["NodeSet"]
    RouteSet = db.entities["RouteSet"]
    Method = db.entities["Method"]

    with db_session:
        ns = NodeSet(**_sample_nodeset_kwargs())
        meth = Method(**_sample_method_kwargs())

        bad = _sample_routeset_kwargs()
        bad.pop("is_normalized")  # Required field → Pony raises ValueError
        with pytest.raises(ValueError):
            RouteSet(nodes=ns, method=meth, **bad)


def test_timestamp_autosets_close_to_now(tmp_path: Path):
    db = open_database(str(tmp_path / "db.sqlite"), create_db=True)
    Method = db.entities["Method"]

    before = datetime.datetime.utcnow()
    with db_session:
        m = Method(**_sample_method_kwargs())
        flush()
        ts = m.timestamp
        assert isinstance(ts, datetime.datetime)
    after = datetime.datetime.utcnow()

    # _naive_utc_now is used → ts should be between before/after
    assert before <= ts <= after


def test_routeset_optional_fields_roundtrip(tmp_path: Path):
    db = open_database(str(tmp_path / "db.sqlite"), create_db=True)
    NodeSet = db.entities["NodeSet"]
    RouteSet = db.entities["RouteSet"]
    Method = db.entities["Method"]

    with db_session:
        ns = NodeSet(**_sample_nodeset_kwargs())
        meth = Method(**_sample_method_kwargs())
        # omit machine on purpose (Optional)
        rs = RouteSet(nodes=ns, method=meth, **_sample_routeset_kwargs())
        flush()
        rid = rs.id

    with db_session:
        got = RouteSet[rid]
        assert got.machine is None
        assert got.misc["note"] == "test"
