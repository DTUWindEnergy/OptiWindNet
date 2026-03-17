# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/
"""Database model v3 for storage of locations and route sets (Peewee).

Tables:
  - NodeSet: location definition
  - RouteSet: routeset (i.e. a record of G)
  - Method: info on algorithm & options to produce routesets
  - Machine: info on machine that generated a routeset
"""

import os
from contextlib import contextmanager
from peewee import (
    AutoField,
    BlobField,
    BooleanField,
    CharField,
    DatabaseProxy,
    DateTimeField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
    SqliteDatabase,
)
from playhouse.sqlite_ext import JSONField

from ._core import _naive_utc_now

__all__ = ()

database_proxy = DatabaseProxy()


class BaseModel(Model):
    class Meta:
        database = database_proxy


class NodeSet(BaseModel):
    # hashlib.sha256(VertexC + boundary).digest()
    digest = BlobField(primary_key=True)
    name = CharField(unique=True)
    T = IntegerField()  # # of non-root nodes
    R = IntegerField()  # # of root nodes
    B = IntegerField()  # num_border_vertices
    # vertices (nodes + roots) coordinates (UTM)
    # np.lib.format.write_array(io, np.empty((R + T + B, 2), dtype=float))
    VertexC = BlobField()
    # the first group is the border (ccw), then obstacles (cw)
    # B is sum(constraint_groups)
    constraint_groups = JSONField()
    # indices to VertexC, concatenation of the groups' ordered vertices
    constraint_vertices = JSONField()
    landscape_angle = FloatField(null=True)


class Method(BaseModel):
    # hashlib.sha256(funhash + pickle(options)).digest()
    digest = BlobField(primary_key=True)
    solver_name = CharField()
    funname = CharField()
    # options is a dict of function parameters
    options = JSONField()
    timestamp = DateTimeField(default=_naive_utc_now)
    funfile = CharField()
    # hashlib.sha256(fun.__code__.co_code)
    funhash = BlobField()


class Machine(BaseModel):
    id = AutoField()
    name = CharField(unique=True)
    attrs = JSONField(null=True)


class RouteSet(BaseModel):
    id = AutoField()
    handle = CharField()
    valid = BooleanField(null=True)
    T = IntegerField()  # num_nodes
    R = IntegerField()  # num_roots
    capacity = IntegerField()
    length = FloatField()
    is_normalized = BooleanField()
    # runtime always in [s]
    runtime = FloatField(null=True)
    num_gates = JSONField()
    # number of contour nodes
    C = IntegerField(default=0)
    # number of detour nodes
    D = IntegerField(default=0)
    # short identifier of routeset origin (redundant with Method)
    creator = CharField(null=True)
    # relative increase from undetoured routeset to the detoured one
    # detoured_length = (1 + detextra)*undetoured_length
    detextra = FloatField(null=True)
    num_diagonals = IntegerField(null=True)
    tentative = JSONField(null=True)
    rogue = JSONField(null=True)
    timestamp = DateTimeField(null=True, default=_naive_utc_now)
    misc = JSONField(default=dict)  # never NULL, defaults to {}
    stuntC = BlobField(null=True)  # coords of border stunts
    # len(clone2prime) == C + D
    clone2prime = JSONField(null=True)
    edges = JSONField()
    nodes = ForeignKeyField(NodeSet, backref='routesets')
    method = ForeignKeyField(Method, backref='routesets')
    machine = ForeignKeyField(Machine, backref='routesets', null=True)


_ALL_MODELS = [NodeSet, Method, Machine, RouteSet]


def open_database(filepath: str, create_db: bool = False) -> SqliteDatabase:
    """Opens the sqlite database v3 file specified in `filepath`.

    Args:
      filepath: path to database file
      create_db: True -> create a new file if it does not exist

    Returns:
      SqliteDatabase object (Peewee)
    """
    filepath = os.path.abspath(os.path.expanduser(str(filepath)))
    if not create_db and not os.path.exists(filepath):
        raise OSError(f'Database file not found: {filepath}')
    db = SqliteDatabase(
        filepath,
        pragmas={'journal_mode': 'wal', 'foreign_keys': 1},
    )
    database_proxy.initialize(db)
    db.connect()
    db.create_tables(_ALL_MODELS)
    return db


@contextmanager
def database_connection(filepath: str, create_db: bool = False):
    """Open the sqlite database for the duration of a context block.

    Args:
      filepath: path to database file
      create_db: True -> create a new file if it does not exist

    Yields:
      Connected SqliteDatabase object (Peewee)
    """
    db = open_database(filepath, create_db=create_db)
    try:
        yield db
    finally:
        db.close()
