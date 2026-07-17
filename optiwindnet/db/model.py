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
    digest = BlobField(primary_key=True)
    """SHA256 hash of `VertexC` coordinates array in `.npy` format."""

    name = CharField(unique=True)
    """Unique name of the wind farm location."""

    T = IntegerField()
    """Number of non-root nodes (wind turbines)."""

    R = IntegerField()
    """Number of root nodes (substations)."""

    B = IntegerField()
    """Number of border-only vertices."""

    VertexC = BlobField()
    """2D coordinates array in UTM, shape (T + B + R, 2)."""

    constraint_groups = JSONField()
    """Sizes of each polygon group (first boundary CCW, then obstacles CW)."""

    constraint_vertices = JSONField()
    """Concatenated indices into VertexC for constraint polygons."""

    landscape_angle = FloatField(null=True)
    """Rotation angle of the landscape layout in radians."""


class Method(BaseModel):
    digest = BlobField(primary_key=True)
    """SHA256 hash of funhash and JSON-serialized options."""

    solver_name = CharField()
    """Name of the solver/algorithm family."""

    funname = CharField()
    """Name of the optimization function."""

    options = JSONField()
    """Dictionary of function parameters."""

    timestamp = DateTimeField(default=_naive_utc_now)
    """UTC timestamp of creation."""

    funfile = CharField()
    """Source filename containing the optimization function."""

    funhash = BlobField()
    """SHA256 hash of the function bytecode."""


class Machine(BaseModel):
    id = AutoField()
    """Primary key."""

    name = CharField(unique=True)
    """Unique hostname/FQDN of the generating machine."""

    attrs = JSONField(null=True)
    """System attributes and specifications dictionary."""


class RouteSet(BaseModel):
    id = AutoField()
    """Primary key."""

    handle = CharField()
    """Standardized wind farm location identifier."""

    T = IntegerField()
    """Number of non-root nodes (turbines)."""

    R = IntegerField()
    """Number of root nodes (substations)."""

    capacity = IntegerField()
    """Cable capacity (maximum turbine load)."""

    length = FloatField()
    """Total cable network length."""

    runtime = FloatField(null=True)
    """Optimization runtime in seconds."""

    feeders_per_root = JSONField()
    """List specifying number of feeders per root substation."""

    C = IntegerField(default=0)
    """Number of contour nodes."""

    D = IntegerField(default=0)
    """Number of detour nodes."""

    creator = CharField(null=True)
    """Short identifier of the routeset origin."""

    detextra = FloatField(null=True)
    """Relative length increase due to detour: length = (1+detextra)*undetoured."""

    num_diagonals = IntegerField(null=True)
    """Number of diagonal crossing structures."""

    tentative = JSONField(null=True)
    """List of vertex pairs representing tentative edges."""

    rogue = JSONField(null=True)
    """List of vertex pairs representing rogue/invalid edges."""

    timestamp = DateTimeField(null=True, default=_naive_utc_now)
    """UTC timestamp of creation."""

    misc = JSONField(default=dict)
    """Dictionary of extra metadata and solver metrics."""

    clone2prime = JSONField(null=True)
    """List of length C + D mapping cloned nodes back to original nodes."""

    edges = JSONField()
    """Terse edge encoding. For a forest routeset it is the positional tree form
    (``edges[i]`` is the target of node ``i``, one entry per non-root node); for
    a RINGED routeset it is a sequence of routes (roots interleaved with each
    ring's node walk), which is longer than the non-root node count."""

    nodes = ForeignKeyField(NodeSet, backref='routesets')
    """Foreign key link to the layout NodeSet."""

    method = ForeignKeyField(Method, backref='routesets')
    """Foreign key link to the Method used."""

    machine = ForeignKeyField(Machine, backref='routesets', null=True)
    """Foreign key link to the generating Machine."""


_ALL_MODELS = [NodeSet, Method, Machine, RouteSet]
_DEFAULT_SQLITE_TIMEOUT = 15


def open_database(
    filepath: str, create_db: bool = False, timeout: int = _DEFAULT_SQLITE_TIMEOUT
) -> SqliteDatabase:
    """Opens the sqlite database v3 file specified in ``filepath``.

    Args:
      filepath: path to database file
      create_db: True → create a new file if it does not exist
      timeout: seconds to wait for a locked database to be released

    Returns:
      SqliteDatabase object (Peewee)
    """
    filepath = os.path.abspath(os.path.expanduser(str(filepath)))
    if not create_db and not os.path.exists(filepath):
        raise OSError(f'Database file not found: {filepath}')
    db = SqliteDatabase(
        filepath,
        pragmas={'foreign_keys': 1},
        timeout=timeout,
    )
    database_proxy.initialize(db)
    db.connect()
    db.create_tables(_ALL_MODELS)
    return db


@contextmanager
def database_connection(
    filepath: str, create_db: bool = False, timeout: int = _DEFAULT_SQLITE_TIMEOUT
):
    """Open the sqlite database for the duration of a context block.

    Args:
      filepath: path to database file
      create_db: True → create a new file if it does not exist
      timeout: seconds to wait for a locked database to be released

    Yields:
      Connected SqliteDatabase object (Peewee)
    """
    db = open_database(filepath, create_db=create_db, timeout=timeout)
    try:
        yield db
    finally:
        db.close()
