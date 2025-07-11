# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/
"""Database model v1 for storage of locations and route sets.

Tables:
  - NodeSet: location definition
  - RouteSet: routeset (i.e. a record of G)
  - Method: info on algorithm & options to produce routesets
  - Machine: info on machine that generated a routeset
"""

import datetime
import os

from pony.orm import Database, IntArray, Json, Optional, PrimaryKey, Required, Set

from ._core import _naive_utc_now

__all__ = ()


def open_database(filepath: str, create_db: bool = False) -> Database:
    """Opens the sqlite database v1 file specified in `filepath`.

    Args:
      filepath: path to database file
      create_db: True -> create a new file if it does not exist

    Returns:
      Database object (Pony ORM)
    """
    db = Database()
    define_entities(db)
    db.bind(
        'sqlite', os.path.abspath(os.path.expanduser(filepath)), create_db=create_db
    )
    db.generate_mapping(create_tables=True)
    return db


def define_entities(db: Database):
    class NodeSet(db.Entity):
        # hashlib.sha256(VertexC + boundary).digest()
        name = Required(str, unique=True)
        T = Required(int)  # # of non-root nodes
        R = Required(int)  # # of root nodes
        # vertices (nodes + roots) coordinates (UTM)
        # pickle.dumps(np.empty((T + R, 2), dtype=float)
        VertexC = Required(bytes)
        # region polygon: P vertices (x, y), ordered ccw
        # pickle.dumps(np.empty((P, 2), dtype=float)
        boundary = Required(bytes)
        landscape_angle = Optional(float)
        digest = PrimaryKey(bytes)
        EdgeSets = Set(lambda: EdgeSet)

    class EdgeSet(db.Entity):
        id = PrimaryKey(int, auto=True)
        handle = Required(str)
        capacity = Required(int)
        length = Required(float)
        # runtime always in [s]
        runtime = Optional(float)
        machine = Optional(lambda: Machine)
        gates = Required(IntArray)
        T = Required(int)
        R = Required(int)
        # number of Detour nodes
        D = Optional(int, default=0)
        timestamp = Optional(datetime.datetime, default=_naive_utc_now)
        misc = Optional(Json)
        clone2prime = Optional(IntArray)
        edges = Required(IntArray)
        nodes = Required(NodeSet)
        method = Required(lambda: Method)

    class Method(db.Entity):
        funname = Required(str)
        # options is a dict of function parameters
        options = Required(Json)
        timestamp = Required(datetime.datetime, default=_naive_utc_now)
        funfile = Required(str)
        # hashlib.sha256(fun.__code__.co_code)
        funhash = Required(bytes)
        # hashlib.sha256(funhash + pickle(options)).digest()
        digest = PrimaryKey(bytes)
        EdgeSets = Set(EdgeSet)

    class Machine(db.Entity):
        name = Required(str, unique=True)
        attrs = Optional(Json)
        EdgeSets = Set(EdgeSet)
