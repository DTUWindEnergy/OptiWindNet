# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/
"""Database model v0 for storage of locations and route sets.

Tables:
  - NodeSet: location definition
  - RouteSet: routeset (i.e. a record of G)
  - Method: info on algorithm & options to produce routesets
  - Machine: info on machine that generated a routeset
"""

import datetime
import os

from pony.orm import Database, IntArray, Optional, PrimaryKey, Required, Set

from ._core import _naive_utc_now

__all__ = ()


def open_database(filepath: str, create_db: bool = False) -> Database:
    """Opens the sqlite database v2 file specified in `filepath`.

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
        digest = PrimaryKey(bytes)
        name = Required(str, unique=True)
        handle = Required(str, unique=True)
        T = Required(int)  # # of non-root nodes
        R = Required(int)  # # of root nodes
        # vertices (nodes + roots) coordinates (UTM)
        # pickle.dumps(np.empty((T + R, 2), dtype=float)
        VertexC = Required(bytes)
        # region polygon: P vertices (x, y), ordered ccw
        # pickle.dumps(np.empty((P, 2), dtype=float)
        boundary = Required(bytes)
        landscape_angle = Optional(float)
        EdgeSets = Set(lambda: EdgeSet)

    class EdgeSet(db.Entity):
        nodes = Required(NodeSet)
        # edges = pickle.dumps(
        # np.array([(u, v)
        #           for u, v in G.edges], dtype=int))
        edges = Required(bytes)
        length = Required(float)
        # number of Detour nodes
        D = Optional(int, default=0)
        clone2prime = Optional(IntArray)
        gates = Required(IntArray)
        method = Required(lambda: Method)
        capacity = Required(int)
        # cables = Optional(lambda: CableSet)
        runtime = Optional(float)
        runtime_unit = Optional(str)
        machine = Optional(lambda: Machine)
        timestamp = Optional(datetime.datetime, default=_naive_utc_now)
        # DetourC = Optional(bytes)  # superceeded by D and clone2prime
        # misc is a pickled python dictionary
        misc = Optional(bytes)

    class Method(db.Entity):
        # hashlib.sha256(funhash + options).digest()
        digest = PrimaryKey(bytes)
        funname = Required(str)
        # hashlib.sha256(esauwilliams.__code__.co_code)
        funhash = Required(bytes)
        # capacity = Required(int)
        # options is a dict of function parameters
        options = Required(str)
        timestamp = Required(datetime.datetime, default=_naive_utc_now)
        EdgeSets = Set(EdgeSet)

    class Machine(db.Entity):
        name = Required(str, unique=True)
        EdgeSets = Set(EdgeSet)
