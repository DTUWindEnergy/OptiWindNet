# SPDX-License-Identifier: MIT
# Typing stubs for optiwindnet.db.model
# Resolves Peewee field descriptors to their runtime Python types on instances.

from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from peewee import DatabaseProxy, Model, SqliteDatabase

database_proxy: DatabaseProxy

class BaseModel(Model): ...

class NodeSet(BaseModel):
    digest: bytes
    name: str
    T: int
    R: int
    B: int
    VertexC: bytes
    constraint_groups: Any
    constraint_vertices: Any
    landscape_angle: float | None

class Method(BaseModel):
    digest: bytes
    solver_name: str
    funname: str
    options: dict[str, Any]
    timestamp: datetime
    funfile: str
    funhash: bytes

class Machine(BaseModel):
    id: int
    name: str
    attrs: dict[str, Any] | None

class RouteSet(BaseModel):
    id: int
    handle: str
    valid: bool | None
    T: int
    R: int
    capacity: int
    length: float
    is_normalized: bool
    runtime: float | None
    num_gates: Any
    C: int
    D: int
    creator: str | None
    detextra: float | None
    num_diagonals: int | None
    tentative: Any
    rogue: Any
    timestamp: datetime | None
    misc: Any
    stuntC: bytes | None
    clone2prime: Any
    edges: Any
    nodes: NodeSet
    method: Method
    machine: Machine | None

_ALL_MODELS: list[type[BaseModel]]
_DEFAULT_SQLITE_TIMEOUT: int

def open_database(
    filepath: str,
    create_db: bool = ...,
    timeout: int = ...,
) -> SqliteDatabase: ...
@contextmanager
def database_connection(
    filepath: str,
    create_db: bool = ...,
    timeout: int = ...,
) -> Generator[SqliteDatabase, None, None]: ...
