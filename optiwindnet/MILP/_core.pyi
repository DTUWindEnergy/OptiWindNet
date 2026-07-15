# SPDX-License-Identifier: MIT
# Typing stubs for optiwindnet.MILP._core
# Surfaces the dynamically generated ModelOptions.__init__ signature.

import abc
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import networkx as nx

def physical_core_count() -> int: ...
def feeder_and_load_bounds(
    T: int,
    capacity: int,
    feeder_limit: FeederLimit,
    max_feeders: int,
    balanced: bool,
    feeders_per_subtree: int = ...,
) -> tuple[int, int | None, int | None, int | None]: ...

class OWNWarmupFailed(Exception): ...
class OWNSolutionNotFound(Exception): ...

class Topology(StrEnum):
    RADIAL: str
    BRANCHED: str
    RINGED: str
    DEFAULT: str

class FeederRoute(StrEnum):
    STRAIGHT: str
    SEGMENTED: str
    DEFAULT: str

class FeederLimit(StrEnum):
    UNLIMITED: str
    EXACTLY: str
    SPECIFIED: str
    MINIMUM: str
    MIN_PLUS1: str
    MIN_PLUS2: str
    MIN_PLUS3: str
    DEFAULT: str

_Link = tuple[int, int]

class ModelOptions(dict[str, Any]):
    """Hold options for the modelling of the cable routing problem."""

    hints: dict[str, type[StrEnum]]
    simple: dict[str, tuple[type, Any, str]]

    def __init__(
        self,
        *,
        topology: Topology | str = ...,
        feeder_route: FeederRoute | str = ...,
        feeder_limit: FeederLimit | str = ...,
        balanced: bool = ...,
        max_feeders: int = ...,
    ) -> None: ...
    @classmethod
    def help(cls) -> None: ...

@dataclass(slots=True)
class ModelMetadata:
    R: int
    T: int
    capacity: int
    linkset: tuple[_Link, ...]
    link_: Mapping[_Link, Any]
    flow_: Mapping[_Link, Any]
    model_options: dict[str, Any]
    fun_fingerprint: dict[str, str | bytes]
    weight_: tuple[float, ...] = ()
    solution_hint: dict[Any, float] = field(default_factory=dict)
    warmed_by: str = ''

@dataclass(slots=True)
class SolutionInfo:
    runtime: float
    bound: float
    objective: float
    relgap: float
    termination: str

def ringed_warmstart_values(
    metadata: ModelMetadata, S: nx.Graph
) -> tuple[dict[_Link, int], dict[_Link, int]]: ...

class Solver(abc.ABC):
    name: str
    metadata: ModelMetadata
    solver: Any
    options: dict[str, Any]
    stopping: dict[str, Any]
    solution_info: SolutionInfo
    applied_options: dict[str, Any]

    @abc.abstractmethod
    def _link_val(self, var: Any) -> int | bool: ...
    @abc.abstractmethod
    def _flow_val(self, var: Any) -> int: ...
    @abc.abstractmethod
    def set_problem(
        self,
        P: nx.PlanarEmbedding,
        A: nx.Graph,
        capacity: int,
        model_options: ModelOptions,
        warmstart: nx.Graph | None = None,
    ) -> None: ...
    @abc.abstractmethod
    def solve(
        self,
        time_limit: float,
        mip_gap: float,
        options: dict[str, Any] = ...,
        verbose: bool = False,
    ) -> SolutionInfo: ...
    @abc.abstractmethod
    def get_solution(self, A: nx.Graph | None = None) -> tuple[nx.Graph, nx.Graph]: ...

class PoolHandler(abc.ABC):
    name: str
    num_solutions: int
    model_options: ModelOptions

    @abc.abstractmethod
    def _objective_at(self, index: int) -> float: ...
    @abc.abstractmethod
    def _topology_from_mip_pool(self) -> nx.Graph: ...
