# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import abc
from enum import StrEnum, auto
from typing import Any
import networkx as nx
import logging
from makefun import with_signature

from ..interarraylib import G_from_S
from ..pathfinding import PathFinder

logger = logging.getLogger(__name__)
error, info = logger.error, logger.info


def _to_kw(c: type) -> str:
    s = c.__name__
    return s[0].lower() + ''.join('_' + c.lower() if c.isupper()
                                  else c for c in s[1:])

class Topology(StrEnum):
    'Set the topology of subtrees in the solution.'
    RADIAL = auto()
    BRANCHED = auto()


class FeederRoute(StrEnum):
    'If feeder routes must be "straight" or can be detoured ("segmented").'
    STRAIGHT = auto()
    SEGMENTED = auto()


class FeederLimit(StrEnum):
    'Whether to limit the maximum number of feeders, if set to "specified", '\
    'additional kwarg "max_feeders" must be given.'
    UNLIMITED = auto()
    SPECIFIED = auto()
    MINIMUM = auto()
    MIN_PLUS1 = auto()
    MIN_PLUS2 = auto()
    MIN_PLUS3 = auto()


class ModelOptions(dict):
    hints = {_to_kw(kind): kind for kind in (Topology, FeederRoute, FeederLimit)}
    @with_signature(
        '__init__(self, '
        + ', '.join(f'{k}: {v.__name__}' for k, v in hints.items()) + ')'
    )
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, str):
                kwargs[k] = self.hints[k](v)
        super().__init__(kwargs)

    @classmethod
    def help(cls):
        for k, v in cls.hints.items():
            print(f'{k} in {{'
                  f'{", ".join(f"\"{m}\"" for m in v.__members__.values())}'
                  f'}}\n    {v.__doc__}\n')


class Solver(abc.ABC):

    @abc.abstractmethod
    def set_problem(self, P, A, capacity, **kwargs):
        pass

    @abc.abstractmethod
    def solve(self, timelimit: int, mipgap: float,
              options: dict[str, Any] = {}, verbose: bool = False) -> tuple:
        pass

    @abc.abstractmethod
    def get_solution(self) -> tuple[nx.Graph, nx.Graph]:
        pass

class PoolHandler(abc.ABC):
    num_solutions: int

    @abc.abstractmethod
    def objective_at(self, index: int) -> float:
        'Get objective value from solution pool at position `index`'
        pass
    
    @abc.abstractmethod
    def topo_from_pool(self) -> nx.Graph:
        'Build S from the pool solution at the last requested position'
        pass


def investigate_pool(P: nx.PlanarEmbedding, A: nx.Graph, pool: PoolHandler
        ) -> nx.Graph:
    '''Go through the CpSat's solutions checking which has the shortest length
    after applying the detours with PathFinder.'''
    Λ = float('inf')
    num_solutions = pool.num_solutions
    info(f'Solution pool has {num_solutions} solutions.')
    for i in range(num_solutions):
        λ = pool.objective_at(i)
        if λ > Λ:
            info(f"#{i} halted pool search: objective ({λ:.3f}) > incumbent's length")
            break
        Sʹ = pool.S_from_pool()
        Gʹ = PathFinder(G_from_S(Sʹ, A), planar=P, A=A).create_detours()
        Λʹ = Gʹ.size(weight='length')
        if Λʹ < Λ:
            S, G, Λ = Sʹ, Gʹ, Λʹ
            G.graph['pool_entry'] = i, λ
            info(f'#{i} -> incumbent (objective: {λ:.3f}, length: {Λ:.3f})')
        else:
            info(f'#{i} discarded (objective: {λ:.3f}, length: {Λ:.3f})')            
    G.graph['pool_count'] = num_solutions
    return S, G

