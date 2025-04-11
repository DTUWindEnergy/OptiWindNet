# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import abc
from typing import Any
import networkx as nx
import logging

from ..interarraylib import G_from_S
from ..pathfinding import PathFinder

logger = logging.getLogger(__name__)
error, info = logger.error, logger.info


class Solver(abc.ABC):

    @abc.abstractmethod
    def set_problem(self, P, A, capacity, **kwargs):
        pass

    @abc.abstractmethod
    def solve(self, timelimit: int, mipgap: float,
              options: dict[str, Any] = {}, verbose: bool = False) -> tuple:
        pass

    @abc.abstractmethod
    def get_solution(self) -> nx.Graph:
        pass

class PoolHandler(abc.ABC):
    num_solutions: int

    @abc.abstractmethod
    def objective_at(self, index: int) -> float:
        'Get objective value from solution pool at position `index`'
        pass
    
    @abc.abstractmethod
    def S_from_pool(self) -> nx.Graph:
        'Build S from the pool solution at the last requested position'
        pass


def summarize_result(result):
    objective = result['Problem'][0]['Upper bound']
    bound = result['Problem'][0]['Lower bound']
    relgap = 1. - bound/objective
    termination = result['Solver'][0]['Termination condition'].name
    info('objective: %f, bound: %f, gap: %.4f, termination: %s',
         objective, bound, relgap, termination)
    return objective, bound, relgap, termination  # runtime, num_solutions


def investigate_pool(P: nx.PlanarEmbedding, A: nx.Graph, pool: PoolHandler
        ) -> nx.Graph:
    '''Go through the CpSat's solutions checking which has the shortest length
    after applying the detours with PathFinder.'''
    Λ = float('inf')
    num_solutions = pool.num_solutions
    info(f'Solution pool has {num_solutions} solutions.')
    for i in range(num_solutions):
        λ = pool.objective_at(i)
        #  print(f'λ[{i}] = {λ}')
        if λ > Λ:
            info(f"#{i} halted pool search: objective ({λ:.3f}) > incumbent's length")
            break
        S = pool.S_from_pool()
        G = G_from_S(S, A)
        Hʹ = PathFinder(G, planar=P, A=A).create_detours()
        Λʹ = Hʹ.size(weight='length')
        if Λʹ < Λ:
            H, Λ = Hʹ, Λʹ
            pool_index = i
            pool_objective = λ
            info(f'#{i} -> incumbent (objective: {λ:.3f}, length: {Λ:.3f})')
        else:
            info(f'#{i} discarded (objective: {λ:.3f}, length: {Λ:.3f})')            
    H.graph['pool_count'] = num_solutions
    if pool_index > 0:
        H.graph['pool_entry'] = pool_index, pool_objective
    return H

