# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import abc
import networkx as nx
import logging

from ..interarraylib import G_from_S
from ..pathfinding import PathFinder


logger = logging.getLogger(__name__)
info = logger.info


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
            info(f'Pool investigation over - next best undetoured length: {λ:.3f}')
            break
        S = pool.S_from_pool()
        G = G_from_S(S, A)
        Hʹ = PathFinder(G, planar=P, A=A).create_detours()
        Λʹ = Hʹ.size(weight='length')
        if Λʹ < Λ:
            H, Λ = Hʹ, Λʹ
            pool_index = i
            pool_objective = λ
            info(f'Incumbent has (detoured) length: {Λ:.3f}')
    H.graph['pool_count'] = num_solutions
    if pool_index > 0:
        H.graph['pool_entry'] = pool_index, pool_objective
    return H

