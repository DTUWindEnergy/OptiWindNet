# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

from functools import partial

import networkx as nx
import numpy as np
import numpy.lib.recfunctions as nprec

from .heuristics import CPEW, NBEW, OBEW
from .interarraylib import F, calcload

__all__ = ('assign_cables',)

heuristics = {
    'CPEW': CPEW,
    'NBEW': NBEW,
    'OBEW': OBEW,
    'OBEW_0.6': partial(OBEW, rootlust='0.6*cur_capacity/capacity'),
}


def translate2global_optimizer(G):
    VertexC = G.graph['VertexC']
    R = G.graph['R']
    T = G.graph['T']
    X, Y = np.hstack((VertexC[-1 : -1 - R : -1].T, VertexC[:T].T))
    return dict(WTc=T, OSSc=R, X=X, Y=Y)


def assign_cables(G, cables):
    """Assign a cable type to each edge of `G` and update attribute 'cost'.

    Each edge is assigned the cheapest cable type that can carry its load. The
    edge attribute 'cable' is the index in `cables` of the type chosen.

    Args:
      G: networkx graph with edges having a 'load' attribute (use calcload(G))
      cables: [(«capacity», «cost»), ...] in increasing capacity order (each
        cable entry must be a tuple) or numpy.ndarray where each row represents
        one cable type
    """

    Nc = len(cables)
    dt = np.dtype([('capacity', int), ('cost', float)])
    if isinstance(cables, np.ndarray):
        cable_ = nprec.unstructured_to_structured(cables, dtype=dt)
    else:
        cable_ = np.fromiter(cables, dtype=dt, count=Nc)
    capacity_ = cable_['capacity']
    capacity = 1

    # for e, data in G.edges.items():
    for u, v, data in G.edges(data=True):
        i = capacity_.searchsorted(data['load']).item()
        if i >= len(capacity_):
            print(
                f'ERROR: Load for edge ⟨{u, v}⟩: {data["load"]} '
                f'exceeds maximum cable capacity {capacity_[-1]}.'
            )
        data['cable'] = i
        data['cost'] = data['length'] * cable_['cost'][i].item()
        if data['load'] > capacity:
            capacity = data['load']
    G.graph['cables'] = cable_
    G.graph['has_costs'] = True
    if 'capacity' not in G.graph:
        G.graph['capacity'] = capacity


def assign_subtree(G):
    start = 0
    queue = []
    for root in range(-G.graph['R'], 0):
        for subtree, gate in enumerate(G[root], start=start):
            queue.append((root, gate))
            while queue:
                parent, node = queue.pop()
                G.nodes[node]['subtree'] = subtree
                for nbr in G[node]:
                    if nbr != parent:
                        queue.append((node, nbr))
        start = subtree + 1


def L_from_XYR(X, Y, R=1, name='unnamed', borderC=None):
    """Create location graph L from node coordinates split in X and Y.

    This function assumes that the first R coordinates in X/Y are OSSs.

    Args:
      X: x coordinates of nodes
      Y: y coordinates of nodes
      R: number of OSSs

    Returns:
      Location graph L.
    """
    assert len(X) == len(Y), 'ERROR: X and Y lengths must match'
    T = len(X) - R

    # create networkx graph
    if borderC is None:
        borderC = np.array(
            ((min(X), min(Y)), (min(X), max(Y)), (max(X), max(Y)), (max(X), min(Y)))
        )
    B = borderC.shape[0]
    border = list(range(T, T + B))
    L = nx.Graph(
        R=R,
        T=T,
        B=B,
        border=border,
        name=name,
        VertexC=np.r_[
            np.c_[X[R:], Y[R:]], np.c_[X[R - 1 :: -1], Y[R - 1 :: -1]], borderC
        ],
    )
    L.add_nodes_from(((n, {'label': F[n], 'kind': 'wtg'}) for n in range(T)))
    L.add_nodes_from(((r, {'label': F[r], 'kind': 'oss'}) for r in range(-R, 0)))
    return L


def G_from_table(
    table: np.ndarray[:, :],
    G_base: nx.Graph,
    capacity: int | None = None,
    cost_scale: float = 1e3,
) -> nx.Graph:
    """Create a networkx Graph with nodes and data from G_base and edges from
    a table.

    (e.g. the S matrix of juru's `global_optimizer`)

    Args:
      table: [ [u, v, length, cable type, load (WT number), cost] ]
    """
    G = nx.Graph()
    G.graph.update(G_base.graph)
    G.add_nodes_from(G_base.nodes(data=True))
    R = G_base.graph['R']

    # indexing differences:
    # table starts at 1, while G starts at -R
    edges = table[:, :2].astype(int) - R - 1

    G.add_edges_from(edges)
    nx.set_edge_attributes(
        G,
        {
            (int(u), int(v)): dict(length=length, cable=cable, load=load, cost=cost)
            for (u, v), length, (cable, load), cost in zip(
                edges, table[:, 2], table[:, 3:5].astype(int), cost_scale * table[:, 5]
            )
        },
    )
    G.graph['has_loads'] = True
    G.graph['has_costs'] = True
    G.graph['creator'] = 'G_from_table()'
    if capacity is not None:
        G.graph['capacity'] = capacity
    return G


def G_from_TG(S, G_base, capacity=None, load_col=4):
    """DEPRECATED in favor of `G_from_table()`

    Creates a networkx graph with nodes and data from G_base and edges from
    a S matrix.
    S matrix: [ [u, v, length, load (WT number), cable type], ...]
    """
    G = nx.Graph()
    G.graph.update(G_base.graph)
    G.add_nodes_from(G_base.nodes(data=True))
    R = G_base.graph['R']
    T = G_base.graph['T']

    # indexing differences:
    # S starts at 1, while G starts at 0
    # S begins with OSSs followed by WTGs,
    # while G begins with WTGs followed by OSSs
    # the line bellow converts the indexing:
    edges = (S[:, :2].astype(int) - R - 1) % (T + R)

    G.add_weighted_edges_from(zip(*edges.T, S[:, 2]), weight='length')
    # nx.set_edge_attributes(G, {(u, v): load for (u, v), load
    #                            in zip(edges, S[:, load_col])},
    #                        name='load')
    # try:
    calcload(G)
    # except AssertionError as err:
    #     print(f'>>>>>>>> SOMETHING WENT REALLY WRONG: {err} <<<<<<<<<<<')
    #     return G
    if S.shape[1] >= 4:
        for (u, v), load in zip(edges, S[:, load_col]):
            Gload = G.edges[u, v]['load']
            assert Gload == load, f'<G.edges[{u}, {v}]> {Gload} != {load} <S matrix>'
    G.graph['has_loads'] = True
    G.graph['creator'] = 'G_from_TG()'
    G.graph['prevented_crossings'] = 0
    return G


def table_from_G(G):
    """Create a table representing the edges of G.

    Args:
      G: graph to convert to table

    Returns:
      table: [ («u», «v», «length», «load (WT number)», «cable type»,
        «edge cost»), ...] (table is a numpy record array)
    """
    R = G.graph['R']
    Ne = G.number_of_edges()

    def edge_parser(edges):
        for u, v, data in edges:
            # OSS index starts at 0
            # u = (u + R) if u > 0 else abs(u) - 1
            # v = (v + R) if v > 0 else abs(v) - 1
            # OSS index starts at 1
            s = (u + R + 1) if u >= 0 else abs(u)
            t = (v + R + 1) if v >= 0 else abs(v)
            # print(u, v, '->', s, t)
            yield (s, t, data['length'], data['load'], data['cable'], data['cost'])

    table = np.fromiter(
        edge_parser(G.edges(data=True)),
        dtype=[
            ('u', int),
            ('v', int),
            ('length', float),
            ('load', int),
            ('cable', int),
            ('cost', float),
        ],
        count=Ne,
    )
    return table


class HeuristicFactory:
    """Initializes a heuristic algorithm.

    Args:
      T: number of nodes
      R: number of roots
      rootC: 2D nympy array (R, 2) of the XY coordinates of the roots
      boundaryC: 2D numpy array (_, 2) of the XY coordinates of the boundary
      cables: [(«cross section», «capacity», «cost»), ...] ordered by capacity
      name: site name

    (increasing capacity along cables' elements)
    """

    def __init__(self, T, R, rootC, boundaryC, heuristic, cables, name='unnamed'):
        self.T = T
        self.R = R
        self.cables = cables
        self.k = cables[-1][1]
        self.VertexC = np.empty((T + R, 2), dtype=float)
        self.VertexC[T:] = rootC
        # create networkx graph
        self.G_base = nx.Graph(R=R, VertexC=self.VertexC, boundary=boundaryC, name=name)
        self.G_base.add_nodes_from(
            ((n, {'label': F[n], 'kind': 'wtg'}) for n in range(T))
        )
        self.G_base.add_nodes_from(
            ((r, {'label': F[r], 'kind': 'oss'}) for r in range(-R, 0))
        )
        self.heuristic = heuristics[heuristic]

    def calccost(self, X, Y):
        assert len(X) == len(Y) == self.T
        self.VertexC[: self.T, 0] = X
        self.VertexC[: self.T, 1] = Y
        self.G = self.heuristic(self.G_base, capacity=self.k)
        calcload(self.G)
        assign_cables(self.G, self.cables)
        return self.G.size(weight='cost')

    def get_table(self):
        """Create a table representing the edges of the solution.

        Must have called cost() at least once. Only the last call's layout is
        available.

        Returns:
          table: [ («u», «v», «length», «load (WT number)», «cable type»,
            «edge cost»), ...] (table is a numpy record array)
        """
        return table_from_G(self.G)


def heuristic_wrapper(X, Y, cables, R=1, heuristic='CPEW', return_graph=False):
    """Run a heuristic on a location defined by X/Y coordinates.

    This function assumes that the first R coordinates in X/Y are OSSs.
    (increasing capacity along `cables`' elements)

    Args:
      X: x coordinates of nodes
      Y: y coordinates of nodes
      R: number of OSSs
      cables: [(«cross section», «capacity», «cost»), ...] ordered by capacity
      R: number of OSSs
      heuristic: {'CPEW', 'OBEW'}

    Returns:
      Location graph L.
    """
    G_base = L_from_XYR(X, Y, R)
    G = heuristics[heuristic](G_base, capacity=cables[-1][1])
    calcload(G)
    assign_cables(G, cables)
    if return_graph:
        return table_from_G(G), G
    else:
        return table_from_G(G)
