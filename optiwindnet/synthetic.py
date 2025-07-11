# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull

from .utils import F

__all__ = ()


def toyfarm():
    VertexC = np.array(
        [
            # Terminals
            [49.0, 993.0],  # row 0
            [-145.0, 388.0],  # row 1
            [275.0, 562.0],
            [699.0, 566.0],
            [-371.0, -147.0],  # row 2
            [371.0, 109.0],
            [972.0, 206.0],
            [-585.0, -655.0],  # row 3
            [90.0, -475.0],
            [707.0, -244.0],
            [-104.0, -966.0],  # row 4
            [494.0, -772.0],
            # Border vertices
            [49.0, 993.0],  # row 0
            [-145.0, 388.0],  # row 1
            [-371.0, -147.0],  # row 2
            [-585.0, -655.0],  # row 3
            [-104.0, -966.0],  # row 4
            [494.0, -772.0],
            [707.0, -244.0],
            [972.0, 206.0],
            [699.0, 566.0],
            # Root
            [0.0, 0.0],  # OSS
        ]
    )
    R = 1
    T = 12
    B = 9
    # create networkx graph
    G = nx.Graph(
        R=R,
        T=T,
        B=B,
        VertexC=VertexC,
        border=np.arange(T, T + B),
        name='toy',
        handle='toy',
    )
    G.add_nodes_from(((n, {'label': F[n], 'kind': 'wtg'}) for n in range(T)))
    G.add_nodes_from(((r, {'label': F[r], 'kind': 'oss'}) for r in range(-R, 0)))
    return G


def synthfarm2graph(RootC, NodeC, BoundaryC=None, name='', handle='synthetic'):
    T = NodeC.shape[0]
    R = RootC.shape[0]

    # build data structures
    VertexC = np.vstack((NodeC, RootC))
    if BoundaryC is None:
        hull = ConvexHull(VertexC)
        for v in hull.vertices:
            # hack to avoid error in .mesh.make_planar_embedding()
            vC = VertexC[v]
            VertexC[v] += 1e-6 * vC
        BoundaryC = VertexC[hull.vertices]

    # create networkx graph
    G = nx.Graph(
        R=R, T=T, VertexC=VertexC, boundary=BoundaryC, name=name, handle=handle
    )
    G.add_nodes_from(((n, {'label': F[n], 'kind': 'wtg'}) for n in range(T)))
    G.add_nodes_from(((r, {'label': F[r], 'kind': 'oss'}) for r in range(-R, 0)))
    return G


def equidistant(R, center='centroid', spacing=1):
    """
    Returns an array of coordinates for the vertices of a regular triangular
    tiling (spacing sets the triangle's side) within radius R.
    The coordinate origin is in the centroid of the central triangle.
    """
    lim = (R / spacing) ** 2
    h = np.sqrt(3) / 2

    if center == 'centroid':

        def iswithin(x, y):
            return x**2 + y**2 <= lim

        Vsector = []

        offset = np.sqrt(3) / 3
        i = 0
        repeat = True
        # this loop fills a 120° sector
        while True:
            # x0 = (3*i + 2)*h/3
            x0 = i * h + offset
            if i % 2 == 0 and repeat:
                # add line starting at 0°
                y0 = 0
                repeat = False
            else:
                # add line starting at 60°
                y0 = x0 * h * 2
                repeat = True
                i += 1
            if iswithin(x0, y0):
                Vsector.append((x0, y0))
                c = 1
                while True:
                    x, y = x0 + c * h, y0 + c / 2
                    if iswithin(x, y):
                        Vsector.append((x, y))
                        r = np.sqrt(x**2 + y**2)
                        θ = 2 * np.pi / 3 - np.arctan2(y, x)
                        Vsector.append((r * np.cos(θ), r * np.sin(θ)))
                    else:
                        break
                    c += 1
            else:
                if not repeat:
                    break
        # replicate the 120° sector created to fill the circle
        Vsector = np.array(Vsector)
        r = np.hypot(*Vsector.T)
        θ = np.arctan2(*Vsector.T[::-1])
        cos_sin = tuple(
            np.c_[np.cos(θ + β), np.sin(θ + β)] for β in (2 * np.pi / 3, 4 * np.pi / 3)
        )
        output = np.r_[
            tuple((Vsector,) + tuple(cs * r[:, np.newaxis] for cs in cos_sin))
        ]

    elif center == 'vertex':

        def addupper(x, y):
            X, Y = (x + 0.5, y + h)
            if X**2 + Y**2 <= lim:
                yield X, Y
                yield from addupper(X, Y)

        def addlower(x, y):
            X, Y = (x + 1, y)
            if X**2 + Y**2 <= lim:
                yield X, Y
                yield from addlower(X, Y)

        def addbranches(x, y):
            yield from addlower(x, y)
            X, Y = (x + 1.5, y + h)
            if X**2 + Y**2 <= lim:
                yield X, Y
                yield from addbranches(X, Y)
            yield from addupper(x, y)

        firstbranch = (1.5, h)
        Vsector = np.array(
            tuple(addlower(0, 0)) + (firstbranch,) + tuple(addbranches(*firstbranch))
        )

        # replicate the 60° sector created to fill the circle
        Vsector = np.array(Vsector)
        r = np.hypot(*Vsector.T)
        θ = np.arctan2(*Vsector.T[::-1])
        cos_sin = tuple(
            np.c_[np.cos(θ + β), np.sin(θ + β)] for β in np.pi / 3 * np.arange(1, 6)
        )
        output = np.r_[
            tuple(
                (np.zeros((1, 2), dtype=float),)
                + (Vsector,)
                + tuple(cs * r[:, np.newaxis] for cs in cos_sin)
            )
        ]
    else:
        print('Unknown option for <center>:', center)
        return None
    return spacing * output
