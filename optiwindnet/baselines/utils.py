# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform


def length_matrix_single_depot_from_G(
    A: nx.Graph, *, scale: float, complete: bool = False
) -> tuple[np.ndarray, float]:
    """Edge length matrix for VRP-based solvers.
    It is assumed that the problem has been pre-scaled, such that multiplying
    all lengths by ``scale`` will place them within a numerically stable range.
    Length of return to depot from all nodes is set to 0 (i.e. Open-VRP).
    Order of nodes in the returned matrix is depot, clients (required by some
    VRP methods), which differs from OptiWindNet order (i.e., clients, depot).

    Args:
      A: Must contain graph attributes ``'R'``, ``'T'``, ``'VertexC'`` and
        ``'d2roots'``.
        A's edges must have the ``'length'`` attribute.
      scale: Factor to multiply all lengths by.
      complete: make the full graph over A available (links not in A assumed direct)

    Returns:
      ``(Λ, λ_max)`` - matrix of lengths and maximum length value (below ``+inf``).
    """
    R, T, d2roots = (A.graph[k] for k in ('R', 'T', 'd2roots'))
    assert R == 1, 'ERROR: only single depot supported'
    if complete:
        # bring depot to before the clients
        VertexC = A.graph['VertexC']
        VertexCmod = np.r_[VertexC[-R:], VertexC[:T]]
        Λv = pdist(VertexCmod) * scale
        λ_max = Λv.max()
        Λ = squareform(Λv)
    else:
        # non-available edges map to infinite length
        Λ = np.full((T + R, T + R), np.inf)
        λ_max = d2roots[:T, 0].max() * scale
    for u, v, length in A.edges(data='length'):
        scaled_length = length * scale
        Λ[u + 1, v + 1] = Λ[v + 1, u + 1] = scaled_length
        λ_max = max(λ_max, scaled_length)
    Λ[0, 1:] = d2roots[:T, 0] * scale
    # make return to depot always free
    Λ[:, 0] = 0.0
    return Λ, λ_max
