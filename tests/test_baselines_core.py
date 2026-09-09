# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import networkx as nx
import numpy as np

from optiwindnet.baselines._core import scaled_length_block


def _make_A(coords, edges=()) -> nx.Graph:
    VertexC = np.asarray(coords, dtype=float)
    T = len(VertexC) - 1
    A = nx.Graph(T=T, R=1, VertexC=VertexC)
    A.add_nodes_from(range(-1, T))
    for u, v, length in edges:
        A.add_edge(u, v, length=length)
    return A


def test_scaled_length_block_writes_only_A_links():
    A = _make_A([(0.0, 0.0), (3.0, 0.0), (0.0, 4.0), (0.0, 0.0)], [(0, 1, 3.0)])
    block, fill_max, edge_max = scaled_length_block(
        A, [0, 1, 2], scale=1.0, complete=False, absent=np.inf, dtype=np.float64
    )
    assert block[0, 1] == block[1, 0] == 3.0
    assert np.isinf(block[0, 2]) and np.isinf(block[1, 2])
    assert fill_max == 0.0
    assert edge_max == 3.0


def test_scaled_length_block_complete_fills_every_pair():
    A = _make_A([(0.0, 0.0), (3.0, 0.0), (0.0, 4.0), (0.0, 0.0)])
    block, fill_max, edge_max = scaled_length_block(
        A, [0, 1, 2], scale=1.0, complete=True, absent=np.inf, dtype=np.float64
    )
    assert np.isfinite(block).all()
    np.testing.assert_allclose(np.diagonal(block), 0.0)
    assert block[0, 1] == 3.0 and block[0, 2] == 4.0 and block[1, 2] == 5.0
    assert fill_max == 5.0
    assert edge_max == 0.0


def test_scaled_length_block_prefers_the_length_A_carries():
    """Stored link lengths take precedence over Euclidean distances."""
    A = _make_A([(0.0, 0.0), (3.0, 0.0), (0.0, 4.0), (0.0, 0.0)], [(0, 1, 7.5)])
    block, fill_max, edge_max = scaled_length_block(
        A, [0, 1, 2], scale=1.0, complete=True, absent=np.inf, dtype=np.float64
    )
    assert block[0, 1] == block[1, 0] == 7.5
    assert block[0, 2] == 4.0  # untouched by A
    assert (fill_max, edge_max) == (5.0, 7.5)


def test_scaled_length_block_scales_and_rounds_for_an_integer_dtype():
    A = _make_A([(0.0, 0.0), (3.0, 0.0), (0.0, 4.0), (0.0, 0.0)], [(0, 1, 0.26)])
    block, _, edge_max = scaled_length_block(
        A, [0, 1, 2], scale=10.0, complete=True, absent=999, dtype=np.int32
    )
    assert block.dtype == np.int32
    assert block[0, 1] == 3  # round(0.26 * 10)
    assert block[0, 2] == 40
    assert edge_max == 2.6  # the maxima are reported unrounded


def test_scaled_length_block_orders_by_the_terminals_given():
    A = _make_A([(0.0, 0.0), (3.0, 0.0), (0.0, 4.0), (0.0, 0.0)], [(0, 1, 3.0)])
    block, _, _ = scaled_length_block(
        A, [2, 1, 0], scale=1.0, complete=False, absent=np.inf, dtype=np.float64
    )
    assert block[1, 2] == block[2, 1] == 3.0  # link (0, 1) at indices of 1 and 0
