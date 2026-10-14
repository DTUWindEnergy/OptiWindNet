# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Regression checks for straight feeder constraints."""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from optiwindnet.api import EWRouter, NetworkConstraints, WindFarmNetwork
from optiwindnet.crossings import _feeder_intersections, _feeder_link_conflicts
from optiwindnet.heuristics import constructor
from optiwindnet.interarraylib import add_link_blockmap
from optiwindnet.mesh import make_planar_embedding
from optiwindnet.synthetic import L_from_synthetic
from optiwindnet.validating import validate_routeset

_REPO = Path(__file__).resolve().parents[1]


def _hi31_network():
    return WindFarmNetwork.from_pbf(
        _REPO / 'docs/notebooks/data/DTU_letters.osm.pbf',
        cables=[(2, 500.0), (5, 800.0)],
        constraints=NetworkConstraints(feeder_route='straight'),
    )


def test_hi31_straight_feeders_do_not_block_links_by_obstructed_headings():
    wfn = _hi31_network()
    wfn.optimize(router=EWRouter())
    A = wfn.A.copy()
    A.remove_node(-1)
    add_link_blockmap(A)

    assert -1 in A.nodes[9]['los_d2root']
    for u, v in ((6, 8), (16, 17)):
        assert A[u][v]['blocked__'][-1][9]
        assert wfn.S.has_edge(u, v)
    # Feeder -1→10 has LOS and still prevents the link 12–13.
    assert -1 not in A.nodes[10].get('los_d2root', {})
    assert A[12][13]['blocked__'][-1][10]
    assert wfn.S.has_edge(-1, 10)
    assert not wfn.S.has_edge(12, 13)
    assert validate_routeset(wfn.G) == []


@pytest.mark.parametrize('obstructed_root', (None, -2, -1))
def test_constructor_excludes_only_the_obstructed_root(obstructed_root):
    L = L_from_synthetic(
        np.array([[0.0, 0.0], [0.0, -10.0]]),
        np.array([[2.0, -0.5], [2.0, 0.5], [3.0, 0.0]]),
        BorderC=np.array([[-1.0, -11.0], [4.0, -11.0], [4.0, 2.0], [-1.0, 2.0]]),
    )
    _, A = make_planar_embedding(L)
    # Terminal 2 remains a feeder while 0–1 is the only candidate union.
    A.remove_edges_from([e for e in A.edges if set(e) != {0, 1}])
    if obstructed_root is not None:
        A.nodes[2]['los_d2root'] = {obstructed_root: 3.0}
    S = constructor(
        A,
        capacity=2,
        method='biased_EW',
        straight_feeder_route=True,
        weigh_detours=False,
    )

    assert S.has_edge(-2, 2)
    assert S.has_edge(0, 1) == (obstructed_root == -2)


@pytest.mark.parametrize('obstructed_root', (None, -2, -1))
def test_milp_conflicts_exclude_only_the_obstructed_root(obstructed_root):
    # Both roots' feeders to 2 cross 0–1. Terminal 3 is a LOS counterexample.
    A = nx.Graph(
        T=4,
        R=2,
        B=0,
        VertexC=np.array(
            [
                [2.0, -1.0],
                [2.0, 1.0],
                [3.0, 0.0],
                [4.0, 0.0],
                [0.0, -0.2],
                [0.0, 0.2],
            ]
        ),
    )
    A.add_nodes_from(range(4))
    A.add_nodes_from((-2, -1))
    A.add_edge(0, 1)
    if obstructed_root is not None:
        A.nodes[2]['los_d2root'] = {obstructed_root: 3.0}
    crossings, passovers = _feeder_link_conflicts(A)

    assert set(map(tuple, crossings.tolist())) == {
        (0, 1, r, n) for r in (-2, -1) for n in (2, 3) if n != 2 or r != obstructed_root
    }
    assert passovers.size == 0
    # Routing still detects the geometric intersections of obstructed feeders.
    assert len(_feeder_intersections(A)) == 4


def test_milp_obstructed_feeder_has_no_passover_conflicts():
    A = nx.Graph(
        T=4,
        R=1,
        B=0,
        VertexC=np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0], [1.0, -1.0], [0.0, 0.0]]),
    )
    A.add_nodes_from(range(4))
    A.add_node(-1)
    A.add_edges_from(((0, 2), (0, 3)))
    assert _feeder_link_conflicts(A)[1].tolist() == [[-1, 1, 0, 2, 0, 3]]
    A.nodes[1]['los_d2root'] = {-1: 2.0}
    crossings, passovers = _feeder_link_conflicts(A)
    assert crossings.size == passovers.size == 0
