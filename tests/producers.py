# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Process-local caches for expensive, read-only producer topology fixtures."""

from collections.abc import Callable
from functools import cache, partial

import networkx as nx

from optiwindnet.baselines.hgs import hgs_cvrp
from optiwindnet.baselines.lkh import lkh3
from optiwindnet.heuristics import constructor
from optiwindnet.transforming import as_normalized

from .cases import BaselineCase, ConstructorCase, case_node_id, expected_topology
from .helpers import run_with_retry, terminal_terminal_crossings
from .sitecache import get_bundle
from .topology_assertions import assert_topology


@cache
def constructor_topology(case: ConstructorCase):
    """Build a typed constructor case once for topology and PathFinder consumers."""
    A = get_bundle(case.site).A
    return constructor(
        A,
        capacity=case.capacity,
        method=case.method,
        bias_margin=case.bias_margin,
        weigh_detours=case.feeder_route.value == 'segmented',
        straight_feeder_route=case.feeder_route.value == 'straight',
    )


def _baseline_topology(case: BaselineCase, solve: Callable[..., nx.Graph]) -> nx.Graph:
    """Produce a baseline case, retrying with a longer limit if it is invalid."""
    A = as_normalized(get_bundle(case.site).A)
    VertexC = A.graph['VertexC']

    def check(S: nx.Graph) -> None:
        assert_topology(S, expected_topology(case), case.capacity)
        assert terminal_terminal_crossings(S, VertexC) == []

    return run_with_retry(
        lambda time_limit: solve(
            A,
            capacity=case.capacity,
            time_limit=time_limit,
            ringed=case.ringed,
            seed=case.seed,
        ),
        check,
        time_limit=case.time_limit,
        label=case_node_id(case),
    )


@cache
def hgs_topology(case: BaselineCase) -> nx.Graph:
    """Produce a typed HGS case once for topology and PathFinder consumers."""
    return _baseline_topology(case, partial(hgs_cvrp, balanced=case.balanced))


@cache
def lkh_topology(case: BaselineCase) -> nx.Graph:
    """Produce a typed LKH-3 case once (requires the ``LKH`` executable)."""
    return _baseline_topology(case, lkh3)
