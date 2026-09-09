# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Process-local caches for expensive, read-only producer topology fixtures."""

from functools import cache

from optiwindnet.baselines.hgs import hgs_cvrp
from optiwindnet.heuristics import constructor
from optiwindnet.transforming import as_normalized

from .cases import BaselineCase, ConstructorCase
from .sitecache import get_bundle


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


@cache
def hgs_topology(case: BaselineCase):
    """Produce a typed HGS case once for topology and PathFinder consumers."""
    A = get_bundle(case.site).A
    return hgs_cvrp(
        as_normalized(A),
        capacity=case.capacity,
        time_limit=case.time_limit,
        balanced=case.balanced,
        ringed=case.ringed,
        seed=case.seed,
    )
