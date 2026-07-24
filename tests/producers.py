"""Process-local caches for expensive, read-only producer topology fixtures."""

from functools import cache

from optiwindnet.baselines.hgs import hgs_cvrp
from optiwindnet.interarraylib import as_normalized

from .cases import BaselineCase
from .sitecache import get_bundle


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
