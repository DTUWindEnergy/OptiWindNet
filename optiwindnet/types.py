# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Types shared across OptiWindNet's routing and solver layers."""

from enum import StrEnum, auto

__all__ = ('Topology',)


class Topology(StrEnum):
    """Architecture of the subtrees in a solution.

    ``ModelOptions`` and ``TerseLinks`` accept the equivalent ``str`` and coerce
    it. Past them the member itself is what travels: every producer stores it in
    ``S.graph['topology']`` and every consumer branches on ``is``, which an equal
    ``str`` would silently fail.
    """

    RADIAL = auto()
    BRANCHED = auto()
    RINGED = auto()
    DEFAULT = BRANCHED
