# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Types shared across OptiWindNet's routing and solver layers."""

from enum import StrEnum, auto

__all__ = ('Topology',)


class Topology(StrEnum):
    """Architecture of the subtrees in a solution."""

    RADIAL = auto()
    BRANCHED = auto()
    RINGED = auto()
    DEFAULT = BRANCHED
