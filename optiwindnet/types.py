# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Types shared across OptiWindNet's modules."""

from collections.abc import Set as AbstractSet
from enum import StrEnum, auto
from typing import NamedTuple

__all__ = ('FeederRoute', 'NodeHighlight', 'OWNWarmupFailed', 'Topology')


class NodeHighlight(NamedTuple):
    """Nodes to emphasize in a plot and their legend label."""

    label: str
    nodes: AbstractSet[int]


class Topology(StrEnum):
    """Architecture of the subtrees in a solution.

    ``NetworkConstraints`` and ``TerseLinks`` accept the equivalent ``str`` and coerce
    it. Past them the member itself is what travels: every producer stores it in
    ``S.graph['topology']`` and every consumer branches on ``is``, which an equal
    ``str`` would silently fail.
    """

    RADIAL = auto()
    BRANCHED = auto()
    RINGED = auto()
    DEFAULT = BRANCHED


class FeederRoute(StrEnum):
    "If feeder routes must be ``'straight'`` or can be detoured (``'segmented'``)."

    STRAIGHT = auto()
    SEGMENTED = auto()
    DEFAULT = SEGMENTED


class OWNWarmupFailed(Exception):
    """A warm start is unusable for the model or solver it was given to."""
