# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Rendering of routeset properties as text for user-facing output."""

import math
from collections.abc import Mapping
from fractions import Fraction

import networkx as nx
import numpy as np

from .loads import calcload, total_inflow

__all__ = ('describe_G',)

# Terminal symbol for each power level: (number of sides, circumradius scale).
# Sides 0 is a circle. The scale √(2π / (n·sin(2π/n))) gives the n-sided
# regular polygon the same area as the circle whose radius it multiplies.
_TERMINAL_SHAPES: tuple[tuple[int, float], ...] = (
    (0, 1.0),
    (8, 1.0539),
    (7, 1.0715),
    (6, 1.0996),
    (5, 1.1495),
)


def _terminal_groups(G: nx.Graph) -> list[tuple[int, float, str, list[int]]]:
    """Partition the terminals of ``G`` by declared power for plotting.

    Power levels are taken in the order of ``G.graph['powers_set']`` and are
    assigned the shapes in ``_TERMINAL_SHAPES`` cyclically. Power levels
    absent from ``G``'s terminals produce no group, but still consume a shape.

    Returns:
      List of ``(sides, scale, label, terminals)``, where ``sides`` is the
      number of sides of the symbol's regular polygon (0 for a circle) and
      ``scale`` multiplies the circle marker's radius to obtain the polygon's
      circumradius of equal area. Graphs without ``'powers_set'`` produce a
      single circle group labeled ``'WTG'``.
    """
    T = G.graph['T']
    powers_set = G.graph.get('powers_set')
    if powers_set is None:
        return [(0, 1.0, 'WTG', list(range(T)))]
    level_from_power = {power: i for i, power in enumerate(powers_set)}
    terminals_: list[list[int]] = [[] for _ in powers_set]
    for t in range(T):
        terminals_[level_from_power.get(G.nodes[t].get('power'), 0)].append(t)
    unit = G.graph.get('power_unit')
    groups = []
    for i, (power, terminals) in enumerate(zip(powers_set, terminals_)):
        if not terminals:
            continue
        sides, scale = _TERMINAL_SHAPES[i % len(_TERMINAL_SHAPES)]
        # The float cast is required only for Python 3.11's lack of Fraction :g.
        label = f'{float(power):g} {unit}' if unit else f'WTG {float(power):g}'
        groups.append((sides, scale, label, terminals))
    return groups


def _format_length(length: float, significant_digits: int = 5) -> str:
    """Format ``length`` with '_' as thousands separator.

    ``significant_digits`` is a minimum, enforced through fraction digits only.
    """
    intdigits = int(np.floor(np.log10(length))) + 1
    fracdigits = max(0, significant_digits - intdigits)
    return f'{{:_.{fracdigits}f}}'.format(round(length, fracdigits))


def _format_inflow_as_power(inflow: int, power_per_inflow: Fraction | int) -> str:
    """Format nominal power, keeping integer notation if ``power_per_inflow`` is 1."""
    # The float cast is required only for Python 3.11's lack of Fraction :g formatting.
    return (
        str(inflow)
        if power_per_inflow == 1
        else f'{float(inflow * power_per_inflow):g}'
    )


def _format_nominal_power(attrs: Mapping, power_per_inflow: Fraction | int) -> str:
    """Render a terminal's nominal power, preferring the value it declares."""
    if 'power' in attrs:
        # The float cast keeps Fraction formatting compatible with Python 3.11.
        return f'{float(attrs["power"]):g}'
    return _format_inflow_as_power(attrs.get('inflow', 1), power_per_inflow)


def _nominal_loads(G: nx.Graph) -> dict[int, Fraction]:
    """Map each node to the nominal power its outgoing cable carries.

    Exact quantization scales the integer loads, so the mapping is derived
    without touching ``G``. Inexact quantization requires accumulating the
    declared ratings along the network, which writes ``'load_nominal'``.
    """
    if G.graph.get('power_quantization_inexact', False):
        calcload(G, nominal=True)
        return {
            n: attrs['load_nominal']
            for n, attrs in G.nodes(data=True)
            if 'load_nominal' in attrs
        }
    power_per_inflow = Fraction(G.graph.get('power_per_inflow', 1))
    return {
        n: attrs['load'] * power_per_inflow
        for n, attrs in G.nodes(data=True)
        if 'load' in attrs
    }


def describe_G(G: nx.Graph, significant_digits: int = 5) -> list[str]:
    """Create a 3-5 line summary of G's properties.

    ``significant_digits`` applies only to total length and is enforced only when the
    integer part has fewer significant digits than ``significant_digits``.

    Args:
      G: routeset instance
      significant_digits: minimum number of significant digits used for total length

    Returns:
      Text lines with capacity and T, the distinct terminal inflow (only where
      some turbine differs from one inflow), excess feeders and feeders per root,
      total length and total cost.
    """
    R = G.graph['R']
    T = G.graph['T']
    capacity = G.graph['capacity']
    roots = range(1, R + 1)
    RootL = {-r: G.nodes[-r].get('label', f'[{-r}]') for r in roots}
    desc = []
    desc.append(f'κ = {capacity}, T = {T}')
    total = total_inflow(G)
    if total != T:
        # capacity counts inflow, so unequal turbines make it more than a count
        inflows = sorted({G.nodes[t].get('inflow', 1) for t in range(T)})
        desc.append('ι ∈ {' + ', '.join(str(inflow) for inflow in inflows) + '}')
    feeder_info = [f'{rootL}: {G.degree[r]}' for r, rootL in RootL.items()]
    excess_feeders = sum(G.degree[-r] for r in roots) - math.ceil(total / capacity)
    desc.append(f'({excess_feeders:+d}) {", ".join(feeder_info)}')
    length = G.size(weight='length')
    if length > 0:
        desc.append(
            'Σλ = '
            + _format_length(length, significant_digits).replace('_', '\u202f')
            + '\u00a0m'
        )
    if 'currency' in G.graph:
        desc.append(
            f'{G.size(weight="cost"):_.0f}\u00a0'.replace('_', '\u202f')
            + G.graph['currency']
        )
    return desc
