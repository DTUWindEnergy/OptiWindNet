# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Rendering of routeset properties as text for user-facing output."""

import math

import networkx as nx
import numpy as np

__all__ = ('describe_G',)


def _format_length(length: float, significant_digits: int = 5) -> str:
    """Format ``length`` with '_' as thousands separator.

    ``significant_digits`` is a minimum, enforced through fraction digits only.
    """
    intdigits = int(np.floor(np.log10(length))) + 1
    fracdigits = max(0, significant_digits - intdigits)
    return f'{{:_.{fracdigits}f}}'.format(round(length, fracdigits))


def describe_G(G: nx.Graph, significant_digits: int = 5) -> list[str]:
    """Create a 3-4 line summary of G's properties.

    ``significant_digits`` applies only to total length and is enforced only when the
    integer part has fewer significant digits than ``significant_digits``.

    Args:
      G: routeset instance
      significant_digits: minimum number of significant digits used for total length

    Returns:
      Text lines with capacity and T, excess feeders and feeders per root, total
      length and total cost.
    """
    R = G.graph['R']
    T = G.graph['T']
    capacity = G.graph['capacity']
    roots = range(1, R + 1)
    RootL = {-r: G.nodes[-r].get('label', f'[{-r}]') for r in roots}
    desc = []
    desc.append(f'κ = {capacity}, T = {T}')
    feeder_info = [f'{rootL}: {G.degree[r]}' for r, rootL in RootL.items()]
    excess_feeders = sum(G.degree[-r] for r in roots) - math.ceil(T / capacity)
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
