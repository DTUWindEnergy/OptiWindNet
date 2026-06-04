# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Deprecation helper for the legacy Esau-Williams heuristic functions.

`ClassicEW`, `CPEW`, `NBEW`, `OBEW` and `EW_presolver` were superseded by the
unified `heuristics.constructor()`. They remain importable but emit a
`DeprecationWarning` when called. Scheduled for removal in v0.3.
"""

import functools
import warnings
from typing import Callable

__all__ = ()

_REMOVAL_VERSION = 'v0.3'


def deprecated_heuristic(*, migrate_to: str) -> Callable:
    """Mark a legacy heuristic as deprecated in favor of `constructor()`.

    The wrapped function keeps working unchanged; it only warns on call.

    Args:
      migrate_to: the equivalent `constructor()` call to suggest to the user.
    """

    def decorator(fn: Callable) -> Callable:
        message = (
            f'`{fn.__name__}()` is deprecated and will be removed in {_REMOVAL_VERSION}'
            '; use `heuristics.constructor()` instead. A similar solution can be '
            f'obtained using the following call: {migrate_to}. Note that `constructor`'
            'takes the available-links graph A (from `make_planar_embedding(L)`), not '
            'the location graph L; alternatively use the high-level `WindFarmNetwork`/'
            '`EWRouter` API, which builds the mesh for you.'
        )

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return fn(*args, **kwargs)

        wrapper.__doc__ = f'DEPRECATED ({_REMOVAL_VERSION}): {migrate_to}\n\n' + (
            fn.__doc__ or ''
        )
        return wrapper

    return decorator
