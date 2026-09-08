# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Compatibility aliases for functions moved to :mod:`optiwindnet.identity`."""

import warnings

from .identity import fingerprint_coordinates, fingerprint_function

warnings.warn(
    'optiwindnet.fingerprint is deprecated and will be removed in v0.4.0; '
    'import these functions from optiwindnet.identity instead',
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ('fingerprint_coordinates', 'fingerprint_function')
