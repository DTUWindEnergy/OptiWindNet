# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Canonical fingerprints for persisted OptiWindNet data."""

import io
from hashlib import sha256
from types import FunctionType

import numpy as np

__all__ = ('fingerprint_coordinates', 'fingerprint_function')


def fingerprint_coordinates(VertexC: np.ndarray) -> tuple[bytes, bytes]:
    """Return the SHA-256 digest and canonical ``.npy`` bytes of ``VertexC``.

    Arrays are normalized to C order before serialization so numerically equal
    C- and Fortran-contiguous inputs have the same digest. Version 3 of NumPy's
    ``.npy`` format is used to match persisted :class:`~optiwindnet.db.NodeSet`
    entries.
    """
    VertexC_npy_io = io.BytesIO()
    np.lib.format.write_array(
        VertexC_npy_io,
        np.ascontiguousarray(VertexC),
        version=(3, 0),
    )
    VertexC_npy = VertexC_npy_io.getvalue()
    return sha256(VertexC_npy).digest(), VertexC_npy


def fingerprint_function(function: FunctionType) -> dict[str, bytes | str]:
    """Return the bytecode digest, source filename, and name of ``function``.

    The digest intentionally covers only ``code.co_code`` to preserve existing
    database identities. It is not a complete semantic fingerprint: changes to
    constants, defaults, globals, or closure values may leave the digest
    unchanged.
    """
    code = function.__code__
    return dict(
        funhash=sha256(code.co_code).digest(),
        funfile=code.co_filename,
        funname=code.co_name,
    )
