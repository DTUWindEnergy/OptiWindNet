# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Identities of OptiWindNet data: coordinates, code, link sets, topologies.

Two vocabularies live here, and the distinction is deliberate. The
``fingerprint_*()`` functions produce SHA-256 identities of records that are
*persisted*: a coordinate fingerprint is the primary key of a
:class:`~optiwindnet.db.NodeSet` row, so its bytes are a stored-data contract
that must not drift. The ``*_id()`` functions produce non-cryptographic xxh3
ids for objects held *in memory* -- a link universe, a topology -- which nothing
reads back from disk and which are free to change with the code that makes
them.
"""

import io
from hashlib import sha256
from types import FunctionType

import networkx as nx
import numpy as np
import xxhash
from bitarray import frozenbitarray

__all__ = ('fingerprint_coordinates', 'fingerprint_function', 'topology_id')


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
    return {
        'funhash': sha256(code.co_code).digest(),
        'funfile': code.co_filename,
        'funname': code.co_name,
    }


_CANONICAL_TERMINAL_LINKS = '_canonical_terminal_links'


def _invalidate_canonical_terminal_links(A: nx.Graph) -> None:
    """Drop the position cache before mutating ``A``'s candidate edge set."""
    A.graph.pop(_CANONICAL_TERMINAL_LINKS, None)


def topology_id(linkbits: frozenbitarray) -> bytes:
    """Return the 128-bit xxh3 id of a topology's canonical ``linkbits``.

    The id covers the length of the link universe together with the packed
    bits, so vectors that differ only in the zero padding of the last byte
    cannot collide. It identifies the set of active links alone: topologies with
    equal ids may still differ in load assignment or in routing.
    """
    return xxhash.xxh3_128_digest(
        len(linkbits).to_bytes(4, 'little') + linkbits.tobytes()
    )
