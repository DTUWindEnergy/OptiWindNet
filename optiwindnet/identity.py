# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Identities of OptiWindNet data: coordinates, code, link sets, topologies.

Two vocabularies live here, and the distinction is deliberate. The
``fingerprint_*()`` functions produce SHA-256 identities of records that are
*persisted*: a coordinate fingerprint is the primary key of a
:class:`~optiwindnet.db.NodeSet` row, so its bytes are a stored-data contract
that must not drift. The ``*_id()`` functions produce non-cryptographic xxh3
ids for objects held *in memory* -- an available-links set, a topology --
which nothing reads back from disk and which are free to change with the code
that makes them.
"""

import io
from hashlib import sha256
from types import FunctionType

import networkx as nx
import numpy as np
import xxhash
from bitarray import frozenbitarray

__all__ = (
    'complete_linkset_id', 'fingerprint_coordinates', 'fingerprint_function',
    'linkset_id', 'topology_id',
)  # fmt: skip


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
_LINKSET_ID = '_linkset_id'


def _invalidate_canonical_linkset(A: nx.Graph) -> None:
    """Drop the cached link order and id before mutating ``A``'s edges."""
    A.graph.pop(_CANONICAL_TERMINAL_LINKS, None)
    A.graph.pop(_LINKSET_ID, None)


def linkset_id(A: nx.Graph) -> bytes:
    """Return the 128-bit xxh3 id of ``A``'s canonical available-links set.

    The id covers ``R``, ``T`` and the canonical terminal-link array, which is
    everything that fixes the meaning of a link-bit position and the total bit
    count. Geometry is deliberately left out: two sites whose meshes have the
    same combinatorics share an id, and coordinates are identified by
    :func:`fingerprint_coordinates` instead.
    """
    R, T = (A.graph[key] for key in 'RT')
    links = A.graph[_CANONICAL_TERMINAL_LINKS]
    return xxhash.xxh3_128_digest(
        R.to_bytes(4, 'little')
        + T.to_bytes(4, 'little')
        + len(links).to_bytes(4, 'little')
        + np.ascontiguousarray(links, dtype='<u4').tobytes()
    )


def complete_linkset_id(R: int, T: int) -> bytes:
    """Return the :func:`linkset_id` of the complete terminal graph over ``T``.

    The digest is the one ``linkset_id()`` would return for a graph whose
    canonical link array is ``np.triu_indices(T, k=1)``, but the array is hashed
    a block of rows at a time and never held whole: it counts ``T*(T - 1)/2``
    links, which is 95 MiB at ``T = 5000`` -- and five times that at the peak of
    building it, against 8 MiB for this.
    """
    count = T * (T - 1) // 2
    hasher = xxhash.xxh3_128()
    hasher.update(
        R.to_bytes(4, 'little') + T.to_bytes(4, 'little') + count.to_bytes(4, 'little')
    )
    # rows per block, so that a block's int64 intermediates stay a few MiB
    step = max(1, (1 << 18) // max(1, T))
    for start in range(0, max(T - 1, 0), step):
        stop = min(start + step, T - 1)
        us = np.arange(start, stop, dtype=np.uint32)
        counts = (T - 1 - us).astype(np.int64)
        lo = np.repeat(us, counts)
        # within each row, hi runs from u + 1 up to T - 1
        offsets = np.repeat(np.cumsum(counts) - counts, counts)
        hi = (np.arange(counts.sum(), dtype=np.int64) - offsets + lo + 1).astype(
            np.uint32
        )
        hasher.update(
            np.ascontiguousarray(np.stack((lo, hi), axis=1), dtype='<u4').tobytes()
        )
    return hasher.digest()


def topology_id(linkbits: frozenbitarray) -> bytes:
    """Return the 128-bit xxh3 id of a topology's canonical ``linkbits``.

    The id covers the length of the available-links set together with the
    packed bits, so vectors that differ only in the zero padding of the last
    byte cannot collide. It identifies the set of active links alone: topologies
    with equal ids may still differ in load assignment or in routing.
    """
    return xxhash.xxh3_128_digest(
        len(linkbits).to_bytes(4, 'little') + linkbits.tobytes()
    )
