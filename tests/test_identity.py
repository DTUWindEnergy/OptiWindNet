from hashlib import sha256

import numpy as np
from bitarray import frozenbitarray

from optiwindnet.identity import (
    fingerprint_coordinates,
    fingerprint_function,
    topology_id,
)


def test_fingerprint_coordinates_canonicalizes_memory_order():
    VertexC = np.array([(0.0, 1.0), (2.0, 3.0)], order='C')
    VertexC_fortran = np.array(VertexC, order='F')

    digest, packed = fingerprint_coordinates(VertexC)
    digest_fortran, packed_fortran = fingerprint_coordinates(VertexC_fortran)

    assert digest_fortran == digest
    assert packed_fortran == packed
    assert b"'fortran_order': False" in packed[:256]


def test_fingerprint_function_reports_bytecode_and_identity():
    def sample_function(x=1):
        return x + 1

    fingerprint = fingerprint_function(sample_function)

    assert fingerprint == {
        'funhash': sha256(sample_function.__code__.co_code).digest(),
        'funfile': sample_function.__code__.co_filename,
        'funname': sample_function.__code__.co_name,
    }


def test_topology_id_separates_vectors_that_share_padded_bytes():
    """Padding to the byte boundary must not make shorter linkbits collide."""
    six = frozenbitarray('010010')
    eight = frozenbitarray('01001000')

    assert topology_id(six) == topology_id(frozenbitarray('010010'))
    assert six.tobytes() == eight.tobytes()
    assert topology_id(six) != topology_id(eight)
    assert len(topology_id(six)) == 16
