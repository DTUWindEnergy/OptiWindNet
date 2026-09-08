from hashlib import sha256

import networkx as nx
import numpy as np
import pytest
from bitarray import frozenbitarray

from optiwindnet.identity import (
    fingerprint_coordinates,
    fingerprint_function,
    linkset_id,
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


def _A(R, T, links):
    """Hand-build the graph attributes ``linkset_id()`` reads."""
    A = nx.Graph(R=R, T=T)
    A.add_edges_from(links)
    A.graph['_canonical_terminal_links'] = np.array(
        sorted(links), dtype=np.uint32
    ).reshape(-1, 2)
    return A


def test_linkset_id_is_a_128_bit_value():
    assert len(linkset_id(_A(1, 3, [(0, 1), (0, 2), (1, 2)]))) == 16


def test_linkset_id_reads_only_the_canonical_linkset():
    """Insertion order and unrelated graph content leave the digest alone."""
    links = [(0, 1), (0, 2), (1, 2)]
    forward = _A(1, 3, links)
    reverse = _A(1, 3, links[::-1])
    reverse.graph['handle'] = 'unrelated'
    reverse.add_edge(-1, 0)

    assert linkset_id(forward) == linkset_id(reverse)


@pytest.mark.parametrize(
    ('R', 'T', 'links'),
    (
        pytest.param(2, 3, [(0, 1), (0, 2), (1, 2)], id='roots'),
        pytest.param(1, 4, [(0, 1), (0, 2), (1, 2)], id='terminals'),
        pytest.param(1, 3, [(0, 1), (1, 2)], id='links'),
    ),
)
def test_linkset_id_separates_distinct_linksets(R, T, links):
    """R and T size the feeder block, so neither may be left out of the hash."""
    base = _A(1, 3, [(0, 1), (0, 2), (1, 2)])

    assert linkset_id(_A(R, T, links)) != linkset_id(base)


def test_linkset_id_ignores_array_dtype_and_memory_order():
    links = [(0, 1), (0, 2), (1, 2)]
    canonical = _A(1, 3, links)
    widened = _A(1, 3, links)
    widened.graph['_canonical_terminal_links'] = np.asfortranarray(
        np.array(sorted(links), dtype=np.int64)
    )

    assert linkset_id(widened) == linkset_id(canonical)
