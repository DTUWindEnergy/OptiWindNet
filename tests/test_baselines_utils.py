import numpy as np

from optiwindnet.baselines.utils import length_matrix_single_depot_from_G

from .helpers import tiny_wfn


def test_length_matrix_shape_and_depot_row():
    """Basic smoke: matrix is (T+R)×(T+R), depot column is all zeros."""
    wfn = tiny_wfn()
    A = wfn.A
    T, R = A.graph['T'], A.graph['R']
    L, len_max = length_matrix_single_depot_from_G(A, scale=1.0)
    assert L.shape == (T + R, T + R)
    # depot column (index 0) must be all-zero (open-VRP)
    assert np.all(L[:, 0] == 0.0)
    assert len_max > 0.0


def test_length_matrix_scale():
    """Scaling factor multiplies every finite length."""
    wfn = tiny_wfn()
    A = wfn.A
    L1, _ = length_matrix_single_depot_from_G(A, scale=1.0)
    L2, _ = length_matrix_single_depot_from_G(A, scale=2.0)
    finite = np.isfinite(L1)
    np.testing.assert_allclose(L2[finite], L1[finite] * 2.0)


def test_length_matrix_complete_no_inf():
    """complete=True builds a full dense matrix with no inf entries."""
    wfn = tiny_wfn()
    A = wfn.A
    L, len_max = length_matrix_single_depot_from_G(A, scale=1.0, complete=True)
    T, R = A.graph['T'], A.graph['R']
    assert L.shape == (T + R, T + R)
    # off-diagonal entries should all be finite (complete graph)
    off_diag = L[np.eye(T + R, dtype=bool) == False]
    assert np.all(np.isfinite(off_diag))


def test_length_matrix_incomplete_has_inf():
    """complete=False leaves unreachable pairs as inf."""
    wfn = tiny_wfn()
    A = wfn.A
    L, _ = length_matrix_single_depot_from_G(A, scale=1.0, complete=False)
    # some pairs should still be inf (not all nodes are directly connected)
    assert np.any(np.isinf(L))
