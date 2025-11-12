import numpy as np
from .helpers import tiny_wfn
from optiwindnet.geometric import minimum_spanning_forest

def test_minimum_spanning_forest():
    wfn = tiny_wfn()
    S = minimum_spanning_forest(wfn.A)
    Edges = np.array(list(S.edges()))
    expected = np.array([(0, 1), (0, -1), (1, 2), (2, 3)])
    assert np.array_equal(Edges, expected)

    # with capacity = 1, there will be detours in G
    wfn2 = tiny_wfn(cables=1)
    S2 = minimum_spanning_forest(wfn2.A)
    Edges2 = np.array(list(S2.edges()))
    expected2 = np.array([(0, 1), (0, -1), (1, 2), (2, 3)])
    assert np.array_equal(Edges2, expected2)