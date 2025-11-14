import numpy as np
from .helpers import tiny_wfn
from optiwindnet.geometric import minimum_spanning_forest, rotate

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


def test_rotate():
    wfn = tiny_wfn()
    G = wfn.G

    vertexC = G.graph['VertexC']
    rotated_vertexC = rotate(coords=vertexC, angle=5)
    expected = np.array(
      [[ 0.9961947 ,  0.08715574],
       [ 1.9923894 ,  0.17431149],
       [ 1.90523365,  1.17050618],
       [ 1.73092217,  3.16289558],
       [-1.81807791, -2.16670088],
       [ 2.16670088, -1.81807791],
       [ 1.64376643,  4.15909028],
       [-2.34101237,  3.81046731],
       [ 1.23901151, -0.39351046],
       [ 1.10827789,  1.10078159],
       [ 1.74957259,  0.65497769],
       [ 1.53786992, -0.36736373],
       [ 0.        ,  0.        ]])
    
    np.array_equal(rotated_vertexC, expected)  