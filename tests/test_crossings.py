import numpy as np
import pytest
from .helpers import tiny_wfn
from optiwindnet.crossings import get_interferences_list, validate_routeset


def test_get_interferences_list():
    wfn = tiny_wfn()
    G = wfn.G
    VertexC = np.array(G.graph["VertexC"])
    Edge = np.array(list(G.edges()))


    with pytest.raises(IndexError):
        get_interferences_list(Edge=Edge, VertexC=VertexC)


    fnT = wfn.G.graph['fnT']

    crossings_0 = get_interferences_list(Edge=Edge, VertexC=VertexC, fnT=fnT)
    assert crossings_0 == []


    Edge[1] = (-1, 11)
    crossings_1 = get_interferences_list(Edge=Edge, VertexC=VertexC, fnT=fnT)
    expected = [((np.int64(0), np.int64(12), np.int64(-1), np.int64(11)), None)]
    assert crossings_1 == expected

def test_validate_routeset():
    wfn = tiny_wfn()
    G = wfn.G

    validate_0 = validate_routeset(G)
    assert validate_0 == []

    G.add_edge(-1, 11)
    validate_1 = validate_routeset(G)
    expected = [(np.int64(0), np.int64(12), np.int64(-1), np.int64(11))]
    assert validate_1 == expected

    G.remove_edge(0, 12)
    with pytest.raises(AssertionError):
        validate_routeset(G)
