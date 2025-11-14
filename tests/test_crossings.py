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

    fnT = G.graph['fnT']

    crossings_0 = get_interferences_list(Edge=Edge, VertexC=VertexC, fnT=fnT)
    assert crossings_0 == []

    Edge[1] = (-1, 11)
    crossings_1 = get_interferences_list(Edge=Edge, VertexC=VertexC, fnT=fnT)
    expected = [((np.int64(0), np.int64(12), np.int64(-1), np.int64(11)), None)]
    assert crossings_1 == expected

    # with detours
    wfn2 = tiny_wfn(cables=1)
    G2 = wfn2.G
    VertexC2 = np.array(G2.graph["VertexC"])
    Edge2 = np.array(list(G2.edges()))
    fnT2 = G2.graph['fnT']

    crossings_2 = get_interferences_list(Edge=Edge2, VertexC=VertexC2, fnT=fnT2)
    assert crossings_2 == []

    crossings_3 = get_interferences_list(Edge=Edge2, VertexC=VertexC2, fnT=fnT)
    expected = [((np.int64(-1), np.int64(13), np.int64(1), np.int64(12)), None)]
    assert crossings_3 == expected


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

    # with detours
    wfn2 = tiny_wfn(cables=1)
    G2 = wfn2.G
    with pytest.raises(KeyError):
        validate_routeset(G2)
