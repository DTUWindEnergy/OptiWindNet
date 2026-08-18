import logging

import numpy as np
import pytest
from shapely.geometry import Polygon

from optiwindnet import api_utils

from .helpers import tiny_wfn


def test_expand_polygon_safely_warns_for_nonconvex_large_buffer(caplog):
    poly = Polygon([(0, 0), (4, 0), (4, 1), (1, 1), (1, 3), (4, 3), (4, 4), (0, 4)])
    with caplog.at_level(logging.WARNING, logger=api_utils.__name__):
        out = api_utils.expand_polygon_safely(poly, buffer_dist=1.0)
    assert out.area > poly.area
    assert any(
        'non-convex and buffering may introduce unexpected changes' in message
        for message in caplog.messages
    )


def test_expand_polygon_safely_convex_no_warning(caplog):
    poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    with caplog.at_level(logging.WARNING, logger=api_utils.__name__):
        out = api_utils.expand_polygon_safely(poly, buffer_dist=0.25)
    assert out.area > poly.area
    assert not any(
        'non-convex and buffering may introduce' in message
        for message in caplog.messages
    )


def test_shrink_polygon_safely_returns_array_normal():
    poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    arr = api_utils.shrink_polygon_safely(poly, shrink_dist=0.2, indx=0)
    assert isinstance(arr, np.ndarray) and arr.shape[1] == 2


def test_shrink_polygon_safely_becomes_empty_warns(caplog):
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    with caplog.at_level(logging.WARNING, logger=api_utils.__name__):
        result = api_utils.shrink_polygon_safely(poly, shrink_dist=10.0, indx=3)
    assert result is None
    assert any(
        'completely removed the obstacle' in message for message in caplog.messages
    )


def test_shrink_polygon_safely_splits_to_multipolygon(caplog):
    big = Polygon([(0, 0), (8, 0), (8, 4), (0, 4)])
    hole = Polygon([(3, -1), (5, -1), (5, 5), (3, 5)])
    shape = big.difference(hole)
    with caplog.at_level(logging.WARNING, logger=api_utils.__name__):
        result = api_utils.shrink_polygon_safely(shape, shrink_dist=0.1, indx=1)
    assert isinstance(result, list) and len(result) >= 2
    assert any('split the obstacle' in message for message in caplog.messages)


def test_enable_ortools_logging_if_jupyter_sets_callback(monkeypatch):
    ZMQInteractiveShell = type('ZMQInteractiveShell', (), {})
    monkeypatch.setattr(
        api_utils,
        'get_ipython',
        lambda: ZMQInteractiveShell(),
        raising=False,
    )

    class DummySolver:
        def __init__(self):
            self.log_callback = None

    solver = DummySolver()
    api_utils.enable_ortools_logging_if_jupyter(solver)
    assert solver.log_callback is print


def test_parse_cables_input_numpy_ints_and_pairs():
    capacities = api_utils.parse_cables_input(np.array([5, 7]))
    assert capacities == [(5, 0.0), (7, 0.0)]
    capacity_cost_pairs = np.array([(3, 10.0), (6, 20.0)], dtype=object)
    assert api_utils.parse_cables_input(capacity_cost_pairs) == [
        (3, 10.0),
        (6, 20.0),
    ]


def test_buffer_border_obs_negative_raises():
    wfn = tiny_wfn()
    with pytest.raises(ValueError, match='must be equal or greater than 0'):
        api_utils.buffer_border_obs(wfn.L, buffer_dist=-1.0)


def test_shrink_polygon_safely_unexpected_geometry_returns_none(caplog):
    from shapely.geometry import Point

    class FakeGeometry:
        def buffer(self, dist):
            return Point(0, 0)

    with caplog.at_level(logging.WARNING, logger=api_utils.__name__):
        res = api_utils.shrink_polygon_safely(FakeGeometry(), shrink_dist=1.0, indx=5)
    assert res is None
    assert any('Unexpected geometry type' in message for message in caplog.messages)


def test_buffer_border_obs_empty_obstacle_entry():
    wfn = tiny_wfn()
    L = wfn.L.copy()
    L.graph['obstacles'] = [np.array([], dtype=int)]
    L_buffered, pre_buf = api_utils.buffer_border_obs(L, buffer_dist=1.0)
    assert L_buffered is not None
    assert isinstance(pre_buf, dict)


def test_plot_org_buff():
    import matplotlib.pyplot as plt

    borderC = np.array([(0, 0), (10, 0), (10, 10), (0, 10)])
    border_bufferedC = np.array([(1, 1), (9, 1), (9, 9), (1, 9)])
    ax = api_utils.plot_org_buff(
        borderC=borderC,
        border_bufferedC=border_bufferedC,
        obstaclesC=[],
        obstacles_bufferedC=[],
    )
    assert ax is not None
    plt.close('all')
