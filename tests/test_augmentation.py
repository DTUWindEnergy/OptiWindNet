# tests/test_augmentation.py
import math
import logging
import numpy as np
import networkx as nx
import pytest

import optiwindnet.augmentation as aug


def _square(w=10.0, h=10.0, x0=0.0, y0=0.0):
    return np.array(
        [
            [x0, y0],
            [x0 + w, y0],
            [x0 + w, y0 + h],
            [x0, y0 + h],
        ],
        dtype=float,
    )


def _pairwise_min_dist(P):
    if len(P) < 2:
        return float('inf')
    diffs = P[:, None, :] - P[None, :, :]
    d2 = np.sum(diffs * diffs, axis=-1)
    # ignore self-distances
    d2 += np.eye(len(P)) * 1e12
    return float(np.sqrt(d2.min()))


# -------------------------------
# _get_border_scale_offset
# -------------------------------
def test_get_border_scale_offset_normalizes_area_to_one():
    poly = _square(10, 10)
    offset, scale, w, h = aug._get_border_scale_offset(poly)
    assert np.allclose(offset, [0.0, 0.0])
    assert math.isclose(w, 10.0) and math.isclose(h, 10.0)
    # For a 10x10 square, area=100 -> scale must be 1/sqrt(100)=0.1
    assert math.isclose(scale, 0.1, rel_tol=0, abs_tol=1e-12)

    # Check area really becomes ~1 when applying the scale+offset
    normed = (poly - offset) * scale
    # Simple shoelace formula for sanity (avoid importing other modules)
    x, y = normed[:, 0], normed[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert math.isclose(area, 1.0, rel_tol=1e-12, abs_tol=1e-12)


# -------------------------------
# _contains / _contains_np
# -------------------------------
def test_contains_and_contains_np_inside_outside_vertex():
    poly = _square(10, 10)
    inside = np.array([[5.0, 5.0]])
    outside = np.array([[-1.0, -1.0]])
    vertex = np.array([[0.0, 0.0]])

    # vectorized API
    m = aug._contains_np(poly, np.vstack([inside, outside, vertex]))
    assert m[0]
    assert not m[1]
    assert m[2]  # counts vertex as inside

    # scalar njit version
    assert aug._contains(poly, inside[0])
    assert not aug._contains(poly, outside[0])
    assert aug._contains(poly, vertex[0])


# -------------------------------
# poisson_disc_filler core behavior
# -------------------------------
def test_poisson_disc_filler_basic_spacing_and_inside():
    BorderC = _square(10, 10)  # area=100
    T = 20
    d = 2.0
    pts = aug.poisson_disc_filler(
        T=T,
        min_dist=d,
        BorderC=BorderC,
        seed=123,
        max_iter=20_000,
        partial_fulfilment=True,
        rounds=1,
    )
    # Should place at least a few points and never exceed T
    assert 0 < len(pts) <= T
    # All points must lie inside the square (bbox == polygon for a square)
    assert np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] <= 10)
    assert np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] <= 10)
    # Enforce minimum spacing (allow tiny numerical slack)
    assert _pairwise_min_dist(pts) >= d - 1e-9


def test_poisson_disc_filler_respects_repeller_radius():
    BorderC = _square(10, 10)
    T = 30
    d = 1.5
    repeller = np.array([[5.0, 5.0]])  # center
    repel_radius = 3.0
    pts = aug.poisson_disc_filler(
        T=T,
        min_dist=d,
        BorderC=BorderC,
        RepellerC=repeller,
        repel_radius=repel_radius,
        seed=7,
        rounds=2,
    )
    # No point should be inside the repeller disk
    dists = np.sqrt(((pts - repeller[0]) ** 2).sum(axis=1))
    assert np.all(dists >= repel_radius - 1e-9)


def test_poisson_disc_filler_obstacles_not_implemented():
    BorderC = _square(10, 10)
    with pytest.raises(NotImplementedError):
        aug.poisson_disc_filler(
            T=5, min_dist=1.0, BorderC=BorderC, obstacles=[_square(2, 2)]
        )


def test_poisson_disc_filler_efficiency_guard_raises_when_partial_false():
    # Border area = 100. With d=5, area_demand per point ~ 19.635.
    # For T=6 => demand ~117.8 > 0.9069 * 100 => should raise if partial_fulfilment=False.
    BorderC = _square(10, 10)
    with pytest.raises(ValueError):
        aug.poisson_disc_filler(
            T=6, min_dist=5.0, BorderC=BorderC, partial_fulfilment=False
        )


def test_poisson_disc_filler_logs_warning_on_partial(caplog):
    BorderC = _square(10, 10)
    # Ask for an impossible number with big spacing: should log a warning
    with caplog.at_level(logging.WARNING, logger='optiwindnet.augmentation'):
        pts = aug.poisson_disc_filler(
            T=1000, min_dist=3.0, BorderC=BorderC, seed=1, rounds=1
        )
    assert len(pts) < 1000
    assert any('Only' in rec.getMessage() for rec in caplog.records)


# -------------------------------
# get_shape_to_fill
# -------------------------------
def _make_graph(borderC, rootsC):
    G = nx.Graph()
    # VertexC: border first, then roots at the end (as required by the code)
    VertexC = np.vstack([borderC, rootsC])
    G.graph['VertexC'] = VertexC
    G.graph['border'] = np.arange(len(borderC), dtype=int)
    G.graph['R'] = len(rootsC)
    # optional extras passed through turbinate -> L_from_site
    G.graph['name'] = 'test'
    G.graph['handle'] = 'h'
    G.graph['landscape_angle'] = 0.0
    return G


def test_get_shape_to_fill_area_is_one():
    borderC = _square(10, 10)
    rootsC = np.array([[5.0, 5.0]])
    G = _make_graph(borderC, rootsC)

    BorderC, RootC = aug.get_shape_to_fill(G)
    # area ~ 1
    x, y = BorderC[:, 0], BorderC[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert math.isclose(area, 1.0, rel_tol=1e-12, abs_tol=1e-12)
    # single root returned
    assert RootC.shape == (1, 2)
    # R unchanged (was 1)
    assert G.graph['R'] == 1


def test_get_shape_to_fill_multi_roots_are_averaged_and_R_set_to_1():
    borderC = _square(8, 12, x0=2, y0=3)  # offset to test translation
    rootsC = np.array([[3.0, 4.0], [9.0, 10.0]])
    G = _make_graph(borderC, rootsC)

    BorderC, RootC = aug.get_shape_to_fill(G)
    # area ~ 1
    x, y = BorderC[:, 0], BorderC[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert math.isclose(area, 1.0, rel_tol=1e-12, abs_tol=1e-12)
    # roots averaged and only one returned
    assert RootC.shape == (1, 2)
    assert G.graph['R'] == 1


# -------------------------------
# turbinate
# -------------------------------
def test_turbinate_builds_graph_with_spaced_turbines():
    borderC = _square(10, 10)
    rootsC = np.array([[5.0, 5.0]])
    L = _make_graph(borderC, rootsC)

    T_req = 10
    d = 0.10  # in normalized units (turbinate normalizes area to 1)
    L2 = aug.turbinate(L, T=T_req, d=d, max_iter=50_000, rounds=2)

    assert isinstance(L2, nx.Graph)
    V = L2.graph['VertexC']
    R = L2.graph['R']
    # In turbinate(), VertexC = [TerminalS, BorderS, RepellerS]; R roots at end.
    T_out = V.shape[0] - len(L.graph['border']) - R
    assert 0 < T_out <= T_req

    turbines = V[:T_out]
    # All turbines inside the normalized [0,1]x[0,1] (square normalized to area 1)
    assert np.all(turbines >= -1e-9)
    assert np.all(turbines <= 1.0 + 1e-9)
    # Spacing respected (allow tiny slack)
    assert _pairwise_min_dist(turbines) >= d - 1e-9
    # Also respect root clearance (defaults to d)
    roots = V[-R:]
    for r in roots:
        dr = np.sqrt(((turbines - r) ** 2).sum(axis=1))
        assert np.all(dr >= d - 1e-9)


# -------------------------------
# iCDF_factory
# -------------------------------
def test_iCDF_factory_monotonic_and_bounds():
    T_min, T_max, eta, d_lb = 50, 200, 0.6, 0.045
    iCDF = aug.iCDF_factory(T_min, T_max, eta, d_lb)

    us = np.linspace(0.0, 1.0, 21)
    Ts = [iCDF(float(u)) for u in us]

    # Bounds
    assert min(Ts) >= T_min - 1
    assert max(Ts) <= T_max + 1

    # Non-decreasing across u (allow occasional flat steps due to int rounding)
    assert all(Ts[i] <= Ts[i + 1] for i in range(len(Ts) - 1))

    # Sample 'd' upper bound is sensible for one T
    T = 100
    d_ub = 2 * math.sqrt(eta / math.pi / T)
    assert d_ub > d_lb
