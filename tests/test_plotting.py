# tests/test_plotting.py
import os
os.environ.setdefault("MPLBACKEND", "Agg")  # headless matplotlib

from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

import optiwindnet.plotting as plotting


def _square(cx=0.0, cy=0.0, s=1.0):
    # counter-clockwise square around center (cx, cy) with side s
    h = s / 2.0
    return np.array([
        [cx - h, cy - h],
        [cx + h, cy - h],
        [cx + h, cy + h],
        [cx - h, cy + h],
    ], dtype=float)


def _make_graph(*, with_obstacle=False, with_supertriangle=False, landscape_angle=0.0):
    """
    Build a minimal graph G with the fields gplot() expects:
      - graph keys: 'R','T','B','VertexC','border', optionally 'obstacles',
                    'capacity','landscape_angle'
      - node ids: turbines: 0..T-1, border: T..T+B-1, extra (e.g. obstacle/ST): next,
                  roots as negative ids -R..-1 (positions taken from last R coords)
    """
    R, T, B = 1, 4, 4

    # geometry blocks (order matters for gplot()'s enumerate scheme)
    turbines = _square(0.0, 0.0, 1.0)                      # 4 turbines
    border    = _square(0.0, 0.0, 2.5)                     # 4 border verts
    chunks = [turbines, border]

    obstacle_idx = None
    if with_obstacle:
        # small triangle inside border
        obstacle = np.array([[0.0, 0.3], [0.3, 0.0], [0.3, 0.3]], dtype=float)
        obstacle_idx = np.arange(T + B, T + B + 3)
        chunks.append(obstacle)

    if with_supertriangle:
        # far away supertriangle points (added BEFORE the roots)
        st = np.array([[-1000, -1000], [2000, 0], [0, 2000]], dtype=float)
        chunks.append(st)

    roots = np.array([[0.0, 0.0]], dtype=float)            # 1 OSS at the end
    VertexC = np.vstack(chunks + [roots])

    G = nx.Graph()
    G.graph["R"], G.graph["T"], G.graph["B"] = R, T, B
    G.graph["VertexC"] = VertexC
    G.graph["border"]  = np.arange(T, T + B)               # indices of border vertices
    G.graph["capacity"] = 25.0
    G.graph["name"] = "test-layout"   # required by plotting.compare()
    if landscape_angle:
        G.graph["landscape_angle"] = float(landscape_angle)

    if with_obstacle:
        G.graph["obstacles"] = [obstacle_idx]

    # Add nodes so NetworkX drawing has them; include negative-root ids too
    # (gplot places roots by negative ids using last R coords)
    all_ids = list(range(VertexC.shape[0] - R)) + list(range(-R, 0))
    G.add_nodes_from(all_ids)

    # Give subtree colors to turbines (optional cosmetic)
    for t in range(T):
        G.nodes[t]["subtree"] = t

    return G


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_gplot_border_only_smoke(tmp_path):
    G = _make_graph(with_obstacle=False, landscape_angle=30.0)  # exercise rotation path

    with patch("matplotlib.pyplot.show"):
        ax = plotting.gplot(
            G,
            node_tag=True,         # label branch (numbers)
            landscape=True,
            infobox=True,
            scalebar=(1.0, "1 unit"),
            hide_ST=True,
            legend=True,
            tag_border=True,       # tag border indices
            min_dpi=120,
        )

    assert hasattr(ax, "figure")
    out = Path(tmp_path) / "gplot_border_only.png"
    ax.figure.savefig(out, dpi=120)
    assert out.exists() and out.stat().st_size > 0


def test_gplot_with_obstacle_polygon(tmp_path):
    G = _make_graph(with_obstacle=True)

    with patch("matplotlib.pyplot.show"):
        ax = plotting.gplot(
            G,
            node_tag="label",   # exercise alternate label branch (empty strings)
            landscape=False,
            infobox=True,
            scalebar=None,
            legend=False,
            hide_ST=False,
        )

    out = Path(tmp_path) / "gplot_with_obstacle.svg"
    ax.figure.savefig(out)
    assert out.exists() and out.stat().st_size > 0


def test_gplot_hides_supertriangle_when_present(tmp_path):
    G = _make_graph(with_supertriangle=True)

    with patch("matplotlib.pyplot.show"):
        ax = plotting.gplot(G, hide_ST=True, node_tag=None, infobox=False)

    # just ensure it renders and can be saved
    out = Path(tmp_path) / "gplot_hide_st.png"
    ax.figure.savefig(out, dpi=96)
    assert out.exists() and out.stat().st_size > 0


def test_pplot_with_planar_embedding(tmp_path):
    # Build source A with supertriangle (as pplot expects coordinates incl. ST)
    A = _make_graph(with_supertriangle=True)
    R, T, B = A.graph["R"], A.graph["T"], A.graph["B"]

    # PlanarEmbedding on turbine nodes only
    P = nx.PlanarEmbedding()
    P.add_nodes_from(range(T))

    # Helper to add an undirected edge by adding both half-edges.
    # If a node already has successors, we pass a cw reference to satisfy the API.
    prev = {}  # remember last neighbor per node to use as cw reference

    def add_undirected(u, v):
        # u -> v
        if u in prev:
            P.add_half_edge(u, v, cw=prev[u])
        else:
            P.add_half_edge(u, v)
        prev[u] = v

        # v -> u
        if v in prev:
            P.add_half_edge(v, u, cw=prev[v])
        else:
            P.add_half_edge(v, u)
        prev[v] = u

    for u, v in [(0, 1), (1, 2)]:
        add_undirected(u, v)

    # optional sanity check
    P.check_structure()

    with patch("matplotlib.pyplot.show"):
        ax = plotting.pplot(P, A)  # wraps gplot()

    out = Path(tmp_path) / "pplot.png"
    ax.figure.savefig(out, dpi=96)
    assert out.exists() and out.stat().st_size > 0


def test_compare_multiple_layouts(tmp_path):
    G1 = _make_graph()
    G2 = _make_graph(with_obstacle=True)

    # compare() creates its own Figure/Axes internally via plt.subplots
    with patch("matplotlib.pyplot.show"):
        plotting.compare([G1, G2])

    # ensure something is on the current figure stack and can be saved
    fig = plt.gcf()
    out = Path(tmp_path) / "compare.png"
    fig.savefig(out, dpi=96)
    assert out.exists() and out.stat().st_size > 0
