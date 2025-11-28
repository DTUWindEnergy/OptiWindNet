# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

from collections.abc import Sequence
from itertools import chain, product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from .geometric import rotate
from .interarraylib import describe_G
from .themes import Colors

__all__ = ('gplot', 'pplot')

FONTSIZE_LABEL = 5
FONTSIZE_LOAD = 7
FONTSIZE_ROOT_LABEL = 4
FONTSIZE_INFO_BOX = 12
FONTSIZE_LEGEND_STRIP = 6
NODESIZE = 35
NODESIZE_LABELED = 135
NODESIZE_LABELED_ROOT = 92
NODESIZE_DETOUR = 90
NODESIZE_LABELED_DETOUR = 215


def _is_ccw(X, Y):
    # Signed area Shoelace (https://stackoverflow.com/a/30408825/287217).
    return (
        X[-1] * Y[0] - Y[-1] * X[0] + np.dot(X[:-1], Y[1:]) - np.dot(Y[:-1], X[1:])
    ) >= 0


def gplot(
    G: nx.Graph,
    ax: Axes | None = None,
    node_tag: str | bool | None = None,
    landscape: bool = True,
    infobox: bool = True,
    scalebar: tuple[float, str] | None = None,
    hide_ST: bool = True,
    legend: bool = False,
    min_dpi: int = 192,
    dark: bool | None = None,
    tag_border: bool = False,
    **kwargs,
) -> Axes:
    """Plot site and routeset contained in G.

    This function relies on matplotlib and networkx's drawing functions. If no
    Axes instance is provided, a Figure with a single Axes will be created.
    Extra arguments given to gplot() will be forwarded to Figure().

    Args:
      ax: Axes instance to plot into. If `None`, opens a new figure.
      node_tag: text tag inside each node (e.g. 'load', 'label' or any of
        the nodes' attributes). If `True`, tags the nodes with their numbers.
      tag_border: if True, all border and obstacle vertices get a number tag.
      landscape: True -> rotate the plot by G's attribute 'landscape_angle'.
      infobox: Draw text box with summary of G's main properties: capacity,
        number of turbines, number of feeders, total cable length.
      scalebar: (span_in_data_units, label) add a small bar to indicate the
        plotted features' scale (lower right corner).
      hide_ST: If coordinates include a Delaunay supertriangle, adjust the
        viewport to fit only the actual vertices (i.e. no ST vertices).
      legend: Add description of linestyles and node shapes.
      min_dpi: Minimum dots per inch to use. matplotlib's default is used if
        it is greater than this value.
      **kwargs: passed on to matplotlib's Figure()

    Returns:
      Axes instance containing the plot.
    """
    c = Colors(dark)

    if node_tag is None:
        kw_axes = dict(aspect='equal', xmargin=0.005, ymargin=0.005)
        root_size = node_size = NODESIZE
        detour_size = NODESIZE_DETOUR
    else:
        kw_axes = dict(aspect='equal', xmargin=0.01, ymargin=0.01)
        root_size = NODESIZE_LABELED_ROOT
        detour_size = NODESIZE_LABELED_DETOUR
        node_size = NODESIZE_LABELED

    R, T, B = (G.graph[k] for k in 'RTB')
    VertexC = G.graph['VertexC']
    C, D = (G.graph.get(k, 0) for k in 'CD')
    border, obstacles, landscape_angle = (
        G.graph.get(k) for k in 'border obstacles landscape_angle'.split()
    )
    if landscape and landscape_angle:
        # landscape_angle is not None and not 0
        VertexC = rotate(VertexC, landscape_angle)

    if ax is None:
        dpi = max(min_dpi, plt.rcParams['figure.dpi'])
        kw_fig = dict(frameon=False, layout='constrained', dpi=dpi)
        fig = plt.figure(**(kw_fig | kwargs))
        ax = fig.add_subplot(**kw_axes)
    else:
        ax.set(**kw_axes)
    ax.set_axis_off()
    # draw farm border
    border_opt = dict(
        facecolor=c.border_face,
        linestyle='dashed',
        edgecolor=c.kind2color['border'],
        linewidth=0.7,
    )
    if border is not None:
        borderC = VertexC[border]

        if obstacles is None:
            ax.fill(*borderC.T, **border_opt)
        else:
            border_is_ccw = _is_ccw(*borderC.T)
            obstacleC_ = [VertexC[obstacle] for obstacle in obstacles]
            # path for the external border
            codes = (
                [Path.MOVETO]
                + (borderC.shape[0] - 1) * [Path.LINETO]
                + [Path.CLOSEPOLY]
            )
            points = [row for row in borderC] + [borderC[0]]
            # paths for the obstacle borders
            for obstacleC in obstacleC_:
                codes.extend(
                    [Path.MOVETO]
                    + (obstacleC.shape[0] - 1) * [Path.LINETO]
                    + [Path.CLOSEPOLY]
                )
                if _is_ccw(*obstacleC.T) != border_is_ccw:
                    points.extend([row for row in obstacleC] + [obstacleC[0]])
                else:
                    points.extend([row for row in obstacleC[::-1]] + [obstacleC[-1]])
            # create and add matplotlib artists
            path = Path(points, codes)
            patch = PathPatch(path, **border_opt)
            ax.add_patch(patch)
    elif obstacles is not None:
        # draw only obstacles
        for obstacle in obstacles:
            ax.fill(*VertexC[obstacle].T, **border_opt)

    # setup
    roots = range(-R, 0)
    pos = dict(enumerate(VertexC[:-R])) | dict(enumerate(VertexC[-R:], start=-R))
    if C > 0 or D > 0:
        fnT = G.graph['fnT']
        contour = range(T + B, T + B + C)
        detour = range(T + B + C, T + B + C + D)
        pos |= dict(zip(detour, VertexC[fnT[detour]]))
        pos |= dict(zip(contour, VertexC[fnT[contour]]))

    # default value for subtree (i.e. color for unconnected nodes)
    # is the last color of the tab20 colormap (i.e. 19)
    subtrees = G.nodes(data='subtree', default=19)
    node_colors = [c.colors[subtrees[n] % len(c.colors)] for n in range(T)]

    edges_width = 1.0
    edges_capstyle = 'round'

    # draw edges
    for graph, edge_kind in product((G, G.graph.get('overlay')), c.kind2style):
        if graph is None:
            continue
        edges = [(u, v) for u, v, kind in graph.edges.data('kind') if kind == edge_kind]
        if edges:
            if 'cables' in G.graph:
                # use variable edge width
                edges_width = [0.8 + 0.8 * G.edges[uv]['cable'] for uv in edges]

            art = nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=edges,
                label=(edge_kind or 'route'),
                width=edges_width,
                style=c.kind2style[edge_kind],
                alpha=c.kind2alpha[edge_kind],
                edge_color=c.kind2color[edge_kind],
                ax=ax,
            )
            art.set_capstyle(edges_capstyle)

    # draw nodes
    arts = nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=roots,
        linewidths=0.3,
        node_color=c.root_face,
        edgecolors=c.root_edge,
        node_size=root_size,
        node_shape='s',
        label='OSS',
    )
    arts.set_clip_on(False)
    arts = nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=range(T),
        edgecolors=c.term_edge,
        ax=ax,
        label='WTG',
        node_color=node_colors,
        node_size=node_size,
        linewidths=0.3,
    )
    arts.set_clip_on(False)
    if D:
        # draw rings around nodes that have Detour clones
        arts = nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=detour,
            alpha=0.4,
            edgecolors=c.detour_ring,
            node_color='none',
            node_size=detour_size,
            label='corner',
        )
        arts.set_clip_on(False)

    # draw labels
    if 'has_loads' in G.graph and node_tag == 'load':
        label_options = dict(
            labels={n: G.nodes[n].get('load', '-') for n in range(-R, T)},
            font_size=(
                {t: FONTSIZE_LOAD for t in range(T)}
                | {r: FONTSIZE_LABEL for r in range(-R, 0)}
            ),
        )
    elif isinstance(node_tag, str):
        # 'label' or some other node attr from node_tag
        label_options = dict(
            labels={n: G.nodes[n].get(node_tag, '') for n in range(-R, T)},
            font_size=(
                {t: FONTSIZE_LABEL for t in range(T)}
                | {r: FONTSIZE_ROOT_LABEL for r in range(-R, 0)}
            ),
        )
    elif node_tag is True:
        # use the node number as label
        label_options = dict(
            labels={n: str(n) for n in range(-R, T)},
            font_size=(
                {t: FONTSIZE_LABEL for t in range(T)}
                | {r: FONTSIZE_LOAD for r in range(-R, 0)}
            ),
        )
    else:
        label_options = dict(labels={})
    arts = nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_color={
            n: (c.root_edge if n < 0 else 'k') for n in label_options['labels'].keys()
        },
        **label_options,
    )
    for artist in arts.values():
        artist.set_clip_on(False)

    if scalebar is not None:
        bar = AnchoredSizeBar(ax.transData, *scalebar, 'lower right', frameon=False)
        ax.add_artist(bar)

    capacity = G.graph.get('capacity')
    if infobox and capacity is not None:
        # using the `legend()` method is a hack to get the `loc='best'` search
        # algorithm of matplotlib to place the info box not covering nodes
        info_art = ax.legend(
            [],
            labelspacing=0,
            facecolor=c.border_face,
            edgecolor=c.fg_color,
            title='\n'.join(describe_G(G)),
            framealpha=0.6,
            title_fontproperties={'size': FONTSIZE_INFO_BOX},
        )
        plt.setp(info_art.get_title(), multialignment='center', color=c.fg_color)
    else:
        info_art = None
    if legend:
        # even if calling `legend()` twice, the info box remains
        ax.legend(
            ncol=8,
            fontsize=FONTSIZE_LEGEND_STRIP,
            loc='lower center',
            columnspacing=1,
            labelcolor=c.fg_color,
            handletextpad=0.3,
            bbox_to_anchor=(0.5, -0.07),
            frameon=False,
        )
        if info_art is not None:
            ax.add_artist(info_art)
    if tag_border:
        border_ = border if border is not None else []
        obstacles_ = obstacles if obstacles is not None else [()]
        print(VertexC.shape)
        for b in chain(border_, *(obstacles_)):
            ax.text(*VertexC[b], str(b), color=c.fg_color, size=FONTSIZE_ROOT_LABEL)
    if hide_ST and VertexC.shape[0] > R + T + B:
        # coordinates include the supertriangle, adjust view limits to hide it
        nonStC = np.r_[VertexC[: T + B], VertexC[-R:]]
        minima = np.min(nonStC, axis=0)
        maxima = np.max(nonStC, axis=0)
        xmargin, ymargin = abs(maxima - minima) * 0.05
        (xlo, xhi), (ylo, yhi) = zip(minima, maxima)
        ax.set_xlim(xlo - xmargin, xhi + xmargin)
        ax.set_ylim(ylo - ymargin, yhi + ymargin)
    return ax


def pplot(P: nx.PlanarEmbedding, A: nx.Graph, **kwargs) -> Axes:
    """Plot PlanarEmbedding `P` using coordinates from `A`.

    Wrapper for `.plotting.gplot()`. Performs what one would expect
    from `gplot(P, ...)` - which does not work because P lacks coordinates and
    node 'kind' attribute. The source needs to be `A` (as opposed to `G` or
    `L`) because only `A` has the supertriangle's vertices coordinates.

    Args:
        P: Planar embedding to plot.
        A: source of vertex coordinates and 'kind'.

    Returns:
        Axes instance containing the plot.
    """
    H = nx.create_empty_copy(A)
    if 'has_loads' in H.graph:
        del H.graph['has_loads']
    R, T, B = (A.graph[k] for k in 'RTB')
    H.add_edges_from(P.edges, kind='planar')
    fnT = np.arange(R + T + B + 3)
    fnT[-R:] = range(-R, 0)
    H.graph['fnT'] = fnT
    return gplot(H, **kwargs)


def compare(positional=None, **title2G_dict):
    """
    Plot layouts side by side. dict keys are inserted in the title.
    Arguments must be either a sequence of graphs or multiple
    `keyword`=«graph_instance»`.
    """
    if positional is not None:
        if isinstance(positional, Sequence):
            title2G_dict |= {
                chr(i): val for i, val in enumerate(positional, start=ord('A'))
            }
        else:
            title2G_dict[''] = positional
    fig, axes = plt.subplots(1, len(title2G_dict), squeeze=False)
    for ax, (title, G) in zip(axes.ravel(), title2G_dict.items()):
        gplot(G, ax=ax, node_tag=None)
        creator = G.graph.get('creator', 'no edges')
        ax.set_title(f'{title} – {G.graph["name"]} ({creator})')
