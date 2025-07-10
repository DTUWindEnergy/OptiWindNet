# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

from collections import defaultdict
from itertools import chain

import networkx as nx
import numpy as np
import svg

from .geometric import rotate
from .interarraylib import describe_G
from .themes import Colors

__all__ = ('SvgRepr', 'svgplot')


class SvgRepr:
    """
    Helper class to get IPython to display the SVG figure encoded in data.
    """

    def __init__(self, data: str):
        self.data = data

    def _repr_svg_(self) -> str:
        return self.data

    def save(self, filepath: str) -> None:
        """write SVG to file `filepath`"""
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(self.data)


class Drawable:
    """
    SVG generator for NetworkX's Graph.
    """

    borderE: list[svg.Element]
    reusableE: list[svg.Element]
    edgesE: list[svg.Element]
    detoursE: list[svg.Element]
    nodesE: list[svg.Element]
    infoboxE: list[svg.Element]
    toplevelE: list[svg.Element]

    def __init__(
        self,
        G: nx.Graph,
        *,
        landscape: bool = True,
        dark: bool | None = None,
        transparent: bool = True,
    ):
        self.borderE = []
        self.reusableE = []
        self.edgesE = []
        self.detoursE = []
        self.nodesE = []
        self.infoboxE = []
        self.toplevelE = []
        self.G, self.landscape = G, landscape
        R, T, B = (G.graph[k] for k in 'RTB')
        self.R, self.T, self.B = R, T, B
        self.c = c = Colors(dark)
        fnT = G.graph.get('fnT')
        if fnT is None:
            fnT = np.arange(R + T + B + 3)
            fnT[-R:] = range(-R, 0)
        self.fnT = fnT

        ##############################
        # Coordinates transformation #
        ##############################
        G = self.G
        w, h = 1920, 1080
        margin = 30
        # TODO: ¿use SVG's attr overflow="visible" instead of margin?
        VertexC = G.graph['VertexC']
        landscape_angle = G.graph.get('landscape_angle', False)
        if self.landscape and landscape_angle:
            # landscape_angle is not None and not 0
            VertexC = rotate(VertexC, landscape_angle)

        # viewport scaling
        idx_B = self.T + self.B
        R = self.R
        Woff = min(VertexC[:idx_B, 0].min(), VertexC[-R:, 0].min())
        W = max(VertexC[:idx_B, 0].max(), VertexC[-R:, 0].max()) - Woff
        Hoff = min(VertexC[:idx_B, 1].min(), VertexC[-R:, 1].min())
        H = max(VertexC[:idx_B, 1].max(), VertexC[-R:, 1].max()) - Hoff
        wr = (w - 2 * margin) / W
        hr = (h - 2 * margin) / H
        if W / H < w / h:
            # tall aspect
            scale = hr
        else:
            # wide aspect
            scale = wr
            h = round(H * scale + 2 * margin)
        offset = np.array((Woff, Hoff))
        VertexS = (VertexC - offset) * scale + margin
        # y axis flipping
        VertexS[:, 1] = h - VertexS[:, 1]
        VertexS = VertexS.round().astype(int)
        self.VertexS = VertexS
        self.bottom_right_anchor = dict(x=round(W * scale + margin), y=h - margin)
        self.viewBox = svg.ViewBoxSpec(0, 0, w, h)

        #######################
        # Background elements #
        #######################
        if not transparent:
            # draw an opaque canvas the same size as the viewport
            self.toplevelE.append(svg.Rect(fill=c.bg_color, width=w, height=h))
        border, obstacles, landscape_angle = (
            G.graph.get(k) for k in 'border obstacles landscape_angle'.split()
        )
        # prepare obstacles
        draw_obstacles = []
        if obstacles is not None:
            for obstacle in obstacles:
                draw_obstacles.append(
                    'M' + ' '.join(str(c) for c in VertexS[obstacle].flat) + 'z'
                )
        # border with obstacles as holes
        if border is not None:
            self.borderE.append(
                svg.Path(
                    id='border',
                    stroke=c.kind2color['border'],
                    stroke_dasharray=[15, 7],
                    stroke_width=2,
                    fill=c.border_face,
                    fill_rule='evenodd',
                    # fill_rule "evenodd" is agnostic to polygon vertices orientation
                    # "nonzero" would depend on orientation (if opposite, no fill)
                    d=' '.join(
                        chain(
                            (
                                'M'
                                + ' '.join(str(c) for c in VertexS[border].flat)
                                + 'z',
                            ),
                            draw_obstacles,
                        )
                    ),
                )
            )

    def add_edges(self):
        fnT, c, VertexS = self.fnT, self.c, self.VertexS
        edges_with_kind = self.G.edges(data='kind')
        edge_lines = defaultdict(list)
        for u, v, edge_kind in edges_with_kind:
            if edge_kind == 'detour':
                # detours are drawn separately as polylines
                continue
            if edge_kind is None:
                edge_kind = 'unspecified'
            u, v = (u, v) if u < v else (v, u)
            edge_lines[edge_kind].append(
                svg.Line(
                    x1=VertexS[fnT[u], 0],
                    y1=VertexS[fnT[u], 1],
                    x2=VertexS[fnT[v], 0],
                    y2=VertexS[fnT[v], 1],
                )
            )
        edgesE = self.edgesE
        for edge_kind, lines in edge_lines.items():
            group_attrs = {}
            if edge_kind in c.kind2dasharray:
                group_attrs['stroke_dasharray'] = c.kind2dasharray[edge_kind]
            edgesE.append(
                svg.G(
                    id='edges_' + edge_kind,
                    stroke=c.kind2color[edge_kind],
                    stroke_width=4,
                    **group_attrs,
                    elements=lines,
                )
            )

    def add_detours(self):
        G, R, T, B = self.G, self.R, self.T, self.B
        C, D = (G.graph.get(k, 0) for k in 'CD')
        fnT, c, VertexS = self.fnT, self.c, self.VertexS
        # reusable ring for indicating clone-vertices
        self.reusableE.append(
            svg.Circle(
                id='dt',
                r=23,
                fill='none',
                stroke_opacity=0.3,
                stroke=c.detour_ring,
                stroke_width=4,
            )
        )

        # Detour edges as polylines (to align the dashes among overlapping lines)
        Points = []
        for r in range(-R, 0):
            detoured = [n for n in G.neighbors(r) if n >= T + B + C]
            for t in detoured:
                s = r
                hops = [s, fnT[t]]
                while True:
                    nbr = set(G.neighbors(t))
                    nbr.remove(s)
                    u = nbr.pop()
                    hops.append(fnT[u])
                    if u < T:
                        break
                    s, t = t, u
                Points.append(' '.join(str(c) for c in VertexS[hops].flat))
        self.detoursE.extend(
            (
                svg.G(
                    id='detours',
                    stroke=c.kind2color['detour'],
                    stroke_width=4,
                    stroke_dasharray=[18, 15],
                    fill='none',
                    elements=[svg.Polyline(points=points) for points in Points],
                ),
                svg.G(  # Detour nodes
                    id='DTgrp',
                    elements=[
                        svg.Use(href='#dt', x=VertexS[d, 0], y=VertexS[d, 1])
                        for d in fnT[T + B + C : T + B + C + D]
                    ],
                ),
            )
        )

    def add_nodes(self, node_size: int = 12):
        c, VertexS = self.c, self.VertexS
        G, R, T = self.G, self.R, self.T

        # reusable elements
        root_side = round(1.77 * node_size)
        self.reusableE.extend(
            (
                svg.Circle(id='wtg', stroke=c.term_edge, stroke_width=2, r=node_size),
                svg.Rect(
                    id='oss',
                    fill=c.root_face,
                    stroke=c.root_edge,
                    stroke_width=2,
                    width=root_side,
                    height=root_side,
                ),
            )
        )

        # nodes
        subtrees = defaultdict(list)
        for n, sub in G.nodes(data='subtree', default=19):
            if 0 <= n < T:
                subtrees[sub].append(n)
        terminals = []
        for sub, nodes in subtrees.items():
            terminals.append(
                svg.G(
                    fill=c.colors[sub % len(c.colors)],
                    elements=[
                        svg.Use(href='#wtg', x=VertexS[n, 0], y=VertexS[n, 1])
                        for n in nodes
                    ],
                )
            )
        self.nodesE.extend(
            (
                svg.G(id='WTGgrp', elements=terminals),
                svg.G(
                    id='OSSgrp',
                    elements=[
                        svg.Use(
                            href='#oss',
                            x=VertexS[r, 0] - root_side / 2,
                            y=VertexS[r, 1] - root_side / 2,
                        )
                        for r in range(-R, 0)
                    ],
                ),
            )
        )

    def add_box(self, github_bugfix: bool = True):
        self.reusableE.append(
            svg.Filter(
                id='bg_textbox',
                x=svg.Length(-5, '%'),
                y=svg.Length(-5, '%'),
                width=svg.Length(110, '%'),
                height=svg.Length(110, '%'),
                elements=[
                    svg.FeFlood(
                        flood_color=self.c.bg_color, flood_opacity=0.6, result='bg'
                    ),
                    svg.FeMerge(
                        elements=[
                            svg.FeMergeNode(in_='bg'),
                            svg.FeMergeNode(in_='SourceGraphic'),
                        ]
                    ),
                ],
            )
        )
        desc_lines = describe_G(self.G)[::-1]

        if github_bugfix:
            # this is a workaround for GitHub's bug in rendering svg utf8 text
            # (only when the svg is inside an ipynb notebook)
            desc_lines = [
                line.encode('ascii', 'xmlcharrefreplace').decode()
                for line in desc_lines
            ]

        linesE: list[svg.Element] = [
            svg.TSpan(
                x=self.bottom_right_anchor['x'],  # dx=svg.Length(-0.2, 'em'),
                dy=svg.Length((-1.3 if i else -0.0), 'em'),
                text=line,
            )
            for i, line in enumerate(desc_lines)
        ]
        self.infoboxE.append(
            svg.Text(
                **self.bottom_right_anchor,
                elements=linesE,
                fill=self.c.fg_color,
                font_size=40,
                text_anchor='end',
                font_family='sans-serif',
                filter='url(#bg_textbox)',
            )
        )

    def to_svg(self) -> str:
        # elements should be added according to the desired z-order
        graphElements = [*self.borderE, *self.edgesE, *self.detoursE, *self.nodesE]

        self.toplevelE.extend(
            (
                svg.Defs(elements=self.reusableE),
                svg.G(
                    id=self.G.graph.get(
                        'handle', self.G.graph.get('name', 'handleless')
                    ),
                    elements=graphElements,
                ),
                *self.infoboxE,
            )
        )

        # Aggregate all elements in the SVG figure.
        out = svg.SVG(
            viewBox=self.viewBox,
            elements=self.toplevelE,
        )
        return out.as_str()


def svgplot(
    G: nx.Graph,
    *,
    landscape: bool = True,
    infobox: bool = True,
    dark: bool | None = None,
    transparent: bool = True,
    node_size: int = 12,
    github_bugfix: bool = True,
) -> SvgRepr:
    """Draw a NetworkX graph representation as SVG markup.

    If using interactively (e.g. Jupyter notebook), the returned object must
    either be the cell's output or be passed to IPython's display() function.

    Alternative to own.plotting.gplot() because matplotlib's svg backend does
    not make efficient use of SVG primitives.

    Args:
      G: graph to plot
      landscape: rotate(?) the plot by G's graph attribute 'landscape_angle'.
      infobox: add(?) text box with summary of G's main properties: capacity,
        number of turbines, excess feeders, total feeders, total cable length.
      dark: color theme to use: True -> dark; False: light; None -> guess
      transparent: background color: True -> transparent; False -> theme-based

    Returns:
      SvgRepr object containing the SVG markup in its 'data' attribute
    """

    drawable = Drawable(G, landscape=landscape, dark=dark, transparent=transparent)

    drawable.add_edges()
    if G.graph.get('D', False):
        drawable.add_detours()
    drawable.add_nodes(node_size=node_size)
    if infobox and G.graph.get('has_loads', False):
        drawable.add_box(github_bugfix=github_bugfix)

    return SvgRepr(drawable.to_svg())
