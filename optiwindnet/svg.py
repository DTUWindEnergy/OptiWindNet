# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

from collections import defaultdict
from itertools import chain
from typing import Any

import networkx as nx
import numpy as np
import svg

from .geometric import rotate
from .interarraylib import describe_G
from .themes import Colors

__all__ = ('SvgRepr', 'svgplot', 'svgpplot')

_NODE_RADII = 12, 20
_RING_RADII = 23, 28
_BORDER_WIDTH = 2
_LINK_WIDTH = 4
_LINK_TYPE_WIDTH_STEP = 3
_NODE_EDGE_WIDTH = 2
_DETOUR_RING_WIDTH = 4


class SvgRepr:
    """
    Helper class to get IPython to display the SVG figure encoded in data.
    """

    def __init__(self, data: str, metadata: dict[str, Any] | None = None):
        self.data = data
        self.metadata: dict[str, Any] = metadata or {}

    def _repr_svg_(self) -> str:
        return self.data

    def __repr__(self) -> str:
        m = self.metadata
        parts = [f'SvgRepr {m["handle"]!r}'] if 'handle' in m else ['SvgRepr']
        name = m.get('name')
        if name and name != m.get('handle'):
            parts.append(f'name={name!r}')
        if 'T' in m:
            parts.append(f'T={m["T"]}')
        if 'R' in m:
            parts.append(f'R={m["R"]}')
        if m.get('capacity') is not None:
            parts.append(f'capacity={m["capacity"]}')
        parts.append(f'{len(self.data)} chars')
        return '<' + ' '.join(parts) + '>'

    def save(self, filepath: str) -> None:
        """write SVG to file ``filepath``"""
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(self.data)


class Drawable:
    """
    SVG generator for NetworkX's Graph.
    """

    margin: int = 30
    borderE: list[svg.Element]
    reusableE: list[svg.Element]
    edgesE: list[svg.Element]
    detoursE: list[svg.Element]
    nodesE: list[svg.Element]
    infoboxE: list[svg.Element]
    toplevelE: list[svg.Element]
    metadata: dict[str, Any]

    def __init__(
        self,
        G: nx.Graph,
        *,
        landscape: bool = True,
        dark: bool | None = None,
        transparent: bool = True,
        legend: bool = False,
    ):
        self.legend = legend
        self.effective_node_radius = 12
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
        name = G.graph.get('name')
        handle = G.graph.get('handle', name if name is not None else 'handleless')
        self.handle = handle
        self.metadata = {'handle': handle, 'T': T, 'R': R}
        if name is not None:
            self.metadata['name'] = name
        capacity = G.graph.get('capacity')
        if capacity is not None:
            self.metadata['capacity'] = capacity
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
        margin = self.margin
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
        self.h_orig = h
        if self.legend:
            h = h + 80
        self.viewBox = svg.ViewBoxSpec(0, 0, w, h)
        self.w, self.h = w, h
        self.overflow = None  # set to 'hidden' by add_edges() if needed

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
        if border is not None:
            # border with obstacles as holes
            self.borderE.append(
                svg.Path(
                    id='border',
                    stroke=c.kind2color['border'],
                    stroke_dasharray=[15, 7],
                    stroke_width=_BORDER_WIDTH,
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
        elif draw_obstacles:
            # draw only the obstacles
            self.borderE.append(
                svg.Path(
                    id='border',
                    stroke=c.kind2color['border'],
                    stroke_dasharray=[15, 7],
                    stroke_width=_BORDER_WIDTH,
                    fill=c.border_face,
                    d=draw_obstacles,
                )
            )

    def _line(self, u, v) -> svg.Line:
        (x1, y1), (x2, y2) = self.VertexS[self.fnT[u]], self.VertexS[self.fnT[v]]
        return svg.Line(x1=x1, y1=y1, x2=x2, y2=y2)

    def _kind_group(self, id: str, kind: str, lines: list, **attrs) -> svg.G:
        c = self.c
        if kind in c.kind2dasharray:
            attrs['stroke_dasharray'] = c.kind2dasharray[kind]
        return svg.G(id=id, stroke=c.kind2color[kind], elements=lines, **attrs)

    def add_edges(self):
        fnT, VertexS = self.fnT, self.VertexS
        w, h = self.w, self.h
        edge_widths = [
            _LINK_TYPE_WIDTH_STEP * (i + 1)
            for i, _ in enumerate(self.G.graph.get('cables', (0,)))
        ]
        edge_lines_ = [defaultdict(list) for _ in edge_widths]
        for u, v, edgeD in self.G.edges(data=True):
            kind = edgeD.get('kind', 'unspecified')
            if kind == 'detour':
                # detours are drawn separately as polylines
                continue
            u, v = (u, v) if u < v else (v, u)
            (x1, y1), (x2, y2) = VertexS[fnT[u]], VertexS[fnT[v]]
            if self.overflow is None and not (
                0 <= x1 <= w and 0 <= y1 <= h and 0 <= x2 <= w and 0 <= y2 <= h
            ):
                self.overflow = 'hidden'
            edge_lines_[edgeD.get('cable', 0)][kind].append(
                svg.Line(x1=x1, y1=y1, x2=x2, y2=y2)
            )
        edges_super_group = self.edgesE
        for cable_type, (stroke_width, edge_lines) in enumerate(
            zip(edge_widths, edge_lines_)
        ):
            if len(edge_widths) > 1:
                # two grouping levels
                edgesE = []
                extra_attrs = {}
            else:
                # single grouping level
                edgesE = edges_super_group
                extra_attrs = dict(stroke_width=_LINK_WIDTH)
            for edge_kind, lines in edge_lines.items():
                edgesE.append(
                    self._kind_group(
                        'edges_' + edge_kind, edge_kind, lines, **extra_attrs
                    )
                )
            if len(edge_widths) > 1:
                # two grouping levels
                edges_super_group.append(
                    svg.G(
                        id=f'cable_{cable_type}',
                        stroke_width=stroke_width,
                        elements=edgesE,
                    )
                )
        # overlay graph (e.g. from PathFinder.best_paths_overlay())
        overlay = self.G.graph.get('overlay')
        if overlay is not None:
            overlay_by_kind = defaultdict(list)
            for u, v, edgeD in overlay.edges(data=True):
                kind = edgeD.get('kind', 'unspecified')
                u, v = (u, v) if u < v else (v, u)
                overlay_by_kind[kind].append(self._line(u, v))
            kind_groups = [
                self._kind_group(
                    f'overlay_{kind}',
                    kind,
                    lines,
                    stroke_width=_LINK_WIDTH,
                    opacity=self.c.kind2alpha[kind],
                )
                for kind, lines in overlay_by_kind.items()
            ]
            self.edgesE.append(svg.G(id='overlay', elements=kind_groups))

    def add_detours(self, size_selector: int = 0):
        G, R, T, B = self.G, self.R, self.T, self.B
        C, D = (G.graph.get(k, 0) for k in 'CD')
        fnT, c, VertexS = self.fnT, self.c, self.VertexS
        # reusable ring for indicating clone-vertices
        self.reusableE.append(
            svg.Circle(
                id='dt',
                r=_RING_RADII[size_selector],
                fill='none',
                stroke_opacity=0.3,
                stroke=c.detour_ring,
                stroke_width=_DETOUR_RING_WIDTH,
            )
        )

        # Detour edges as polylines (to align the dashes among overlapping lines)
        points__ = defaultdict(list)
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
                points__[G[s][t].get('cable', None)].append(
                    ' '.join(str(c) for c in VertexS[hops].flat)
                )
        common_attr = dict(
            stroke=c.kind2color['detour'],
            stroke_dasharray=[18, 15],
            fill='none',
        )
        if None in points__:
            detours = [
                svg.G(
                    id='detours',
                    **common_attr,
                    stroke_width=_LINK_WIDTH,
                    elements=[svg.Polyline(points=points) for points in points__[None]],
                ),
            ]
        else:
            detours = [
                svg.G(
                    id=f'detours_{cable_type}',
                    **common_attr,
                    stroke_width=_LINK_TYPE_WIDTH_STEP * (cable_type + 1),
                    elements=[svg.Polyline(points=points) for points in points_],
                )
                for cable_type, points_ in points__.items()
            ]

        self.detoursE.extend(
            (
                *detours,
                svg.G(  # Detour nodes
                    id='DTgrp',
                    elements=[
                        svg.Use(href='#dt', x=VertexS[d, 0], y=VertexS[d, 1])
                        for d in fnT[T + B + C : T + B + C + D]
                    ],
                ),
            )
        )

    def add_nodes(self, node_tag: str | bool | None = None):
        node_radius = _NODE_RADII[node_tag is not None]
        c, VertexS = self.c, self.VertexS
        G, R, T = self.G, self.R, self.T

        # reusable elements
        self.root_side = root_side = round(1.77 * node_radius)
        self.reusableE.extend(
            (
                svg.Circle(
                    id='wtg',
                    stroke=c.term_edge,
                    stroke_width=_NODE_EDGE_WIDTH,
                    r=node_radius,
                ),
                svg.Rect(
                    id='oss',
                    fill=c.root_face,
                    stroke=c.root_edge,
                    stroke_width=_NODE_EDGE_WIDTH,
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

        # node labels
        if node_tag is not None:
            has_loads = G.graph.get('has_loads', False)

            def get_label(n):
                if node_tag is True:
                    return str(n)
                if node_tag == 'load' and has_loads:
                    return str(G.nodes[n].get('load', '-'))
                if isinstance(node_tag, str):
                    val = G.nodes[n].get(node_tag, '')
                    return str(val) if val != '' else ''
                return ''

            base_attrs = {
                'font-family': 'sans-serif',
                'text-anchor': 'middle',
                'dominant-baseline': 'central',
            }
            # turbine/root font sizes mirror gplot's per-tag scheme, whose
            # FONTSIZE_ROOT_LABEL : FONTSIZE_LABEL : FONTSIZE_LOAD = 4 : 5 : 7
            small, normal, large = (round(node_radius * f) for f in (0.8, 1.0, 1.4))
            if node_tag == 'load' and has_loads:
                wtg_font, oss_font = large, normal
            elif node_tag is True:
                wtg_font, oss_font = normal, large
            else:
                wtg_font, oss_font = normal, small
            wtg_labels = [
                svg.Text(x=VertexS[n, 0], y=VertexS[n, 1], text=lbl)
                for n in range(T)
                if (lbl := get_label(n))
            ]
            oss_labels = [
                svg.Text(x=VertexS[r, 0], y=VertexS[r, 1], text=lbl)
                for r in range(-R, 0)
                if (lbl := get_label(r))
            ]
            if wtg_labels:
                self.nodesE.append(
                    svg.G(
                        id='WTGlabels',
                        fill='black',
                        extra={'font-size': wtg_font, **base_attrs},
                        elements=wtg_labels,
                    )
                )
            if oss_labels:
                self.nodesE.append(
                    svg.G(
                        id='OSSlabels',
                        fill=c.root_edge,
                        extra={'font-size': oss_font, **base_attrs},
                        elements=oss_labels,
                    )
                )

    def add_border_tags(self, node_radius: int = 12):
        G, c, VertexS = self.G, self.c, self.VertexS
        border = G.graph.get('border')
        obstacles = G.graph.get('obstacles')
        border_ = border if border is not None else []
        obstacles_ = obstacles if obstacles is not None else [()]
        tags = [
            svg.Text(x=VertexS[b, 0], y=VertexS[b, 1], text=str(b))
            for b in chain(border_, *obstacles_)
        ]
        if tags:
            self.nodesE.append(
                svg.G(
                    id='border_tags',
                    fill=c.fg_color,
                    extra={
                        'font-size': round(node_radius * 1.3),
                        'font-family': 'sans-serif',
                    },
                    elements=tags,
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

    def add_legend(self):
        c, G = self.c, self.G
        legend_items = []

        # 1. WTG
        legend_items.append(('node', 'wtg', 'WTG', c.colors[0], 'circle'))

        # 2. OSS
        legend_items.append(('node', 'oss', 'OSS', c.root_face, 'rect'))

        # 3. corner (if detour/clone exists)
        if G.graph.get('D', 0) > 0:
            legend_items.append(('node', 'corner', 'corner', 'none', 'ring'))

        # 4. Edges (collect unique kinds from G and overlay if any)
        kinds = set()
        for u, v, k in G.edges.data('kind'):
            if k is not None:
                kinds.add(k)
            else:
                kinds.add('route')

        overlay = G.graph.get('overlay')
        if overlay is not None:
            for u, v, k in overlay.edges.data('kind'):
                if k is not None:
                    kinds.add(k)
                else:
                    kinds.add('route')

        for kind in sorted(kinds):
            color_key = None if kind == 'route' else kind
            legend_items.append(
                (
                    'edge',
                    kind,
                    kind,
                    c.kind2color.get(color_key, c.fg_color),
                    c.kind2dasharray.get(color_key),
                )
            )

        # Layout metrics
        item_width = 180
        N = len(legend_items)
        total_width = N * item_width
        bbox_center_x = (self.bottom_right_anchor['x'] + self.margin) / 2
        start_x = bbox_center_x - total_width / 2
        y_pos = self.h_orig + 40

        elements = []
        labels = []
        for i, item in enumerate(legend_items):
            x_pos = start_x + i * item_width
            item_type = item[0]
            label = ''

            if item_type == 'node':
                _, name, label, color, shape = item
                if shape == 'circle':
                    elements.append(
                        svg.Use(href='#wtg', x=x_pos + 20, y=y_pos, fill=color)
                    )
                elif shape == 'rect':
                    elements.append(
                        svg.Use(
                            href='#oss',
                            x=x_pos + 20 - self.root_side / 2,
                            y=y_pos - self.root_side / 2,
                        )
                    )
                elif shape == 'ring':
                    elements.append(svg.Use(href='#dt', x=x_pos + 20, y=y_pos))
            elif item_type == 'edge':
                _, name, label, color, dash = item
                attrs = {
                    'x1': x_pos,
                    'y1': y_pos,
                    'x2': x_pos + 40,
                    'y2': y_pos,
                    'stroke': color,
                    'stroke_width': _LINK_WIDTH,
                }
                if dash:
                    attrs['stroke_dasharray'] = dash
                elements.append(svg.Line(**attrs))

            labels.append(svg.Text(x=x_pos + 50, y=y_pos, text=label))

        elements.append(
            svg.G(
                id='legend_labels',
                fill=c.fg_color,
                extra={
                    'font-size': '24',
                    'font-family': 'sans-serif',
                    'dominant-baseline': 'central',
                },
                elements=labels,
            )
        )
        self.toplevelE.append(svg.G(id='legend', elements=elements))

    def to_svg(self) -> str:
        if self.legend:
            self.add_legend()
        # elements should be added according to the desired z-order
        graphElements = [*self.borderE, *self.edgesE, *self.detoursE, *self.nodesE]

        self.toplevelE.extend(
            (
                svg.Defs(elements=self.reusableE),
                svg.G(id=self.handle, elements=graphElements),
                *self.infoboxE,
            )
        )

        # Aggregate all elements in the SVG figure.
        out = svg.SVG(
            viewBox=self.viewBox,
            overflow=self.overflow,
            elements=self.toplevelE,
        )
        return out.as_str()


def svgplot(
    G: nx.Graph,
    *,
    landscape: bool = True,
    node_tag: str | bool | None = None,
    tag_border: bool = False,
    infobox: bool = True,
    legend: bool = False,
    dark: bool | None = None,
    transparent: bool = True,
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
      node_tag: text label inside each node. Use True for node numbers, 'load'
        for power flow values (requires has_loads), or any node attribute name.
      tag_border: if True, label all border and obstacle vertices with their
        index numbers (useful for geometry debugging).
      infobox: add(?) text box with summary of G's main properties: capacity,
        number of turbines, excess feeders, total feeders, total cable length.
      legend: if True, add a legend strip at the bottom of the SVG plot.
      dark: color theme to use: True -> dark; False: light; None -> guess
      transparent: background color: True -> transparent; False -> theme-based

    Returns:
      SvgRepr object containing the SVG markup in its 'data' attribute
    """

    drawable = Drawable(
        G,
        landscape=landscape,
        dark=dark,
        transparent=transparent,
        legend=legend,
    )

    drawable.add_edges()
    if G.graph.get('D', False):
        drawable.add_detours(size_selector=int(node_tag is not None))
    drawable.add_nodes(node_tag=node_tag)
    if tag_border:
        drawable.add_border_tags()
    if infobox and G.graph.get('capacity') is not None:
        drawable.add_box(github_bugfix=github_bugfix)

    return SvgRepr(drawable.to_svg(), drawable.metadata)


def svgpplot(P: nx.PlanarEmbedding, A: nx.Graph, **kwargs) -> SvgRepr:
    """Plot PlanarEmbedding ``P`` using coordinates from ``A`` as SVG markup.

    SVG equivalent of :func:`.plotting.pplot`. Accepts the same keyword arguments
    as :func:`svgplot`.

    Args:
      P: planar embedding to plot.
      A: source of vertex coordinates and node attributes.

    Returns:
      SvgRepr object containing the SVG markup in its ``'data'`` attribute
    """
    H = nx.create_empty_copy(A)
    if 'has_loads' in H.graph:
        del H.graph['has_loads']
    R, T, B = (A.graph[k] for k in 'RTB')
    H.add_edges_from(P.edges, kind='planar')
    fnT = np.arange(R + T + B + 3)
    fnT[-R:] = range(-R, 0)
    H.graph['fnT'] = fnT
    return svgplot(H, **kwargs)
