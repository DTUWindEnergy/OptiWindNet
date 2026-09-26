# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import math
from collections import defaultdict
from collections.abc import Mapping
from itertools import chain
from types import MappingProxyType
from typing import Any

import networkx as nx
import numpy as np
import svg

from .geometric import rotate
from .presenting import (
    _format_length,
    _format_nominal_power,
    _nominal_loads,
    _terminal_groups,
    describe_G,
)
from .themes import Colors

__all__ = ('SvgRepr', 'svgplot', 'svgpplot')

# SvgRepr's repr, first line: problem and configuration properties
_STATIC_KEYS = ('name', 'T', 'R', 'capacity', 'topology', 'cables')
# SvgRepr's repr, second line: properties that change with the solution
_SOLUTION_KEYS = ('feeders', 'C', 'D', 'length', 'cost')
# SvgRepr's repr, third line: the topology's identity
_ID_KEY = 'topology_id'
# keys used only by other keys' formatters
_ANCILLARY_KEYS = ('currency',)
# keys not rendered as '«key»«sep»«value»'
_KEY_FORMATTER = {
    'cables': lambda meta: 'cables=' + '|'.join(str(c) for c in meta['cables']),
    'feeders': lambda meta: 'feeders = ' + '|'.join(str(f) for f in meta['feeders']),
    'length': lambda meta: 'Σλ = ' + _format_length(meta['length']) + ' m',
    'cost': lambda meta: (
        'Σ¤ = {:_.0f} '.format(meta['cost']) + meta.get('currency', '')
    ).rstrip(),
    # bytes() so that a non-bytes value raises TypeError into _render's fallback
    'topology_id': lambda meta: 'topology_id = ' + bytes(meta['topology_id']).hex(),
}
_KNOWN_KEYS = frozenset(_STATIC_KEYS + _SOLUTION_KEYS + _ANCILLARY_KEYS + (_ID_KEY,))

_NODE_RADII = 12, 20
_RING_RADII = 23, 28
_BORDER_WIDTH = 2
_LINK_WIDTH = 4
_LINK_TYPE_WIDTH_STEP = 3
_NODE_EDGE_WIDTH = 2
_DETOUR_RING_WIDTH = 4


def _closed_path(S: np.ndarray) -> str:
    """Make SVG path data for the closed polygon with vertices ``S``."""
    return 'M' + ' '.join(str(c) for c in S.flat) + 'z'


def _render(key: str, metadata: dict[str, Any], sep: str = '=') -> str:
    """Render metadata entry ``key`` for SvgRepr's repr (never raises)."""
    formatter = _KEY_FORMATTER.get(key)
    if formatter is not None:
        try:
            return formatter(metadata)
        except (TypeError, ValueError, KeyError):
            # unexpected value type: fall back to the default rendering
            pass
    return f'{key}{sep}{metadata[key]}'


class SvgRepr:
    """
    Helper class to get IPython to display the SVG figure encoded in data.

    ``metadata`` holds unformatted values, so that instances may be compared
    programmatically. Its repr renders ``_STATIC_KEYS`` in the first line,
    ``_SOLUTION_KEYS`` in the second and the topology id in the third.
    """

    def __init__(self, data: str, metadata: Mapping[str, Any] = MappingProxyType({})):
        self.data = data
        self.metadata = dict(metadata)
        self.handle = self.metadata.pop('handle', '')
        if self.handle == self.metadata.get('name'):
            del self.metadata['name']

    def _repr_svg_(self) -> str:
        return self.data

    def __repr__(self) -> str:
        metadata = self.metadata
        static = [_render(key, metadata) for key in _STATIC_KEYS if key in metadata]
        # unrecognized keys go to the first line
        static.extend(
            _render(key, metadata) for key in metadata if key not in _KNOWN_KEYS
        )
        solution = [
            _render(key, metadata, ' = ') for key in _SOLUTION_KEYS if key in metadata
        ]
        # markup size changes with the solution
        solution.append(f'{len(self.data)} chars')
        if len(solution) == 1:
            lines = ['; '.join(static + solution)]
        else:
            lines = ['; '.join(static), '; '.join(solution)]
        if _ID_KEY in metadata:
            lines.append(_render(_ID_KEY, metadata, ' = '))
        return f'<SvgRepr[{self.handle}]: ' + '\n '.join(lines) + '>'

    def save(self, filepath: str) -> None:
        """Write SVG to file ``filepath``."""
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(self.data)


class Drawable:
    """
    SVG generator for NetworkX's Graph.
    """

    margin: int = 30
    #: Narrowest width-to-height ratio of a tight viewBox with a legend, so that
    #: the legend of a very tall site gets a few items per row.
    legend_min_aspect: float = 9 / 16
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
        tight: bool = False,
    ):
        self.legend = legend
        self.transparent = transparent
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
        self.handle = G.graph.get('handle', name if name is not None else '')
        self.metadata = {'handle': self.handle} | {
            key: G.graph[key]
            for key in ('name', 'T', 'R', 'capacity', 'topology')
            if key in G.graph
        }
        if 'cables' in G.graph:
            self.metadata['cables'] = tuple(κ for κ, _ in G.graph['cables'])
        if G.graph.get('has_loads'):
            # routeset: add the solution properties
            self.metadata['feeders'] = tuple(G.degree[r] for r in range(-1, -R - 1, -1))
            self.metadata['C'] = G.graph.get('C', 0)
            self.metadata['D'] = G.graph.get('D', 0)
            length = G.size(weight='length')
            if length > 0:
                self.metadata['length'] = length
            if 'currency' in G.graph:
                self.metadata['cost'] = G.size(weight='cost')
                self.metadata['currency'] = G.graph['currency']
            # absent from routesets read back from the database: pack_G() withholds
            # the underscore-prefixed graph attributes
            topology_id = G.graph.get('_topology_id')
            if topology_id is not None:
                self.metadata[_ID_KEY] = topology_id
        self.c = c = Colors(dark)
        fnT = G.graph.get('fnT')
        if fnT is None:
            fnT = np.arange(R + T + B + 3)
            fnT[len(fnT) - R :] = range(-R, 0)
        self.fnT = fnT

        ##############################
        # Coordinates transformation #
        ##############################
        G = self.G
        w, h = 1920, 1080
        margin = self.margin
        # TODO: ¿use SVG's attr overflow="visible" instead of margin?
        VertexC = G.graph['VertexC']
        boundaryC_ = G.graph.get('_original_boundaries', ())
        landscape_angle = G.graph.get('landscape_angle', False)
        if self.landscape and landscape_angle:
            # landscape_angle is not None and not 0
            VertexC = rotate(VertexC, landscape_angle)
            boundaryC_ = [rotate(C, landscape_angle) for C in boundaryC_]

        # viewport scaling
        idx_B = self.T + self.B
        extentC = np.vstack(
            (VertexC[:idx_B], VertexC[VertexC.shape[0] - self.R :], *boundaryC_)
        )
        Woff, Hoff = extentC.min(axis=0)
        W, H = extentC.max(axis=0) - (Woff, Hoff)
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

        def to_svg(C: np.ndarray) -> np.ndarray:
            S = (C - offset) * scale + margin
            # y axis flipping
            S[:, 1] = h - S[:, 1]
            return S.round().astype(int)

        self.VertexS = VertexS = to_svg(VertexC)
        boundaryS_ = [to_svg(C) for C in boundaryC_]
        self.bottom_right_anchor = {'x': round(W * scale + margin), 'y': h - margin}
        min_x = 0
        if tight:
            w = self.bottom_right_anchor['x'] + margin
            if legend and w < h * self.legend_min_aspect:
                # widen evenly on both sides to keep the drawing centred
                min_x = -math.ceil((h * self.legend_min_aspect - w) / 2)
        self.viewBox = svg.ViewBoxSpec(min_x, 0, w - 2 * min_x, h)
        self.w, self.h, self.min_x = w, h, min_x
        self.overflow = None  # set to 'hidden' by add_edges() if needed

        #######################
        # Background elements #
        #######################
        border, obstacles, landscape_angle = (
            G.graph.get(k) for k in ['border', 'obstacles', 'landscape_angle']
        )
        # prepare obstacles
        draw_obstacles = []
        if obstacles is not None:
            for obstacle in obstacles:
                draw_obstacles.append(_closed_path(VertexS[obstacle]))
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
                    # svg.py types `d` as list[PathData], but it renders a
                    # pre-joined path string just as well
                    d=' '.join(  # pyrefly: ignore[bad-argument-type]
                        chain((_closed_path(VertexS[border]),), draw_obstacles)
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
        if boundaryS_:
            # pre-buffering border and obstacles (outline only)
            self.borderE.append(
                svg.Path(
                    id='original_boundaries',
                    stroke=c.kind2color['original_boundaries'],
                    stroke_width=_BORDER_WIDTH,
                    fill='none',
                    # svg.py types `d` as list[PathData], but it renders a
                    # pre-joined path string just as well
                    d=' '.join(  # pyrefly: ignore[bad-argument-type]
                        _closed_path(S) for S in boundaryS_
                    ),
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
            if edgeD.get('load') == 0:
                # ring zero-load link: keep geometry, draw with the 'split' style
                kind = 'split'
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
                extra_attrs = {'stroke_width': _LINK_WIDTH}
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
                if edgeD.get('load') == 0:
                    kind = 'split'
                u, v = (u, v) if u < v else (v, u)
                overlay_by_kind[kind].append(self._line(u, v))
            kind_groups: list[svg.Element] = [
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
        common_attr: dict[str, Any] = {
            'stroke': c.kind2color['detour'],
            'stroke_dasharray': [18, 15],
            'fill': 'none',
        }
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
        self.terminal_groups = terminal_groups = _terminal_groups(G)
        href_from_terminal = {}
        scale_from_sides = {}
        for sides, scale, _, terminals in terminal_groups:
            href = '#wtg' if sides == 0 else f'#wtg{sides}'
            href_from_terminal |= dict.fromkeys(terminals, href)
            scale_from_sides[sides] = scale
        for sides, scale in sorted(scale_from_sides.items()):
            if sides == 0:
                self.reusableE.append(
                    svg.Circle(
                        id='wtg',
                        stroke=c.term_edge,
                        stroke_width=_NODE_EDGE_WIDTH,
                        r=node_radius,
                    )
                )
            else:
                # same area as the circle, vertex pointing up
                angles = 2 * np.pi * np.arange(sides) / sides
                vertices = scale * node_radius * np.c_[np.sin(angles), -np.cos(angles)]
                self.reusableE.append(
                    svg.Polygon(
                        id=f'wtg{sides}',
                        stroke=c.term_edge,
                        stroke_width=_NODE_EDGE_WIDTH,
                        # adding 0.0 turns -0.0 into 0.0
                        points=(vertices.round(1) + 0.0).ravel().tolist(),
                    )
                )
        self.reusableE.extend(
            (
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
                        svg.Use(
                            href=href_from_terminal[n],
                            x=VertexS[n, 0],
                            y=VertexS[n, 1],
                        )
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
            power_per_inflow = G.graph.get('power_per_inflow', 1)
            nominal = (
                _nominal_loads(G) if has_loads and node_tag == 'load_nominal' else {}
            )

            def get_label(n):
                if node_tag is True:
                    return str(n)
                if node_tag in ('load', 'load_nominal') and has_loads:
                    if node_tag == 'load_nominal':
                        load = nominal.get(n)
                        return '-' if load is None else f'{float(load):g}'
                    load = G.nodes[n].get('load')
                    return '-' if load is None else str(load)
                if node_tag == 'power':
                    if n < 0:
                        return ''
                    return _format_nominal_power(G.nodes[n], power_per_inflow)
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
            if node_tag in ('load', 'load_nominal') and has_loads:
                wtg_font, oss_font = large, normal
            elif node_tag is True:
                wtg_font, oss_font = normal, large
            else:
                wtg_font, oss_font = normal, small
            wtg_labels: list[svg.Element] = [
                svg.Text(x=VertexS[n, 0], y=VertexS[n, 1], text=lbl)
                for n in range(T)
                if (lbl := get_label(n))
            ]
            oss_labels: list[svg.Element] = [
                svg.Text(x=VertexS[r, 0], y=VertexS[r, 1], text=lbl)
                for r in range(-R, 0)
                if (lbl := get_label(r))
            ]
            if wtg_labels:
                self.nodesE.append(
                    svg.G(
                        id='WTGlabels',
                        fill='black',
                        extra={'font-size': str(wtg_font), **base_attrs},
                        elements=wtg_labels,
                    )
                )
            if oss_labels:
                self.nodesE.append(
                    svg.G(
                        id='OSSlabels',
                        fill=c.root_edge,
                        extra={'font-size': str(oss_font), **base_attrs},
                        elements=oss_labels,
                    )
                )

    def add_border_tags(self, node_radius: int = 12):
        G, c, VertexS = self.G, self.c, self.VertexS
        border = G.graph.get('border')
        obstacles = G.graph.get('obstacles')
        border_ = border if border is not None else []
        obstacles_ = obstacles if obstacles is not None else [()]
        tags: list[svg.Element] = [
            svg.Text(x=VertexS[b, 0], y=VertexS[b, 1], text=str(b))
            for b in chain(border_, *obstacles_)
        ]
        if tags:
            self.nodesE.append(
                svg.G(
                    id='border_tags',
                    fill=c.fg_color,
                    extra={
                        'font-size': str(round(node_radius * 1.3)),
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

        # 1. WTG (one item per power level)
        for sides, _, label, _ in self.terminal_groups:
            href = '#wtg' if sides == 0 else f'#wtg{sides}'
            legend_items.append(('node', href, label, c.colors[0], 'terminal'))

        # 2. OSS
        legend_items.append(('node', 'oss', 'OSS', c.root_face, 'rect'))

        # 3. corner (if detour/clone exists)
        if G.graph.get('D', 0) > 0:
            legend_items.append(('node', 'corner', 'corner', 'none', 'ring'))

        # 4. Edges (collect unique kinds from G and overlay if any). A ring
        # zero-load link is drawn with the 'split' style regardless of its
        # geometry kind, so it contributes 'split' to the legend.
        def _legend_kind(d):
            if d.get('load') == 0:
                return 'split'
            k = d.get('kind')
            return k if k is not None else 'route'

        kinds = set()
        for u, v, d in G.edges(data=True):
            kinds.add(_legend_kind(d))

        overlay = G.graph.get('overlay')
        if overlay is not None:
            for u, v, d in overlay.edges(data=True):
                kinds.add(_legend_kind(d))

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

        if '_original_boundaries' in G.graph:
            legend_items.append(
                (
                    'edge',
                    'original_boundaries',
                    'pre-buffer',
                    c.kind2color['original_boundaries'],
                    None,
                )
            )

        # Layout metrics: rows span the viewBox (which a tight one fits to the
        # drawing), each centred under the drawing unless that crosses an edge
        item_width, row_pitch = 180, 50
        N = len(legend_items)
        margin = self.margin
        left, right = self.min_x + margin, self.w - self.min_x - margin
        center_x = (self.bottom_right_anchor['x'] + margin) / 2
        per_row = max(1, (right - left) // item_width)
        rows = -(-N // per_row)
        self.viewBox.height = self.h + 80 + (rows - 1) * row_pitch

        elements = []
        labels = []
        for i, item in enumerate(legend_items):
            row, col = divmod(i, per_row)
            row_len = min(per_row, N - row * per_row)
            row_width = row_len * item_width
            start_x = min(max(center_x - row_width / 2, left), right - row_width)
            x_pos = start_x + col * item_width
            y_pos = self.h + 40 + row * row_pitch
            item_type = item[0]
            label = ''

            if item_type == 'node':
                _, name, label, color, shape = item
                if shape == 'terminal':
                    elements.append(
                        svg.Use(href=name, x=x_pos + 20, y=y_pos, fill=color)
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
                _, _name, label, color, dash = item
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
        if not self.transparent:
            # opaque canvas covering the final viewBox, legend rows included
            vb = self.viewBox
            self.toplevelE.insert(
                0,
                svg.Rect(
                    fill=self.c.bg_color,
                    x=vb.min_x or None,
                    width=vb.width,
                    height=vb.height,
                ),
            )
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
    tight: bool = False,
) -> SvgRepr:
    """Draw a NetworkX graph representation as SVG markup.

    If using interactively (e.g. Jupyter notebook), the returned object must
    either be the cell's output or be passed to IPython's display() function.

    Alternative to own.plotting.gplot() because matplotlib's svg backend does
    not make efficient use of SVG primitives.

    Args:
      G: graph to plot
      landscape: rotate(?) the plot by G's graph attribute ``'landscape_angle'``.
      node_tag: text label inside each node. Use ``True`` for node numbers,
        ``'load'`` for cumulative integer inflow (requires ``has_loads``), or any
        node attribute name. ``'power'`` displays declared turbine ratings,
        falling back to inflow times ``'power_per_inflow'``.
        ``'load_nominal'`` displays cumulative nominal power, requiring current
        routing loads. Exact quantization scales ``'load'`` for display only;
        inexact quantization accumulates turbine ratings through the network,
        refreshing the graph's ``'load_nominal'`` attributes.
      tag_border: if ``True``, label all border and obstacle vertices with their
        index numbers (useful for geometry debugging).
      infobox: add(?) text box with summary of G's main properties: capacity,
        number of turbines, excess feeders, total feeders, total cable length.
      legend: if ``True``, add a legend strip at the bottom of the SVG plot. Its
        rows are centred under the drawing, shifted inward where that would cross
        an edge of the viewBox, and wrap where they would not fit its width.
      dark: color theme to use: ``True`` → dark; ``False``: light; ``None`` → guess
      transparent: background color: ``True`` → transparent; ``False`` → theme-based
      tight: if ``True``, narrow the viewBox to the drawing's width. A site
        taller than 16:9 otherwise sits on the left of a 1920-wide viewBox, with
        the rest blank. Wide sites are unaffected, as their viewBox height already
        fits the drawing. With ``legend``, the viewBox is kept at least 9:16 by
        padding both sides of the drawing evenly, so that a very tall site's legend
        still fits a few items per row.

    Returns:
      SvgRepr object containing the SVG markup in its ``'data'`` attribute
    """

    drawable = Drawable(
        G,
        landscape=landscape,
        dark=dark,
        transparent=transparent,
        legend=legend,
        tight=tight,
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
    fnT[len(fnT) - R :] = range(-R, 0)
    H.graph['fnT'] = fnT
    return svgplot(H, **kwargs)
