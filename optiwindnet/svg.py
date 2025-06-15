# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

from collections import defaultdict
from itertools import chain

import numpy as np
import networkx as nx
import svg

from .geometric import rotate
from .interarraylib import describe_G
from .themes import Colors

class SvgRepr():
    '''
    Helper class to get IPython to display the SVG figure encoded in data.
    '''

    def __init__(self, data: str):
        self.data = data

    def _repr_svg_(self) -> str:
        return self.data

    def save(self, filepath: str) -> None:
        '''write SVG to file `filepath`'''
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(self.data)


def svgplot(G: nx.Graph, *, landscape: bool = True, infobox: bool = True,
            dark=None, transparent: bool = True, node_size: int = 12,
            github_bugfix: bool = True) -> SvgRepr:
    '''Draw a NetworkX graph representation as SVG markup.

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
    '''
    c = Colors(dark)
    w, h = 1920, 1080
    margin = 30
    root_side = round(1.77*node_size)
    # TODO: ¿use SVG's attr overflow="visible" instead of margin?
    R, T, B = (G.graph[k] for k in 'RTB')
    VertexC = G.graph['VertexC']
    C, D = (G.graph.get(k, 0) for k in 'CD')
    border, obstacles, landscape_angle = (
        G.graph.get(k) for k in 'border obstacles landscape_angle'.split())
    if landscape and landscape_angle:
        # landscape_angle is not None and not 0
        VertexC = rotate(VertexC, landscape_angle)

    # viewport scaling
    idx_B = T + B
    Woff = min(VertexC[:idx_B, 0].min(), VertexC[-R:, 0].min())
    W = max(VertexC[:idx_B, 0].max(), VertexC[-R:, 0].max()) - Woff
    Hoff = min(VertexC[:idx_B, 1].min(), VertexC[-R:, 1].min())
    H = max(VertexC[:idx_B, 1].max(), VertexC[-R:, 1].max()) - Hoff
    wr = (w - 2*margin)/W
    hr = (h - 2*margin)/H
    if W/H < w/h:
        # tall aspect
        scale = hr
    else:
        # wide aspect
        scale = wr
        h = round(H*scale + 2*margin)
    offset = np.array((Woff, Hoff))
    VertexS = (VertexC - offset)*scale + margin
    # y axis flipping
    VertexS[:, 1] = h - VertexS[:, 1]
    VertexS = VertexS.round().astype(int)

    fnT = G.graph.get('fnT')
    if fnT is None:
        fnT = np.arange(R + T + B + 3)
        fnT[-R:] = range(-R, 0)

    #############################
    # generate the SVG elements #
    #############################
    # Defs (i.e. reusable elements)
    reusableE = [
        svg.Circle(id='wtg', stroke=c.node_edge, stroke_width=2, r=node_size),
        svg.Rect(id='oss', fill=c.root_color, stroke=c.border_face, stroke_width=2,
                 width=root_side, height=root_side),
    ]

    # prepare obstacles
    draw_obstacles = []
    if obstacles is not None:
        for obstacle in obstacles:
            draw_obstacles.append(
                'M' + ' '.join(str(c) for c in VertexS[obstacle].flat) + 'z')
    # border with obstacles as holes
    borderE: list[svg.Element]  = []
    if border is not None:
        borderE.append(svg.Path(
            id='border', stroke=c.kind2color['border'], stroke_dasharray=[15, 7],
            stroke_width=2, fill=c.border_face, fill_rule='evenodd',
            # fill_rule "evenodd" is agnostic to polygon vertices orientation
            # "nonzero" would depend on orientation (if opposite, no fill)
            d=' '.join(chain(
                ('M' + ' '.join(str(c) for c in VertexS[border].flat) + 'z',),
                draw_obstacles
            )),
        ))

    # Edges
    edges_with_kind = G.edges(data='kind')
    edge_lines = defaultdict(list)
    for u, v, edge_kind in edges_with_kind:
        if edge_kind == 'detour':
            # detours are drawn separately as polylines
            continue
        if edge_kind is None:
            edge_kind = 'unspecified'
        u, v = (u, v) if u < v else (v, u)
        edge_lines[edge_kind].append(
            svg.Line(x1=VertexS[fnT[u], 0], y1=VertexS[fnT[u], 1],
                     x2=VertexS[fnT[v], 0], y2=VertexS[fnT[v], 1]))
    edgesE: list[svg.Element]  = []
    for edge_kind, lines in edge_lines.items():
        group_attrs = {}
        if edge_kind in c.kind2dasharray:
            group_attrs['stroke_dasharray'] = c.kind2dasharray[edge_kind]
        edgesE.append(svg.G(
            id='edges_' + edge_kind, stroke=c.kind2color[edge_kind],
            stroke_width=4, **group_attrs, elements=lines,
        ))

    # detour elements
    detoursE: list[svg.Element]  = []
    if D > 0:
        # reusable ring for indicating clone-vertices
        reusableE.append(svg.Circle(id='dt', fill='none', stroke_opacity=0.3,
                         stroke=c.detour_ring, stroke_width=4, r=23))

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
        detoursE.extend((
            svg.G(
                id='detours', stroke=c.kind2color['detour'], stroke_width=4,
                stroke_dasharray=[18, 15], fill='none',
                elements=[svg.Polyline(points=points) for points in Points]
            ),
            svg.G( # Detour nodes
                id='DTgrp', elements=[
                    svg.Use(href='#dt', x=VertexS[d, 0], y=VertexS[d, 1])
                    for d in fnT[T + B + C: T + B + C + D]
                ]
            )
        ))

    # nodes (terminals and roots)
    subtrees = defaultdict(list)
    for n, sub in G.nodes(data='subtree', default=19):
        if 0 <= n < T:
            subtrees[sub].append(n)
    terminals = []
    for sub, nodes in subtrees.items():
        terminals.append(svg.G(
            fill=c.colors[sub % len(c.colors)],
            elements=[svg.Use(href='#wtg', x=VertexS[n, 0], y=VertexS[n, 1])
                      for n in nodes]))
    nodesE: list[svg.Element]  = [
        svg.G(id='WTGgrp', elements=terminals),
        svg.G(id='OSSgrp', elements=[
            svg.Use(href='#oss', x=VertexS[r, 0] - root_side/2,
                    y=VertexS[r, 1] - root_side/2) for r in range(-R, 0)
            ]
        ),
    ]

    # Infobox
    infoboxE: list[svg.Element] = []
    if infobox and G.graph.get('has_loads', False):
        reusableE.append(svg.Filter(
            id='bg_textbox',
            x=svg.Length(-5, '%'), y=svg.Length(-5, '%'),
            width=svg.Length(110, '%'), height=svg.Length(110, '%'),
            elements=[
                svg.FeFlood(flood_color=c.bg_color,
                            flood_opacity=0.6, result='bg'),
                svg.FeMerge(elements=[svg.FeMergeNode(in_='bg'),
                                      svg.FeMergeNode(in_='SourceGraphic')])
            ]
        ))
        right_anchor = round(W*scale + margin)
        desc_lines = describe_G(G)[::-1]

        if github_bugfix:
            # this is a workaround for GitHub's bug in rendering svg utf8 text
            # (only when the svg is inside an ipynb notebook)
            desc_lines = [l.encode('ascii', 'xmlcharrefreplace').decode()
                          for l in desc_lines]

        linesE: list[svg.Element] = [
            svg.TSpan(x=right_anchor,# dx=svg.Length(-0.2, 'em'),
                      dy=svg.Length((-1.3 if i else -0.), 'em'), text=line)
            for i, line in enumerate(desc_lines)
        ]
        infoboxE.append(svg.Text(
            x=right_anchor, y=h - margin, elements=linesE, fill=c.fg_color,
            font_size=40, text_anchor='end', font_family='sans-serif',
            filter='url(#bg_textbox)',
        ))

    # elements should be added according to the desired z-order
    graphElements = [*borderE, *edgesE, *detoursE, *nodesE]

    # Aggregate the SVG root elements
    rootElements: list[svg.Element] = []
    if not transparent:
        rootElements.append(svg.Rect(fill=c.bg_color, width=w, height=h))
    rootElements.extend((
        svg.Defs(elements=reusableE),
        svg.G(id=G.graph.get('handle', G.graph.get('name', 'handleless')),
              elements=graphElements),
        *infoboxE,
    ))

    # Aggregate all elements in the SVG figure.
    out = svg.SVG(
        viewBox=svg.ViewBoxSpec(0, 0, w, h),
        elements=rootElements,
    )
    return SvgRepr(out.as_str())
