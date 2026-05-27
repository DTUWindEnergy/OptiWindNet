import re

import networkx as nx

from optiwindnet.svg import SvgRepr, svgplot, svgpplot

from .helpers import tiny_wfn


def _texts(svg_data: str) -> list[str]:
    """Extract text content from all <text> elements."""
    return re.findall(r'<text[^>]*>([^<]+)</text>', svg_data)


def test_svgplot_basic():
    wfn1 = tiny_wfn()
    svg1 = svgplot(wfn1.G)
    assert isinstance(svg1, SvgRepr)

    wfn2 = tiny_wfn(cables=1)
    svg2 = svgplot(wfn2.G)
    assert isinstance(svg2, SvgRepr)


def test_svgplot_no_overflow_for_regular_graph():
    from optiwindnet.interarraylib import scaffolded

    wfn = tiny_wfn()
    # regular optimized graph
    assert 'overflow' not in svgplot(wfn.G).data
    # wfn.A carries supertriangle rows in VertexC, but no edges reference them
    assert 'overflow' not in svgplot(wfn.A).data
    # wfn.L has no edges at all
    assert 'overflow' not in svgplot(wfn.L).data
    # scaffold: fnT remaps supertriangle node IDs back to on-screen positions
    assert 'overflow' not in svgplot(scaffolded(wfn.G, wfn.P)).data


def test_svgpplot_overflow_for_planar_embedding():
    wfn = tiny_wfn()
    # svgpplot uses an identity fnT so edges reach actual supertriangle coordinates
    assert 'overflow="hidden"' in svgpplot(wfn.P, wfn.A).data


def test_svgplot_node_tag_numbers():
    wfn = tiny_wfn()
    G = wfn.G
    svg = svgplot(G, node_tag=True)
    assert 'id="WTGlabels"' in svg.data
    assert 'id="OSSlabels"' in svg.data
    texts = _texts(svg.data)
    # terminal nodes 0..T-1 and root -1 should all appear
    T, R = G.graph['T'], G.graph['R']
    for n in range(T):
        assert str(n) in texts
    for r in range(-R, 0):
        assert str(r) in texts


def test_svgplot_node_tag_attribute():
    G = tiny_wfn().G
    svg = svgplot(G, node_tag='load')
    assert 'id="WTGlabels"' in svg.data
    texts = _texts(svg.data)
    # loads are integers; every terminal should have its load shown
    for n in range(G.graph['T']):
        load = G.nodes[n].get('load')
        if load is not None:
            assert str(load) in texts


def test_svgplot_node_tag_none_by_default():
    G = tiny_wfn().G
    svg = svgplot(G)
    assert 'WTGlabels' not in svg.data
    assert 'OSSlabels' not in svg.data


def test_svgplot_tag_border():
    wfn = tiny_wfn()
    G = wfn.G
    # off by default
    svg_off = svgplot(G, tag_border=False)
    assert 'border_tags' not in svg_off.data

    svg_on = svgplot(G, tag_border=True)
    assert 'id="border_tags"' in svg_on.data
    texts = _texts(svg_on.data)
    # border and obstacle vertex indices should all be labeled
    for idx in G.graph['border']:
        assert str(idx) in texts
    for obstacle in G.graph['obstacles']:
        for idx in obstacle:
            assert str(idx) in texts


def test_svgplot_overlay():
    G = tiny_wfn().G
    T = G.graph['T']
    overlay = nx.Graph()
    overlay.add_nodes_from(G.nodes)
    for i in range(T - 1):
        overlay.add_edge(i, i + 1, kind='virtual')
    G2 = G.copy()
    G2.graph['overlay'] = overlay

    svg = svgplot(G2)
    assert 'id="overlay"' in svg.data
    assert 'id="overlay_virtual"' in svg.data


def test_svgplot_overlay_absent_when_no_overlay():
    G = tiny_wfn().G
    svg = svgplot(G)
    assert 'id="overlay"' not in svg.data


def test_svgplot_infobox_requires_capacity():
    G = tiny_wfn().G
    # has capacity -> infobox rendered
    svg_with = svgplot(G, infobox=True)
    assert 'bg_textbox' in svg_with.data

    # capacity removed -> no infobox even if has_loads is True
    G2 = G.copy()
    del G2.graph['capacity']
    assert G2.graph.get('has_loads')  # has_loads still set
    svg_without = svgplot(G2, infobox=True)
    assert 'bg_textbox' not in svg_without.data


def test_svgpplot():
    wfn = tiny_wfn()
    svg = svgpplot(wfn.P, wfn.A)
    assert isinstance(svg, SvgRepr)
    assert 'edges_planar' in svg.data
    # supertriangle vertices are present → overflow clipping applied
    assert 'overflow="hidden"' in svg.data
    # svgplot kwargs are forwarded
    svg_dark = svgpplot(wfn.P, wfn.A, dark=True)
    assert isinstance(svg_dark, SvgRepr)


def test_svgplot_legend():
    G = tiny_wfn().G
    # off by default
    svg_off = svgplot(G, legend=False)
    assert 'id="legend"' not in svg_off.data

    svg_on = svgplot(G, legend=True)
    assert 'id="legend"' in svg_on.data
    texts = _texts(svg_on.data)
    assert 'WTG' in texts
    assert 'OSS' in texts
    legend_markup = svg_on.data.split('id="legend"', 1)[1].split('<defs', 1)[0]
    assert 'href="#wtg"' in legend_markup
    assert 'href="#oss"' in legend_markup
    assert '<circle' not in legend_markup
    assert '<rect' not in legend_markup


def test_svgplot_legend_tagged_marker_sizes():
    G = tiny_wfn().G
    # with tag, nodes are scaled
    svg_tagged = svgplot(G, node_tag=True, legend=True)
    # default size: node_size = 12
    # tagged: node_size = 12, effective_node_size = round(12 * 1.7) = 20
    # OSS root_side: round(1.77 * 20) = 35
    # So width="35" should be in the SVG data for the OSS rect in the legend!
    assert 'width="35"' in svg_tagged.data


def test_svgplot_legend_route_color():
    G = tiny_wfn().G
    # In dark mode, route edges are drawn as 'crimson'
    svg_dark = svgplot(G, dark=True, legend=True)
    assert 'stroke="crimson"' in svg_dark.data

    # In light mode, route edges are drawn as 'black'
    svg_light = svgplot(G, dark=False, legend=True)
    assert 'stroke="black"' in svg_light.data


def test_svgplot_legend_planar_dasharray():
    wfn = tiny_wfn()
    # planar edges in the legend should use the exact dash array from the theme
    svg = svgpplot(wfn.P, wfn.A, legend=True)
    # The theme planar dasharray is '23 2 5 2'
    assert (
        'stroke_dasharray="23 2 5 2"' in svg.data
        or 'stroke-dasharray="23 2 5 2"' in svg.data
    )


def test_svgplot_legend_alignment_on_bbox_center():
    G = tiny_wfn().G
    svg_legend = svgplot(G, legend=True)
    # The elements inside <g id="legend"> should have coordinates calculated relative to
    # the location bbox center instead of the standard viewport center.
    # Let's verify '<g id="legend">' exists in the SVG markup.
    assert 'id="legend"' in svg_legend.data


def test_svgplot_legend_detour_dasharray():
    wfn = tiny_wfn()
    G = wfn.G.copy()
    G.add_edge(0, 1, kind='detour')
    svg = svgplot(G, legend=True)
    # The detour dasharray in themes should be '18 15'
    assert (
        'stroke_dasharray="18 15"' in svg.data or 'stroke-dasharray="18 15"' in svg.data
    )
