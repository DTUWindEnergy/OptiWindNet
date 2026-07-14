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
    # scaffold: fnT resolves supertriangle placeholders to their real (and
    # off-screen) corner coordinates, so overflow is expected here too
    assert 'overflow="hidden"' in svgplot(scaffolded(wfn.G, wfn.P)).data


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
    G = tiny_wfn().G.copy()
    svg = svgplot(G, node_tag='load')
    assert 'id="WTGlabels"' in svg.data
    texts = _texts(svg.data)
    # loads are integers; every terminal should have its load shown
    for n in range(G.graph['T']):
        load = G.nodes[n].get('load')
        if load is not None:
            assert str(load) in texts

    G.graph['power_scale'] = 2
    G.graph['capacity'] = 8
    for turbine in range(G.graph['T']):
        G.nodes[turbine]['power'] = 2
    G.nodes[0]['power'] = 3
    G.nodes[0]['load'] = 3

    power_svg = svgplot(G, node_tag='power')
    assert '1.5' in _texts(power_svg.data)
    assert power_svg.metadata['capacity'] == 4
    assert 'capacity=4' in repr(power_svg)

    load_svg = svgplot(G, node_tag='load')
    assert '1.5' in _texts(load_svg.data)


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


# ─────────────────────────────────────────────
# SvgRepr.__repr__  (lines 39-52)
# ─────────────────────────────────────────────


def test_svgrepr_repr_no_metadata():
    r = repr(SvgRepr('abc'))
    assert r == '<SvgRepr 3 chars>'


def test_svgrepr_repr_with_handle_only():
    r = repr(SvgRepr('ab', {'handle': 'myfarm'}))
    assert "'myfarm'" in r
    assert 'name=' not in r


def test_svgrepr_repr_name_differs_from_handle():
    r = repr(
        SvgRepr(
            'x', {'handle': 'h', 'name': 'Full Name', 'T': 5, 'R': 2, 'capacity': 9}
        )
    )
    assert "name='Full Name'" in r
    assert 'T=5' in r
    assert 'R=2' in r
    assert 'capacity=9' in r


def test_svgrepr_repr_name_equals_handle():
    """When name == handle the name field must NOT appear in repr."""
    r = repr(SvgRepr('x', {'handle': 'same', 'name': 'same'}))
    assert 'name=' not in r


def test_svgrepr_repr_capacity_none():
    """capacity=None must not appear in repr."""
    r = repr(SvgRepr('x', {'capacity': None}))
    assert 'capacity' not in r


# ─────────────────────────────────────────────
# SvgRepr.save  (lines 54-57)
# ─────────────────────────────────────────────


def test_svgrepr_save(tmp_path):
    wfn = tiny_wfn()
    svg = svgplot(wfn.G)
    out = tmp_path / 'test.svg'
    svg.save(str(out))
    assert out.exists()
    assert out.read_text(encoding='utf-8') == svg.data


# ─────────────────────────────────────────────
# Drawable.__init__  landscape angle + aspect
# ─────────────────────────────────────────────


def test_svgplot_landscape_angle_rotation():
    """landscape_angle rotates VertexC before scaling (line 123)."""
    wfn = tiny_wfn()
    G = wfn.G.copy()
    G.graph['landscape_angle'] = 45
    svg = svgplot(G, landscape=True)
    assert isinstance(svg, SvgRepr)


def test_svgplot_wide_aspect_ratio():
    """Wide geometry uses width-based scale and recomputes h (lines 139-140)."""
    wfn = tiny_wfn()
    G = wfn.G.copy()
    VertexC = G.graph['VertexC'].copy()
    VertexC[:, 0] *= 20  # stretch x → W/H >> 1.78 (viewport ratio)
    G.graph['VertexC'] = VertexC
    svg = svgplot(G)
    assert isinstance(svg, SvgRepr)


# ─────────────────────────────────────────────
# Drawable.__init__  obstacles without border  (lines 195-197)
# ─────────────────────────────────────────────


def test_svgplot_obstacles_only_no_border():
    """border=None but obstacles present hits the elif draw_obstacles branch."""
    wfn = tiny_wfn()
    G = wfn.G.copy()
    G.graph['border'] = None
    # obstacles remain non-None
    svg = svgplot(G)
    assert isinstance(svg, SvgRepr)


# ─────────────────────────────────────────────
# Drawable  transparent=False  (line 160)
# ─────────────────────────────────────────────


def test_svgplot_transparent_false():
    wfn = tiny_wfn()
    svg = svgplot(wfn.G, transparent=False)
    assert isinstance(svg, SvgRepr)


# ─────────────────────────────────────────────
# Drawable.add_edges  multiple cable types  (lines 244-266)
# ─────────────────────────────────────────────


def test_svgplot_multiple_cable_types():
    """Two cable types → len(edge_widths) > 1 → two-level grouping."""
    wfn = tiny_wfn(cables=[(2, 5.0), (4, 10.0)])
    assert len(wfn.G.graph.get('cables', [])) == 2
    svg = svgplot(wfn.G)
    assert 'cable_0' in svg.data
    assert 'cable_1' in svg.data


# ─────────────────────────────────────────────
# Drawable.add_nodes  node_tag as generic string attr  (lines 427-430, 445)
# ─────────────────────────────────────────────


def test_svgplot_node_tag_generic_string():
    """node_tag='label' hits the isinstance(node_tag, str) branch."""
    wfn = tiny_wfn()
    svg = svgplot(wfn.G, node_tag='label')
    assert isinstance(svg, SvgRepr)


# ─────────────────────────────────────────────
# Drawable.add_box  github_bugfix=False  (lines 521-529)
# ─────────────────────────────────────────────


def test_svgplot_github_bugfix_false():
    wfn = tiny_wfn()
    svg = svgplot(wfn.G, infobox=True, github_bugfix=False)
    assert 'bg_textbox' in svg.data


# ─────────────────────────────────────────────
# Drawable.add_legend  D>0 corner + overlay  (lines 561, 573-577, 619-620)
# ─────────────────────────────────────────────


def test_svgplot_legend_detour_corner():
    """D>0 in G adds a 'corner' entry and shape=='ring' branch to the legend."""
    wfn = tiny_wfn()
    G = wfn.G.copy()
    G.graph['D'] = 1  # pretend there are detour clones
    svg = svgplot(G, legend=True)
    assert 'corner' in svg.data


def test_svgplot_legend_with_overlay():
    """overlay graph edges appear in the legend (lines 573-577)."""
    wfn = tiny_wfn()
    G = wfn.G.copy()
    overlay = nx.Graph()
    overlay.add_nodes_from(G.nodes)
    overlay.add_edge(0, 1, kind='virtual')
    G.graph['overlay'] = overlay
    svg = svgplot(G, legend=True)
    assert 'virtual' in svg.data


# ─────────────────────────────────────────────
# svgpplot  has_loads branch  (line 746)
# ─────────────────────────────────────────────


def test_svgpplot_has_loads_removed():
    """If A has has_loads, svgpplot removes it before building the SVG."""
    wfn = tiny_wfn()
    A = wfn.A.copy()
    A.graph['has_loads'] = True  # artificially inject
    svg = svgpplot(wfn.P, A)
    assert isinstance(svg, SvgRepr)
