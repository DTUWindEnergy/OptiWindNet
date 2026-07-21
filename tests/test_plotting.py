import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes

from optiwindnet.plotting import compare, gplot, pplot

from .helpers import tiny_wfn


@pytest.fixture(scope='module')
def wfn():
    return tiny_wfn()


def test_gplot_returns_axes(wfn):
    ax = gplot(wfn.G)
    assert isinstance(ax, Axes)
    plt.close('all')


def test_gplot_with_provided_axes(wfn):
    """When an existing Axes is passed, gplot reuses it and returns the same object."""
    fig, ax = plt.subplots()
    result = gplot(wfn.G, ax=ax)
    assert result is ax
    plt.close('all')


def test_gplot_node_tag_load(wfn):
    """node_tag='load' path uses has_loads branch and load-specific font sizes."""
    ax = gplot(wfn.G, node_tag='load')
    assert isinstance(ax, Axes)
    plt.close('all')


def test_gplot_node_tag_string_attr(wfn):
    """node_tag with any other string falls into the generic attribute branch."""
    ax = gplot(wfn.G, node_tag='label')
    assert isinstance(ax, Axes)
    plt.close('all')


def test_gplot_dark_mode(wfn):
    ax = gplot(wfn.G, dark=True)
    assert isinstance(ax, Axes)
    plt.close('all')


def test_gplot_light_mode(wfn):
    ax = gplot(wfn.G, dark=False)
    assert isinstance(ax, Axes)
    plt.close('all')


def test_gplot_border_with_no_obstacles(wfn):
    """Border present, no obstacles → simple ax.fill() path."""
    G = wfn.G.copy()
    G.graph['obstacles'] = None
    ax = gplot(G)
    assert isinstance(ax, Axes)
    plt.close('all')


def test_gplot_obstacles_without_border(wfn):
    """No border, obstacles present → draw-only-obstacles elif branch."""
    G = wfn.G.copy()
    G.graph['border'] = None
    ax = gplot(G)
    assert isinstance(ax, Axes)
    plt.close('all')


def test_gplot_no_border_no_obstacles(wfn):
    """Neither border nor obstacles → both branches skipped."""
    G = wfn.G.copy()
    G.graph['border'] = None
    G.graph['obstacles'] = None
    ax = gplot(G)
    assert isinstance(ax, Axes)
    plt.close('all')


def test_gplot_infobox_false(wfn):
    """With infobox=False the legend-based info box is not added."""
    ax = gplot(wfn.G, infobox=False)
    assert isinstance(ax, Axes)
    plt.close('all')


def test_gplot_landscape_false(wfn):
    """landscape=False skips the coordinate-rotation step."""
    ax = gplot(wfn.G, landscape=False)
    assert isinstance(ax, Axes)
    plt.close('all')


def test_gplot_hide_st_false(wfn):
    """hide_ST=False skips the supertriangle viewport clipping."""
    ax = gplot(wfn.G, hide_ST=False)
    assert isinstance(ax, Axes)
    plt.close('all')


def test_pplot_returns_axes(wfn):
    ax = pplot(wfn.P, wfn.A)
    assert isinstance(ax, Axes)
    plt.close('all')


def test_compare_list_of_graphs(wfn):
    """Positional list of graphs is mapped to letter keys A, B, …"""
    compare([wfn.G, wfn.G])
    plt.close('all')


def test_compare_single_graph(wfn):
    """A single non-Sequence positional argument is stored under key ''."""
    compare(wfn.G)
    plt.close('all')


def test_compare_keyword_graphs(wfn):
    """Keyword arguments are used directly as subplot titles."""
    compare(Before=wfn.G, After=wfn.G)
    plt.close('all')
