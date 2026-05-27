from optiwindnet.themes import Colors


def test_themes():
    dark = Colors(dark=True)
    assert dark.fg_color == 'white' and dark.bg_color == 'black'
    light = Colors(dark=False)
    assert light.bg_color == 'white' and light.fg_color == 'black'


def test_themes_auto_detect():
    """Colors(dark=None) uses darkdetect and must not raise."""
    c = Colors()
    assert c.fg_color in ('white', 'black')
    assert c.bg_color in ('white', 'black')
    assert c.fg_color != c.bg_color


def test_themes_kind2dasharray():
    """Verify dasharray keys exist for SVG-specific edge kinds."""
    c = Colors(dark=False)
    for kind in ('planar', 'detour', 'tentative', 'scaffold'):
        assert kind in c.kind2dasharray

    c_dark = Colors(dark=True)
    assert c_dark.kind2dasharray == c.kind2dasharray
