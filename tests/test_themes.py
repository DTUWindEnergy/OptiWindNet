from optiwindnet.themes import Colors
import pytest

def test_themes():
    dark = Colors(dark=True)
    assert dark.fg_color == 'white' and dark.bg_color == 'black'
    light = Colors(dark=False)
    assert light.bg_color == 'white' and light.fg_color == 'black'
 