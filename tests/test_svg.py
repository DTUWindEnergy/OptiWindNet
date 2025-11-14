from .helpers import tiny_wfn
from optiwindnet.svg import svgplot, SvgRepr

def test_svgplot():
    
    wfn1 = tiny_wfn()
    G1 = wfn1.G
    svg_obj1 = svgplot(G1)
    assert isinstance(svg_obj1, SvgRepr)

    wfn2 = tiny_wfn(cables=1)
    G2 = wfn2.G
    svg_obj2 = svgplot(G2)
    assert isinstance(svg_obj2, SvgRepr)