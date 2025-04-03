# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

from collections import defaultdict

import darkdetect


class Colors:
    kind2color: dict
    kind2alpha: dict
    kind2style: dict
    kind2dasharray: dict
    fg_color: str
    bg_color: str
    root_color: str
    root_edge: str
    term_edge: str
    detour_ring: str
    border_face: str

    def __init__(self, dark: bool | None = None):
        if dark is None:
            dark = darkdetect.isDark()

        self.kind2alpha = defaultdict(lambda: 1.0)
        self.kind2alpha['virtual'] = 0.4
        # kind2style is used only by plotting.py
        self.kind2style = {
            'scaffold': 'dotted',
            'delaunay': 'solid',
            'extended': 'dashed',
            'tentative': 'dashdot',
            'rogue': 'dashed',
            'contour_delaunay': 'solid',
            'contour_extended': 'dashed',
            'contour': 'solid',
            'planar': 'dashdot',
            'constraint': 'solid',
            'border': 'dashed',
            None: 'solid',
            'detour': (0, (3, 3)),
            'virtual': 'solid',
        }
        # kind2dasharray is used only by svg.py
        self.kind2dasharray = dict(
            tentative='18 15',
            rogue='25 5',
            extended='18 15',
            contour_extended='18 15',
            scaffold='10 10',
        )
        if dark:
            self.kind2color = {
                'scaffold': 'gray',
                'delaunay': 'darkcyan',
                'extended': 'darkcyan',
                'tentative': 'red',
                'rogue': 'yellow',
                'contour_delaunay': 'green',
                'contour_extended': 'green',
                'contour': 'red',
                'planar': 'darkorchid',
                'constraint': 'purple',
                'border': 'silver',
                'unspecified': 'crimson',
                None: 'crimson',
                'detour': 'darkorange',
                'virtual': 'gold',
            }
            self.fg_color = 'white'
            self.bg_color = 'black'
            self.term_edge = 'none'
            self.detour_ring = 'orange'
            self.border_face = '#111'
            self.root_face = 'lawngreen'
            self.root_edge = self.border_face
        else:
            self.kind2color = {
                'scaffold': 'gray',
                'delaunay': 'darkgreen',
                'extended': 'darkgreen',
                'tentative': 'darkorange',
                'rogue': 'magenta',
                'contour_delaunay': 'firebrick',
                'contour_extended': 'firebrick',
                'contour': 'black',
                'planar': 'darkorchid',
                'constraint': 'darkcyan',
                'border': 'dimgray',
                'unspecified': 'black',
                None: 'black',
                'detour': 'royalblue',
                'virtual': 'gold',
            }
            self.fg_color = 'black'
            self.bg_color = 'white'
            self.term_edge = 'black'
            self.detour_ring = 'deepskyblue'
            self.border_face = '#eee'
            self.root_face = 'black'
            self.root_edge = self.border_face
        # matplotlib tab20
        self.colors = (
            '#1f77b4',
            '#aec7e8',
            '#ff7f0e',
            '#ffbb78',
            '#2ca02c',
            '#98df8a',
            '#d62728',
            '#ff9896',
            '#9467bd',
            '#c5b0d5',
            '#8c564b',
            '#c49c94',
            '#e377c2',
            '#f7b6d2',
            '#7f7f7f',
            '#c7c7c7',
            '#bcbd22',
            '#dbdb8d',
            '#17becf',
            '#9edae5',
        )
