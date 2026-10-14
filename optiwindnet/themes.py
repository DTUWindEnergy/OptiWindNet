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
    fallback_face: str
    highlight_edge: str
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
            'split': 'solid',
        }
        # kind2dasharray is used only by svg.py
        self.kind2dasharray = {
            'tentative': '18 15',
            'rogue': '25 5',
            'extended': '18 15',
            'contour_extended': '18 15',
            'scaffold': '10 10',
            'planar': '23 2 5 2',  # dash-dot, mirroring kind2style['planar']
            'detour': '18 15',
            'split': '24 8 4 8',
        }
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
                'original_boundaries': 'tomato',
                'unspecified': 'crimson',
                None: 'crimson',
                'detour': 'darkorange',
                'virtual': 'gold',
                'split': 'orangered',
            }
            self.fg_color = 'white'
            self.bg_color = 'black'
            self.term_edge = 'none'
            self.fallback_face = '#d0d0d0'
            self.highlight_edge = '#0060ff'
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
                'original_boundaries': 'firebrick',
                'unspecified': 'black',
                None: 'black',
                'detour': 'royalblue',
                'virtual': 'gold',
                'split': 'dimgray',
            }
            self.fg_color = 'black'
            self.bg_color = 'white'
            self.term_edge = 'black'
            self.fallback_face = '#d0d0d0'
            self.highlight_edge = '#001080'
            self.detour_ring = 'deepskyblue'
            self.border_face = '#eee'
            self.root_face = 'black'
            self.root_edge = self.border_face
        # First 20 colors from Colorcet's Glasbey palettes, rounded to RGB.
        # https://colorcet.holoviz.org/user_guide/Categorical.html
        # https://creativecommons.org/licenses/by/4.0/
        self.colors = (
            (
                '#d70000', '#028800', '#b600ff', '#06acc6', '#98ff00',
                '#ffa530', '#ff8fc8', '#79525f', '#00fecf', '#b0a5ff',
                '#94ad84', '#9a6900', '#376a62', '#d3008c', '#fef58f',
                '#c86f66', '#9ee3ff', '#00c946', '#a977ad', '#b8bb02',
            )
            if dark
            else (
                '#d70000', '#8c3cff', '#028800', '#00acc7', '#e7a500',
                '#ff7fd1', '#6c004f', '#583b00', '#005759', '#15e18d',
                '#0000dd', '#a2766a', '#bcb7ff', '#c004b9', '#645473',
                '#790000', '#0774d8', '#739b7d', '#ff7852', '#004b00',
            )
        )  # fmt: skip
