# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import pylunasvg
from PIL import Image

from ..svg import Drawable


def make_reel(S, G, filename):
    log = S.graph['log']
    drawable = Drawable(G, transparent=False)

    drawable.add_terminals_ungrouped()

    reel = []
    c = drawable.c
    num_colors = len(drawable.c.colors)
    for u, v in log:
        drawable.add_edge(u, v)

        # update node colors
        drawable.terminals_group.elements[u].fill = \
            c.colors[S.nodes[u]['subtree'] % num_colors]
        drawable.terminals_group.elements[v].fill = \
            c.colors[S.nodes[v]['subtree'] % num_colors]

        # plot and rasterize
        luna = pylunasvg.Document.load_from_data(drawable.to_svg())
        bmp = luna.render_to_bitmap()
        bmp.convert_to_rgba()
        im = Image.frombytes('RGBA', (int(luna.width), int(luna.height)), bmp.data)
        reel.append(im)

    if G.graph.get('D', False):
        # last frame shows the detours
        drawable.add_detours()
        luna = pylunasvg.Document.load_from_data(drawable.to_svg())
        bmp = luna.render_to_bitmap()
        bmp.convert_to_rgba()
        im = Image.frombytes('RGBA', (int(luna.width), int(luna.height)), bmp.data)
        reel.append(im)

    reel[0].save(filename, save_all=True, append_images=reel[1:], duration=333, loop=0)
    return 

