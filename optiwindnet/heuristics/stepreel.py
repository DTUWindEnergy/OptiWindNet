# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import subprocess
from shutil import which

import pylunasvg
from PIL import Image

from ..svg import Drawable


def make_reel(S, G, filename):
    steps_log = S.graph['solver_details']['steps_log']
    drawable = Drawable(G, transparent=False)

    drawable.add_terminals_ungrouped()

    reel = []
    c = drawable.c
    num_colors = len(drawable.c.colors)
    for i, links in steps_log.items():
        for u, v in links:
            drawable.add_edge(u, v)
            # update node colors
            if u >= 0:
                drawable.terminals_group.elements[u].fill = c.colors[
                    S.nodes[u]['subtree'] % num_colors
                ]
            if v >= 0:
                drawable.terminals_group.elements[v].fill = c.colors[
                    S.nodes[v]['subtree'] % num_colors
                ]

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

    gifsicle = which('gifsicle')
    if gifsicle is not None:
        subprocess.run([gifsicle, '--batch', '--optimize=3', '--colors=256', filename])
    return
