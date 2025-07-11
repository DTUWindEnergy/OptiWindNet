from collections import defaultdict
from shutil import which

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import animation
from scipy.spatial.distance import cdist

from optiwindnet.interarraylib import calcload


class LayoutPlotter:
    """Plotter for making step-by-step animations. The heuristics actually
    generate a log of actions and the plotter is only called after the entire
    process is over. This class is only instantiated inside of `animate()`."""

    edge_color = 'crimson'
    root_color = 'lawngreen'
    node_tag = 'label'
    colors = plt.get_cmap('tab20', 20).colors

    def __init__(
        self, G_base, ax=None, dpi=None, node_tag='label', start_from_star=False
    ):
        # fail early if graph does not have a log
        loglist = G_base.graph['log']
        if 'has_loads' not in G_base.graph:
            calcload(G_base)

        if dpi is None:
            dpi = plt.rcParams['figure.dpi']
        # self.mm = mm = 25.4*dpi
        self.node_tag = node_tag
        self.start_from_star = start_from_star
        self.G_base = G_base
        self.root_node_size = (
            NODESIZE_LABELED_ROOT if self.node_tag is not None else NODESIZE
        )
        self.node_size = NODESIZE_LABELED if self.node_tag is not None else NODESIZE

        log = [(0, [['nop', None]])]
        iprev = 0
        for i, *entry in loglist:
            # print(i)
            if i != iprev:
                out = [entry]
                iprev = i
                log.append((i, out))
            else:
                out.append(entry)

        log.append((i, [['end', None]]))
        self.log = log

        VertexC = G_base.graph['VertexC']
        self.VertexC = VertexC
        R = G_base.graph['R']
        self.R = R
        T = G_base.graph['T']
        self.T = T
        self.fnT = G_base.graph.get('fnT')
        pos = dict(zip(range(T), VertexC[:T]))
        pos |= dict(zip(range(-R, 0), VertexC[-R:]))
        D = G_base.graph.get('D')
        if D is not None:
            T -= D
        self.pos = pos

        G = nx.Graph(name=G_base.name, R=R, T=T, VertexC=VertexC)
        G.add_nodes_from(G_base.nodes(data=True))

        # make star graph
        if self.start_from_star:
            d2roots = G_base.graph.get('d2roots')
            if d2roots is None:
                d2roots = cdist(VertexC[:T], VertexC[-R:])
            for n in range(T):
                # root = G_base.nodes[n]['root']
                root = G.nodes[n]['root']
                G.add_edge(root, n, length=d2roots[n, root])

        # for node in G.nbunch_iter(range(T, T + D)):
        # G[node]['color'] = 'none'
        self.G = G

        subtrees = G.nodes(data='subtree', default=-1)
        self.node_colors = np.array(
            [self.colors[subtrees[n] % len(self.colors)] for n in range(T)]
        )
        Subtree = defaultdict(list)
        for node, subtreeI in G.nodes(data='subtree'):
            if subtreeI is None or node >= T:
                continue
            Subtree[subtreeI].append(node)
        self.Subtree = Subtree
        self.uncolored = set(Subtree.keys())
        self.DetourNodeA = {}

        if ax is None:
            # limX, limY = 1920/dpi, 1080/dpi
            limX, limY = 1440 / dpi, 900 / dpi
            lR = limX / limY
            boundary = G_base.graph['boundary']
            XYrange = np.abs(np.amax(boundary, axis=0) - np.amin(boundary, axis=0))
            ratio = XYrange[0] / XYrange[1]
            figsize = (limX, limX / ratio) if ratio > lR else (limY * ratio, limY)
            self.fig = plt.figure(figsize=figsize)
            # self.fig.facecolor = '#1b1c17'
        self.init_plt()

    def init_plt(self):
        G = self.G
        R = self.R
        pos = self.pos
        # ax = self.ax
        ax = self.fig.add_subplot(aspect='equal')
        ax.axis('off')
        # ax.facecolor = '#1b1c17'
        self.ax = ax

        redraw = []
        # draw farm boundary
        # area_polygon = Polygon(self.G_base.graph['boundary'],
        #                        color='#111111', zorder=0)
        area_polygon = Polygon(self.G_base.graph['boundary'], color='black', zorder=0)
        self.boundaryA = ax.add_patch(area_polygon)
        redraw.append(self.boundaryA)
        ax.update_datalim(area_polygon.get_xy())
        ax.autoscale()
        ax.set_aspect('equal')

        # draw root nodes
        roots = range(-R, 0)
        RootL = {r: G.nodes[r]['label'] for r in roots[::-1]}
        redraw.append(
            nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                nodelist=roots,
                node_color=self.root_color,
                node_shape='s',
                node_size=self.root_node_size,
            )
        )

        # draw regular nodes, one subtree at a time
        Subtree = self.Subtree
        SubtreeA = np.empty((len(Subtree)), dtype=object)
        for subtreeI, nodes in Subtree.items():
            SubtreeA[subtreeI] = nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                nodelist=Subtree[subtreeI],
                node_color=[self.colors[-1]],
                node_size=self.node_size,
            )
            redraw.append(SubtreeA[subtreeI])
        self.SubtreeA = SubtreeA

        # draw labels
        font_size = {'load': FONTSIZE_LOAD, 'label': FONTSIZE_LABEL}
        node_tag = self.node_tag
        if node_tag is not None:
            if node_tag == 'load' and 'has_loads' not in G.graph:
                node_tag = 'label'
            labels = nx.get_node_attributes(G, node_tag)
            for root in roots:
                if root in labels:
                    labels.pop(root)
            self.labelsA = nx.draw_networkx_labels(
                G, pos, ax=ax, font_size=font_size[node_tag], labels=labels
            )
            redraw.extend(self.labelsA.values())
        # root nodes' labels
        redraw.extend(
            nx.draw_networkx_labels(
                G, pos, ax=ax, labels=RootL, font_size=FONTSIZE_ROOT_LABEL
            ).values()
        )
        redraw.append(self.draw_edges())

        # create text element for iteration number
        self.iteration = ax.text(
            0.01,
            0.99,
            'i =   0',
            ha='left',
            va='top',
            transform=ax.transAxes,
            fontname='DejaVu Sans Mono',
            color='white',
        )
        # transform=ax.transAxes, fontname='Inconsolata')
        # transform=ax.transAxes, fontname='Iosevka')
        redraw.append(self.iteration)
        # create text element for total length
        self.length = ax.text(
            0.99,
            0.99,
            f'{G.size(weight="length"):.0f} m',
            ha='right',
            va='top',
            transform=ax.transAxes,
            fontname='DejaVu Sans Mono',
            color='white',
        )
        # fontname='Inconsolata')
        # fontname='Iosevka')
        redraw.append(self.length)
        return redraw

    def draw_edges(self):
        G = self.G
        edge_colors = [
            color for u, v, color in G.edges(data='color', default=self.edge_color)
        ]
        edge_style = [style for s, t, style in G.edges(data='style', default='solid')]
        edgesA = nx.draw_networkx_edges(
            G, self.pos, ax=self.ax, edge_color=edge_colors, style=edge_style
        )
        self.prevEdgesA = edgesA
        return edgesA

    def update(self, step):
        redraw = []
        #  n2s = NodeStr(self.fnT, self.T)
        detourprop = dict(style='dashed', color='yellow')
        G = self.G
        pos = self.pos
        VertexC = self.VertexC
        # TODO: clear highlighting from previous iteration

        i, operations = step
        self.iteration.set_text('i = ' + f'{i:d}'.rjust(3, ' '))
        redraw.append(self.iteration)
        for oper, args in operations:
            if oper == 'addE':
                # if args in self.G_base.edges:
                #     length = self.G_base[args]['length']
                # else:
                #     u, v = args
                #     length = np.hypot(*(VertexC[u] - VertexC[v]).T)
                G.add_edge(*args, length=self.G_base.edges[args]['length'])
            elif oper == 'remE':
                G.remove_edge(*args)
            elif oper == 'addDE':
                s, t, s_, t_ = args
                length = np.hypot(*(VertexC[s_] - VertexC[t_]).T)
                G.add_edge(s, t, **detourprop, length=length)
            elif oper == 'addDN':
                t_, new = args
                G.add_node(new)
                pos[new] = VertexC[t_]
                self.DetourNodeA[new] = nx.draw_networkx_nodes(
                    G,
                    pos,
                    ax=self.ax,
                    nodelist=[new],
                    alpha=0.4,
                    edgecolors='orange',
                    node_color='none',
                    node_size=NODESIZE_LABELED_DETOUR,
                )
                redraw.append(self.DetourNodeA[new])
            elif oper == 'movDN':
                hook, corner, hook_, corner_ = args
                root = self.G_base.nodes[hook]['root']
                pos[corner] = VertexC[corner_]
                if corner > 0:
                    self.DetourNodeA[corner].remove()
                    redraw.append(self.DetourNodeA[corner])
                    self.DetourNodeA[corner] = nx.draw_networkx_nodes(
                        G,
                        pos,
                        ax=self.ax,
                        nodelist=[corner],
                        alpha=0.4,
                        edgecolors='orange',
                        node_color='none',
                        node_size=NODESIZE_LABELED_DETOUR,
                    )
                    redraw.append(self.DetourNodeA[corner])
                hook2cornerL = np.hypot(*(VertexC[hook_] - VertexC[corner_]).T)
                # corner2rootL = self.G_base.graph['d2roots'][t_, root]
                G.edges[hook, corner]['length'] = hook2cornerL
                # G.edges[blocked, root]['length'] = corner2rootL
            elif oper == 'finalG':
                # TODO: this coloring of the subtree is misleading for the
                # cases where the gate is marked as final before the full
                # capacity is reached, causing nodes still not part of the
                # final subtree to receive its color too early
                gate, root = args
                subtreeI = G.nodes[gate]['subtree']
                self.SubtreeA[subtreeI].remove()
                nodes = self.Subtree[subtreeI]
                self.SubtreeA[subtreeI] = nx.draw_networkx_nodes(
                    G,
                    pos,
                    ax=self.ax,
                    node_size=self.node_size,
                    nodelist=nodes,
                    node_color=self.node_colors[nodes],
                )
                redraw.append(self.SubtreeA[subtreeI])
                # print(i, self.uncolored, f' - {{{subtreeI}}}')
                # self.uncolored.remove(subtreeI)
                self.uncolored.discard(subtreeI)
            elif oper == 'remN':
                G.remove_node(args)
                self.DetourNodeA[args].remove()
                redraw.append(self.DetourNodeA[args])
                del self.DetourNodeA[args]
            elif oper == 'end':
                # color the remaining subtrees
                for subtreeI in self.uncolored:
                    self.SubtreeA[subtreeI].remove()
                    nodes = self.Subtree[subtreeI]
                    self.SubtreeA[subtreeI] = nx.draw_networkx_nodes(
                        G,
                        pos,
                        ax=self.ax,
                        node_size=self.node_size,
                        nodelist=nodes,
                        node_color=self.node_colors[nodes],
                    )
                    redraw.append(self.SubtreeA[subtreeI])
                # make text bold to mark the last frame
                self.length.set_color('yellow')
                self.iteration.set_color('yellow')
            elif oper == 'nop':
                pass
            else:
                print('Unknown operation:', oper)

        self.length.set_text(f'{G.size(weight="length"):.0f} m')
        redraw.append(self.length)
        self.prevEdgesA.remove()
        self.prevEdgesA.set_visible(False)
        redraw.append(self.prevEdgesA)
        redraw.append(self.draw_edges())
        return redraw


def animate(
    G,
    interval=250,
    blit=True,
    workpath='./tmp/',
    node_tag='label',
    savepath='./videos/',
    remove_apng=True,
    use_apng2gif=False,
):
    # old_dpi = plt.rcParams['figure.dpi']
    # dpi = plt.rcParams['figure.dpi'] = 192
    # layplt = LayoutPlotter(G, dpi=dpi)
    layplt = LayoutPlotter(G, dpi=192, node_tag=node_tag)
    #  savefig_kwargs = {'facecolor': '#1b1c17'}
    savefig_kwargs = {'facecolor': '#000000'}
    anim = animation.FuncAnimation(
        layplt.fig, layplt.update, frames=layplt.log, interval=interval, blit=blit
    )
    if use_apng2gif:
        print('apng2gif is disabled in the source code.')
        #  fname = f'{G.name}_{G.graph["creator"]}_' \
        #          f'{G.graph["capacity"]}.apng'
        # from numpngw import AnimatedPNGWriter
        # writer = AnimatedPNGWriter(fps=1000/interval)
        #  anim.save(workpath + fname, writer=writer,
        #            savefig_kwargs=savefig_kwargs)
        #  subprocess.run(['apng2gif', workpath + fname, savepath + fname[:-4]
        #                    + 'gif'])
        #  if remove_apng:
        #      os.remove(workpath + fname)
    else:
        fname = f'{G.name}_{G.graph["creator"]}_{G.graph["capacity"]}.gif'
        writer = animation.PillowWriter(fps=round(1000 / interval))
        anim.save(savepath + fname, writer=writer, savefig_kwargs=savefig_kwargs)
        from pygifsicle import gifsicle

        gifsicle(sources=savepath + fname, options=['--optimize=3'])
    # plt.rcParams['figure.dpi'] = old_dpi
    return fname
