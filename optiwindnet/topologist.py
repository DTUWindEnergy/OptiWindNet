"""Draw a topology over a location graph.

Run from the repository root::

    MPLCONFIGDIR="$PWD/.mplconfig" NUMBA_CACHE_DIR="$PWD/.numba_cache" \
        .venv/bin/python -m optiwindnet.topologist [PATH]

PATH may be a directory of locations, an .osm.pbf file, or an OWN .yaml file.
Without PATH, the bundled locations are used.

Left-click a node to start a chain, then left-click more nodes to add links.
Right-click to end the chain. A drawing that matches the selected topology
enables the button that copies its TerseLinks constructor.
"""

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import ttk

import darkdetect
import networkx as nx
import numpy as np
from matplotlib import rcParams
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from optiwindnet.geometric import rotate
from optiwindnet.importer import L_from_pbf, L_from_yaml, load_repository
from optiwindnet.plotting import gplot
from optiwindnet.terse import LinkScope, TerseLinks
from optiwindnet.types import Topology


def load_locations(path: Path | None = None) -> dict[str, nx.Graph[int]]:
    """Load bundled locations, a directory, or one location file."""
    if path is not None and not path.exists():
        raise FileNotFoundError(f'Path does not exist: {path}')
    if path is None or path.is_dir():
        repository = load_repository(path)
        locations = dict(zip(repository._fields, repository))
    elif path.is_file() and path.name.endswith('.osm.pbf'):
        location = L_from_pbf(path)
        locations = {location.graph['handle']: location}
    elif path.is_file() and path.suffix == '.yaml':
        location = L_from_yaml(path)
        locations = {location.graph['handle']: location}
    else:
        raise ValueError(f'Expected a directory, .osm.pbf file, or .yaml file: {path}')
    if not locations:
        raise ValueError(f'No location files found in {path}')
    return locations


def ring_links(graph: nx.Graph[int]) -> tuple[int, ...] | None:
    """Encode disjoint root-to-root walks, including rings returning to one root."""
    T, R = graph.graph['T'], graph.graph['R']
    links: list[int] = []
    seen_terminals: set[int] = set()
    seen_feeders: set[tuple[int, int]] = set()
    for root in range(-R, 0):
        for first in sorted(graph[root]):
            if (root, first) in seen_feeders:
                continue
            if first < 0:
                return None
            walk = []
            previous, current = root, first
            while current >= 0:
                if current in walk or current in seen_terminals:
                    return None
                walk.append(current)
                onward = [node for node in graph[current] if node != previous]
                if len(onward) > 1:
                    return None
                if not onward:
                    current = root
                    break
                previous, current = current, onward[0]
            seen_terminals.update(walk)
            seen_feeders.add((root, first))
            seen_feeders.add((current, walk[-1]))
            links.extend((root, *walk, current))
    return tuple(links) if len(seen_terminals) == T else None


def creation_line(graph: nx.Graph[int], topology: Topology) -> str | None:
    """Return a constructor when the drawn links match the selected topology."""
    T, R = graph.graph['T'], graph.graph['R']
    if topology is Topology.RINGED:
        links = ring_links(graph)
        if links is None:
            return None
        try:
            encoding = TerseLinks(
                links=links, topology=topology, scope=LinkScope.TOPOLOGY, T=T, R=R
            )
            restored = encoding.to_topology()
        except ValueError:
            return None
        if {frozenset(edge) for edge in restored.edges} != {
            frozenset(edge) for edge in graph.edges
        }:
            return None
    else:
        if not nx.is_forest(graph):
            return None
        if topology is Topology.RADIAL and any(graph.degree(t) > 2 for t in range(T)):
            return None
        roots = set(range(-R, 0))
        parents = [None] * T
        for component in nx.connected_components(graph):
            component_roots = component & roots
            if len(component_roots) != 1:
                return None
            for parent, child in nx.bfs_edges(graph, next(iter(component_roots))):
                if child in range(T):
                    parents[child] = parent
        if any(parent is None for parent in parents):
            return None
        encoding = TerseLinks(
            links=tuple(parents),
            topology=topology,
            scope=LinkScope.TOPOLOGY,
            T=T,
            R=R,
        )
    return (
        f'TerseLinks(links={encoding.links!r}, topology={encoding.topology.value!r}, '
        f'scope={encoding.scope.value!r}, T={T}, R={R})'
    )


def link_error(
    graph: nx.Graph[int], source: int, target: int, topology: Topology
) -> str | None:
    """Reject links that cannot belong to the selected topology."""
    if topology is Topology.RINGED:
        if source < 0 and target < 0:
            return 'A ring route must contain a turbine.'
        if any(graph.degree(node) >= 2 for node in (source, target) if node >= 0):
            return 'Ringed turbines can have at most two links.'
        component = nx.node_connected_component(graph, source)
        if target in component and not component.intersection(
            range(-graph.graph['R'], 0)
        ):
            return 'A ring must touch a substation.'
    else:
        if topology is Topology.RADIAL and any(
            graph.degree(node) >= 2 for node in (source, target) if node >= 0
        ):
            return 'Radial turbines can have at most two links.'
        source_group = nx.node_connected_component(graph, source)
        target_group = nx.node_connected_component(graph, target)
        if source_group == target_group:
            return 'That link would make a cycle.'
        roots = set(range(-graph.graph['R'], 0))
        if len((source_group | target_group) & roots) > 1:
            return 'A component can contain only one substation.'
    return None


class Topologist:
    """Keep the location, drawing, and widget state together."""

    def __init__(
        self, root: tk.Tk, locations: dict[str, nx.Graph[int]] | None = None
    ) -> None:
        self.root = root
        self.root.title('OptiWindNet Topologist')
        self.locations = load_locations() if locations is None else locations
        self.graph: nx.Graph[int]
        self.cursor: int | None = None
        self.constructor: str | None = None
        self.nodes: list[int]
        self.coords: np.ndarray
        ui_font = tkfont.nametofont('TkDefaultFont')
        gap = max(1, ui_font.metrics('linespace') // 2)

        controls = ttk.Frame(root, padding=gap)
        controls.pack(fill='x')
        ttk.Label(controls, text='Location:').pack(side='left')
        self.selection = ttk.Combobox(
            controls,
            values=sorted(self.locations),
            state='readonly',
            width=max(map(len, self.locations)),
        )
        self.selection.pack(side='left', padx=max(1, gap // 2))
        self.selection.bind('<<ComboboxSelected>>', self.select_location)
        ttk.Label(controls, text='Topology:').pack(side='left')
        self.topology_selection = ttk.Combobox(
            controls,
            values=[topology.value for topology in Topology],
            state='readonly',
            width=max(len(topology.value) for topology in Topology),
        )
        self.topology_selection.pack(side='left', padx=max(1, gap // 2))
        self.topology_selection.set(Topology.BRANCHED.value)
        self.topology_selection.bind(
            '<<ComboboxSelected>>', lambda _event: self.redraw()
        )
        tag_group = ttk.Frame(
            controls, relief='groove', borderwidth=1, padding=max(1, gap // 4)
        )
        tag_group.pack(side='left', padx=(max(1, gap // 2), 0))
        self.tags_enabled = tk.BooleanVar(root, value=False)
        ttk.Checkbutton(
            tag_group,
            text='Node tags',
            variable=self.tags_enabled,
            command=self.change_tags,
        ).pack(side='left', padx=max(1, gap // 2))
        self.tag_kind = tk.StringVar(root, value='id')
        self.id_radio = ttk.Radiobutton(
            tag_group,
            text='id',
            variable=self.tag_kind,
            value='id',
            command=self.redraw,
        )
        self.id_radio.pack(side='left')
        self.label_radio = ttk.Radiobutton(
            tag_group,
            text='label',
            variable=self.tag_kind,
            value='label',
            command=self.redraw,
        )
        self.label_radio.pack(side='left', padx=(gap, 0))
        self.id_radio.state(['disabled'])
        self.label_radio.state(['disabled'])

        footer = ttk.Frame(root)
        footer.pack(side='bottom', fill='x')
        summary = ttk.Frame(footer, padding=(gap, max(1, gap // 2)))
        summary.pack(fill='x')
        ttk.Label(summary, text='Disconnected:').pack(side='left')
        self.disconnected_count = tk.StringVar(root, value='0')
        ttk.Entry(
            summary,
            textvariable=self.disconnected_count,
            width=len(str(max(site.graph['T'] for site in self.locations.values()))),
            state='readonly',
            justify='right',
        ).pack(side='left', padx=max(1, gap // 2))
        self.copy_button = ttk.Button(
            summary, text='Copy TerseLinks', command=self.copy_constructor
        )
        self.copy_button.pack(side='right')
        ttk.Button(summary, text='Reset', command=self.reset).pack(
            side='right', padx=(0, gap)
        )
        ttk.Separator(footer, orient='horizontal').pack(fill='x')
        self.status_font = ui_font.copy()
        status_size = self.status_font.cget('size')
        self.status_font.configure(
            size=max(1, status_size - 1)
            if status_size > 0
            else min(-1, status_size + 1)
        )
        self.status = ttk.Label(
            footer,
            padding=(gap, max(1, gap // 3)),
            font=self.status_font,
            anchor='w',
        )
        self.status.pack(fill='x')
        self.status_timer: str | None = None

        self.figure = Figure(
            figsize=(3, 2),
            dpi=max(192, rcParams['figure.dpi']),
            layout='constrained',
        )
        self.ax = self.figure.add_subplot()
        # Matplotlib changes the canvas's requested size after mapping on HiDPI.
        # Keep that request from feeding back into the window's initial size.
        figure_height = self.figure.get_figheight()
        plot_area = ttk.Frame(root, height=round(figure_height * self.figure.dpi))
        plot_area.pack(fill='both', expand=True)
        plot_area.pack_propagate(False)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_area)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.bind(
            '<Map>',
            lambda _event: plot_area.configure(
                height=round(figure_height * self.figure.dpi)
            ),
            add='+',
        )
        canvas_widget.pack(fill='both', expand=True)
        self.canvas.mpl_connect('button_press_event', self.click)

        self.style = ttk.Style(root)
        self.style.theme_use('clam')
        self.dark = darkdetect.isDark()
        self.apply_theme()
        self.root.after(2000, self.check_theme)
        self.selection.set(min(self.locations))
        self.select_location()

    def apply_theme(self) -> None:
        """Apply the detected system colors to Tk and the Matplotlib canvas."""
        background = '#202124' if self.dark else '#f3f3f3'
        field = '#303134' if self.dark else '#ffffff'
        foreground = '#f1f3f4' if self.dark else '#202124'
        active = '#45474a' if self.dark else '#e3e6ea'
        disabled_background = '#292a2c' if self.dark else '#e5e5e5'
        disabled_foreground = '#85898d' if self.dark else '#888888'
        self.plot_background = background
        self.root.configure(bg=background)
        self.style.configure('TFrame', background=background)
        self.style.configure('TLabel', background=background, foreground=foreground)
        for widget in ('TCheckbutton', 'TRadiobutton'):
            self.style.configure(
                widget,
                background=background,
                foreground=foreground,
            )
            self.style.map(
                widget,
                background=[('active', background)],
                foreground=[('disabled', disabled_foreground)],
            )
        self.style.configure(
            'TButton',
            background=field,
            foreground=foreground,
        )
        self.style.map(
            'TButton',
            background=[('disabled', disabled_background), ('active', active)],
            foreground=[('disabled', disabled_foreground)],
        )
        self.style.configure(
            'TEntry',
            fieldbackground=field,
            foreground=foreground,
        )
        self.style.map(
            'TEntry',
            fieldbackground=[('readonly', field)],
            foreground=[('readonly', foreground)],
        )
        self.style.configure(
            'TCombobox',
            fieldbackground=field,
            background=field,
            foreground=foreground,
            arrowcolor=foreground,
        )
        self.style.map(
            'TCombobox',
            fieldbackground=[('readonly', field)],
            foreground=[('readonly', foreground)],
            selectbackground=[('readonly', field)],
            selectforeground=[('readonly', foreground)],
        )
        self.root.option_add('*TCombobox*Listbox.background', field)
        self.root.option_add('*TCombobox*Listbox.foreground', foreground)
        self.figure.set_facecolor(background)

    def check_theme(self) -> None:
        """Follow system theme changes while the window is open."""
        detected = darkdetect.isDark()
        if detected != self.dark:
            self.dark = detected
            self.apply_theme()
            self.redraw()
        self.root.after(2000, self.check_theme)

    def select_location(self, _event=None) -> None:
        location = self.locations[self.selection.get()]
        self.graph = nx.create_empty_copy(location, with_data=True)
        self.cursor = None
        T, R = self.graph.graph['T'], self.graph.graph['R']
        self.nodes = [*range(T), *range(-R, 0)]
        vertices = self.graph.graph['VertexC']
        self.coords = np.vstack((vertices[:T], vertices[-R:]))
        angle = self.graph.graph.get('landscape_angle')
        if angle:
            self.coords = rotate(self.coords, angle)
        self.redraw()

    def reset(self) -> None:
        """Clear the drawn links and cursor for the current location."""
        self.graph.clear_edges()
        self.cursor = None
        self.redraw()

    def change_tags(self) -> None:
        """Enable the tag choice only while tags are shown."""
        state = ['!disabled'] if self.tags_enabled.get() else ['disabled']
        self.id_radio.state(state)
        self.label_radio.state(state)
        self.redraw()

    def copy_constructor(self) -> None:
        """Copy the current topology constructor to the system clipboard."""
        if self.constructor is None:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.constructor)
        self.show_status('TerseLinks copied to clipboard')

    def show_status(self, message: str) -> None:
        """Show a temporary message, then restore the mouse instruction."""
        if self.status_timer is not None:
            self.root.after_cancel(self.status_timer)
        self.status.config(text=message)
        self.status_timer = self.root.after(2500, self.restore_instruction)

    def restore_instruction(self) -> None:
        """Restore the instruction for the current cursor state."""
        self.status_timer = None
        self.status.config(text=self.instruction())

    def instruction(self) -> str:
        """Describe the mouse action available in the current cursor state."""
        if self.cursor is None:
            return 'Left-click: place cursor'
        return 'Left-click: connect · Right-click: clear cursor'

    def nearest_node(self, event) -> int | None:
        if event.inaxes is not self.ax:
            return None
        screen = self.ax.transData.transform(self.coords)
        distances = np.linalg.norm(screen - (event.x, event.y), axis=1)
        index = int(np.argmin(distances))
        pick_radius = self.figure.dpi * 8 / 72
        return self.nodes[index] if distances[index] <= pick_radius else None

    def click(self, event) -> None:
        if event.button == 3:
            self.cursor = None
            self.redraw()
            return
        if event.button != 1:
            return
        node = self.nearest_node(event)
        if node is None:
            return
        if self.cursor is None:
            self.cursor = node
        elif node != self.cursor:
            source = self.cursor
            if not self.graph.has_edge(source, node):
                error = link_error(
                    self.graph, source, node, Topology(self.topology_selection.get())
                )
                if error is not None:
                    self.show_status(error)
                    return
                self.graph.add_edge(source, node)
            self.cursor = node
        self.redraw()

    def redraw(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor(self.plot_background)
        node_tag = (
            (True if self.tag_kind.get() == 'id' else 'label')
            if self.tags_enabled.get()
            else None
        )
        gplot(self.graph, ax=self.ax, dark=self.dark, node_tag=node_tag)
        if self.cursor is not None:
            index = self.nodes.index(self.cursor)
            x, y = self.coords[index]
            self.ax.scatter(
                [x],
                [y],
                s=170,
                facecolors='none',
                edgecolors='tab:red',
                linewidths=1.5,
                zorder=10,
            )
        self.canvas.draw_idle()

        topology = Topology(self.topology_selection.get())
        self.constructor = creation_line(self.graph, topology)
        self.copy_button.state(
            ['!disabled'] if self.constructor is not None else ['disabled']
        )
        roots = range(-self.graph.graph['R'], 0)
        served = set().union(
            *(nx.node_connected_component(self.graph, root) for root in roots)
        )
        disconnected = len(set(range(self.graph.graph['T'])) - served)
        self.disconnected_count.set(str(disconnected))
        self.status.config(text=self.instruction())


def main(argv: list[str] | None = None) -> None:
    """Open Topologist with the requested locations."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        'path', nargs='?', type=Path, help='location directory, .osm.pbf, or OWN .yaml'
    )
    args = parser.parse_args(argv)
    try:
        locations = load_locations(args.path)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    root = tk.Tk()
    Topologist(root, locations)
    root.mainloop()


if __name__ == '__main__':
    main()
