"""Generate the SVG figures embedded in the prose pages of the documentation.

The notebooks under ``docs/notebooks`` carry their own figures as committed cell
outputs (``nbsphinx_execute = 'never'``), but the Markdown pages have no such
mechanism. This script is it: it writes standalone SVG files into ``_static``,
which are committed alongside the pages that embed them.

Run it after any change that alters what the figures show::

    make -C docs figures PYTHON=/path/to/.venv/bin/python

Each figure is written twice, ``_light`` and ``_dark``, for furo's
``only-light``/``only-dark`` image classes. A figure is solved once and drawn
once per theme, so the MILP runs behind `topologies` are not paid for twice.

Panels are composed by nesting each ``svgplot()`` output in a fixed-size cell.
A nested ``<svg>`` without width/height fills its cell, and its default
``preserveAspectRatio="xMidYMid meet"`` fits and centres the drawing there, so
the panel's own dimensions never have to be measured: ``svgplot()`` shortens the
viewBox height for a wide-aspect site, and this absorbs that without noticing.

Every figure asserts what it is meant to show. A location and capacity that
produce no detour, or no contour, would yield panels that quietly illustrate
nothing, so the builders raise instead of writing such a figure.
"""

# ruff: noqa: E402, RUF100 -- MPLCONFIGDIR precedes Matplotlib intentionally.

import argparse
import functools
import io
import os
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Set `MPLCONFIGDIR` before importing Matplotlib, including indirectly through
# `optiwindnet.api`, to keep its cache inside the repository.
os.environ.setdefault('MPLCONFIGDIR', str(HERE.parent / '.mplconfig'))

import matplotlib.pyplot as plt
import networkx as nx

from docs.figuredata import routers_problem
from docs.svg_utils import prettify_svg
from optiwindnet.api import MILPRouter, ModelOptions, WindFarmNetwork
from optiwindnet.baselines.hgs import hgs_cvrp
from optiwindnet.importer import load_repository
from optiwindnet.interarraylib import G_from_S, as_normalized
from optiwindnet.mesh import make_planar_embedding
from optiwindnet.pathfinding import PathFinder
from optiwindnet.svg import Drawable
from optiwindnet.synthetic import toyfarm

STATIC = HERE / '_static'

#: Theme-specific colors. `series` contains four categorical colors, `ink` is
#: used for text, and `backdrop` is used for marker outlines.
THEME = {
    'light': {
        'series': ('#2a78d6', '#eb6834', '#1baf7a', '#8a5cd0'),
        'ink': '#0b0b0b',
        'muted': '#52514e',
        'grid': '#d8d7d2',
        'backdrop': '#ffffff',
        'caption': '#5a5a5a',
    },
    'dark': {
        'series': ('#3987e5', '#d95926', '#199e70', '#a37ce8'),
        'ink': '#ffffff',
        'muted': '#c3c2b7',
        'grid': '#3a3a38',
        'backdrop': '#131416',
        'caption': '#a3a3a3',
    },
}

#: Use the same fixed hash salt as `run_notebooks.py` to produce stable Matplotlib
#: SVG element IDs.
SVG_HASHSALT = 'fixed-salt-for-this-project'

# Captions are sized as a fraction of the cell so that they read the same
# whether a figure's panels are wide or narrow. At this ratio a caption has
# room for roughly 38 characters before it runs past its cell.
LABEL_RATIO = 20

Panels = tuple[int, list[tuple[str, nx.Graph]]]
#: Callable that renders SVG markup for a theme.
Builder = Callable[[str], str]


def draw(G: nx.Graph, dark: bool) -> tuple[str, int, int]:
    """Draw one graph, cropped to the width its drawing actually occupies.

    ``svgplot()`` always emits a 1920-wide viewBox, so a site taller than 16:9
    is left-aligned in it with the remainder blank — the toy farm behind
    `topologies` fills 45% of it. ``Drawable`` is the same drawing code with the
    viewport still open to adjustment, so the panel comes out already cropped
    and no markup has to be rewritten afterwards.

    This repeats the element order of ``svgplot()`` in ``optiwindnet/svg.py``,
    minus the infobox and the border tags, and has to stay in step with it.

    Returns the markup and the size it was cropped to, which is what lets
    `compose` size its cells to the drawing rather than to svgplot's canvas.
    """
    drawable = Drawable(G, landscape=True, dark=dark, transparent=True)
    drawable.add_edges()
    if G.graph.get('D', False):
        drawable.add_detours()
    drawable.add_nodes()
    used_w = drawable.bottom_right_anchor['x'] + drawable.margin
    drawable.viewBox = type(drawable.viewBox)(0, 0, used_w, drawable.h)
    return prettify_svg(drawable.to_svg()), used_w, drawable.h


def compose(panels: list[tuple[str, str, int, int]], columns: int, theme: str) -> str:
    """Lay out ``(label, markup, width, height)`` panels on a grid, captions above.

    Cells take the size of the largest panel, so a panel is only ever centred
    against its neighbours rather than against a fixed canvas.
    """
    rows = -(-len(panels) // columns)
    cell_w = max(panel_w for _, _, panel_w, _ in panels)
    cell_h = max(panel_h for _, _, _, panel_h in panels)
    label_size = round(cell_w / LABEL_RATIO)
    label_h = round(label_size * 1.55)
    width, height = columns * cell_w, rows * (cell_h + label_h)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        (
            f' <style>.caption{{font:{label_size}px sans-serif;'
            f'fill:{THEME[theme]["caption"]}}}</style>'
        ),
    ]
    for index, (label, panel, _, _) in enumerate(panels):
        x = (index % columns) * cell_w
        y = (index // columns) * (cell_h + label_h)
        parts.extend(
            (
                (
                    f' <text class="caption" x="{x + label_size // 2}" '
                    f'y="{y + label_size}">{label}</text>'
                ),
                f' <svg x="{x}" y="{y + label_h}" width="{cell_w}" height="{cell_h}">',
            )
        )
        parts.extend(f'  {line}' for line in panel.rstrip().splitlines())
        parts.append(' </svg>')
    parts.append('</svg>')
    return '\n'.join(parts) + '\n'


def _solve(
    location: str, capacity: int, time_limit: float
) -> tuple[nx.Graph, nx.Graph, nx.Graph]:
    """Take one repository location through the pipeline, returning `A`, `S` and `G`.

    `S` comes back as the physical graph `G_from_S()` builds from it, so that it
    can be drawn on the site; PathFinder gets its own copy to work on.
    """
    L = getattr(load_repository(), location)
    P, A = make_planar_embedding(L)
    S = hgs_cvrp(as_normalized(A), capacity=capacity, time_limit=time_limit)
    straight = G_from_S(S, A)
    detoured = PathFinder(G_from_S(S, A), planar=P, A=A).create_detours()
    return A, straight, detoured


def graph_model(location: str = 'kfA', capacity: int = 5) -> Panels:
    """The `L` -> `A` -> `S` -> `G` progression, for `problem.md`'s graph model."""
    L = getattr(load_repository(), location)
    A, straight, detoured = _solve(location, capacity, time_limit=0.5)
    if not detoured.graph.get('D', 0):
        raise SystemExit(
            f'{location} at capacity {capacity} yields no detour, so panels S and G '
            'would be identical — pick another location or capacity.'
        )
    return 2, [
        ('L — location', L),
        ('A — available links', A),
        ('S — solution topology', straight),
        ('G — routeset', detoured),
    ]


def crossings(location: str = 'cazzaro_2022', capacity: int = 5) -> Panels:
    """Straight feeders against path-found ones, for `problem.md`'s crossings section.

    What differs between the panels is the feeders: `G_from_S()` already lays the
    non-feeder links around obstacles, so both panels carry the same contours and
    only the second has detours. The location still needs obstacles, for the
    contours to be there to see at all; only the cazzaro_2022 sites have any.
    """
    _, straight, detoured = _solve(location, capacity, time_limit=2.0)
    contours, detours = detoured.graph.get('C', 0), detoured.graph.get('D', 0)
    if not contours or not detours:
        raise SystemExit(
            f'{location} at capacity {capacity} yields {contours} contour(s) and '
            f'{detours} detour(s); the figure needs both — pick another location '
            'or capacity.'
        )
    return 2, [
        ('S — straight feeders', straight),
        ('G — feeders after path-finding', detoured),
    ]


def topologies(capacity: int = 4, time_limit: int = 5) -> Panels:
    """The three network topologies side by side, for `problem.md`.

    The same toy farm and MILP settings as the figure in `hi00_quickstart`.
    """
    panels = []
    for topology in ('branched', 'radial', 'ringed'):
        wfn = WindFarmNetwork(L=toyfarm(), cables=capacity)
        wfn.optimize(
            router=MILPRouter(
                solver_name='ortools.cp_sat',
                time_limit=time_limit,
                mip_gap=0.001,
                model_options=ModelOptions(topology=topology, feeder_limit='minimum'),
            )
        )
        panels.append((topology.capitalize(), wfn.G))
    return 3, panels


def fig_to_svg(fig) -> str:
    """Serialize a Matplotlib figure as SVG markup without the XML preamble.

    Remove the preamble and DOCTYPE for embedding, and close the figure after
    serialization.

    Use a fixed hash salt and omit date metadata to make the SVG output byte-stable,
    consistent with `run_notebooks.py`.
    """
    # Preserve SVG text so page fonts apply and labels remain selectable and
    # searchable.
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['svg.hashsalt'] = SVG_HASHSALT
    buffer = io.StringIO()
    fig.savefig(
        buffer,
        format='svg',
        transparent=True,
        metadata={'Date': None, 'Creator': 'OptiWindNet'},
    )
    plt.close(fig)
    markup = buffer.getvalue()
    # Replace the machine-specific font name with `sans-serif`; the browser resolves
    # the family because `svg.fonttype` is `none`.
    markup = markup.replace(f"'{plt.rcParams['font.sans-serif'][0]}'", 'sans-serif')
    markup = markup[markup.index('<svg') :]
    return '\n'.join(line.rstrip() for line in markup.splitlines()) + '\n'


@functools.cache
def routers_data() -> tuple[dict, dict]:
    """Load and cache the routers and NumPy time-series artifacts."""
    data = routers_problem.load_summary()
    timeseries = routers_problem.load_timeseries()
    return data, timeseries


def routers(theme: str) -> str:
    """Plot percentage above the proven optimum versus time for `routers.md`.

    Load the committed artifacts generated by the scripts in `figuredata/`; the
    MILP is not run during figure generation.

    Plot topology lengths to match the MILP objective. Offset MILP timestamps by
    the warm-start duration.

    Every series is keyed to an optimization approach: the two constructive
    heuristics, the four meta-heuristic budgets and the exact solver's incumbent
    each get their own color and marker, so the legend names approaches rather
    than data files.

    Read the problem description from its definition.
    """
    data, timeseries = routers_data()
    observations = data['observations']
    optimum = data['optimum']
    color = THEME[theme]
    ink, muted, grid = color['ink'], color['muted'], color['grid']
    exact_c, bound_c, heuristic_c, meta_c = color['series']
    backdrop = color['backdrop']

    def excess(length: float) -> float:
        """Return a length as a percentage above the proven optimum."""
        return (length / optimum - 1) * 100

    offset, _length = observations[routers_problem.MILP_TIMESERIES_WARMSTART]
    bound_seconds = [t + offset for t in timeseries['bound_time']]
    bound_excess = [excess(length) for length in timeseries['bound']]
    announced = [
        (t + offset, excess(length))
        for t, length in zip(timeseries['incumbent_time'], timeseries['incumbent'])
    ]
    # The solver re-announces an unchanged incumbent — down to the last bits of
    # the objective — so keep only the entries that actually improved it. The
    # series is non-increasing by construction, so dropping a repeated value
    # leaves the staircase unchanged while giving every drawn point a marker.
    updates = [
        point
        for i, point in enumerate(announced)
        if i == 0 or point[1] < announced[i - 1][1] - 1e-6
    ]
    # The last step is carried to the end of the run so that both lines span the
    # same time range; `markevery` below leaves that added point unmarked.
    updates.append((max(bound_seconds[-1], updates[-1][0]), updates[-1][1]))

    # Split the sampled runs by approach: a key carrying a time budget is a
    # meta-heuristic run, one without is a constructive-heuristic variant.
    heuristic = {k: v for k, v in observations.items() if ':' not in k}
    meta = {k: v for k, v in observations.items() if ':' in k}

    # Point sizes for the annotations and for the axis labels. The page scales
    # the SVG to its own width, so these set how large the text reads there
    # relative to the figure, not its size on any particular screen.
    note_size, axis_size = 10, 11

    fig, ax = plt.subplots(figsize=(7.2, 4.0), layout='constrained')
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    ax.axhline(0.0, color=muted, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.annotate(
        'proven optimum',
        xy=(0.005, 0.0),
        xycoords=('axes fraction', 'data'),
        ha='left',
        va='bottom',
        fontsize=note_size,
        color=muted,
    )
    # Draw one artist per approach, in legend order: `ax.legend()` collects the
    # labelled artists in the order they were added, and the explicit `zorder`
    # keeps that order from deciding what is drawn on top.
    for runs, marker, face, label in (
        (heuristic, 'o', heuristic_c, 'Heuristic'),
        (meta, 's', meta_c, 'Meta-heuristic'),
    ):
        ax.plot(
            [elapsed for elapsed, _ in runs.values()],
            [excess(length) for _, length in runs.values()],
            marker=marker,
            ms=8,
            mfc=face,
            mec=backdrop,
            mew=1.5,
            ls='none',
            label=label,
            zorder=4,
        )
    ax.step(
        [t for t, _ in updates],
        [y for _, y in updates],
        where='post',
        lw=2,
        color=exact_c,
        # `markevery` indexes the data points, so this marks every incumbent
        # update and skips the point carrying the last step to the end.
        marker='D',
        markevery=slice(0, -1),
        ms=6.5,
        mfc=exact_c,
        mec=backdrop,
        mew=1.2,
        label='Exact',
        zorder=3,
    )
    ax.step(
        bound_seconds,
        bound_excess,
        where='post',
        lw=1.3,
        color=bound_c,
        label='Exact: bound',
        zorder=2,
    )

    # Heuristic points sit close together, so only the topmost one is labelled
    # above; a lower label would otherwise run into the marker above it.
    highest = max(heuristic, key=lambda key: heuristic[key][1])
    for key, (elapsed, length) in observations.items():
        family, _, budget = key.partition(':')
        # The legend names the approach, so a point only has to carry what tells
        # it apart from its siblings: the variant, or the time budget.
        above = key == highest
        ax.annotate(
            f'{budget} s' if budget else family,
            xy=(elapsed, excess(length)),
            xytext=(0, 8) if above else (0, -11),
            textcoords='offset points',
            ha='center',
            va='bottom' if above else 'top',
            fontsize=note_size,
            color=ink,
        )

    # Build the figure title from the problem definition.
    L = routers_problem.load_location()
    ax.set_title(
        f'{L.name} · {L.graph["T"]} turbines · capacity {routers_problem.CAPACITY}'
        f' · {routers_problem.MODEL_OPTIONS["topology"]}',
        color=muted,
        fontsize=note_size,
        loc='left',
        pad=8,
    )

    ax.set_xscale('log')
    ax.set_xlabel('time (s)', color=ink, fontsize=axis_size)
    ax.set_ylabel('length above the optimum (%)', color=ink, fontsize=axis_size)
    ax.tick_params(colors=muted, labelsize=note_size, which='both')
    ax.grid(True, which='major', color=grid, lw=0.6, zorder=0)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ('left', 'bottom'))
        spine.set_color(grid)
    # Exclude the initial LP relaxation from y-axis scaling because it would
    # compress the remaining data.
    settled = [e for e in bound_excess if e > -5]
    top = max(excess(length) for _, length in observations.values())
    ax.set_ylim(min(settled) - 0.8, top + 1.6)
    legend = ax.legend(
        frameon=False, fontsize=note_size, loc='upper right', labelcolor=ink
    )
    legend.set_zorder(5)
    return fig_to_svg(fig)


def panelled(builder) -> Builder:
    """Adapt a panel builder to the `Builder` interface.

    Cache generated panels so both themes reuse the same solver run.
    """
    solve = functools.cache(builder)

    def render(theme: str) -> str:
        columns, panels = solve()
        return compose(
            [(label, *draw(G, dark=theme == 'dark')) for label, G in panels],
            columns=columns,
            theme=theme,
        )

    return render


#: Map figure names to theme-aware SVG builders. `panelled()` adapts graph-panel
#: builders to this interface.
BUILDERS: dict[str, Builder] = {
    'graph_model': panelled(graph_model),
    'crossings': panelled(crossings),
    'topologies': panelled(topologies),
    'routers': routers,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        'names',
        nargs='*',
        choices=[*BUILDERS, []],
        help=f'figures to build (default: all of {", ".join(BUILDERS)})',
    )
    parser.add_argument(
        '--outdir', type=Path, default=STATIC, help='where to write the SVG files'
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    for name in args.names or BUILDERS:
        for theme in ('light', 'dark'):
            markup = BUILDERS[name](theme)
            target = args.outdir / f'fig_{name}_{theme}.svg'
            target.write_text(markup, encoding='utf8')
            print(f'wrote {target} ({target.stat().st_size} bytes)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
