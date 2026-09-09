# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Regenerate the expected available-links sets of the bundled locations.

Run from the repository root with::

    python -m tests.update_mesh_goldens

Every bundled location is meshed and its available-links graph is reduced to
the pair recorded in ``tests/mesh_goldens.py``: the number of canonical
terminal-terminal links and the hexadecimal ``linkset_id()`` of the whole
available-links set. The link count is redundant with the id, but it makes a
golden diff legible -- a reviewer sees how many candidate links the mesh
gained or lost, instead of an opaque id change.
"""

import argparse
import time
from pathlib import Path

from .sitecache import get_bundle, location_repository

MESH_GOLDENS_FILE = Path(__file__).with_name('mesh_goldens.py')

_TEMPLATE = '''"""Expected canonical available-links sets of the bundled locations.

Each entry maps a location handle to the number of canonical terminal-terminal
links in its mesh and the hexadecimal id of the whole available-links set (that
count plus the ``R * T`` feeders). A change here means
``make_planar_embedding()`` selected a different candidate link set: review it,
do not refresh it to make a test pass.

Regenerate with: python -m tests.update_mesh_goldens
"""

__all__ = ('MESH_LINKSET_GOLDENS',)

MESH_LINKSET_GOLDENS: dict[str, tuple[int, str]] = {{
{entries}}}
'''


def generate(*, verbose: bool = True) -> dict[str, tuple[int, str]]:
    """Mesh every bundled location and reduce it to its golden pair."""
    handles = location_repository()._fields  # pyrefly: ignore[missing-attribute]
    goldens = {}
    started = time.perf_counter()
    for count, handle in enumerate(handles, start=1):
        A = get_bundle(handle).A
        goldens[handle] = (
            len(A.graph['_canonical_terminal_links']),
            A.graph['_linkset_id'].hex(),
        )
        if verbose:
            elapsed = time.perf_counter() - started
            print(f'[{count}/{len(handles)}] {elapsed:7.1f}s {handle}', flush=True)
    distinct = {link_id for _, link_id in goldens.values()}
    if len(distinct) != len(goldens):
        raise ValueError('two bundled locations share a linkset id')
    return goldens


def render(goldens: dict[str, tuple[int, str]]) -> str:
    """Return the source of the golden module, one location per line."""
    entries = ''.join(
        f'    {handle!r}: ({links}, {link_id!r}),\n'
        for handle, (links, link_id) in goldens.items()
    )
    return _TEMPLATE.format(entries=entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=MESH_GOLDENS_FILE)
    args = parser.parse_args()
    goldens = generate()
    args.output.write_text(render(goldens))
    print(f'Saved {len(goldens)} available-links sets to {args.output}')


if __name__ == '__main__':
    main()
