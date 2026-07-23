"""Regenerate the bundled-node-set digest to location-name mapping.

Run from the repository root with::

    NUMBA_CACHE_DIR="$PWD/.numba_cache" \
        .venv/bin/python -m tests.update_nodeset_digest_location_map

Both the original and, where applicable, single-root coordinate sets map to
the base location name. A location name is its bundled data filename stem.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import networkx as nx

from optiwindnet.fingerprint import fingerprint_coordinates
from optiwindnet.importer import load_repository
from optiwindnet.interarraylib import as_single_root

from .sitecache import (
    NODESET_DIGEST_LOCATION_MAP_FILE,
    bundled_location_path,
)


def _add_location(
    mapping: dict[bytes, str],
    L: nx.Graph,
    *,
    name: str,
) -> None:
    digest = fingerprint_coordinates(L.graph['VertexC'])[0]
    previous = mapping.setdefault(digest, name)
    if previous != name:
        raise ValueError(
            f'nodeset digest collision between {previous!r} and {name!r}: '
            f'{digest.hex()}'
        )


def generate() -> dict[bytes, str]:
    """Map every bundled base and single-root node set to its filename stem."""
    mapping = {}
    for L in load_repository():
        name = L.graph['name']
        bundled_location_path(name)
        _add_location(mapping, L, name=name)
        if L.graph['R'] > 1:
            _add_location(mapping, as_single_root(L), name=name)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output',
        type=Path,
        default=NODESET_DIGEST_LOCATION_MAP_FILE,
    )
    args = parser.parse_args()
    mapping = generate()
    with args.output.open('wb') as file:
        pickle.dump(mapping, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved {len(mapping)} node-set mappings to {args.output}')


if __name__ == '__main__':
    main()
