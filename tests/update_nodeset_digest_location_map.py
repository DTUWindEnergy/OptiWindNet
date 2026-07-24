"""Regenerate the bundled-node-set digest to location-name mapping.

Run from the repository root with::

    python -m tests.update_nodeset_digest_location_map

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

from .nodeset_digest_aliases import NODESET_DIGEST_ALIASES
from .sitecache import (
    NODESET_DIGEST_LOCATION_MAP_FILE,
    bundled_location_path,
)


def _add_digest(
    mapping: dict[bytes, str],
    digest: bytes,
    *,
    name: str,
) -> None:
    previous = mapping.setdefault(digest, name)
    if previous != name:
        raise ValueError(
            f'nodeset digest collision between {previous!r} and {name!r}: '
            f'{digest.hex()}'
        )


def _add_location(
    mapping: dict[bytes, str],
    L: nx.Graph,
    *,
    name: str,
) -> None:
    _add_digest(
        mapping,
        fingerprint_coordinates(L.graph['VertexC'])[0],
        name=name,
    )


def generate() -> dict[bytes, str]:
    """Map bundled node sets and verified base-site aliases to filename stems."""
    mapping = {}
    for L in load_repository():
        name = L.graph['name']
        bundled_location_path(name)
        _add_location(mapping, L, name=name)
        if L.graph['R'] > 1:
            _add_location(mapping, as_single_root(L), name=name)
    for alias, canonical in NODESET_DIGEST_ALIASES.items():
        try:
            name = mapping[canonical]
        except KeyError as exc:
            raise ValueError(
                f'alias target is not a canonical bundled digest: {canonical.hex()}'
            ) from exc
        _add_digest(mapping, alias, name=name)
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
