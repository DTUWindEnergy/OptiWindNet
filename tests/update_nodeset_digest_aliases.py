"""Regenerate verified historical coordinate payloads for node-set aliases.

Run from the repository root with::

    python -m tests.update_nodeset_digest_aliases \
        docs/notebooks/optiwindnet-routesets-r26.05-v4.sqlite
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import networkx as nx
import numpy as np

from optiwindnet.db import NodeSet, database_connection
from optiwindnet.db.storage import L_from_nodeset, packnodes
from optiwindnet.fingerprint import fingerprint_coordinates
from optiwindnet.importer import load_repository
from optiwindnet.interarraylib import as_single_root
from optiwindnet.mesh import make_planar_embedding

from .nodeset_digest_aliases import (
    NODESET_DIGEST_ALIASES,
    NODESET_DIGEST_ALIAS_COORDINATES_FILE,
    NODESET_DIGEST_ALIAS_NAMES,
)


def _edge_set(graph: nx.Graph) -> set[tuple[int, int]]:
    return {tuple(sorted(edge)) for edge in graph.edges}


def _canonical_locations() -> dict[bytes, nx.Graph]:
    locations = {}
    for base in load_repository():
        variants = (base, as_single_root(base)) if base.graph['R'] > 1 else (base,)
        for location in variants:
            digest = fingerprint_coordinates(location.graph['VertexC'])[0]
            if digest in locations:
                raise ValueError(f'duplicate canonical location digest: {digest.hex()}')
            locations[digest] = location
    return locations


def generate(database: Path) -> dict[bytes, np.ndarray]:
    """Load and verify historical coordinates for every curated alias."""
    canonical_locations = _canonical_locations()
    coordinates = {}
    with database_connection(str(database)):
        for historical, canonical in NODESET_DIGEST_ALIASES.items():
            try:
                current = canonical_locations[canonical]
            except KeyError as exc:
                raise ValueError(
                    f'alias target is not a canonical bundled digest: {canonical.hex()}'
                ) from exc

            nodeset = NodeSet.get_or_none(NodeSet.digest == historical)
            if nodeset is None:
                raise ValueError(
                    f'alias digest is absent from the database: {historical.hex()}'
                )
            expected_name = NODESET_DIGEST_ALIAS_NAMES[historical]
            if nodeset.name != expected_name:
                raise ValueError(
                    f'alias {historical.hex()} names {nodeset.name!r}; '
                    f'expected {expected_name!r}'
                )

            database_location = L_from_nodeset(nodeset)
            current_pack = packnodes(current)
            for field in (
                'T',
                'R',
                'B',
                'constraint_groups',
                'constraint_vertices',
            ):
                if getattr(nodeset, field) != current_pack[field]:
                    raise ValueError(
                        f'{expected_name} alias disagrees on {field}: '
                        f'{getattr(nodeset, field)!r} != {current_pack[field]!r}'
                    )

            _database_planar, database_available = make_planar_embedding(
                database_location
            )
            _current_planar, current_available = make_planar_embedding(current)
            if set(database_available) != set(current_available) or _edge_set(
                database_available
            ) != _edge_set(current_available):
                raise ValueError(
                    f'{expected_name} alias does not generate the canonical A topology'
                )
            coordinates[historical] = database_location.graph['VertexC']
    return coordinates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument(
        '--output',
        type=Path,
        default=NODESET_DIGEST_ALIAS_COORDINATES_FILE,
    )
    args = parser.parse_args()
    if args.database.suffix != '.sqlite':
        parser.error('database must be a .sqlite file')
    coordinates = generate(args.database)
    with args.output.open('wb') as file:
        pickle.dump(coordinates, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved {len(coordinates)} node-set alias coordinates to {args.output}')


if __name__ == '__main__':
    main()
