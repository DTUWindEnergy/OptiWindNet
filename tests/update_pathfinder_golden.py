"""Manually regenerate the PathFinder golden routesets.

Run from the repository root with::

    python -m tests.update_pathfinder_golden path/to/routesets.sqlite

Additional positional arguments override ``ROUTESET_IDS``. The output file can
be overridden with ``--output``.

The compressed routeset database the IDS refer to is available from Zenodo:
https://zenodo.org/records/20053479/files/optiwindnet-routesets-r26.05-v4.sqlite.xz?download=1
"""

import argparse
import pickle
from collections.abc import Sequence
from pathlib import Path

from optiwindnet.db import G_from_routeset, RouteSet, database_connection
from optiwindnet.terse import TerseLinks

ROUTESET_IDS = (
    21623,
    32884,
    32753,
    21624,
    21625,
    36270,
    33644,
    21626,
    33409,
    21628,
    21645,
    21655,
    37716,
    21647,
    37557,
    21657,
    21654,
    37607,
    37294,
    39827,
    37250,
    21830,
    21832,
    37605,
    37876,
    21833,
    21836,
    37439,
    21890,
    37376,
    223,
    191,
    3179,
    208,
)


DEFAULT_OUTPUT = Path(__file__).with_name('pathfinder_golden.pkl')


def generate(database: Path, routeset_ids: Sequence[int]) -> tuple[TerseLinks, ...]:
    """Load ``routeset_ids`` in order and encode them as routed terse links."""
    if len(routeset_ids) != len(set(routeset_ids)):
        raise ValueError('route-set IDs must be unique')

    golden = []
    with database_connection(str(database)):
        for routeset_id in routeset_ids:
            G = G_from_routeset(RouteSet.get_by_id(routeset_id))
            if G.graph['capacity'] != G.graph['max_load']:
                raise ValueError(
                    f'route set {routeset_id} does not use its full cable capacity; '
                    'capacity cannot be represented by TerseLinks'
                )
            golden.append(TerseLinks.from_routeset(G))
    return tuple(golden)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path, help='source .sqlite routeset database')
    parser.add_argument(
        'routeset_ids',
        metavar='ID',
        type=int,
        nargs='*',
        help='route-set IDs to store (defaults to the curated ROUTESET_IDS)',
    )
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.database.suffix != '.sqlite':
        parser.error('database must be a .sqlite file')

    routeset_ids = args.routeset_ids or ROUTESET_IDS
    golden = generate(args.database, routeset_ids)
    with args.output.open('wb') as file:
        pickle.dump(golden, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved {len(golden)} TerseLinks instances to {args.output}')


if __name__ == '__main__':
    main()
