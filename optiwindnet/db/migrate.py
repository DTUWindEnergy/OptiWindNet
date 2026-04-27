# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/
"""Migrate a v2 (Pony ORM) or v3 (Peewee) database to v4 (Peewee, slim RouteSet schema).

Uses sqlite3 to read the source file (no Pony dependency), writes v4 with
Peewee. The RouteSet field `num_gates` is renamed to `feeders_per_root`,
and the columns dropped in v4 (`valid`, `is_normalized`, `stuntC`) are
discarded.

Usage:
    python -m optiwindnet.db.migrate input.sqlite output.v4.sqlite
"""

import json
import os
import sqlite3
import sys

from . import open_database, Machine, Method, NodeSet, RouteSet
from .model import database_proxy


def _get_source_table_name(src_conn, candidate_names):
    """Find the actual table name in the source database from candidates."""
    cursor = src_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing = {row[0] for row in cursor}
    for name in candidate_names:
        if name in existing:
            return name
    raise ValueError(
        f'None of {candidate_names} found in source database. Existing tables: {existing}'
    )


def migrate(src_path, v4_path):
    """Migrate a v2 (Pony ORM) or v3 (Peewee) database to v4 Peewee format.

    Args:
        src_path: Path to the existing v2 or v3 database file.
        v4_path: Path for the new v4 database file (must not exist).
    """
    src_path = os.path.abspath(os.path.expanduser(src_path))
    v4_path = os.path.abspath(os.path.expanduser(v4_path))

    if not os.path.exists(src_path):
        raise OSError(f'source database not found: {src_path}')
    if os.path.exists(v4_path):
        raise OSError(f'v4 database already exists: {v4_path}')

    # Open the source with raw sqlite3
    src_conn = sqlite3.connect(src_path)
    src_conn.row_factory = sqlite3.Row

    # Discover table names (v2/Pony uses PascalCase, v3/Peewee uses lowercase)
    ns_table = _get_source_table_name(src_conn, ['NodeSet', 'nodeset'])
    method_table = _get_source_table_name(src_conn, ['Method', 'method'])
    machine_table = _get_source_table_name(src_conn, ['Machine', 'machine'])
    rs_table = _get_source_table_name(src_conn, ['RouteSet', 'routeset'])

    # Create v4 database
    v4_db = open_database(v4_path, create_db=True)

    with database_proxy.atomic():
        # --- Migrate NodeSet ---
        src_rows = src_conn.execute(f'SELECT * FROM "{ns_table}"').fetchall()
        for row in src_rows:
            row_dict = dict(row)
            # IntArray columns are stored as JSON text
            for col in ('constraint_groups', 'constraint_vertices'):
                val = row_dict[col]
                if isinstance(val, str):
                    row_dict[col] = json.loads(val)
            NodeSet.create(**row_dict)
        ns_count = len(src_rows)
        print(f'  NodeSet: {ns_count} rows migrated')

        # --- Migrate Method ---
        src_rows = src_conn.execute(f'SELECT * FROM "{method_table}"').fetchall()
        for row in src_rows:
            row_dict = dict(row)
            # options is stored as JSON text
            val = row_dict.get('options')
            if isinstance(val, str):
                row_dict['options'] = json.loads(val)
            Method.create(**row_dict)
        method_count = len(src_rows)
        print(f'  Method: {method_count} rows migrated')

        # --- Migrate Machine ---
        src_rows = src_conn.execute(f'SELECT * FROM "{machine_table}"').fetchall()
        for row in src_rows:
            row_dict = dict(row)
            val = row_dict.get('attrs')
            if isinstance(val, str):
                row_dict['attrs'] = json.loads(val)
            Machine.create(**row_dict)
        machine_count = len(src_rows)
        print(f'  Machine: {machine_count} rows migrated')

        # --- Migrate RouteSet ---
        src_rows = src_conn.execute(f'SELECT * FROM "{rs_table}"').fetchall()

        # JSON columns in RouteSet (`num_gates` is the legacy v2/v3 name)
        json_cols = {
            'num_gates',
            'feeders_per_root',
            'edges',
            'tentative',
            'rogue',
            'clone2prime',
            'misc',
        }

        for row in src_rows:
            row_dict = dict(row)

            # Parse JSON text columns
            for col in json_cols:
                if col in row_dict:
                    val = row_dict[col]
                    if isinstance(val, str):
                        row_dict[col] = json.loads(val)

            # `num_gates` was renamed to `feeders_per_root` in v4
            if 'feeders_per_root' not in row_dict and 'num_gates' in row_dict:
                row_dict['feeders_per_root'] = row_dict.pop('num_gates')
            else:
                row_dict.pop('num_gates', None)

            # Drop columns removed in v4
            for col in ('valid', 'is_normalized', 'stuntC'):
                row_dict.pop(col, None)

            # Ensure misc is never NULL
            if row_dict.get('misc') is None:
                row_dict['misc'] = {}

            RouteSet.create(**row_dict)
        rs_count = len(src_rows)
        print(f'  RouteSet: {rs_count} rows migrated')

    src_conn.close()

    # Verify counts
    assert NodeSet.select().count() == ns_count
    assert Method.select().count() == method_count
    assert Machine.select().count() == machine_count
    assert RouteSet.select().count() == rs_count

    print(f'\nMigration complete: {v4_path}')
    print(
        f'  NodeSet: {ns_count}, Method: {method_count}, '
        f'Machine: {machine_count}, RouteSet: {rs_count}'
    )

    v4_db.close()


def main():
    if len(sys.argv) != 3:
        print('Usage: python -m optiwindnet.db.migrate <v2-or-v3.sqlite> <v4.sqlite>')
        sys.exit(1)
    migrate(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()
