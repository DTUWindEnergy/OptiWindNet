# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/
"""Migrate a v2 (Pony ORM) database to v3 (Peewee).

Uses sqlite3 to read the v2 file (no Pony dependency), writes v3 with Peewee.

Usage:
    python -m optiwindnet.db.migrate input.v2.sqlite output.v3.sqlite
"""

import json
import os
import sqlite3
import sys

from .model import (
    Machine,
    Method,
    NodeSet,
    RouteSet,
    database_proxy,
)
from .storage import open_database


def _get_v2_table_name(v2_conn, candidate_names):
    """Find the actual table name in the v2 database from candidates."""
    cursor = v2_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing = {row[0] for row in cursor}
    for name in candidate_names:
        if name in existing:
            return name
    raise ValueError(
        f'None of {candidate_names} found in v2 database. Existing tables: {existing}'
    )


def migrate(v2_path, v3_path):
    """Migrate a v2 Pony ORM database to v3 Peewee format.

    Args:
        v2_path: Path to existing v2 database file.
        v3_path: Path for the new v3 database file (must not exist).
    """
    v2_path = os.path.abspath(os.path.expanduser(v2_path))
    v3_path = os.path.abspath(os.path.expanduser(v3_path))

    if not os.path.exists(v2_path):
        raise OSError(f'v2 database not found: {v2_path}')
    if os.path.exists(v3_path):
        raise OSError(f'v3 database already exists: {v3_path}')

    # Open v2 with raw sqlite3
    v2_conn = sqlite3.connect(v2_path)
    v2_conn.row_factory = sqlite3.Row

    # Discover table names (Pony uses PascalCase)
    ns_table = _get_v2_table_name(v2_conn, ['NodeSet', 'nodeset'])
    method_table = _get_v2_table_name(v2_conn, ['Method', 'method'])
    machine_table = _get_v2_table_name(v2_conn, ['Machine', 'machine'])
    rs_table = _get_v2_table_name(v2_conn, ['RouteSet', 'routeset'])

    # Create v3 database
    v3_db = open_database(v3_path, create_db=True)

    with database_proxy.atomic():
        # --- Migrate NodeSet ---
        v2_rows = v2_conn.execute(f'SELECT * FROM "{ns_table}"').fetchall()
        for row in v2_rows:
            row_dict = dict(row)
            # IntArray columns are stored as JSON text in Pony
            for col in ('constraint_groups', 'constraint_vertices'):
                val = row_dict[col]
                if isinstance(val, str):
                    row_dict[col] = json.loads(val)
            NodeSet.create(**row_dict)
        ns_count = len(v2_rows)
        print(f'  NodeSet: {ns_count} rows migrated')

        # --- Migrate Method ---
        v2_rows = v2_conn.execute(f'SELECT * FROM "{method_table}"').fetchall()
        for row in v2_rows:
            row_dict = dict(row)
            # options is stored as JSON text
            val = row_dict.get('options')
            if isinstance(val, str):
                row_dict['options'] = json.loads(val)
            Method.create(**row_dict)
        method_count = len(v2_rows)
        print(f'  Method: {method_count} rows migrated')

        # --- Migrate Machine ---
        v2_rows = v2_conn.execute(f'SELECT * FROM "{machine_table}"').fetchall()
        for row in v2_rows:
            row_dict = dict(row)
            val = row_dict.get('attrs')
            if isinstance(val, str):
                row_dict['attrs'] = json.loads(val)
            Machine.create(**row_dict)
        machine_count = len(v2_rows)
        print(f'  Machine: {machine_count} rows migrated')

        # --- Migrate RouteSet ---
        v2_rows = v2_conn.execute(f'SELECT * FROM "{rs_table}"').fetchall()

        # JSON columns in RouteSet
        json_cols = {
            'num_gates',
            'edges',
            'tentative',
            'rogue',
            'clone2prime',
            'misc',
        }

        for row in v2_rows:
            row_dict = dict(row)

            # Parse JSON text columns
            for col in json_cols:
                if col in row_dict:
                    val = row_dict[col]
                    if isinstance(val, str):
                        row_dict[col] = json.loads(val)

            # Ensure misc is never NULL
            if row_dict.get('misc') is None:
                row_dict['misc'] = {}

            RouteSet.create(**row_dict)
        rs_count = len(v2_rows)
        print(f'  RouteSet: {rs_count} rows migrated')

    v2_conn.close()

    # Verify counts
    assert NodeSet.select().count() == ns_count
    assert Method.select().count() == method_count
    assert Machine.select().count() == machine_count
    assert RouteSet.select().count() == rs_count

    print(f'\nMigration complete: {v3_path}')
    print(
        f'  NodeSet: {ns_count}, Method: {method_count}, '
        f'Machine: {machine_count}, RouteSet: {rs_count}'
    )

    v3_db.close()


def main():
    if len(sys.argv) != 3:
        print('Usage: python -m optiwindnet.db.migrate <v2.sqlite> <v3.sqlite>')
        sys.exit(1)
    migrate(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()
