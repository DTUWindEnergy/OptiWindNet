from pathlib import Path
import pytest
from optiwindnet.db.modelv2 import open_database

tmp_path = Path(__file__).parent / "temp"

def test_open_database(tmp_path):
    """
    1) Guarantee DB file does not exist and assert open_database(..., create_db=False) raises OSError
    2) Create DB with create_db=True
    3) Assert file exists and DB exposes expected attributes
    """
    dbfile = tmp_path / "db_test.sqlite"

    # ensure file is not present
    if dbfile.exists():
        dbfile.unlink()

    # Expect OSError when trying to open a non-existent DB without create flag
    with pytest.raises(OSError):
        open_database(str(dbfile), create_db=False)
        assert dbfile.exists(), "database file should have been created"

    # create the DB
    db = open_database(str(dbfile), create_db=True)

    expected_attrs = ["Entity", "Machine", "Method", "NodeSet", "RouteSet"]

    for name in expected_attrs:
        assert hasattr(db, name), f"db should expose attribute '{name}'"