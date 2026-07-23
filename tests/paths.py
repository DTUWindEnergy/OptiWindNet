# tests/paths.py
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent if HERE.name == 'tests' else Path.cwd()

LOCATIONS_DIR = (REPO_ROOT / 'tests' / 'locations').resolve()
DATA_DIR = (REPO_ROOT / 'optiwindnet' / 'data').resolve()
