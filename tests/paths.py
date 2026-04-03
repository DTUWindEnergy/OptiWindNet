# tests/paths.py
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent if HERE.name == 'tests' else Path.cwd()

LOCATIONS_DIR = (REPO_ROOT / 'tests' / 'locations').resolve()
DATA_DIR = (REPO_ROOT / 'optiwindnet' / 'data').resolve()
SOLUTIONS_FILE = (REPO_ROOT / 'tests' / 'solutions.pkl').resolve()

# Optional script locations (used by conftest for regeneration hints)
GEN_END2END_SCRIPT = (REPO_ROOT / 'update_expected_values.py').resolve()
