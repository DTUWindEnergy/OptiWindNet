# tests/paths.py
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent if HERE.name == 'tests' else Path.cwd()

TEST_FILES_DIR = (REPO_ROOT / 'tests' / 'test_files').resolve()
SITES_DIR = (TEST_FILES_DIR / 'sites').resolve()

END_TO_END_DILL = (TEST_FILES_DIR / 'expected_end_to_end.dill').resolve()

# Optional script locations (used by conftest for regeneration hints)
GEN_END2END_SCRIPT = (REPO_ROOT / 'update_expected_values.py').resolve()
