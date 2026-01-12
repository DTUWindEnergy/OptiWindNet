# tests/conftest.py
"""
Central pytest fixtures for optiwindnet tests.

Responsibilities:
 - Ensure deterministic test environment (disable numba JIT).
 - Resolve repository/test-files paths.
 - Load expected dill blobs with helpful messages.
 - Provide factory fixtures (router construction, L/G loader, site extractor).
 - Optionally regenerate expected data when `--regen-expected` is passed.
"""

import os
import sys
import subprocess
from pathlib import Path

import pytest

from . import paths

# required env variables for coverage to work with multiprocessing
os.environ['PYTHONPATH'] = '.'
os.environ['COVERAGE_PROCESS_START'] = '.coveragerc'

REPO_ROOT = paths.REPO_ROOT
END_TO_END_DILL = paths.END_TO_END_DILL
TEST_FILES_DIR = paths.TEST_FILES_DIR
SITES_DIR = paths.SITES_DIR
GEN_END2END_SCRIPT = paths.GEN_END2END_SCRIPT

# Ensure Numba JIT is disabled for tests
os.environ['NUMBA_DISABLE_JIT'] = '1'


# -----------------------
# Utility helpers
# -----------------------
def _maybe_run_generator(script_path: Path) -> None:
    """Run a generator script via subprocess (fresh Python interpreter)."""
    if not script_path.exists():
        raise FileNotFoundError(f'Generator script not found: {script_path}')
    # Use the same python interpreter
    proc = subprocess.run([sys.executable, str(script_path)], check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f'Generator script failed: {script_path} (rc={proc.returncode})'
        )


# -----------------------
# Pytest CLI option (optional regeneration)
# -----------------------
def pytest_addoption(parser):
    group = parser.getgroup('optiwindnet', 'optiwindnet test helpers')
    group.addoption(
        '--regen-expected',
        action='store_true',
        default=False,
        help=(
            'If set, pytest will attempt to regenerate missing expected dill files '
            'by running the repository generator scripts. Use with care (generators '
            'may be slow or require external solvers).'
        ),
    )


def pytest_sessionstart(session):
    """If user passed --regen-expected and files are missing, try regenerate them."""
    regen = session.config.getoption('--regen-expected')
    if not regen:
        return

    # Attempt to regenerate missing expected files (best-effort; fail loudly if generator fails)
    if not END_TO_END_DILL.exists() and GEN_END2END_SCRIPT.exists():
        session.config.warn(
            'optiwindnet', f'Regenerating {END_TO_END_DILL} via {GEN_END2END_SCRIPT}'
        )
        _maybe_run_generator(GEN_END2END_SCRIPT)


# -----------------------
# Lazy-loaded repository locations fixture
# -----------------------
@pytest.fixture(scope='session')
def locations():
    """Load repository-backed sites (the same loader used by generator scripts)."""
    try:
        from optiwindnet.importer import load_repository  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            'Failed to import load_repository from optiwindnet.importer'
        ) from exc
    return load_repository(SITES_DIR)
