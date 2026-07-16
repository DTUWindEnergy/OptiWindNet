# tests/conftest.py
"""
Central pytest fixtures for optiwindnet tests.

Responsibilities:
 - Ensure deterministic test environment (disable numba JIT).
 - Resolve repository/test-files paths.
 - Load expected instances blobs with helpful messages.
 - Provide factory fixtures (router construction, L/G loader, site extractor).
 - Optionally regenerate expected data when `--regen-expected` is passed.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from . import isolation, paths

# required env variables for coverage to work with multiprocessing
os.environ['PYTHONPATH'] = '.'
os.environ['COVERAGE_PROCESS_START'] = '.coveragerc'

REPO_ROOT = paths.REPO_ROOT
SOLUTIONS_FILE = paths.SOLUTIONS_FILE
LOCATIONS_DIR = paths.LOCATIONS_DIR
GEN_END2END_SCRIPT = paths.GEN_END2END_SCRIPT

# Ensure Numba JIT is disabled for tests
os.environ['NUMBA_DISABLE_JIT'] = '1'


# -----------------------
# Utility helpers
# -----------------------
def _maybe_run_generator(script_path: Path) -> None:
    """Run the generator as a module (fresh interpreter, from the repo root).

    The generator uses package-relative imports (``from . import matrix`` etc.),
    so it must run as ``python -m tests.update_expected_values`` rather than as a
    bare script path.
    """
    if not script_path.exists():
        raise FileNotFoundError(f'Generator script not found: {script_path}')
    proc = subprocess.run(
        [sys.executable, '-m', 'tests.update_expected_values'],
        cwd=str(REPO_ROOT),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f'Generator failed: tests.update_expected_values (rc={proc.returncode})'
        )


# -----------------------
# Isolated worker for ortools-touching test code
# -----------------------
# See tests/isolation.py for why ortools needs process isolation. This
# fixture provides ONE persistent worker process (not a fresh spawn per call,
# which costs ~2-4s of cold interpreter + reimport overhead against
# sub-second solves) that any test needing to touch `optiwindnet.MILP.ortools`
# can dispatch work to.
@pytest.fixture(scope='session')
def ortools_worker():
    worker = isolation.ortools_worker_factory()
    yield worker
    worker.shutdown()


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
            'If set, pytest will attempt to regenerate missing expected'
            ' instances files '
            'by running the repository generator scripts. Use with care (generators '
            'may be slow or require external solvers).'
        ),
    )


def pytest_sessionstart(session):
    """If user passed --regen-expected and files are missing, try regenerate them."""
    regen = session.config.getoption('--regen-expected')
    if not regen:
        return

    # Attempt to regenerate missing expected files
    # (best-effort; fail loudly if generator fails)
    if not SOLUTIONS_FILE.exists() and GEN_END2END_SCRIPT.exists():
        session.config.warn(
            'optiwindnet',
            f'Regenerating {SOLUTIONS_FILE} via {GEN_END2END_SCRIPT}',
        )
        _maybe_run_generator(GEN_END2END_SCRIPT)


# -----------------------
# Lazy-loaded repository locations fixture
# -----------------------
@pytest.fixture(scope='session')
def locations():
    """Load the locations used by the end-to-end tests.

    Backed by the single source of truth in ``tests/sites.py`` so this fixture
    and ``update_expected_values.py`` can never drift apart.
    """
    from . import sites

    return sites.load_locations()
