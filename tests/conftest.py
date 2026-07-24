# tests/conftest.py
"""
Central pytest fixtures for optiwindnet tests.

Responsibilities:
 - Ensure deterministic test environment (disable numba JIT).
 - Provide the persistent OR-Tools isolation worker.
 - Load the whole repository only for tests that intentionally exercise it.
"""

import os

# Set these before importing any OptiWindNet module that defines Numba functions.
os.environ['PYTHONPATH'] = '.'
os.environ['COVERAGE_PROCESS_START'] = '.coveragerc'
os.environ['NUMBA_DISABLE_JIT'] = '1'

import pytest

from . import isolation


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
# Repository-wide locations fixture
# -----------------------
@pytest.fixture(scope='session')
def locations():
    """Load all bundled sites only for repository-wide importer/clustering tests."""
    from .sitecache import location_repository

    return location_repository()
