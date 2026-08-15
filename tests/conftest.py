# tests/conftest.py
"""
Central pytest fixtures for optiwindnet tests.

Responsibilities:
 - Ensure deterministic test environment (disable numba JIT).
 - Provide the persistent OR-Tools isolation worker and the per-solver dispatch
   that routes each job to the process it is allowed to run in.
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
#
# It is ortools-only: `solver_factory()` refuses pyscipopt/highspy in a process
# that has loaded ortools, so solver-parametrized tests must go through
# `run_isolated` below rather than send every isolated job here.
@pytest.fixture(scope='session')
def ortools_worker():
    worker = isolation.ortools_worker_factory()
    yield worker
    worker.shutdown()


# -----------------------
# Per-solver job dispatch
# -----------------------
@pytest.fixture(scope='session')
def run_isolated(request):
    """Run a solver job in whichever process that solver is allowed to use.

    Returns a ``run(solver_name, func, args, timeout)`` callable yielding either
    the job's result or the exception it raised (never re-raising), so callers
    handle both the in-process and the subprocess outcome the same way.
    """

    def run(solver_name: str, func, args: tuple, timeout: float):
        match isolation.worker_kind(solver_name):
            case 'ortools':
                # getfixturevalue: leave the worker unspawned until a test needs it
                return request.getfixturevalue('ortools_worker').run(
                    func, args, timeout
                )
            case 'scip':
                return isolation.run_disposable(func, args, timeout)
            case _:
                try:
                    return func(*args)
                except BaseException as exc:
                    return exc

    return run


# -----------------------
# Repository-wide locations fixture
# -----------------------
@pytest.fixture(scope='session')
def locations():
    """Load all bundled sites only for repository-wide importer/clustering tests."""
    from .sitecache import location_repository

    return location_repository()
