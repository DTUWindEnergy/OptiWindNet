# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

# tests/conftest.py
"""
Central pytest fixtures for optiwindnet tests.

Responsibilities:
 - Ensure deterministic test environment (disable numba JIT).
 - Provide the persistent OR-Tools isolation worker and the per-solver dispatch
   that routes each job to the process it is allowed to run in.
 - Run SCIP's concurrent solves in opportunistic mode.
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
            case _:
                try:
                    return func(*args)
                # marshalled back to the caller, so nothing may escape
                except BaseException as exc:  # noqa: BLE001
                    return exc

    return run


# -----------------------
# SCIP concurrent mode
# -----------------------
@pytest.fixture(scope='session', autouse=True)
def scip_opportunistic_mode():
    """Make every SolverSCIP in the tests solve in opportunistic parallel mode.

    SCIP 10.0's deterministic mode (`parallel/mode` = 1, the default) can
    deadlock in `solveConcurrent()`: a solver waits at a synchronization point
    for others that have already stopped. The ringed albatros model of
    `test_ringed_warmstart_is_accepted_by_scip` hangs this way every time.
    Opportunistic mode does not wait for the other solvers; the test problems are
    simple enough that its run-to-run variability does not affect the checks.
    """
    try:
        from optiwindnet.MILP.scip import SolverSCIP
    except ModuleNotFoundError:
        yield
        return
    original_init = SolverSCIP.__init__

    def __init__(self):
        original_init(self)
        self.options['parallel/mode'] = 0  # 0: opportunistic

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SolverSCIP, '__init__', __init__)
        yield


# -----------------------
# Repository-wide locations fixture
# -----------------------
@pytest.fixture(scope='session')
def locations():
    """Load all bundled sites only for repository-wide importer/clustering tests."""
    from .sitecache import location_repository

    return location_repository()
