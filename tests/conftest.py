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

import multiprocessing
import multiprocessing.queues
import os
import queue
import subprocess
import sys
from pathlib import Path

import pytest

from . import paths

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
# Isolated worker for ortools-touching test code
# -----------------------
# ortools.math_opt bundles its own copies of HiGHS and SCIP under the same
# soname as the standalone highspy/pyscipopt packages; loading ortools and one
# of those standalone packages into the same process breaks whichever one
# loads second (undefined native symbols). Rather than isolate every solver
# defensively, only `ortools` ever needs to be pushed into a subprocess -- it's
# the party vendoring copies of other solvers' libraries. This fixture provides
# ONE persistent worker process (not a fresh spawn per call, which costs ~2-4s
# of cold interpreter + reimport overhead against sub-second solves) that any
# test needing to touch `optiwindnet.MILP.ortools` can dispatch work to.
def _job_dispatcher_loop(job_queue, result_queue) -> None:
    """Run in the persistent worker process: execute jobs until told to stop.

    Deliberately solver-agnostic: no ortools/highspy import here, only inside
    whatever `func` a job provides.
    """
    while True:
        job = job_queue.get()
        if job is None:
            return
        func, args = job
        try:
            result = func(*args)
        except BaseException as exc:
            result_queue.put(exc)
        else:
            result_queue.put(result)


class _IsolatedWorker:
    """Persistent subprocess that executes `(func, args)` jobs one at a time.

    Holds mutable process/queue state (rather than being a plain namedtuple)
    so `run()` can transparently respawn a dead worker and retry once, without
    every caller having to implement that dance itself.
    """

    def __init__(self, ctx: multiprocessing.context.SpawnContext):
        self._ctx = ctx
        self.job_queue: multiprocessing.queues.Queue
        self.result_queue: multiprocessing.queues.Queue
        self.process: multiprocessing.process.BaseProcess
        self._spawn()

    def _spawn(self) -> None:
        self.job_queue = self._ctx.Queue()
        self.result_queue = self._ctx.Queue()
        self.process = self._ctx.Process(
            target=_job_dispatcher_loop, args=(self.job_queue, self.result_queue)
        )
        self.process.start()

    def run(self, func, args: tuple, timeout: float):
        self.job_queue.put((func, args))
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            if self.process.is_alive():
                raise TimeoutError(
                    f'{func.__module__}.{func.__qualname__} did not respond'
                    f' within {timeout}s (worker still alive -- not retrying)'
                ) from None
            # Worker died silently (e.g. OOM-killed): replace it and retry once.
            self.process.terminate()
            self.process.join(timeout=5)
            self._spawn()
            self.job_queue.put((func, args))
            return self.result_queue.get(timeout=timeout)

    def shutdown(self) -> None:
        if self.process.is_alive():
            self.job_queue.put(None)
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join()


@pytest.fixture(scope='session')
def ortools_worker():
    ctx = multiprocessing.get_context('spawn')
    worker = _IsolatedWorker(ctx)
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
    """Load locations used by end-to-end tests, by explicit file name."""
    from collections import namedtuple

    from optiwindnet.importer import L_from_yaml

    data_dir = paths.DATA_DIR
    location_files = {
        'hornsea': (L_from_yaml, data_dir / 'Hornsea One.yaml'),
        'london': (L_from_yaml, data_dir / 'London Array.yaml'),
        'taylor_2023': (L_from_yaml, data_dir / 'Taylor-2023.yaml'),
        'yi_2019': (L_from_yaml, data_dir / 'Yi-2019.yaml'),
        'borkum2': (L_from_yaml, data_dir / 'Borkum Riffgrund 2.yaml'),
        'example_location': (L_from_yaml, LOCATIONS_DIR / 'example_location.yaml'),
    }
    loaded = {handle: loader(path) for handle, (loader, path) in location_files.items()}
    Locations = namedtuple('Locations', loaded.keys())
    return Locations(**loaded)
