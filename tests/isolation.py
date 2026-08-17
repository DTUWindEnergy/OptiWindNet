# tests/isolation.py
"""
Isolated subprocess workers for solver code that cannot share a process with
other solvers.

``ortools.math_opt`` bundles its own copies of HiGHS and SCIP under the same
soname as the standalone highspy/pyscipopt packages; loading ortools and one
of those standalone packages into the same process breaks whichever one loads
second (undefined native symbols) -- and `solver_factory()` refuses the second
one outright. Rather than isolate every solver defensively, only `ortools` is
pushed into a subprocess for that reason -- it's the party vendoring copies of
other solvers' libraries.

Windows adds a second, unrelated reason to isolate: consecutive calls to SCIP's
native concurrent solver (solveConcurrent()) can trigger an access violation in
C thread state on msvcrt. That one is not solved by *a* subprocess but by a
*fresh* one per job -- and never by the ortools worker, which holds precisely
the package pyscipopt must not meet.

Hence two shapes of isolation, one per hazard:
 - `'ortools'`: one persistent, pre-warmed worker, reused across jobs;
 - `'scip'`: a throwaway worker per job (`run_disposable`).

This module has no pytest dependency so it can be used both from
`conftest.py` (as the `ortools_worker` and `run_isolated` fixtures) and from
plain scripts such as topology-golden generation.
"""

import multiprocessing
import multiprocessing.queues
import queue
import sys


def worker_kind(solver_name: str) -> str | None:
    """Which isolated worker family a solver's test execution needs, if any.

    Returns:
      ``'ortools'``, ``'scip'``, or None if the solver may run in-process.

    OR-Tools always requires process isolation (DLL symbol collision with
    highspy/pyscipopt). SCIP and FSCIP on Windows also require process isolation
    because consecutive calls to SCIP's native C concurrent solver
    (solveConcurrent()) can trigger an access violation in C thread state on
    msvcrt -- but they need a *different* process from the ortools one, since
    pyscipopt and ortools cannot share an interpreter instance.
    """
    if solver_name.startswith('ortools'):
        return 'ortools'
    if sys.platform == 'win32' and solver_name in ('scip', 'fscip'):
        return 'scip'
    return None


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
        # sent across the process boundary, so nothing may escape
        except BaseException as exc:  # noqa: BLE001
            result_queue.put(exc)
        else:
            result_queue.put(result)


class IsolatedWorker:
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


def _warmup_ortools() -> None:
    """Force ortools' native libraries to load in the worker process.

    `ortools.math_opt` is a large extension module; importing it cold can
    take much longer than any individual solve (worse under 'spawn', which
    re-execs and re-imports the interpreter from scratch, and worse still
    when several pytest-xdist workers do this at once on a loaded CI
    runner). Paying that cost once here -- under a generous timeout, before
    any test's tighter per-call timeout starts ticking -- keeps that cost
    off of every other `IsolatedWorker.run()` call.
    """
    from optiwindnet.MILP import solver_factory

    solver_factory('ortools.cp_sat')


# Generous on purpose: covers 'spawn' process creation plus cold import of
# ortools' bundled native libraries under CI contention, not just a solve.
_WARMUP_TIMEOUT = 120


def ortools_worker_factory() -> IsolatedWorker:
    """Spawn a fresh `IsolatedWorker` on a 'spawn' multiprocessing context,
    pre-warmed so its first real job isn't also paying import-time cost.

    The worker is ortools-only: no job that loads highspy or pyscipopt may be
    sent to it (see `worker_kind`).
    """
    worker = IsolatedWorker(multiprocessing.get_context('spawn'))
    # Return value ignored: if ortools is simply unavailable this comes back
    # as a (Module)NotFoundError, not a raise -- individual tests already
    # handle that and skip. A genuine TimeoutError here does raise, which is
    # the desired outcome (fail fast in setup rather than once per test).
    worker.run(_warmup_ortools, (), _WARMUP_TIMEOUT)
    return worker


def run_disposable(func, args: tuple, timeout: float):
    """Run one job in a throwaway worker process, then dispose of it.

    Used for the SCIP family on Windows, where the point is not just "some
    other process" but a *pristine* one: no ortools (which pyscipopt cannot
    share an interpreter instance with) and no earlier SCIP solve (whose
    solveConcurrent() thread state is what faults on the next call).
    """
    worker = IsolatedWorker(multiprocessing.get_context('spawn'))
    try:
        return worker.run(func, args, timeout)
    finally:
        worker.shutdown()
