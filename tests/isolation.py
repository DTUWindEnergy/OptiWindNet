# tests/isolation.py
"""
Isolated subprocess worker for solver code that cannot share a process with
other solvers.

``ortools.math_opt`` bundles its own copies of HiGHS and SCIP under the same
soname as the standalone highspy/pyscipopt packages; loading ortools and one
of those standalone packages into the same process breaks whichever one loads
second (undefined native symbols). Rather than isolate every solver
defensively, only `ortools` ever needs to be pushed into a subprocess -- it's
the party vendoring copies of other solvers' libraries.

This module has no pytest dependency so it can be used both from
`conftest.py` (as the `ortools_worker` fixture) and from plain scripts such as
`update_expected_values.py`.
"""

import multiprocessing
import multiprocessing.queues
import queue


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
    """
    worker = IsolatedWorker(multiprocessing.get_context('spawn'))
    # Return value ignored: if ortools is simply unavailable this comes back
    # as a (Module)NotFoundError, not a raise -- individual tests already
    # handle that and skip. A genuine TimeoutError here does raise, which is
    # the desired outcome (fail fast in setup rather than once per test).
    worker.run(_warmup_ortools, (), _WARMUP_TIMEOUT)
    return worker
