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


def ortools_worker_factory() -> IsolatedWorker:
    """Spawn a fresh `IsolatedWorker` on a 'spawn' multiprocessing context."""
    return IsolatedWorker(multiprocessing.get_context('spawn'))
