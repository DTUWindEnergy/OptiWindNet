# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Sweep harness: run variants over bundled sites and record metrics in SQLite.

A variant is a callable taking one :class:`Task` and returning either a
solution topology ``S`` or a routed solution ``G``. ``S`` is routed with
``G_from_S()`` + ``PathFinder`` so that every variant is measured on a routeset.
Only the variant call is timed; routing, validation and metrics come after.

Example::

    from tests.sweep import EASY_CASES, run_sweep
    from optiwindnet.heuristics import constructor

    def ew(t):
        return constructor(t.A, t.capacity, method='esau_williams')

    def rootlust(t):
        return constructor(t.A, t.capacity, method='rootlust')

    variants = {'ew': ew, 'rootlust': rootlust}
    run_sweep(EASY_CASES[:10], variants, 'artifacts/sweep.sqlite')

Rerunning with the same ``run_id`` resumes: tasks already recorded as ``ok`` or
``invalid`` are skipped and errored tasks are retried. Resuming is refused if a
variant's source or the code state changed. The code state is the HEAD commit
plus a hash of ``git diff HEAD`` (untracked files are not covered).

To measure a library edit, sweep once before and once after the edit with
different run ids, then compare the runs: ``report(db_path, ['before',
'after'])``.

Inspect results with ``python -m tests.sweep DB_PATH [RUN_ID ...] [--baseline
LABEL] [--per-case]``, rebuild a recorded solution with :func:`load_solution`,
or query the tables directly:

- ``runs(run_id, started, code, executor, workers, threads, variants)``
- ``results(id, run_id, variant, fingerprint, site, capacity, options, rep,
  status, time_s, length, violations, extras, linkbits, linkset_id, error,
  finished)``

``status`` is ``'ok'``, ``'invalid'`` (``violations`` > 0; the first violation
is in ``error``) or ``'error'`` (the traceback is in ``error``). ``options``
and ``extras`` are JSON; ``extras`` holds the scalar graph attributes of the
variant's return value (e.g. ``runtime``, ``iterations``, ``bound``), so a
variant records custom metrics by setting them in ``S.graph``. ``linkbits`` is
the solution topology encoded over the site's ``A`` (see
``optiwindnet.converting.linkbits_from_S``), serialized with
``bitarray.util.serialize``.
"""

import argparse
import concurrent.futures
import copy
import datetime
import hashlib
import inspect
import json
import multiprocessing
import signal
import sqlite3
import statistics
import subprocess
import time
import traceback
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import networkx as nx
from bitarray.util import deserialize, serialize
from prettyTables import Table

from optiwindnet.converting import (
    G_from_S,
    S_from_G,
    S_from_linkbits,
    linkbits_from_S,
)
from optiwindnet.identity import linkset_id
from optiwindnet.loads import calcload, nonunit_inflow, split_rings_and_calc_loads
from optiwindnet.MILP._core import physical_core_count
from optiwindnet.pathfinding import PathFinder
from optiwindnet.types import Topology
from optiwindnet.validating import validate_routeset

from .sitecache import SiteBundle, get_bundle

__all__ = [
    'EASY_CASES',
    'HARD_CASES',
    'MEDIUM_CASES',
    'Task',
    'load_solution',
    'report',
    'run_sweep',
]

# Cases are (site, capacity) pairs tiered by the runtime of the branched MILP
# (0.5% gap target, warm-started) stored in optiwindnet-routesets-r26.05-v4:
# EASY solves in < 10 s, MEDIUM in 10 s to 10 min, HARD takes longer or ends
# at the time limit. Each tier lists one capacity (4 to 12) per site, spread
# over sites of different sizes, sorted by increasing MILP runtime.
EASY_CASES = (
    ('mermaid', 8),  # T=27, R=1, MILP 0.0301 s
    ('bucht', 10),  # T=31, R=1, MILP 0.0304 s
    ('albatros', 12),  # T=16, R=1, MILP 0.0331 s
    ('galloper', 11),  # T=38, R=1, MILP 0.046 s
    ('amalia', 8),  # T=60, R=1, MILP 0.0973 s
    ('robin', 10),  # T=60, R=1, MILP 0.244 s
    ('nordsee', 12),  # T=54, R=1, MILP 0.332 s
    ('nordseeost', 6),  # T=48, R=1, MILP 0.592 s
    ('luchterduinen', 5),  # T=43, R=1, MILP 0.666 s
    ('rudongdemo', 9),  # T=38, R=1, MILP 0.696 s
    ('race', 9),  # T=91, R=2, MILP 0.812 s
    ('dudgeon', 9),  # T=67, R=1, MILP 0.87 s
    ('kfB', 4),  # T=48, R=1, MILP 0.955 s
    ('changhua1', 5),  # T=75, R=1, MILP 0.968 s
    ('brieuc', 11),  # T=62, R=1, MILP 1.09 s
    ('rudongH8', 6),  # T=65, R=1, MILP 1.21 s
    ('morayeast', 11),  # T=100, R=3, MILP 2.25 s
    ('glotech1', 4),  # T=80, R=1, MILP 2.35 s
    ('gangkou2', 12),  # T=64, R=1, MILP 2.58 s
    ('eagle', 4),  # T=50, R=1, MILP 2.82 s
    ('fecamp', 7),  # T=71, R=1, MILP 3.4 s
    ('inchcape', 4),  # T=72, R=1, MILP 3.87 s
    ('nazaire', 11),  # T=80, R=1, MILP 3.97 s
    ('gemini2', 10),  # T=75, R=1, MILP 5.1 s
    ('bard', 4),  # T=80, R=1, MILP 5.75 s
    ('walney2', 7),  # T=51, R=1, MILP 6.13 s
    ('rudongH6', 5),  # T=100, R=1, MILP 7.55 s
    ('jiaxing1', 5),  # T=74, R=1, MILP 8.75 s
    ('doggerA', 4),  # T=95, R=1, MILP 9.7 s
    ('thor', 7),  # T=72, R=1, MILP 9.73 s
)

MEDIUM_CASES = (
    ('triborkum', 4),  # T=72, R=1, MILP 10.2 s
    ('noirmoutier', 8),  # T=61, R=1, MILP 10.3 s
    ('race', 4),  # T=91, R=2, MILP 11.4 s
    ('nordsee', 8),  # T=54, R=1, MILP 11.5 s
    ('kfB', 7),  # T=48, R=1, MILP 11.6 s
    ('kaskasi', 6),  # T=38, R=1, MILP 11.7 s
    ('rudongH8', 11),  # T=65, R=1, MILP 13.5 s
    ('walneyext', 4),  # T=87, R=2, MILP 13.6 s
    ('morayeast', 7),  # T=100, R=3, MILP 16.3 s
    ('shengsi2', 9),  # T=63, R=1, MILP 19.7 s
    ('gemini1', 5),  # T=75, R=1, MILP 22.7 s
    ('hornsea2w', 4),  # T=110, R=1, MILP 28.2 s
    ('gwynt', 12),  # T=160, R=2, MILP 33.5 s
    ('fecamp', 5),  # T=71, R=1, MILP 60.4 s
    ('rampion', 9),  # T=116, R=1, MILP 63 s
    ('borkum', 6),  # T=78, R=1, MILP 63.8 s
    ('nazaire', 9),  # T=80, R=1, MILP 71.1 s
    ('bard', 7),  # T=80, R=1, MILP 74.2 s
    ('triton', 9),  # T=90, R=2, MILP 76.6 s
    ('nanpeng', 10),  # T=55, R=1, MILP 120 s
    ('anglia', 11),  # T=102, R=1, MILP 130 s
    ('jiaxing1', 10),  # T=74, R=1, MILP 135 s
    ('hornsea', 11),  # T=174, R=3, MILP 136 s
    ('borssele', 12),  # T=173, R=2, MILP 187 s
    ('glotech1', 12),  # T=80, R=1, MILP 260 s
    ('northwind', 12),  # T=72, R=1, MILP 384 s
    ('eagle', 11),  # T=50, R=1, MILP 421 s
    ('rødsand2', 8),  # T=90, R=1, MILP 427 s
    ('rudongH6', 10),  # T=100, R=1, MILP 435 s
    ('doggerB', 6),  # T=95, R=1, MILP 458 s
)

HARD_CASES = (
    ('thanet', 6),  # T=100, R=1, MILP 617 s
    ('rudongH6', 12),  # T=100, R=1, MILP 671 s
    ('london', 4),  # T=175, R=2, MILP 688 s
    ('bodhi', 10),  # T=75, R=1, MILP 724 s
    ('borkum3', 8),  # T=83, R=1, MILP 812 s
    ('anglia', 10),  # T=102, R=1, MILP 846 s
    ('seagreen', 12),  # T=114, R=1, MILP 888 s
    ('doggerB', 10),  # T=95, R=1, MILP 907 s
    ('inchcape', 7),  # T=72, R=1, MILP 943 s
    ('rampion', 11),  # T=116, R=1, MILP 1.06e+03 s
    ('doggerC', 8),  # T=87, R=1, MILP 1.21e+03 s
    ('borssele', 4),  # T=173, R=2, MILP 1.26e+03 s
    ('meerwind', 12),  # T=80, R=1, MILP 1.36e+03 s
    ('sofia', 9),  # T=100, R=1, MILP 1.44e+03 s
    ('rudongH10', 8),  # T=100, R=1, MILP 1.81e+03 s
    ('binhainorthH2', 11),  # T=100, R=1, MILP 2.51e+03 s
    ('hornsea', 7),  # T=174, R=3, MILP 2.66e+03 s
    ('anholt', 7),  # T=111, R=1, MILP 3.29e+03 s
    ('northwind', 9),  # T=72, R=1, MILP 5.63e+03 s
    ('rødsand2', 6),  # T=90, R=1, MILP 6.72e+03 s
    ('treport', 8),  # T=62, R=1, MILP 6.72e+03 s
    ('baltic2', 8),  # T=80, R=1, MILP 1.48e+04 s
    ('doggerA', 11),  # T=95, R=1, MILP gap 0.6% at 6 h
    ('gabbin', 7),  # T=102, R=1, MILP gap 0.7% at 3 h
    ('hornsea2w', 5),  # T=110, R=1, MILP gap 0.8% at 3 h
    ('sands', 9),  # T=108, R=1, MILP gap 0.9% at 3 h
    ('horns', 5),  # T=80, R=1, MILP gap 0.9% at 3 h
    ('gwynt', 5),  # T=160, R=2, MILP gap 1.0% at 3 h
    ('belwind', 5),  # T=55, R=1, MILP gap 1.1% at 6 h
    ('amrumbank', 6),  # T=80, R=1, MILP gap 1.5% at 6 h
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started TEXT NOT NULL,
    code TEXT,
    executor TEXT NOT NULL,
    workers INTEGER NOT NULL,
    threads INTEGER NOT NULL,
    variants TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    variant TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    site TEXT NOT NULL,
    capacity INTEGER NOT NULL,
    options TEXT NOT NULL,
    rep INTEGER NOT NULL,
    status TEXT NOT NULL,
    time_s REAL,
    length REAL,
    violations INTEGER,
    extras TEXT,
    linkbits BLOB,
    linkset_id BLOB,
    error TEXT,
    finished TEXT NOT NULL
);
"""


@dataclass(frozen=True, slots=True)
class Task:
    """Input of one variant call; ``L``, ``P`` and ``A`` are private copies.

    Attributes:
        site: Location handle, e.g. ``'cazzaro_2022'``.
        capacity: Cable capacity (maximum turbines per feeder).
        options: Case options, forwarded untouched from the case tuple.
        rep: Repetition number, starting at 1.
        threads: Thread budget for multithreaded solvers.
        L: Location graph.
        P: Navigation mesh.
        A: Available-links graph.
    """

    site: str
    capacity: int
    options: dict[str, Any]
    rep: int
    threads: int
    L: nx.Graph = field(repr=False)
    P: nx.PlanarEmbedding = field(repr=False)
    A: nx.Graph = field(repr=False)


@dataclass(frozen=True, slots=True)
class _Job:
    variant: str
    fn: Callable[[Task], nx.Graph]
    fingerprint: str
    site: str
    capacity: int
    options: dict[str, Any]
    rep: int
    threads: int
    timeout: float | None

    @property
    def key(self) -> tuple[str, str, int, str, int]:
        return (
            self.variant,
            self.site,
            self.capacity,
            json.dumps(self.options, sort_keys=True),
            self.rep,
        )


# Jobs are published here before forking so that workers receive only an index
# and variants need not be picklable (closures and notebook functions work).
_JOBS: list[_Job] = []


def _fingerprint(fn: Callable) -> str:
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        source = repr(fn)
    return hashlib.sha256(source.encode()).hexdigest()[:12]


def _code_state() -> str | None:
    """Return the HEAD commit, suffixed with a hash of the uncommitted changes."""
    try:
        head, diff = (
            subprocess.run(
                ['git', *args],
                cwd=Path(__file__).parent,
                capture_output=True,
                check=True,
            ).stdout
            for args in (
                ('rev-parse', '--short', 'HEAD'),
                ('diff', '--no-ext-diff', '--no-textconv', 'HEAD'),
            )
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    state = head.decode().strip()
    return f'{state}+{hashlib.sha256(diff).hexdigest()[:8]}' if diff else state


def _scalar_attrs(graph_attrs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in graph_attrs.items()
        if not k.startswith('_') and isinstance(v, int | float | str | bool)
    }


def _raise_timeout(signum, frame) -> None:
    raise TimeoutError('task exceeded its timeout')


def _execute(job: _Job) -> dict[str, Any]:
    # case capacities count turbines, whatever power the site declares
    bundle = get_bundle(job.site, copy=True, read_powers=False)
    task = Task(
        job.site,
        job.capacity,
        copy.deepcopy(job.options),
        job.rep,
        job.threads,
        bundle.L,
        bundle.P,
        bundle.A,
    )
    row: dict[str, Any] = {
        'variant': job.variant,
        'fingerprint': job.fingerprint,
        'site': job.site,
        'capacity': job.capacity,
        'options': json.dumps(job.options, sort_keys=True),
        'rep': job.rep,
    }
    previous_handler = None
    if job.timeout is not None:
        previous_handler = signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, job.timeout)
    try:
        t0 = time.perf_counter()
        out = job.fn(task)
        row['time_s'] = time.perf_counter() - t0
        extras = _scalar_attrs(out.graph)
        if 'VertexC' in out.graph:
            G, S = out, S_from_G(out)
        else:
            G = PathFinder(G_from_S(out, task.A), task.P, task.A).create_detours()
            S = out
        violations = validate_routeset(G)
        row.update(
            status='invalid' if violations else 'ok',
            length=G.size(weight='length'),
            violations=len(violations),
            extras=json.dumps(extras),
            error=violations[0] if violations else None,
        )
        try:
            row['linkbits'] = serialize(linkbits_from_S(task.A, S))
            row['linkset_id'] = linkset_id(task.A)
        except ValueError:
            pass  # S uses links outside the site's A (e.g. from another mesh)
    except Exception:  # noqa: BLE001 (any variant failure becomes a recorded row)
        row.update(status='error', error=traceback.format_exc())
    finally:
        if job.timeout is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
    return row


def _execute_forked(index: int) -> dict[str, Any]:
    return _execute(_JOBS[index])


def _connect(db_path: Path | str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _pending_jobs(
    conn: sqlite3.Connection, run_id: str, jobs: list[_Job]
) -> list[_Job]:
    recorded = conn.execute(
        'SELECT variant, fingerprint, site, capacity, options, rep, status '
        'FROM results WHERE run_id = ?',
        (run_id,),
    ).fetchall()
    fingerprints = {job.variant: job.fingerprint for job in jobs}
    for r in recorded:
        if fingerprints.get(r['variant'], r['fingerprint']) != r['fingerprint']:
            raise ValueError(
                f'variant {r["variant"]!r} changed since run {run_id!r} recorded it; '
                'use a new run_id'
            )
    done = {
        (r['variant'], r['site'], r['capacity'], r['options'], r['rep'])
        for r in recorded
        if r['status'] != 'error'
    }
    with conn:
        conn.execute(
            "DELETE FROM results WHERE run_id = ? AND status = 'error'", (run_id,)
        )
    return [job for job in jobs if job.key not in done]


def run_sweep(
    cases: Iterable[tuple],
    variants: Mapping[str, Callable[[Task], nx.Graph]],
    db_path: Path | str,
    *,
    run_id: str | None = None,
    reps: int = 1,
    workers: int = 1,
    threads: int | None = None,
    executor: Literal['process', 'thread'] = 'process',
    timeout: float | None = None,
    verbose: bool = True,
) -> str:
    """Run every variant on every case ``reps`` times and record the results.

    Results are written by the calling process as tasks finish, so an
    interrupted sweep keeps its finished rows and resumes when rerun with the
    same ``run_id``. Resuming is refused if a variant's source or the
    repository's code state (see module docstring) changed.

    Args:
        cases: ``(site, capacity)`` or ``(site, capacity, options)`` tuples;
            ``options`` is a dict passed to variants as ``Task.options``.
        variants: Mapping from variant name to callable.
        db_path: SQLite file, created if missing (e.g. under ``artifacts/``).
        run_id: Run identifier; defaults to a UTC timestamp.
        reps: Repetitions per case and variant.
        workers: Concurrent tasks. ``1`` runs in this process, which suits
            debugging and short sweeps.
        threads: Thread budget passed as ``Task.threads``; defaults to the
            physical cores divided by ``workers``.
        executor: ``'process'`` forks workers (Linux only) and suits Python-
            bound code (heuristics, PathFinder). ``'thread'`` suits variants
            that release the GIL, such as MILP solvers. Ignored if
            ``workers == 1``.
        timeout: Seconds allowed per task (variant call, routing and
            validation), after which the task is recorded as a
            ``TimeoutError``. It interrupts Python code only: a call stuck in
            native code (e.g. a solver) is interrupted when it returns, so
            solvers still need their own time limits. Not available with
            ``executor='thread'`` and ``workers > 1``.
        verbose: Print one progress line per task and the final report.

    Returns:
        The run identifier.
    """
    if run_id is None:
        run_id = datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S')
    if threads is None:
        threads = max(1, physical_core_count() // workers)
    if (
        workers > 1
        and executor == 'process'
        and 'fork' not in multiprocessing.get_all_start_methods()
    ):
        raise ValueError("executor='process' needs fork; use 'thread' or workers=1")
    if timeout is not None and (
        not hasattr(signal, 'setitimer') or (workers > 1 and executor == 'thread')
    ):
        raise ValueError("timeout needs SIGALRM and not executor='thread'")

    jobs = []
    for site, capacity, *rest in cases:
        options = rest[0] if rest else {}
        for name, fn in variants.items():
            fingerprint = _fingerprint(fn)
            for rep in range(1, reps + 1):
                jobs.append(
                    _Job(
                        name,
                        fn,
                        fingerprint,
                        site,
                        capacity,
                        options,
                        rep,
                        threads,
                        timeout,
                    )
                )

    code = _code_state()
    conn = _connect(db_path)
    run = conn.execute('SELECT code FROM runs WHERE run_id = ?', (run_id,)).fetchone()
    if run is not None and run['code'] != code:
        conn.close()
        raise ValueError(
            f'code changed since run {run_id!r} ({run["code"]} -> {code}); '
            'use a new run_id'
        )
    with conn:
        conn.execute(
            'INSERT OR IGNORE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)',
            (
                run_id,
                datetime.datetime.now(datetime.UTC).isoformat(),
                code,
                executor if workers > 1 else 'inline',
                workers,
                threads,
                json.dumps(list(variants)),
            ),
        )
    pending = _pending_jobs(conn, run_id, jobs)
    if verbose:
        print(
            f'run {run_id} (code {code}): {len(pending)} tasks to do '
            f'({len(jobs) - len(pending)} already recorded), '
            f'{workers} workers x {threads} threads'
        )
    # Bundles built before forking are inherited by every worker.
    for site in {job.site for job in pending}:
        get_bundle(site)

    t_start = time.perf_counter()

    def record(i: int, row: dict[str, Any]) -> None:
        row['run_id'] = run_id
        row['finished'] = datetime.datetime.now(datetime.UTC).isoformat()
        with conn:
            conn.execute(
                f'INSERT INTO results ({", ".join(row)}) '
                f'VALUES ({", ".join("?" * len(row))})',
                tuple(row.values()),
            )
        if verbose:
            length = f' length={row["length"]:.1f}' if row.get('length') else ''
            time_s = f' {row["time_s"]:.3f}s' if row.get('time_s') else ''
            print(
                f'[{i}/{len(pending)} {time.perf_counter() - t_start:.0f}s] '
                f'{row["site"]} cap={row["capacity"]} {row["variant"]} '
                f'rep={row["rep"]}: {row["status"]}{time_s}{length}'
            )

    try:
        if workers == 1:
            for i, job in enumerate(pending, 1):
                record(i, _execute(job))
        else:
            if executor == 'process':
                _JOBS[:] = pending
                pool = concurrent.futures.ProcessPoolExecutor(
                    workers, mp_context=multiprocessing.get_context('fork')
                )
                # The warning counts OpenBLAS's idle native threads, which are
                # fork-safe; the executor forks all workers on the first submit.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'ignore',
                        'This process .* is multi-threaded',
                        DeprecationWarning,
                    )
                    futures = [
                        pool.submit(_execute_forked, i) for i in range(len(pending))
                    ]
            else:
                pool = concurrent.futures.ThreadPoolExecutor(workers)
                futures = [pool.submit(_execute, job) for job in pending]
            with pool:
                for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    record(i, future.result())
    finally:
        _JOBS.clear()
        conn.close()

    if verbose:
        print('\n' + report(db_path, run_id))
    return run_id


def load_solution(db_path: Path | str, row_id: int) -> tuple[nx.Graph, SiteBundle]:
    """Rebuild the topology ``S`` recorded in a result row, with its site.

    ``S`` has loads, ``capacity`` and ``topology`` set, so the routeset can be
    reproduced with ``PathFinder(G_from_S(S, bundle.A), bundle.P,
    bundle.A).create_detours()`` and plotted with ``gplot()``.

    Args:
        db_path: SQLite file written by :func:`run_sweep`.
        row_id: ``results.id`` of the row.

    Returns:
        ``S`` and a private copy of the site bundle (``L``, ``P``, ``A``).

    Raises:
        ValueError: If the row has no link bits (errored task, or ``S`` used
            links outside the site's ``A``) or the site's available links
            changed since the row was recorded.
    """
    conn = _connect(db_path)
    row = conn.execute('SELECT * FROM results WHERE id = ?', (row_id,)).fetchone()
    conn.close()
    if row is None or row['linkbits'] is None:
        raise ValueError(f'results row {row_id} has no recorded solution')
    bundle = get_bundle(row['site'], copy=True, read_powers=False)
    A = bundle.A
    if linkset_id(A) != row['linkset_id']:
        raise ValueError(f'available links of {row["site"]!r} changed since the run')
    S = S_from_linkbits(deserialize(row['linkbits']), A)
    topology = Topology(json.loads(row['extras']).get('topology', 'branched'))
    nx.set_node_attributes(S, nonunit_inflow(A), 'inflow')
    if topology is Topology.RINGED:
        split_rings_and_calc_loads(S, A)
    else:
        calcload(S)
    S.graph.update(topology=topology, capacity=row['capacity'])
    return S, bundle


def _spread(rows: list[dict[str, Any]], column: str) -> str | None:
    """Return the median over cases of the reps' relative range, as a % string."""
    by_case = defaultdict(list)
    for r in rows:
        if r['status'] == 'ok':
            by_case[r['site'], r['capacity'], r['options']].append(r[column])
    spreads = [
        (max(values) - min(values)) / statistics.median(values)
        for values in by_case.values()
        if len(values) > 1
    ]
    return f'{100 * statistics.median(spreads):.4g}' if spreads else None


def report(
    db_path: Path | str,
    runs: str | Sequence[str] | None = None,
    *,
    baseline: str | None = None,
    per_case: bool = False,
) -> str:
    """Tabulate results per variant, compared with a baseline variant.

    Given several runs, variants are labeled ``'run_id:variant'``, which
    compares code states: e.g. a run before and a run after a library edit.
    ``Δlength`` is the mean relative length difference to the baseline over the
    tasks (same site, capacity, options and rep) where both are ``ok``;
    ``better``/``worse`` count those tasks by the sign of the difference.

    With repetitions, ``time_spread_%`` and ``length_spread_%`` show the
    run-to-run noise: the median over cases of the range of the ``ok`` reps
    relative to their median. Differences between variants smaller than the
    spread are not meaningful.

    Args:
        db_path: SQLite file written by :func:`run_sweep`.
        runs: Run or runs to report; defaults to the latest run.
        baseline: Variant label to compare against; defaults to the first
            variant given to the first run.
        per_case: Add a table of median length and time per case and variant.

    Returns:
        GitHub-flavored Markdown tables.
    """
    conn = _connect(db_path)
    if runs is None:
        latest = conn.execute(
            'SELECT run_id FROM runs ORDER BY started DESC LIMIT 1'
        ).fetchone()
        if latest is None:
            return 'no runs recorded'
        runs = [latest['run_id']]
    elif isinstance(runs, str):
        runs = [runs]
    rows, headers, labels = [], [], []
    parallel = False
    for run_id in runs:
        run = conn.execute('SELECT * FROM runs WHERE run_id = ?', (run_id,)).fetchone()
        if run is None:
            raise ValueError(f'unknown run {run_id!r}')
        headers.append(
            f'run {run_id}: code {run["code"]}, {run["executor"]}, '
            f'{run["workers"]} workers x {run["threads"]} threads'
        )
        parallel |= run['workers'] > 1
        labels.extend(
            v if len(runs) == 1 else f'{run_id}:{v}'
            for v in json.loads(run['variants'])
        )
        for r in conn.execute(
            'SELECT * FROM results WHERE run_id = ? ORDER BY id', (run_id,)
        ):
            label = r['variant'] if len(runs) == 1 else f'{run_id}:{r["variant"]}'
            rows.append({**dict(r), 'label': label})
    conn.close()
    if not rows:
        return '\n'.join(headers) + '\n\nno results'

    labels = [label for label in labels if any(r['label'] == label for r in rows)]
    baseline = baseline or labels[0]
    headers.append(f'baseline {baseline!r}')
    if parallel:
        headers.append(
            'note: concurrent tasks compete for memory bandwidth and boost clocks; '
            'confirm small time differences with workers=1'
        )
    ok_length = {
        (r['label'], r['site'], r['capacity'], r['options'], r['rep']): r['length']
        for r in rows
        if r['status'] == 'ok'
    }

    table = []
    for label in labels:
        lrows = [r for r in rows if r['label'] == label]
        times = [r['time_s'] for r in lrows if r['time_s'] is not None]
        rel = [
            length / ok_length[(baseline, *key[1:])] - 1
            for key, length in ok_length.items()
            if key[0] == label and (baseline, *key[1:]) in ok_length
        ]
        table.append(
            {
                'variant': label,
                'tasks': len(lrows),
                'ok': sum(r['status'] == 'ok' for r in lrows),
                'invalid': sum(r['status'] == 'invalid' for r in lrows),
                'error': sum(r['status'] == 'error' for r in lrows),
                'median_s': f'{statistics.median(times):.4g}' if times else None,
                'total_s': f'{sum(times):.4g}',
                'Δlength_%': f'{100 * statistics.mean(rel):.4g}' if rel else None,
                'better': sum(x < -1e-9 for x in rel),
                'worse': sum(x > 1e-9 for x in rel),
                'time_spread_%': _spread(lrows, 'time_s'),
                'length_spread_%': _spread(lrows, 'length'),
            }
        )
    for column in ('time_spread_%', 'length_spread_%'):
        if all(entry[column] is None for entry in table):
            for entry in table:
                del entry[column]
    out = [
        '\n'.join(headers),
        Table.from_dicts(table).to_markdown(),
    ]

    if per_case:
        cases = list(
            dict.fromkeys((r['site'], r['capacity'], r['options']) for r in rows)
        )
        case_table = []
        for site, capacity, options in cases:
            entry: dict[str, Any] = {'site': site, 'cap': capacity}
            if options != '{}':
                entry['options'] = options
            for label in labels:
                crows = [
                    r
                    for r in rows
                    if (r['label'], r['site'], r['capacity'], r['options'])
                    == (label, site, capacity, options)
                ]
                lengths = [r['length'] for r in crows if r['status'] == 'ok']
                times = [r['time_s'] for r in crows if r['time_s'] is not None]
                entry[f'{label} length'] = (
                    f'{statistics.median(lengths):.6g}' if lengths else None
                )
                entry[f'{label} s'] = (
                    f'{statistics.median(times):.6g}' if times else None
                )
            case_table.append(entry)
        out.append(Table.from_dicts(case_table).to_markdown())

    failed = [r for r in rows if r['status'] != 'ok']
    if failed:
        out.append(f'{len(failed)} tasks not ok; first ones:')
        out.extend(
            f'- [id {r["id"]}] {r["site"]} cap={r["capacity"]} {r["label"]} '
            f'rep={r["rep"]} [{r["status"]}]: {r["error"].strip().splitlines()[-1]}'
            for r in failed[:5]
        )
    return '\n\n'.join(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Report the results of sweeps.')
    parser.add_argument('db_path', type=Path)
    parser.add_argument('runs', nargs='*', help='defaults to the latest run')
    parser.add_argument('--baseline', help='defaults to the first variant')
    parser.add_argument('--per-case', action='store_true')
    args = parser.parse_args()
    print(
        report(
            args.db_path,
            args.runs or None,
            baseline=args.baseline,
            per_case=args.per_case,
        )
    )
