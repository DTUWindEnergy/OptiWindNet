# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Tests for the sweep harness."""

import json
import sqlite3
import time
from pathlib import Path
from typing import Literal

import pytest

from optiwindnet.converting import G_from_S
from optiwindnet.heuristics import constructor
from optiwindnet.pathfinding import PathFinder

from . import sweep
from .sitecache import location_repository
from .sweep import (
    EASY_CASES,
    HARD_CASES,
    MEDIUM_CASES,
    Task,
    load_solution,
    report,
    run_sweep,
)

CASES = [('toy', 5), ('example_location', 3, {'method': 'rootlust'})]


def ew(t: Task):
    return constructor(t.A, t.capacity, method=t.options.get('method', 'esau_williams'))


def ew_routed(t: Task):
    return PathFinder(G_from_S(ew(t), t.A), t.P, t.A).create_detours()


def failing(t: Task):
    raise RuntimeError('intentional')


def sleeping(t: Task):
    time.sleep(10)
    return ew(t)


def _rows(db_path: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT * FROM results ORDER BY id').fetchall()
    conn.close()
    return rows


def test_suites_use_repository_handles() -> None:
    handles = set(location_repository()._fields)
    for site, _ in EASY_CASES + MEDIUM_CASES + HARD_CASES:
        assert site in handles


def test_inline_sweep_records_and_resumes(tmp_path: Path) -> None:
    db_path = tmp_path / 'sweep.sqlite'
    variants = {'ew': ew, 'routed': ew_routed, 'failing': failing}
    run_sweep(CASES, variants, db_path, run_id='r', reps=2, verbose=False)

    rows = _rows(db_path)
    assert len(rows) == 12
    by_variant = {v: [r for r in rows if r['variant'] == v] for v in variants}
    for r in by_variant['ew'] + by_variant['routed']:
        assert r['status'] == 'ok'
        assert r['length'] > 0
        assert r['violations'] == 0
    assert json.loads(by_variant['ew'][0]['extras'])['creator'] == 'constructor'
    assert json.loads(by_variant['ew'][2]['options']) == {'method': 'rootlust'}
    # S and its routed G measure the same length and record the same topology
    for r_S, r_G in zip(by_variant['ew'], by_variant['routed'], strict=True):
        assert r_S['length'] == pytest.approx(r_G['length'])
        assert r_S['linkbits'] == r_G['linkbits']
    S, bundle = load_solution(db_path, by_variant['ew'][2]['id'])
    G = PathFinder(G_from_S(S, bundle.A), bundle.P, bundle.A).create_detours()
    assert G.size(weight='length') == pytest.approx(by_variant['ew'][2]['length'])
    for r in by_variant['failing']:
        assert r['status'] == 'error'
        assert 'RuntimeError: intentional' in r['error']

    text = report(db_path, baseline='ew', per_case=True)
    assert '| failing' in text
    assert 'time_spread_%' in text
    assert 'RuntimeError: intentional' in text

    # resuming keeps finished rows and retries only the errors
    ok_ids = {r['id'] for r in rows if r['status'] == 'ok'}
    run_sweep(CASES, variants, db_path, run_id='r', reps=2, verbose=False)
    rows = _rows(db_path)
    assert len(rows) == 12
    assert ok_ids == {r['id'] for r in rows if r['status'] == 'ok'}

    with pytest.raises(ValueError, match='changed'):
        run_sweep(CASES, {'ew': ew_routed}, db_path, run_id='r', verbose=False)


@pytest.mark.parametrize('executor', ['process', 'thread'])
def test_concurrent_sweep(
    tmp_path: Path, executor: Literal['process', 'thread']
) -> None:
    db_path = tmp_path / 'sweep.sqlite'
    method = 'esau_williams'

    def closure(t: Task):
        # closures are not picklable; the process executor must still run them
        assert t.threads == 1
        return constructor(t.A, t.capacity, method=method)

    def slow(t: Task):
        time.sleep(0.1)
        return ew(t)

    # the slow variant finishes last, yet remains the default baseline
    run_sweep(
        CASES,
        {'slow': slow, 'closure': closure},
        db_path,
        reps=2,
        workers=2,
        threads=1,
        executor=executor,
        verbose=False,
    )
    rows = _rows(db_path)
    assert len(rows) == 8
    assert all(r['status'] == 'ok' for r in rows), [r['error'] for r in rows]
    assert "baseline 'slow'" in report(db_path)


def test_report_compares_runs_and_refuses_code_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / 'sweep.sqlite'

    def rootlust(t: Task):
        return constructor(t.A, t.capacity, method='rootlust')

    run_sweep(CASES[:1], {'ew': ew}, db_path, run_id='before', verbose=False)
    run_sweep(CASES[:1], {'ew': rootlust}, db_path, run_id='after', verbose=False)
    text = report(db_path, ['before', 'after'])
    assert "baseline 'before:ew'" in text
    assert '| after:ew' in text

    monkeypatch.setattr(sweep, '_code_state', lambda: 'edited')
    with pytest.raises(ValueError, match='code changed'):
        run_sweep(CASES[:1], {'ew': ew}, db_path, run_id='before', verbose=False)


@pytest.mark.parametrize('workers', [1, 2])
def test_timeout(tmp_path: Path, workers: int) -> None:
    db_path = tmp_path / 'sweep.sqlite'
    t0 = time.perf_counter()
    run_sweep(
        CASES,
        {'sleeping': sleeping},
        db_path,
        workers=workers,
        timeout=0.2,
        verbose=False,
    )
    assert time.perf_counter() - t0 < 5
    rows = _rows(db_path)
    assert len(rows) == 2
    assert all('TimeoutError' in r['error'] for r in rows)
