"""Generate the golden data consumed by the end-to-end tests.

Writes ``tests/solutions.pkl`` with two sections:

``snapshots``
    Exact ``terse_links`` for the deterministic (EWRouter) cases in
    ``matrix.build_matrix()``, stored as ``(topology, links)`` tuples. These are
    compared byte-for-byte by ``test_snapshot_matches``.

``ref_lengths``
    Solver-independent reference optima keyed by ``matrix.problem_key`` -- the
    objective length of a MILP problem whenever *some* available solver proved
    it optimal during generation. ``test_milp_solution_properties`` uses these
    for a gap-tolerant length-regression check; they are optional, so problems
    no available solver could prove (large sites) are simply absent and only
    their structural properties are checked.

MILP property cases need no snapshot: their correctness is self-validating
(validity + topology shape + capacity + feeder/balance constraints), so this
generator only has to record the reference optima it can obtain. Solvers that
are missing or unlicensed are skipped, not fatal -- run it in a fuller solver
environment to populate more references.

'ortools*' solves are dispatched to the shared ``ortools_worker`` subprocess
(see tests/isolation.py); every other solver runs in-process.
"""

import pickle

from optiwindnet.api import WindFarmNetwork

from . import isolation, matrix, paths, sites
from .helpers import (
    needs_process_isolation,
    router_factory,
    solve_milp_property_metrics,
    solver_unavailable,
)


def print_header(title: str) -> None:
    print('\n' + '=' * 10)
    print(title)
    print('=' * 10)


def generate() -> dict:
    cases = matrix.build_matrix()
    locations = sites.load_locations()

    snapshots: dict = {}
    ref_lengths: dict = {}

    snapshot_cases = [(s, sp) for s, sp in cases
                      if sp.get('determinism') == 'snapshot']
    milp = [(s, sp) for s, sp in cases if sp['class'] == 'MILPRouter']
    n_property = sum(1 for _, sp in cases if sp.get('determinism') == 'property')

    # --- deterministic snapshots (EWRouter forest topologies) ----------------
    print_header(f'Deterministic snapshots ({len(snapshot_cases)})')
    for i, (site, spec) in enumerate(snapshot_cases, 1):
        key = matrix.case_key(site, spec)
        L = getattr(locations, site)
        wfn = WindFarmNetwork(L=L, cables=spec['cables'])
        terse = wfn.optimize(router=router_factory(spec))
        snapshots[key] = {
            'location': site,
            'router_spec': spec,
            'terse_links': (terse.topology.value, tuple(terse.tolist())),
        }
        print(f'[{i}/{len(snapshot_cases)}] {key}')

    # --- MILP reference optima (solver-independent, best-effort) --------------
    print_header(f'MILP reference optima ({len(milp)} cases)')
    ortools_worker = isolation.ortools_worker_factory()
    try:
        for i, (site, spec) in enumerate(milp, 1):
            pkey = matrix.problem_key(site, spec)
            if pkey in ref_lengths:
                continue  # optimum already recorded by another solver
            L = getattr(locations, site)
            solver_name = spec['params']['solver_name']
            if needs_process_isolation(spec):
                timeout = spec['params'].get('time_limit', 10) + 30
                result = ortools_worker.run(
                    solve_milp_property_metrics, (spec, L), timeout
                )
            else:
                try:
                    result = solve_milp_property_metrics(spec, L)
                except BaseException as exc:  # noqa: BLE001
                    result = exc
            if isinstance(result, BaseException):
                if solver_unavailable(result):
                    print(f'[{i}/{len(milp)}] skip {solver_name}: unavailable')
                    continue
                raise result
            _metrics, termination, length = result
            status = termination
            if termination == 'OPTIMAL':
                ref_lengths[pkey] = length
                status += f' -> ref {length:.1f}'
            print(f'[{i}/{len(milp)}] {matrix.case_key(site, spec)}: {status}')
    finally:
        ortools_worker.shutdown()

    # --- property cases (HGS, ringed EWRouter): self-validating, no golden data
    print_header(f'Property-only cases ({n_property}) -- no golden data recorded')

    return {'snapshots': snapshots, 'ref_lengths': ref_lengths}


if __name__ == '__main__':
    print_header('Generating end-to-end expected values...')
    data = generate()
    with paths.SOLUTIONS_FILE.open('wb') as f:
        pickle.dump(data, f)
    print_header(
        f'Saved {len(data["snapshots"])} snapshots and '
        f'{len(data["ref_lengths"])} reference optima to {paths.SOLUTIONS_FILE}'
    )
