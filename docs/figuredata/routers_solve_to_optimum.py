"""Solve the configured problem to proven optimality.

Gurobi solves to zero gap with ``mipfocus=2``, using the shortest stored
heuristic topology as a warm start. Only the certified objective is stored; the
optimal network itself is not needed by the figure.

Because the optimum is the denominator used by the figure, a result is written
only after Gurobi proves optimality at zero gap.

Usage::

    python -m docs.figuredata.routers_solve_to_optimum
    python -m docs.figuredata.routers_solve_to_optimum --dry-run
"""

import argparse
import time

from optiwindnet.MILP import ModelOptions, solver_factory

from . import routers_problem as problem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--time-limit', type=float, default=3600.0)
    parser.add_argument('--mip-focus', type=int, default=2)
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument(
        '--dry-run', action='store_true', help='build the model and stop before solving'
    )
    args = parser.parse_args()

    _, P, A = problem.setup()
    data = problem.load_summary()
    warm_key = min(data['observations'], key=lambda key: data['observations'][key][1])
    warm_len = data['observations'][warm_key][1]
    S_warm = data['best_topology'].to_topology(
        capacity=problem.CAPACITY, creator='figuredata-best'
    )
    print(f'warm start: {warm_key} at {warm_len:.1f} m')

    solver = solver_factory('gurobi')
    # Set options before `set_problem()`, which creates the persistent Gurobi
    # environment. Copy the defaults to avoid mutating the class-level mapping.
    solver.options = dict(solver.options, mipfocus=args.mip_focus)
    if args.threads is not None:
        solver.options['threads'] = args.threads
    print(f'gurobi options: {solver.options}')

    solver.set_problem(
        P,
        A,
        capacity=problem.CAPACITY,
        model_options=ModelOptions(**problem.MODEL_OPTIONS),
        warmstart=S_warm,
    )
    if args.dry_run:
        print('dry run: model built and warm-started; not solving.')
        return

    t0 = time.perf_counter()
    info = solver.solve(time_limit=args.time_limit, mip_gap=0.0, verbose=True)
    wall = time.perf_counter() - t0
    print(
        f'\ntermination: {info.termination}\n'
        f'objective:   {info.objective:.4f} m\n'
        f'bound:       {info.bound:.4f} m\n'
        f'rel. gap:    {info.relgap:.6%}\n'
        f'solver time: {info.runtime:.1f} s (wall {wall:.1f} s)'
    )
    if info.termination != 'optimal':
        raise SystemExit(
            f'terminated as {info.termination!r}, not proven optimal; '
            f'{problem.SUMMARY.name} left untouched.'
        )

    data['optimum'] = info.objective
    problem.save_summary(data)
    print(f'wrote {problem.SUMMARY.name}: {info.objective:.4f} m, proven')


if __name__ == '__main__':
    main()
