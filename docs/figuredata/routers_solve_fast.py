"""Solve the configured problem with the fast routers used by the figure.

The constructive and meta-heuristic samples cover the millisecond-to-second
range preceding the MILP time series. ``problem.EW_METHODS`` and
``problem.HGS_TIME_LIMITS`` define the sampled variants and time limits.

This script generates the initial data without requiring a MILP solver or proven
optimum. Each run contributes only its elapsed time and objective length. The
two retained topologies serve distinct purposes: the shortest result starts the
optimality proof, while the 0.1-second HGS result starts the MILP time series.

Usage::

    python -m docs.figuredata.routers_solve_fast
"""

from optiwindnet.baselines.hgs import hgs_cvrp
from optiwindnet.heuristics import constructor
from optiwindnet.interarraylib import G_from_S, as_normalized
from optiwindnet.terse import TerseLinks

from . import routers_problem as problem


def observation(S, A) -> tuple[float, float]:
    """Return only the elapsed time and MILP-comparable objective length."""
    return float(S.graph['runtime']), G_from_S(S, A).size(weight='length')


def main() -> None:
    L, _P, A = problem.setup()

    observations = {}
    solutions = {}

    def retain(key, S) -> None:
        observations[key] = observation(S, A)
        solutions[key] = S

    for method in problem.EW_METHODS:
        S = constructor(A, capacity=problem.CAPACITY, method=method)
        retain(method, S)
    for time_limit in problem.HGS_TIME_LIMITS:
        # Match `HGSRouter` by normalizing the graph as recommended by `hgs_cvrp()`.
        S = hgs_cvrp(
            as_normalized(A),
            capacity=problem.CAPACITY,
            time_limit=time_limit,
            seed=problem.HGS_SEED,
        )
        key = f'hgs_cvrp:{time_limit:g}'
        retain(key, S)

    best_key = min(observations, key=lambda key: observations[key][1])
    digest = problem.nodeset_digest(L)
    best_topology = TerseLinks.from_topology(solutions[best_key], nodeset_digest=digest)
    timeseries_topology = TerseLinks.from_topology(
        solutions[problem.MILP_TIMESERIES_WARMSTART], nodeset_digest=digest
    )
    problem.save_summary(
        {
            'problem': problem.problem_id(),
            'observations': observations,
            'best_topology': best_topology,
            'timeseries_topology': timeseries_topology,
        }
    )
    print(f'wrote {problem.SUMMARY.name}: {len(observations)} observations')
    for key, (elapsed, length) in observations.items():
        roles = []
        if key == best_key:
            roles.append('optimality proof')
        if key == problem.MILP_TIMESERIES_WARMSTART:
            roles.append('MILP time series')
        marker = f' ({", ".join(roles)})' if roles else ''
        print(f'  {key:20s} {elapsed:8.3f} s  {length:12.1f} m{marker}')


if __name__ == '__main__':
    main()
