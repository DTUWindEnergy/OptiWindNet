"""Record the MILP incumbent and bound time series from an early warm start.

HiGHS runs to ``problem.TARGET_GAP`` from the stored 0.1-second HGS topology.
Incumbent and bound values come from HiGHS MIP callbacks and are stored as two
independent series: the bound is polled densely by the interrupt callback, so it
progresses in many small steps, while the incumbent holds one entry per announced
improvement. Reusing that exact topology aligns the MILP time series with its
figure marker.

Lengths are stored in metres without normalization. The figure computes
percentages against the current proven optimum.

Usage::

    python -m docs.figuredata.routers_solve_and_log [--time-limit 1800]
"""

import argparse
import time

from optiwindnet.MILP import ModelOptions, solver_factory

from . import routers_problem as problem

#: One sample: elapsed solver time and the reported length in metres.
Sample = tuple[float, float]


def subscribe(solver) -> tuple[list[Sample], list[Sample]]:
    """Attach HiGHS MIP callbacks to `solver`, returning their sample lists.

    Pyomo exposes no progress API, so this reaches the underlying ``highspy``
    model through its persistent-solver wrapper. The model is only built by
    ``set_instance()``, hence the explicit call: the later ``solve()`` then takes
    its update path and keeps the callbacks.

    The two quantities are collected apart, from the callbacks that report them:
    the dual bound from the interrupt callback, which is polled throughout the
    search, and from the logging callback, which fires once per MIP log line; the
    incumbent from the improving-solution callback, which fires on each new
    solution and nowhere else.

    Returns:
      The dual-bound samples and the incumbent improvements, both filled in
      during ``solve()``.
    """
    backend = solver.solver
    backend.set_instance(solver.model)
    highs = backend._solver_model
    if highs is None:
        raise SystemExit("pyomo's HiGHS wrapper no longer exposes the highspy model")
    bound: list[Sample] = []
    incumbent: list[Sample] = []

    def sampler(sink: list[Sample], quantity: str):
        def sample(event) -> None:
            out = event.data_out
            sink.append((out.running_time, getattr(out, quantity)))

        return sample

    highs.cbMipInterrupt.subscribe(sampler(bound, 'mip_dual_bound'))
    highs.cbMipLogging.subscribe(sampler(bound, 'mip_dual_bound'))
    highs.cbMipImprovingSolution.subscribe(sampler(incumbent, 'mip_primal_bound'))
    return bound, incumbent


def condense(samples: list[Sample]) -> list[Sample]:
    """Order the bound samples by time and drop those that carry no new value.

    Two callbacks poll the same dual bound, so the samples interleave and repeat.
    A sample is kept when it first reaches a new value, which is what a step plot
    needs, plus the final one, which sets the end of the series. Samples taken
    before the root relaxation has a bound are dropped.
    """
    condensed: list[Sample] = []
    previous: float | None = None
    for sample in sorted(samples):
        _, value = sample
        if value == float('-inf'):
            continue
        if value != previous:
            condensed.append(sample)
            previous = value
    if condensed and condensed[-1] != (last := max(samples)):
        condensed.append(last)
    return condensed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--time-limit', type=float, default=1800.0)
    args = parser.parse_args()

    _, P, A = problem.setup()
    data = problem.load_summary()
    optimum = data['optimum']
    warm_key = problem.MILP_TIMESERIES_WARMSTART
    warm_len = data['observations'][warm_key][1]
    S_warm = data['timeseries_topology'].to_topology(
        capacity=problem.CAPACITY, creator='figuredata-timeseries'
    )
    print(
        f'warm start: {warm_key} -> {warm_len:.1f} m '
        f'({warm_len / optimum - 1:+.3%} vs optimum, topology)'
    )

    solver = solver_factory('highs')
    solver.set_problem(
        P,
        A,
        capacity=problem.CAPACITY,
        model_options=ModelOptions(**problem.MODEL_OPTIONS),
        warmstart=S_warm,
    )
    samples, improvements = subscribe(solver)
    print(f'solving to a proven {problem.TARGET_GAP:.1%} gap')
    t0 = time.perf_counter()
    info = solver.solve(
        time_limit=args.time_limit, mip_gap=problem.TARGET_GAP, verbose=True
    )
    wall = time.perf_counter() - t0
    # `improvements` is stored as it came: every announced improvement is kept,
    # including one that repeats the preceding length.
    bound = condense(samples)
    if not bound or not improvements:
        raise SystemExit('HiGHS reported no MIP progress callbacks')

    print(
        f'termination: {info.termination}\n'
        f'objective:   {info.objective:.1f} m '
        f'({info.objective / optimum - 1:+.4%} vs optimum)\n'
        f'bound:       {info.bound:.1f} m\n'
        f'rel. gap:    {info.relgap:.4%}\n'
        f'solver time: {info.runtime:.1f} s (wall {wall:.1f} s)'
    )

    problem.save_timeseries(bound, improvements)
    print(
        f'wrote {problem.TIMESERIES.name}: {len(bound)} bound points out of '
        f'{len(samples)} samples over {bound[-1][0]:.1f} s, '
        f'{len(improvements)} incumbent improvements'
    )
    for seconds, length in improvements:
        print(f'  {seconds:7.1f} s  {length:12.1f} m  {length / optimum - 1:+.4%}')


if __name__ == '__main__':
    main()
