# Figure data

This directory contains the precomputed router results used by `../figures.py` and the scripts that generate them.

The directory is excluded from the Sphinx build through `exclude_patterns` in `../conf.py` and from documentation checks through `NON_PAGE_DIRS` in `../check_docs.py`.

## Contents

| File | Purpose |
| --- | --- |
| `routers_problem.py` | Defines the problem and shared data operations. |
| `routers_solve_fast.py` | Runs the constructive-heuristic and meta-heuristic routers. |
| `routers_solve_to_optimum.py` | Solves (MILP) to the proven optimum and records it. |
| `routers_solve_and_log.py` | Solves (MILP) to a small gap and records its time series. |
| `routers_summary.pkl` | Stores the problem, observations, two topologies, and optimum. |
| `routers_timeseries.npz` | Stores the MILP bound and incumbent series. |

Each heuristic run stores only `(elapsed time, objective length)`. Two site-bound `TerseLinks` values are retained: the minimum-length topology for the optimality proof and the 0.1-second HGS topology for the MILP time series.

The MILP bound and incumbent are stored as two independent series, `<name>_time` and `<name>`, because HiGHS reports them on different occasions: the bound is polled densely by the interrupt and logging callbacks, while the incumbent has one entry per improvement the solver announced.

Lengths are stored in metres. `../figures.py` computes all percentages relative to the proven optimum. Both artifacts fingerprint the site geometry and complete experiment configuration, and their loaders reject data for a different problem. The NumPy archive contains plain arrays and loads with `allow_pickle=False`.

## Regenerating

Run the scripts in order from the repository root:

```sh
python -m docs.figuredata.routers_solve_fast
python -m docs.figuredata.routers_solve_to_optimum
python -m docs.figuredata.routers_solve_and_log
```

`routers_solve_to_optimum.py` requires a Gurobi licence, and `routers_solve_and_log.py` requires `highspy`.

The documentation build uses the committed artifacts and does not run the routers.

## Changing the problem

Edit the problem definition in `routers_problem.py`, then run the three generation scripts above. The figure obtains its problem description from that definition, so no corresponding prose changes should be necessary.

Running the fast solves deliberately removes the old optimum and time series; recording a new optimum removes the old time series. This prevents downstream data from silently surviving a changed input.

Results produced with wall-clock limits may vary between runs, even when the seed and time limit are unchanged. The committed heuristic data therefore represents one observation; the recorded optimum is solver-certified.
