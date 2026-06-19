# feat/continuous_power_flow

## 2026-06-09

- Added opt-in `ModelOptions(continuous_power_flow=True)` support for OR-Tools MathOpt backends that support continuous variables (`ortools.highs`, `ortools.gscip`).
- Kept default MILP flow behavior integer-scaled and unchanged.
- High-level `WindFarmNetwork` now preserves nominal float `turbine_power` values when the router model option enables continuous power flow; otherwise it keeps the existing integer normalization path.
- OR-Tools CP-SAT, Pyomo, and SCIP backends now reject `continuous_power_flow=True` explicitly.
- `assign_cables()` now handles float edge loads by choosing the first cable capacity that can carry the load.
- `assign_cables()` also tolerates tiny floating-point capacity overshoots from continuous solvers, e.g. `5.000000000000169` for a capacity-5 cable.
- Added focused tests for model option handling, OR-Tools continuous bounds, unsupported backend rejection, high-level power storage, and float-load cable assignment.
- Configured `artifacts/timing_benchmark.py` for the continuous benchmark variant:
  `MODEL_VARIANT='continuous_power_flow_v1'`, `SOLVER_NAME='ortools.highs'`, and
  `ModelOptions(continuous_power_flow=True)`.
- Cleaned continuous benchmark diagnostics in `artifacts/timing_benchmark.py`:
  schema version 5, explicit `nominal_*` fields, explicit `integer_*` baseline
  fields, and solution `max_load_minus_capacity`.
- Added benchmark controls in `artifacts/timing_benchmark.py`:
  `OWN_BENCH_SITES`, `OWN_BENCH_CABLES`, `OWN_BENCH_CASE_REGEX`,
  `OWN_BENCH_CASE_LIMIT`, and `OWN_BENCH_SOLVERS`. Default timing cables are
  now only `5` and `10`; multi-solver runs store backend-specific case keys.

Verification:

- `pytest -q tests/test_MILP.py::test_model_options_accepts_continuous_power_flow tests/test_MILP.py::test_ortools_continuous_power_flow_uses_nominal_bounds tests/test_MILP.py::test_ortools_cp_sat_rejects_continuous_power_flow tests/test_MILP.py::test_pyomo_rejects_continuous_power_flow tests/test_api_WindFarmNetwork.py::test_turbine_power_default_path_is_integer_scaled tests/test_api_WindFarmNetwork.py::test_continuous_power_flow_keeps_nominal_turbine_power tests/test_interarraylib.py::test_assign_cables tests/test_interarraylib.py::test_assign_cables_accepts_float_loads`
- `pytest -q tests/test_interarraylib.py::test_as_single_root_no_root_in_border`
- `ruff check optiwindnet/MILP/_core.py optiwindnet/MILP/_core.pyi optiwindnet/MILP/ortools.py optiwindnet/MILP/pyomo.py optiwindnet/MILP/scip.py optiwindnet/api.py optiwindnet/interarraylib.py tests/test_MILP.py tests/test_api_WindFarmNetwork.py tests/test_interarraylib.py`
- `python -m py_compile optiwindnet/MILP/_core.py optiwindnet/MILP/ortools.py optiwindnet/MILP/pyomo.py optiwindnet/MILP/scip.py optiwindnet/api.py optiwindnet/interarraylib.py`
- Tiny `ortools.highs` smoke solve with `turbine_power=[1.0, 1.5]` returned `OPTIMAL`, float topology loads `[1.5, 2.5]`, and `continuous_power_flow=True` in graph method options.
- Reproduced and fixed Ormonde false cable-capacity errors caused by solver tolerance; rerun with `OWN_BENCH_RETRY_ERRORS=1` to replace the error entries.
- Reproduced and fixed Horns Rev 3 false detour load assertion errors caused by
  floating-point load propagation (`math.isclose` tolerance in detour/tentative
  hook load checks).
- Horns Rev 3 hard-case backend comparison (`ortools.highs` vs `ortools.gscip`)
  showed no advantage for `gscip`: the near-uniform hard cases still hit the
  60s limit, and `highs` had slightly better gaps.

## 2026-06-16

- Extended `continuous_power_flow=True` support to the Pyomo model, so Pyomo/Gurobi
  can use continuous nominal-power flow variables instead of only the integer-scaled
  workaround.
- Updated the Gurobi persistent wrapper so binary link values are rounded, while
  continuous flow values are preserved as floats.
- Added high-level API rounding for `turbine_power`: values are rounded before
  optimization and a warning is emitted when inputs are changed.
- Added configurable `WindFarmNetwork(..., turbine_power_decimals=1)` support.
  The value is the number of decimal places to keep before optimization. Use
  `turbine_power_decimals=2` for two decimal digits or `10` for ten.
- Fixed integer-scaled high-level results so public `S` and `G` graphs are converted
  back to nominal units consistently. This now covers edge loads, node loads,
  node powers, `capacity`, `max_load`, and stored cable capacities.
- Fixed router switching with `turbine_power`: `WindFarmNetwork.optimize(router=...)`
  now reapplies either nominal powers or integer-scaled powers according to the
  selected router's `continuous_power_flow` option.
- Added focused tests for Pyomo continuous bounds, Gurobi float flow extraction,
  API turbine-power rounding, nominal-unit result graphs, and router switching.
- Added and iterated on `artifacts/gurobi_power_flow_compare.py` for paired
  Gurobi benchmarks comparing integer-scaled power flow against continuous
  Pyomo/Gurobi flow. The script records wall/setup/solver timing, supports repeats,
  alternates variant order, and prints paired summaries.
- Added `OWN_COMPARE_TURBINE_POWER_DECIMALS` and
  `OWN_BENCH_TURBINE_POWER_DECIMALS` benchmark controls so benchmark diagnostics
  and `WindFarmNetwork` construction use the same decimal-place setting.
- Updated benchmark resume metadata after the rounding semantic change:
  `artifacts/gurobi_power_flow_compare.py` now records
  `rounding_semantics=decimal_places_v1`, and `artifacts/timing_benchmark.py`
  records the same metadata.
- Extended `artifacts/gurobi_power_flow_compare.py` site selection with
  `OWN_COMPARE_SITES=all`, `OWN_COMPARE_SKIP_SITES`, `OWN_COMPARE_SITE_REGEX`,
  `OWN_COMPARE_SITE_LIMIT`, and `OWN_COMPARE_SITE_OFFSET` for broader sampled
  runs.

Audit findings:

- The Pyomo/Gurobi continuous MILP equations now match the OR-Tools continuous
  formulation at the important modeling points: real flow variables, destination
  power based terminal-arc capacity, total-power feeder lower bound, and float
  solution extraction.
- The main correctness issue found during audit was not the MILP formulation but
  high-level unit conversion after integer-scaled solves. Public graphs could
  previously expose mixed scaled/nominal attributes; this is now fixed.
- Gurobi/Pyomo can report solver termination `"error"` while still exposing a
  solution. Benchmark summaries should exclude or separately report those rows
  unless explicitly investigating solver-return behavior.

Verification:

- `pytest -q tests/test_api_WindFarmNetwork.py`
- `pytest -q tests/test_MILP.py tests/test_api_WindFarmNetwork.py tests/test_interarraylib.py::test_assign_cables_accepts_float_loads tests/test_interarraylib.py::test_assign_cables_accepts_tiny_capacity_overshoot`
- `ruff check optiwindnet/api.py optiwindnet/MILP/pyomo.py optiwindnet/MILP/gurobi.py optiwindnet/MILP/ortools.py optiwindnet/interarraylib.py tests/test_api_WindFarmNetwork.py tests/test_MILP.py tests/test_interarraylib.py`
- `ruff format --check optiwindnet/api.py optiwindnet/MILP/pyomo.py optiwindnet/MILP/gurobi.py tests/test_api_WindFarmNetwork.py tests/test_MILP.py`
- `pytest -q tests/test_api_WindFarmNetwork.py`
- `ruff check optiwindnet/api.py tests/test_api_WindFarmNetwork.py artifacts/gurobi_power_flow_compare.py artifacts/timing_benchmark.py`
- `ruff format --check optiwindnet/api.py tests/test_api_WindFarmNetwork.py artifacts/gurobi_power_flow_compare.py artifacts/timing_benchmark.py`
- `python -m py_compile optiwindnet/api.py artifacts/gurobi_power_flow_compare.py artifacts/timing_benchmark.py`
- Tiny solve invariant checks for `ortools.highs` and `gurobi`, each in both
  integer-scaled and continuous modes, confirmed public `S` and `G` expose
  nominal loads/capacity/cables consistently.
- After clarifying turbine-power rounding as decimal places:
  `pytest -q tests/test_api_WindFarmNetwork.py` passed 63 tests;
  `pytest -q tests/test_MILP.py tests/test_api_WindFarmNetwork.py tests/test_interarraylib.py::test_assign_cables_accepts_float_loads tests/test_interarraylib.py::test_assign_cables_accepts_tiny_capacity_overshoot`
  passed 94 tests with 1 skipped; ruff check/format and py_compile passed for
  the touched API, tests, and benchmark scripts.

## 2026-06-19

- Renamed the high-level turbine-power rounding option from
  `turbine_power_precision` to `turbine_power_decimals`.
- Changed the option semantics from a scale factor to decimal places:
  `turbine_power_decimals=1` keeps one decimal place, while
  `turbine_power_decimals=10` keeps ten decimal places.
- Updated benchmark controls and metadata names from `*_TURBINE_POWER_PRECISION`
  / `precision_semantics` to `*_TURBINE_POWER_DECIMALS` /
  `rounding_semantics`.
- Bumped timing benchmark schema metadata for the new decimal-place semantics.
- Fixed stale `continuous_power_flow` help/error text that still described the
  feature as OR-Tools-only after Pyomo/Gurobi support was added.
- Updated the weighted-turbine-power notebook source text/examples to describe
  `turbine_power_decimals` instead of the old denominator-approximation behavior.
