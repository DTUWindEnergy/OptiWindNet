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
