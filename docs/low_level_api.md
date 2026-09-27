# Advanced API

These notebooks are developer-facing examples that import lower-level modules directly. Those modules may evolve independently of the Network/Router API, so pin an integration built on them to a tested _OptiWindNet_ version.

The graphs these notebooks pass between functions — `L`, `P`, `A`, `S` and `G` — are described in [](/problem.md#graph-representations); the routers they call, and the model and solver options they pass, are in {doc}`/routers`. Checking a result is covered by {doc}`/reference/validation`.

[](/reference/tasks.md#paired-examples) maps each notebook here to its counterpart in the {doc}`/high_level_api`, and the {doc}`/reference/tasks` indexes them by goal. Complete signatures are in the generated {doc}`API Reference </autoapi/index>`.

## Updating code written for v0.3.0

Import graph conversions from `optiwindnet.converting`, load calculations from `optiwindnet.loads`, graph transformations from `optiwindnet.transforming`, validation from `optiwindnet.validating`, compact link encodings from `optiwindnet.terse`, and text descriptions from `optiwindnet.presenting`. Their old `interarraylib` aliases are deprecated and scheduled for removal in v0.4.0. Fingerprinting functions move from `optiwindnet.fingerprint` to `optiwindnet.identity`.

Pass every `Solver.set_problem()` argument after `P` and `A` by keyword:

```python
solver.set_problem(P, A, capacity=capacity, model_options=options, warmstart=S)
```

Use `solver.get_solution()` for the topology and routed graph. The removed `get_incumbent_topology()` method is replaced, for advanced uses that intentionally skip routing, by decoding `solver.incumbent_linkbits` with `S_from_linkbits()`. This recovers connectivity only: restore the topology type, capacity and terminal inflow, then calculate loads before using or validating it. {doc}`/notebooks/lo32_clustering` demonstrates the ringed, unit-inflow case. `SolutionInfo.topology_id` identifies the model incumbent; a solution pool can deliver a different topology after ranking by routed length.

`ModelOptions` accepts strings such as `topology='radial'`. Direct calls to backend `make_min_length_model()` functions require enum members for `topology`, `feeder_route` and `feeder_limit`.

For power declarations, use `set_turbine_powers()` for nominal ratings and `'inflow'` for integer demands. Graphs that used `'power'` for integer demands need that attribute renamed. Imported locations read nominal powers by default; use `read_powers=False` for turbine-count examples, or follow [](/reference/input_formats.md#turbines-of-unequal-output) for nominal capacities and router restrictions.

```{toctree}
:titlesonly:
:caption: Getting started

notebooks/lo00_quickstart
```

```{toctree}
:titlesonly:
:caption: Basics

notebooks/lo11_data_input
notebooks/lo12_locations
notebooks/lo14_plotting
notebooks/lo16_mixed_power
```

```{toctree}
:titlesonly:
:caption: Routers

notebooks/lo20_heuristic
notebooks/lo21_hgs
notebooks/lo22_lkh
```

```{toctree}
:titlesonly:
:caption: MILP backends

notebooks/lo23_milp_ortools
notebooks/lo24_milp_gurobi
notebooks/lo25_milp_cplex
notebooks/lo26_milp_highs
notebooks/lo27_milp_scip
notebooks/lo28_milp_cbc
```

```{toctree}
:titlesonly:
:caption: Shaping the solution

notebooks/lo30_topologies
notebooks/lo32_clustering
```

```{toctree}
:titlesonly:
:caption: Worked examples

notebooks/lo40_example_taylor_2023
notebooks/lo41_example_iea_wind_task_55
```

```{toctree}
:titlesonly:
:caption: Extending

extending
```

```{toctree}
:titlesonly:
:caption: Appendix

notebooks/lo90_removed_heuristics
```
