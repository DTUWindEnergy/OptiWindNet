# Advanced API

These notebooks are developer-facing examples that import lower-level modules directly. Those modules may evolve independently of the Network/Router API, so pin an integration built on them to a tested _OptiWindNet_ version.

The graphs these notebooks pass between functions — `L`, `P`, `A`, `S` and `G` — are described in [](/problem.md#the-graph-model); the routers they call, and the model and solver options they pass, are in {doc}`/routers`. Checking a result is covered by {doc}`/reference/validation`.

[](/reference/tasks.md#paired-examples) maps each notebook here to its counterpart in the {doc}`/high_level_api`, and the {doc}`/reference/tasks` indexes them by goal. Complete signatures are in the generated {doc}`API Reference </autoapi/index>`.

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
