# Task Index

A task-oriented index into the rest of the documentation. Each entry names the concept page that explains the _what_ and the notebooks that show the _how_, once per API.

## Get a network at all

| Goal | Read | Network/Router | Advanced |
| --- | --- | --- | --- |
| Optimize a layout for the first time | {doc}`/routers` | {doc}`/notebooks/hi00_quickstart` | {doc}`/notebooks/lo00_quickstart` |
| Load my own turbine coordinates | [](/reference/input_formats.md#coordinate-arrays) | {doc}`/notebooks/hi11_data_input` | {doc}`/notebooks/lo11_data_input` |
| Load a windIO, YAML or `.osm.pbf` file | [Input formats](/reference/input_formats.md#input-formats) | {doc}`/notebooks/hi11_data_input` | {doc}`/notebooks/lo11_data_input` |
| Try it on a real wind farm without any data of my own | [](/reference/input_formats.md#location-repositories) | {doc}`/notebooks/hi12_locations` | {doc}`/notebooks/lo12_locations` |
| Get a cost instead of a cable length | [](/reference/input_formats.md#cable-types) | {doc}`/notebooks/hi11_data_input` | — |

## Shape the solution

| Goal | Read | Network/Router | Advanced |
| --- | --- | --- | --- |
| Avoid branching at turbines (radial network) | [](/problem.md#network-topologies) | {doc}`/notebooks/hi31_options` | {doc}`/notebooks/lo30_topologies` |
| Build rings for fault redundancy | [](/problem.md#ring-semantics) | {doc}`/notebooks/hi30_topologies` | {doc}`/notebooks/lo30_topologies` |
| Limit or pin the number of feeders | [](/routers.md#model-options) | {doc}`/notebooks/hi31_options` | {doc}`/notebooks/lo23_milp_ortools` |
| Balance the load across subtrees | [](/routers.md#model-options) | {doc}`/notebooks/hi31_options` | {doc}`/notebooks/lo23_milp_ortools` |
| Keep feeder routes straight | [](/routers.md#model-options) | {doc}`/notebooks/hi20_heuristic` | {doc}`/notebooks/lo20_heuristic` |
| Keep cables out of an exclusion zone | [](/problem.md#crossings-contours-and-detours) | {doc}`/notebooks/hi13_border_obstacles` | — |
| Add a safety margin to the boundaries | [](/reference/input_formats.md#preparing-the-geometry) | {doc}`/notebooks/hi13_border_obstacles` | — |
| Handle several substations | [](/routers.md#meta-heuristics) · [](/problem.md#ring-semantics) | — | {doc}`/notebooks/lo32_clustering` |

## Trade runtime for quality

| Goal | Read | Network/Router | Advanced |
| --- | --- | --- | --- |
| Choose between optimization approaches | [](/routers.md#choosing-a-router) | {doc}`/notebooks/hi00_quickstart` | {doc}`/notebooks/lo00_quickstart` |
| Get the fastest possible answer | [](/routers.md#constructive-heuristics) | {doc}`/notebooks/hi20_heuristic` | {doc}`/notebooks/lo20_heuristic` |
| Get a better network for a fixed time budget | [](/routers.md#meta-heuristics) | {doc}`/notebooks/hi21_hgs` | {doc}`/notebooks/lo21_hgs` |
| Prove how good the solution is | [](/routers.md#exact-optimization) | {doc}`/notebooks/hi23_milp` | {doc}`/notebooks/lo23_milp_ortools` |
| Speed up a MILP solve with a warm start | [](/routers.md#warm-starting) | {doc}`/notebooks/hi23_milp` | {doc}`/notebooks/lo40_example_taylor_2023` |
| Pick and install a MILP solver | [](/routers.md#exact-optimization) · {doc}`/install` | {doc}`/notebooks/hi23_milp` | {doc}`/notebooks/lo23_milp_ortools` |
| Decide when to stop a solve | [](/routers.md#how-long-a-solve-takes) | {doc}`/notebooks/hi23_milp` | {doc}`/notebooks/lo23_milp_ortools` |

## Inspect and integrate

| Goal | Read | Network/Router | Advanced |
| --- | --- | --- | --- |
| Plot a location or a result | [](/problem.md#the-graph-model) | {doc}`/notebooks/hi14_plotting` | {doc}`/notebooks/lo14_plotting` |
| Label the turbines in a figure | — | {doc}`/notebooks/hi14_plotting` | {doc}`/notebooks/lo14_plotting` |
| See why a route bends the way it does | [](/problem.md#crossings-contours-and-detours) | {doc}`/notebooks/hi14_plotting` | {doc}`/notebooks/lo14_plotting` |
| Check that a solution is valid | [Validation](/reference/validation.md#validation) | — | {doc}`/notebooks/lo30_topologies` |
| Get gradients for an outer optimization loop | — | {doc}`/notebooks/hi50_gradient` | — |
| Drive it from TOPFARM | — | {doc}`/notebooks/hi51_topfarm` | — |
| Migrate from a removed heuristic entry point | [](/routers.md#constructive-heuristics) | — | {doc}`/notebooks/lo90_removed_heuristics` |
| Add a constraint or an objective of my own | {doc}`/extending` | — | {doc}`/notebooks/lo23_milp_ortools` |

## Paired examples

Most notebooks have a counterpart in the other section that performs the same task through the other API. Use this table to cross over — to compare the two styles, or to find the equivalent of something you already know.

| Task | 📦 Network/Router API | 🛠️ Advanced API |
| --- | --- | --- |
| Minimal working example | {doc}`/notebooks/hi00_quickstart` | {doc}`/notebooks/lo00_quickstart` |
| Loading input data | {doc}`/notebooks/hi11_data_input` | {doc}`/notebooks/lo11_data_input` |
| Bundled locations | {doc}`/notebooks/hi12_locations` | {doc}`/notebooks/lo12_locations` |
| Plotting | {doc}`/notebooks/hi14_plotting` | {doc}`/notebooks/lo14_plotting` |
| Constructive heuristic | {doc}`/notebooks/hi20_heuristic` | {doc}`/notebooks/lo20_heuristic` |
| Meta-heuristic | {doc}`/notebooks/hi21_hgs` | {doc}`/notebooks/lo21_hgs` |
| Exact optimization | {doc}`/notebooks/hi23_milp` | {doc}`/notebooks/lo23_milp_ortools` |
| Network topologies | {doc}`/notebooks/hi30_topologies` | {doc}`/notebooks/lo30_topologies` |
| Example: Taylor 2023 | {doc}`/notebooks/hi40_example_taylor_2023` | {doc}`/notebooks/lo40_example_taylor_2023` |
| Example: IEA Wind Task 55 | {doc}`/notebooks/hi41_example_iea_wind_task_55` | {doc}`/notebooks/lo41_example_iea_wind_task_55` |

Some material exists in one section only: the {py:class}`WindFarmNetwork <optiwindnet.api.WindFarmNetwork>` class, gradients, geometry buffering and TOPFARM integration are specific to the Network/Router API, while the LKH-3 meta-heuristic, the per-solver MILP examples, substation clustering and the removed heuristics migration guide are specific to the Advanced API.

## When something goes wrong

<!-- prettier-ignore-start -->

`ValueError` about turbines outside the border
: Every terminal and root is checked against the allowed area before optimizing. Either the coordinates are wrong, or the border is — see [](/reference/input_formats.md#preparing-the-geometry). Plotting the location before optimizing is the fastest way to tell which.

The MILP solver ignored my warm start
: Not every solution is a valid start for every model. A constructive heuristic does not limit the feeder count and is not radial, so it cannot start a model that requires either. The full matrix is in [](/routers.md#warm-starting); run with `verbose=True` to see the solver report what it accepted.

The result is not what the options asked for
: Only the exact routers enforce every model option. A router given an option it cannot honor returns a solution that ignores it, without failing — see [](/routers.md#model-options).

I need to see what the algorithm is doing
: Logging is configured per module, and non-Python solvers have their own verbosity — see {doc}`/notebooks/hi15_debugging`.

The MILP solve is taking too long
: Solve time grows steeply with the terminal count and the cable capacity — see [](/routers.md#how-long-a-solve-takes) for the three ways to bound it.

<!-- prettier-ignore-end -->

```{admonition} Planned
:class: seealso

One topic is not yet covered and is being written: **extending the model** — adding custom
constraints and objective functions to the MILP formulation. See {doc}`/extending` for what
exists in the meantime.
```
