(howto)=
# How Do I…?

A task-oriented index into the rest of the documentation.
Each entry names the concept page that explains the *what* and the notebooks that show the *how*, once per API.

(howto-getting-a-network)=
## Get a network at all

| Goal | Read | Network/Router | Advanced |
| --- | --- | --- | --- |
| Optimize a layout for the first time | {doc}`methods` | {doc}`notebooks/hi00_quickstart` | {doc}`notebooks/lo00_quickstart` |
| Load my own turbine coordinates | {ref}`data-format-arrays` | {doc}`notebooks/hi11_data_input` | {doc}`notebooks/lo11_data_input` |
| Load a windIO, YAML or `.osm.pbf` file | {ref}`data-formats` | {doc}`notebooks/hi11_data_input` | {doc}`notebooks/lo11_data_input` |
| Try it on a real wind farm without any data of my own | {ref}`data-repository` | {doc}`notebooks/hi12_repositories` | {doc}`notebooks/lo12_repositories` |
| Get a cost instead of a cable length | {ref}`data-cables` | {doc}`notebooks/hi11_data_input` | — |

(howto-shaping)=
## Shape the solution

| Goal | Read | Network/Router | Advanced |
| --- | --- | --- | --- |
| Avoid branching at turbines (radial network) | {ref}`problem-topologies` | {doc}`notebooks/hi30_options` | {doc}`notebooks/lo29_topologies` |
| Build rings for fault redundancy | {ref}`problem-rings` | {doc}`notebooks/hi29_topologies` | {doc}`notebooks/lo29_topologies` |
| Limit or pin the number of feeders | {ref}`methods-model-options` | {doc}`notebooks/hi30_options` | {doc}`notebooks/lo23_milp_ortools` |
| Balance the load across subtrees | {ref}`methods-model-options` | {doc}`notebooks/hi30_options` | {doc}`notebooks/lo23_milp_ortools` |
| Keep feeder routes straight | {ref}`methods-model-options` | {doc}`notebooks/hi20_heuristic` | {doc}`notebooks/lo20_heuristic` |
| Keep cables out of an exclusion zone | {ref}`data-geometry` | {doc}`notebooks/hi31_border_obstacles` | — |
| Add a safety margin to the boundaries | {ref}`data-geometry` | {doc}`notebooks/hi31_border_obstacles` | — |
| Handle several substations | {ref}`methods-metaheuristic` | {doc}`notebooks/hi33_clustering` | — |

(howto-quality)=
## Trade runtime for quality

| Goal | Read | Network/Router | Advanced |
| --- | --- | --- | --- |
| Choose between the three method families | {ref}`methods-choosing` | {doc}`notebooks/hi00_quickstart` | {doc}`notebooks/lo00_quickstart` |
| Get the fastest possible answer | {ref}`methods-constructive` | {doc}`notebooks/hi20_heuristic` | {doc}`notebooks/lo20_heuristic` |
| Get a better network for a fixed time budget | {ref}`methods-metaheuristic` | {doc}`notebooks/hi21_hgs` | {doc}`notebooks/lo21_hgs` |
| Prove how good the solution is | {ref}`methods-exact` | {doc}`notebooks/hi23_milp` | {doc}`notebooks/lo23_milp_ortools` |
| Speed up a MILP solve with a warm start | {ref}`methods-warmstart` | {doc}`notebooks/hi23_milp` | {doc}`notebooks/lo40_example_taylor_2023` |
| Pick and install a MILP solver | {ref}`methods-exact` · {doc}`setup` | {doc}`notebooks/hi23_milp` | {doc}`notebooks/lo23_milp_ortools` |

(howto-inspect)=
## Inspect and integrate

| Goal | Read | Network/Router | Advanced |
| --- | --- | --- | --- |
| Plot a location or a result | {ref}`data-plotting` | {doc}`notebooks/hi13_plotting` | {doc}`notebooks/lo13_plotting` |
| Label the turbines in a figure | {ref}`data-plot-options` | {doc}`notebooks/hi13_plotting` | {doc}`notebooks/lo13_plotting` |
| See why a route bends the way it does | {ref}`problem-crossings` | {doc}`notebooks/hi13_plotting` | {doc}`notebooks/lo13_plotting` |
| Check that a solution is valid | {ref}`problem-validation` | — | {doc}`notebooks/lo29_topologies` |
| Get gradients for an outer optimization loop | — | {doc}`notebooks/hi32_gradient` | — |
| Drive it from TOPFARM | — | {doc}`notebooks/hi50_topfarm` | — |
| Migrate from a removed heuristic entry point | {ref}`methods-constructive` | — | {doc}`notebooks/lo34_legacy_heuristics` |

(howto-trouble)=
## When something goes wrong

`ValueError` about turbines outside the border
: Every terminal and root is checked against the allowed area before optimizing. Either the coordinates are wrong, or the border is — see {ref}`data-geometry`. Plotting the location before optimizing is the fastest way to tell which.

The MILP solver ignored my warm start
: Not every solution is a valid start for every model. A constructive heuristic does not limit the feeder count and is not radial, so it cannot start a model that requires either. The full matrix is in {ref}`methods-warmstart`; run with `verbose=True` to see the solver report what it accepted.

The result is not what the options asked for
: Only the exact methods enforce every model option. A heuristic given an option it cannot honor returns a solution that ignores it, without failing — see {ref}`methods-model-options`.

I need to see what the algorithm is doing
: Logging is configured per module, and non-Python solvers have their own verbosity — see {doc}`notebooks/hi14_debugging`.

```{admonition} Planned
:class: seealso

Two topics are not yet covered and are being written:

* **Extending the model** — adding custom constraints and objective functions to the
  MILP formulation. This is the stated purpose of the {doc}`low_level_api`, and it
  currently has no page of its own.
* **Performance guidance** — how solve time responds to turbine count, cable capacity
  and solver choice, and how to decide when to stop a solve.
```
