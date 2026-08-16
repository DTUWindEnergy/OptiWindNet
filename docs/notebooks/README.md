# Example Notebooks Overview

These notebooks are the worked examples of the _OptiWindNet_ documentation. They show **how to call** the package; what the tool computes, what it accepts as input, and which routers and options exist are covered by the API-agnostic pages one level up.

Worth reading once, whichever of the two APIs you use:

- [`../problem.md`](../problem.md) — the optimization problem, the graph model, the plot views and the network topologies
- [`../routers.md`](../routers.md) — the optimization approaches, warm-starting and how to choose

Worth consulting as needed:

- [`../reference/tasks.md`](../reference/tasks.md) — a task-oriented index into all of the above
- [`../reference/glossary.md`](../reference/glossary.md) — the vocabulary used throughout
- [`../reference/input_formats.md`](../reference/input_formats.md) — input formats and location repositories
- [`../reference/solvers.md`](../reference/solvers.md) — the MILP backends and how to install them

## Naming scheme

`<api><band><n>_<topic>`

| Prefix | Meaning                                               |
| ------ | ----------------------------------------------------- |
| `hi`   | Network/Router API — the high-level path              |
| `lo`   | Advanced API — direct use of the internal modules     |
| `p`    | Reproduction of the paper's computational experiments |

The band digit means the same thing in both API sections, and paired notebooks share the band, the index and the topic stem — so `hi21_hgs` and `lo21_hgs` are the same task through the two APIs.

| Band | Contents                  |
| ---- | ------------------------- |
| `0`  | quickstart                |
| `1`  | input data and inspection |
| `2`  | routers                   |
| `3`  | shaping the solution      |
| `4`  | worked examples           |
| `5`  | integration               |
| `9`  | appendix                  |

## Network/Router API (recommended for most users)

Drives _OptiWindNet_ through the `WindFarmNetwork` and `Router` classes: provide input data, call a few intuitive methods, read the results.

- `hi00_quickstart`: minimum steps to optimize a network
- `hi10_windfarmnetwork`, `hi11_data_input`, `hi12_locations`, `hi13_border_obstacles`, `hi14_plotting`, `hi15_debugging`
- `hi20_heuristic`, `hi21_hgs`, `hi23_milp`: the three optimization approaches
- `hi30_topologies`, `hi31_options`: shaping the solution
- `hi40_example_taylor_2023`, `hi41_example_iea_wind_task_55`: worked examples
- `hi50_gradient`, `hi51_topfarm`: integration with an outer optimization loop

> Use these if you want fast prototyping or plan to integrate `OptiWindNet` into a larger workflow.

## Advanced API

Imports the internal modules directly, calling each intermediate step (mesh generation, warm start, optimization, routing) explicitly.

- `lo00_quickstart`: the `L` → `P`,`A` → `S` → `G` pipeline
- `lo11_data_input`, `lo12_locations`, `lo14_plotting`
- `lo20_heuristic`, `lo21_hgs`, `lo22_lkh`: the constructive-heuristic and meta-heuristic routers
- `lo23_milp_ortools`–`lo28_milp_cbc`: one notebook per MILP backend
- `lo30_topologies`: topologies and validation
- `lo32_clustering`: automatic vs. manual substation clustering
- `lo40_example_taylor_2023`, `lo41_example_iea_wind_task_55`: worked examples
- `lo90_removed_heuristics`: migration from removed entry points

> Use these if you're exploring the algorithm, debugging, or building on top of the library internals.

## Paper reproduction

`p01`–`p07` reproduce the computational experiments of the paper; each notebook's title carries the section number it corresponds to.

## Paired examples

| Task | Network/Router API | Advanced API |
| --- | --- | --- |
| Minimal working example | `hi00_quickstart` | `lo00_quickstart` |
| Loading input data | `hi11_data_input` | `lo11_data_input` |
| Bundled locations | `hi12_locations` | `lo12_locations` |
| Plotting | `hi14_plotting` | `lo14_plotting` |
| Constructive heuristic | `hi20_heuristic` | `lo20_heuristic` |
| Meta-heuristic | `hi21_hgs` | `lo21_hgs` |
| Exact optimization | `hi23_milp` | `lo23_milp_ortools` |
| Topologies | `hi30_topologies` | `lo30_topologies` |
| Example: Taylor 2023 | `hi40_example_taylor_2023` | `lo40_example_taylor_2023` |
| Example: IEA Wind Task 55 | `hi41_example_iea_wind_task_55` | `lo41_example_iea_wind_task_55` |

Learning **one** of the two APIs is enough to start using _OptiWindNet_.

> **Note:** many of these notebooks have SVG figures as cell outputs, which JupyterLab and Jupyter Notebook only display if the notebook is marked as **trusted** (in JupyterLab: `Ctrl+Shift+C` → **Trust Notebook**).
