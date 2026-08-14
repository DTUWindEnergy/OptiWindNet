# Example Notebooks Overview

These notebooks are the worked examples of the *OptiWindNet* documentation.
They show **how to call** the package; what the tool computes, what it accepts as input, and which solution methods and options exist are covered by the API-agnostic pages one level up:

- [`../problem.md`](../problem.md) — the optimization problem, the graph model, the vocabulary and the network topologies
- [`../data.md`](../data.md) — input formats, location repositories, geometry preparation and plotting
- [`../methods.md`](../methods.md) — the solution methods, model options, solver options and warm-starting
- [`../howto.md`](../howto.md) — a task-oriented index into all of the above

Read those once; they apply regardless of which of the two APIs you use.

## Naming scheme

`<api><band><n>_<topic>`

| Prefix | Meaning |
| --- | --- |
| `hi` | Network/Router API — the high-level path |
| `lo` | Advanced API — direct use of the internal modules |
| `p` | Reproduction of the paper's computational experiments |

The band digit means the same thing in both API sections, and paired notebooks share the band, the index and the topic stem — so `hi21_hgs` and `lo21_hgs` are the same task through the two APIs.

| Band | Contents |
| --- | --- |
| `0` | quickstart |
| `1` | data and visualization |
| `2` | solution methods |
| `3` | features |
| `4` | worked examples |
| `5` | integration |

## Network/Router API (recommended for most users)

Drives *OptiWindNet* through the `WindFarmNetwork` and `Router` classes: provide input data, call a few intuitive methods, read the results.

- `hi00_quickstart`: minimum steps to optimize a network
- `hi10`–`hi14`: the `WindFarmNetwork` object, data input, location repositories, plotting, debugging
- `hi20`–`hi29`: heuristic, meta-heuristic and MILP routers; ringed topologies
- `hi30`–`hi33`: model and solver options, borders and obstacles, gradients, substation clustering
- `hi40`–`hi41`: worked examples
- `hi50_topfarm`: TOPFARM integration

> Use these if you want fast prototyping or plan to integrate `OptiWindNet` into a larger workflow.

## Advanced API

Imports the internal modules directly, calling each intermediate step (mesh generation, warm start, optimization, routing) explicitly.

- `lo00_quickstart`: the `L` → `P`,`A` → `S` → `G` pipeline
- `lo11`–`lo13`: data input, bundled locations, plotting
- `lo20`–`lo29`: constructive heuristic, HGS-CVRP, LKH-3, one notebook per MILP solver, topologies and validation
- `lo34_legacy_heuristics`: migration from removed entry points
- `lo40`–`lo41`: worked examples

> Use these if you're exploring the algorithm, debugging, or building on top of the library internals.

## Paper reproduction

`p01`–`p07` reproduce the computational experiments of the paper; each notebook's title carries the section number it corresponds to.

## Paired examples

| Task | Network/Router API | Advanced API |
| --- | --- | --- |
| Minimal working example | `hi00_quickstart` | `lo00_quickstart` |
| Loading input data | `hi11_data_input` | `lo11_data_input` |
| Bundled locations | `hi12_repositories` | `lo12_repositories` |
| Plotting | `hi13_plotting` | `lo13_plotting` |
| Constructive heuristic | `hi20_heuristic` | `lo20_heuristic` |
| Meta-heuristic | `hi21_hgs` | `lo21_hgs` |
| Exact optimization | `hi23_milp` | `lo23_milp_ortools` |
| Topologies | `hi29_topologies` | `lo29_topologies` |
| Example: Taylor 2023 | `hi40_example_taylor_2023` | `lo40_example_taylor_2023` |
| Example: IEA Wind Task 55 | `hi41_example_iea_wind_task_55` | `lo41_example_iea_wind_task_55` |

Learning **one** of the two APIs is enough to start using *OptiWindNet*.

> **Note:** many of these notebooks have SVG figures as cell outputs, which JupyterLab and
> Jupyter Notebook only display if the notebook is marked as **trusted** (in JupyterLab:
> `Ctrl+Shift+C` → **Trust Notebook**).
