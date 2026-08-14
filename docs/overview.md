(overview)=
# How to Use This Documentation

*OptiWindNet* can be driven through two different APIs, and this documentation covers both.
The material is arranged so that everything the two have in common comes first:

{doc}`problem`, {doc}`data` and {doc}`methods`
: API-agnostic. What the tool computes, what you must feed it, and which solution methods
  and options are available. Read these once, regardless of how you intend to call the
  package.

{doc}`high_level_api` and {doc}`low_level_api`
: How to actually invoke it. The two sections cover largely the same ground in two
  different styles — **learning only one of them is enough** to start using
  *OptiWindNet*.

(overview-apis)=
## Which API?

* 📦 **Network/Router API**: high-level, user-friendly interface (recommended for most users)
* 🛠️ **Advanced API**: low-level, fine-grained control for performance, customization and research

### 📦 Network/Router API

The {doc}`Network/Router API <high_level_api>` (high level) makes the main functionality of *OptiWindNet* available through two classes: `WindFarmNetwork` and `Router`.
This approach enables quick experimentation and includes some guardrails for beginners.

* Simple to use, more forgiving on mistakes;
* Gentler learning curve;
* Focused on productivity and ease of interaction.

### 🛠️ Advanced API

The {doc}`Advanced API <low_level_api>` (low level) offers fine-grained control of all data structures and functions of *OptiWindNet*.

* Allows picking and choosing exactly what is needed from *OptiWindNet*;
* May perform faster by avoiding unnecessary checks and offering more tuning options;
* The API to use for extending *OptiWindNet* with custom algorithms, models, objective functions or constraints.

These lower-level interfaces are developer-facing and may evolve independently of the Network/Router API; keep advanced integrations pinned to a tested *OptiWindNet* version.

(overview-pairs)=
## Paired examples

Most notebooks have a counterpart in the other section that performs the same task through the other API.
Use this table to cross over — to compare the two styles, or to find the equivalent of something you already know.

| Task | 📦 Network/Router API | 🛠️ Advanced API |
| --- | --- | --- |
| Minimal working example | {doc}`notebooks/hi00_quickstart` | {doc}`notebooks/lo00_quickstart` |
| Loading input data | {doc}`notebooks/hi11_data_input` | {doc}`notebooks/lo11_data_input` |
| Bundled locations | {doc}`notebooks/hi12_repositories` | {doc}`notebooks/lo12_repositories` |
| Plotting | {doc}`notebooks/hi13_plotting` | {doc}`notebooks/lo13_plotting` |
| Constructive heuristic | {doc}`notebooks/hi20_heuristic` | {doc}`notebooks/lo20_heuristic` |
| Meta-heuristic | {doc}`notebooks/hi21_hgs` | {doc}`notebooks/lo21_hgs` |
| Exact optimization | {doc}`notebooks/hi23_milp` | {doc}`notebooks/lo23_milp_ortools` |
| Ringed topology | {doc}`notebooks/hi29_topologies` | {doc}`notebooks/lo29_topologies` |
| Example: Taylor 2023 | {doc}`notebooks/hi40_example_taylor_2023` | {doc}`notebooks/lo40_example_taylor_2023` |
| Example: IEA Wind Task 55 | {doc}`notebooks/hi41_example_iea_wind_task_55` | {doc}`notebooks/lo41_example_iea_wind_task_55` |

Some material exists in one section only: the `WindFarmNetwork` class, gradients, geometry buffering and TOPFARM integration are specific to the Network/Router API, while the LKH-3 meta-heuristic, the per-solver MILP examples and the legacy heuristic migration guide are specific to the Advanced API.

(overview-notebooks)=
## Running the notebooks

Both API sections consist of Jupyter notebooks, which can be viewed online, executed in the cloud through the *Binder* link at the top of each notebook page, or run on a personal computer with [Jupyter](https://jupyter.org/install) (JupyterLab recommended).
The notebooks and the data they read are available in [the repository](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/tree/main/docs/notebooks).

```{admonition} Trust the notebook to see the figures
:class: important

Many of the notebooks have SVG figures as cell outputs, which JupyterLab and Jupyter
Notebook only display if the notebook is marked as trusted. In JupyterLab: press
`Ctrl+Shift+C`, then **Trust Notebook**.
```
