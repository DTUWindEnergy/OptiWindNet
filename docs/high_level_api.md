# Network/Router API

The high-level API exposes _OptiWindNet_ through two classes: {py:class}`WindFarmNetwork <optiwindnet.api.WindFarmNetwork>`, which holds the problem instance and its solution, and {py:class}`Router <optiwindnet.api.Router>`, which represents the algorithm used to solve it.

These notebooks show how to drive that API. What the tool computes and how it solves are covered by the API-agnostic {doc}`/problem` and {doc}`/routers`; the formats it accepts are catalogued in {doc}`/reference/input_formats`.

[](/reference/tasks.md#paired-examples) maps each notebook here to its counterpart in the {doc}`/low_level_api`, and the {doc}`/reference/tasks` indexes them by goal. Complete signatures are in the generated {doc}`API Reference </autoapi/index>`.

<!-- ## Running -->
<!--  -->
<!-- *OptiWindNet* is not an application and has no *main* program to be executed. The recommended way to use it is in an interactive Python notebook such as [JupyterLab](https://jupyterlab.readthedocs.io/en/latest/) or the [Jupyter Extension for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter]. -->

```{toctree}
:titlesonly:
:caption: Basics

notebooks/hi10_windfarmnetwork
notebooks/hi11_data_input
notebooks/hi12_locations
notebooks/hi13_border_obstacles
notebooks/hi14_plotting
notebooks/hi15_debugging
```

```{toctree}
:titlesonly:
:caption: Routers

notebooks/hi20_heuristic
notebooks/hi21_hgs
notebooks/hi23_milp
```

```{toctree}
:titlesonly:
:caption: Shaping the solution

notebooks/hi30_topologies
notebooks/hi31_options
```

```{toctree}
:titlesonly:
:caption: Worked examples

notebooks/hi40_example_taylor_2023
notebooks/hi41_example_iea_wind_task_55
```

```{toctree}
:titlesonly:
:caption: Integration

notebooks/hi50_gradient
notebooks/hi51_topfarm
```
