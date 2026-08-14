# Network/Router API

The high-level API exposes *OptiWindNet* through two classes: `WindFarmNetwork`, which holds the problem instance and its solution, and `Router`, which represents the method used to solve it.

These notebooks show how to drive that API. For what the tool computes and which methods and options are available, see {doc}`problem`, {doc}`data` and {doc}`methods` — those apply to both APIs. {ref}`overview-pairs` maps each notebook here to its counterpart in the {doc}`low_level_api`, and {doc}`howto` indexes them by task.

Complete signatures for every class and function are in the generated {doc}`API Reference <autoapi/index>`.

<!-- ## Running -->
<!--  -->
<!-- *OptiWindNet* is not an application and has no *main* program to be executed. The recommended way to use it is in an interactive Python notebook such as [JupyterLab](https://jupyterlab.readthedocs.io/en/latest/) or the [Jupyter Extension for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter]. -->

```{toctree}
:glob:
:titlesonly:
:caption: Fundamentals

notebooks/hi1*
```

```{toctree}
:glob:
:titlesonly:
:caption: Methods

notebooks/hi2*
```

```{toctree}
:glob:
:titlesonly:
:caption: Features

notebooks/hi3*
```

```{toctree}
:glob:
:titlesonly:
:caption: Worked examples

notebooks/hi4*
```

```{toctree}
:glob:
:titlesonly:
:caption: Integration

notebooks/hi5*
```
