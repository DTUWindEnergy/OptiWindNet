# Advanced API

These notebooks are developer-facing examples that import lower-level modules directly. Their interfaces offer fine control but are not covered by the compatibility guarantees of the Network/Router API.

The graphs these notebooks pass between functions — `L`, `P`, `A`, `S` and `G` — are described in {ref}`problem-graph-model`; the methods they call and the options they set are described in {doc}`methods`. {ref}`overview-pairs` maps each notebook here to its counterpart in the {doc}`high_level_api`, and {doc}`howto` indexes them by task.

Complete signatures for every module, class and function are in the generated {doc}`API Reference <autoapi/index>`.

```{toctree}
:glob:
:titlesonly:
:caption: Getting started

notebooks/lo00_quickstart
```

```{toctree}
:glob:
:titlesonly:
:caption: Fundamentals

notebooks/lo1*
```

```{toctree}
:glob:
:titlesonly:
:caption: Methods

notebooks/lo2*
```

```{toctree}
:glob:
:titlesonly:
:caption: Features

notebooks/lo3*
```

```{toctree}
:glob:
:titlesonly:
:caption: Worked examples

notebooks/lo4*
```
