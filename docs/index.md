<p></p>

```{image} _static/OptiWindNet.svg
:alt: OptiWindNet
:width: 40%
:align: center
```

# OptiWindNet Documentation

**OptiWindNet: Wind Farm Electrical Network Optimizer**\
(distributed under the [MIT License](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/blob/main/LICENSE))

|  |  |
| --: | :-- |
| Python Package Index (PyPI) | <https://pypi.org/project/optiwindnet/> |
| Source code repository | <https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet> |
| Issue tracker | <https://github.com/DTUWindEnergy/OptiWindNet/issues> |
| Jupyter notebooks used in this manual | <https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/tree/main/docs/notebooks> |

## About OptiWindNet

OptiWindNet is an electrical network design tool for offshore wind farms developed at the Technical University of Denmark -- DTU. The package offers a framework to obtain optimal or near-optimal cable routes for a given turbine layout within the cable-laying boundaries. It provides high-level access to constructive-heuristic, meta-heuristic and exact optimization routers.

The tool is distributed as the open-source Python package **optiwindnet**, which can be used either within an interactive Python session (e.g. Jupyter notebook) or as a library, by invoking OptiWindNet's API directly from another application (e.g. [TOPFARM](https://topfarm.pages.windenergy.dtu.dk/TopFarm2/notebooks/cables.html), [Ard](https://github.com/NLRWindSystems/Ard)).

## What can OptiWindNet do?

- Optimize the network of array cables (aka collection system, infield cables, internal grid, inter-array cables);
- Route the cables to avoid exclusion zones and cable-to-cable crossings;
- Assign cable types and calculate network costs;
- Use different optimization approaches according to the preferred time/quality trade-off;
- Employ user-provided models and objective functions within the mathematical optimization approach.

## How this documentation is arranged

**Start** gets you running: {doc}`/install`, the {doc}`Quick Start </notebooks/hi00_quickstart>`, and {doc}`/apis`, which introduces the two APIs and helps you pick one.

**Concepts** is two pages, API-agnostic and worth reading once whichever API you pick: what the tool computes ({doc}`/problem`) and how it solves it ({doc}`/routers`).

**{doc}`/high_level_api`** and **{doc}`/low_level_api`** are the two sets of worked notebooks. Most notebooks have a counterpart in the other section that performs the same task.

**Reference** holds the lookup material, meant to be consulted rather than read through: the {doc}`/reference/tasks` finds a page by what you are trying to do, plus the {doc}`/reference/glossary`, {doc}`/reference/input_formats`, {doc}`/reference/solvers`, {doc}`/reference/milp_formulation`, {doc}`/reference/validation` and the generated {doc}`API Reference </autoapi/index>`.

**Papers** holds the two scientific articles behind the tool: {doc}`/paper` presents the framework and reproduces the computational experiments of the article below, while {doc}`/dataset` presents the open database of routing solutions produced with it.

## How to Cite

A peer-reviewed scientific article explaining the OptiWindNet framework and benchmarking it against state-of-the-art methods is available (open-access) at:

- Mauricio Souza de Alencar, Tuhfe Göçmen, Nicolaos A. Cutululis, _Flexible cable routing framework for wind farm collection system optimization_, European Journal of Operational Research, 329(3):1037-1051, 2026, ISSN 0377-2217, <https://doi.org/10.1016/j.ejor.2025.07.069>.

The BibTeX entry is in {doc}`/paper`, together with the notebooks that reproduce the article's results.

A second article introduces **OptiWindNet RouteSets**, an open database of cable-routing solutions produced with _OptiWindNet_. It is under review; the preprint is open-access at:

- Mauricio Souza de Alencar, Tuhfe Göçmen, Nicolaos A. Cutululis, _OptiWindNet RouteSets: a solver-diverse benchmark dataset for the offshore wind-farm cable routing problem_, Wind Energy Science Discussions [preprint], 2026, <https://doi.org/10.5194/wes-2026-124>, in review.

Cite it if you use the database, whose own DOI and BibTeX entry are in {doc}`/dataset`.

The OptiWindNet software package can be cited (unversioned) as:

> Souza de Alencar, M., Arasteh, A., & Friis-Møller, M. (2026). OptiWindNet by DTU Wind Energy. Zenodo. https://doi.org/10.5281/zenodo.18388438

To cite a specific version, get the version-specific DOI at [OptiWindNet's entry at Zenodo](https://doi.org/10.5281/zenodo.18388438). Select the desired version on the right column and use one of the ready-to-use citation formats available at the bottom right of that page.

## Acknowledgements

The development of OptiWindNet was carried out as part of a Ph.D. project at the Technical University of Denmark (DTU Wind), financially supported by the Independent Research Fund Denmark / Danmarks Frie Forskningsfond (DFF) under grant no. 1127-00188B, project _Integrated Design of Offshore Wind Power Plants_.

The heuristics implemented in this repository (release 0.0.1) are presented and analyzed in the MSc thesis [Optimization heuristics for offshore wind power plant collection systems design](https://fulltext-gateway.cvt.dk/oafilestore?oid=62dddf809a5e7116caf943f3&targetid=62dddf80a41ba354e4ed35bc) (DTU Wind - Technical University of Denmark, July 4, 2022).

The meta-heuristic used is [vidalt/HGS-CVRP](https://github.com/vidalt/HGS-CVRP) — a modern implementation of the hybrid genetic search (HGS) algorithm specialized to the capacitated vehicle routing problem (CVRP), including an additional neighborhood called SWAP\* — via its Python bindings [mdealencar/HybGenSea](https://github.com/mdealencar/HybGenSea).

The cable routing relies on a navigation mesh generated by the library [artem-ogre/CDT](https://github.com/artem-ogre/CDT) (Constrained Delaunay Triangulation, C++) via its Python bindings [artem-ogre/PythonCDT](https://github.com/artem-ogre/PythonCDT).

```{toctree}
:hidden:
:caption: Start

install
notebooks/hi00_quickstart
apis
```

```{toctree}
:hidden:
:caption: Concepts

problem
routers
```

```{toctree}
:hidden:
:caption: Guides

high_level_api
low_level_api
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Reference

reference/tasks
reference/glossary
reference/input_formats
reference/solvers
reference/milp_formulation
reference/validation
autoapi/index
```

```{toctree}
:hidden:
:caption: Papers

paper
dataset
```
