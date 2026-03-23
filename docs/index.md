<p></p>

```{image} _static/OptiWindNet.svg
:alt: OptiWindNet
:width: 40%
:align: center
```
# OptiWindNet Documentation
**OptiWindNet: Wind Farm Electrical Network Optimizer**\
(distributed under the [MIT License](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/blob/main/LICENSE))

|||
|--:|:--|
Python Package Index (PyPI) | <https://pypi.org/project/optiwindnet/>
Source code repository | <https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet>
Issue tracker | <https://github.com/DTUWindEnergy/OptiWindNet/issues>
Jupyter notebooks used in this manual | <https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/tree/main/docs/notebooks>

## About OptiWindNet

OptiWindNet is an electrical network design tool for offshore wind farms developed at the Technical University of Denmark -- DTU.
The package offers a framework to obtain optimal or near-optimal cable routes for a given turbine layout within the cable-laying boundaries. It provides high-level access to heuristic, meta-heuristic and mathematical optimization approaches to the problem.

The tool is distributed as the open-source Python package **optiwindnet**, which can be used either within an interactive Python session (e.g. Jupyter notebook) or as a library, by invoking OptiWindNet's API directly from another application.

## What can OptiWindNet do?

* Optimize the network of array cables (aka collection system, infield cables, internal grid, inter-array cables);
* Route the cables so as to avoid crossings;
* Assign cable types and calculate network costs;
* Use different optimization approaches according to the preferred time/quality trade-off;
* Employ user-provided models and objective functions within the mathematical optimization approach.

## Getting Started

[](setup) your Python environment and check the {doc}`Quickstart <notebooks/quickstart_high>` to begin using OptiWindNet.


## How to Cite

A peer-reviewed scientific article explaining the OptiWindNet framework and benchmarking it against state-of-the-art methods is available (open-access) at:
- Mauricio Souza de Alencar, Tuhfe Göçmen, Nicolaos A. Cutululis,
_Flexible cable routing framework for wind farm collection system optimization_,
European Journal of Operational Research,
2025, ISSN 0377-2217, <https://doi.org/10.1016/j.ejor.2025.07.069>.

```{code-block} bib
@article{
    SOUZADEALENCAR2025,
    title = {Flexible cable routing framework for wind farm collection system optimization},
    journal = {European Journal of Operational Research},
    year = {2025},
    issn = {0377-2217},
    doi = {https://doi.org/10.1016/j.ejor.2025.07.069},
    url = {https://www.sciencedirect.com/science/article/pii/S0377221725005946},
    author = {Mauricio {Souza de Alencar} and Tuhfe Göçmen and Nicolaos A. Cutululis},
    keywords = {Combinatorial optimization, Network design, Collection system, Wind farm},
}
```

The OptiWindNet software package can be cited (unversioned) as:
> Souza de Alencar, M., Arasteh, A., & Friis-Møller, M. (2026). OptiWindNet by DTU Wind Energy. Zenodo. https://doi.org/10.5281/zenodo.18388438

To cite a specific version, get the version-specific DOI at [OptiWindNet's entry at Zenodo](https://doi.org/10.5281/zenodo.18388438). Select the desired version on the right column and use one of the ready-to-use citation formats available at the bottom right of that page.

```{toctree}
setup
notebooks/quickstart_high
overview
high_level_api
low_level_api
theory
paper
```
