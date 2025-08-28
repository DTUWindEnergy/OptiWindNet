<p></p>

```{image} _static/OptiWindNet.svg
:alt: OptiWindNet
:width: 40%
:align: center
```
# OptiWindNet Documentation
**OptiWindNet = Optimize Windfarm Network**\
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

The tool is distributed as an open-source Python package that is suitable for use within an interactive Python session (e.g. Jupyter notebook). Alternatively, OptiWindNet's high-level API can be invoked directly from another application.

## What can OptiWindNet do?

* Optimize the network of array cables;
* Route the cables so as to avoid crossings;
* Assign cable types and calculate network costs;
* Use different optimization approaches according to the preferred time/quality trade-off;
* Employ user-provided models and objective functions within the mathematical optimization approach.

## Getting Started

[](setup) your Python environment and check the {doc}`Quickstart <notebooks/quickstart_high>` to begin using OptiWindNet.


## How to Cite

A peer-reviewed cientific article explaining the OptiWindNet framework and benchmarking it against state-of-the art methods is available (open-access) at:
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

The OptiWindNet software package can be cited as:
- Mauricio Souza de Alencar, Amir Arasteh and Mikkel Friis-Møller.
_OptiWindNet: An open-source wind farm electrical network optimization tool_,
DTU Wind and Energy Systems, Technical University of Denmark (2025, March),
url: <https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet>

```{code-block} bib
@article{
    optiwindnet_2025,
    title={OptiWindNet: Tool for designing and optimizing the electrical cable network of offshore wind farms},
    author={Mauricio {Souza de Alencar} and Amir Arasteh and Mikkel Friis-Møller},
    url="https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet",
    publisher={DTU Wind and Energy Systems, Technical University of Denmark},
    year={2025},
    month={4}
}
```

```{toctree}
setup
notebooks/quickstart_high
overview
high_level_api
low_level_api
theory
paper
```
