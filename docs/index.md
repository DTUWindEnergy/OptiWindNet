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

The tool is distributed as an open-source Python package that is suitable for use within an interactive Python session (e.g. Jupyter notebook). Alternatively, OptiWindNet's API can be invoked directly from another application.

## What can OptiWindNet do?

* Optimize the network of array cables;
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

The OptiWindNet software package can be cited as:
- Mauricio Souza de Alencar, Amir Arasteh and Mikkel Friis-Møller.
_OptiWindNet: An open-source wind farm electrical network optimization tool_,
DTU Wind and Energy Systems, Technical University of Denmark (2025, March),
url: <https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet>

```{code-block} bib
@software{souza_de_alencar_2026_18388439,
  author       = {Souza de Alencar, Mauricio and
                  Arasteh, Amir and
                  Friis-Møller, Mikkel},
  title        = {DTUWindEnergy/OptiWindNet: Release of v0.1.7},
  month        = jan,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v0.1.7},
  doi          = {10.5281/zenodo.18388439},
  url          = {https://doi.org/10.5281/zenodo.18388439},
  swhid        = {swh:1:dir:289eeb68be1214f069d0634666fe0c3bad2df84d
                   ;origin=https://doi.org/10.5281/zenodo.18388438;vi
                   sit=swh:1:snp:8ca52a089ae33194f300b466a6168ef29e58
                   2eb4;anchor=swh:1:rel:c660f0924a7df34ae674a91b178c
                   715c43f6317c;path=DTUWindEnergy-
                   OptiWindNet-9c6db85
                  },
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
