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
Version 1.0.0

`Mauricio Souza de Alencar, Amir Arasteh and Mikkel Friis-Møller. (2025, March).
OptiWindNet 1.0.0: An open-source wind farm electrical network optimization tool. DTU Wind, Technical University of Denmark.`

```{code-block} bib
	@article{
    	    optiwindnet1.0.0_2025,
    	    title={OptiWindNet 1.0.0: An open-source wind farm electrical network optimization tool},
    	    author={Mauricio Souza de Alencar, Amir Arasteh and Mikkel Friis-Møller},
    	    url="https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet",
    	    publisher={DTU Wind, Technical University of Denmark},
    	    year={2025},
    	    month={3}
	    }
```
```{toctree}

setup
notebooks/quickstart_high
high_level_api
low_level_api
theory
paper
```
