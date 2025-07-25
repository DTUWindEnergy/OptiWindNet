[![pipeline status](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/badges/main/pipeline.svg)](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/commits/main)
[![PyPi](https://img.shields.io/pypi/v/optiwindnet)](https://pypi.org/project/optiwindnet/)
[![License](https://img.shields.io/pypi/l/optiwindnet)](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/blob/main/LICENSE)
<!---
[![coverage report](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/badges/main/coverage.svg)](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/commits/main)
[![DOI](https://zenodo.org/badge/164115313.svg)](https://zenodo.org/badge/latestdoi/164115313)
-->

OptiWindNet
===========

Tool for designing and optimizing the electrical cable network (collection system) for offshore wind power plants.

Documentation: [https://optiwindnet.readthedocs.io](https://optiwindnet.readthedocs.io)
- [Quickstart](https://optiwindnet.readthedocs.io/stable/notebooks/Quickstart.html)
- [Download the Jupyter notebooks](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/tree/main/docs/notebooks) used in the documentation.
- [Report an issue](https://github.com/DTUWindEnergy/OptiWindNet/issues) (mirror of OptiWindNet on GitHub)
- [API Reference](https://optiwindnet.readthedocs.io/stable/autoapi/index.html)
- [How to Cite](https://optiwindnet.readthedocs.io/stable/index.html#how-to-cite)

Installation
------------

```
pip install optiwindnet
```

Detailed instructions in [Installation](https://optiwindnet.readthedocs.io/stable/setup.html#installation).

Requirements
------------

Python 3.10+. The use of a Python virtual environment is recommended. OptiWindNet's dependencies will be installed automatically when using `pip install optiwindnet`.

One may pre-install the dependencies in a python environment by using either:
- [requirements.txt](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/raw/main/requirements.txt?ref_type=heads&inline=false): `pip install -r requirements.txt`
- [environment.yml](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/raw/main/environment.yml?ref_type=heads&inline=false): `conda env create -f environment.yml` (name: *optiwindnet_env*)

Acknowledgements
----------------

The heuristics implemented in this repository (release 0.0.1) are presented and analyzed in the MSc thesis [Optimization heuristics for offshore wind power plant collection systems design](https://fulltext-gateway.cvt.dk/oafilestore?oid=62dddf809a5e7116caf943f3&targetid=62dddf80a41ba354e4ed35bc) (DTU Wind - Technical University of Denmark, July 4, 2022)

The meta-heuristic used is [vidalt/HGS-CVRP: Modern implementation of the hybrid genetic search (HGS) algorithm specialized to the capacitated vehicle routing problem (CVRP). This code also includes an additional neighborhood called SWAP\*.](https://github.com/vidalt/HGS-CVRP) via its Python bindings [chkwon/PyHygese: A Python wrapper for the Hybrid Genetic Search algorithm for Capacitated Vehicle Routing Problems (HGS-CVRP)](https://github.com/chkwon/PyHygese).

The cable routing relies on a navigation mesh generated by the library [artem-ogre/CDT: Constrained Delaunay Triangulation (C++)](https://github.com/artem-ogre/CDT) via its Python bindings - [artem-ogre/PythonCDT: Constrained Delaunay Triangulation (Python)](https://github.com/artem-ogre/PythonCDT).
