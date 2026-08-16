# Framework Paper

The methodology implemented in _OptiWindNet_ is described in the peer-reviewed scientific article:

- Mauricio Souza de Alencar, Tuhfe Göçmen, Nicolaos A. Cutululis, _Flexible cable routing framework for wind farm collection system optimization_, European Journal of Operational Research, 329(3):1037-1051, 2026, ISSN 0377-2217, <https://doi.org/10.1016/j.ejor.2025.07.069>.

```{code-block} bib
@article{SOUZADEALENCAR20261037,
  title = {Flexible cable routing framework for wind farm collection system optimization},
  journal = {European Journal of Operational Research},
  volume = {329},
  number = {3},
  pages = {1037-1051},
  year = {2026},
  issn = {0377-2217},
  doi = {https://doi.org/10.1016/j.ejor.2025.07.069},
  url = {https://www.sciencedirect.com/science/article/pii/S0377221725005946},
  author = {Mauricio {Souza de Alencar} and Tuhfe Göçmen and Nicolaos A. Cutululis},
  keywords = {Combinatorial optimization, Network design, Collection system, Wind farm},
}
```

If you arrived here looking for the _OptiWindNet_ software package, please proceed either to {doc}`/install` or to the {doc}`Quick Start </notebooks/hi00_quickstart>`. Continue here to explore the computational experiments of the paper. A second article, covered in {doc}`/dataset`, introduces the open database of routing solutions produced with _OptiWindNet_.

The plots and tables in the paper were generated with [this OptiWindNet version](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/tree/cf8420fd1f5ef64e089c9d96012789eaaf0b4e86) (notebooks are inside the project folder **paper**).

This section contains the results and the code to reproduce the computational experiments with the **current OptiWindNet version**. Some small differences with respect to the paper data/figures may be observed, but the results still support the same analysis and conclusions reached there.

Alternatively, the notebooks (`p` series), along with the required _data_ folder, can be [downloaded here](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/tree/main/docs/notebooks/).

```{admonition} Important
:class: important

Many of the jupyter notebooks provided here have SVG figures as cell outputs, which will only be displayed by JupyterLab or Jupyter Notebook if the notebook is marked as trusted (In JupyterLab: press `Ctrl+Shift+C`, then **Trust Notebook**).
```

```{toctree}
:glob:
:maxdepth: 1

notebooks/p0*
```
