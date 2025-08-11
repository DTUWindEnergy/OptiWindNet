# Paper

The methodology implemented in *OptiWindNet* is described in the peer-reviewed cientific article:
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

If you arrived here looking for the *OptiWindNet* software package, please proceed either to the [](setup.md) (how to install) or to the [Quick Start](notebooks/quickstart_high) (how to use). Continue here to explore the computational experiments of the paper.

The plots and tables in the paper were generated with [this OptiWindNet version](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/tree/cf8420fd1f5ef64e089c9d96012789eaaf0b4e86) (notebooks are inside the project folder **paper**). This section contains the results and the code to reproduce the computational experiments with the **current OptiWindNet version**. Some small differences with respect to the paper data/figures may be observed, but the results still support the same analysis and conclusions reached there. 

Alternatively, the notebooks (40 series), along with the required *data* folder, can be [downloaded here](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/tree/main/docs/notebooks/).

```{admonition} Important
:class: important

Many of the jupyter notebooks provided here have SVG figures as cell outputs, which will only be displayed by JupyterLab or Jupyter Notebook if the notebook is marked as trusted (In JupyterLab: press `Ctrl+Shift+C`, then **Trust Notebook**).
```

```{toctree}
:glob:
:maxdepth: 1

notebooks/4*
```
