# Paper

The methodology implemented in *OptiWindNet* is described in the paper titled **"Flexible cable routing framework for wind farm collection system optimization"** <a href="https://authors.elsevier.com/tracking/article/details.do?aid=19691&jid=EOR&surname=Souza%20de%20Alencar">accepted for publication</a> in the *European Journal of Operational Research* on 2025-07-30. The paper's authors are *Mauricio Souza de Alencar*, *Tuhfe Göçmen* and *Nicolaos A. Cutululis*.

If you arrived here looking for *OptiWindNet*, please proceed either to the [](setup.md) (how to install) or to the [Quick Start](notebooks/quickstart_high) (how to use). Continue here to explore the computational experiments of the paper.

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
