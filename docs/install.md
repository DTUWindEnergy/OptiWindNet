# Install

## No-install trial

The easiest way to experiment with OptiWindNet is in JupyterLab. Click the ![launch|binder](https://mybinder.org/badge_logo.svg) button at the top of supported pages to launch the corresponding notebook in a cloud-based JupyterLab session, ready to run directly in your browser (via [Binder](https://mybinder.org/)).

## Requirements

_OptiWindNet_ has been tested on Windows 10/11 and on Linux systems, but should run on MacOSX as well.

Python version 3.11+ is required. The last version to support Python 3.10 was v0.0.6.

Running _OptiWindNet_ within a dedicated Python virtual environment is recommended. This can be achieved by installing **either**:

- [Python](https://www.python.org/downloads/), which provides: `venv` virtual environment creator and `pip` package manager;
- or [Miniforge](https://conda-forge.org/download/), which provides: `conda` environment and package manager.

```{admonition} Anaconda and Miniconda
:class: important

[Anaconda or Miniconda](https://www.anaconda.com/download/success) may be used to provide the `conda` manager, as long as the environment is configured to use the **conda-forge** channel.
```

## Installation

The following commands must be run from the system's command line interface (e.g. _git-bash_, _cmd_, _powershell_).

### If using `venv` + `pip`

Create a new venv:

    python -m venv optiwindnet_env

Activate _optiwindnet_env_ (choose the one that matches your command prompt):

- cmd: `optiwindnet_env\Scripts\activate.bat`
- bash: `source optiwindnet_env/Scripts/activate`
- powershell: `optiwindnet_env\Scripts\Activate.ps1`

And finally:

    pip install optiwindnet

The PyPI package installs `ortools` as a dependency.

### If using `conda`

    conda create --name optiwindnet_env --channel conda-forge python=3.12 optiwindnet
    conda activate optiwindnet_env

The flag `--channel conda-forge` may be omitted if using _miniforge_ or if the global _conda_ configuration already sets **conda-forge** as the highest-priority channel.

The conda package installs `highspy` as a dependency.

## Interactive use

The **launch|binder** button is an easy way to get started, but a local installation of a notebook interface is recommended for more serious work. Here are some links to comprehensive tutorials on popular Jupyter interfaces:

- [Get Started with JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html)
- [Jupyter Notebooks in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
- [How to Use Jupyter Notebook: A Beginner’s Tutorial – Dataquest](https://www.dataquest.io/blog/jupyter-notebook-tutorial/)

## Updating

Activate the Python environment for _OptiWindNet_ and enter:

    pip install --upgrade optiwindnet
    conda update optiwindnet

## Running the notebooks

Both API sections consist of Jupyter notebooks, which can be viewed online, executed in the cloud through the _Binder_ link at the top of each notebook page, or run on a personal computer with [Jupyter](https://jupyter.org/install) (JupyterLab recommended). The notebooks and the data they read are available in [the repository](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/tree/main/docs/notebooks).

```{admonition} Trust the notebook to see the figures
:class: important

Many of the notebooks have SVG figures as cell outputs, which JupyterLab and Jupyter
Notebook only display if the notebook is marked as trusted. In JupyterLab: press
`Ctrl+Shift+C`, then **Trust Notebook**.
```
