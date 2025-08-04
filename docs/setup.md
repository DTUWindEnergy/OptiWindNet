# Setup

## Requirements

*OptiWindNet* has been tested on Windows 10/11 and on Linux systems, but should run on MacOSX as well.

A recent Python version (3.10+) is required to run *OptiWindNet*, and the use of a dedicated Python virtual environment is recommended. This can be achieved by installing **either**:

* [Python](https://www.python.org/downloads/), which provides: `venv` virtual environment creator and `pip` package manager;
* or [Miniforge](https://conda-forge.org/download/), which provides: `conda` environment and package manager.

```{admonition} Anaconda and Miniconda
:class: important

[Anaconda or Miniconda](https://www.anaconda.com/download/success) may be used to provide the `conda` manager, as long as the environment is configured to use the **conda-forge** channel.
```

## Installation

The following commands must be run from the system's command line interface (e.g. *git-bash*, *cmd*, *powershell*).

### If using `venv` + `pip`

Create a new venv:

    python -m venv optiwindnet_env

Activate *optiwindnet_env* (choose the one that matches your command prompt):
* cmd: `optiwindnet_env\Scripts\activate.bat`
* bash: `source optiwindnet_env/Scripts/activate`
* powershell: `optiwindnet_env\Scripts\Activate.ps1`

And finally:

    pip install optiwindnet

### If using `conda`

Download <a href="https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/raw/main/environment.yml?ref_type=heads&inline=false">environment.yml</a>, then run:

    conda env create -f environment.yml
    conda activate optiwindnet_env
    pip install optiwindnet

## Optional - Solvers

The installation procedure above enables *OptiWindNet*'s heuristics, meta-heuristic and mathematical optimization with [Google's OR-Tools](https://developers.google.com/optimization) (open-source software).

Other solvers can be used for mathematical optimization, but they are not installed by default.

The commands suggested here assume that the Python environment for *OptiWindNet* has been already activated and that `conda` is configured for the `conda-forge` channel.
For packages that are installable with both `pip` and `conda`, **enter only one** of the commands.

Solvers perform a search accross the branch-and-bound tree. This process can be accelerated in multi-core computers by using concurrent threads, but not all solvers have this feature. As of Mar/2025, only `gurobi`, `cplex` and `cbc` have this multi-threaded search capability. The `ortools` solver also benefits from multi-core systems by launching a portfolio of algorithms in parallel, with some information exchange among them.

### Gurobi

[Gurobi](https://www.gurobi.com/academia/academic-program-and-licenses/) is proprietary software (academic license available). The trial version can only handle very small problems:

    pip install gurobipy
    conda install -c gurobi gurobi

### CPLEX

[IBM ILOG CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio) is proprietary software (academic license available). The Community Edition version can only handle very small problems:

    pip install cplex

### HiGHS

[HiGHS](https://highs.dev/) is open source software:

    pip install highspy
    conda install highspy

### SCIP

[SCIP](https://www.scipopt.org/) is open source software:

    conda install scip

### CBC

[COIN-OR's Optimization Suite](https://coin-or.github.io/user_introduction.html) is open source software and its MILP solver is [coin-or/Cbc: COIN-OR Branch-and-Cut solver](https://github.com/coin-or/Cbc).

Pyomo's interface with CBC is through a system call, so it does not need to be part of a python environment, but Pyomo must be able to find the solver's executable file. Conda has a package for it, but it may also be installed by following the instructions in the links above:

    conda install coin-or-cbc


## Updating

Activate the Python environment for *OptiWindNet* and enter:

    pip install --upgrade optiwindnet
