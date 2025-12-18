# Setup

## Requirements

*OptiWindNet* has been tested on Windows 10/11 and on Linux systems, but should run on MacOSX as well.

Python version 3.11 or 3.12 is recommended to run *OptiWindNet*. Python 3.13+ may cause issues with the `optiwindnet.db` module, but all other features work fine. The last version to support Python 3.10 was v0.1.0.

Running *OptiWindNet* within a dedicated Python virtual environment is recommended. This can be achieved by installing **either**:

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

    conda create --name optiwindnet_env --channel conda-forge python=3.12 optiwindnet 
    conda activate optiwindnet_env

The flag `--channel conda-forge` may be omitted if using *miniforge* or if the global *conda* configuration already sets **conda-forge** as the highest-priority channel.

## Optional - Solvers

The installation procedure above enables *OptiWindNet*'s heuristics, meta-heuristic and mathematical optimization with [Google's OR-Tools](https://developers.google.com/optimization) (open-source software).

Other solvers can be used for mathematical optimization, but they are not installed by default.

The commands suggested here assume that the Python environment for *OptiWindNet* has been already activated and that `conda` is configured for the `conda-forge` channel.
For packages that are installable with both `pip` and `conda`, **enter only one** of the commands.

Solvers perform a search across the branch-and-bound tree. This process can be accelerated in multi-core computers by using concurrent threads, but not all solvers have this feature. As of Dec/2025, only `gurobi`, `cplex`, `cbc` and `fscip` have this multi-threaded search capability.

Solvers `ortools` and `scip` also benefit from multi-core systems by launching multiple concurrent solvers in parallel, with some information exchange among them. `ortools` diversifies the algorithms/strategies among threads, while `scip` diversifies the random seeds and may use different emphasis among threads. Both have several user-configurable settings regarding that diversification.

For installing all pip-available solvers:

    pip install optiwindnet[solvers]

See below for specific instructions for each solver.

### Gurobi

[Gurobi](https://www.gurobi.com/academia/academic-program-and-licenses/) is proprietary software (academic license available). The trial version can only handle very small problems:

    pip install gurobipy
    conda install -c gurobi gurobi

### CPLEX

[IBM ILOG CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio) is proprietary software (academic license available). The Community Edition version can only handle very small problems:

    pip install cplex
    conda install -c IBMDecisionOptimization cplex

### HiGHS

[HiGHS](https://highs.dev/) is open source software:

    pip install highspy
    conda install -c conda-forge highspy

### CBC

[COIN-OR's Optimization Suite](https://coin-or.github.io/user_introduction.html) is open source software and its MILP solver is [coin-or/Cbc: COIN-OR Branch-and-Cut solver](https://github.com/coin-or/Cbc).

Pyomo's interface with CBC is through a system call, so it does not need to be part of a python environment, but Pyomo must be able to find the solver's executable file. Conda has a package for CBC, but it may also be installed by following the instructions in the links above:

    conda install -c conda-forge coin-or-cbc


### SCIP

[SCIP](https://www.scipopt.org/) is open source software.

> **Attention**: Avoid loading both `scip` and `ortools` solvers within the same Python interpreter instance, since `ortools` contains a SCIP library and its version may be different from the one used by **pyscipopt**.

    pip install pyscipopt
    conda install -c conda-forge pyscipopt

Note that these **pyscipopt** packages may not have been compiled with multi-threading capability. If you get the warning:
```
optiwindnet\MILP\scip.py:96: UserWarning: SCIP was compiled without task processing interface. Parallel solve not possible - using optimize() instead of solveConcurrent()
  model.solveConcurrent()
```
Then SCIP will still work, but will under-perform as it is limited to a single core. To overcome that, you will need to replace your current **pyscipopt** with a multi-threading-enabled one. PyPI-distributed **pyscipopt** version 6.0.0 is multi-threading-capable on all platforms (**recommended**). The one distributed via conda-forge is still at version 5.6.0 and is **not** multi-threading-capable.

### FiberSCIP

FiberSCIP is a parallelized version of SCIP based on the [Ubiquity Generator framework](https://ug.zib.de/index.php#reference). It splits the branch-and-bound search tree among multiple SCIP threads (in a shared-memory system). It is different from SCIP's `concurrentSolve()` in that each thread works on a different part of the tree, reducing duplicate work.

The `'fscip'` solver in *OptiWindNet* is currently **experimental**, use at your own risk. The executable `fscip` must be reachable through the **PATH** environment. The **pyscipopt** package is required (see the SCIP section above), as well as a recent [SCIP Optimization Suite](https://scipopt.org/index.php#download) (10.0.0+). Not all binary distributions of SCIPOptSuite include `fscip`, using one of the precompiled packages from that page is recommended.

## Updating

Activate the Python environment for *OptiWindNet* and enter:

    pip install --upgrade optiwindnet
    conda update optiwindnet
