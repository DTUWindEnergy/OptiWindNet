# Setup

## No-install trial

The easiest way to experiment with OptiWindNet is in JupyterLab. Click the ![launch|binder](https://mybinder.org/badge_logo.svg) button at the top of supported pages to launch the corresponding notebook in a cloud-based JupyterLab session, ready to run directly in your browser (via [Binder](https://mybinder.org/)).

## Requirements

*OptiWindNet* has been tested on Windows 10/11 and on Linux systems, but should run on MacOSX as well.

Python version 3.11+ is required. The last version to support Python 3.10 was v0.0.6.

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

The PyPI package installs `ortools` as a dependency.

### If using `conda`

    conda create --name optiwindnet_env --channel conda-forge python=3.12 optiwindnet 
    conda activate optiwindnet_env

The flag `--channel conda-forge` may be omitted if using *miniforge* or if the global *conda* configuration already sets **conda-forge** as the highest-priority channel.

The conda package installs `highspy` as a dependency.

## Interactive use

The **launch|binder** button is an easy way to get started, but a local installation of a notebook interface is recommended for more serious work. Here are some links to comprehensive tutorials on popular Jupyter interfaces:

- [Get Started with JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html)
- [Jupyter Notebooks in VS Code](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
- [How to Use Jupyter Notebook: A Beginner’s Tutorial – Dataquest](https://www.dataquest.io/blog/jupyter-notebook-tutorial/)

## Solvers (optional)

The installation procedure above enables *OptiWindNet*'s heuristics, meta-heuristic, and mathematical optimization with [Google's OR-Tools](https://developers.google.com/optimization) when installed from PyPI, or with [HiGHS](https://highs.dev/) when installed from conda.

Without installing any extra solver package, a PyPI installation of *OptiWindNet* can use `ortools.cp_sat` for CP-SAT, `ortools.gscip` for SCIP, and `ortools.highs` for HiGHS, while a conda installation can use `highs` for HiGHS. The legacy alias `ortools` is still accepted and maps to `ortools.cp_sat`.

Other mathematical optimization backends can also be used, but they must be installed separately.

The commands suggested here assume that the Python environment for *OptiWindNet* has been already activated and that `conda` is configured for the `conda-forge` channel.
For packages that are installable with both `pip` and `conda`, **enter only one** of the commands.

Solvers perform a search across the branch-and-bound tree. On multi-core computers, some solvers parallelize the tree search itself, while others run several coordinated searches in parallel. As of Dec/2025, `gurobi`, `cplex`, `cbc`, and `fscip` support multi-threaded tree search in *OptiWindNet*.

The OR-Tools backends and native `scip` can also benefit from multiple cores by running concurrent searches with some information exchange among them. OR-Tools diversifies algorithms and strategies across workers, while SCIP diversifies random seeds and may vary emphasis settings. Both expose user-configurable controls for that behavior.

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

[HiGHS](https://highs.dev/) can be called from *OptiWindNet* in two ways:

* `ortools.highs`: uses the HiGHS backend exposed through OR-Tools;
* `highs`: uses the native Pyomo + `highspy` backend.

For the PyPI package, `ortools.highs` is available out of the box. The `highs` backend requires `highspy` to be installed separately when it is not already present in the environment:

      pip install highspy
      conda install -c conda-forge highspy

### CBC

[COIN-OR's Optimization Suite](https://coin-or.github.io/user_introduction.html) is open source software and its MILP solver is [coin-or/Cbc: COIN-OR Branch-and-Cut solver](https://github.com/coin-or/Cbc).

Pyomo's interface with CBC is through a system call, so it does not need to be part of a python environment, but Pyomo must be able to find the solver's executable file. Conda has a package for CBC, but it may also be installed by following the instructions in the links above:

    conda install -c conda-forge coin-or-cbc


Users on Windows might find it difficult to get a multi-threaded CBC on that platform (the symptom of single-threaded CBC binary is it not recognizing the `threads` parameter). If that is the case, use the CBC binary from <https://github.com/mdealencar/Cbc/releases>.

### SCIP

[SCIP](https://www.scipopt.org/) can be called from *OptiWindNet* in two ways:

* `ortools.gscip`: uses the SCIP backend exposed through OR-Tools;
* `scip`: uses the native **pyscipopt** backend.

For the PyPI package, `ortools.gscip` is available out of the box. The native `scip` backend requires a separate installation:

> **Attention**: Avoid loading both `scip` and `ortools` solvers within the same Python interpreter instance, since `ortools` contains a SCIP library and its version may be different from the one used by **pyscipopt**.

    pip install pyscipopt
    conda install -c conda-forge pyscipopt

If a call to `WindFarmNetwork().optimize()` or to `Solver.solve()` produces the warning:
> UserWarning: SCIP was compiled without task processing interface. Parallel solve not possible - using optimize() instead of solveConcurrent()

It means that the **pyscipopt** package currently installed was not compiled with multi-threading capability. SCIP will still work, but will under-perform as it is limited to a single core. To overcome that, you will need to upgrade **pyscipopt** to version 6.0.0+, which is multi-threading-capable on all platforms.

### FiberSCIP

FiberSCIP is a parallelized version of SCIP based on the [Ubiquity Generator framework](https://ug.zib.de/index.php#reference). It splits the branch-and-bound search tree among multiple SCIP threads (in a shared-memory system). It is different from SCIP's `concurrentSolve()` in that each thread works on a different part of the tree, reducing duplicate work.

The `'fscip'` solver in *OptiWindNet* is currently **experimental**, use at your own risk. The executable `fscip` must be reachable through the **PATH** environment. The **pyscipopt** package is required (see the SCIP section above), as well as a recent [SCIP Optimization Suite](https://scipopt.org/index.php#download) (10.0.0+). Not all binary distributions of SCIPOptSuite include `fscip`, using one of the precompiled packages from that page is recommended.

## Updating

Activate the Python environment for *OptiWindNet* and enter:

    pip install --upgrade optiwindnet
    conda update optiwindnet
