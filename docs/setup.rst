.. _Requirements:

Requirements
============

* Python
* Git
* C/C++ Build Utilities

*OptiWindNet* has been tested on Windows 10 and on Linux systems, but should run on MacOSX as well.

Python
------
A recent Python version (3.10+) is required to run *OptiWindNet*, and the use of a dedicated Python virtual environment is recommended. This can be achieved by installing **either**:

* `Python <https://www.python.org/downloads/>`_, which provides: ``venv`` virtual environment creator and ``pip`` package manager;
* or `Miniforge <https://conda-forge.org/download/>`_ (`Anaconda or Miniconda <https://www.anaconda.com/download/success>`_ also work), which provides: ``conda`` environment and package manager.

Git
---

*OptiWindNet* and one of its dependencies (*PythonCDT*) are installable only from their source code repositories and require the program ``git``. This software can be obtained from `Git <https://git-scm.com/downloads>`_ for a standalone version or from `Git for Windows <https://gitforwindows.org/>`_ to get bundle of git and other useful tools for the Windows platform (recommended).

C Build Utilities
-----------------
Two of *OptiWindNet*'s dependencies (*PythonCDT* and *hygese*) require C/C++ source code to be compiled at installation time. If ``gcc``, ``g++``, ``cmake`` and ``make`` commands are already available in the target system, no further action is necessary. If not, the easiest way to fullfill that requirement on Windows (as a regular user) is using `Scoop <https://scoop.sh/>`_.

To install Scoop, open a PowerShell prompt and enter::

    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression

Next, use ``scoop`` to install `WinLibs <https://winlibs.com/>`_::

    scoop install mingw-winlibs

Open a new shell to ensure that the updated system PATH is loaded. To check the build system is ready, enter these commands::

    gcc --version
    make --version
    cmake --version

.. _Installation:

Installation
============
The following commands must be run from the system's command line interface (e.g. *git-bash*, *cmd*, *powershell*).

If using ``venv``/``pip``
-------------------------

Run::

    python -m venv optiwindnet_env

* cmd: ``optiwindnet_env\Scripts\activate.bat``
* bash: ``source optiwindnet_env/Scripts/activate``
* powershell: ``optiwindnet_env\Scripts\Activate.ps1``

Then::

    pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet.git

If using ``conda``
------------------

Download `environment.yml <https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/raw/main/environment.yml?ref_type=heads&inline=false>`_, then run::

    conda env create -f environment.yml
    conda activate optiwindnet_env
    pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet.git


Running
=======

*OptiWindNet* is not an application and has no *main* program to be executed. The recommended way to use it is in an interactive Python notebook such as `JupyterLab <https://jupyterlab.readthedocs.io/en/latest/>`_ or the `Jupyter Extension for Visual Studio Code <https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter>`_.

Optional - Solvers
==================

The installation procedure above enables *OptiWindNet*'s heuristics, meta-heuristic and mathematical optimization with `Google's OR-Tools <https://developers.google.com/optimization>`_ (open-source software).

Other solvers can be used for mathematical optimization, but they are not installed by default.
See the documentation section **Solvers** for relevant parameters when calling each solver.

The commands suggested here assume that the Python environment for *OptiWindNet* has been already activated.
For packages that are installable with both ``pip`` and ``conda``, **enter only one** of the commands.

Solvers perform a search accross the branch-and-bound tree. This process can be accelerated in multi-core computers by using concurrent threads, but not all solvers have this feature. As of Mar/2025, only `gurobi`, `cplex` and `cbc` have this multi-threaded search capability. The `ortools` solver also benefits from multi-core systems by launching a portfolio of algorithms in parallel, with some information exchange among them.

Gurobi
------

`Gurobi <https://www.gurobi.com/academia/academic-program-and-licenses/>`_ is proprietary software (academic license available). The trial version can only handle very small problems.::

    pip install gurobipy
    conda install -c gurobi gurobi

CPLEX
-----

`IBM ILOG CPLEX <https://www.ibm.com/products/ilog-cplex-optimization-studio>`_ is proprietary software (academic license available). The Community Edition version can only handle very small problems.::

    pip install cplex

HiGHS
-----

`HiGHS <https://highs.dev/>`_ is open source software.::

    pip install highspy
    conda install highspy

SCIP
----

`SCIP <https://www.scipopt.org/>`_ is open source software.::

    conda install scip

CBC
---

`COIN-OR's Optimization Suite <https://coin-or.github.io/user_introduction.html>`_ is open source software and its MILP solver is `coin-or/Cbc: COIN-OR Branch-and-Cut solver <https://github.com/coin-or/Cbc>`_.

Pyomo's interface with CBC is through a system call, so it does not need to be part of a python environment, but Pyomo must be able to find the solver's executable file. Conda has a package for it, but it may also be installed by following the instructions in the links above.::

    conda install coin-or-cbc


Updating
========

Activate the Python environment for *OptiWindNet* and enter::

    pip install --upgrade --force-reinstall git+https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet.git
