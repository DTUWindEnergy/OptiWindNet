Requirements
============
A recent Python version (3.10+) is required to run *OptiWindNet*, and we recommend to install it in its own virtual environment. This can be achieved by installing **either**:

* `Python <https://www.python.org/downloads/>`_ and using the built-ins ``venv`` virtual environment creator and ``pip`` package manager;
* or `Miniforge <https://conda-forge.org/download/>`_ (`Anaconda or Miniconda <https://www.anaconda.com/download/success>`_ also work) and using ``conda`` to create and populate the virtual environment.

In the near future OptiWindNet will be turned into a Python package installable with the usual package managers, but currently it must be installed from the project tree obtained via ``git``. This software can be obtained from `Git <https://git-scm.com/downloads>`_ for a standalone version or from `Git for Windows <https://gitforwindows.org/>`_ to get bundle of git and other useful tools for the Windows platform (recommended).

Installation
============
The following commands must be run from the system's command line interface (e.g. *git-bash*, *cmd*, *powershell*), first making sure that `git` and `python` or `conda` are available (see section above)::

    git clone https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet.git

If using ``venv``/``pip``
-------------------------

Run::

    python -m venv optiwindnet_env

* cmd: ``optiwindnet_env\Scripts\activate.bat``
* bash: ``source optiwindnet_env/Scripts/activate``
* powershell: ``optiwindnet_env\Scripts\Activate.ps1``

Then::

    pip install -r OptiWindNet/requirements.txt
    pip install --editable OptiWindNet/


If using ``conda``
------------------

Run::

    conda env create -f OptiWindNet/environment.yml
    activate optiwindnet_env
    pip install --editable OptiWindNet/


Optional - Solvers
==================

The installation procedure above enables *OptiWindNet*'s heuristics, meta-heuristic and mathematical optimization with `Google's OR-Tools <https://developers.google.com/optimization>`_ (open-source software).

Other solvers can be used for mathematical optimization, but they are not installed by default.
See the documentation section **Solvers** for relevant parameters when calling each solver.

The commands suggested here assume that the Python environment for *OptiWindNet* has been already activated.
For packages that are installable with both ``pip`` and ``conda``, only one of the commands.

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

Run::

    cd OptiWindNet
    git pull

If ``pip`` was given the ``--editable`` option when installing, the new version will be immediately available within the Python environment ``optiwindnet_env``.
