# MILP Solvers

Which MILP backends _OptiWindNet_ can drive, how to configure them, and how to install them. What the exact routers do with them is described in [](/routers.md#exact-optimization), and the settings they accept in [](#solver-options).

## Supported solvers

| Solver | Licensing | Identifier |
| --- | --- | --- |
| Google OR-Tools | open source | `'ortools.cp_sat'`, `'ortools.gscip'`, `'ortools.highs'` |
| HiGHS | open source | `'highs'`, `'pyomo.highs'` |
| SCIP | open source | `'scip'` |
| COIN-OR CBC | open source | `'ortools.cbcbox'`, `'pyomo.cbc'` (alias `'cbc'`) |
| FiberSCIP | open source (experimental) | `'fscip'` |
| Gurobi | commercial (academic license available) | `'gurobi'` |
| IBM ILOG CPLEX | commercial (academic license available) | `'cplex'` |

All of them solve the same model, so the choice does not change what counts as a valid solution — but it does change how long the search takes, sometimes by a wide margin on the same instance. If a solve is slower than expected, trying another backend costs a one-word change.

Solvers perform a search across the branch-and-bound tree. On multi-core computers, some solvers parallelize the tree search itself, while others run several coordinated searches in parallel. As of Jul/2026, `gurobi`, `cplex`, `highs`, `pyomo.highs`, `ortools.cbcbox`, `cbc`, and `fscip` support multi-threaded tree search in _OptiWindNet_.

The OR-Tools backends and native `scip` can also benefit from multiple cores by running concurrent searches with some information exchange among them. OR-Tools diversifies algorithms and strategies across workers, while SCIP diversifies random seeds and may vary emphasis settings. Both expose user-configurable controls for that behavior.

The base installation includes a solver backend; additional backends are optional dependencies. The rest of this page describes both.

_In use:_ {doc}`/notebooks/hi23_milp` (Network/Router API) · {doc}`/notebooks/lo23_milp_ortools` and the other MILP notebooks (Advanced API).

## Solver options

Solver options say **how** the solver searches, once the model is already built. They do not change what counts as a valid solution, and they apply to this approach only.

| Option | Effect |
| --- | --- |
| `time_limit` | Maximum solve time, in seconds. |
| `mip_gap` | Optimality tolerance — stop once the gap falls below this, e.g. `0.01` for 1%. |
| `threads` | Number of threads or workers the solver may use. |
| `mip_emphasis` | Whether to prioritize bound quality, feasibility, or integrality. |
| `verbose` | Whether to surface the solver's own log. |

Through the {doc}`/high_level_api`, `time_limit`, `mip_gap` and `verbose` are arguments of {py:class}`MILPRouter <optiwindnet.api.MILPRouter>` itself, while the rest are passed in its `solver_options` mapping.

_OptiWindNet_ sets a handful of solver-specific defaults when a solver is initialized, chosen to suit this problem class; these are readable afterwards from the router or solver object. Every solver accepts many more options than the ones above — consult the solver's own documentation, and pass them through as additional options.

_In use:_ {doc}`/notebooks/hi31_options` (Network/Router API) · {doc}`/notebooks/lo23_milp_ortools` (Advanced API).

## Switching backends in a notebook

OR-Tools bundles native HiGHS and SCIP libraries. Loading it alongside standalone `highspy` or `pyscipopt` can cause a native library conflict, so `solver_factory()` raises `RuntimeError` if a rival package is already imported. Restart the Python interpreter or Jupyter kernel before switching between `ortools*` and native `highs`, `pyomo.highs`, `scip` or `fscip`, or run the backends in separate processes. Switching among `ortools.cp_sat`, `ortools.gscip` and `ortools.highs` does not require a restart.

## Installing a solver

The base installation in {doc}`/install` enables _OptiWindNet_'s constructive heuristics and meta-heuristic, and its exact optimization with [Google's OR-Tools](https://developers.google.com/optimization) when installed from PyPI, or with [HiGHS](https://highs.dev/) when installed from conda.

Without installing any extra solver package, a PyPI installation of _OptiWindNet_ can use `ortools.cp_sat` for CP-SAT, `ortools.gscip` for SCIP, and `ortools.highs` for HiGHS, while a conda installation can use `highs` for HiGHS. The legacy alias `ortools` is still accepted and maps to `ortools.cp_sat`.

Other mathematical optimization backends can also be used, but they must be installed separately.

The commands suggested here assume that the Python environment for _OptiWindNet_ has been already activated and that `conda` is configured for the `conda-forge` channel. For packages that are installable with both `pip` and `conda`, **enter only one** of the commands.

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

[HiGHS](https://highs.dev/) can be called from _OptiWindNet_ in two ways:

- `ortools.highs`: uses the HiGHS backend exposed through OR-Tools;
- `highs`: uses the native **highspy** backend;
- `pyomo.highs`: uses **highspy** through Pyomo.

Of these, only `highs` keeps the solutions found during the search in a pool, from which the one with the shortest routed length is taken (see [](/routers.md#exact-optimization)); the other two take the best-objective solution. The `solver_options` of `highs` are HiGHS option names, passed unchanged to `highspy`.

For the PyPI package, `ortools.highs` is available out of the box. The `highs` and `pyomo.highs` backends require `highspy` to be installed separately when it is not already present in the environment:

      pip install highspy
      conda install -c conda-forge highspy

> **Attention**: Avoid loading both `highs`/`pyomo.highs` and `ortools*` solvers within the same Python interpreter instance, since `ortools` contains a vendored copy of HiGHS and its version may be different from the one used by **highspy**.

### CBC

[COIN-OR's Optimization Suite](https://coin-or.github.io/user_introduction.html) is open source software and its MILP solver is [coin-or/Cbc: COIN-OR Branch-and-Cut solver](https://github.com/coin-or/Cbc).

The recommended way to use CBC is the `ortools.cbcbox` backend: the model is built with OR-Tools (as for the other `ortools.*` backends) and solved by the `cbc` executable bundled in the Python package [cbcbox](https://pypi.org/project/cbcbox/), so no separate executable needs to be installed or placed on the system PATH:

    pip install cbcbox
    conda install -c conda-forge cbcbox

#### Alternative: `pyomo.cbc`

The `pyomo.cbc` backend (also accepted as `cbc`) uses Pyomo. Pyomo's interface with CBC is through a system call, so it does not need to be part of a python environment, but Pyomo must be able to find the solver's executable file. Conda has a package for CBC, but it may also be installed by following the instructions in the links above:

    conda install -c conda-forge coin-or-cbc

Users on Windows might find it difficult to get a multi-threaded CBC on that platform (the symptom of single-threaded CBC binary is it not recognizing the `threads` parameter). If that is the case, use the CBC binary from <https://github.com/mdealencar/Cbc/releases>.

### SCIP

[SCIP](https://www.scipopt.org/) can be called from _OptiWindNet_ in two ways:

- `ortools.gscip`: uses the SCIP backend exposed through OR-Tools;
- `scip`: uses the native **pyscipopt** backend.

For the PyPI package, `ortools.gscip` is available out of the box. The native `scip` backend requires a separate installation:

    pip install pyscipopt
    conda install -c conda-forge pyscipopt

> **Attention**: Avoid loading both `scip` and `ortools*` solvers within the same Python interpreter instance, since `ortools` contains a SCIP library and its version may be different from the one used by **pyscipopt**.

If a call to {py:meth}`WindFarmNetwork.optimize() <optiwindnet.api.WindFarmNetwork.optimize>` or to {py:meth}`Solver.solve() <optiwindnet.MILP.Solver.solve>` produces the warning:

> UserWarning: SCIP was compiled without task processing interface. Parallel solve not possible - using optimize() instead of solveConcurrent()

It means that the **pyscipopt** package currently installed was not compiled with multi-threading capability. SCIP will still work, but will under-perform as it is limited to a single core. To overcome that, you will need to upgrade **pyscipopt** to version 6.0.0+, which is multi-threading-capable on all platforms.

### FiberSCIP

FiberSCIP is a parallelized version of SCIP based on the [Ubiquity Generator framework](https://ug.zib.de/index.php#reference). It splits the branch-and-bound search tree among multiple SCIP threads (in a shared-memory system). It is different from SCIP's `concurrentSolve()` in that each thread works on a different part of the tree, reducing duplicate work.

The `'fscip'` solver in _OptiWindNet_ is currently **experimental**, use at your own risk. The executable `fscip` must be reachable through the **PATH** environment. The **pyscipopt** package is required (see the SCIP section above), as well as a recent [SCIP Optimization Suite](https://scipopt.org/index.php#download) (10.0.0+). Not all binary distributions of SCIPOptSuite include `fscip`, using one of the precompiled packages from that page is recommended.
