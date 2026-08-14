(methods)=
# Solving Methods & Options

*OptiWindNet* offers three families of solution methods and a common set of options that shape the problem they are asked to solve.
Both APIs expose the same methods and the same options — the {doc}`high_level_api` wraps each family in a `Router` class, while the {doc}`low_level_api` calls the underlying functions directly.
This section describes what the methods do and when to choose them; the API sections show how to invoke them.

(methods-overview)=
## The three families

| Family | Method | Typical runtime | Quality | Topologies |
| --- | --- | --- | --- | --- |
| Constructive heuristic | Esau-Williams variants | sub-second | no guarantee, usually within a few percent | branched, radial, ringed |
| Meta-heuristic | HGS-CVRP, LKH-3 | seconds to minutes (you set the budget) | no guarantee, typically better than the heuristic | radial, ringed |
| Exact optimization | MILP models solved by branch-and-cut | minutes to hours | bounded — the solver reports an optimality gap | branched, radial, ringed |

The families are complementary rather than competing: a heuristic solution is a valid starting point for a meta-heuristic or a MILP solve, and chaining them is the normal way to get a good solution quickly.
See {ref}`methods-warmstart`.

```{admonition} What "quality" means here
:class: note

Only the exact methods can certify how far a solution is from optimal. A heuristic result
may well *be* optimal — you simply have no way to know. Compare methods on routesets
rather than topologies, since detours added during routing change the total length; see
{ref}`problem-crossings`.
```

(methods-constructive)=
## Constructive heuristics

These build a solution incrementally, starting from every terminal connected directly to a root and repeatedly merging subtrees while capacity allows.
They are extensions of the Esau-Williams heuristic for the CMSTP, modified to account for cable crossings.
They run in a fraction of a second even on large layouts, which makes them the right default for interactive work, for warm-starting the slower methods, and for use inside an outer optimization loop where the network is re-solved on every iteration.

The variants differ in how they break ties and how strongly they bias growth towards the root:

| Method | Behavior | Replaces |
| --- | --- | --- |
| `'biased_EW'` | Esau-Williams with a bias towards moving radially (root-ward) on quasi-ties. The default. | `CPEW` |
| `'esau_williams'` | The classic Esau-Williams C-MST heuristic, modified to avoid crossings. | `ClassicEW` |
| `'rootlust'` | A tunable root-ward bias that increases as remaining capacity decreases. | `OBEW` |
| `'radial_EW'` | Produces radial subtrees — simple paths from the root. | `NBEW` |
| `'ringed'` | Closes each subtree into a ring: both endpoints connect to the same root, joined at a zero-load link. | — |

The *Replaces* column refers to the removed legacy entry points; see {doc}`notebooks/lo34_legacy_heuristics` for migration.

*In use:* {doc}`notebooks/hi20_heuristic` (Network/Router API) · {doc}`notebooks/lo20_heuristic` (Advanced API).

(methods-metaheuristic)=
## Meta-heuristics

Meta-heuristics search the solution space under a time budget you set.
They treat the problem as a capacitated vehicle routing problem (CVRP), which is why they produce radial topologies by default: a CVRP route is a path, not a tree.
Solving the *closed* CVRP instead, where every route returns to the depot, yields a ringed topology.

Both wrappers handle multiple substations by clustering the terminals by root and solving one instance per cluster, and both repair crossings iteratively after the search.
That repair work is bounded by a retry count, so total wall time can exceed the time limit you set — with `max_retries` retries, up to `(max_retries + 1) × time_limit`.

### HGS-CVRP

[vidalt/HGS-CVRP](https://github.com/vidalt/HGS-CVRP) is a modern implementation of the hybrid genetic search algorithm specialized to the CVRP, including an additional neighborhood called SWAP\*.
It is described in [Vidal (2022)](https://doi.org/10.1016/j.cor.2021.105643) and reached through the Python bindings [mdealencar/HybGenSea](https://github.com/mdealencar/HybGenSea).
It is bundled with *OptiWindNet*, so it needs no separate installation.

Its distinctive options concern the feeder count:

* the feeder limit is normally an **upper bound** — the search is free to use fewer, and normally settles at the minimum feasible number;
* pinning the count to that limit exactly additionally requires balanced subtrees and a single substation, and rules out the ringed topology;
* balancing makes subtree loads differ by at most one terminal;
* with multiple substations the feeder limit is ignored and the count is fixed to the minimum required;
* a seed is available for reproducible runs.

Because HGS produces radial topologies, and radial is a special case of branched, its solutions can warm-start both branched and radial models.

### LKH-3

[LKH-3](http://akira.ruc.dk/~keld/research/LKH-3/) is K. Helsgaun's extension of the Lin-Kernighan-Helsgaun TSP solver to constrained TSP and vehicle routing problems.

Unlike HGS-CVRP, it is **not bundled**: *OptiWindNet* interfaces with it through temporary files and system calls, so the `LKH` executable must be on the `PATH` as seen from the Python process.
Keld Helsgaun distributes it as C source code and as a Windows binary, for academic and non-commercial use.

Its options cover the search itself — number of runs, a per-run time limit, a pseudo-random seed — plus whether to fill missing candidate links with direct Euclidean links or to restrict the search to the allowed graph `A`.

*In use:* {doc}`notebooks/hi21_hgs` (Network/Router API) · {doc}`notebooks/lo21_hgs`, {doc}`notebooks/lo22_lkh` (Advanced API).

(methods-exact)=
## Exact optimization

The problem is formulated as a mixed-integer linear program and handed to a branch-and-cut solver.
This is the only family that quantifies solution quality: the solver maintains a bound on the best possible solution and reports the remaining **optimality gap**.
Stopping at a 1% gap means the solution is provably within 1% of optimal, which is usually a better use of time than proving optimality outright.

The exact methods honor every option in {ref}`methods-model-options`, which the other families only partly support.

### Available solvers

| Solver | Licensing | Identifier |
| --- | --- | --- |
| Google OR-Tools | open source | `'ortools.cp_sat'`, `'ortools.gscip'`, `'ortools.highs'` |
| HiGHS | open source | `'highs'` |
| SCIP | open source | `'scip'` |
| COIN-OR CBC | open source | `'cbc'` |
| Gurobi | commercial (academic license available) | `'gurobi'` |
| IBM ILOG CPLEX | commercial (academic license available) | `'cplex'` |

Solvers are optional dependencies and are installed separately — see {doc}`setup` for per-solver instructions.
The remaining arguments (time limit, gap, verbosity) are the same across solvers, so switching between them is a one-word change.

*In use:* {doc}`notebooks/hi23_milp` (Network/Router API) · {doc}`notebooks/lo23_milp_ortools` and the other MILP notebooks (Advanced API).

(methods-model-options)=
## Model options

Model options control **what is solved** — the structure of the problem and the constraints imposed on any acceptable solution.
They are independent of the solver and, in principle, meaningful for every family, though not every family can enforce all of them.

| Option | Values | Meaning |
| --- | --- | --- |
| `topology` | `'branched'` (default), `'radial'`, `'ringed'` | The architecture of the subtrees; see {ref}`problem-topologies`. |
| `feeder_route` | `'segmented'` (default), `'straight'` | Whether feeder routes may be detoured (`'segmented'`) or must run straight. Straight feeders tend to give more direct connections, but routes may still bend around exclusion zones. |
| `feeder_limit` | `'unlimited'` (default), `'minimum'`, `'min_plus1'`, `'min_plus2'`, `'min_plus3'`, `'specified'`, `'exactly'` | How many feeders the solution may use. `'minimum'` is the fewest possible given the terminal count and cable capacity; the `min_plusN` values allow N more than that. |
| `max_feeders` | integer | Required by `'specified'` (an upper bound) and `'exactly'` (an exact count). For a ringed topology this counts physical substation connections, so it must be a multiple of 2 — each ring uses two. |
| `balanced` | `False` (default), `True` | Whether subtree loads must be balanced, i.e. differ by at most one terminal. Only enforceable when the feeder count is pinned to a single value, that is with `'minimum'` or `'exactly'`. |

### What each family can enforce

| Family | Topology | Feeder route | Feeder limit | Balanced |
| --- | --- | --- | --- | --- |
| Constructive heuristic | via the method chosen — `'radial_EW'` for radial, `'ringed'` for rings, others branched | yes | no | no |
| HGS-CVRP | radial, or ringed on request | no | yes, as an upper bound | yes |
| MILP | yes | yes | yes | yes |

A method that cannot enforce an option does not fail when given one — it simply produces a solution that may not satisfy it.
This matters when chaining methods, which is the subject of the next section.

*In use:* {doc}`notebooks/hi30_options` (Network/Router API) · {doc}`notebooks/lo29_topologies` (Advanced API).

(methods-solver-options)=
## Solver options

Solver options control **how** the MILP solver searches, once the model is already built.
They do not change what counts as a valid solution.

| Option | Effect |
| --- | --- |
| `time_limit` | Maximum solve time, in seconds. |
| `mip_gap` | Optimality tolerance — stop once the gap falls below this, e.g. `0.01` for 1%. |
| `threads` | Number of threads or workers the solver may use. |
| `mip_emphasis` | Whether to prioritize bound quality, feasibility, or integrality. |
| `verbose` | Whether to surface the solver's own log. |

*OptiWindNet* sets a handful of solver-specific defaults when a solver is initialized, chosen to suit this problem class; these are readable afterwards from the router or solver object.
Every solver accepts many more options than the ones above — consult the solver's own documentation, and pass them through as additional options.

```{admonition} Model options or solver options?
:class: tip

Use **model options** to say *what kind of solution* you want: structure, constraints,
flexibility. Use **solver options** to say *how long and how hard* the solver should try
to find it.
```

(methods-warmstart)=
## Warm-starting

A feasible solution supplied up front lets a MILP solver start from a known bound instead of searching for its first incumbent, which usually shortens the time to a good gap considerably.
The natural chain is fast to slow: a constructive heuristic or a meta-heuristic produces a solution, and the MILP model starts from it.

A warm start is only usable if it satisfies the constraints of the model being started.
This is where the capability gaps of the previous section become concrete:

| Model option | Value | Usable warm starts |
| --- | --- | --- |
| `feeder_limit` | `'unlimited'` | any method |
| | `'minimum'` | meta-heuristic only |
| `feeder_route` | `'straight'` | any method |
| | `'segmented'` | any method |
| `topology` | `'branched'` | any method |
| | `'radial'` | meta-heuristic only |

The two restricted cases have the same cause: a constructive heuristic does not limit the number of feeders and does not produce radial subtrees unless asked for the radial variant, so its output violates those two models.
A meta-heuristic solution is radial and uses the minimum feeder count, so it satisfies both.

*In use:* {doc}`notebooks/hi40_example_taylor_2023` (Network/Router API) · {doc}`notebooks/lo40_example_taylor_2023` (Advanced API).

(methods-choosing)=
## Choosing a method

* **Interactive exploration, or a network re-solved inside an outer loop** — constructive heuristic. Sub-second, and the quality is adequate for comparing layouts against each other.
* **A good network without a long wait** — meta-heuristic with a modest time limit. Check the reported solution times: if raising the limit stops improving the length, lower it and save the time.
* **A network you intend to defend** — MILP, warm-started by one of the above, stopped at an optimality gap you consider acceptable.
* **A specific structure is required** — ringed for redundancy, radial to avoid branching at turbines, a pinned feeder count to match available switchgear — check {ref}`methods-model-options` for which families can enforce it, and use MILP when the constraint must be guaranteed.
