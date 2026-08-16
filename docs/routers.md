# Routers

_OptiWindNet_ offers three optimization approaches. Both APIs expose the same ones — the {doc}`/high_level_api` wraps each approach in a {py:class}`Router <optiwindnet.api.Router>` class, while the {doc}`/low_level_api` calls the underlying functions directly. This page describes what the routers do, which settings they read and when to choose them; the API sections show how to invoke them.

## Optimization approaches

| Approach | Implementations | Typical runtime | Quality | Topologies |
| --- | --- | --- | --- | --- |
| Constructive heuristic | Esau-Williams variants | sub-second | no guarantee, usually within a few percent | branched, radial, ringed |
| Meta-heuristic | HGS-CVRP, LKH-3 | seconds to minutes (you set the budget) | no guarantee, typically better than the heuristic | radial, ringed |
| Exact optimization | MILP models solved by branch-and-cut | minutes to hours | bounded — the solver reports an optimality gap | branched, radial, ringed |

The approaches are complementary rather than competing: a heuristic solution is a valid starting point for a meta-heuristic or a MILP solve, and chaining them is the normal way to get a good solution quickly. See [](/routers.md#warm-starting). They also differ in which constraints they can actually enforce; the matrix is in [](/routers.md#what-each-approach-can-enforce).

```{admonition} What "quality" means here
:class: note

Only the exact routers can certify how far a solution is from optimal. A heuristic result
may well *be* optimal — you simply have no way to know. Compare routers on routesets
rather than topologies, since detours added during path-finding change the total length; see
[](/problem.md#crossings-contours-and-detours).
```

_In use:_ {doc}`/notebooks/hi20_heuristic` (Network/Router API) · {doc}`/notebooks/hi21_hgs` (Network/Router API) · {doc}`/notebooks/hi23_milp` (Network/Router API).

### What each approach can enforce

| Approach | Topology | Feeder route | Feeder limit | Balanced |
| --- | --- | --- | --- | --- |
| Constructive heuristic | via the `method` chosen — `'radial_EW'` for radial, `'ringed'` for rings, others branched | yes | no | no |
| HGS-CVRP | radial, or ringed on request | no | yes, as an upper bound | yes |
| MILP | yes | yes | yes | yes |

A router that cannot enforce an option does not fail when given one — it simply produces a solution that may not satisfy it. This matters when chaining routers; see [](/routers.md#warm-starting).

_In use:_ {doc}`/notebooks/hi31_options` (Network/Router API) · {doc}`/notebooks/lo30_topologies` (Advanced API).

## Model options

Model options say **what is solved**: the structure of the problem, and the constraints any acceptable solution has to meet. They are independent of the solver, and in principle meaningful for every approach — subject to the table above.

- `topology` picks the architecture of the subtrees. The three are defined in [](/problem.md#network-topologies).
- `feeder_route` decides whether feeder routes may be detoured or must run straight. Straight feeders tend to give more direct connections, but the routes may still bend around exclusion zones.
- `feeder_limit`, together with `max_feeders`, bounds how many feeders the solution may use: the fewest the terminal count and cable capacity allow, a small number more than that, an upper bound, or an exact count. For a ringed topology `max_feeders` counts physical substation connections, so it must be a multiple of 2 — each ring uses two.
- `balanced` requires the subtree loads to differ by at most one terminal. It is only enforceable when the feeder count is pinned to a single value.

Only {py:class}`MILPRouter <optiwindnet.api.MILPRouter>` accepts these as a {py:class}`ModelOptions <optiwindnet.MILP.ModelOptions>` mapping; {py:class}`EWRouter <optiwindnet.api.EWRouter>` and {py:class}`HGSRouter <optiwindnet.api.HGSRouter>` expose the subset they can enforce as ordinary constructor arguments instead.

The permitted values and the defaults are deliberately not repeated here: {py:meth}`ModelOptions.help() <optiwindnet.MILP.ModelOptions.help>` prints them from the library itself, and {doc}`/notebooks/hi31_options` shows that output alongside what each setting does to a solution.

_In use:_ {doc}`/notebooks/hi31_options` (Network/Router API) · {doc}`/notebooks/lo30_topologies` (Advanced API).

## Constructive heuristics

These build a solution incrementally, starting from every terminal connected directly to a root and repeatedly merging subtrees while capacity allows. They are extensions of the Esau-Williams heuristic for the CMSTP, modified to account for cable crossings. They run in a fraction of a second even on large layouts, which makes them the right default for interactive work, for warm-starting the slower routers, and for use inside an outer optimization loop where the network is re-solved on every iteration.

The variants differ in how they break ties and how strongly they bias growth towards the root. They are selected by the `method` argument, which names a variant of the constructive-heuristic approach — it never selects among the optimization approaches:

| `method` | Behavior | Replaces |
| --- | --- | --- |
| `'biased_EW'` | Esau-Williams with a rootward bias in near-tie cases. The default for {py:class}`EWRouter <optiwindnet.api.EWRouter>`. | `CPEW` |
| `'esau_williams'` | The classic Esau-Williams C-MST heuristic, modified to avoid crossings. | `ClassicEW` |
| `'rootlust'` | A configurable rootward bias that increases as remaining capacity decreases. The default for {py:func}`constructor() <optiwindnet.heuristics.constructor>`. | `OBEW` |
| `'radial_EW'` | Produces radial subtrees — simple paths from the root. | `NBEW` |
| `'ringed'` | Closes each subtree into a ring: both endpoints connect to the same root, joined at a zero-load link. | — |

The two entry points use different default variants and may therefore produce different networks for the same input.

The _Replaces_ column refers to the removed entry points; see {doc}`/notebooks/lo90_removed_heuristics` for migration.

_In use:_ {doc}`/notebooks/hi20_heuristic` (Network/Router API) · {doc}`/notebooks/lo20_heuristic` (Advanced API).

## Meta-heuristics

Meta-heuristics search the solution space under a time budget you set. They treat the problem as a capacitated vehicle routing problem (CVRP), which is why they produce radial topologies by default: a CVRP route is a path, not a tree. Solving the _closed_ CVRP instead, where every route returns to the depot, yields a ringed topology.

Both wrappers handle multiple substations by clustering the terminals by root and solving one instance per cluster, and both repair crossings iteratively after the search. That repair work is bounded by a retry count, so total wall time can exceed the time limit you set — with `max_retries` retries, up to `(max_retries + 1) × time_limit`.

### HGS-CVRP

[vidalt/HGS-CVRP](https://github.com/vidalt/HGS-CVRP) is a modern implementation of the hybrid genetic search algorithm specialized to the CVRP, including an additional neighborhood called SWAP\*. It is described in [Vidal (2022)](https://doi.org/10.1016/j.cor.2021.105643) and reached through the Python bindings [mdealencar/HybGenSea](https://github.com/mdealencar/HybGenSea). It is bundled with _OptiWindNet_, so it needs no separate installation.

Its distinctive options concern the feeder count:

- the feeder limit is normally an **upper bound** — the search is free to use fewer, and normally settles at the minimum feasible number;
- pinning the count to that limit exactly additionally requires balanced subtrees and a single substation, and rules out the ringed topology;
- balancing makes subtree loads differ by at most one terminal;
- with multiple substations the feeder limit is ignored and the count is fixed to the minimum required;
- a seed is available for reproducible runs.

Because HGS produces radial topologies, and radial is a special case of branched, its solutions can warm-start both branched and radial models.

_In use:_ {doc}`/notebooks/hi21_hgs` (Network/Router API) · {doc}`/notebooks/lo21_hgs` (Advanced API).

### LKH-3

[LKH-3](http://akira.ruc.dk/~keld/research/LKH-3/) is Keld Helsgaun's implementation of the Lin-Kernighan-Helsgaun meta-heuristic, extended to constrained TSP and vehicle routing problems.

Unlike HGS-CVRP, it is **not bundled**: _OptiWindNet_ interfaces with it through temporary files and system calls, so the `LKH` executable must be on the `PATH` as seen from the Python process. Keld Helsgaun distributes it as C source code and as a Windows binary, for academic and non-commercial use.

Its options cover the search itself — number of runs, a per-run time limit, a pseudo-random seed — plus whether to fill missing candidate links with direct Euclidean links or to restrict the search to the allowed graph `A`.

_In use:_ {doc}`/notebooks/lo22_lkh` (Advanced API).

## Exact optimization

The problem is formulated as a mixed-integer linear program and handed to a branch-and-cut solver. This is the only approach that quantifies solution quality: the solver maintains a bound on the best possible solution and reports the remaining **optimality gap**. Stopping at a 1% gap means the solution is provably within 1% of optimal, which typically needs much less time than proving optimality outright.

The exact routers honor every option in [](/routers.md#model-options), which the other approaches only partly support.

The common flow model, objective and crossing constraints are given in {doc}`/reference/milp_formulation`.

The model is handed to one of several interchangeable backends; which ones are supported, and how to install them, is in {doc}`/reference/solvers`. The remaining arguments (time limit, gap, verbosity) are the same across solvers, so switching between them is a one-word change.

_In use:_ {doc}`/notebooks/hi23_milp` (Network/Router API) · {doc}`/notebooks/lo23_milp_ortools` and the other MILP notebooks (Advanced API).

### Solver options

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

## Warm-starting

A feasible solution supplied up front lets a MILP solver start from a known bound instead of searching for its first incumbent, which usually shortens the time to a good gap considerably. The natural chain is fast to slow: a constructive heuristic or a meta-heuristic produces a solution, and the MILP model starts from it.

A warm start is only usable if it satisfies the constraints of the model being started. This is where the capability gaps in [](/routers.md#what-each-approach-can-enforce) become concrete:

| Model option   | Value         | Usable warm starts  |
| -------------- | ------------- | ------------------- |
| `feeder_limit` | `'unlimited'` | any router          |
|                | `'minimum'`   | meta-heuristic only |
| `feeder_route` | `'straight'`  | any router          |
|                | `'segmented'` | any router          |
| `topology`     | `'branched'`  | any router          |
|                | `'radial'`    | meta-heuristic only |

The two restricted cases have the same cause: a constructive heuristic does not limit the number of feeders and does not produce radial subtrees unless asked for the radial variant, so its output violates those two models. A meta-heuristic solution is radial and uses the minimum feeder count, so it satisfies both.

_In use:_ {doc}`/notebooks/hi40_example_taylor_2023` (Network/Router API) · {doc}`/notebooks/lo40_example_taylor_2023` (Advanced API).

## Choosing a router

The figure compares all three optimization approaches on one instance, relative to its proven optimum:

```{image} /_static/fig_routers_light.svg
:alt: Solution length above the proven optimum versus computation time for the three optimization approaches
:class: only-light
:width: 100%
```

```{image} /_static/fig_routers_dark.svg
:alt: Solution length above the proven optimum versus computation time for the three optimization approaches
:class: only-dark
:width: 100%
```

The routers differ by orders of magnitude in runtime and show diminishing improvements in solution quality. Constructive heuristics finish in milliseconds with the largest optimality gap. The meta-heuristic closes most of the gap within about one second, with limited subsequent improvement. The warm-started MILP reaches the optimum before termination and then improves the bound until optimality is certified; its flat incumbent curve indicates that it is working on proving optimality, not that optimization has stalled.

The differences between optimization approaches depend on the site, capacity, and model options. This figure represents a single instance and should be interpreted qualitatively.

- **Interactive exploration, or a network re-solved inside an outer loop** — constructive heuristic. Sub-second, and the quality is adequate for comparing layouts against each other.
- **A good network without a long wait** — meta-heuristic with a modest time limit. Check the reported solution times: if raising the limit stops improving the length, lower it and save the time.
- **A network you intend to defend** — MILP, warm-started by one of the above, stopped at an optimality gap you consider acceptable.
- **A specific structure is required** — ringed for redundancy, radial to avoid branching at turbines, a pinned feeder count to match available switchgear — check [](/routers.md#model-options) for which approaches can enforce it, and use MILP when the constraint must be guaranteed.

_In use:_ {doc}`/notebooks/hi00_quickstart` (Network/Router API) · {doc}`/notebooks/lo00_quickstart` (Advanced API).

### How long a solve takes

Only the exact routers make this an open question. A constructive heuristic finishes in a fraction of a second whatever the instance, and a meta-heuristic takes the budget you give it and stops.

A MILP solve is different: it gets harder with the number of terminals and with the cable capacity — a larger capacity admits more feasible subtrees, so the tree the solver has to explore grows — and the growth is steep enough that predicting a runtime is not worth the effort. Bound it instead. Set a `time_limit` and a `mip_gap` and let whichever comes first end the solve; both are in [](/routers.md#solver-options). Warm-starting shortens the way to a usable gap, and the backends themselves differ in speed on the same model — see {doc}`/reference/solvers`. The {doc}`/paper` reports solve times across a range of problem sizes.

_In use:_ {doc}`/notebooks/hi23_milp` (Network/Router API) · {doc}`/notebooks/lo23_milp_ortools` (Advanced API).
