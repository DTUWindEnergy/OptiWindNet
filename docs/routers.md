# Routers

_Router_ is the term used in _OptiWindNet_ to refer to each optimization approach to the cable routing problem. Three different classes of routers are offered in both the {doc}`/high_level_api` and the {doc}`/low_level_api`. The former wraps each approach in a {py:class}`Router <optiwindnet.api.Router>` class, while latter exposes the underlying functions directly. This page describes what the routers do, which options they accept and when to choose them; the API sections show how to invoke them.

## Optimization approaches

| Approach | Implementations | Typical runtime | Quality |
| --- | --- | --- | --- |
| Heuristic | Esau-Williams variants | tens of milliseconds | unbounded – typically a few percent over the minimum |
| Meta-heuristic | HGS-CVRPᵃ, LKH-3ᵇ | tenths of a second to tens of seconds | unbounded – typically closer to the minimum than via heuristics |
| Exact optimization | MILPᶜ & branch-and-cut solvers | minutes to hours | bounded – user may impose a gap or have the gap reported for a fixed budget |

ᵃ HGS-CVRP = [Hybrid Genetic Search for the CVRP](https://github.com/vidalt/HGS-CVRP)\
ᵇ LKH-3 = [Lin–Kernighan–Helsgaun TSP solver, version 3](http://akira.ruc.dk/~keld/research/LKH-3/)\
ᶜ MILP = Mixed-Integer Linear Programming

The approaches are complementary rather than competing: a heuristic solution is a valid starting point for a meta-heuristic or a MILP solve, and chaining them is the normal way to get a good solution quickly. See [](/routers.md#warm-starting). They also differ in which constraints they can actually enforce; the matrix is in [](/routers.md#which-constraints-each-approach-can-enforce).

```{admonition} What "quality" means here
:class: note

Quality refers to additional cable length relative to the best length for the modeled problem over a set of representative instances. For an *unbounded* approach, any typical excess is a statistical estimate; the excess length of a specific solution is unknown. Only the exact routers can certify a gap for their length objective. A meta-heuristic result may be optimal for that objective, but the approach cannot certify it. Cable prices may give these solutions a different ordering by cost; see [](/problem.md#objective-and-reported-cost).
```

_In use:_ {doc}`/notebooks/hi20_heuristic` (Network/Router API) · {doc}`/notebooks/hi21_hgs` (Network/Router API) · {doc}`/notebooks/hi23_milp` (Network/Router API).

## Problem options

Problem options complement the problem data (component counts, positions, boundaries, ratings) in defining **what is solved**. They constrain the structure of the networks that are considered feasible solutions. They are independent of the solver, and in principle meaningful for every approach — subject to the table in [](/routers.md#which-constraints-each-approach-can-enforce).

- `topology` picks the architecture of the subtrees (branched, radial, ringed); see [](/problem.md#network-topologies) for details.
- `feeder_route` decides whether feeder routes may be detoured (multiple straight segments) or must run straight. Even when straight feeders are requested, exclusion zones may introduce some bends in the route.
- `feeder_limit`, together with `max_feeders`, bounds how many feeders the solution may use, which may be given as an absolute count or relative to the minimum count that meets other constraints. For a ringed topology each ring uses two feeders, which entails that bounds must include at least one even number.
- `balanced` requires the subtree loads to differ by at most one terminal. It is only enforceable when the feeder count is pinned to a single value.

Only {py:class}`MILPRouter <optiwindnet.api.MILPRouter>` accepts these as a {py:class}`ModelOptions <optiwindnet.MILP.ModelOptions>` mapping; {py:class}`EWRouter <optiwindnet.api.EWRouter>` and {py:class}`HGSRouter <optiwindnet.api.HGSRouter>` expose the subset they can enforce as ordinary constructor arguments instead. Permitted values and the defaults can be displayed with {py:meth}`ModelOptions.help() <optiwindnet.MILP.ModelOptions.help>`, and {doc}`/notebooks/hi31_options` compares the effects of different choices.

### Which constraints each approach can enforce

| Approach | Topology | Feeder route | Feeder (count) limit | Balanced feeder loads |
| --- | --- | --- | --- | --- |
| Heuristic | branched, radial, ringed | straight, segmented | no | no |
| Meta-heuritic | radial, ringed | segmented | upper bound, exact countᵃ | yes |
| Exact optimization | branched, radial, ringed | straight, segmented | yes | yes |

ᵃ only if _balanced_ is also enforced

Those differences must be taken into account when chaining routers; see [](/routers.md#warm-starting).

_In use:_ {doc}`/notebooks/hi31_options` (Network/Router API) · {doc}`/notebooks/lo30_topologies` (Advanced API).

## Constructive heuristics

These build a solution incrementally, starting from every terminal connected directly to a root and repeatedly merging subtrees while capacity allows. They are extensions of the Esau-Williams heuristic for the CMSTP, modified to account for cable crossings. They run in a fraction of a second even on large layouts, which makes them the right default for interactive work, for warm-starting the slower routers, and for use inside an outer optimization loop where the network is re-solved on every iteration.

The variants differ in how they break ties and how strongly they bias growth towards the root. They are selected by the `method` argument, which names a variant of the constructive-heuristic approach — it never selects among the optimization approaches:

| `method` | Topology | Description |
| --- | --- | --- |
| `'esau_williams'` | branched | The classic Esau-Williams C-MST heuristic, modified to avoid crossings. |
| `'biased_EW'` | branched | Esau-Williams with a rootward bias in near-tie cases. The default for {py:class}`EWRouter <optiwindnet.api.EWRouter>`. |
| `'rootlust'` | branched | A configurable rootward bias that increases as remaining capacity decreases. The default for {py:func}`constructor() <optiwindnet.heuristics.constructor>`. |
| `'radial_EW'` | radial | Produces radial subtrees — simple paths from the root. |
| `'ringed'` | ringed | Network of closed paths starting and ending on a root. |

The two entry points use different default variants and may therefore produce different networks for the same input.

Versions of _OptiWindNet_ before v0.3.0 had a different Advanced API for the heuristics; see {doc}`/notebooks/lo90_removed_heuristics` for migration.

_In use:_ {doc}`/notebooks/hi20_heuristic` (Network/Router API) · {doc}`/notebooks/lo20_heuristic` (Advanced API).

## Meta-heuristics

Meta-heuristics search the solution space under a time budget you set. They treat the problem as a capacitated vehicle routing problem (CVRP), which is why they produce radial topologies by default: a CVRP route is a path, not a tree. Solving the _closed_ CVRP instead, where every route returns to the depot, yields a ringed topology.

Both wrappers handle multiple substations by clustering the terminals by root and solving one instance per cluster, and both repair crossings iteratively after the search. That repair work is bounded by a retry count, so total wall time can exceed the time limit you set — with `max_retries` retries, up to `(max_retries + 1) × time_limit`.

### HGS-CVRP

[vidalt/HGS-CVRP](https://github.com/vidalt/HGS-CVRP) is a modern implementation of the hybrid genetic search algorithm specialized to the CVRP, including an additional neighborhood called SWAP\*. It is described in [Vidal (2022)](https://doi.org/10.1016/j.cor.2021.105643) and reached through the Python bindings [mdealencar/HybGenSea](https://github.com/mdealencar/HybGenSea). It is bundled with _OptiWindNet_, so it needs no separate installation.

Its distinctive options concern the feeder count:

- the feeder limit is normally an **upper bound** — the search is free to use fewer, and normally settles at the minimum feasible number;
- pinning the count to that limit exactly additionally requires balanced subtrees and a single substation (for the ringed topology, the limit is then at most one ring per two turbines);
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

The problem is formulated as a mixed-integer optimization model and handed to a solver. The solver maintains a lower bound and reports the remaining **optimality gap** between that bound and its best model objective — the _MIP gap_ of solver logs and of the `mip_gap` option. A gap of `0.01` requests a maximum of 1% relative increase above the verified bound; however, the time limit always take precedence as stopping criterion, so reaching it may leave a larger gap than requested.

The certificate applies to the selected candidate links and model constraints. Detours are computed after the search and can increase the delivered length, so the reported gap is not automatically the gap of the final routeset, and it is not a cost guarantee. Compare the routed length with the bound in the same length units when assessing the delivered network. Solver backends with a solution pool may select a different topology than that of the best incumbent solution after ranking candidates by routed length.

The exact routers support the model options in [](/routers.md#problem-options), subject to their combination rules and the restrictions for mixed powers. Feeder load balancing requires a pinned feeder count; otherwise it is not enforced.

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

A feasible solution supplied up front lets a MILP solver start from a known bound instead of searching for its first incumbent, which may shorten the time to a small gap. The natural chain is fast to slow: a constructive heuristic or a meta-heuristic produces a solution, and the MILP model starts from it.

A warm start is only usable if it satisfies the model it is given to. The model checks it against its own definition and refuses it at the first failure. Its **topology** must be the model's own, with a single relaxation: a radial solution also satisfies a branched model, since a path is a valid tree. Every **link** it uses must exist in the model, which holds whenever both derive from the same available-link set `A`. And every **constraint** the model emitted must hold. The last check is where the capability gaps in [](/routers.md#which-constraints-each-approach-can-enforce) become concrete, one model option at a time:

| Model option | What it demands of a warm start | Fast routers that produce it |
| --- | --- | --- |
| `topology='branched'` | branched or radial | any router |
| `topology='radial'` | radial | `HGSRouter(…)`; `EWRouter(method='radial_EW')` |
| `topology='ringed'` | ringed | `HGSRouter(ringed=True)`; `EWRouter(method='ringed')` |
| `feeder_route='segmented'` | no requirement | any router |
| `feeder_route='straight'` | no link blocking a feeder route | `EWRouter(feeder_route='straight')` |
| `feeder_limit='unlimited'` | no requirement | any router |
| `feeder_limit='minimum'`, `'min_plus1..3'`, `'specified'` | a feeder count within the allowed range | `HGSRouter(feeder_limit=n)`, `n` being the highest count the model allows |
| `feeder_limit='exactly'` | exactly `max_feeders` feeders | `HGSRouter(feeder_limit=max_feeders, feeder_exact=True, balanced=True)`, single substation only |
| `balanced=True` | feeder loads within one terminal of each other | `HGSRouter(balanced=True)`, with the feeder count pinned |

The requirements are cumulative: a model imposes all of them at once, and no single option decides acceptance on its own. The two fast routers satisfy disjoint subsets of them. A constructive heuristic does not constrain the feeder count, so its feeder count must be checked before using it to warm-start a model that constrains the count, but it is the only router that can be required to keep the links clear of the feeder routes. HGS constrains the feeder count and the load balance, and its solutions are repaired until no two non-feeder links cross, but that repair does not examine the feeders — a link blocking a feeder route may remain. Under `feeder_route='segmented'` this is not a defect, since path-finding detours the feeder around the link; a straight-feeder model forbids the detour and refuses such a solution. Some models consequently have no fast producer at all, `feeder_route='straight'` together with a constrained feeder count being the clearest example.

Two restrictions apply to the last column. Option `balanced` is only enforced where the feeder count is pinned to a single value (`'minimum'` or `'exactly'`), which is where the router settings above pin it. For a ringed model, `max_feeders` and the router's `feeder_limit` both count substation connections, two per ring, so either must be even; an exact count is also limited to one ring per two turbines.

A solution that does not fit can be replaced instead of dropped: a heuristic matched to the model's topology and feeder limit attempts to build a fitting one. The model itself does not arrange that — it accepts the warm start it is given or refuses it — so wherever the warm start is passed explicitly, replacing a refused one is the caller's move. {py:class}`MILPRouter <optiwindnet.api.MILPRouter>` is the exception, and the only one: it warm-starts by default, starting from the solution its network already carries when the model takes it and building a replacement when it does not, with `warmup` and `warmup_time` to switch that off or to budget the build. It builds the replacement with the constructive heuristic for a branched model with `feeder_limit='unlimited'`, keeping the links clear of the feeder routes if the model requires straight ones, and with HGS, its feeder count set to the model's bounds, for every other model. Either way, a model that no quickly built solution fits is solved cold.

`warmup_time` defaults to 0.2 seconds and bounds the HGS search, capped by `time_limit`; HGS repair retries can extend it, and the constructive heuristic runs without a budget. `warmup=False` disables both construction and reuse, even when the network already has a solution. With unequal turbine powers, fewer models can be warm-started; see [](/reference/input_formats.md#turbines-of-unequal-output).

In the Advanced API, supply a topology through `solver.set_problem(P, A, model_options=options, capacity=capacity, warmstart=S)`. This call does not build a replacement warm start; an incompatible one raises `OWNWarmupFailed`. All arguments after `P` and `A` are keyword-only.

_In use:_ {doc}`/notebooks/hi31_options` and {doc}`/notebooks/hi40_example_taylor_2023` (Network/Router API) · {doc}`/notebooks/lo40_example_taylor_2023` (Advanced API).

## Choosing a router

The figure compares all three optimization approaches on one instance, relative to its proven optimum:

```{image} /_static/fig_routers_light.svg
:alt: Solution length above the proven optimum versus computation time for the three optimization approaches, with the MIP gap marked between the MILP bound and incumbent
:class: only-light
:width: 100%
```

```{image} /_static/fig_routers_dark.svg
:alt: Solution length above the proven optimum versus computation time for the three optimization approaches, with the MIP gap marked between the MILP bound and incumbent
:class: only-dark
:width: 100%
```

The routers differ by orders of magnitude in runtime and show diminishing improvements in solution quality. Constructive heuristics finish in milliseconds with the largest optimality gap. The meta-heuristic closes most of the gap within about one second, with limited subsequent improvement. The warm-started MILP reaches the optimum before termination and then improves the bound until optimality is certified; its flat incumbent curve indicates that it is working on proving optimality, not that optimization has stalled. The double arrow marks the MIP gap at one instant — the incumbent above the bound, with the unknown optimum somewhere between them — and the solve ends when that distance has shrunk to the requested tolerance.

The differences between optimization approaches depend on the site, capacity, and model options. This figure represents a single instance and should be interpreted qualitatively.

- **Interactive exploration, or a network re-solved inside an outer loop** — constructive heuristic. Sub-second, and the quality is adequate for comparing layouts against each other.
- **A good network without a long wait** — meta-heuristic with a modest time limit. Check the reported solution times: if raising the limit stops improving the length, lower it and save the time.
- **A network you intend to defend** — MILP, warm-started by one of the above, stopped at an optimality gap you consider acceptable.
- **A specific structure is required** — ringed for redundancy, radial to avoid branching at turbines, a pinned feeder count to match available switchgear — check [](/routers.md#problem-options) for which approaches can enforce it, and use MILP when the constraint must be guaranteed.

_In use:_ {doc}`/notebooks/hi00_quickstart` (Network/Router API) · {doc}`/notebooks/lo00_quickstart` (Advanced API).

### How long a solve takes

Constructive heuristics are usually the fastest choice. Meta-heuristics have a search budget, but clustering, crossing repair and retries add to wall time. MILP model construction, warm-start construction and routing also happen outside the solver search budget; `MILPRouter` can repeat the search if it finds no feasible incumbent.

A MILP solve is different: it gets harder with the number of terminals and with the cable capacity — a larger capacity admits more feasible subtrees, so the tree the solver has to explore grows — and the growth is steep enough that predicting a runtime is not worth the effort. Bound it instead. Set a `time_limit` and a `mip_gap` and let whichever comes first end the solve; both are in [](/routers.md#solver-options). Warm-starting shortens the way to a usable gap, and the backends themselves differ in speed on the same model — see {doc}`/reference/solvers`. The {doc}`/paper` reports solve times across a range of problem sizes.

_In use:_ {doc}`/notebooks/hi23_milp` (Network/Router API) · {doc}`/notebooks/lo23_milp_ortools` (Advanced API).
