# Optimization approaches

_OptiWindNet_ calls its optimization approaches **routers**. Choosing an approach decides how the cable network is optimized; choosing an API decides how to use that approach from Python. The Network/Router API wraps approaches in router classes, while the Advanced API exposes the underlying functions. No API choice is needed to compare the approaches here.

Both APIs offer constructive heuristics, the HGS meta-heuristic and [](#exact-optimization). LKH is an additional option available only through the Advanced API.

## Choosing a router

| Approach | Implementations | Typical runtime | Quality |
| --- | --- | --- | --- |
| Heuristic | Esau-Williams variants | tens of milliseconds | unbounded – typically a few percent over the minimum |
| Meta-heuristic | HGS-CVRPᵃ, LKH-3ᵇ | tenths of a second to tens of seconds | unbounded – typically closer to the minimum than via heuristics |
| Exact optimization | MILPᶜ & branch-and-cut solvers | minutes to hours | bounded – user may impose a gap or have the gap reported for a fixed budget |

ᵃ HGS-CVRP = [Hybrid Genetic Search for the CVRP](https://github.com/vidalt/HGS-CVRP)\
ᵇ LKH-3 = [Lin–Kernighan–Helsgaun TSP solver, version 3](http://akira.ruc.dk/~keld/research/LKH-3/)\
ᶜ MILP = Mixed-Integer Linear Programming

The approaches are complementary rather than competing: a compatible heuristic or meta-heuristic solution can give a [MILP solver](#exact-optimization) a useful starting point. See [](/routers.md#warm-starting). They also differ in which constraints they can actually enforce; the matrix is in [](/routers.md#which-constraints-each-approach-can-enforce).

```{admonition} What "quality" means here
:class: note

Quality refers to additional cable length relative to the best length for the modeled problem over a set of representative instances. For an *unbounded* approach, any typical excess is a statistical estimate; the excess length of a specific solution is unknown. Only the exact routers can certify a gap for their length objective. A meta-heuristic result may be optimal for that objective, but the approach cannot certify it. Cable prices may give these solutions a different ordering by cost; see [](/problem.md#objective-and-reported-cost).
```

- **Interactive exploration, or a network re-solved inside a loop** — constructive heuristic. Sub-second, and the quality is adequate for comparing layouts against each other.
- **A good network without a long wait** — meta-heuristic with a modest time limit. Check the reported solution times: if raising the limit stops improving the length, lower it and save the time.
- **A quantified bound on solution quality** — MILP, warm-started by one of the above, stopped at an optimality gap you consider acceptable.
- **A specific structure is required** — ringed for redundancy, radial to avoid branching at turbines, a pinned feeder count to match available switchgear — check [](#which-constraints-each-approach-can-enforce) and select an approach that supports the required combination.

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

## Which constraints each approach can enforce

The [](/problem.md#problem-options) define the feeder requirements; [](/problem.md#network-topologies) describes the architectures. Required constraints should guide the choice of approach before speed or solution quality.

| Approach | Topology | Feeder route | Feeder (count) limit | Balanced feeder loads |
| --- | --- | --- | --- | --- |
| Heuristic | branched, radial, ringed | straight, segmented | no | no |
| Meta-heuristic | radial, ringed | segmented | upper bound, exact countᵃ | yes |
| Exact optimization | branched, radial, ringed | straight, segmented | yes | yes |

ᵃ only if _balanced_ is also enforced

Those differences must be taken into account when chaining routers; see [](/reference/warm_starting.md#warm-start-acceptance-requirements).

The capability table assumes uniform turbine power. Unequal powers restrict supported topologies and feeder limits; see [](/reference/power.md#router-support). Some unsupported option combinations raise errors; others are ignored with a warning, including MILP balancing when the feeder count is not pinned.

## How the approaches work

The following sections explain how each family searches for a network.

### Constructive heuristics

These build a solution incrementally. Initially, each turbine has its own connection to a substation; the heuristic then merges subtrees while capacity allows. They are extensions of the Esau-Williams heuristic for the CMSTP, modified to account for cable crossings. They run in a fraction of a second even on large layouts, which makes them the right default for interactive work, for warm-starting the slower routers, and for use inside an outer optimization loop where the network is re-solved on every iteration.

Esau-Williams variants differ in how they break ties and how strongly they favor growth towards a substation. Some variants construct branched networks; others construct radial paths or rings. The guides explain how to select a variant in each API.

_In use:_ {doc}`/notebooks/hi20_heuristic` (Network/Router API) · {doc}`/notebooks/lo20_heuristic` (Advanced API).

### Meta-heuristics

Meta-heuristics search the solution space under a time budget you set. They treat the problem as a capacitated vehicle routing problem (CVRP), which is why they produce radial topologies by default: a CVRP route is a path, not a tree. Solving the _closed_ CVRP instead, where every route returns to the depot, yields a ringed topology.

Both implementations handle multiple substations by assigning turbines to one cluster per substation, then solving the clusters concurrently. This fixes the assignment before the per-cluster search. Crossing repair and repeated searches can add to the total runtime.

#### HGS-CVRP

HGS-CVRP uses hybrid genetic search for the CVRP. It is bundled with _OptiWindNet_ and available through both APIs. It supports radial and ringed networks; feeder-count and balancing support depends on the topology, turbine powers and number of substations.

_In use:_ {doc}`/notebooks/hi21_hgs` (Network/Router API) · {doc}`/notebooks/lo21_hgs` (Advanced API).

#### LKH-3

LKH-3 uses the Lin–Kernighan–Helsgaun search method for constrained travelling-salesperson and vehicle-routing problems. It is available only through the Advanced API and requires a separate executable. See the guide for setup and usage.

_In use:_ {doc}`/notebooks/lo22_lkh` (Advanced API).

### Exact optimization

Exact optimization expresses the network design as a mixed-integer linear programming (MILP) model: mathematical variables describe the connections, and constraints describe which networks are allowed. A **MILP solver** searches for a network satisfying those constraints while minimizing cable length.

The solver maintains a lower bound and reports the remaining **optimality gap** between that bound and its best model objective — the _MIP gap_ in solver logs. A requested gap of 1% sets the relative optimality tolerance; however, the time limit takes precedence as a stopping criterion, so reaching it may leave a larger gap than requested.

The certificate applies to the selected candidate links and model constraints. Detours are computed after the search and can increase the delivered length, so the reported gap is not automatically the gap of the final routeset, and it is not a cost guarantee. Compare the routed length with the bound in the same length units when assessing the delivered network. Solver backends with a solution pool may select a different topology than that of the best incumbent solution after ranking candidates by routed length.

The exact routers support the model options in [](/problem.md#problem-options), subject to their combination rules and the restrictions for mixed powers. Feeder load balancing requires a pinned feeder count; otherwise it is not enforced.

The common flow model, objective and crossing constraints are given in {doc}`/reference/milp_formulation`.

The model is handed to one of several interchangeable backends; which ones are supported, and how to install them, is in {doc}`/reference/solvers`. That reference page also describes solver settings.

_In use:_ {doc}`/notebooks/hi23_milp` (Network/Router API) · {doc}`/notebooks/lo23_milp_ortools` and the other MILP notebooks (Advanced API).

## How long a solve takes

Constructive heuristics are usually the fastest choice. Meta-heuristics have a search time budget, but clustering, crossing repair and retries add to wall time. MILP model construction, warm-start construction and routing also happen outside the solver budget, but the solving step is the dominant one.

A MILP solve becomes harder as the turbine count increases. The capacity also influences problem difficulty, which is easiest at capacities 2 and 3, and gets increasingly harder until around 7–9; higher capacities then cause difficulty to decrease. The sensible approach to sizing the solver budget is experimentation in the available optimization setup (representative problem set, a chosen solver, hardware, competing CPU loads). Set a time budget and an acceptable optimality gap, and let whichever comes first end the solve; the settings are described in [](/reference/solvers.md#solver-options). [](#warm-starting) shortens the way to a usable gap, and the backends themselves differ in speed on the same model — see {doc}`/reference/solvers`. The {doc}`/paper` reports solve times across a range of problem sizes.

_In use:_ {doc}`/notebooks/hi23_milp` (Network/Router API) · {doc}`/notebooks/lo23_milp_ortools` (Advanced API).

## Warm-starting

**Warm-starting means giving an optimization method an existing solution as a starting point.** For [](#exact-optimization), a typical workflow is to build a network quickly with a constructive heuristic or HGS, then give it to the MILP solver. The solver can search for improvements and assess solution quality from that starting point. This may save time, but does not guarantee faster completion. Starting without an initial solution is called a **cold start**.

The initial network must meet the requirements of the problem being solved. See {doc}`/reference/warm_starting` for the compatibility table, restrictions and links to examples for each API.

## Using an approach

Choose the {doc}`/high_level_api` for the Network/Router workflow or the {doc}`/low_level_api` for direct control of the underlying functions and graphs. If undecided, return to {doc}`/apis`; the problem and optimization concepts above apply to both.
