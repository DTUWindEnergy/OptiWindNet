# v0.3.1 (unreleased)

## Important Changes

- **`MILPRouter` warm-start controls, with a default behavior change.** Two new parameters expose warm-starting explicitly: `warmup` (default `True`) is a master switch, and `warmup_time` (default `0.2` s) is the time budget for *building* a warm start with the heuristic (replacing a previously hardcoded value).
  - **Behavior change:** with `warmup=True` (the default), a *fresh* MILP solve now builds a warm start via HGS/constructor before solving, where it previously solved cold. Reusing the solution carried across successive `WindFarmNetwork.optimize()` calls is unchanged. Pass `warmup=False` to force a fully cold solve (any stored or supplied solution is then ignored).
  - Warm-start construction now also covers the `feeder_limit` values `'exactly'` (with `balanced`), `'min_plus1/2/3'`, and `'minimum'`+`balanced` across the `branched`, `radial`, and `ringed` topologies; these previously fell back to a cold solve.
  - The feeder-count options of the HGS-CVRP warm start come from the model's own bounds (`feeder_and_load_bounds()`), through the new `MILP._core.hgs_count_kwargs()`. A pinned `'exactly'` above the minimum is warm-started without `balanced` too, since a balanced solution satisfies an unbalanced model, and `'specified'` caps the HGS-CVRP count at `max_feeders`, for a branched model as well, which took the constructor's layout.

- **Exact ring counts with HGS-CVRP.** `hgs_cvrp(ringed=True)` accepts `vehicles_exact=True` (with `balanced=True`), pinning the number of rings, up to `T // 2` so that every ring holds two or more turbines; it previously raised `NotImplementedError`. `HGSRouter(ringed=True, feeder_exact=True)` therefore pins the feeder count at twice the ring count, and `MILPRouter` builds a warm start for a balanced RINGED model with `feeder_limit='exactly'` above the minimum, which previously solved cold.

- **`make_min_length_model()` rejects the `str` spelling of its enum options (`topology`, `feeder_route`, `feeder_limit`).** It raises `TypeError` rather than defaulting silently or failing deeper in the build; `ModelOptions` continues to accept the `str` spelling.

- **A MILP search settles on link bits, not on a topology.** `Solver.solve()` reads the incumbent's binary link variables into a canonical bit vector, exposed as the new read-only `Solver.incumbent_linkbits`, and derives the `topology_id` it stamps on `SolutionInfo` from it; no topology graph is built. `get_solution()` decodes that vector — and every solution-pool entry it ranks — through `S_from_linkbits()`. Identifying a solution therefore costs a bit vector rather than a decode, and a ringed MILP solution is closed into rings by the same `split_rings_and_calc_loads()` every other ringed producer uses (its subtree ids are numbered in that function's order).

- **`Solver.set_problem()` is unified in the base `Solver` class and keyword-only from `model_options`.**
  - **Behavior change / Pitfall:** `Solver.set_problem(P, A, *, model_options, capacity=None, capacity_nominal=None, power_rtol=0.01, warmstart=None)` requires all parameters after geometry graphs `P` and `A` to be passed as keyword arguments. Calling `solver.set_problem(P, A, capacity, model_options)` positionally raises `TypeError`.

- **Modularization of `interarraylib`.** `interarraylib.py` was decomposed into single-concern modules with strictly one-way dependencies: `identity`, `loads`, `converting`, `transforming`, `validating`, `terse`, and `presenting`. Topologies and candidate link sets are canonically identified by `topology_id()` and `linkset_id()` in `optiwindnet.identity`, and `SolutionInfo` gains the `topology_id` attribute with a column-aligned summary `__repr__`.

- **Load calculation and solver bounds use terminal `'inflow'`.** `calcload()` and `bfs_subtree_loads()` use each terminal's declared integer inflow, defaulting to one.
  - `calcload()` checks that the traversal reaches all `T` terminals independently of their inflow. Its error message reports the number reached. Passing `nominal=True` accumulates nominal power into `'load_nominal'` without altering integer loads, edge directions, or graph metadata.
  - `bfs_subtree_loads()` defaults a root's own contribution to zero.
  - `S_from_G()` preserves terminal inflow and quantization graph attributes (`power_per_inflow`, `capacity_nominal`, `power_rtol`) so that validation reproduces routed loads. Computational topologies omit declared nominal power.
  - All three MILP model builders derive minimum feeder counts and balanced load bounds from total inflow. `feeder_and_load_bounds()` accepts `inflow_total`, defaulting to `T`.
  - `hgs_cvrp()` and `lkh3()` support nonunitary inflow for unbalanced, single-substation radial solves. Both pass inflow as customer demand. LKH-3 adjusts route-size bounds for those demands and checks route loads when retrying an over-capacity solution.
  - `loads.terminal_inflow()` extracts nonunitary demands; `validate_terminal_power()` checks inflow and optional nominal power at graph-input boundaries.

- **Turbines of unequal power and explicit power units.** Documented in the Input Formats reference, section "Turbines of unequal output".
  - **Behavior change / Pitfall:** the library separates nominal physical rating from integer solver flow. Terminals declare physical power via the `'power'` attribute, while solver demands are strictly integer `'inflow'`. User code setting `node['power'] = <int>` directly on NetworkX graphs without matching inflow or `power_per_inflow` declarations causes `validate_terminal_power()` to raise `ValueError`. Use `loads.set_turbine_powers(L, powers, power_unit)` or set `'inflow'` directly.
  - `loads.set_turbine_powers(L, powers, power_unit=None)` declares nominal power: equal powers as the graph attribute `power_per_inflow`, unequal ones as terminal `'power'` fractions and the graph attribute `powers_set`. `validate_terminal_power()` checks the convention and maintains the graph attribute `power_quantization_inexact`.
  - `loads.quantize_for_capacity(powers_set, capacity_nominal, power_rtol)` maps unequal powers to integer inflow of the coarsest `power_per_inflow` within `power_rtol` that packs turbines into a cable of that nominal capacity exactly as their nominal powers do; `loads.quantized()` applies it to a copy of `A`. Location and available-links graphs never carry quantized inflow for unequal power.
  - `constructor()`, `hgs_cvrp()`, `lkh3()` and `Solver.set_problem()` take `capacity_nominal` and `power_rtol` as an alternative to `capacity`, and record `power_per_inflow`, `capacity`, `capacity_nominal` and `power_rtol` on the topology; `G_from_S()` combines them with the power declared on `A`. `Solver.set_problem()` recomputes the loads of a warm start quantized otherwise. Every `Router` takes `power_rtol` (default 0.01).
  - `WindFarmNetwork` takes `power_unit` (e.g. `'MW'`), the unit of `turbine_powers` and of the cable capacities, which `cables` returns as exact fractions; each solve quantizes unequal powers for the largest capacity. With `power_unit=None` (the default), `turbine_powers` and capacities are integer inflow, floats are refused, and a power declaration on `L` is dropped (logged as a warning for unequal powers, at INFO level for equal ones).
  - **Behavior change:** assigning `WindFarmNetwork.cables` whose largest capacity is below the solution's largest load invalidates the solution (call `optimize()` again) instead of raising `ValueError`; a solution that fits keeps its links and gets its cable types reassigned. `assign_cables()` compares nominal capacities with nominal loads where `G` has `capacity_nominal`, and raises `ValueError` if any load exceeds the largest capacity.
  - `EWRouter` and `constructor()` reject nonunitary inflow (`NotImplementedError`); `HGSRouter`, `hgs_cvrp()` and `lkh3()` accept it only for an unbalanced, single-substation radial solve (raising `NotImplementedError` for multi-root, balanced, or ringed solves). A loose `power_rtol` may quantize unequal powers to unitary inflow, which every router supports.
  - **Behavior change:** `feeder_and_load_bounds()`, and therefore all three MILP model builders, raises `ValueError` for `feeder_limit` `'minimum'` and `'min_plus1/2/3'` when terminal inflow differ from one: `ceil(inflow_total / capacity)` treats terminal inflow as divisible among feeders. Use `'unlimited'`, `'exactly'` or `'specified'`.
  - **Behavior change:** all three MILP model builders raise `NotImplementedError` for a RINGED topology whose terminals declare nonunitary inflow. The ring model states the two-arm split by doubling the capacity of a single path, which only bounds the arms while every terminal contributes one inflow.
  - `MILPRouter` warm starts honour terminal inflow: a solve with nonunitary inflow takes a plain HGS-CVRP warm start where HGS supports it, and is otherwise solved cold.
  - **Behavior change / Persistence pitfall:** `pack_G()`, and so `store_G()`, raises `NotImplementedError` for a routeset with nonunitary terminal inflow or `powers_set`, since a `RouteSet` record keeps no terminal power. Routesets of equal power are stored with their power attributes.
  - `gplot()` and `svgplot()` accept `node_tag='power'` and `node_tag='load_nominal'`; capacities stay in inflow in the graph attributes, the infobox and `describe_G()`, which adds a line with the distinct terminal inflow where turbines differ. `WindFarmNetwork.get_network()` reports loads in nominal units, through `calcload(G, nominal=True)` where quantization is inexact.
  - **Behavior change:** `TerseLinks.to_routeset(L)` and `to_topology(A)` require `capacity_nominal` if the site graph declares unequal turbine power (`powers_set`), raising `ValueError` otherwise.

- **`L_from_pbf()` loads the generators' declared electricity output.** The OpenStreetMap tag `generator:output:electricity` (for instance `8 MW`) sets the location's `power_unit` and declares each generator's power with `set_turbine_powers()`. It is taken only when every generator declares a positive output in a common unit; values carrying no number, such as `yes`, are ignored, a partial declaration is dropped with a warning, and inconsistent units raise `ValueError`. `read_powers=False` ignores the tags. Of the 74 bundled `.osm.pbf` locations, 56 declare their turbines' power this way.

- **`L_from_yaml()` loads the `TURBINE` section's declared power.** A mapping states the `power_MW` of every turbine; a list describes a site of turbines of unequal power, one entry per turbine model, each stating the `qty` of turbines it accounts for. The entries apply to `TURBINES` in blocks, in the order given, and their quantities must add up to `T`. Where the listing does not group the models, an entry's optional `prefix` claims every turbine whose label starts with it instead; prefixes are all or nothing, must claim each turbine exactly once, and turn `qty` into a cross-check on how many each one matched. Either form sets `power_unit` to `'MW'` and declares the powers with `set_turbine_powers()`. A list whose entries carry no `power_MW` declares no power. An entry declaring a power without a positive integer `qty` or a `prefix`, quantities that do not add up, prefixes that do not claim every turbine exactly once, and a `power_MW` that is not a positive finite number all raise `ValueError`. `read_powers=False` ignores the section, as it does for `load_repository()`. Of the 36 bundled `.yaml` locations, 29 built of one turbine model declare their power, as do Borssele and Walney Extension, each built of two.
  - **Behavior change / Low-level API pitfall:** `L_from_yaml()` and `load_repository()` default to `read_powers=True`. Because `Borssele.yaml` and `Walney Extension.yaml` declare unequal turbine power, loading them attaches `powers_set` to `L`. Passing these graphs directly to low-level solvers (`constructor()`, `hgs_cvrp()`, `lkh3()`, `Solver.set_problem()`) with an integer `capacity` raises `ValueError: A declares unequal turbine power: pass capacity_nominal`. Calling `constructor()` with `capacity_nominal` on them raises `NotImplementedError`. Pass `read_powers=False` to `L_from_yaml()` or `load_repository()` if unweighted turbine count behavior is intended without `WindFarmNetwork`.

- **New module `optiwindnet.presenting` gathers routeset property rendering as text.** `describe_G()` moved there from `interarraylib`.

- **`solver_factory()` refuses solver backends with native library clashes.** Calling `solver_factory()` raises `RuntimeError` if conflicting native solver packages (such as `highs` or `scip` alongside `ortools`) are already present in `sys.modules`.

- **Consistent complete-graph solves across baselines.** HGS and LKH support complete solves consistently across baselines, with distance-matrix utilities in `optiwindnet.baselines.utils` removed.

## Deprecations

- **Relocated functions in `optiwindnet.interarraylib` and `optiwindnet.fingerprint`.** Functions relocated during the modularization of `interarraylib` (`G_from_S`, `L_from_G`, `L_from_site`, `S_from_G`, `S_from_terse_links`, `terse_links_from_S`, `bfs_subtree_loads`, `calcload`, `split_rings_and_calc_loads`, `describe_G`, `TerseLinks`, `as_hooked_to_head`, `as_hooked_to_nearest`, `as_normalized`, `as_obstacle_free`, `as_rescaled`, `as_single_root`, `as_stratified_vertices`, `as_undetoured`, `validate_routeset`, `validate_topology`) remain accessible from `optiwindnet.interarraylib` as deprecated aliases emitting a `DeprecationWarning`, scheduled for removal in v0.4.0. Similarly, `optiwindnet.fingerprint` is a deprecated compatibility shim delegating to `optiwindnet.identity`.

## Removed APIs

- **`Solver.get_incumbent_topology()` was removed** (added in v0.3.0). `get_solution()` is the one way to obtain a routed topology. To obtain the incumbent topology without routing, decode the bits the search recorded: `S_from_linkbits(solver.incumbent_linkbits, A)`, followed by `calcload(S)` or `split_rings_and_calc_loads(S, A)` for the loads — this is what `docs/notebooks/lo32_clustering.ipynb` now does for its per-cluster sub-problems.
- **`optiwindnet.crossings.describe_crossings` was removed.** It was superseded by geometric crossing checks in `validate_routeset()` and has no remaining callers in the library.
- **`optiwindnet.interarraylib.add_ring_to_S` and `rings_from_S` were removed.** Ring topology construction and decoding are handled through `WindFarmNetwork` or `converting.S_from_linkbits()` / `terse.TerseLinks`.

## Bug Fixes

- **`hgs_cvrp()` and `lkh3()` left options that shape the solve out of `method_options`.** Both record `balanced`, `ringed`, `repair` and `max_retries` there, so that the stored method tells apart solves that differ in any of them.
  - **Persistence pitfall:** the method digest that `packmethod()` computes from `method_options` changes for every HGS-CVRP and LKH-3 routeset, so routesets stored from now on reference a new `Method` record even for settings identical to earlier ones.
- **`lkh3()` accepted `balanced=True` without enforcing it.** The flag only set LKH-3's `MTSP_MIN_SIZE`, which LKH-3 ignores for open routes (`TYPE=OVRP`), and ringed solves left it at 0, so route loads came out as unbalanced as with `balanced=False`. `lkh3()` raises `NotImplementedError` for `balanced=True`; use `hgs_cvrp(balanced=True)` for balanced loads.
  - **Behavior change:** `lkh3(balanced=True)`, accepted in v0.3.0, raises `NotImplementedError`.
- **`HGSRouter` with `ringed=True` read `feeder_limit` as a ring count.** Each ring connects to the substation twice, so `feeder_limit=4` allowed up to 4 rings (8 feeders). `feeder_limit` counts substation connections, as in `MILPRouter`, and is halved into the ring count.
  - **Behavior change:** a ringed `HGSRouter` with a given `feeder_limit` allows half as many rings as in v0.3.0, and an odd `feeder_limit` raises `ValueError`.
- **`find_geometric_crossings()` rejected valid RINGED routesets.** It traced every route into a polyline, and a ring comes out *closed* — its first and last segments meet at the substation by construction — so the closure, and every corridor a ring's two legs shared, was reported as `self_cross`/`self_overlap`. The identical geometry between two separate routes was tolerated, so rings were held to a standard radial routes are not; `validate_routeset()` inherited the false positives. Same-root rings are now recognized as closed polylines: their root segments are cyclically adjacent, and routing vertices or corridors shared by their two arms are tolerated. Rings bridging two substations remain open polylines. Intersections between distinct routes use the same centerline checks regardless of topology, while genuine boundary and self-crossings are still reported as `'cross'` and `'self_cross'`.
- **The route decomposition double-counted one link per ring.** Each ring's far feeder was emitted a second time as a two-node stub, breaking the documented "cover every edge exactly once" contract of the polyline tracing.
- **Strengthened topology and routeset validation.** `validate_routeset()` verifies stored loads, reduces G to S, and checks route geometry independently. `validate_topology()` checks node-set, orientation, load, capacity, connectivity, and topology invariants. `find_geometric_crossings()` detects self-intersections, branch splits, degenerate routes, and crossings at nonterminal route vertices.
- **Each solver instance gets its own options dict.** Default solver options in `SolverGurobi` and `SolverCplex` now live in `__init__` rather than class attributes, preventing `solver.options` mutations from leaking across solver instances or into the solver class.
- **`as_hooked_to_nearest()` cleared subtree loads incompletely.** Rehooking a subtree now clears loads on clone vertices to prevent stale clone load artifacts.
- **`ortools.highs` solver options were ignored.** Solver options passed to the HiGHS backend in OR-Tools are now forwarded to the solver parameters.

# v0.3.0

[Commit history since v0.2.3](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.2.3...v0.3.0)

This release adds library-wide support for **ringed** cable networks (those with cable cycles or "loops") – check the documentation for details.

## Important Changes

- New `topology=` alternative: `'ringed'` or `Topology.RINGED`.
- Solution graphs now carry a mandatory `topology` attribute. The new `Topology`, `ModelOptions`, and self-describing `TerseLinks` types make solver configuration, warm starts, and solution exchange topology-aware.
- MILP users can now retrieve an incumbent topology skipping `PathFinder` calls with `.get_incumbent_topology()`.
- Multi-root clustering was rewritten to keep turbines closer to their substations without adding feeders. HGS and LKH-3 now handle empty clusters.
- Validation now reports topology, load, capacity, and crossing violations without modifying the graph. `validate_routeset()` moved from `optiwindnet.crossings` to `optiwindnet.interarraylib` and now returns `list[str]`; `clusterize()` now returns only the cluster list.
- Planar-embedding generation is about 1.5 times faster, and Poisson-disc site generation avoids more unnecessary border and obstacle checks.
- Solver recovery and retry handling was improved for SCIP and FiberSCIP, including concurrent SCIP use on Windows. Documentation now includes dedicated guides for topology choices, ringed networks, and multi-substation clustering.

## Removed Deprecated APIs

- The legacy EW implementations (`ClassicEW`, `CPEW`, `NBEW`, `OBEW`, `EW_presolver`) and `optiwindnet.interface` were removed; use `heuristics.constructor()` or the `WindFarmNetwork` router API.
- `hgs_multiroot()` and `iterative_hgs_cvrp()` were removed in favor of `hgs_cvrp()`; `lkh()` and `iterative_lkh()` were removed in favor of `lkh3()`.
- The new implementations are more capable than the ones they replace, please report if you find a regression.

## Refactoring & Maintenance

- Major overhaul of the test scripts.

# v0.2.3

[Commit history since v0.2.2](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.2.2...v0.2.3)

Drop-in replacement for v0.2.2. The APIs deprecated in v0.2.2 are still available and still emit `DeprecationWarning`; they are scheduled for removal in v0.3.

## Important Changes

- **Balanced Subtrees Actually Balanced (MILP and HGS)**: `balanced=True` promises subtree loads differing at most by one unit, but both solver families only bounded those loads from below, letting a subtree grow up to `capacity`. This is now fixed consistently on both sides:
  - _MILP_: the upper bound `ceil(T / feeders)` is now enforced across the pyomo, SCIP and OR-Tools model builders.
  - _HGS_: `hgs_cvrp()` used to request more slack nodes than there were routes whenever `T < feeders * (capacity - 1)`, forcing the surplus through clipped `inf` arcs and inflating the reported objective. The balanced sub-problem is now solved at `capacity_effective = ceil(T / feeders)`, reported in `solver_details`.
- **Exact Feeder Count (MILP and HGS)**: a new, mirrored way to pin the feeder count instead of only bounding it from above. Because `balanced` is only expressible once the feeder count is pinned, this makes balanced solutions reachable above the minimum feeder count.
  - _MILP_: `feeder_limit="exactly"` pins the feeder count to `max_feeders` (whereas `"specified"` remains an upper bound).
  - _HGS_: `hgs_cvrp(vehicles=F, vehicles_exact=True)` — exposed as `HGSRouter(feeder_limit=F, feeder_exact=True)` — pins the feeder count to `F`, whereas `vehicles` alone remains an upper bound that HGS-CVRP normally undershoots. It currently requires `balanced=True` and a single substation; `F` must lie between `ceil(T / capacity)` and `T`.
- **Reversible lat/lon Coordinates**: `L_from_yaml()` and `L_from_pbf()` now project all coordinates into the single UTM zone holding the most turbines (instead of the zone of the first point), minimizing distortion for the bulk of the layout, and retain that zone as the graph attributes `utm_zone_number` and `utm_zone_letter`. This makes `VertexC` reversible back to lat/lon via `utm.to_latlon()`. Multi-zone `.yaml` input no longer raises an assertion.
- **Tunable `EWRouter`**: the `method` and `bias_margin` parameters of `heuristics.constructor()` are now exposed on `EWRouter`, giving access to the `esau_williams`, `biased_EW`, `rootlust` and `radial_EW` methods from the high-level API.
- **Concurrent HiGHS**: the pyomo HiGHS solver now runs a concurrent branch-and-bound tree search, in line with the other MILP backends. This speedup requires `highspy` v1.15 or newer; older versions accept the setting but run the search serially.
- **New Locations**: added Revolution, Sunrise, Hornsea 2, Norfolk Vanguard West, East Anglia 3 and Hollandse Kust Noord.

## Bug Fixes

- **Multi-Root Warmstart Eligibility**: `is_warmstart_eligible()` compared only the first root's feeder count against the feeder limit, while the model constrains the total across all roots.
- **`PathFinder` Malformed Chain**: two spanning fences of the same subtree meeting at a single chain-end vertex form a dead-end pocket, which could leave a chain short an access cone. Chain detection is now keyed on the subtree alone within a chain-end vertex.
- **`scaffolded()` Correctness**: fence hops and shortened-contour hops are now converted to primed edges (in both `PathFinder.scaffolded()` and `interarraylib.scaffolded()`), and a supertriangle/clone id collision was fixed.
- **Sexagesimal Coordinate Parsing**: `_translate_latlonstr()` failed to reset minutes/seconds between coordinates, corrupting the parsed values.

## Deprecated

- `WindFarmNetwork.from_yaml()` is renamed to `WindFarmNetwork.from_own_yaml()`; the old name still works and emits a `DeprecationWarning`.
- Reminder — the following remain available in this release and will be removed in v0.3. Users are advised to migrate now:
  - Standalone EW heuristics (`ClassicEW`, `CPEW`, `NBEW`, `OBEW`, `EW_presolver`) → `heuristics.constructor()` (or the high-level `WindFarmNetwork`/`EWRouter`). See the [Legacy heuristics migration guide](https://optiwindnet.readthedocs.io/stable/notebooks/lo34_legacy_heuristics.html), which pairs each legacy call with its `constructor()` equivalent.
  - `optiwindnet.interface` (`heuristic_wrapper()`, `HeuristicFactory`) → `WindFarmNetwork`/`EWRouter`.
  - HGS aliases `hgs_multiroot()` / `iterative_hgs_cvrp()` → `hgs_cvrp()`.
  - LKH entry points `lkh()` / `iterative_lkh()` → `lkh3()`.

## Refactoring & Maintenance

- **Docstrings**: project-wide docstring formatting pass — standardized markup of string values for Sphinx rendering, unicode arrows, and docstrings for the database model.
- **Test Coverage**: expanded coverage for `geometric`, `interarraylib`, `plotting`, `svg`, `repair`, `themes` and `baselines.utils`.
- **CI**: isolated OR-Tools in a shared subprocess to resolve solver DLL conflicts, switched SCIP download to GitHub, bumped SCIPOptSuite, installed python tooling from conda, and added manual-dispatch pipeline targets.

# v0.2.2

[Commit history since v0.2.1](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.2.1...v0.2.2)

## Breaking Changes

- **Advanced API Cleanups**: Helper functions for root-assignment and link-blockage moved from `optiwindnet.geometric` to `optiwindnet.interarraylib`. Users should import `add_terminal_closest_root()`, `add_link_blockmap()`, and `add_link_cosines()` from `optiwindnet.interarraylib`.

## Important Changes

- **Default Vector SVG Plotting**: High-level `WindFarmNetwork` plotting methods (`plot()`, `plot_location()`, `plot_available_links()`, `plot_navigation_mesh()`, and `plot_selected_links()`) now use a modern, interactive vector SVG plotting backend (`svgplot`/`svgpplot`) by default. This delivers clean, high-resolution inline displays in Jupyter notebooks. The legacy Matplotlib-based backend remains fully accessible by passing an explicit `ax` argument (including `ax=None` to dynamically instantiate Matplotlib figures).
- **svgplot() matches gplot()'s features**: SVG plots now support node labeling, boundary/obstacle vertex tagging, and figure legend.
- **Informative String Representations**: Added descriptive, debugger-safe string representations (`__repr__`) for `WindFarmNetwork` and `Router` subclasses (`EWRouter`, `HGSRouter`, `MILPRouter`) displaying key configuration parameters and solved network metrics.
- **Shorter Substation Labels**: Pre-packaged offshore wind farm datasets (.osm.pbf format) have been updated with short, human-readable substation abbreviations (such as "Alpha", "Beta", "OSS") to fit cleanly in visualization labels.
- **New Fused Heuristic**: Added `heuristics.constructor()` with `esau_williams`, `biased_EW`, `rootlust`, and `radial_EW` methods, unifying the constructive routing heuristics. The high-level `EWRouter` now uses this path, offering radial topology and the performant rootlust method.
- **LKH-3 Solver Parity**: Added `lkh3()` as the preferred LKH entry point, bringing it to feature parity with the HGS solver. It supports single- and multi-root configurations, per-root clustering, warm starts, capacity-violation retries, crossing repair, and improved solver metadata.
- **Expanded Crossing Diagnostics**: Added Shapely-based `find_geometric_crossings()` for geometry-first validation of arbitrary routesets, including detours, contour clones, shared-run overlap crossings, and branch-split cases.
- **Robust PathFinder Detours**: Major robustness improvements when routing detours among cable routes that follow boundaries or exclusion zones, significantly reducing cable use on sites with many obstacles.

## Deprecated

- Standalone EW heuristics (`ClassicEW`, `CPEW`, `NBEW`, `OBEW`, and `EW_presolver`) are deprecated and will be removed in v0.3. They are superseded by the new unified `heuristics.constructor()`. Note that `constructor` expects the available-links graph `A`, not the location graph `L`.
- The legacy `optiwindnet.interface` module (`heuristic_wrapper()`, `HeuristicFactory`) is deprecated and will be removed in v0.3; use `WindFarmNetwork`/`EWRouter` instead.

## Fixes

- **Pathfinder Robustness**: Resolved fatal crashes (`KeyError` and triangulation flip failures) when constructing detours.
- **Diagonal Mesh Exclusion**: Prevented invalid diagonal paths by skipping edges in the site's boundary polygon during navigation mesh generation.
- **Logging & Diagnostics**: Replaced all remaining raw `print()` statements across the API and utility modules with standard Python logging.
- **LKH and Heuristic Repairs**: Fixed LKH warm-start tour construction (indexing, walk order within clusters) and aligned HGS/LKH repair behavior for capacity-violating and crossing routes.
- **Overflow Prevention**: Added checks for LKH weight-matrix construction with clear guidance when inputs need normalization.
- **Crossing Detection**: Fixed shared-route overlap crossing detection and added geometric handling for route intersections not expressible as available-edge crossings.

## Refactoring & Maintenance

- **Python 3.11–3.14 Support**: Explicitly declared support for Python 3.11 through 3.14 with standard Trove classifiers on PyPI.
- **Updated OR-Tools Floor**: Aligned OR-Tools requirements in `pyproject.toml` to `>=9.14.6206` for consistency across development and production environments.
- **Strict Deprecation Testing**: Test suite configured to treat `DeprecationWarning` as errors to guarantee API health.
- **Linting & Code Quality**: Enforced strict Ruff linting and formatting rules via continuous integration.
- **Performance Optimizations**: Optimized pathfinding sector lookups and precomputed chain-end topologies to speed up execution.

# v0.2.1

[Commit history since v0.2.0](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.2.0...v0.2.1)

## Breaking Changes

- `optiwindnet.db` module only: **RouteSet schema slimmed (v4)**: `num_gates` was renamed to `feeders_per_root`; the unused `valid`, `is_normalized`, and `stuntC` columns were removed. The `python -m optiwindnet.db.migrate` script now writes the v4 schema and accepts both v2 (Pony ORM) and v3 (Peewee) source databases.

## Important Changes

- **OR-Tools MILP backend switched to MathOpt API** (replacing `cp_model`), enabling multiple backends through a unified wrapper. No impact on use through either API.
- **Pyomo CPLEX/Gurobi solvers switched to the persistent interfaces** (`cplex_persistent`, `gurobi_persistent`). Relevant for successive calls to solver.solve().
- **`PathFinder`**: `A` is now a mandatory argument, it is relied upon to inform about tentative feeder crossings (saves the repeated check done before); default options were updated; search heuristics improved.

## Features

- LKH improvements: iterations are also triggered on capacity violations, new `warmstart` argument.
- Better estimation of obstructed feeder lengths pre-optimization (make_planar_embedding).
- Default thread count for MathOpt solvers set to the number of physical cores; non-OR-Tools solvers also use `physical_core_count()`.
- New context-managed database connection API; `open_database()` and `database_connection()` accept a `timeout` argument.
- `.osm.pbf` parsing now accepts locations without borders.
- Added typing stubs and improved type annotations.
- Informative string repr for `SvgRepr`.

## Fixes

- Multiple PathFinder robustness fixes: collinear vertices in funnel apex update, expansion of `P_paths` shortcuts when building contour clones, shortcut provenance tracking for barriers, cumulative turning check for dropping traversers, and `bad_streak` decay on first arrival.
- `make_planar_embedding` fixes: constraint checks and line-of-sight tagging now use Shapely's `STRtree`, proper handling of diagonal promotion conflicts in concave meshes, string-pulling skipped when only one border vertex is on the path (enabled by STRtree check).
- `validate_routeset()`: corrected detour index range; touchpoint set as bunch-split corner apex.
- LKH: replaced stale `_add_link_blockage` call with `add_link_blockmap`.
- Removed WAL mode from the SQLite open pragma (caused issues on shared clusters).
- Stunt vertices are no longer placed in `G` (regression since a70b575).
- Gracefully handle repeated extents' vertices in .yaml input files.
- Fixed `migrate.py` ImportError.

## Refactoring & Maintenance

- Replaced `dill` with `pickle` everywhere it was used in tests.
- Removed `stuntC` from the routeset saving path.
- CI now runs a test matrix covering Python 3.11–3.14 (default bumped to 3.14); release requires passing on all versions.
- Increased test coverage and added topology-aware routeset comparison to prevent spurious failures.

## Documentation

- Major refactor of the Topfarm integration example (now including substation trajectory); several notebook updates.
- Added links to TOPFARM and Ard, updated preamble with Jupyter tutorial links.
- Acknowledged the DFF grant in README and doc index.
- Improved docstrings and setup instructions.

# v0.2.0

[Commit history since v0.1.6](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.1.6...v0.2.0)

## Breaking Changes

- **HGS-CVRP interface unified**: `baselines.hgs` functions `hgs_multiroot()` and `iterative_hgs_cvrp()` are deprecated; `hgs_cvrp()` replaces them with no loss in functionality. Users of HGSRouter from the Network/Router API will not notice the change.
- **Database format updated to v3**: Switched from Pony ORM (incompatible with Python 3.13+) to Peewee. Use `python -m optiwindnet.db.migrate input.v2.sqlite output.v3.sqlite` to migrate existing databases.

## Features

- Obstacles are now supported in `turbinate()` and `poisson_disc_filler()`.
- `as_normalized()` now works also with `L`.
- Replaced `multiprocessing.Pool` with `concurrent.futures.ThreadPoolExecutor` in HGS-CVRP calls, enabling concurrent solver instances without the quirks of the multiprocessing module. This requires a new version (v0.1.1+) of dependency hybgensea.
- Removed dependency `py`.

## Fixes

- Fixed potential infinite loop in `PathFinder` for inconsistent graphs.

# v0.1.6

[Commit history since v0.1.5](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.1.5...v0.1.6)

Drop-in replacement for v0.1.5. This release provides maily two important fixes:

- fix bugs caused by ortools v9.15.6755 released on 2026-01-12
- remove a duplicate turbine from the included location Gangkou 2

In addition, the graph attribute 'creator' of solutions produced by OWN was reverted back to using the naming convention adopted in earlier OWN versions, which includes the 'pyomo' string if the solver was called through it (e.g. 'MILP.pyomo.cplex' instead of 'MILP.cplex').

# v0.1.5

[Commit history since v0.1.4](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.1.4...v0.1.5)

Drop-in replacement for v0.1.4.

## Features

- Added new offshore wind locations: Dogger Bank B/C, Coastal Virginia, Inch Cape, Changhua 1, Gangkou 1/2, Yunlin, Noirmoutier, Tréport, Borkum Riffgrund 3, He Dreiht.
- Experimental **FiberSCIP (fscip)** solver support (system call, file-based interface).
- Improved automatic `landscape_angle` calculation
- Added `as_obstacle_free()` method to remove location obstacles; improved `as_single_root()`.
- `.osm.pbf` parsing now prioritizes tag `ref` over `name` for node labels.

## Fixes

- Fixed dangling reference in diagonals (`make_planar_embedding()`) which could cause errors when checking for crossings.
- Applied rounding in `_link_val()`/`_flow_val()` for MILP Solvers CPLEX and SCIP to eliminate tiny non-zero values (error manifested as cyclic solutions).
- Corrected setting of `B` in `L_from_windIO()`.
- Resolved `_hull_processor()` edge case (wrong P for Yunlin).
- Ensured roots are added to solution topology `S` even if disconnected.
- Enforced integer values for SCIP model variables.
- Updated deprecated Shapely `buffer()` argument name.
- Adjusted graph attributes in MILP solvers.
- Multiple robustness improvements in tests and solver handling.

# v0.1.4

[Commit history since v0.1.3](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.1.3...v0.1.4)

Drop-in replacement for v0.1.3.

- gplot() and svgplot() now draw links with different line thickness to represent cable type (after assign_cables() is called)
- improve number formatting inside infobox of gplot() and svgplot()
- switch SCIP modelling from Pyomo to PySCIPOpt, enabling the launching of concurrent solvers for the same problem (competitive mode)
- refactor MILP code for reducing code duplication and improving consistency between model descriptions for the different APIs
- add information on how to install missing solvers when a requested solver is not available
- bump dependency NetworkX version to 3.6 (resolves pickling issues with nx.PlanarEmbedding)
- update the documentation to reflect the changes involving solver SCIP and plotting functions
- fix the assignment of graph attributes 'creator' (all solvers) and 'runtime' (scip)

# v0.1.3

[Commit history since v0.1.2](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.1.2...v0.1.3)

Another minor version bump to enable conda-forge recipe to work.

- improve tests coverage
- restructure tests to skip unavailable MILP solvers
- make db.modelv2 handle only schema definition
- get correct runtime for MILP solver SCIP

# v0.1.2

[Commit history since v0.1.1](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.1.1...v0.1.2)

Minor version bump to enable conda-forge recipe to work.

- include tests in source distribution (sdist tarball)
- update docs to state Python 3.11 and 3.12 are recommended

# v0.1.1

[Commit history since v0.1.0](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.1.0...v0.1.1)

## 📦 Packaging

- drop Python 3.10 support (v0.1.0 had an inconsistency due to NetworkX v3.5)
- minor syntax fix in pyproject.toml to make conda-forge package possible

# v0.1.0

[Commit history since v0.0.6](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.0.6...v0.1.0)

## ✨ New Features

- **Thor Offshore Wind Farm**: Added to location repository.
- **Lin-Kernighan-Helsgaun Meta-Heuristics solver (Advanced API only)**:
  - Introduced `iterative_lkh()` to deal with crossings.
  - Switched LKH to OVRP problem type.
  - Automatic prunning poor links from the available choices given to LKH.

## 🛠️ Fixes & Improvements

- Fixed runtime reporting for solver HiGHS.
- Adapted MILP code to Pyomo API v2.
- Enforced radial topology in HGSRouter.
- Improved hull construction and shortcut creation in planar embedding.
- Handled multiple crossings by single link in iterative meta-heuristics calls.
- Reduced rogue link usage in LKH.
- Improved precision handling in `lkh_acvrp()`.
- Improved handling of scaling parameters and significant digits.

## 🔧 Refactoring & Code Quality

- Removed `**kwargs` from key initializers.
- Improved consistency across HGS and LKH meta-heurists functions.
- Cleaned up angle helper utilities.
- Increased test coverage.

## 📚 Documentation

- Added advanced example notebook for LKH.
- Fixed typos and improved clarity in README and notebooks.
- Updated figures and notebook outlines for better HTML rendering.

## 📦 Dependencies

- Removed `pyyaml-include` dependency.
- Bumped `numba` version and removed `numpy` version cap.

# v0.0.6

[Commit history since v0.0.5](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.0.5...v0.0.6)

- Almost a drop-in replacement for v0.0.5
  - single existing API change: argument name of HGS meta-heuristics: from max_reruns to max_retries
- Introduction of Network/Router high-level API for easier on-boarding of new users
  - Two new components -- WindFarmNetwork and Router -- expose most of OWN's features
- Major expansion and improvement of the documentation
  - Improved the Advanced API docs
  - Fully documented the Network/Router API
  - Added Topfarm integration example
  - Added the OptiWindNet logo
- Added automated code testing based on pytest and tests for the main components
- MILP model warm-starting is now checked for feasibility before invoking the solver (Pyomo-only)
- Silenced warnings of Pyomo-based solvers when the search times out before the gap is reached
- Other small fixes and improvements

# v0.0.5

[Commit history since v0.0.4](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.0.4...v0.0.5)

- drop-in replacement for v0.0.4
- gplot()' options improvements:
  - 'node_tag=True' plots node numbers
  - 'node_tag="load"' now also plots the roots' loads
  - 'tag_border=True' plots numbers of border/obstacle vertices
- gplot() and svgplot() now can plot sites without borders
- bug fixes and improvements in path-finding
- bug fixes and improvements in navigation mesh generation
- mesh generation now can handle terminals placed on border lines
- some paperdb incomplete or incorrect entries were fixed
- other small fixes and improvements

# v0.0.4

[Commit history since v0.0.3](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.0.3...v0.0.4)

- fixed exception AttributeError on MacOS ('Process' object has no attribute 'cpu_affinity')
- added 3 more locations (Hollandse Kust Zuid, Vineyard 1, Sofia)
- enabled easy wind farm creation and import using JOSM (external program with GUI)
- many improvements in docstrings and documentation in general

# v0.0.3

[Commit history since v0.0.2](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.0.2...v0.0.3)

- merged all features from the paper's computational experiments
- introduced a new API for MILP solvers
- introduced a multi-root capable HGS-CVRP wrapper
- several bug fixes

# v0.0.2

[Commit history since v0.0.1](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/interarray-0.0.1...v0.0.2)

- project renamed to OptiWindNet and package to optiwindnet
- many more changes and bug fixes

# interarray-0.0.1

First release.
