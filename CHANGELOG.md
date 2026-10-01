# v0.3.1

[Commit history since v0.3.0](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.3.0...HEAD)

## Breaking Changes

### Network–Router API

- **Automatic MILP warm starts:** `MILPRouter` builds a warm start for a fresh solve by default. `warmup=False` disables both construction and reuse; `warmup_time` sets the construction budget (default 0.2 s). [29ab4cdd](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/29ab4cdd9825c9b855d5a8a0c25273d5d90414d8)
- **Ringed feeder limits:** `HGSRouter.feeder_limit` counts substation connections, so it allows half as many rings as before. Odd limits raise `ValueError`. [f8d0c132](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/f8d0c132e21b1823308d6ae02dfcbf2dfb5d4cf1)
- **Power units:** with `power_unit` supplied, turbine powers and cable capacities are nominal power in that unit. Without it, they must be integer inflow; existing power declarations on the location are discarded. [efa6d5b6](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/efa6d5b696d954ee747cb6320177a27db2ebd3fe)
- **Cable reassignment:** assigning cables too small for the existing solution invalidates it instead of raising `ValueError`; call `optimize()` again. A solution that fits is retained and its cable types reassigned. [efa6d5b6](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/efa6d5b696d954ee747cb6320177a27db2ebd3fe)
- **HiGHS options:** `'highs'` uses native `highspy` and accepts native HiGHS option names only. Use `'pyomo.highs'` for the previous backend and its appsi configuration. [c92d28e1](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/c92d28e1e2b5ad000566e422f513b0c0ab60290b)
- **Buffered-location plots:** `plot_original_vs_buffered()` returns SVG by default, matching the other plot methods. Pass `ax` for Matplotlib. [6ce61fcc](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/6ce61fcc8727b252d66efeab2124874a483eb416)

### Advanced API

- **Keyword-only problem setup:** all `Solver.set_problem()` arguments after `P, A`, including `capacity` and `model_options`, must be named. [e4e87836](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/e4e878369ba26417956f932a8514e36530c711dd)
- **Terminal demand:** integer solver demand is stored as `'inflow'`, defaulting to one; `'power'` denotes nominal physical power. Rename integer-demand attributes or declare nominal power with `loads.set_terminal_power()`. [85bfaf3f](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/85bfaf3fbdf8efea7742c804bc000275eae574b3) [dd6d90ac](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/dd6d90ac1c9cd1fc61f433f3de9ef2e3cfc01bbe)
- **Mixed-power solves:** sites declaring unequal power require `capacity_nominal`, including in `TerseLinks` decoders. `EWRouter`/`constructor()` reject nonunitary inflow; HGS and LKH support it only for unbalanced, single-substation radial solves. MILP rejects nonunitary ringed models and feeder limits `'minimum'`/`'min_plus1/2/3'`. [b6275757](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/b62757570da519f974a8d8c54efc0803142d0771) [17d2c202](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/17d2c202d0432f4cfdc8800aa16570110a297177) [54904b26](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/54904b26a34f3a6864e27550b00c1b10aaf1e529)
- **Imported power:** YAML and OSM importers, and `load_repository()`, read declared turbine power by default. Pass `read_powers=False` for turbine-count inputs; bundled Borssele and Walney Extension declare unequal powers. [05519b1e](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/05519b1eabe4213c0a82500f62419b89c93cde13)
- **Persistence:** `pack_G()` and `store_G()` reject routesets with unequal power or nonunitary inflow because the storage schema has no terminal-power field. Equal-power attributes remain storable. [dd6d90ac](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/dd6d90ac1c9cd1fc61f433f3de9ef2e3cfc01bbe)
- **Enum options:** direct `make_min_length_model()` calls require enum values for `topology`, `feeder_route`, and `feeder_limit`; strings raise `TypeError`. `ModelOptions` still accepts strings. [7aeda317](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/7aeda317d85a0ebb3e23494803e650751d5193aa)
- **LKH balancing:** `lkh3(balanced=True)` raises `NotImplementedError` because LKH did not enforce it. Use `hgs_cvrp(balanced=True)`. [ce3df91f](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/ce3df91f43bf4cece9c223526d04f926a45c83e8)
- **Incumbent access:** `Solver.get_incumbent_topology()` is removed. Use `get_solution()` for routed results, or `S_from_linkbits(solver.incumbent_linkbits, A)` followed by `calcload()` or `split_rings_and_calc_loads()` for an unrouted topology. [531dcf0c](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/531dcf0cb1771ce5313df42d5f03c1e55f46d052)
- **Crossing diagnostics:** `crossings.describe_crossings()` is removed; use `validate_routeset()` for geometric validation. [c123ed2d](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/c123ed2d345f5ffabd03a2f1accd85f4e0e3df5f)
- **Ring helpers:** `add_ring_to_S()` and `rings_from_S()` are private implementation details and no longer public exports. [fd27af0b](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/fd27af0b70a0a1b80860962733203faf549e386b) [c4c84866](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/c4c848664ff2b9e05713f3848bbf8271e2822ec8)
- **Baseline utilities:** `optiwindnet.baselines.utils` and its distance-matrix utilities are removed. [d0c5e882](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/d0c5e8820462f204e21cec61b67ec0c539d658b3)
- **Deprecated imports:** relocated `interarraylib` functions and classes remain available through aliases that emit `DeprecationWarning`, scheduled for removal in v0.4.0. Import from `converting`, `loads`, `terse`, `transforming`, `validating`, or `presenting`; `fingerprint` similarly delegates to `identity`. [6bc02121](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/6bc02121b2422db9224cc2b0ff00f60b3b1d8922) [c123ed2d](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/c123ed2d345f5ffabd03a2f1accd85f4e0e3df5f)
- **Method identities:** HGS and LKH record `balanced`, `ringed`, `repair`, and `max_retries` in `method_options`. Their stored method digests change even for settings identical to earlier runs. [7708d170](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/7708d170ed67e07d33494d3c5737552a57277fd0)
- **Warm-start validation:** infeasible starts consistently raise `OWNWarmupFailed` at the solver boundary; `MILPRouter` handles replacement or cold-solve fallback. [56e3ba66](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/56e3ba6691a6b3936697da0220a51c62b0ce33c3) [c92d28e1](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/c92d28e1e2b5ad000566e422f513b0c0ab60290b)
- **Buffering helpers:** `buffer_border_obs()` returns only the location graph; `plot_org_buff()` is removed in favor of the standard plotting paths. [6ce61fcc](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/6ce61fcc8727b252d66efeab2124874a483eb416)
- **Diagonal maps:** mesh diagonals use `utils.BiMap`; mutate the forward mapping, not its `inv` dictionary. Previously stored `bidict` mappings remain supported. [bd2eff7c](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/bd2eff7c3eec8ced6fa83272794ee98a63c1f48e)
- **Native solver conflicts:** `solver_factory()` rejects loading incompatible native solver packages in one interpreter with `RuntimeError`. [be725269](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/be72526990e37677105ac1270cf89bf595539839)

## Features & Improvements

- **Mixed turbine powers:** `WindFarmNetwork` accepts nominal powers and cable capacities in an explicit unit, quantizes demands per solve with `power_rtol` (default 0.01), and exports network loads in nominal units. [efa6d5b6](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/efa6d5b696d954ee747cb6320177a27db2ebd3fe) [dd6d90ac](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/dd6d90ac1c9cd1fc61f433f3de9ef2e3cfc01bbe) [b6275757](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/b62757570da519f974a8d8c54efc0803142d0771)
- **Power declarations and quantization:** `set_terminal_power()` stores exact nominal ratings; `quantize_for_capacity()` preserves which turbine combinations fit a cable. `calcload(nominal=True)` and `assign_cables()` support nominal loads and capacities. [dd6d90ac](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/dd6d90ac1c9cd1fc61f433f3de9ef2e3cfc01bbe)
- **Power import:** YAML supports turbine-model blocks or label prefixes; OSM reads `generator:output:electricity`. [05519b1e](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/05519b1eabe4213c0a82500f62419b89c93cde13)
- **Demand-aware loads:** load calculation and topology conversion preserve terminal inflow and verify coverage independently of its magnitude. [f1166ede](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/f1166ede5e5143e488739876338f1add175a0fb7) [85bfaf3f](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/85bfaf3fbdf8efea7742c804bc000275eae574b3)
- **Demand-aware MILP bounds:** flow bounds account for each terminal's inflow, tightening the model for unequal demands. [6d1b96f1](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/6d1b96f1e15871d11908d5ade891f35c93602d33) [17d2c202](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/17d2c202d0432f4cfdc8800aa16570110a297177)
- **Demand-aware baselines:** HGS and LKH pass supported nonunitary inflows as customer demands; LKH checks route loads when retrying capacity violations. [68bfd132](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/68bfd132604d93e3585c255e6587105a681a4360) [95db911b](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/95db911bbdd4a1b6d011c5687b8c4e218d8962a2)
- **Exact ring counts:** balanced `hgs_cvrp(ringed=True, vehicles_exact=True)` pins the ring count, up to `T // 2`; `HGSRouter(feeder_exact=True)` exposes this through feeder counts. [7df0a6e4](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/7df0a6e47f83faa9f95ac2fe93015beed52b492e)
- **Broader MILP warm starts:** construction covers more balanced and feeder-limited models, derives counts from the model's bounds, and respects the heuristic's inflow support. [29ab4cdd](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/29ab4cdd9825c9b855d5a8a0c25273d5d90414d8) [b6848b76](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/b6848b767792c5d2f0d42650396ec9b86dc383a5) [7df0a6e4](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/7df0a6e47f83faa9f95ac2fe93015beed52b492e)
- **Native HiGHS:** the `highspy` backend ranks saved improving solutions by routed length. `'pyomo.highs'` remains available, and `'pyomo.cbc'` explicitly names the existing `'cbc'` backend. [c92d28e1](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/c92d28e1e2b5ad000566e422f513b0c0ab60290b)
- **CBCbox:** `'ortools.cbcbox'` builds with MathOpt and uses the CBC executable supplied by `cbcbox`, with warm-start support and DINS enabled by default. [7e3d9dbc](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/7e3d9dbc610cd5262b93245829b24d21bedcf1ea) [b5f888bc](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/b5f888bc90bb2b705ab3757753841b3cab92a1e3)
- **Complete-graph baselines:** HGS and LKH consistently support complete solves, explicitly or through edgeless input graphs; mesh-based repair is rejected for these solves. [d0c5e882](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/d0c5e8820462f204e21cec61b67ec0c539d658b3)
- **Canonical link bits:** candidate links have a shared order across producers; solutions carry a compact, solver-independent link representation. [b781c120](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/b781c120b0979ece429075ae0354a2f7704dce8f) [56a2443e](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/56a2443e97fc7a7a9d4c177259c63fde56008c77)
- **Solution identities:** `topology_id()` and `linkset_id()` identify topologies and their candidate-link sets; `SolutionInfo` exposes `topology_id`. [4f8e55b4](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/4f8e55b48070d133705545db848eaf8ec022a01c) [2c5d8d2b](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/2c5d8d2b112bd93b9f0f5be25b63a3cf199188d5) [1c8ee72b](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/1c8ee72ba2819ead90dfba42276fea8551c40e70)
- **Topology reconstruction:** `S_from_linkbits()` decodes a solution against its available-links graph. [9152a5fb](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/9152a5fb2d1b3517a20c3328fe110c214251954f)
- **Solution summaries:** `SolutionInfo` has a column-aligned summary, and `SvgRepr` reports solution properties alongside configuration. [71400404](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/71400404be276b1790d025d4bf6829bebbb49dc6) [e125fde4](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/e125fde4019ffb4943ff2c902cca9e9844fb2eb0)
- **Power-aware plots:** turbine symbols and legends distinguish ratings; `node_tag='power'` and `'load_nominal'` label nominal values. [a143825a](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/a143825a0e3fc4cc03b6510eaa45e9b829afea8d)
- **Tight SVG plots:** `tight=True` trims the viewBox; legends wrap to fit, and opaque backgrounds cover the final bounds. [ddacdbbc](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/ddacdbbc808caeaf32aeca1b10c2f277ece7c555)
- **Pre-buffer boundaries:** location plots overlay the original border and obstacles on their buffered versions. [6ce61fcc](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/6ce61fcc8727b252d66efeab2124874a483eb416)

## Bug Fixes

- **Warm-start recovery:** failed construction falls back to a cold MILP solve; SCIP permits repeated solves on an already-solved model. [56e3ba66](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/56e3ba6691a6b3936697da0220a51c62b0ce33c3)
- **PathFinder robustness:** searches converge more reliably on open layouts and bend at collinear turbines instead of routing directly over them. [c61a2177](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/c61a217752f01a8fa40225cc301b1ac1901b2665)
- **Straight feeders over turbines:** MILP models let a straight feeder pass over a turbine whose links stay on one side. [a25dde43](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/a25dde43ba54715ca413a9e6f46430355a2a1d5f) [9ccfdae0](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/9ccfdae039d1869540443bf438889dce2bca02fb)
- **Ring geometry:** route decomposition covers each ring edge once, and crossing checks accept root closures and corridors shared by ring arms. [c82c0570](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/c82c0570cc812543b5bb9f1f3821b6dff7bfeb8f)
- **Validation:** topology and routeset checks cover stored loads, connectivity, capacity, orientation, and geometry, including self-intersections and branch splits. [6f1c1252](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/6f1c12526c377802395fdf7136f6d85c3d9b4d37)
- **Corridor foldbacks:** geometric validation accepts noncrossing retraces and detects crossings at shared-run exits more accurately. [ad6fa982](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/ad6fa982d790f7ba047d382205b892345a9854b9)
- **Repair crossings:** conflict checks avoid false rejections, missed crossings, and `KeyError` on remapped diagonals. [822c5cba](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/822c5cbae69080338b93f16502076036c29a4d08)
- **Nearly straight quadrilaterals:** convexity tests reject spurious diagonals over collinear turbines. Rebuilt meshes can therefore omit links present in older stored routesets. [654360ab](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/654360ab9f3880fdc51053acb0169143388e6890)
- **Rehooked loads:** `as_hooked_to_nearest()` clears stale loads on clone vertices before recalculation. [d4bf19d9](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/d4bf19d92ee0fed1f1539b0a1587f72fbc518cf9)
- **Solver options:** Gurobi and CPLEX instances no longer share mutable option dictionaries. [31fb1c2e](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/31fb1c2ed0742fd60a3b0705a3df3fbadb05e320)
- **OR-Tools HiGHS options:** options reach the correctly typed HiGHS parameter maps. [146c5e94](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/146c5e94902ceae89c3bc97f8ed5b0001c26e81d)
- **Verbose output:** HGSRouter forwards its verbose flag, OR-Tools log forwarding recognizes the backend family, and CBCbox prints output when requested. [6c4e778c](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/6c4e778c81c8f7ee11747894781056bdceb4f9b9) [56e3ba66](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/56e3ba6691a6b3936697da0220a51c62b0ce33c3) [76a6ff1d](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/76a6ff1d735bca92553d14fd8158f1ce3f529ca3)
- **Rootless SVG plots:** graphs containing only borders and obstacles can be drawn without substation-slice errors. [97be6b3d](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/97be6b3dcfa4e1d441b080d5562b90e631d328fb)

## Performance

- **Multi-substation MILP models:** only each terminal's feeder to its closest substation, by contour-adjusted distance, remains available. [c2c680fa](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/c2c680facb20a3bab8289c232f3c6a4b4fd13e02)
- **MILP search:** flow-variable integrality is implied instead of branched on; CBC defaults are retuned for that formulation. [db5a83ad](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/db5a83adcf5774b1b10f0ae786f59337e2645b02) [9e043b05](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/9e043b055bb6b13c3a22fb8d90c438dd5562bb62)
- **Deferred topology decoding:** MILP searches retain incumbent link bits and decode topologies only when retrieving or ranking solutions. [531dcf0c](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/531dcf0cb1771ce5313df42d5f03c1e55f46d052)
- **Mesh construction:** triangulation uses CDT's NumPy array API, and contour detours use a lean A* search. [26014fed](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/26014fed1e27a6fd4e82406bf27b0635591a987b) [8dfe558f](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/8dfe558f26181b71cf5b8559ab319bb0db7dd426)
- **Geometric tests:** root visibility uses a Numba kernel with Shapely fallback; quadrilateral tests avoid repeated NumPy indexing and dispatch. [9256daa4](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/9256daa4f2f2b177218633e7149d93564e230c8a) [c49317da](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/c49317da406b94e3c7622aa4d9a5cfc03985c0ff)
- **PathFinder inner loops:** list-based coordinates, direct adjacency access, and guarded debug calls reduce overhead. [4efefee1](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/4efefee14a9c8141d7760128a2f7892da9f184a8) [f580ad1c](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/f580ad1c5a385cc6e670b0ae1eb90bfabe771e3b) [a1b7c0b6](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/a1b7c0b6ec8f0ef33adf8463b7b4d3c2c2803e7a)
- **Diagonal maps:** dict-based `BiMap` speeds map construction and crossing lookups. [bd2eff7c](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/bd2eff7c3eec8ced6fa83272794ee98a63c1f48e)
- **Load calculation:** iterative traversal avoids graph-view overhead and recursion-depth limits. [8dd0763b](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/8dd0763b6bd35fe589379c233d35b7381c2c4a44)
- **API imports:** Matplotlib loads only when a Matplotlib plot is requested. [6ce61fcc](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/6ce61fcc8727b252d66efeab2124874a483eb416)

## Refactoring & Maintenance

- **Module organization:** `interarraylib` is split into `identity`, `loads`, `converting`, `transforming`, and `validating`, with text rendering in `presenting` and encodings in `terse`. Shared problem setup lives in the base `Solver`. [2c5d8d2b](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/2c5d8d2b112bd93b9f0f5be25b63a3cf199188d5) [c123ed2d](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/c123ed2d345f5ffabd03a2f1accd85f4e0e3df5f) [e4e87836](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/e4e878369ba26417956f932a8514e36530c711dd)
- **Path annotations:** YAML constructors explicitly accept `Path` as well as `str`, matching existing runtime behavior. [628faf13](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/628faf133ce9500c4ac52931732444047120d1f1)
- **Regression identities:** mesh and solver tests use canonical link-set and topology identities. [ce5c9b65](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/ce5c9b65a1f1e40d0c7511d58ee8de436848a0cd) [97173ceb](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/97173ceb9e296745f3e9b121c0ad26eef4292318)
- **Sweep harness:** resumable A/B experiments record code state, metrics, and reconstructible solutions. [bb3932d8](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/bb3932d8b951447e449fdec1af995aab8d25b378)
- **Code checks:** linting, formatting, and type checking are enforced in CI, with declared tooling and stub dependencies. [9ce1a115](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/9ce1a1156cee2bf496caa70fb6dff1ab556b5a35) [e63eb9bc](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/e63eb9bc78e7c7f35c83df5a9415c2b1d32814bc) [9db83cff](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/9db83cff7610d16ff6d721b440f9f5e06e964c7e)
- **Dependencies:** minimum versions rise to `condeltri 0.0.6` and `peewee 3.17.9`; `xxhash` supplies in-memory identities. [26014fed](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/26014fed1e27a6fd4e82406bf27b0635591a987b) [76947c9d](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/76947c9d2ef0825b6279e7d7b7747d3bd952a8fe) [4f8e55b4](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/4f8e55b48070d133705545db848eaf8ec022a01c)

## Documentation

- **Manual and guides:** reorganized navigation, paired API tutorials, and expanded solver, power, validation, and warm-start references. [b3104957](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/b3104957f2367ef3469399ccf872734afdaa8023) [d066b245](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/d066b2455f80914031bbef4e3757d047f83b7a15) [c7e0b245](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/c7e0b245db30b7c08ae91fdf13ccb6546c84b2bd)
- **Notebook results:** stochastic examples specify seeds, solution summaries replace routine solver logs, and published outputs are refreshed. [6e5ccc47](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/6e5ccc471ab649d807934ceb156b9a7d387e20b6) [9fe41cb8](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/commit/9fe41cb837f87cc3135ace7d798d613b7f0a5853)

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
