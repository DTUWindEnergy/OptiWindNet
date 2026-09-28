# Turbine Power and Cable Capacity

Cable capacity can count turbines or represent nominal power, depending on the supplied power declarations. This page explains those inputs, their effect on router support, and the units used in results. For coordinates and file schemas, see {doc}`/reference/input_formats`.

## Power inputs and defaults

With the default `power_unit=None` and no `turbine_powers`, each turbine contributes one integer inflow, so cable capacity counts turbines. To use nominal ratings, supply one `turbine_powers` value per turbine in coordinate order and set `power_unit` (e.g. `'MW'`) on `WindFarmNetwork`. Cable capacities then use that same unit.

Equal nominal ratings use one inflow per turbine, worth the common rating. Unequal ratings are converted for each solve; choose a compatible router from [](#router-support).

| Input | Interpretation |
| --- | --- |
| `power_unit=None`, no `turbine_powers` (defaults) | Each turbine contributes one inflow, so cable capacities count turbines. |
| `power_unit=None`, explicit `turbine_powers` | Powers and cable capacities are integer inflow; floats are refused, including whole-valued floats. |
| `power_unit='MW'`, explicit `turbine_powers` | Turbine ratings and cable capacities are nominal MW. |
| `power_unit` matching a declaration on `L`, no `turbine_powers` | Use the location's declared ratings. OptiWindNet YAML and OSM imports can supply these. |

If `L` declares a power unit, `power_unit` must match it or be `None`. Choosing `None` drops the declaration from the network's copy: ignoring unequal ratings logs a warning, while ignoring equal ratings logs at INFO level.

With a physical unit, the `cables` property holds nominal capacities as exact fractions. Float inputs are rationalized to the simplest fractions that round back to them; supplied fractions keep their exact values.

_In use:_ {doc}`/notebooks/hi16_mixed_power` (Network/Router API). Graph-based inputs are covered in [](#advanced-api) and {doc}`/notebooks/lo16_mixed_power`.

## Cable specifications

Supply cable capacities in the units established by [](#power-inputs-and-defaults). The package does not convert electrical current ratings to power capacities; supply capacities appropriate to the site. The Network/Router API accepts three ways of declaring the available types:

- a single number — the maximum capacity among all available cables, when cost is not of interest;
- a list of capacities — one entry per cable type;
- a list of `(capacity, linear_cost)` pairs — capacities must be increasing, and cost is per unit of length.

Only the last form supplies prices for reporting a network cost. All built-in routers optimize cable length using the largest capacity, including when prices are supplied. Cable types are assigned afterwards: each link receives the first type in increasing capacity order that can carry its load. Use costs that are nondecreasing with capacity if this is to select the cheapest feasible type. A shorter network need not have a lower cost when its loads require more expensive cables.

In the Advanced API, pass `capacity` (integer inflow) or `capacity_nominal` (declared power units) to the routing function, following [](#quantization-contract), then use `assign_cables()` with `(capacity, linear_cost)` pairs to price the routed result.

_In use:_ {doc}`/notebooks/hi11_data_input` (Network/Router API).

## Router support

Routers use integer inflow internally. Unequal ratings are converted for each solve (see [](#quantization)); the following restrictions apply when the resulting inflow is nonunitary. The default `EWRouter` does not support that case. With a loose `power_rtol`, unequal ratings may quantize to unitary inflow, which every router supports.

| Router | Topology | `feeder_limit` | Other conditions |
| --- | --- | --- | --- |
| `MILPRouter`, `solver_factory()` backends | radial, branched | `'unlimited'`, `'exactly'`, `'specified'` | — |
| `HGSRouter`, `hgs_cvrp()`, `lkh3()` | radial | upper bound only | unbalanced, single substation; inflow becomes customer demand |
| `EWRouter`, `constructor()` | unsupported | — | subtree construction uses turbine counts rather than unequal demands |

`'minimum'` and `'min_plus*'` are rejected because the count they derive from total inflow treats a turbine as divisible among feeders. A `ringed` model is rejected because a ring within twice the capacity need not split into two arms that each fit within it. These restrictions also limit which models can be warm-started; see [](/routers.md#warm-starting).

## Quantization

Each solve converts unequal nominal powers to integer inflow for one nominal capacity — in `WindFarmNetwork`, the largest cable's. This conversion is called **quantization**. It preserves exactly which groups of turbines fit that capacity.

Every `Router` (`EWRouter`, `HGSRouter`, `MILPRouter`) accepts `power_rtol`, defaulting to `0.01`. This bounds the relative error per turbine between its declared power and `inflow * power_per_inflow`. Zero tolerance represents the declared powers exactly; a looser tolerance can yield smaller integers, which are easier on MILP solvers.

For 5 MW and 6.35 MW ratings with a 20 MW routing capacity:

| `power_rtol` | `power_per_inflow` (MW) | Inflow for 5 MW | Inflow for 6.35 MW |
| --- | --- | --- | --- |
| `0` | `1/20` | 100 | 127 |
| `0.01` (default) | `17/24` | 7 | 9 |

Both conversions admit the same turbine groups under the 20 MW limit. The guarantee concerns that routing capacity; smaller cable types are assigned using nominal loads as described below. The precise conversion rule and its failure case are in [](#quantization-contract).

## Interpreting results

Integer routing loads and nominal power are different quantities. Read the solution's conversion unit from `wfn.G.graph['power_per_inflow']`.

| Result or display | Meaning |
| --- | --- |
| Graph `capacity` and `max_load`, plot infobox, `describe_G()` | Integer inflow. `describe_G()` also lists distinct turbine inflows when they differ. |
| `node_tag='load'` | Cumulative integer inflow. |
| `node_tag='power'` in `svgplot()` or `gplot()` | Declared turbine rating, falling back to `inflow * power_per_inflow`. |
| `node_tag='load_nominal'` or `get_network()` | Nominal downstream loads, generated on demand. |

With exact quantization, nominal loads are integer loads scaled by `power_per_inflow`. With inexact quantization, they are sums of declared ratings, so feeder loads add up to the site's declared power. Multiplying integer loads by the conversion unit would only approximate those sums. The graph mechanism is described in [](#loads-and-graph-conversion).

`assign_cables(G, cables)` compares nominal capacities with nominal loads when `G` has `capacity_nominal`, and inflow capacities with integer loads otherwise. Reassigning `WindFarmNetwork.cables` keeps the solution and reassigns its cable types unless its largest load (nominal or integer) exceeds the largest capacity, which invalidates it. `L` and `A` are left untouched.

## Advanced API

Power declarations belong to the site graphs; quantization belongs to each solve. The following contracts describe how to carry both through the graph pipeline.

### Power declarations

In graph vocabulary, turbines are **terminals**. `set_turbine_powers(L, powers, power_unit=None)` declares power on a location or available-links graph. `validate_terminal_power()` checks the following conventions at graph-input boundaries and normalizes declared powers to fractions.

| Declaration | Graph representation |
| --- | --- |
| Equal nominal powers | Graph `power_per_inflow` holds the common power. |
| Unequal nominal powers | Each terminal's `power` is a `Fraction`; graph `powers_set` holds the sorted distinct powers. `L` and `A` carry no quantized inflow. |
| Integer inflow without a unit | Terminal `inflow` is a positive integer; absence means one. |

The integer-injection helpers are `terminal_inflow()` and `total_inflow()`.

### Quantization contract

`constructor()`, `hgs_cvrp()`, `lkh3()` and `Solver.set_problem()` accept exactly one of these capacity arguments:

| Argument                                  | Requirement on `A`            |
| ----------------------------------------- | ----------------------------- |
| `capacity` (integer inflow)               | No unequal-power declaration. |
| `capacity_nominal` (declared power units) | A power declaration.          |

These entry points quantize through `quantized(A, *, capacity=None, capacity_nominal=None, power_rtol)`. It validates `A` and returns `(A_solve, capacity, attrs)`, where `A_solve` is a copy of `A` whose terminals carry quantized `inflow` if the powers are unequal. The topology `S` records `power_per_inflow`, `capacity`, `capacity_nominal` and `power_rtol`.

The underlying `quantize_for_capacity(powers_set, capacity_nominal, power_rtol)` returns `(inflow_by_power, power_per_inflow, capacity)`. It chooses the coarsest exact rational `power_per_inflow` satisfying both rules:

- Each power divided by `power_per_inflow`, rounded half up and at least one, gives an inflow whose represented power stays within `power_rtol` of the declaration.
- Integer capacity `floor(capacity_nominal / power_per_inflow)` admits exactly the same sets of turbines as the nominal capacity.

The exact conversion always preserves packing. A tolerance that no conversion meets without inflow beyond what a MILP can carry raises `ValueError`. Results are memoized.

### Loads and graph conversion

`G_from_S(S, A)` combines the declaration from `A` with the quantization from `S`. `S_from_G()` keeps terminal inflow and the quantization, not the declared power. `Solver.set_problem()` recomputes the loads of a warm start whose quantization differs from the target solve.

On a routeset `G`, `validate_terminal_power()` establishes `power_quantization_inexact`, recording whether any declared power differs from its inflow times `power_per_inflow`. For inexact quantization, nominal reporting uses `calcload(G, nominal=True)` to accumulate the declared ratings into `load_nominal`. After editing terminal power or inflow directly, call `validate_terminal_power()` and recalculate the loads.

## Storage and compatibility

The database and compact link encoding do not preserve individual turbine ratings:

- `pack_G()`, and therefore `store_G()`, refuses a routeset with nonunitary inflow or `powers_set`. Equal-power routesets are stored with their power attributes.
- `TerseLinks.to_topology()` and `to_routeset()` quantize the power of the supplied site graph. `WindFarmNetwork.update_from_terse_links()` quantizes the location as its router does, so a round trip through that method keeps the loads.

The power convention breaks compatibility with v0.3.0 graphs that used `power` for integer injections: rename that attribute to `inflow`.
