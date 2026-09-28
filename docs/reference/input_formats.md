# Input Formats

A problem instance for _OptiWindNet_ consists of a **location** — the geometry — and the properties of the **available cable types**. This page is the catalogue of what an instance must contain and of the formats it can be given in, independently of which API you use to load it. What the loaded instance becomes is described in [](/problem.md#graph-representations).

## What an instance requires

| Item | Required | Notes |
| --- | --- | --- |
| Turbine coordinates | yes | One planar `(x, y)` position for each turbine. |
| Substation coordinates | yes | One planar `(x, y)` position for each substation, in the same coordinate system. |
| Cable types | yes | See [](/reference/input_formats.md#cable-types). |
| Border | no | A polygon delimiting the area where cables may be laid. |
| Obstacles | no | Polygons inside the border where cables may not be laid. |

_OptiWindNet_ works without a border and obstacles; the geometry constraints simply do not apply in that case. The geometric data carries much more volume and complexity than the cable properties, which is why the input formats below are concerned almost entirely with it.

_In use:_ {doc}`/notebooks/hi11_data_input` (Network/Router API) · {doc}`/notebooks/lo11_data_input` (Advanced API).

## Cable types

In the Network/Router API, cable capacity uses the unit selected by `power_unit`: nominal power when a unit such as `'MW'` is supplied, or integer inflow when it is `None`. With the default unitary turbine inflow, capacity counts turbines. The package does not convert electrical current ratings to power capacities; supply capacities appropriate to the site. Three ways of declaring the available types are accepted:

- a single number — the maximum capacity among all available cables, when cost is not of interest;
- a list of capacities — one entry per cable type;
- a list of `(capacity, linear_cost)` pairs — capacities must be increasing, and cost is per unit of length.

Only the last form supplies prices for reporting a network cost. All built-in routers optimize cable length using the largest capacity, including when prices are supplied. Cable types are assigned afterwards: each link receives the first type in increasing capacity order that can carry its load. Use costs that are nondecreasing with capacity if this is to select the cheapest feasible type. A shorter network need not have a lower cost when its loads require more expensive cables.

In the Advanced API, pass `capacity` (integer inflow) or `capacity_nominal` (declared power units) to the routing function, then use `assign_cables()` with `(capacity, linear_cost)` pairs to price the routed result.

_In use:_ {doc}`/notebooks/hi11_data_input` (Network/Router API).

### Turbines of unequal output

Pass `power_unit` (e.g. `'MW'`) and `turbine_powers` to `WindFarmNetwork` to give each turbine a nominal power in that unit. Cable capacities are then nominal power as well, held as exact fractions in the `cables` property. Floats are rationalized to the simplest fractions that round back to them; fractions keep their exact values. A location `L` that declares a `power_unit` (as the OptiWindNet YAML and OSM importers produce) supplies the powers where `turbine_powers` is omitted, and requires `power_unit` to be that same unit or None.

With `power_unit=None` (the default), `turbine_powers` and the cable capacities are integer inflow, and floats are refused, even whole ones. Without `turbine_powers`, each turbine injects unitary inflow, so capacities count turbines. A power declaration on `L` is dropped from the network's copy, logged as a warning where the turbines differ, since their ratings are then ignored, and at INFO level where they are equal.

**Declaration.** `set_turbine_powers(L, powers, power_unit=None)` declares power on a location or available-links graph. Equal powers become the graph attribute `power_per_inflow`, the common power. Unequal powers become each terminal's `'power'` (a `Fraction`) and the graph attribute `powers_set`, the sorted distinct powers. Without a unit, a terminal's `'inflow'` (a positive integer, one where absent) states its integer contribution. `L` and `A` carry no quantized inflow for unequal powers. `validate_terminal_power()` checks these conventions at graph-input boundaries and normalizes declared powers to fractions.

**Quantization.** Loads and capacities are integer inflow inside every router, so unequal powers are quantized for each solve, for one nominal capacity — in `WindFarmNetwork`, the largest cable's. `quantize_for_capacity(powers_set, capacity_nominal, power_rtol)` returns `(inflow_by_power, power_per_inflow, capacity)`. It takes the coarsest `power_per_inflow` for which every `inflow * power_per_inflow` stays within `power_rtol` of the declared power (inflow being their ratio rounded half up, at least one) and which **packs the turbines into the cable exactly as their nominal power does**: a cable of `floor(capacity_nominal / power_per_inflow)` inflow admits the same sets of turbines as its nominal capacity. The routed network is thus the one the nominal powers call for. The exact `power_per_inflow` always preserves the packing; a tolerance that no `power_per_inflow` meets without inflow beyond what a MILP can carry raises `ValueError`. Results are memoized.

`power_per_inflow` is an exact rational, so `power_rtol=0` makes `inflow * power_per_inflow` the declared power itself. A looser tolerance yields smaller inflow, which is easier on the MILP solvers: under a 20 MW cable, 5 MW and 6.35 MW turbines weigh 100 and 127 inflow of 1/20 MW at `power_rtol=0`, and 7 and 9 inflow of 17/24 MW at the default 0.01. `power_rtol` is an argument of each `Router` (`EWRouter`, `HGSRouter`, `MILPRouter`).

`quantized(A, *, capacity=None, capacity_nominal=None, power_rtol)` validates `A` and returns `(A_solve, capacity, attrs)`, where `A_solve` is a copy of `A` whose terminals carry the quantized `'inflow'` if the powers are unequal. `capacity` (inflow) requires `A` to declare no unequal power; `capacity_nominal` requires `A` to declare power. `constructor()`, `hgs_cvrp()`, `lkh3()` and `Solver.set_problem()` accept exactly one of the two, quantize through `quantized()`, and record `power_per_inflow`, `capacity`, `capacity_nominal` and `power_rtol` on the topology `S`. `G_from_S(S, A)` combines the declaration from `A` with the quantization from `S`; `S_from_G()` keeps terminal inflow and the quantization, not the declared power. `Solver.set_problem()` recomputes the loads of a warm start quantized otherwise.

**Reporting.** `assign_cables(G, cables)` compares nominal capacities with nominal loads where `G` has `capacity_nominal`, and inflow capacities with integer loads otherwise. Reassigning `WindFarmNetwork.cables` keeps the solution, with its cable types reassigned, unless its largest load (nominal or integer) exceeds the largest capacity, which invalidates it; `L` and `A` are left untouched. The graph attributes `capacity` and `max_load`, the plot infobox and `describe_G()` stay in inflow; `describe_G()` lists the distinct terminal inflow where they differ. The quantization of a solution is read from `wfn.G.graph['power_per_inflow']`.

The graph attribute `power_quantization_inexact` on a routeset `G` records whether any declared power differs from its inflow times `power_per_inflow`; `validate_terminal_power()` establishes it. `node_tag='power'` in `svgplot()` and `gplot()` labels declared ratings, falling back to `inflow * power_per_inflow`; `node_tag='load'` shows cumulative integer inflow. `node_tag='load_nominal'` and `get_network()` generate nominal loads on demand: with exact quantization, they scale the integer loads by `power_per_inflow`; with inexact quantization, `calcload(G, nominal=True)` accumulates the declared ratings into `'load_nominal'`, so feeder loads add up to the site's declared power. After editing terminal power or inflow directly, call `validate_terminal_power()` and recalculate the loads.

This convention breaks compatibility with v0.3.0 graphs that used `'power'` for integer injections: rename that attribute to `'inflow'`. The integer-injection helpers are `terminal_inflow()` and `total_inflow()`, and `TerseLinks.to_topology()` and `to_routeset()` quantize the power of the site graph they are given.

**Router support.** Nonunitary inflow restricts the routers:

| Router | Topology | `feeder_limit` | Other conditions |
| --- | --- | --- | --- |
| `MILPRouter`, `solver_factory()` backends | radial, branched | `'unlimited'`, `'exactly'`, `'specified'` | — |
| `HGSRouter`, `hgs_cvrp()`, `lkh3()` | radial | upper bound only | unbalanced, single substation; inflow becomes customer demand |
| `EWRouter`, `constructor()` | unsupported | — | subtree construction uses turbine counts rather than unequal demands |

`'minimum'` and `'min_plus*'` are rejected because the count they derive from total inflow treats a turbine as divisible among feeders. A `ringed` model is rejected because a ring within twice the capacity need not split into two arms that each fit within it. With a loose `power_rtol`, unequal powers may quantize to unitary inflow, which every router supports.

**Storage.** The database and the compact link encoding do not preserve individual turbine ratings. `pack_G()`, and so `store_G()`, refuses a routeset with nonunitary inflow or `powers_set`; a routeset of equal powers is stored with its power attributes. `WindFarmNetwork.update_from_terse_links()` quantizes the location as its router does, so a round trip through that method keeps the loads.

_In use:_ {doc}`/notebooks/hi16_mixed_power` (Network/Router API), {doc}`/notebooks/lo16_mixed_power` (Advanced API).

## Input formats

Four formats are accepted. All of them produce the same location graph `L` described in [](/problem.md#graph-representations), so the choice is purely one of convenience.

### Coordinate arrays

Coordinates are passed as _numpy_ arrays of `(x, y)` pairs — if the coordinates are held in separate one-dimensional arrays `X` and `Y`, use `np.column_stack((X, Y))`. Use a common planar coordinate system and length unit for turbines, substations and boundaries; reported lengths use that unit, and linear cable costs must use it too. A border polygon is defined by its sequence of vertices, with the segment closing the last vertex back to the first left implicit. Obstacles are given as a sequence of such polygons.

This is the format to use when the layout is generated programmatically, for instance inside an optimization loop driven by another tool.

_In use:_ {doc}`/notebooks/hi11_data_input` (Network/Router API) · {doc}`/notebooks/lo11_data_input` (Advanced API).

### windIO YAML

[windIO](https://github.com/IEAWindSystems/windIO) is a community data format for inputs and outputs of wind energy system models. Originally focused on systems engineering models, it has since been adopted across other areas of wind energy modeling. See the [windIO documentation](https://ieawindsystems.github.io/windIO/main/index.html) for the format itself.

_In use:_ {doc}`/notebooks/hi11_data_input` (Network/Router API) · {doc}`/notebooks/lo11_data_input` (Advanced API).

### OptiWindNet YAML

_OptiWindNet_'s own YAML schema is a compact way to keep a location in a file:

| Key | Required | Content |
| --- | --- | --- |
| `COORDINATE_FORMAT` | no | `planar` or `latlon` — defaults to `latlon`. |
| `EXTENTS` | yes | The border polygon. Do not repeat the initial vertex at the end. |
| `OBSTACLES` | no | A list of polygons, even when there is only one. |
| `SUBSTATIONS` | yes | Positions of the substations. |
| `TURBINE` | no | The turbine model's `make`, `model` and `power_MW` — a list of them, each with its `qty` and optional `prefix`, for a site of turbines of unequal power. |
| `TURBINES` | yes | Positions of the turbines, in input order. |

Coordinates are given either as lists of `[x, y]` pairs, for `planar`, or as a text block of latitude/longitude, for `latlon`:

```yaml
COORDINATE_FORMAT: latlon

SUBSTATIONS: |-
  OSS 56°35.748'N 11°09.174'E

TURBINES: |-
  A01 56°30.477'N 11°11.026'E
  A02 56°30.810'N 11°11.078'E
```

In the `latlon` form, any identifier placed _before_ the coordinates — `OSS`, `A01`, `A02` above — is loaded as the node's `label` attribute and can be shown in plots. Several examples are bundled in the folder `optiwindnet/data`; look for them in Python's `site-packages` or in [the repository](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/tree/main/optiwindnet/data).

The `TURBINE` section declares the turbines' nominal power in MW. A mapping states one `power_MW` for the whole site. A list gives one entry per turbine model. Each entry's `qty` claims the next group of positions in `TURBINES`, so the groups must follow the coordinate order and their quantities must add up to the total turbine count:

```yaml
TURBINE:
  - make: Siemens Gamesa
    model: SG 8.0-167 DD
    power_MW: 8
    qty: 94
  - make: Vestas
    model: V164-9.5
    power_MW: 9.5
    qty: 79
```

Where `TURBINES` is not grouped by turbine model, use a `prefix` for each model entry instead. It selects turbines by the start of their labels, regardless of their position in the coordinate list:

```yaml
TURBINE:
  - make: MHI Vestas
    model: V164-8.0
    power_MW: 8.25
    qty: 40
    prefix: 3-
  - make: Siemens Gamesa
    model: SWT-7.0-154
    power_MW: 7
    qty: 47
    prefix: 4-
```

Prefixes are all or nothing, must claim every turbine exactly once, and turn `qty` into a cross-check on how many each one matched.

`power_unit` becomes `MW` and the declared values are set with `set_turbine_powers()`, as described in [](/reference/input_formats.md#turbines-of-unequal-output). A list whose entries carry no `power_MW` declares no power. `L_from_yaml(..., read_powers=False)` ignores the section. A list entry with neither `qty` nor `prefix`, quantities that do not add up, prefixes that do not claim every turbine exactly once, and a `power_MW` that is not a positive finite number all raise `ValueError`.

_In use:_ {doc}`/notebooks/hi11_data_input` (Network/Router API) · {doc}`/notebooks/lo11_data_input` (Advanced API).

### OpenStreetMap PBF

`.osm.pbf` stands for _OpenStreetMap Protocol Buffer Binary Format_. It is the format to use when the location is digitized from a map.

The [JOSM](https://josm.openstreetmap.de/) open-source map editor is recommended for producing these files. The JOSM plugin **pbf** is required to save in the `.osm.pbf` format; the plugin **opendata** is useful for importing many common GIS file formats.

The OpenStreetMap objects used to represent a wind farm location are _nodes_, _ways_ and _multipolygons_ (a relation between closed ways):

| Element | Represented by |
| --- | --- |
| Wind turbine | a _node_ tagged `power=generator` |
| Substation | a _node_, or a closed _way_, tagged `power=substation` or `power=transformer` |
| Border | a closed _way_ tagged `power=plant` |
| Border with obstacles | a _multipolygon_ tagged `power=plant`, combining the closed _ways_ for the border and the obstacles — the _ways_ themselves are then left untagged |

A substation based on a _way_ is reduced to the centroid of the polygon that the _way_ defines. The node tags `name` or `ref` are loaded as the node's `label` attribute.

A generator's `generator:output:electricity` tag — a number followed by an optional unit, such as `8 MW` — declares its nominal power. It is loaded only if every generator declares a positive output in a common unit: the unit becomes the graph attribute `power_unit` and the outputs are set with `set_turbine_powers()`, as described in [](/reference/input_formats.md#turbines-of-unequal-output). Values with no number, such as `yes`, are ignored, and a location where only some generators declare their output carries no power at all. Generators declaring their output in more than one unit raise `ValueError`. `L_from_pbf(..., read_powers=False)` ignores the tags.

_In use:_ {doc}`/notebooks/hi11_data_input` (Network/Router API) · {doc}`/notebooks/lo11_data_input` (Advanced API).

## Location repositories

{py:func}`load_repository() <optiwindnet.importer.load_repository>` reads every `.osm.pbf` and `.yaml` file in a directory into a _namedtuple_ of NetworkX graphs, one per location. Called without arguments, it loads the locations distributed with _OptiWindNet_; called with a path, it loads a repository of your own. `read_powers=False` loads the locations without their declared turbine power.

The bundled locations are real offshore wind farms and are used throughout this documentation as ready-made examples.

Power declarations are read by default. In particular, Borssele, Trianel Windpark Borkum, and Walney Extension declare unequal turbine ratings: low-level routing calls on these sites require `capacity_nominal` and a router that supports the resulting inflow. To study them with capacity measured in turbine counts, load them with `load_repository(read_powers=False)`. The Network/Router API makes this choice through `power_unit`; its default `None` ignores imported nominal ratings.

_In use:_ {doc}`/notebooks/hi12_locations` (Network/Router API) · {doc}`/notebooks/lo12_locations` (Advanced API).

## Preparing the geometry

Boundaries digitized from a map are frequently not directly usable: obstacles may touch or intersect the border, concavities may be too narrow for a cable to pass, and turbines may sit marginally outside the allowed area. Merging obstacles into the border, buffering the boundaries to add a safety margin, and validating that every turbine and substation lies inside the allowed area are covered — with the geometry plotted before and after each operation — in {doc}`/notebooks/hi13_border_obstacles`.
