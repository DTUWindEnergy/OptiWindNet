# The cable routing problem

_OptiWindNet_ designs the array cable network connecting wind turbines to substations. This page explains the inputs, design choices and results shared by both APIs. No API choice is needed to read it; {doc}`/routers` explains how to choose an optimization approach.

## Inputs and results

A problem starts with turbine and substation positions, the allowed cable-laying area and any exclusion zones, and turbine power and cable capacity. The result specifies which components connect, where their cables run and, when cable types are supplied, which type each connection uses. Cable prices can be supplied to report the resulting cost.

A **feeder** is a cable connection at a substation. A **subtree** is a group of turbines connected to a substation through a feeder; ringed networks have two arms, each served by a feeder. Turbines and substations are also called **terminals** and **roots** in the graph model. The {doc}`/reference/glossary` collects these terms.

Turbines have equal power by default, so capacity can be expressed as a number of turbines. Unequal ratings can be declared in explicit power units, but restrict the available optimization approaches and network architectures. See {doc}`/reference/power` for the input rules and {doc}`/reference/input_formats` for supported file formats.

## Feasibility and design choices

To be feasible, a solution must satisfy the following electrical and geometric constraints:

- branching, when permitted, can occur only inside a wind turbine;
- cables cannot cross;
- cable routes must remain inside the allowed area and avoid obstacles;
- the load carried by each cable must not exceed its capacity.

The network architecture and feeder requirements further constrain which solutions are acceptable.

## Network topologies

_OptiWindNet_ supports three electrical topologies. Each constrains how turbines connect to substations.

```{glossary}
branched
  Each subtree connects a group of turbines to a substation, with branching allowed at any turbine. In graph terms, the network is a forest of rooted trees. This is the default and least constrained topology, and therefore generally permits the shortest networks.

radial
  Each subtree follows a single path from a substation through its turbines. A turbine has at most two neighbors, so cables do not branch there. This simplifies switchgear but may increase cable length.

ringed
  Turbines are arranged in loops that begin and end at a substation, or connect two substations that are implicitly electrically interconnected. Only substations may belong to more than one loop. Each turbine has exactly two neighbors, except that a loop serving just one turbine is represented by a single link.
```

The following figure shows the same example wind farm solved under each topology:

```{image} /_static/fig_topologies_light.svg
:alt: The same small wind farm solved as a branched, a radial and a ringed network
:class: only-light
:width: 100%
```

```{image} /_static/fig_topologies_dark.svg
:alt: The same small wind farm solved as a branched, a radial and a ringed network
:class: only-dark
:width: 100%
```

Allowing branches at turbines generally enables the shortest network (total cable length). A radial network prohibits these junctions and may require additional cable. A ringed network requires more cable to preserve a path from every turbine to a substation after the failure of any single ring link.

Not all routers support every topology; see [](/routers.md#which-constraints-each-approach-can-enforce) for the capability matrix.

_In use:_ {doc}`/notebooks/hi30_topologies` (Network/Router API) · {doc}`/notebooks/lo30_topologies` (Advanced API).

### Ring semantics

Capacity accounting for rings differs from that of branched and radial topologies:

- a ring serving two or more turbines is split into two arms serving equal numbers of turbines, or differing by one;
- cable capacity limits the load on each feeder of the _split_ ring; with equal turbine power, the two arms together can serve up to twice the turbine count allowed on one feeder;
- each ring uses two physical connections at the substation, so the requested feeder count must be even for a ringed topology;
- a ring serving two or more turbines provides an alternative path to a substation after a single cable failure. The model checks capacity in the normal split configuration; it does not guarantee that the surviving arm can carry the entire ring's output after a fault;
- the single-turbine case uses just one feeder, with no redundant path.

With multiple substations, a MILP model may connect the two ends of a ring to different substations. To require both ends to connect to the same substation, assign the turbines to one cluster per substation and solve the clusters separately.

_In use:_ {doc}`/notebooks/hi30_topologies` (Network/Router API) · {doc}`/notebooks/lo30_topologies` (Advanced API) · {doc}`/notebooks/lo32_clustering` (multiple substations).

## Problem options

Beyond the architecture, three choices define what counts as an acceptable network:

- **Feeder routing:** allow detours made of multiple straight segments, or require straight feeders. Even with straight feeders, exclusion zones may introduce bends.
- **Feeder count:** leave the count unrestricted, bound it, or pin it to a particular value. Bounds can be absolute or relative to the minimum count required by the other constraints. For rings, the allowed range must include an even count because each ring uses two substation connections.
- **Balancing:** require turbine counts per subtree to differ by at most one. This assumes equal turbine power and requires a pinned feeder count; it does not mean balancing unequal power outputs.

These choices define the problem independently of the optimization method, but methods differ in which choices they can enforce. See [](/routers.md#which-constraints-each-approach-can-enforce).

## Objective and reported cost

A low-cost network is often the practical goal. The built-in routers seek a short feasible network: their optimization objective is cable length, which often correlates strongly with total cable cost. When cable types have different prices per unit of length, the shortest feasible network need not be the cheapest.

The routers select the electrical connections using the lengths of available links and feeders. Path-finding then adds any necessary detours to produce the physical cable routes, so their total cable length can exceed the objective value used by a solver. If priced cable types are supplied, _OptiWindNet_ assigns a type to each routed link according to its load and reports the sum of routed length times that type's price per unit of length. This is the **cost of the resulting network**, not a claim that its cost is minimal. See [](/reference/power.md#cable-specifications) for cable inputs and [](/routers.md#exact-optimization) for the meaning of a MILP optimality gap.

_In use:_ {doc}`/notebooks/hi00_quickstart` (Network/Router API) · {doc}`/notebooks/lo00_quickstart` (Advanced API).

## Crossings, contours and detours

Selecting electrical connections is only part of the design: the cables also need feasible physical routes.

- **routes must remain inside the allowed area.** Where a straight link would leave the border or intersect an obstacle, the route follows the relevant boundary and forms a _contour_. A navigation mesh supports this routing.
- **routes must not cross.** A router may return a topology whose straight-line representation contains crossings. The path-finding step adds _detours_ until the crossings are eliminated. Although detours increase the length of the routeset relative to the solver's solution, allowing them can produce shorter routesets than restricting all links to straight routes.

The following figure illustrates both constraints on a 50-turbine site with six exclusion zones. The first panel shows straight feeders; the second shows the routes after path-finding.

```{image} /_static/fig_crossings_light.svg
:alt: Straight feeders crossing obstacles and turbine-turbine links, and the same solution after path-finding
:class: only-light
:width: 100%
```

```{image} /_static/fig_crossings_dark.svg
:alt: Straight feeders crossing obstacles and turbine-turbine links, and the same solution after path-finding
:class: only-dark
:width: 100%
```

Contours appear in both panels because selected links follow boundaries as soon as the topology is converted into a physical graph. The feeders, shown as dashed lines, differ between the panels. In the first panel, they extend directly from the substation and intersect exclusion zones and other routes. In the second, they follow the navigation mesh; ringed markers identify the vertices of a detour added to eliminate a crossing.

Because a detour changes the length of a solution, the total length of a routed network is generally greater than that of the selected connections from which it was derived. Plot conventions for contours and detours are described in [](/problem.md#graph-representations).

_In use:_ {doc}`/notebooks/hi13_border_obstacles` (Network/Router API) · {doc}`/notebooks/lo40_example_taylor_2023` (Advanced API).

## Graph representations

Both APIs represent the same stages using NetworkX graphs. These are useful views of the design process, not steps that every user must carry out manually:

| Graph | View | What it shows |
| --- | --- | --- |
| `L` | Location | Turbines, substations, border and obstacles: the input geometry. |
| `P` | Navigation mesh | Geometry used to find routes around boundaries, obstacles and cables. |
| `A` | Available links | Candidate connections between turbines. Possible substation feeders are also considered by the optimizer, but are omitted from this plot. |
| `S` | Solution topology | The selected electrical connections. Its plot includes boundary-following contours but not crossing-avoidance detours. |
| `G` | Routeset | The physical cable routes, including detours and, when assigned, cable types. |

The progression is `L` → (`P`, `A`) → `S` → `G`. In the following 24-turbine example, the final routing step adds one detour. Cable capacity is five turbines.

```{image} /_static/fig_graph_model_light.svg
:alt: One wind farm site drawn as the graphs L, A, S and G
:class: only-light
:width: 100%
```

```{image} /_static/fig_graph_model_dark.svg
:alt: One wind farm site drawn as the graphs L, A, S and G
:class: only-dark
:width: 100%
```

In the routeset view, detours are dashed, turbines in the same subtree share a color, and thicker lines indicate higher-capacity cable types when assigned.

_Topology_ can mean either the selected connection graph `S` or the network architecture (branched, radial or ringed). The phrase "topology `S`" refers to the graph.

_In use:_ {doc}`/notebooks/hi10_windfarmnetwork` and {doc}`/notebooks/hi14_plotting` (Network/Router API) · {doc}`/notebooks/lo30_topologies` and {doc}`/notebooks/lo14_plotting` (Advanced API).

## Further detail

The branched problem is related to the capacitated minimum spanning tree problem (CMSTP); ringed networks are related to the capacitated vehicle routing problem (CVRP), and radial networks to its open-route variant. _OptiWindNet_ extends these formulations with crossing-free cable routing.

See {doc}`/reference/milp_formulation` for the mathematical model, {doc}`/reference/validation` for checking a solution and {doc}`/paper` for the methodology. Continue to {doc}`/routers` to compare optimization approaches, or return to {doc}`/apis` to choose an interface.
