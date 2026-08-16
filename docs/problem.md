# Problem

This page defines the optimization problem that _OptiWindNet_ solves. The definition is independent of the API: the concepts apply equally to the {doc}`/high_level_api` and the {doc}`/low_level_api`. The {doc}`/reference/glossary` collects the terms used throughout the documentation.

## Formulation

The design of an offshore wind farm collection system can be formulated as a graph problem. Its vertices represent the predefined positions of the wind farm components. A solution consists of edges that specify electrical connections between these components and routes that specify where to lay the corresponding cables.

To be feasible, a solution must satisfy the following electrical and geometric constraints:

- branching, when permitted, can occur only inside a wind turbine;
- cables cannot cross;
- cable routes must remain inside the allowed area and avoid obstacles;
- the current through each cable must not exceed its capacity.

This problem is related to two classical operations research problems:

- the capacitated minimum spanning tree problem (CMSTP);
- the capacitated vehicle routing problem (CVRP).

Neither classical formulation accounts for route crossings. _OptiWindNet_ extends these formulations by incorporating crossing-free cable routing. The selected [](/problem.md#network-topologies) determines the underlying formulation: CMSTP for a _branched_ topology, CVRP for a _ringed_ topology, and the open-route variant of CVRP for a _radial_ topology.

A detailed analysis of the methodology is available in the open-access article referenced in {doc}`/paper`.

_In use:_ {doc}`/notebooks/hi00_quickstart` (Network/Router API) · {doc}`/notebooks/lo00_quickstart` (Advanced API).

## The graph model

_OptiWindNet_ represents a problem instance and its solution as a sequence of _networkx_ graphs. Both APIs build the same graphs and provide corresponding plot views. The high-level API stores the graphs as attributes of a {py:class}`WindFarmNetwork <optiwindnet.api.WindFarmNetwork>` instance, whereas the low-level API passes them explicitly between functions. The location and routeset views present the input and final result; the other three expose intermediate graphs for inspection and diagnostics.

| Graph | Name | Contents | Plot representation |
| --- | --- | --- | --- |
| `L` | location | The problem geometry: terminal and root coordinates, the border and obstacles. No links. | Terminals as circles, roots as squares, and the border and obstacles. This view can be used to validate the input before optimization. |
| `P` | planar embedding | The _navigation mesh_: a constrained Delaunay triangulation of every vertex of the location, used to find routes around borders, obstacles and cables. | The complete triangulation, including the supertriangle that contains the location. |
| `A` | available links | The search space. Its edges are non-feeder links derived from the Delaunay triangulation of terminals and roots, together with diagonal links. Each possible root-to-terminal feeder is also available, but `A` stores its obstacle-avoiding length in the `d2roots` attribute rather than representing the feeder as an edge. | Direct Delaunay links as solid lines and diagonals as dashed lines; color distinguishes obstructed links. Feeders are omitted for clarity. |
| `S` | solution topology | The electrical parent–child relations of the solution: node-to-node connectivity without physical routing geometry. | A physical rendering of the selected links and feeders. Links that must avoid the border or obstacles follow their contours; all others remain straight. Detours are not yet applied. |
| `G` | routeset | The solution as physical cable routes, including contours and detours, optionally with cable types assigned. | The actual cable routes, with detours dashed. Terminals in the same subtree share a color; line thickness encodes cable type, with thicker lines indicating higher capacity. |

The usual progression is `L` → (`P`, `A`) → `S` → `G`. The optimizer selects from the links in `A` and from all possible root-to-terminal feeders. Feeders are not edges of `A` and are omitted from its plot because drawing every possible feeder would obscure the non-feeder search space.

The following figure shows four graph representations of Kriegers Flak A, with 24 turbines and `capacity = 5`. `L` contains only the input geometry. `A` shows the available non-feeder links. `S` shows the selected links and feeders, with required contours but without detours. `G` adds one detour to eliminate a crossing and thereby completes the cable routes.

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

This sequence describes the data pipeline rather than a required user workflow. The {doc}`/high_level_api` accepts `L` and returns `G`, while constructing and storing `P`, `A` and `S` internally. Direct access to these intermediate graphs is useful for inspecting the search space, reusing a topology across runs or controlling individual processing steps through the {doc}`/low_level_api`.

{py:func}`G_from_S(S, A) <optiwindnet.interarraylib.G_from_S>` renders the selected-links graph `S` as a tentative physical graph. Links that must avoid the border or obstacles already follow their contours at this stage. Path-finding then resolves crossings by adding detours, producing the routeset `G`. The selected-links and routeset views therefore differ in routing completeness: the former includes contours but no detours, whereas the latter shows the finalized physical cable routes. See {doc}`/reference/validation` for procedures that verify solution validity.

```{admonition} Two meanings of "topology"
:class: note

*Topology* may refer to the solution graph `S` or to its network architecture
(*branched*, *radial* or *ringed*). Where the distinction matters, this documentation uses "the topology `S`" for the former.
```

_In use:_ {doc}`/notebooks/hi10_windfarmnetwork` and {doc}`/notebooks/hi14_plotting` (Network/Router API) · {doc}`/notebooks/lo30_topologies` and {doc}`/notebooks/lo14_plotting` (Advanced API).

## Network topologies

_OptiWindNet_ supports three electrical topologies. Each constrains the structure of the solution and can be selected through the `topology` option described in [](/routers.md#model-options).

```{glossary}
branched
  A forest of rooted trees; terminals may have any number of neighbors. This is the default and least constrained topology, and therefore generally permits the shortest networks.

radial
  A collection of root-to-leaf paths; each terminal has at most two neighbors. Cables do not branch at turbines, which simplifies switchgear but may increase cable length.

ringed
  A collection of cyclic paths (multi-terminal *loops*), each including one or two roots (roots are implicitly neighboring each other). Only roots may belong to more than one path. Each terminal has exactly two neighbors, and single-terminal cycles degenerate to a single link.
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

Allowing branches at turbines generally reduces the cable length of a branched network. A radial network prohibits these junctions and may require additional cable. A ringed network requires more cable to preserve a path from every turbine to a substation after the failure of any single ring segment.

Not all routers support every topology; see [](/routers.md#optimization-approaches) for the capability matrix.

_In use:_ {doc}`/notebooks/hi30_topologies` (Network/Router API) · {doc}`/notebooks/lo30_topologies` (Advanced API).

### Ring semantics

Capacity accounting for rings differs from that of branched and radial topologies:

- each cycle contains one link with `load = 0`, which splits the ring at its midpoint;
- `capacity` is the feeder limit of the _split_ ring, so a complete ring can contain up to `2 × capacity` terminals;
- each ring uses two physical connections at the substation, so `max_feeders` must be an even integer for a ringed topology;
- if a fault occurs anywhere along a ring, every terminal on it retains an intact path to a substation. The additional cable provides this redundancy.

With multiple substations, a MILP model for a ringed topology may produce a ring that begins at one root and ends at another. Partitioning the terminals by root and solving each cluster separately ensures that every ring is anchored to a single root.

_In use:_ {doc}`/notebooks/hi30_topologies` (Network/Router API) · {doc}`/notebooks/lo30_topologies` (Advanced API) · {doc}`/notebooks/lo32_clustering` (multiple substations).

## Crossings, contours and detours

Two geometric constraints distinguish _OptiWindNet_ from a standard CMSTP or CVRP solver:

- **routes must remain inside the allowed area.** Where a straight link would leave the border or intersect an obstacle, the route follows the relevant boundary and forms a _contour_. The navigation mesh `P` supports this routing.
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

Contours appear in both panels because links selected from `A` follow boundaries as soon as the topology is converted into a physical graph. The feeders, shown as dashed lines, differ between the panels. In the first panel, they extend directly from the substation and intersect exclusion zones and other routes. In the second, they follow the navigation mesh; ringed markers identify the vertices of a detour added to eliminate a crossing.

Because a detour changes the length of a solution, the total length of a routeset `G` is generally greater than that of the topology `S` from which it was derived. Plot conventions for contours and detours are described in [](/problem.md#the-graph-model).

_In use:_ {doc}`/notebooks/hi13_border_obstacles` (Network/Router API) · {doc}`/notebooks/lo40_example_taylor_2023` (Advanced API).
