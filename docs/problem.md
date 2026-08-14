(problem)=
# The Problem

This section defines the optimization problem *OptiWindNet* solves and the vocabulary used throughout this documentation.
It is independent of the API you choose: the concepts below apply equally to the {doc}`high_level_api` and to the {doc}`low_level_api`.

(problem-formulation)=
## Formulation

The design of the collection system of an offshore wind farm can be formulated as a graph problem.
The vertices of the problem represent positions of the power plant's components and are assumed given.
The solution for the problem is a set of edges that represent electrical connections between the components, along with the route along which to lay the electrical cables.

For the solution to be useful, it must meet the constraints that ensure sound electrical circuits and the feasibility of the network:

* circuits can only branch inside a wind turbine, if at all;
* cables cannot cross each other;
* cable routes must fall inside the allowed area, avoiding obstacles within it;
* the maximum current capacity of the cable must be respected.

This problem has similarities with two classic operations research problems:

* the capacitated minimum spanning tree problem (CMSTP);
* the open and capacitated vehicle routing problem (OCVRP).

Neither of the classic formulations considers route crossings, which is the main achievement of *OptiWindNet*.
Whether the approach is via the CMSTP or via the OCVRP depends on the viability of branching the circuits on turbines.
*OptiWindNet* can produce both branched and radial (non-branching) networks — see {ref}`problem-topologies`.

A full analysis of the methodology is available in the open-access article referenced in {doc}`paper`.

(problem-vocabulary)=
## Vocabulary

```{glossary}
terminal
  A wind turbine. Terminals are numbered from `0` in the order they appear in the input data.

root
  A substation. Roots carry negative node numbers, also assigned in order of appearance.

subtree
  The group of terminals served through a single connection to a root.

feeder
  The link that connects a subtree to a root. A *branched* or *radial* subtree has one
  feeder; a *ringed* subtree is a cycle with two.

capacity
  The maximum number of terminals a feeder may serve, derived from the current-carrying
  capacity of the available cable types.

load
  The number of terminals exporting power through a given node or link, including itself
  for a terminal. A link's load determines which cable type is assigned to it.

link
  An electrical connection between two nodes, considered without regard to how the cable
  physically gets there.

route
  The geographical path a cable follows to implement a link. A route may bend around
  boundaries (a *contour*) or deviate to avoid another cable (a *detour*), so its length
  is at least the straight-line distance of the link it implements.

contour
  The part of a route that follows the border or an obstacle boundary, because a straight
  run would leave the allowed area.

detour
  A deviation added to a route so that it no longer crosses another route. Detours are
  drawn with dashed lines in the plots.

crossing
  Two cable routes intersecting at a point that is not a shared node. Crossings are
  forbidden in a valid solution.
```

(problem-graph-model)=
## The graph model

*OptiWindNet* represents a problem instance and its solution as a series of *networkx* graphs.
Both APIs build the same graphs; the high-level API keeps them as attributes of a `WindFarmNetwork` instance, while the low-level API passes them between functions explicitly.

| Graph | Name | What it holds |
| --- | --- | --- |
| `L` | location | The problem instance geometry: terminal and root coordinates, the border and the obstacles. No links. |
| `P` | planar embedding | The *navigation mesh*: a constrained Delaunay triangulation of every vertex of the location, used to find routes around boundaries. |
| `A` | available links | The search space: the links the optimizer is allowed to choose from, derived from the Delaunay triangulation of terminals and roots plus diagonal links. |
| `S` | topology | The solution as electrical parent–child relations only — which node connects to which, without geometry. |
| `G` | routeset | The solution as physical cable routes, including contours and detours, with cable types assigned. |

The usual progression is `L` → (`P`, `A`) → `S` → `G`.
A topology `S` becomes a tentative physical graph through `G_from_S(S, A)`, and that graph becomes a routeset once the feeders have been routed so as to avoid crossings.

```{admonition} Two meanings of "topology"
:class: note

*Topology* names both the solution graph `S` and the architecture of its subtrees
(*branched*, *radial* or *ringed*). Where the distinction matters, this documentation
writes "the topology `S`" for the former.
```

*In use:* {doc}`notebooks/hi10_windfarmnetwork` (Network/Router API) · {doc}`notebooks/lo29_topologies` (Advanced API).

(problem-topologies)=
## Network topologies

*OptiWindNet* supports three electrical topologies.
They constrain the shape of the solution and are requested through the `topology` option described in {ref}`methods-model-options`.

Branched
: A forest of rooted trees; terminals may have any number of neighbors.
  This is the default and the least constrained topology, so it admits the shortest networks.

Radial
: A collection of root-to-leaf paths; each terminal has at most two neighbors.
  No cable branches at a turbine, which simplifies the switchgear at the cost of some cable length.

Ringed
: A collection of cyclic paths (multi-terminal *loops*), each including one or two roots
  (roots are implicitly neighboring each other). Only roots may belong to more than one
  path. Each terminal has exactly two neighbors, and single-terminal cycles degenerate to
  a single link.

Not every method can produce every topology; see {ref}`methods-overview` for the capability matrix.

(problem-rings)=
### Ring semantics

Rings deserve a closer look, because their capacity accounting differs from the other two topologies:

* each cycle contains one link with `load = 0`, which splits the ring at its mid-point;
* `capacity` is the feeder limit of the *split* ring, so a full ring can hold up to `2 × capacity` terminals;
* each ring uses two physical connections at the substation, which is why a `max_feeders` value for a ringed topology must be a multiple of 2;
* if a fault occurs anywhere along a ring, every terminal on it still retains an intact path to a substation. This redundancy is the reason to pay for the extra cable.

*In use:* {doc}`notebooks/hi29_topologies` (Network/Router API) · {doc}`notebooks/lo29_topologies` (Advanced API).

(problem-crossings)=
## Crossings, contours and detours

Two constraints are geometric rather than electrical, and they are what distinguishes
*OptiWindNet* from a textbook CMSTP or OCVRP solver:

* **routes must stay inside the allowed area.** Where a straight link would leave the
  border or pass through an obstacle, the route follows the boundary instead, producing a
  *contour*. The navigation mesh `P` is what makes this search possible.
* **routes must not cross each other.** A solution method may return a topology whose
  straight-line drawing has crossings; the routing step then adds *detours* until none
  remain. Detours make the network longer, so a method that avoids creating them in the
  first place tends to produce shorter networks.

Because a detour changes the length of the solution, the total length of a routeset `G`
is generally greater than the total length of the links in the topology `S` it came from.
Comparisons between methods are only meaningful on routesets.

*In use:* {doc}`notebooks/hi31_border_obstacles` (Network/Router API) · {doc}`notebooks/lo40_example_taylor_2023` (Advanced API).

(problem-validation)=
## Validating a solution

Two checks are available wherever a solution is produced:

* `validate_topology()` — checks that a topology `S` is feasible from the electrical
  perspective (capacity, degree limits, connectivity to a root).
* `validate_routeset()` — checks that a routeset `G` is correct, calling
  `validate_topology()` internally.

`describe_G()` gives a compact diagnostic summary that is useful next to a plot.
When working with the {doc}`low_level_api`, validate each representation at the boundary
where it is created; the {doc}`high_level_api` applies these checks for you.
