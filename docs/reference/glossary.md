# Glossary

The vocabulary used throughout this documentation. The network topologies — _branched_, _radial_ and _ringed_ — are defined separately, in [](/problem.md#network-topologies).

```{glossary}
turbine
  A wind turbine, represented by a terminal node in the graph model. Network/Router guides and shared explanations normally use the physical name, turbine.

substation
  The collection point for power from a group of turbines, represented by a root node in the graph model.

terminal
  The graph node representing a turbine. This term is common in the Advanced API. Terminal nodes are numbered `0` through `T - 1` in the order they appear in the input data.

root
  The graph node representing a substation. This term is common in the Advanced API. Root nodes are numbered `-R` through `-1` in the order they appear in the input data.

subtree
  For *branched* or *radial* topology: the group of turbines served through a single connection to a substation. For *ringed* topology: the group of turbines in a ring.

feeder
  The link from a substation to the first turbine in a subtree or an arm of a ring.

capacity
  The maximum load a cable can carry. Routers represent it as integer inflow; user inputs can instead declare nominal power with an explicit unit. With unitary inflow, it counts turbines. See [](/reference/input_formats.md#cable-types).

load
  The cumulative inflow carried by a node or link. At a turbine, this includes its own contribution and those of the turbines feeding into it. Load counts turbines only when each contributes one unit. Nominal loads sum the declared turbine powers; cable assignment uses nominal loads when capacities are nominal power.

inflow
  A turbine's positive integer contribution to the solver's flow, defaulting to one. Unequal nominal powers are quantized to integer inflow for the selected cable capacity; see [](/reference/input_formats.md#turbines-of-unequal-output).

link
  An electrical connection between two nodes, considered without regard to how the cable is physically routed.

route
  The geographical path a cable follows to implement a link. A route may bend around boundaries (a *contour*) or deviate to avoid another cable (a *detour*), so its length is at least the straight-line distance of the link it implements.

contour
  The part of a route that follows the border or an obstacle boundary, because a straight run would leave the allowed area.

detour
  A deviation added to a route so that it no longer crosses another route. Detours are drawn with dashed lines in the plots.

crossing
  Two cable routes intersecting at a point that is not a shared node. Crossings are forbidden in a valid routeset.

routeset
  The solution as physical cable routes — the graph `G` of [](/problem.md#graph-representations). It carries the contours and detours that path-finding added, and optionally a cable type per link, so its total length is at least that of the topology `S` it came from.

router
  An algorithm that turns a problem instance into a solution topology. The three optimization approaches — constructive heuristic, meta-heuristic and exact optimization — are described in {doc}`/routers`. The Network/Router API wraps each approach in a {py:class}`Router <optiwindnet.api.Router>` subclass; the Advanced API calls the same algorithms as functions.

solver
  A third-party mixed-integer programming backend — Gurobi, CPLEX, HiGHS, SCIP, CBC, OR-Tools — that an exact router hands its model to. The term is reserved for these backends and is never used for a heuristic or a meta-heuristic. The roster is in {doc}`/reference/solvers`.

method
  The argument that picks an Esau-Williams variant within the constructive-heuristic approach, such as `'biased_EW'`. It names a variant, never an optimization approach; the variants are listed in [](/routers.md#constructive-heuristics).
```
