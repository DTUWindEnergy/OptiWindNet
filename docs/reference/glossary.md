# Glossary

The vocabulary used throughout this documentation. The network topologies — _branched_, _radial_ and _ringed_ — are defined separately, in [](/problem.md#network-topologies).

```{glossary}
terminal
  A wind turbine. Terminals are numbered from `0` in the order they appear in the input data.

root
  A substation. Roots carry negative node numbers, also assigned in order of appearance.

subtree
  For *branched* or *radial* topology: the group of terminals served through a single connection to a root. For *ringed* topology: the group of terminals in a ring.

feeder
  The link that connects a root to a group of terminals.

capacity
  The maximum number of terminals a feeder may serve, derived from the current-carrying capacity of the available cable types.

load
  The number of terminals exporting power through a given node or link, including itself for a terminal. A link's load determines which cable type is assigned to it.

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
