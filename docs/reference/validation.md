# Validation

A solution can be invalid in two independent ways: it can be electrically infeasible (capacity exceeded, a terminal not connected to any root, a degree limit broken), or it can be geometrically invalid (two routes crossing). _OptiWindNet_ currently lacks a check for route×boundaries crossings.

_OptiWindNet_ releases are expected to produce valid solution topologies and routesets, hence these functions are intended primarily for developers attempting to modify the solvers or the path-finder, or to code their own implementations.

End-users are advised to validate a solution ({py:func}`find_geometric_crossings(G) <optiwindnet.crossings.find_geometric_crossings>`) only for problem instances with capacities 2 and 3 and more than 40 terminals, for which {py:class}`PathFinder <optiwindnet.pathfinding.PathFinder>` (with the default parameters) may produce routesets with feeder crossings. These require increasing some internal limits of {py:class}`PathFinder <optiwindnet.pathfinding.PathFinder>` via its arguments.

Please report any invalid solution topology or invalid routesets that are not resolved by the tuning of {py:class}`PathFinder <optiwindnet.pathfinding.PathFinder>`'s parameters.

## Electrical feasibility

<!-- prettier-ignore-start -->

{py:func}`validate_topology(S) <optiwindnet.interarraylib.validate_topology>`
: Checks that a solution topology `S` adheres to the network topology (architecture) and to the cable capacity it declares. Ensures that all terminals are connected to a root and that the required edge attributes are set.

<!-- prettier-ignore-end -->

## Geometric feasibility

Three routines detect crossings, differing in what they can accept as input and in how much they catch. {py:func}`find_geometric_crossings(G) <optiwindnet.crossings.find_geometric_crossings>` is the most robust option and also the most resource-intensive. The right tool to validate {py:class}`PathFinder <optiwindnet.pathfinding.PathFinder>`'s outputs.

{py:func}`find_routeset_crossings(G) <optiwindnet.crossings.find_routeset_crossings>` is a faster, segment-level diagnostic. It may miss crossings that involve routes with overlapping segments and is not used by the full routeset validator.

{py:func}`list_edge_crossings(S, A) <optiwindnet.crossings.list_edge_crossings>` can only identify crossings if S is limited to using only the links available in A. Very low resource use as no geometric calculations are performed. Its typical application is to check if the solvers correctly implemented non-crossings constraints.

_In use:_ {doc}`/notebooks/lo30_topologies` (Advanced API).

## Full routeset validation

<!-- prettier-ignore-start -->

{py:func}`validate_routeset(G) <optiwindnet.interarraylib.validate_routeset>`
: Checks the complete routed solution. It verifies stored loads against the routes, reduces the routes to their solution topology and calls {py:func}`validate_topology() <optiwindnet.interarraylib.validate_topology>`, then checks the complete route polylines with {py:func}`find_geometric_crossings() <optiwindnet.crossings.find_geometric_crossings>`. Calling `validate_topology()` separately is redundant.

<!-- prettier-ignore-end -->
