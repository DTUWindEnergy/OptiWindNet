# Validation

A solution can be invalid in two independent ways: it can be electrically infeasible (a cable exceeds capacity, a turbine has no connection to a substation, or a node has too many neighbors for the chosen topology), or it can be geometrically invalid (two routes crossing). _OptiWindNet_ currently lacks a check for route×boundaries crossings.

Use `validate_routeset(wfn.G)` to check a delivered network, especially when using custom graph transformations, hand-built topologies or path-finder settings. The function returns a list of violations, empty when the checks pass, and leaves the graph unchanged:

```python
from optiwindnet.validating import validate_routeset

violations = validate_routeset(wfn.G)
if violations:
    raise ValueError('\n'.join(violations))
```

On sites with more than 40 turbines and capacities of 2 or 3, the default `PathFinder` search limits may leave feeder crossings. Its arguments allow those limits to be increased. Report invalid results that persist after tuning, including the site, capacity and router settings.

These checks cover the graph's declared topology and integer capacity, stored loads and route intersections. They do not check every requested model option, such as an exact feeder count, or replace a check of route containment within the site boundaries.

## Electrical feasibility

<!-- prettier-ignore-start -->

{py:func}`validate_topology(S) <optiwindnet.validating.validate_topology>`
: Checks that a solution topology `S` adheres to the network topology (architecture) and to the cable capacity it declares. Ensures that all terminals are connected to a root and that the required edge attributes are set.

<!-- prettier-ignore-end -->

## Geometric feasibility

Three routines detect crossings, differing in what they can accept as input and in how much they catch. {py:func}`find_geometric_crossings(G) <optiwindnet.crossings.find_geometric_crossings>` is the most robust option and also the most resource-intensive. The right tool to validate {py:class}`PathFinder <optiwindnet.pathfinding.PathFinder>`'s outputs.

{py:func}`find_routeset_crossings(G) <optiwindnet.crossings.find_routeset_crossings>` is a faster, segment-level diagnostic. It may miss crossings that involve routes with overlapping segments and is not used by the full routeset validator.

{py:func}`list_edge_crossings(S, A) <optiwindnet.crossings.list_edge_crossings>` can only identify crossings if S is limited to using only the links available in A. Very low resource use as no geometric calculations are performed. Its typical application is to check if the solvers correctly implemented non-crossings constraints.

_In use:_ {doc}`/notebooks/lo30_topologies` (Advanced API).

## Full routeset validation

<!-- prettier-ignore-start -->

{py:func}`validate_routeset(G) <optiwindnet.validating.validate_routeset>`
: Checks the complete routed solution. It verifies stored loads against the routes, reduces the routes to their solution topology and calls {py:func}`validate_topology() <optiwindnet.validating.validate_topology>`, then checks route crossings, self-intersections, invalid overlaps, branch splits and degenerate geometry with {py:func}`find_geometric_crossings() <optiwindnet.crossings.find_geometric_crossings>`. Calling `validate_topology()` separately is redundant.

<!-- prettier-ignore-end -->
