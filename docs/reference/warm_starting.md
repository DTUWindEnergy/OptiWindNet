# Warm-starting

A warm start supplies an existing solution to the MILP solver used for [](/routers.md#exact-optimization). This page describes which solutions are accepted, how to choose a fast method to construct one, and how requirements interact. For the basic idea and benefits, see [](/routers.md#warm-starting).

For example, an HGS solution with radial paths can start a branched optimization problem: paths are allowed when branching is optional. A solution containing branches cannot start a radial problem, which forbids them. The other constraints must match too.

## Warm-start acceptance requirements

A warm start must already be a valid solution to the problem being solved. Check three things:

1. **Capacity:** every connection carries a load within the requested capacity.
2. **Available connections:** every selected connection is allowed in the receiving problem. Using the same candidate-link graph for both searches ensures this.
3. **Network requirements:** the solution satisfies the requested architecture, feeder routes, feeder count and balancing.

The table helps with the third check. Read all rows that apply to your problem together. It assumes equal turbine power; for unequal powers, also check [](/reference/power.md#router-support).

| Requested network | Acceptable initial solution | Ways to construct a candidate quickly |
| --- | --- | --- |
| Branched | Branched or radial. | A constructive heuristic, or a meta-heuristic producing radial paths. |
| Radial | Paths with no branching at turbines. | A radial constructive heuristic or radial meta-heuristic. |
| Ringed | Rings. | A ringed constructive heuristic or ringed meta-heuristic. |
| Feeders may detour | No additional straight-feeder restriction. | Any approach. |
| Straight feeders | No selected link blocks a feeder route. | A constructive heuristic configured to keep feeder routes clear. |
| Unrestricted feeder count | Any otherwise feasible count. | Any approach. |
| Feeder count within bounds | Count lies within the allowed range. | HGS with the permitted upper bound; also check the resulting count against the lower bound. |
| Exact feeder count | Count equals the requested value. | HGS with an exact count and balancing, for a single substation. |
| Balanced feeder loads | Turbine counts per feeder differ by at most one. | HGS with balancing and a fixed feeder count. |

These are ways to produce candidates, not guarantees of acceptance. A candidate must pass all three checks above.

## Combining feeder requirements

A constructive heuristic can keep feeder routes clear, but cannot enforce a feeder count. Its solution may still have an acceptable count; check the result before using it as a warm start.

HGS can enforce feeder counts and balancing, but its crossing repair only checks links between turbines. A link may therefore still block a feeder route. If feeder detours are allowed, routing can go around that link. If straight feeders are required, the candidate is rejected. This explains why combining straight feeders with a constrained count is difficult: neither fast approach guarantees both requirements together.

Two additional rules apply:

- **Balancing requires a fixed count:** use the minimum feasible feeder count or an explicitly requested exact count.
- **Rings use two substation connections:** requested feeder counts must be even. For an exact count, HGS additionally allows at most one ring per two turbines.

## Using a warm start

How the initial solution is supplied or constructed depends on the API. The Network/Router API can reuse or construct it automatically; the Advanced API lets the caller supply it explicitly. The guides explain the settings and what happens when a candidate is rejected.

_In use:_ {doc}`/notebooks/hi31_options` and {doc}`/notebooks/hi40_example_taylor_2023` (Network/Router API) · {doc}`/notebooks/lo23_milp_ortools` and {doc}`/notebooks/lo40_example_taylor_2023` (Advanced API).
