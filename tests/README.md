# Golden test data

Run regeneration commands from the repository root.

## Node-set digest map

Regenerate `tests/nodeset_digest-location-map.pkl` after bundled location files
or single-root conversion change:

```bash
python -m tests.update_nodeset_digest_location_map
```

## Deterministic solver topologies

Regenerate `tests/solver_topologies.pkl` from the exact constructor and required
MILP cases:

```bash
python -m tests.update_solver_topologies
```

## PathFinder routesets

Regenerate `tests/pathfinder_golden.pkl` from the curated route-set IDs:

```bash
python -m tests.update_pathfinder_golden \
  docs/notebooks/optiwindnet-routesets-r26.05-v4.sqlite
```

## MILP references

MILP regeneration has two stages. First solve the candidate matrix with CPLEX;
only proven optima are retained in the gitignored provisional JSON:

```bash
python -m tests.update_milp_reference_candidates \
  --matrix --solver-name cplex --time-limit 30 --mip-gap 1e-8
```

Then validate and deploy the reviewed bound, objective, and topology records to
`tests/milp_references.pkl` without rerunning a solver:

```bash
python -m tests.update_milp_references
```

Review golden changes before committing them; a failing regression test alone
is not sufficient reason to replace an expected value.
