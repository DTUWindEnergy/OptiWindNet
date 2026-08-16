# Dataset Paper

A second scientific article introduces **OptiWindNet RouteSets** — a database of cable-routing solutions produced with _OptiWindNet_ and released as an open benchmark for the wind farm cable-routing problem. The article is under review at _Wind Energy Science_; the preprint is open access:

- Mauricio Souza de Alencar, Tuhfe Göçmen, Nicolaos A. Cutululis, _OptiWindNet RouteSets: a solver-diverse benchmark dataset for the offshore wind-farm cable routing problem_, Wind Energy Science Discussions [preprint], 2026, <https://doi.org/10.5194/wes-2026-124>, in review.

```{code-block} bib
@Article{wes-2026-124,
  author = {Souza de Alencar, M. and G\"o\c{c}men, T. and Cutululis, N. A.},
  title = {OptiWindNet RouteSets: a solver-diverse benchmark dataset for the offshore wind-farm cable routing problem},
  journal = {Wind Energy Science Discussions},
  volume = {2026},
  year = {2026},
  pages = {1--13},
  url = {https://wes.copernicus.org/preprints/wes-2026-124/},
  doi = {10.5194/wes-2026-124},
}
```

The framework that produced the solutions is the subject of the {doc}`/paper`.

## Getting the database

Release 26.05-v4 is a single SQLite file, `optiwindnet-routesets-r26.05-v4.sqlite`, deposited on Zenodo under CC-BY-4.0:

> Souza de Alencar, M. (2026). _OptiWindNet RouteSets_ (26.05-v4) [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.20053479>

The download is `optiwindnet-routesets-r26.05-v4.sqlite.xz` (~28 MB compressed, ~88 MB decompressed). Nothing else is needed to read it — the `optiwindnet` package you already have provides the models.

## What is in it

49 204 route sets over 12 814 distinct wind farm geometries, covering 13 954 distinct **problem instances**, where an instance is a (location, cable capacity) pair. Instances span 50–200 turbines and cable capacities of 2–12 turbines, in radial and branched topology. A solution with a proven optimality gap below 1 % is available for 63 % of the instances, and below 2 % for 79 %.

Several route sets may solve the same instance — that is deliberate, since the informative comparisons are paired ones in which a single method attribute differs.

The four solvers behind the route sets:

| `creator` | backend | topology | route sets | % |
| --- | --- | --- | --: | --: |
| `MILP.pyomo.cplex` | CPLEX 22.1.1, via _pyomo_ | radial, branched | 22 623 | 46 |
| `baselines.hgs` | HGS-CVRP, via _hybgensea_ | radial | 13 857 | 28 |
| `baselines.lkh` | LKH-3.0.14 binary | radial | 12 166 | 25 |
| `MILP.pyomo.gurobi` | Gurobi 12.0, via _pyomo_ | radial, branched | 558 | 1 |

The locations have a core of 98 built offshore wind farms — most of the existing ones in the covered turbine-count range — plus five hypothetical sites proposed by [Cazzaro and Pisinger (2022)](https://doi.org/10.1002/net.22100), [Yi et al. (2019)](https://doi.org/10.1049/iet-rpg.2018.5805) and [Taylor et al. (2023)](https://doi.org/10.1049/rpg2.12593). The remaining geometries were derived from those by the data augmentation in `optiwindnet.augmentation`: borders and substations are kept, turbines are re-scattered over the available area. Augmented names carry their donor, as `!donor_name!digest`, with suffixes recording transformations (`.1_OSS` and `.1st_OSS` reduce to a single substation, `.solid` removes obstacles).

```{admonition} Site families matter for machine learning
:class: important

Variants sharing a donor are geometrically related, so a random split across geometries leaks between train and test. Split by site family instead — it separates within-family sensitivity to turbine placement from between-family generalization.
```

### The four tables

<!-- prettier-ignore-start -->

`nodeset`
: One row per distinct wind farm geometry, keyed by the SHA digest of that geometry, so identical geometries loaded from different files collapse into one row. All coordinates live in the `VertexC` binary blob as planar easting/northing (the UTM zone is not stored, so the geodesic position cannot be recovered from the file alone); the other fields say which coordinate plays which role — turbine count `T`, substation count `R`, and the `B` coordinates that only define borders and exclusion-zone polygons.

`routeset`
: One row per solution. Solutions refer to coordinates by index into their `nodeset` row and carry no coordinates of their own. Each row references one `nodeset`, one `method` and one `machine`.

`method`
: One row per distinct (`solver_name`, `options`, `funhash`) triple, describing the solver call.

`machine`
: The computer that produced the solution.

<!-- prettier-ignore-end -->

```{admonition} Do not query methods by primary key
:class: warning

`funhash` hashes the solver function's bytecode, so it drifts across _OptiWindNet_ versions: the same solver call may be recorded under several `method` rows. Match on `solver_name` and the `options` (key, value) pairs instead of on `digest`.
```

### Objective, length and gap

The optimizer minimizes a weighted connectivity problem defined over the available-links graph, which abstracts away the geometric detail. The best weight it found is `misc['objective']`, and, for MILP runs, the best dual bound is `misc['bound']` — together they give the proven relative gap `misc['relgap']`.

The route set itself is the embedded network: the cable routes in the plane that implement that connectivity, after feeder detours are added. Its total length is the `length` field, which may exceed the objective; `detextra` is the relative increase between the two, so `length = (1 + detextra) * objective`. The pre-detour graph is recoverable from the route set and is therefore not stored.

## Reading the database

The tables are [_peewee_](https://docs.peewee-orm.com/) models, so the database is queried through {py:class}`RouteSet <optiwindnet.db.RouteSet>` and its siblings, and a row is turned back into a routed graph by {py:func}`G_from_routeset() <optiwindnet.db.G_from_routeset>`:

```{code-block} python
from optiwindnet.db import RouteSet, G_from_routeset, database_connection
from optiwindnet.plotting import gplot

with database_connection('optiwindnet-routesets-r26.05-v4.sqlite'):
    routeset = (
        RouteSet.select()
        .where((RouteSet.handle == 'dudgeon') & (RouteSet.capacity == 5))
        .order_by(RouteSet.length)
        .first()
    )
    print(routeset.creator, routeset.T, routeset.length, routeset.misc['relgap'])
    G = G_from_routeset(routeset)

gplot(G)
```

A location alone, without a solution, comes from {py:func}`L_from_nodeset() <optiwindnet.db.L_from_nodeset>`. Both functions rebuild the graph metadata the rest of the library expects — see [](/problem.md#the-graph-model) — which a raw SQL query would leave behind.

## What the article reports

Four analyses over the release, each on paired instances:

- **Meta-heuristic solution quality.** Against the shortest MILP-radial route set of the same instance, and with 120 s of single-core time, HGS-CVRP has a median length increase of 0.0 % (90th percentile 0.225 %) and LKH-3 of 0.555 % (90th percentile 2.187 %).
- **Problem difficulty.** Read through the MILP gap and runtime, difficulty rises monotonically with turbine count, while its dependence on capacity peaks at intermediate values and depends on the count. Median runtime goes from seconds to the 6 h cap between 70 and 110 turbines. Branched instances are somewhat harder than radial ones.
- **Branched vs. radial topology.** Over 8 696 matched instances, branching shortens the network by a median 0.34 % (mean 0.64 %), and is shorter in 6 741 of them. The advantage grows with cable capacity and shrinks with turbine count, with enough spread to warrant deciding case by case.
- **Detour impact.** The length added by feeder detours over the solver-optimized objective has a median of 0.171 % for radial and 0.154 % for branched, largest at capacities 3–6.

## Reproducing the analyses

The four notebooks that produce the article's figures, together with the SQL access layer they use, are archived separately:

> Souza de Alencar, M. (2026). _Code and Computational Artifacts for the PhD Thesis "Wind Farm Collection System Optimization for Integrated Design"_. Zenodo. <https://doi.org/10.5281/zenodo.20140812>

Download `optiwindnet_routesets.zip` from that record. It holds `case1_metaheuristic_solution_quality.ipynb`, `case2_problem_difficulty.ipynb`, `case3_branched_vs_radial_topology.ipynb` and `case4_detour_impact.ipynb` — one per analysis above, independent of each other — plus `db_access.py`, which fetches and verifies the database on first run and pushes the projection, filtering and best-per-instance selection down into SQL. The archive also ships the figures, so it can be read without running anything.

## How to cite

Cite the article for the analyses and the Zenodo record for the data:

> Souza de Alencar, M., Göçmen, T., and Cutululis, N. A.: OptiWindNet RouteSets: a solver-diverse benchmark dataset for the offshore wind-farm cable routing problem, Wind Energ. Sci. Discuss. [preprint], https://doi.org/10.5194/wes-2026-124, in review, 2026.

> Souza de Alencar, M. (2026). _OptiWindNet RouteSets_ (26.05-v4) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.20053479

The software itself is cited as described in [](/index.md#how-to-cite).
