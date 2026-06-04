# v0.2.2

[Commit history since v0.2.1](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.2.1...v0.2.2)

## Breaking Changes
- **Advanced API Cleanups**: Helper functions for root-assignment and link-blockage moved from `optiwindnet.geometric` to `optiwindnet.interarraylib`. Users should import `add_terminal_closest_root()`, `add_link_blockmap()`, and `add_link_cosines()` from `optiwindnet.interarraylib`.

## Important Changes
- **Default Vector SVG Plotting**: High-level `WindFarmNetwork` plotting methods (`plot()`, `plot_location()`, `plot_available_links()`, `plot_navigation_mesh()`, and `plot_selected_links()`) now use a modern, interactive vector SVG plotting backend (`svgplot`/`svgpplot`) by default. This delivers clean, high-resolution inline displays in Jupyter notebooks. The legacy Matplotlib-based backend remains fully accessible by passing an explicit `ax` argument (including `ax=None` to dynamically instantiate Matplotlib figures).
- **svgplot() matches gplot()'s features**: SVG plots now support node labeling, boundary/obstacle vertex tagging, and figure legend.
- **Informative String Representations**: Added descriptive, debugger-safe string representations (`__repr__`) for `WindFarmNetwork` and `Router` subclasses (`EWRouter`, `HGSRouter`, `MILPRouter`) displaying key configuration parameters and solved network metrics.
- **Shorter Substation Labels**: Pre-packaged offshore wind farm datasets (.osm.pbf format) have been updated with short, human-readable substation abbreviations (such as "Alpha", "Beta", "OSS") to fit cleanly in visualization labels.
- **New Fused Heuristic**: Added `heuristics.constructor()` with `esau_williams`, `biased_EW`, `rootlust`, and `radial_EW` methods, unifying the constructive routing heuristics. The high-level `EWRouter` now uses this path, offering radial topology and the performant rootlust method.
- **LKH-3 Solver Parity**: Added `lkh3()` as the preferred LKH entry point, bringing it to feature parity with the HGS solver. It supports single- and multi-root configurations, per-root clustering, warm starts, capacity-violation retries, crossing repair, and improved solver metadata.
- **Expanded Crossing Diagnostics**: Added Shapely-based `find_geometric_crossings()` for geometry-first validation of arbitrary routesets, including detours, contour clones, shared-run overlap crossings, and branch-split cases.
- **Robust PathFinder Detours**: Major robustness improvements when routing detours among cable routes that follow boundaries or exclusion zones, significantly reducing cable use on sites with many obstacles.

## Deprecated
- Standalone EW heuristics (`ClassicEW`, `CPEW`, `NBEW`, `OBEW`, and `EW_presolver`) are deprecated and will be removed in v0.3. They are superseded by the new unified `heuristics.constructor()`. Note that `constructor` expects the available-links graph `A`, not the location graph `L`.
- The legacy `optiwindnet.interface` module (`heuristic_wrapper()`, `HeuristicFactory`) is deprecated and will be removed in v0.3; use `WindFarmNetwork`/`EWRouter` instead.

## Fixes
- **Pathfinder Robustness**: Resolved fatal crashes (`KeyError` and triangulation flip failures) when constructing detours.
- **Diagonal Mesh Exclusion**: Prevented invalid diagonal paths by skipping edges in the site's boundary polygon during navigation mesh generation.
- **Logging & Diagnostics**: Replaced all remaining raw `print()` statements across the API and utility modules with standard Python logging.
- **LKH and Heuristic Repairs**: Fixed LKH warm-start tour construction (indexing, walk order within clusters) and aligned HGS/LKH repair behavior for capacity-violating and crossing routes.
- **Overflow Prevention**: Added checks for LKH weight-matrix construction with clear guidance when inputs need normalization.
- **Crossing Detection**: Fixed shared-route overlap crossing detection and added geometric handling for route intersections not expressible as available-edge crossings.

## Refactoring & Maintenance
- **Python 3.11–3.14 Support**: Explicitly declared support for Python 3.11 through 3.14 with standard Trove classifiers on PyPI.
- **Updated OR-Tools Floor**: Aligned OR-Tools requirements in `pyproject.toml` to `>=9.14.6206` for consistency across development and production environments.
- **Strict Deprecation Testing**: Test suite configured to treat `DeprecationWarning` as errors to guarantee API health.
- **Linting & Code Quality**: Enforced strict Ruff linting and formatting rules via continuous integration.
- **Performance Optimizations**: Optimized pathfinding sector lookups and precomputed chain-end topologies to speed up execution.

# v0.2.1

[Commit history since v0.2.0](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.2.0...v0.2.1)

## Breaking Changes
- `optiwindnet.db` module only: **RouteSet schema slimmed (v4)**: `num_gates` was renamed to `feeders_per_root`; the unused `valid`, `is_normalized`, and `stuntC` columns were removed. The `python -m optiwindnet.db.migrate` script now writes the v4 schema and accepts both v2 (Pony ORM) and v3 (Peewee) source databases.

## Important Changes
- **OR-Tools MILP backend switched to MathOpt API** (replacing `cp_model`), enabling multiple backends through a unified wrapper. No impact on use through either API. 
- **Pyomo CPLEX/Gurobi solvers switched to the persistent interfaces** (`cplex_persistent`, `gurobi_persistent`). Relevant for successive calls to solver.solve().
- **`PathFinder`**: `A` is now a mandatory argument, it is relied upon to inform about tentative feeder crossings (saves the repeated check done before); default options were updated; search heuristics improved.

## Features
- LKH improvements: iterations are also triggered on capacity violations, new `warmstart` argument.
- Better estimation of obstructed feeder lengths pre-optimization (make_planar_embedding).
- Default thread count for MathOpt solvers set to the number of physical cores; non-OR-Tools solvers also use `physical_core_count()`.
- New context-managed database connection API; `open_database()` and `database_connection()` accept a `timeout` argument.
- `.osm.pbf` parsing now accepts locations without borders.
- Added typing stubs and improved type annotations.
- Informative string repr for `SvgRepr`.

## Fixes
- Multiple PathFinder robustness fixes: collinear vertices in funnel apex update, expansion of `P_paths` shortcuts when building contour clones, shortcut provenance tracking for barriers, cumulative turning check for dropping traversers, and `bad_streak` decay on first arrival.
- `make_planar_embedding` fixes: constraint checks and line-of-sight tagging now use Shapely's `STRtree`, proper handling of diagonal promotion conflicts in concave meshes, string-pulling skipped when only one border vertex is on the path (enabled by STRtree check).
- `validate_routeset()`: corrected detour index range; touchpoint set as bunch-split corner apex.
- LKH: replaced stale `_add_link_blockage` call with `add_link_blockmap`.
- Removed WAL mode from the SQLite open pragma (caused issues on shared clusters).
- Stunt vertices are no longer placed in `G` (regression since a70b575).
- Gracefully handle repeated extents' vertices in .yaml input files.
- Fixed `migrate.py` ImportError.

## Refactoring & Maintenance
- Replaced `dill` with `pickle` everywhere it was used in tests.
- Removed `stuntC` from the routeset saving path.
- CI now runs a test matrix covering Python 3.11–3.14 (default bumped to 3.14); release requires passing on all versions.
- Increased test coverage and added topology-aware routeset comparison to prevent spurious failures.

## Documentation
- Major refactor of the Topfarm integration example (now including substation trajectory); several notebook updates.
- Added links to TOPFARM and Ard, updated preamble with Jupyter tutorial links.
- Acknowledged the DFF grant in README and doc index.
- Improved docstrings and setup instructions.

# v0.2.0

[Commit history since v0.1.6](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.1.6...v0.2.0)

## Breaking Changes
- **HGS-CVRP interface unified**: `baselines.hgs` functions `hgs_multiroot()` and `iterative_hgs_cvrp()` are deprecated; `hgs_cvrp()` replaces them with no loss in functionality. Users of HGSRouter from the Network/Router API will not notice the change.
- **Database format updated to v3**: Switched from Pony ORM (incompatible with Python 3.13+) to Peewee. Use `python -m optiwindnet.db.migrate input.v2.sqlite output.v3.sqlite` to migrate existing databases.

## Features
- Obstacles are now supported in `turbinate()` and `poisson_disc_filler()`.
- `as_normalized()` now works also with `L`.
- Replaced `multiprocessing.Pool` with `concurrent.futures.ThreadPoolExecutor` in HGS-CVRP calls, enabling concurrent solver instances without the quirks of the multiprocessing module. This requires a new version (v0.1.1+) of dependency hybgensea.
- Removed dependency `py`.

## Fixes
- Fixed potential infinite loop in `PathFinder` for inconsistent graphs.

# v0.1.6

[Commit history since v0.1.5](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.1.5...v0.1.6)

Drop-in replacement for v0.1.5. This release provides maily two important fixes:
- fix bugs caused by ortools v9.15.6755 released on 2026-01-12
- remove a duplicate turbine from the included location Gangkou 2

In addition, the graph attribute 'creator' of solutions produced by OWN was reverted back to using the naming convention adopted in earlier OWN versions, which includes the 'pyomo' string if the solver was called through it (e.g. 'MILP.pyomo.cplex' instead of 'MILP.cplex').

# v0.1.5

[Commit history since v0.1.4](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.1.4...v0.1.5)

Drop-in replacement for v0.1.4.

## Features
- Added new offshore wind locations: Dogger Bank B/C, Coastal Virginia, Inch Cape, Changhua 1, Gangkou 1/2, Yunlin, Noirmoutier, Tréport, Borkum Riffgrund 3, He Dreiht.
- Experimental **FiberSCIP (fscip)** solver support (system call, file-based interface).
- Improved automatic `landscape_angle` calculation
- Added `as_obstacle_free()` method to remove location obstacles; improved `as_single_root()`.
- `.osm.pbf` parsing now prioritizes tag `ref` over `name` for node labels.

## Fixes
-  Fixed dangling reference in diagonals (`make_planar_embedding()`) which could cause errors when checking for crossings.
- Applied rounding in `_link_val()`/`_flow_val()` for MILP Solvers CPLEX and SCIP to eliminate tiny non-zero values (error manifested as cyclic solutions).
- Corrected setting of `B` in `L_from_windIO()`.
- Resolved `_hull_processor()` edge case (wrong P for Yunlin).
- Ensured roots are added to solution topology `S` even if disconnected.
- Enforced integer values for SCIP model variables.
- Updated deprecated Shapely `buffer()` argument name.
- Adjusted graph attributes in MILP solvers.
- Multiple robustness improvements in tests and solver handling.


# v0.1.4

[Commit history since v0.1.3](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.1.3...v0.1.4)

Drop-in replacement for v0.1.3.

- gplot() and svgplot() now draw links with different line thickness to represent cable type (after assign_cables() is called)
- improve number formatting inside infobox of gplot() and svgplot()
- switch SCIP modelling from Pyomo to PySCIPOpt, enabling the launching of concurrent solvers for the same problem (competitive mode)
- refactor MILP code for reducing code duplication and improving consistency between model descriptions for the different APIs
- add information on how to install missing solvers when a requested solver is not available
- bump dependency NetworkX version to 3.6 (resolves pickling issues with nx.PlanarEmbedding)
- update the documentation to reflect the changes involving solver SCIP and plotting functions
- fix the assignment of graph attributes 'creator' (all solvers) and 'runtime' (scip)

# v0.1.3

[Commit history since v0.1.2](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.1.2...v0.1.3)

Another minor version bump to enable conda-forge recipe to work.

- improve tests coverage
- restructure tests to skip unavailable MILP solvers
- make db.modelv2 handle only schema definition
- get correct runtime for MILP solver SCIP

# v0.1.2

[Commit history since v0.1.1](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.1.1...v0.1.2)

Minor version bump to enable conda-forge recipe to work.

- include tests in source distribution (sdist tarball)
- update docs to state Python 3.11 and 3.12 are recommended

# v0.1.1

[Commit history since v0.1.0](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.1.0...v0.1.1)

## 📦 Packaging
- drop Python 3.10 support (v0.1.0 had an inconsistency due to NetworkX v3.5)
- minor syntax fix in pyproject.toml to make conda-forge package possible

# v0.1.0

[Commit history since v0.0.6](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.0.6...v0.1.0)

## ✨ New Features
- **Thor Offshore Wind Farm**: Added to location repository.
- **Lin-Kernighan-Helsgaun Meta-Heuristics solver (Advanced API only)**:
  - Introduced `iterative_lkh()` to deal with crossings.
  - Switched LKH to OVRP problem type.
  - Automatic prunning poor links from the available choices given to LKH.

## 🛠️ Fixes & Improvements
- Fixed runtime reporting for solver HiGHS.
- Adapted MILP code to Pyomo API v2.
- Enforced radial topology in HGSRouter.
- Improved hull construction and shortcut creation in planar embedding.
- Handled multiple crossings by single link in iterative meta-heuristics calls.
- Reduced rogue link usage in LKH.
- Improved precision handling in `lkh_acvrp()`.
- Improved handling of scaling parameters and significant digits.

## 🔧 Refactoring & Code Quality
- Removed `**kwargs` from key initializers.
- Improved consistency across HGS and LKH meta-heurists functions.
- Cleaned up angle helper utilities.
- Increased test coverage.

## 📚 Documentation
- Added advanced example notebook for LKH.
- Fixed typos and improved clarity in README and notebooks.
- Updated figures and notebook outlines for better HTML rendering.

## 📦 Dependencies
- Removed `pyyaml-include` dependency.
- Bumped `numba` version and removed `numpy` version cap.

# v0.0.6

[Commit history since v0.0.5](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.0.5...v0.0.6)

- Almost a drop-in replacement for v0.0.5
  - single existing API change: argument name of HGS meta-heuristics: from max_reruns to max_retries
- Introduction of Network/Router high-level API for easier on-boarding of new users
  - Two new components -- WindFarmNetwork and Router -- expose most of OWN's features
- Major expansion and improvement of the documentation
  - Improved the Advanced API docs
  - Fully documented the Network/Router API
  - Added Topfarm integration example
  - Added the OptiWindNet logo
- Added automated code testing based on pytest and tests for the main components
- MILP model warm-starting is now checked for feasibility before invoking the solver (Pyomo-only)
- Silenced warnings of Pyomo-based solvers when the search times out before the gap is reached
- Other small fixes and improvements

# v0.0.5

[Commit history since v0.0.4](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.0.4...v0.0.5)

- drop-in replacement for v0.0.4
- gplot()' options improvements:
  - 'node_tag=True' plots node numbers
  - 'node_tag="load"' now also plots the roots' loads
  - 'tag_border=True' plots numbers of border/obstacle vertices
- gplot() and svgplot() now can plot sites without borders
- bug fixes and improvements in path-finding
- bug fixes and improvements in navigation mesh generation
- mesh generation now can handle terminals placed on border lines
- some paperdb incomplete or incorrect entries were fixed
- other small fixes and improvements

# v0.0.4

[Commit history since v0.0.3](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.0.3...v0.0.4)

- fixed exception AttributeError on MacOS ('Process' object has no attribute 'cpu_affinity')
- added 3 more locations (Hollandse Kust Zuid, Vineyard 1, Sofia)
- enabled easy wind farm creation and import using JOSM (external program with GUI)
- many improvements in docstrings and documentation in general

# v0.0.3

[Commit history since v0.0.2](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/v0.0.2...v0.0.3)

- merged all features from the paper's computational experiments
- introduced a new API for MILP solvers
- introduced a multi-root capable HGS-CVRP wrapper
- several bug fixes

# v0.0.2

[Commit history since v0.0.1](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/compare/interarray-0.0.1...v0.0.2)

- project renamed to OptiWindNet and package to optiwindnet
- many more changes and bug fixes

# interarray-0.0.1

First release.
