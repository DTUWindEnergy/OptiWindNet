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
