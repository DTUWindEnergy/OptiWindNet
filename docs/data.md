(data)=
# Input Data

A problem instance for *OptiWindNet* consists of a **location** — the geometry — and the properties of the **available cable types**.
This section describes the accepted formats and the operations available for preparing and inspecting the data, independently of which API you use to load it.

(data-instance)=
## What an instance requires

| Item | Required | Notes |
| --- | --- | --- |
| Turbine coordinates | yes | 2D planar coordinates, one `(x, y)` pair per terminal. |
| Substation coordinates | yes | One `(x, y)` pair per root. |
| Cable types | yes | See {ref}`data-cables`. |
| Border | no | A polygon delimiting the area where cables may be laid. |
| Obstacles | no | Polygons inside the border where cables may not be laid. |

*OptiWindNet* works without a border and obstacles; the geometry constraints simply do not apply in that case.
The geometric data carries much more volume and complexity than the cable properties, which is why the input formats below are concerned almost entirely with it.

(data-cables)=
## Cable types

Cable capacity is expressed in **number of turbines**, not in amperes: it is the number of terminals whose full output the cable can carry.
Three equivalent ways of declaring the available types are accepted:

* a single number — the maximum capacity among all available cables, when cost is not of interest;
* a list of capacities — one entry per cable type;
* a list of `(capacity, linear_cost)` pairs — capacities must be increasing, and cost is per unit of length.

Only the last form allows *OptiWindNet* to report a network cost; with the other two, the objective is total cable length.
Each link is assigned the cheapest type that can carry its load, so a link's cable type follows from the {term}`load` it carries.

(data-formats)=
## Input formats

Four formats are accepted. All of them produce the same location graph `L` described in {ref}`problem-graph-model`, so the choice is purely one of convenience.

(data-format-arrays)=
### Coordinate arrays

Coordinates are passed as *numpy* arrays of `(x, y)` pairs — if the coordinates are held in separate arrays `X` and `Y`, use `np.hstack((X, Y))`.
A border polygon is defined by its sequence of vertices, with the segment closing the last vertex back to the first left implicit.
Obstacles are given as a sequence of such polygons.

This is the format to use when the layout is generated programmatically, for instance inside an optimization loop driven by another tool.

(data-format-windio)=
### windIO YAML

[windIO](https://github.com/IEAWindSystems/windIO) is a community data format for inputs and outputs of wind energy system models.
Originally focused on systems engineering models, it has since been adopted across other areas of wind energy modeling.
See the [windIO documentation](https://ieawindsystems.github.io/windIO/main/index.html) for the format itself.

(data-format-yaml)=
### OptiWindNet YAML

*OptiWindNet*'s own YAML schema is a compact way to keep a location in a file:

| Key | Required | Content |
| --- | --- | --- |
| `COORDINATE_FORMAT` | no | `planar` or `latlon` — defaults to `latlon`. |
| `EXTENTS` | yes | The border polygon. Do not repeat the initial vertex at the end. |
| `OBSTACLES` | no | A list of polygons, even when there is only one. |
| `SUBSTATIONS` | yes | Root coordinates. |
| `TURBINES` | yes | Terminal coordinates. |

Coordinates are given either as lists of `[x, y]` pairs, for `planar`, or as a text block of latitude/longitude, for `latlon`:

```yaml
COORDINATE_FORMAT: latlon

SUBSTATIONS: |-
  OSS 56°35.748'N 11°09.174'E

TURBINES: |-
  A01 56°30.477'N 11°11.026'E
  A02 56°30.810'N 11°11.078'E
```

In the `latlon` form, any identifier placed *before* the coordinates — `OSS`, `A01`, `A02` above — is loaded as the node's `label` attribute and can be shown in plots.
Several examples are bundled in the folder `optiwindnet/data`; look for them in Python's `site-packages` or in [the repository](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/tree/main/optiwindnet/data).

(data-format-pbf)=
### OpenStreetMap PBF

`.osm.pbf` stands for *OpenStreetMap Protocolbuffer Binary Format*.
It is the format to use when the location is digitized from a map.

The [JOSM](https://josm.openstreetmap.de/) open-source map editor is recommended for producing these files.
The JOSM plugin **pbf** is required to save in the `.osm.pbf` format; the plugin **opendata** is useful for importing many common GIS file formats.

The OpenStreetMap objects used to represent a wind farm location are *nodes*, *ways* and *multipolygons* (a relation between closed ways):

| Element | Represented by |
| --- | --- |
| Wind turbine | a *node* tagged `power=generator` |
| Substation | a *node*, or a closed *way*, tagged `power=substation` or `power=transformer` |
| Border | a closed *way* tagged `power=plant` |
| Border with obstacles | a *multipolygon* tagged `power=plant`, combining the closed *ways* for the border and the obstacles — the *ways* themselves are then left untagged |

A substation based on a *way* is reduced to the centroid of the polygon that the *way* defines.
The node tags `name` or `ref` are loaded as the node's `label` attribute.

*In use:* {doc}`notebooks/hi11_data_input` (Network/Router API) · {doc}`notebooks/lo11_data_input` (Advanced API).

(data-repository)=
## Location repositories

`load_repository()` reads every `.osm.pbf` and `.yaml` file in a directory into a *namedtuple* of *networkx* graphs, one per location.
Called without arguments, it loads the locations distributed with *OptiWindNet*; called with a path, it loads a repository of your own.

The bundled locations are real offshore wind farms and are used throughout this documentation as ready-made examples.

*In use:* {doc}`notebooks/hi12_repositories` (Network/Router API) · {doc}`notebooks/lo12_repositories` (Advanced API).

(data-geometry)=
## Preparing the geometry

Boundaries digitized from a map are frequently not directly usable: obstacles may touch or intersect the border, concavities may be too narrow for a cable to pass, and turbines may sit marginally outside the allowed area.
Three operations address this.

Merging obstacles into the border
: Resolves obstacles that intersect or touch the exterior border by absorbing them into it, drops obstacles that turn out to be irrelevant, and simplifies the resulting boundary. Fewer, simpler constraints mean a smaller navigation mesh.

Buffering
: Expands the border outward and shrinks obstacles inward by a given distance, adding a safety margin between the cables and the boundaries. Buffering is destructive in a useful way: a concavity narrower than the buffer distance disappears, and an obstacle smaller than the buffer distance is removed entirely. Both effects are reported when they occur, and the original and buffered geometry can be plotted together for comparison.

Validating placement
: Before optimizing, every turbine and substation is checked against the allowed area. A terminal outside the border, or inside an obstacle, raises a `ValueError` rather than producing a silently invalid network.

*In use:* {doc}`notebooks/hi31_border_obstacles` (Network/Router API).

(data-plotting)=
## Inspecting data and results

Five views are available, whichever API is used to produce them.
The first two cover ordinary use; the last three expose the internal graphs from {ref}`problem-graph-model` and are mainly for debugging and development.

| View | Shows |
| --- | --- |
| Location | The raw location `L`: terminals as circles, roots as squares, border and obstacles. No links — useful to confirm the data was loaded correctly, before optimizing. |
| Routeset | The optimized network `G`: actual cable routes, with detours dashed. Terminals of the same subtree share a color, and cable types are encoded in line thickness — thicker means higher capacity. |
| Navigation mesh | The triangulation `P` of every vertex of the location, including the supertriangle that contains them all. |
| Available links | The search space `A`, with solid lines for direct Delaunay links and dashed lines for diagonals; obstructed links are colored differently from direct ones. Plots omit the star links from each root to each terminal, for clarity. |
| Selected links | The links the method chose out of `A` — the topology `S`, drawn straight. |

Comparing the last two views is the clearest way to see the difference between a {term}`link` and a {term}`route`: *selected links* shows the electrical connections as straight lines, while *routeset* shows the paths the cables actually take, contours and detours included.

(data-plot-options)=
### Common plot options

`node_tag`
: Labels drawn inside the node symbols.

  | Value | Effect |
  | --- | --- |
  | `True` | The node number — from order of appearance in the data, starting at `0` for terminals and negative for roots. |
  | `'label'` | The node's `label` attribute, as defined in the `.yaml` or `.osm.pbf` file, or assigned programmatically. |
  | `'load'` | The {term}`load` exported by the node: for a terminal, the number of terminals upstream including itself; for a root, the total number of terminals connected to it. |
  | `'attribute_name'` | The value of any other node attribute. |

  Default symbol sizes suit tags of up to three characters; longer tags may not fit.

`dark`
: Color theme. By default the theme is matched to the operating system's via the `darkdetect` package — note that this detects the *system* theme, not JupyterLab's, so the two can disagree. Set `True` or `False` to decide explicitly.

`landscape`
: Each bundled location carries a `landscape_angle` graph attribute — the rotation that orients its widest dimension horizontally, suiting a landscape figure. Pass `landscape=False` to orient the location with north up instead.

*In use:* {doc}`notebooks/hi13_plotting` (Network/Router API) · {doc}`notebooks/lo13_plotting` (Advanced API).
