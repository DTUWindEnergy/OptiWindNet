# Input Formats

A problem instance for _OptiWindNet_ consists of a **location** — the geometry — and the properties of the **available cable types**. This page is the catalogue of what an instance must contain and of the formats it can be given in, independently of which API you use to load it. What the loaded instance becomes is described in [](/problem.md#graph-representations).

## What an instance requires

| Item | Required | Notes |
| --- | --- | --- |
| Turbine coordinates | yes | 2D planar coordinates, one `(x, y)` pair per terminal. |
| Substation coordinates | yes | One `(x, y)` pair per root. |
| Cable types | yes | See [](/reference/input_formats.md#cable-types). |
| Border | no | A polygon delimiting the area where cables may be laid. |
| Obstacles | no | Polygons inside the border where cables may not be laid. |

_OptiWindNet_ works without a border and obstacles; the geometry constraints simply do not apply in that case. The geometric data carries much more volume and complexity than the cable properties, which is why the input formats below are concerned almost entirely with it.

_In use:_ {doc}`/notebooks/hi11_data_input` (Network/Router API) · {doc}`/notebooks/lo11_data_input` (Advanced API).

## Cable types

Cable capacity is expressed in **number of turbines**, not in amperes: it is the number of terminals whose full output the cable can carry. Three equivalent ways of declaring the available types are accepted:

- a single number — the maximum capacity among all available cables, when cost is not of interest;
- a list of capacities — one entry per cable type;
- a list of `(capacity, linear_cost)` pairs — capacities must be increasing, and cost is per unit of length.

Only the last form allows _OptiWindNet_ to report a network cost; with the other two, the objective is total cable length. Each link is assigned the cheapest type that can carry its load, so a link's cable type follows from the {term}`load` it carries.

_In use:_ {doc}`/notebooks/hi11_data_input` (Network/Router API).

## Input formats

Four formats are accepted. All of them produce the same location graph `L` described in [](/problem.md#graph-representations), so the choice is purely one of convenience.

### Coordinate arrays

Coordinates are passed as _numpy_ arrays of `(x, y)` pairs — if the coordinates are held in separate arrays `X` and `Y`, use `np.hstack((X, Y))`. A border polygon is defined by its sequence of vertices, with the segment closing the last vertex back to the first left implicit. Obstacles are given as a sequence of such polygons.

This is the format to use when the layout is generated programmatically, for instance inside an optimization loop driven by another tool.

_In use:_ {doc}`/notebooks/hi11_data_input` (Network/Router API) · {doc}`/notebooks/lo11_data_input` (Advanced API).

### windIO YAML

[windIO](https://github.com/IEAWindSystems/windIO) is a community data format for inputs and outputs of wind energy system models. Originally focused on systems engineering models, it has since been adopted across other areas of wind energy modeling. See the [windIO documentation](https://ieawindsystems.github.io/windIO/main/index.html) for the format itself.

_In use:_ {doc}`/notebooks/hi11_data_input` (Network/Router API) · {doc}`/notebooks/lo11_data_input` (Advanced API).

### OptiWindNet YAML

_OptiWindNet_'s own YAML schema is a compact way to keep a location in a file:

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

In the `latlon` form, any identifier placed _before_ the coordinates — `OSS`, `A01`, `A02` above — is loaded as the node's `label` attribute and can be shown in plots. Several examples are bundled in the folder `optiwindnet/data`; look for them in Python's `site-packages` or in [the repository](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/tree/main/optiwindnet/data).

_In use:_ {doc}`/notebooks/hi11_data_input` (Network/Router API) · {doc}`/notebooks/lo11_data_input` (Advanced API).

### OpenStreetMap PBF

`.osm.pbf` stands for _OpenStreetMap Protocolbuffer Binary Format_. It is the format to use when the location is digitized from a map.

The [JOSM](https://josm.openstreetmap.de/) open-source map editor is recommended for producing these files. The JOSM plugin **pbf** is required to save in the `.osm.pbf` format; the plugin **opendata** is useful for importing many common GIS file formats.

The OpenStreetMap objects used to represent a wind farm location are _nodes_, _ways_ and _multipolygons_ (a relation between closed ways):

| Element | Represented by |
| --- | --- |
| Wind turbine | a _node_ tagged `power=generator` |
| Substation | a _node_, or a closed _way_, tagged `power=substation` or `power=transformer` |
| Border | a closed _way_ tagged `power=plant` |
| Border with obstacles | a _multipolygon_ tagged `power=plant`, combining the closed _ways_ for the border and the obstacles — the _ways_ themselves are then left untagged |

A substation based on a _way_ is reduced to the centroid of the polygon that the _way_ defines. The node tags `name` or `ref` are loaded as the node's `label` attribute.

_In use:_ {doc}`/notebooks/hi11_data_input` (Network/Router API) · {doc}`/notebooks/lo11_data_input` (Advanced API).

## Location repositories

{py:func}`load_repository() <optiwindnet.importer.load_repository>` reads every `.osm.pbf` and `.yaml` file in a directory into a _namedtuple_ of _networkx_ graphs, one per location. Called without arguments, it loads the locations distributed with _OptiWindNet_; called with a path, it loads a repository of your own.

The bundled locations are real offshore wind farms and are used throughout this documentation as ready-made examples.

_In use:_ {doc}`/notebooks/hi12_locations` (Network/Router API) · {doc}`/notebooks/lo12_locations` (Advanced API).

## Preparing the geometry

Boundaries digitized from a map are frequently not directly usable: obstacles may touch or intersect the border, concavities may be too narrow for a cable to pass, and turbines may sit marginally outside the allowed area. Merging obstacles into the border, buffering the boundaries to add a safety margin, and validating that every turbine and substation lies inside the allowed area are covered — with the geometry plotted before and after each operation — in {doc}`/notebooks/hi13_border_obstacles`.
