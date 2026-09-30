# Input Formats

A location supplies the turbine and substation positions and any boundaries that constrain cable routing. This page describes how to supply that location through arrays, files, or a repository, independently of which API loads it. The resulting location graph `L` is described in [](/problem.md#graph-representations). Turbine ratings and cable specifications are covered in {doc}`/reference/power`.

## Required inputs

| Item | Required | Notes |
| --- | --- | --- |
| Turbine coordinates | yes | One planar `(x, y)` position for each turbine. |
| Substation coordinates | yes | One planar `(x, y)` position for each substation, in the same coordinate system. |
| Border | no | A polygon delimiting the area where cables may be laid. |
| Obstacles | no | Polygons inside the border where cables may not be laid. |
| Cable types | for routing | Supplied separately from the location; see [](/reference/power.md#cable-specifications). |

_OptiWindNet_ works without a border and obstacles; the geometry constraints simply do not apply in that case.

_In use:_ {doc}`/notebooks/hi11_data_input` (Network/Router API) · {doc}`/notebooks/lo11_data_input` (Advanced API).

## Supported formats

All four formats produce the location graph `L`. Choose the format that fits the source of the data:

| Format | Typical use |
| --- | --- |
| [](#coordinate-arrays) | Coordinates generated or already held in Python. |
| [](#windio-yaml) | Inputs shared with wind energy system models. |
| [](#optiwindnet-yaml) | A compact site file with coordinates, boundaries, and optional turbine ratings. |
| [](#openstreetmap-pbf) | A location digitized from a map. |

### Coordinate arrays

Coordinates are passed as _numpy_ arrays of `(x, y)` pairs — if the coordinates are held in separate one-dimensional arrays `X` and `Y`, use `np.column_stack((X, Y))`. Use a common planar coordinate system and length unit for turbines, substations and boundaries; reported lengths use that unit, and linear cable costs must use it too. A border polygon is defined by its sequence of vertices, with the segment closing the last vertex back to the first left implicit. Obstacles are given as a sequence of such polygons.

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
| `SUBSTATIONS` | yes | Positions of the substations. |
| `TURBINE` | no | The turbine model's `make`, `model` and `power_MW` — a list of them, each with its `qty` and optional `prefix`, for a site of turbines of unequal power. |
| `TURBINES` | yes | Positions of the turbines, in input order. |

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

The `TURBINE` section declares the turbines' nominal power in MW. A mapping states one `power_MW` for the whole site. A list gives one entry per turbine model. Each entry's `qty` claims the next group of positions in `TURBINES`, so the groups must follow the coordinate order and their quantities must add up to the total turbine count:

```yaml
TURBINE:
  - make: Siemens Gamesa
    model: SG 8.0-167 DD
    power_MW: 8
    qty: 94
  - make: Vestas
    model: V164-9.5
    power_MW: 9.5
    qty: 79
```

Where `TURBINES` is not grouped by turbine model, use a `prefix` for each model entry instead. It selects turbines by the start of their labels, regardless of their position in the coordinate list:

```yaml
TURBINE:
  - make: MHI Vestas
    model: V164-8.0
    power_MW: 8.25
    qty: 40
    prefix: 3-
  - make: Siemens Gamesa
    model: SWT-7.0-154
    power_MW: 7
    qty: 47
    prefix: 4-
```

Prefixes are all or nothing, must claim every turbine exactly once, and turn `qty` into a cross-check on how many each one matched.

`power_unit` becomes `MW` and the declared values are set with `set_terminal_power()`. See [](/reference/power.md#power-inputs-and-defaults) for how to use the imported ratings. A list whose entries carry no `power_MW` declares no power. `L_from_yaml(..., read_powers=False)` ignores the section. A list entry with neither `qty` nor `prefix`, quantities that do not add up, prefixes that do not claim every turbine exactly once, and a `power_MW` that is not a positive finite number all raise `ValueError`.

_In use:_ {doc}`/notebooks/hi11_data_input` (Network/Router API) · {doc}`/notebooks/lo11_data_input` (Advanced API).

### OpenStreetMap PBF

`.osm.pbf` stands for _OpenStreetMap Protocol Buffer Binary Format_. It is the format to use when the location is digitized from a map.

The [JOSM](https://josm.openstreetmap.de/) open-source map editor is recommended for producing these files. The JOSM plugin **pbf** is required to save in the `.osm.pbf` format; the plugin **opendata** is useful for importing many common GIS file formats.

The OpenStreetMap objects used to represent a wind farm location are _nodes_, _ways_ and _multipolygons_ (a relation between closed ways):

| Element | Represented by |
| --- | --- |
| Wind turbine | a _node_ tagged `power=generator` |
| Substation | a _node_, or a closed _way_, tagged `power=substation` or `power=transformer` |
| Border | a closed _way_ tagged `power=plant` |
| Border with obstacles | a _multipolygon_ tagged `power=plant`, combining the closed _ways_ for the border and the obstacles — the _ways_ themselves are then left untagged |

A substation based on a _way_ is reduced to the centroid of the polygon that the _way_ defines. The node tags `name` or `ref` are loaded as the node's `label` attribute.

A generator's `generator:output:electricity` tag — a number followed by an optional unit, such as `8 MW` — declares its nominal power. It is loaded only if every generator declares a positive output in a common unit: the unit becomes the graph attribute `power_unit` and the outputs are set with `set_terminal_power()`. See [](/reference/power.md#power-inputs-and-defaults) for how to use the imported ratings. Values with no number, such as `yes`, are ignored, and a location where only some generators declare their output carries no power at all. Generators declaring their output in more than one unit raise `ValueError`. `L_from_pbf(..., read_powers=False)` ignores the tags.

_In use:_ {doc}`/notebooks/hi11_data_input` (Network/Router API) · {doc}`/notebooks/lo11_data_input` (Advanced API).

## Location repositories

{py:func}`load_repository() <optiwindnet.importer.load_repository>` reads every `.osm.pbf` and `.yaml` file in a directory into a _namedtuple_ of NetworkX graphs, one per location. Called without arguments, it loads the locations distributed with _OptiWindNet_; called with a path, it loads a repository of your own. `read_powers=False` loads the locations without their declared turbine power.

The bundled locations are real offshore wind farms and are used throughout this documentation as ready-made examples.

Power declarations are read by default. In particular, Borssele, Trianel Windpark Borkum, and Walney Extension declare unequal turbine ratings. To load them without ratings for a study in turbine counts, use `load_repository(read_powers=False)`. For using imported ratings through either API, see [](/reference/power.md#power-inputs-and-defaults), [](/reference/power.md#router-support), and the Advanced API's [](/reference/power.md#quantization-contract).

_In use:_ {doc}`/notebooks/hi12_locations` (Network/Router API) · {doc}`/notebooks/lo12_locations` (Advanced API).

## Preparing geometry

Boundaries digitized from a map are frequently not directly usable: obstacles may touch or intersect the border, concavities may be too narrow for a cable to pass, and turbines may sit marginally outside the allowed area. Merging obstacles into the border, buffering the boundaries to add a safety margin, and validating that every turbine and substation lies inside the allowed area are covered — with the geometry plotted before and after each operation — in {doc}`/notebooks/hi13_border_obstacles`.
