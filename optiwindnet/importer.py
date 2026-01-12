# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import math
import re
from collections import namedtuple
from importlib.resources import files
from itertools import chain
from pathlib import Path

import esy.osm.pbf
import networkx as nx
import numpy as np
import shapely as shp
from scipy.spatial import ConvexHull
import utm
import yaml

from .geometric import rotating_calipers
from .interarraylib import L_from_site
from .utils import make_handle

_lggr = logging.getLogger(__name__)
info, warn = _lggr.info, _lggr.warning

__all__ = ('L_from_yaml', 'L_from_pbf', 'L_from_windIO', 'load_repository')


_coord_sep = r',\s*|;\s*|\s{1,}|,|;'
_coord_lbraces = '(['
_coord_rbraces = ')]'


def _get_entries(entries):
    if isinstance(entries, str):
        for entry in entries.splitlines():
            *opt, lat, lon = re.split(_coord_sep, entry)
            lat = lat.lstrip(_coord_lbraces)
            lon = lon.rstrip(_coord_rbraces)
            if opt:
                yield opt[0], lat, lon
            else:
                yield None, lat, lon
    else:
        for entry in entries:
            if len(entry) > 2:
                yield entry
            else:
                yield (None, *entry)


def _translate_latlonstr(entry_list):
    translated = []
    min = sec = 0.0
    for label, lat, lon in _get_entries(entry_list):
        latlon = []
        for ll in (lat, lon):
            deg, *tail = ll.split('°')
            if tail:
                min, *tail = tail[0].split("'")
                if not tail:
                    hemisphere = min.strip()
                    min = 0.0
                else:
                    sec, *tail = tail[0].split('"')
                    if not tail:
                        hemisphere = sec.strip()
                        sec = 0.0
                    else:
                        hemisphere = tail[0].strip()
                latlon.append(
                    (float(deg) + (float(min) + float(sec) / 60) / 60)
                    * (1 if hemisphere in ('N', 'E') else -1)
                )
            else:
                # entry is a signed fractional degree without hemisphere letter
                latlon.append(float(deg))
        translated.append((label, *utm.from_latlon(*latlon)))
    return translated


def _parser_latlon(entry_list):
    # separate data into columns
    labels, eastings, northings, zone_numbers, zone_letters = zip(
        *_translate_latlonstr(entry_list)
    )
    # all coordinates must belong to the same UTM zone
    assert all(num == zone_numbers[0] for num in zone_numbers[1:])
    assert all(letter == zone_letters[0] for letter in zone_letters[1:])
    return np.c_[eastings, northings], (labels if any(labels) else ())


def _parser_planar(entry_list):
    labels = []
    coords = []
    for label, easting, northing in _get_entries(entry_list):
        labels.append(label)
        coords.append((float(easting), float(northing)))
    return np.array(coords, dtype=float), (labels if any(labels) else ())


coordinate_parser = dict(
    latlon=_parser_latlon,
    planar=_parser_planar,
)


def L_from_yaml(filepath: Path | str, handle: str | None = None) -> nx.Graph:
    """Import wind farm data from .yaml file.

    Two options available for COORDINATE_FORMAT: "planar" and "latlon".

    Format "planar" is: [label] easting northing. Example::

      LABEL 234.2 5212.5

    Format "latlon" is: [label] latitude longitude. Example::

      LABEL1 11°22.333'N 44°55.666'E
      LABEL2 11.3563°N 44.8903°E
      LABEL3 11°22'20"N 44°55'40"E

    The [label] is optional. Ensure no spaces within a latitude or longitude.

    The coordinate pair may be separated by "," or ";" and may be enclosed in
    "[]" or "()". Example::

      LABEL [234.2, 5212.5]

    Args:
      filepath: path to `.yaml` file to read.
      handle: Short moniker for the site.

    Returns:
      Unconnected location graph L.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    # read wind power plant site YAML file
    parsed_dict = yaml.safe_load(open(filepath, 'r', encoding='utf8'))
    name = filepath.stem
    handle = parsed_dict.get('HANDLE')
    if handle is None:
        handle = make_handle(name)
    # default format is "latlon"
    format = parsed_dict.get('COORDINATE_FORMAT', 'latlon')
    Border, BorderLabel = coordinate_parser[format](parsed_dict['EXTENTS'])
    Root, RootLabel = coordinate_parser[format](parsed_dict['SUBSTATIONS'])
    Terminal, TerminalLabel = coordinate_parser[format](parsed_dict['TURBINES'])
    T = Terminal.shape[0]
    R = Root.shape[0]
    node_xy = {xy: i for i, xy in enumerate(map(tuple, Terminal))}
    node_xy.update({xy: i for i, xy in enumerate(map(tuple, Root), start=-R)})
    i = T
    border_xy = []
    border = []
    for xy in map(tuple, Border):
        if xy not in node_xy:
            border_xy.append(xy)
            border.append(i)
            node_xy[xy] = i
            i += 1
        else:
            border.append(node_xy[xy])
    B = len(border_xy)
    optional = {}
    obstacles = parsed_dict.get('OBSTACLES')
    obstacleC_ = []
    if obstacles is not None:
        # obstacle has to be a list of arrays, so parsing is a bit different
        indices = []
        for obstacle_entry in parsed_dict['OBSTACLES']:
            obstacleC, poly_tag = coordinate_parser[format](obstacle_entry)

            obstacle_xy = []
            obstacle = []
            for xy in map(tuple, obstacleC):
                if xy not in node_xy:
                    obstacle_xy.append(xy)
                    obstacle.append(i)
                    node_xy[xy] = i
                    i += 1
                else:
                    obstacle_xy.append(node_xy[xy])
            B += len(obstacle_xy)

            indices.append(np.array(obstacle, dtype=np.int_))
            obstacleC_.extend(obstacle_xy)
        optional['obstacles'] = indices

    VertexC = np.vstack((Terminal, *border_xy, *obstacleC_, Root))

    lsangle = parsed_dict.get('LANDSCAPE_ANGLE')
    if lsangle is not None:
        optional['landscape_angle'] = lsangle

    # create networkx graph
    G = nx.Graph(
        T=T,
        R=R,
        B=B,
        VertexC=VertexC,
        border=np.array(border, dtype=np.int_),
        name=name,
        handle=handle,
        **optional,
    )

    # populate graph G
    G.add_nodes_from(range(T), kind='wtg')
    if TerminalLabel:
        nx.set_node_attributes(G, {t: TerminalLabel[t] for t in range(T)}, name='label')
    G.add_nodes_from(range(-R, 0), kind='oss')
    if RootLabel:
        nx.set_node_attributes(
            G, {-R + r: RootLabel[r] for r in range(R)}, name='label'
        )
    return G


def L_from_pbf(filepath: Path | str, handle: str | None = None) -> nx.Graph:
    """Import wind farm data from .osm.pbf file.

    Args:
        filepath: path to `.osm.pbf` file to read.
        handle: Short moniker for the site.

    Returns:
        Unconnected location graph L.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    assert ['.osm', '.pbf'] == filepath.suffixes[-2:], (
        'Argument `filepath` does not have `.osm.pbf` extension.'
    )
    name = filepath.stem[:-4]
    osm = esy.osm.pbf.File(filepath)
    plant_name = None
    nodes = {}
    substations = []
    substation_labels = []
    turbines = []
    turbine_labels = []
    border_raw = None
    obstacles_raw = []
    ways = {}
    for e in osm:
        match e:
            case esy.osm.pbf.Node():
                nodes[e.id] = e
                power_kind = e.tags.get('power')
                if power_kind is None:
                    power_kind = e.tags.get('construction:power')
                label = e.tags.get('ref') or e.tags.get('name')
                match power_kind:
                    case 'substation' | 'transformer':
                        substations.append(e.lonlat[::-1])
                        substation_labels.append(label)
                    case 'generator':
                        turbines.append(e.lonlat[::-1])
                        turbine_labels.append(label)
                    case _:
                        info('Unhandled power category for Node: %s', power_kind)

            case esy.osm.pbf.Way():
                power_kind = e.tags.get('power')
                if power_kind is None:
                    power_kind = e.tags.get('construction:power')
                match power_kind:
                    case 'plant':
                        plant_name = e.tags.get('name:en') or e.tags.get('name')
                        handle = e.tags.get('handle') or make_handle(name)
                        if border_raw is not None:
                            raise ValueError('Only a single border is supported.')
                        border_raw = [nodes[nid].lonlat[::-1] for nid in e.refs[:-1]]
                    case 'substation' | 'transformer':
                        label = e.tags.get('ref') or e.tags.get('name')
                        substations.append(
                            [nodes[nid].lonlat[::-1] for nid in e.refs[:-1]]
                        )
                        substation_labels.append(label)
                    case 'generator':
                        info('Generator must be Node, not Way.')
                    case None:
                        # likely to be used in a Relation
                        ways[e.id] = e
                    case _:
                        info('Unhandled power category for Way: %s', power_kind)
            case esy.osm.pbf.Relation():
                if e.tags.get('type') == 'multipolygon':
                    power_kind = e.tags.get('power')
                    if power_kind is None:
                        power_kind = e.tags.get('construction:power')
                    match power_kind:
                        case 'plant':
                            plant_name = e.tags.get('name:en') or e.tags.get('name')
                            handle = e.tags.get('handle') or make_handle(name)
                            for m in e.members:
                                eid, cls, kind = m
                                match cls:
                                    case 'WAY':
                                        match kind:
                                            case 'outer':
                                                if border_raw is not None:
                                                    raise ValueError(
                                                        'Only a single border is supported.'
                                                    )
                                                border_raw = [
                                                    nodes[nid].lonlat[::-1]
                                                    for nid in ways[eid].refs[:-1]
                                                ]
                                            case 'inner':
                                                obstacles_raw.append(
                                                    [
                                                        nodes[nid].lonlat[::-1]
                                                        for nid in ways[eid].refs[:-1]
                                                    ]
                                                )
                        case _:
                            info(
                                'Unhandled power category for Relation: %s', power_kind
                            )

    T = len(turbines)
    R = len(substations)
    if T == 0 or R == 0:
        raise ValueError(
            f'Location: "{name}" -> Unable to identify at least one substation and one generator.'
        )

    #  for i, substation in enumerate(tuple(substations)):
    for i, substation in enumerate(tuple(substations)):
        if isinstance(substation, list):
            # Substation defined as a polygon, reduce it to a point
            easting, northing, zone_num, zone_let = utm.from_latlon(
                *np.array(tuple(zip(*substation)))
            )
            centroid = shp.Polygon(shell=list(zip(easting, northing))).centroid
            latlon = utm.to_latlon(centroid.x, centroid.y, zone_num, zone_let)
            substations[i] = latlon

    node_latlon = {node: i for i, node in enumerate(turbines)}
    node_latlon.update({node: i for i, node in enumerate(substations, start=-R)})

    i = T
    border_list = []
    border_latlon = []
    for latlon in border_raw:
        if latlon not in node_latlon:
            border_latlon.append(latlon)
            border_list.append(i)
            node_latlon[latlon] = i
            i += 1
        else:
            border_list.append(node_latlon[latlon])
    B = len(border_latlon)

    obstacles = []
    obstacles_latlon = []
    for obstacle_entry in obstacles_raw:
        obstacle_latlon = []
        obstacle = []
        for latlon in obstacle_entry:
            if latlon not in node_latlon:
                obstacle_latlon.append(latlon)
                obstacle.append(i)
                node_latlon[latlon] = i
                i += 1
            else:
                obstacle.append(node_latlon[latlon])
        B += len(obstacle_latlon)

        obstacles.append(np.array(obstacle, dtype=np.int_))
        obstacles_latlon.extend(obstacle_latlon)

    # Build site data structure
    latlon = np.array(
        tuple(
            chain(
                turbines,
                border_latlon,
                obstacles_latlon,
                substations,
            )
        ),
        dtype=float,
    )

    # TODO: find the UTM sector that includes the most coordinates among
    # vertices and boundary (bin them in 6° sectors and count). Then insert
    # a dummy coordinate as the first array row, such that utm.from_latlon()
    # will pick the right zone for all points.
    VertexC = np.c_[utm.from_latlon(*latlon.T)[:2]]

    if handle is None:
        handle = make_handle(name)
    L = L_from_site(
        T=T,
        R=R,
        VertexC=VertexC,
        name=name,
        handle=handle,
    )
    for labels, start in ((substation_labels, -R), (turbine_labels, 0)):
        if any(labels):
            for i, label in enumerate(labels, start=start):
                if label is not None:
                    L.nodes[i]['label'] = label
    if border_list is not None:
        border = np.array(border_list, dtype=np.int_)
        L.graph['border'] = border
        # for now, obstacles are allowed only if a border is defined
        if obstacles:
            L.graph['obstacles'] = [
                np.array(obstacle, dtype=np.int_) for obstacle in obstacles
            ]
        border_list.extend([r for r in range(-R, 0) if r not in set(border_list)])
        hullC_ = VertexC[
            np.array(border_list)[ConvexHull(VertexC[border_list]).vertices]
        ]
    else:
        # if no border is defined, pass all vertices to ConvexHull
        hullC_ = VertexC[ConvexHull(VertexC).vertices]
    _, best_caliper_angle, _, _ = rotating_calipers(hullC_, metric='height')
    best_caliper_angle_deg = 180 * best_caliper_angle / math.pi
    ls_angle = 90 - best_caliper_angle_deg
    ls_angle = (
        ls_angle if -90 <= ls_angle < 90 else ls_angle + (180 if ls_angle < 0 else -180)
    )
    L.graph['landscape_angle'] = ls_angle

    L.graph['B'] = B

    if plant_name is not None:
        L.graph['OSM_name'] = plant_name

    return L


def _yaml_include_constructor(loader, node):
    filename = node.value
    with open(filename, 'r') as f:
        return yaml.load(f, Loader=type(loader))


class IncludeLoader(yaml.SafeLoader):
    def __init__(self, stream):
        # Store the directory of the currently loaded YAML file
        self._parent = Path(stream.name).parent
        super().__init__(stream)
        self.add_constructor('!include', IncludeLoader.include)

    def include(self, node):
        # Construct the full path of the file to include, relative to parent YAML
        include_path = Path(self.construct_scalar(node))
        if include_path.suffix not in ('.yml', '.yaml'):
            warn(
                'Ignoring YAML "!include" directive to unsupported file type (%s)',
                include_path,
            )
            return {}
        if not include_path.is_absolute():
            include_path = self._parent / include_path
        with open(include_path, 'r') as f:
            # When processing includes, use IncludeLoader to maintain correct directory context
            return yaml.load(f, IncludeLoader)


def L_from_windIO(filepath: Path | str, handle: str | None = None) -> nx.Graph:
    """Import wind farm data from a windIO .yaml file.

    Args:
      filepath: path to windIO `.yaml` file to read.
      handle: Short moniker for the site.

    Returns:
      Unconnected location geometry L.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    name = filepath.stem
    system = yaml.load(filepath.open(), IncludeLoader)
    coords = system['wind_farm']['layouts']['initial_layout']['coordinates']
    terminalC = np.c_[coords['x'], coords['y']]
    coords = system['wind_farm']['electrical_substations']['coordinates']
    rootC = np.c_[coords['x'], coords['y']]
    coords = system['site']['boundaries']['polygons'][0]
    borderC = np.c_[coords['x'], coords['y']]

    T = terminalC.shape[0]
    R = rootC.shape[0]
    B = borderC.shape[0]
    if handle is None:
        handle = make_handle(name)

    L = L_from_site(
        R=R,
        T=T,
        B=B,
        VertexC=np.vstack((terminalC, borderC, rootC)),
        **({'border': np.arange(T, T + B)} if (borderC is not None and B >= 3) else {}),
        name=name,
        handle=handle,
    )
    return L


def load_repository(path: Path | str | None = None) -> tuple[nx.Graph, ...]:
    """Load locations from files of known formats into a namedtuple.

    Each file (.yaml or .osm.pbf) is translated into a location graph and
    included as an attribute in the returned namedtuple. The attribute name
    can be specified in the .yaml with the field `HANDLE` or in the .osm.pbf
    file with the tag `handle` applied to the power plant object.

    Args:
      path: Path to look for location files (non-recursive). If omited, the
        locations included in optiwindnet are loaded.
    Returns:
      Named tuple which has the location handles as attribute identifiers.
    """
    if path is None:
        path = files(__package__) / 'data'
    else:
        path = Path(path)
    locations = [L_from_yaml(file) for file in path.glob('*.yaml')]
    locations.extend(L_from_pbf(file) for file in path.glob('*.osm.pbf'))
    handles = tuple(L.graph['handle'] for L in locations)
    return namedtuple('Locations', handles)(*locations)
