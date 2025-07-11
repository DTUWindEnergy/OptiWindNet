# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import re
from collections import namedtuple
from importlib.resources import files
from itertools import chain, zip_longest
from pathlib import Path

import esy.osm.pbf
import networkx as nx
import numpy as np
import shapely as shp
import utm
import yaml

from .interarraylib import L_from_site
from .utils import F, make_handle

_lggr = logging.getLogger(__name__)
info = _lggr.info

__all__ = ('L_from_yaml', 'L_from_pbf', 'load_repository')


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
    for tag, lat, lon in _get_entries(entry_list):
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
        translated.append((tag, *utm.from_latlon(*latlon)))
    return translated


def _parser_latlon(entry_list):
    # separate data into columns
    tags, eastings, northings, zone_numbers, zone_letters = zip(
        *_translate_latlonstr(entry_list)
    )
    # all coordinates must belong to the same UTM zone
    assert all(num == zone_numbers[0] for num in zone_numbers[1:])
    assert all(letter == zone_letters[0] for letter in zone_letters[1:])
    return np.c_[eastings, northings], (tags if any(tags) else ())


def _parser_planar(entry_list):
    tags = []
    coords = []
    for tag, easting, northing in _get_entries(entry_list):
        tags.append(tag)
        coords.append((float(easting), float(northing)))
    return np.array(coords, dtype=float), (tags if any(tags) else ())


coordinate_parser = dict(
    latlon=_parser_latlon,
    planar=_parser_planar,
)


def L_from_yaml(filepath: Path | str, handle: str | None = None) -> nx.Graph:
    """Import wind farm data from .yaml file.

    Two options available for COORDINATE_FORMAT: "planar" and "latlon".

    Format "planar" is: [tag] easting northing. Example::

      TAG 234.2 5212.5

    Format "latlon" is: [tag] latitude longitude. Example::

      TAG1 11°22.333'N 44°55.666'E
      TAG2 11.3563°N 44.8903°E
      TAG3 11°22'20"N 44°55'40"E

    The [tag] is optional. Ensure no spaces within a latitude or longitude.

    The coordinate pair may be separated by "," or ";" and may be enclosed in
    "[]" or "()". Example::

      TAG [234.2, 5212.5]

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
    Border, BorderTag = coordinate_parser[format](parsed_dict['EXTENTS'])
    Root, RootTag = coordinate_parser[format](parsed_dict['SUBSTATIONS'])
    Terminal, TerminalTag = coordinate_parser[format](parsed_dict['TURBINES'])
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
    G.add_nodes_from(
        (n, {'kind': 'wtg', 'label': (tag if tag else F[n])})
        for n, tag in zip_longest(range(T), TerminalTag)
    )
    G.add_nodes_from(
        (r, {'kind': 'oss', 'label': (tag if tag else F[r])})
        for r, tag in zip_longest(range(-R, 0), RootTag)
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
    turbines = []
    border_raw = None
    obstacles_raw = []
    ways = {}
    for e in osm:
        match e:
            case esy.osm.pbf.Node():
                nodes[e.id] = e
                power_kind = e.tags.get('power')
                match power_kind:
                    case 'substation' | 'transformer':
                        substations.append(e.lonlat[::-1])
                    case 'generator':
                        turbines.append(e.lonlat[::-1])
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
                        substations.append(
                            [nodes[nid].lonlat[::-1] for nid in e.refs[:-1]]
                        )
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
    border = []
    border_latlon = []
    for latlon in border_raw:
        if latlon not in node_latlon:
            border_latlon.append(latlon)
            border.append(i)
            node_latlon[latlon] = i
            i += 1
        else:
            border.append(node_latlon[latlon])
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

    L = L_from_site(
        T=T,
        R=R,
        VertexC=VertexC,
        name=name,
        handle=handle,
    )
    if border is not None:
        L.graph['border'] = np.array(border, dtype=np.int_)
        if obstacles:
            L.graph['obstacles'] = [
                np.array(obstacle, dtype=np.int_) for obstacle in obstacles
            ]
        # landscape_angle calculation
        border_utm = shp.Polygon(shell=VertexC[border])
        x, y = border_utm.minimum_rotated_rectangle.exterior.coords.xy
        side0 = np.hypot(x[1] - x[0], y[1] - y[0])
        side1 = np.hypot(x[2] - x[1], y[2] - y[1])
        if side0 < side1:
            angle = np.arctan2((x[1] - x[0]), (y[1] - y[0])).item()
        else:
            angle = np.arctan2((x[2] - x[1]), (y[2] - y[1])).item()
        if abs(angle) > np.pi / 2:
            angle += np.pi if angle < 0 else -np.pi
        L.graph['landscape_angle'] = 180 * angle / np.pi

    L.graph['B'] = B

    if plant_name is not None:
        L.graph['OSM_name'] = plant_name

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
