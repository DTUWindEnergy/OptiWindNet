# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import re
from itertools import chain, zip_longest
from collections import namedtuple, defaultdict
from pathlib import Path
from typing import NamedTuple, Iterable
from importlib.resources import files
from functools import reduce

import networkx as nx
import numpy as np
import utm
import yaml
import osmium
import shapely as shp
import shapely.wkb as wkblib

from .utils import NodeTagger
from .interarraylib import L_from_site

F = NodeTagger()


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
    for tag, lat, lon in _get_entries(entry_list):
        latlon = []
        for ll in (lat, lon):
            val, hemisphere = ll.split("'")
            deg, sec = val.split('°')
            latlon.append((float(deg) + float(sec)/60)
                          * (1 if hemisphere in ('N', 'E') else -1))
        translated.append((tag, *utm.from_latlon(*latlon)))
    return translated


def _parser_latlon(entry_list):
    # separate data into columns
    tags, eastings, northings, zone_numbers, zone_letters = \
        zip(*_translate_latlonstr(entry_list))
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
    '''Import wind farm data from .yaml file.

    Two options available for COORDINATE_FORMAT: "planar" and "latlon".

    Format "planar" is: [tag] easting northing. Example:
    TAG 234.2 5212.5

    Format "latlon" is: [tag] latitude longitude. Example:
    TAG 11°22.333'N 44°55.666'E

    The [tag] is optional. Only this specific latlon format is supported.

    The coordinate pair may be separated by "," or ";" and may be enclosed in
    "[]" or "()". Example:
    TAG [234.2, 5212.5]

    Args:
        filepath: path to `.yaml` file to read.
        handle: Short moniker for the site.

    Returns:
        Unconnected locations graph L.
    '''
    if isinstance(filepath, str):
        filepath = Path(filepath)
    # read wind power plant site YAML file
    parsed_dict = yaml.safe_load(open(filepath, 'r', encoding='utf8'))
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

    VertexC=np.vstack((Terminal, *border_xy, *obstacleC_, Root))

    lsangle = parsed_dict.get('LANDSCAPE_ANGLE')
    if lsangle is not None:
        optional['landscape_angle'] = lsangle

    # create networkx graph
    G = nx.Graph(T=T, R=R, B=B,
                 VertexC=VertexC,
                 border=np.array(border, dtype=np.int_),
                 name=filepath.stem,
                 handle=handle,
                 **optional)

    # populate graph G
    G.add_nodes_from((n, {'kind': 'wtg', 'label': (tag if tag else F[n])})
                      for n, tag in zip_longest(range(T), TerminalTag))
    G.add_nodes_from((r, {'kind': 'oss', 'label': (tag if tag else F[r])})
                      for r, tag in zip_longest(range(-R, 0), RootTag))
    return G


wkbfab = osmium.geom.WKBFactory()


class GetRelations(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.borderIDs = []
        self.obstacleIDs = []
        self.plant_tags = None
        self.plant_name = None

    def relation(self, r):
        power = r.tags.get('power') or r.tags.get('construction:power')
        if power == 'plant':
            if self.plant_tags is None:
                self.plant_tags = dict(r.tags)
                self.plant_name = r.tags.get('name:en') or r.tags.get('name')
            else:
                raise RuntimeError('Only single-power-plant .osm.pbf files are supported.')
            for member in r.members:
                if member.type == 'w':
                    if member.role == 'outer':
                        self.borderIDs.append(member.ref)
                    elif member.role == 'inner':
                        self.obstacleIDs.append(member.ref)


class GetAllData(osmium.SimpleHandler):
    def __init__(self, borderIDs, obstacleIDs):
        super().__init__()
        self.borderIDs = borderIDs
        self.obstacleIDs = obstacleIDs
        self.turbines = []
        self.substations = []
        self.border = None
        self.obstacles = []
        self.plant_name = None

    def node(self, n):
        power = n.tags.get('power') or n.tags.get('construction:power')
        if power == 'generator':
            self.turbines.append(
                wkblib.loads(wkbfab.create_point(n), hex=True)
            )
        if power in ['substation', 'transformer']:
            self.substations.append(
                wkblib.loads(wkbfab.create_point(n), hex=True)
            )

    def way(self, w):
        power = w.tags.get('power') or w.tags.get('construction:power')
        if power in ['substation', 'transformer']:
            self.substations.append(
                # reduce polygon-shaped substation/transformer to its centroid
                wkblib.loads(wkbfab.create_linestring(w), hex=True).centroid
            )
        elif power == 'plant':
            assert not self.borderIDs and not self.obstacleIDs
            assert self.border is None, 'The outer location border must be a single polygon.'
            self.border = wkblib.loads(wkbfab.create_linestring(w), hex=True)
            self.plant_name = w.tags.get('name:en') or w.tags.get('name')
        elif w.id in self.obstacleIDs:
            self.obstacles.append(
                wkblib.loads(wkbfab.create_linestring(w), hex=True)
            )
        elif w.id in self.borderIDs:
            assert self.border is None, 'The outer location border must be a single polygon.'
            self.border = wkblib.loads(wkbfab.create_linestring(w), hex=True)


def L_from_pbf(filepath: Path | str, handle: str | None = None) -> nx.Graph:
    '''Import wind farm data from .osm.pbf file.

    Args:
        filepath: path to `.osm.pbf` file to read.
        handle: Short moniker for the site.

    Returns:
        Unconnected locations graph L.
    '''
    if isinstance(filepath, str):
        filepath = Path(filepath)
    assert ['.osm', '.pbf'] == filepath.suffixes[-2:], \
        'Argument `filepath` does not have `.osm.pbf` extension.'
    name = filepath.stem[:-4]

    first_pass = GetRelations()
    file = osmium.io.File(filepath)
    first_pass.apply_file(file, idx='flex_mem')
    getter = GetAllData(first_pass.borderIDs, first_pass.obstacleIDs)
    getter.apply_file(file, locations=True, idx='flex_mem')

    T = len(getter.turbines)
    R = len(getter.substations)
    if T == 0 or R == 0:
        raise ValueError(f'Location: "{name}" -> Unable to identify at least one substation and one generator.')
    node_latlon = {(node.y, node.x): i for i, node in
                   enumerate(getter.turbines)}
    node_latlon.update({(node.y, node.x): i for i, node in
                        enumerate(getter.substations, start=-R)})

    i = T
    border_latlon = []
    border = []
    for latlon in map(tuple, getter.border.coords.__array__()[:-1, ::-1].tolist()):
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
    for obstacle_entry in getter.obstacles:
        obstacle_latlon = []
        obstacle = []
        for latlon in map(tuple, obstacle_entry.coords.__array__()[:-1, ::-1].tolist()):
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
    latlon = np.array(tuple(chain(
        ((n.y, n.x) for n in getter.turbines),
        border_latlon,
        obstacles_latlon,
        ((n.y, n.x) for n in getter.substations),
        )), dtype=float
    )

    # TODO: find the UTM sector that includes the most coordinates among
    # vertices and boundary (bin them in 6° sectors and count). Then insert
    # a dummy coordinate as the first array row, such that utm.from_latlon()
    # will pick the right zone for all points.
    VertexC = np.c_[utm.from_latlon(*latlon.T)[:2]]

    L = L_from_site(
            T=T, R=R,
            VertexC=VertexC,
            name=name,
            handle=handle,
            )
    if getter.border is not None:
        L.graph['border'] = np.array(border, dtype=np.int_)
        if obstacles:
            L.graph['obstacles'] = [np.array(obstacle, dtype=np.int_)
                                    for obstacle in obstacles]
        # landscape_angle calculation
        border_utm = shp.Polygon(shell=VertexC[border])
        x, y = border_utm.minimum_rotated_rectangle.exterior.coords.xy
        side0 = np.hypot(x[1] - x[0], y[1] - y[0])
        side1 = np.hypot(x[2] - x[1], y[2] - y[1])
        if side0 < side1:
            angle = np.arctan2((x[1] - x[0]), (y[1] - y[0])).item()
        else:
            angle = np.arctan2((x[2] - x[1]), (y[2] - y[1])).item()
        if abs(angle) > np.pi/2:
            angle += np.pi if angle < 0 else -np.pi
        L.graph['landscape_angle'] = 180*angle/np.pi

    L.graph['B'] = B

    if getter.plant_name is not None:
        L.graph['OSM_name'] = getter.plant_name
    elif first_pass.plant_name is not None:
        L.graph['OSM_name'] = first_pass.plant_name

    return L


_site_handles_yaml = dict(
    anholt='Anholt',
    borkum='Borkum Riffgrund 1',
    borkum2='Borkum Riffgrund 2',
    borssele='Borssele',
    butendiek='Butendiek',
    dantysk='DanTysk',
    doggerA='Dogger Bank A',
    dudgeon='Dudgeon',
    anglia='East Anglia ONE',
    gode='Gode Wind 1',
    gabbin='Greater Gabbard Inner',
    gwynt='Gwynt y Mor',
    hornsea='Hornsea One',
    hornsea2w='Hornsea Two West',
    horns='Horns Rev 1',
    horns2='Horns Rev 2',
    horns3='Horns Rev 3',
    london='London Array',
    moray='Moray East',
    moraywest='Moray West',
    ormonde='Ormonde',
    race='Race Bank',
    rampion='Rampion',
    rødsand2='Rødsand 2',
    thanet='Thanet',
    triton='Triton Knoll',
    walney1='Walney 1',
    walney2='Walney 2',
    walneyext='Walney Extension',
    sands='West of Duddon Sands',
    yi_2019='Yi-2019',
    cazzaro_2022='Cazzaro-2022',
    cazzaro_2022G140='Cazzaro-2022G-140',
    cazzaro_2022G210='Cazzaro-2022G-210',
    taylor_2023='Taylor-2023',
)

_site_handles_pbf = dict(
    amalia='Princess Amalia',
    amrumbank='Amrumbank West',
    arkona='Arkona',
    baltic2='Baltic 2',
    bard='BARD Offshore 1',
    beatrice='Beatrice',
    belwind='Belwind',
    binhainorthH2='SPIC Binhai North H2',
    bodhi='Laoting Bodhi Island',
    eagle='Baltic Eagle',
    fecamp='Fecamp',
    gemini1='Gemini 1',
    gemini2='Gemini 2',
    glotech1='Global Tech 1',
    hohesee='Hohe See',
    jiaxing1='Zhejiang Jiaxing 1',
    kfA='Kriegers Flak A',
    kfB='Kriegers Flak B',
    kustzuid='Hollandse Kust Zuid',
    lillgrund='Lillgrund',
    lincs='Lincs',
    meerwind='Meerwind',
    merkur='Merkur',
    nanpeng='CECEP Yangjiang Nanpeng Island',
    neart='Neart na Gaoithe',
    nordsee='Nordsee One',
    northwind='Northwind',
    nysted='Nysted',
    robin='Robin Rigg',
    rudongdemo='CGN Rudong Demonstration',
    rudongH10='Rudong H10',
    rudongH6='Rudong H6',
    rudongH8='Rudong H8',
    sandbank='Sandbank',
    seagreen='Seagreen',
    shengsi2='Shengsi 2',
    sheringham='Sheringham Shoal',
    vejamate='Veja Mate',
    wikinger='Wikinger',
    brieuc='Saint-Brieuc',
    nazaire='Saint-Nazaire',
    riffgat='Riffgat',
    humber='Humber Gateway',
    rough='Westermost Rough',
    bucht='Deutsche Bucht',
    nordseeost='Nordsee Ost',
    kaskasi='Kaskasi',
    albatros='Albatros',
    luchterduinen='Luchterduinen',
    norther='Norther',
    mermaid='Mermaid',
    rentel='Rentel',
    triborkum='Trianel Windpark Borkum',
    galloper='Galloper Inner',
)

_site_handles = _site_handles_yaml | _site_handles_pbf

_READERS = {
        '.yaml': L_from_yaml,
        '.osm.pbf': L_from_pbf,
        }


def load_repository(handles2names=(
        ('.yaml', _site_handles_yaml),
        ('.osm.pbf', _site_handles_pbf),
        )) -> NamedTuple:
    base_dir = files(__package__ + '.data')
    if isinstance(handles2names, dict):
        # assume all files have .yaml extension
        return namedtuple('SiteRepository', handles2name)(
            *(L_from_yaml(base_dir / fname, handle)
              for handle, fname in handles2names.items()))
    elif isinstance(handles2names, Iterable):
        # handle multiple file extensions
        return namedtuple('SiteRepository',
                          reduce(lambda a, b: a | b,
                                 (m for _, m in handles2names)))(
            *sum((tuple(_READERS[ext](base_dir / (fname + ext), handle)
                        for handle, fname in handle2name.items())
                 for ext, handle2name in handles2names),
                 tuple()))
