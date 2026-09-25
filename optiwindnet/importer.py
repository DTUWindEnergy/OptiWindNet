# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import logging
import math
import re
from collections import Counter, namedtuple
from collections.abc import Iterator, Sequence
from decimal import Decimal
from fractions import Fraction
from importlib.resources import files
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import esy.osm.pbf
import networkx as nx
import numpy as np
import shapely as shp
import utm
import yaml
from scipy.spatial import ConvexHull

from .converting import L_from_site
from .geometric import rotating_calipers
from .loads import _validated_powers, set_turbine_powers
from .utils import make_handle

_lggr = logging.getLogger(__name__)
_info, _warn = _lggr.info, _lggr.warning

__all__ = (
    'L_from_pbf', 'L_from_windIO', 'L_from_yaml',
    'LocationsRepository', 'load_repository',
)  # fmt: skip


_coord_sep = r',\s*|;\s*|\s{1,}|,|;'

# OSM's generator:output:electricity holds a number followed by an optional unit
# (e.g. '8 MW'), or a non-numeric value such as 'yes', which conveys no power.
_power_output = re.compile(r'\s*(?P<number>\d+(?:\.\d*)?|\.\d+)\s*(?P<unit>\S.*?)?\s*$')
_coord_lbraces = '(['
_coord_rbraces = ')]'


def _get_entries(entries) -> Iterator[tuple[str | None, str, str]]:
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
    for label, lat, lon in _get_entries(entry_list):
        latlon = []
        for ll in (lat, lon):
            # reset per component so minutes/seconds never leak between
            # coordinates (e.g. a "D°M'S\"" latitude into a "D°" longitude)
            min = sec = 0.0
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
        # (label, latitude, longitude) in signed decimal degrees
        translated.append((label, *latlon))
    return translated


def _utm_zone_tally(latlon):
    """Tally of ``(zone_number, zone_letter)`` over ``(latitude, longitude)`` pairs.

    The most common zone (``tally.most_common(1)[0][0]``) is used as the single
    projection zone for the whole location.
    """
    return Counter(
        (utm.latlon_to_zone_number(lat, lon), utm.latitude_to_zone_letter(lat))
        for lat, lon in latlon
    )


def _parser_latlon(entry_list, force_zone_number=None, force_zone_letter=None):
    labels, lats, lons = zip(*_translate_latlonstr(entry_list))
    # project all points into a single (optionally forced) UTM zone
    eastings, northings, *_ = utm.from_latlon(
        np.array(lats),
        np.array(lons),
        force_zone_number=force_zone_number,
        force_zone_letter=force_zone_letter,
    )
    return np.c_[eastings, northings], (labels if any(labels) else ())


def _parser_planar(entry_list, force_zone_number=None, force_zone_letter=None):
    # planar coordinates carry no UTM zone; the force_zone_* args are ignored
    labels = []
    coords = []
    for label, easting, northing in _get_entries(entry_list):
        labels.append(label)
        coords.append((float(easting), float(northing)))
    return np.array(coords, dtype=float), (labels if any(labels) else ())


coordinate_parser = {
    'latlon': _parser_latlon,
    'planar': _parser_planar,
}


def _entry_power_MW(entry: dict, name: str) -> float:
    """Return a TURBINE entry's ``power_MW`` as a positive finite number.

    Raises:
        ValueError: the entry's ``power_MW`` is not one.
    """
    power_MW = entry['power_MW']
    if (
        isinstance(power_MW, bool)
        or not isinstance(power_MW, (int, float))
        or not math.isfinite(power_MW)
        or power_MW <= 0
    ):
        raise ValueError(
            f'Location: "{name}" -> a TURBINE power_MW must be a positive finite'
            f' number: got {power_MW!r}.'
        )
    return float(power_MW)


def _turbine_power_by_prefix(
    entries: list[dict], T: int, name: str, labels: Sequence[str | None]
) -> list[float]:
    """Assign turbine powers by the prefix of their ``TURBINES`` label.

    Raises:
        ValueError: labels are missing, a prefix is not a non-empty string or
            matches no turbine, ``qty`` disagrees with the matches, a
            ``power_MW`` is not a positive finite number, or a turbine is claimed
            zero or twice.
    """
    if len(labels) < T or any(labels[t] is None for t in range(T)):
        raise ValueError(
            f'Location: "{name}" -> a TURBINE prefix claims turbines by their'
            ' TURBINES label, which this location does not give to all of them.'
        )
    turbine_power = [0.0] * T
    claimed_by: list[str] = [''] * T
    for entry in entries:
        prefix = entry['prefix']
        if not isinstance(prefix, str) or not prefix:
            raise ValueError(
                f'Location: "{name}" -> a TURBINE prefix must be a non-empty'
                f' string: got {prefix!r}.'
            )
        matched = [t for t in range(T) if str(labels[t]).startswith(prefix)]
        if not matched:
            raise ValueError(
                f'Location: "{name}" -> the TURBINE prefix {prefix!r} matches no'
                ' turbine label.'
            )
        qty = entry.get('qty')
        if qty is not None and qty != len(matched):
            raise ValueError(
                f'Location: "{name}" -> the TURBINE prefix {prefix!r} matches'
                f' {len(matched)} turbines, but its qty states {qty!r}.'
            )
        power_MW = _entry_power_MW(entry, name)
        for t in matched:
            if claimed_by[t]:
                raise ValueError(
                    f'Location: "{name}" -> turbine {labels[t]!r} is claimed by'
                    f' both the TURBINE prefix {claimed_by[t]!r} and {prefix!r}.'
                )
            turbine_power[t] = power_MW
            claimed_by[t] = prefix
    unclaimed = [labels[t] for t in range(T) if not claimed_by[t]]
    if unclaimed:
        raise ValueError(
            f'Location: "{name}" -> no TURBINE prefix claims {len(unclaimed)} of'
            f' the {T} turbines, starting with {unclaimed[0]!r}.'
        )
    return turbine_power


def _turbine_power_from_spec(
    turbine: object, T: int, name: str, labels: Sequence[str | None] = ()
) -> list[Fraction] | None:
    """Read the turbines' declared power from the YAML ``TURBINE`` section.

    A mapping supplies one ``power_MW`` for all turbines. List entries assign
    power by consecutive ``qty`` blocks or label ``prefix``. Partial power
    declarations are ignored with a warning.

    Args:
        turbine: the file's ``TURBINE`` section, or None if it has none.
        T: number of turbines in the location.
        name: location name, used in the diagnostic messages.
        labels: the terminals' ``TURBINES`` labels, empty where absent.

    Returns:
        Declared power of each turbine in MW, in terminal order, or None if power
        is unspecified or incomplete.

    Raises:
        ValueError: quantities or prefixes do not cover each turbine exactly
            once, or power is not positive and finite.
    """
    if isinstance(turbine, dict):
        if turbine.get('power_MW') is None:
            return None
        turbine_power = [_entry_power_MW(turbine, name)] * T
    elif isinstance(turbine, (list, tuple)):
        entries = [entry if isinstance(entry, dict) else {} for entry in turbine]
        declared = [entry for entry in entries if entry.get('power_MW') is not None]
        if not declared:
            # a list of makes, models and quantities declares no power
            return None
        if len(declared) < len(entries):
            _warn(
                'Location "%s" -> ignoring the power of %d turbine models, as the'
                ' other %d do not declare theirs.',
                name,
                len(declared),
                len(entries) - len(declared),
            )
            return None
        by_prefix = [entry for entry in entries if entry.get('prefix') is not None]
        if by_prefix:
            if len(by_prefix) < len(entries):
                raise ValueError(
                    f'Location: "{name}" -> either every TURBINE entry claims its'
                    ' turbines by prefix or none does, and'
                    f' {len(entries) - len(by_prefix)} of {len(entries)} carry no'
                    ' prefix.'
                )
            turbine_power = _turbine_power_by_prefix(entries, T, name, labels)
        else:
            turbine_power = []
            for entry in entries:
                qty = entry.get('qty')
                if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
                    raise ValueError(
                        f'Location: "{name}" -> every TURBINE entry declaring a'
                        f' power_MW must state the qty of turbines of that model'
                        f' as a positive integer: got {qty!r}.'
                    )
                turbine_power.extend([_entry_power_MW(entry, name)] * qty)
            if len(turbine_power) != T:
                raise ValueError(
                    f'Location: "{name}" -> the TURBINE quantities add up to'
                    f' {len(turbine_power)} turbines, but the location has {T}.'
                )
    else:
        return None
    return _validated_powers(turbine_power, T)


def L_from_yaml(
    filepath: Path | str, handle: str | None = None, read_powers: bool = True
) -> nx.Graph:
    """Import wind farm data from .yaml file.

    Two options available for ``COORDINATE_FORMAT``: ``'planar'`` and ``'latlon'``.

    Format ``'planar'`` is: ``[label] easting northing``. Example::

      LABEL 234.2 5212.5

    Format ``'latlon'`` is: [label] latitude longitude. Example::

      LABEL1 11°22.333'N 44°55.666'E
      LABEL2 11.3563°N 44.8903°E
      LABEL3 11°22'20"N 44°55'40"E

    The [label] is optional. Ensure no spaces within a latitude or longitude.

    The coordinate pair may be separated by ``','`` or ``';'`` and may be enclosed in
    ``'[]'`` or ``'()'``. Example::

      LABEL [234.2, 5212.5]

    The optional ``TURBINE`` section declares the turbines' nominal power in
    MW (see :func:`~optiwindnet.loads.set_turbine_powers`), with graph
    attribute ``'power_unit'`` set to ``'MW'``. A mapping states one
    ``power_MW`` for the whole site; a
    list states one entry per turbine model, each claiming its turbines by
    ``qty``, as a block of consecutive terminals, or by a label ``prefix``::

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

    Args:
      filepath: path to ``.yaml`` file to read.
      handle: Short moniker for the site.
      read_powers: whether to declare the ``TURBINE`` section's power on the
        location; if False, the section is ignored.

    Returns:
      Unconnected location graph L.

    Raises:
      ValueError: a ``TURBINE`` list entry has neither a positive integer
        ``qty`` nor a ``prefix``, the quantities do not add up to the number of
        turbines, the prefixes do not claim every turbine exactly once, or a
        ``power_MW`` is not a positive finite number.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    # read wind power plant site YAML file
    with open(filepath, encoding='utf8') as f:
        parsed_dict = yaml.safe_load(f)
    name = filepath.stem
    handle = parsed_dict.get('HANDLE')
    if handle is None:
        handle = make_handle(name)
    # default format is "latlon"
    format = parsed_dict.get('COORDINATE_FORMAT', 'latlon')
    # Pick the UTM zone holding the most turbines and project every coordinate
    # into it, so the bulk of the layout has minimal projection distortion. The
    # zone is stored so VertexC can be reverse-projected to lat/lon.
    zone_number = zone_letter = None
    if format == 'latlon':
        turbines_latlon = _translate_latlonstr(parsed_dict['TURBINES'])
        zone_tally = _utm_zone_tally((lat, lon) for _label, lat, lon in turbines_latlon)
        (zone_number, zone_letter), _ = zone_tally.most_common(1)[0]
    Border, _BorderLabel = coordinate_parser[format](
        parsed_dict['EXTENTS'], zone_number, zone_letter
    )
    Root, RootLabel = coordinate_parser[format](
        parsed_dict['SUBSTATIONS'], zone_number, zone_letter
    )
    Terminal, TerminalLabel = coordinate_parser[format](
        parsed_dict['TURBINES'], zone_number, zone_letter
    )
    T = Terminal.shape[0]
    R = Root.shape[0]
    vertex_xy = {xy: i for i, xy in enumerate(map(tuple, Terminal.tolist()))}
    vertex_xy.update(
        {xy: i for i, xy in enumerate(map(tuple, Root.tolist()), start=-R)}
    )
    i = T
    border_xy_ = []
    border = []
    for xy in map(tuple, Border.tolist()):
        j = vertex_xy.get(xy)
        if j is None:
            border_xy_.append(xy)
            border.append(i)
            vertex_xy[xy] = i
            i += 1
        else:
            if j >= T:
                _warn(
                    'Repeated EXTENTS vertex detected: %s. This is not supported for a '
                    "location's border. Skipping this vertex, please fix the file: %s.",
                    xy,
                    filepath,
                )
                continue
            border.append(vertex_xy[xy])
    B = len(border_xy_)
    optional = {}
    obstacles = parsed_dict.get('OBSTACLES')
    obstacleC_ = []
    if obstacles is not None:
        # obstacle has to be a list of arrays, so parsing is a bit different
        indices = []
        for obstacle_entry in parsed_dict['OBSTACLES']:
            obstacleC, _poly_tag = coordinate_parser[format](
                obstacle_entry, zone_number, zone_letter
            )

            obstacle_xy_ = []
            obstacle = []
            for xy in map(tuple, obstacleC.tolist()):
                if xy not in vertex_xy:
                    obstacle_xy_.append(xy)
                    obstacle.append(i)
                    vertex_xy[xy] = i
                    i += 1
                else:
                    obstacle_xy_.append(vertex_xy[xy])
            B += len(obstacle_xy_)

            indices.append(np.array(obstacle, dtype=np.int_))
            obstacleC_.extend(obstacle_xy_)
        optional['obstacles'] = indices

    VertexC = np.vstack((Terminal, *border_xy_, *obstacleC_, Root))

    lsangle = parsed_dict.get('LANDSCAPE_ANGLE')
    if lsangle is not None:
        optional['landscape_angle'] = lsangle

    # store the UTM zone needed to reverse-project VertexC back to lat/lon via
    # utm.to_latlon(easting, northing, utm_zone_number, utm_zone_letter)
    if zone_number is not None:
        optional['utm_zone_number'] = zone_number
        optional['utm_zone_letter'] = zone_letter

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
    if read_powers:
        turbine_powers = _turbine_power_from_spec(
            parsed_dict.get('TURBINE'), T, name, TerminalLabel
        )
        if turbine_powers is not None:
            set_turbine_powers(G, turbine_powers, 'MW')
    return G


def _turbine_power_from_tags(
    tag_values: list[str | None], name: str
) -> tuple[list[Fraction], str] | None:
    """Read the generators' declared power from OSM electricity-output tags.

    All generators must declare positive output in a common unit, since a partial
    declaration leaves the remaining turbines without a demand to compare
    against. Missing or nonnumeric tags cause the power declaration to be
    ignored.

    Args:
        tag_values: ``generator:output:electricity`` value of each turbine, in
            terminal order, with ``None`` wherever the tag is absent.
        name: location name, used in the diagnostic messages.

    Returns:
        Declared power of each generator, in terminal order, and the unit string
        (empty if absent), or None if power is unspecified or incomplete.

    Raises:
        ValueError: output units differ between generators.
    """
    parsed = []
    unusable = Counter()
    for value in tag_values:
        match = None if value is None else _power_output.match(value)
        if match is not None and Decimal(match['number']) <= 0:
            match = None
        if match is None and value is not None:
            unusable[value] += 1
        parsed.append(match)
    if unusable:
        _info(
            'Location "%s" -> unusable generator output: %s',
            name,
            ', '.join(f'{value} ({count}x)' for value, count in unusable.most_common()),
        )
    declared = [match for match in parsed if match is not None]
    if not declared:
        return None
    if len(declared) < len(parsed):
        _warn(
            'Location "%s" -> ignoring the electricity output of %d generators,'
            ' as the other %d do not declare theirs.',
            name,
            len(declared),
            len(parsed) - len(declared),
        )
        return None
    units = {match['unit'] or '' for match in declared}
    if len(units) > 1:
        raise ValueError(
            f'Location: "{name}" -> generators declare their electricity output'
            f' in inconsistent units: {", ".join(sorted(units))}.'
        )
    return [Fraction(match['number']) for match in declared], units.pop()


def L_from_pbf(
    filepath: Path | str, handle: str | None = None, read_powers: bool = True
) -> nx.Graph:
    """Import wind farm data from .osm.pbf file.

    Generators tagged with ``generator:output:electricity`` declare their
    nominal power (see :func:`~optiwindnet.loads.set_turbine_powers`) and
    set the graph attribute ``'power_unit'``, provided every generator declares
    a positive output in a common unit.

    Args:
        filepath: path to ``.osm.pbf`` file to read.
        handle: Short moniker for the site.
        read_powers: whether to declare the generators' tagged output on the
            location; if False, the tags are ignored.

    Returns:
        Unconnected location graph L.

    Raises:
        ValueError: no substation or generator was found, more than one border
            was defined, or the generators declare their electricity output in
            inconsistent units.
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
    turbine_outputs = []
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
                        turbine_outputs.append(
                            e.tags.get('generator:output:electricity')
                        )
                    case _:
                        _info('Unhandled power category for Node: %s', power_kind)

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
                        _info('Generator must be Node, not Way.')
                    case None:
                        # likely to be used in a Relation
                        ways[e.id] = e
                    case _:
                        _info('Unhandled power category for Way: %s', power_kind)
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
                                                        'Only a single border'
                                                        ' is supported.'
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
                            _info(
                                'Unhandled power category for Relation: %s', power_kind
                            )

    T = len(turbines)
    R = len(substations)
    if T == 0 or R == 0:
        raise ValueError(
            f'Location: "{name}" -> Unable to identify at least one'
            ' substation and one generator.'
        )

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
    border_latlon = []
    border_list = []
    if border_raw is None:
        B = 0
    else:
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

    # Pick the UTM zone holding the most turbines (the first T rows of latlon)
    # and project every coordinate into it, so the bulk of the layout has minimal
    # projection distortion. zone_number and zone_letter are retained so the
    # projection can be reversed via utm.to_latlon().
    zone_tally = _utm_zone_tally(latlon[:T])
    (zone_number, zone_letter), _ = zone_tally.most_common(1)[0]
    eastings, northings, *_ = utm.from_latlon(
        *latlon.T, force_zone_number=zone_number, force_zone_letter=zone_letter
    )
    VertexC = np.c_[eastings, northings]

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
    declaration = (
        _turbine_power_from_tags(turbine_outputs, name) if read_powers else None
    )
    if declaration is not None:
        turbine_powers, power_unit = declaration
        set_turbine_powers(L, turbine_powers, power_unit or None)
    if border_list:
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
    L.graph['utm_zone_number'] = zone_number
    L.graph['utm_zone_letter'] = zone_letter

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
            _warn(
                'Ignoring YAML "!include" directive to unsupported file type (%s)',
                include_path,
            )
            return {}
        if not include_path.is_absolute():
            include_path = self._parent / include_path
        with open(include_path, 'r') as f:
            # When processing includes, use IncludeLoader to maintain
            # correct directory context
            return yaml.load(f, Loader=IncludeLoader)


def L_from_windIO(filepath: Path | str, handle: str | None = None) -> nx.Graph:
    """Import wind farm data from a windIO .yaml file.

    Args:
      filepath: path to windIO ``.yaml`` file to read.
      handle: Short moniker for the site.

    Returns:
      Unconnected location geometry L.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    name = filepath.stem
    system = yaml.load(filepath.open(), Loader=IncludeLoader)
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
        border=np.arange(T, T + B) if (borderC is not None and B >= 3) else None,
        name=name,
        handle=handle,
    )
    return L


if TYPE_CHECKING:

    class LocationsRepository(tuple[nx.Graph, ...]):
        """Locations addressable both by position and by handle.

        The real object is a ``namedtuple`` whose field names come from the
        loaded files, so they are unknown until run time; this declaration
        lets type checkers accept any handle as an attribute.
        """

        def __getattr__(self, handle: str) -> nx.Graph: ...
else:
    LocationsRepository = tuple


def load_repository(
    path: Path | str | None = None, read_powers: bool = True
) -> 'LocationsRepository':
    """Load locations from files of known formats into a namedtuple.

    Each file (.yaml or .osm.pbf) is translated into a location graph and
    included as an attribute in the returned namedtuple. The attribute name
    can be specified in the .yaml with the field ``HANDLE`` or in the .osm.pbf
    file with the tag ``handle`` applied to the power plant object.

    Args:
      path: Path to look for location files (non-recursive). If omited, the
        locations included in optiwindnet are loaded.
      read_powers: whether to declare the turbine power the files state (see
        :func:`L_from_yaml` and :func:`L_from_pbf`).
    Returns:
      Named tuple which has the location handles as attribute identifiers.
    """
    if path is None:
        # `__package__` is set for any imported submodule (PEP 366), but its declared
        # type allows None, which `files()` rejects on the minimum supported Python
        anchor = __package__ or __name__.rpartition('.')[0]
        # the bundled data always lives on the filesystem
        root = Path(str(files(anchor) / 'data'))
    else:
        root = Path(path)
    locations = [
        L_from_yaml(file, read_powers=read_powers) for file in root.glob('*.yaml')
    ]
    locations.extend(
        L_from_pbf(file, read_powers=read_powers) for file in root.glob('*.osm.pbf')
    )
    handles = tuple(L.graph['handle'] for L in locations)
    # field names are only known at run time -- see LocationsRepository
    return namedtuple('Locations', handles)(*locations)  # pyrefly: ignore
