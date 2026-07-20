"""Single source of truth for the locations used by the end-to-end tests.

Both ``conftest.py`` (the ``locations`` fixture) and ``update_expected_values.py``
(the golden-value generator) build their location set from this table, so the two
can never drift apart.

Each entry maps a short handle to the file that defines the site. Handles are
grouped by size so the matrix in ``matrix.py`` can pick a cheap site for the dense
option sweeps and the larger, multi-substation sites for breadth.
"""

from collections import namedtuple

from optiwindnet.importer import L_from_pbf, L_from_yaml, LocationsRepository

from . import paths

# handle -> location file name (resolved against DATA_DIR unless it lives in
# tests/locations, which only holds the tiny hand-made example).
_DATA_FILES = {
    # small, single-substation -- used for the dense per-option MILP sweeps
    'example_location': 'example_location.yaml',  # R=1, T=12  (tests/locations)
    'borkum2': 'Borkum Riffgrund 2.yaml',  # R=1, T=52
    # larger, multi-substation -- breadth for heuristics and a few MILP cases
    'hornsea': 'Hornsea One.yaml',  # R=3, T=174
    'london': 'London Array.yaml',  # R=2, T=175
    'taylor_2023': 'Taylor-2023.yaml',  # R=2, T=122
    'yi_2019': 'Yi-2019.yaml',  # R=2, T=119
    # repository locations reused by the ringed tests
    'albatros': 'Albatros.osm.pbf',  # R=1, T=16
    'neart': 'Neart na Gaoithe.osm.pbf',  # R=2, T=54
    'riffgat': 'Riffgat.osm.pbf',  # R=1, T=30
    'galloper': 'Galloper Inner.osm.pbf',  # R=1, T=38
    # remaining multi-substation locations used by the clustering tests
    'beatrice': 'Beatrice.osm.pbf',  # R=2, T=84
    'borssele': 'Borssele.yaml',  # R=2, T=173
    'cazzaro_2022G140': 'Cazzaro-2022G-140.yaml',  # R=3, T=140
    'cazzaro_2022G210': 'Cazzaro-2022G-210.yaml',  # R=3, T=210
    'coastalva': 'Coastal Virginia.osm.pbf',  # R=3, T=176
    'gwynt': 'Gwynt y Mor.yaml',  # R=2, T=160
    'kustzuid': 'Hollandse Kust Zuid.osm.pbf',  # R=2, T=139
    'morayeast': 'Moray East.yaml',  # R=3, T=100
    'moraywest': 'Moray West.yaml',  # R=2, T=60
    'race': 'Race Bank.yaml',  # R=2, T=91
    'revolution': 'Revolution.osm.pbf',  # R=2, T=65
    'sheringham': 'Sheringham Shoal.osm.pbf',  # R=2, T=88
    'triton': 'Triton Knoll.yaml',  # R=2, T=90
    'walneyext': 'Walney Extension.yaml',  # R=2, T=87
    'yunlin': 'Yunlin.osm.pbf',  # R=2, T=80
}

# handles whose YAML lives in tests/locations rather than optiwindnet/data
_IN_LOCATIONS_DIR = frozenset({'example_location'})

# convenience groupings referenced by matrix.py
SMALL_SITES = ('example_location',)
MEDIUM_SITES = ('borkum2',)
LARGE_SITES = ('hornsea', 'london', 'taylor_2023', 'yi_2019')
ALL_SITES = tuple(_DATA_FILES)


def _path_for(handle: str):
    base = paths.LOCATIONS_DIR if handle in _IN_LOCATIONS_DIR else paths.DATA_DIR
    return base / _DATA_FILES[handle]


def load_locations(handles=ALL_SITES) -> 'LocationsRepository':
    """Return a namedtuple of loaded ``L`` graphs for the requested handles."""
    loaded = {}
    for handle in handles:
        path = _path_for(handle)
        loader = L_from_pbf if path.suffixes[-2:] == ['.osm', '.pbf'] else L_from_yaml
        loaded[handle] = loader(path)
    # field names are only known at run time -- see LocationsRepository
    Locations = namedtuple('Locations', loaded.keys())  # pyrefly: ignore
    return Locations(**loaded)  # pyrefly: ignore
