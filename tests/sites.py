"""Single source of truth for the locations used by the end-to-end tests.

Both ``conftest.py`` (the ``locations`` fixture) and ``update_expected_values.py``
(the golden-value generator) build their location set from this table, so the two
can never drift apart.

Each entry maps a short handle to the YAML file that defines the site. Handles are
grouped by size so the matrix in ``matrix.py`` can pick a cheap site for the dense
option sweeps and the larger, multi-substation sites for breadth.
"""

from collections import namedtuple

from optiwindnet.importer import L_from_yaml

from . import paths

# handle -> yaml file name (resolved against DATA_DIR unless it lives in
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


def load_locations(handles=ALL_SITES):
    """Return a namedtuple of loaded ``L`` graphs for the requested handles."""
    loaded = {h: L_from_yaml(_path_for(h)) for h in handles}
    Locations = namedtuple('Locations', loaded.keys())
    return Locations(**loaded)
