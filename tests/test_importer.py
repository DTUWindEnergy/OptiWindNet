import networkx as nx
import numpy as np

from optiwindnet.importer import (
    L_from_yaml,
    _get_entries,
    _parser_planar,
    _translate_latlonstr,
    load_repository,
)

from . import paths


# --- _get_entries ---


def test_get_entries_string_no_labels():
    entries = '1.0 2.0\n3.0 4.0'
    result = list(_get_entries(entries))
    assert result == [(None, '1.0', '2.0'), (None, '3.0', '4.0')]


def test_get_entries_string_with_labels():
    entries = 'WTG1 1.0 2.0\nWTG2 3.0 4.0'
    result = list(_get_entries(entries))
    assert result == [('WTG1', '1.0', '2.0'), ('WTG2', '3.0', '4.0')]


def test_get_entries_string_with_braces():
    entries = '[1.0, 2.0]\n(3.0; 4.0)'
    result = list(_get_entries(entries))
    assert result == [(None, '1.0', '2.0'), (None, '3.0', '4.0')]


def test_get_entries_list_tuples():
    entries = [(1.0, 2.0), (3.0, 4.0)]
    result = list(_get_entries(entries))
    assert result == [(None, 1.0, 2.0), (None, 3.0, 4.0)]


def test_get_entries_list_with_labels():
    entries = [('A', 1.0, 2.0), ('B', 3.0, 4.0)]
    result = list(_get_entries(entries))
    assert result == [('A', 1.0, 2.0), ('B', 3.0, 4.0)]


# --- _parser_planar ---


def test_parser_planar_no_labels():
    entries = [[100.0, 200.0], [300.0, 400.0]]
    coords, labels = _parser_planar(entries)
    np.testing.assert_allclose(coords, [[100.0, 200.0], [300.0, 400.0]])
    assert labels == ()


def test_parser_planar_with_labels():
    entries = [('T1', '100.0', '200.0'), ('T2', '300.0', '400.0')]
    coords, labels = _parser_planar(entries)
    np.testing.assert_allclose(coords, [[100.0, 200.0], [300.0, 400.0]])
    assert labels == ['T1', 'T2']


# --- _translate_latlonstr ---


def test_translate_latlonstr_dms():
    entries = '55°30\'0"N 7°30\'0"E'
    result = _translate_latlonstr(entries)
    assert len(result) == 1
    label, easting, northing, zone_num, zone_letter = result[0]
    assert label is None
    assert isinstance(easting, float)
    assert isinstance(northing, float)


def test_translate_latlonstr_decimal_deg():
    entries = '55.5 7.5'
    result = _translate_latlonstr(entries)
    assert len(result) == 1
    label, easting, northing, zone_num, zone_letter = result[0]
    assert label is None
    assert isinstance(easting, float)


# --- L_from_yaml ---


def test_L_from_yaml_example_location():
    filepath = paths.LOCATIONS_DIR / 'example_location.yaml'
    L = L_from_yaml(filepath)

    assert isinstance(L, nx.Graph)
    T = L.graph['T']
    R = L.graph['R']
    assert T == 12
    assert R == 1
    assert L.graph['B'] > 0
    assert L.number_of_nodes() == T + R
    assert L.number_of_edges() == 0

    # Check node kinds
    for n in range(T):
        assert L.nodes[n]['kind'] == 'wtg'
    for r in range(-R, 0):
        assert L.nodes[r]['kind'] == 'oss'

    # Check VertexC shape
    assert L.graph['VertexC'].shape[1] == 2


def test_L_from_yaml_data_dir():
    """Test loading a latlon-format YAML from the data directory."""
    filepath = paths.DATA_DIR / 'Yi-2019.yaml'
    L = L_from_yaml(filepath)

    assert isinstance(L, nx.Graph)
    assert L.graph['T'] > 0
    assert L.graph['R'] > 0


def test_L_from_yaml_string_path():
    """Test that string paths work."""
    filepath = str(paths.LOCATIONS_DIR / 'example_location.yaml')
    L = L_from_yaml(filepath)
    assert isinstance(L, nx.Graph)
    assert L.graph['T'] == 12


# --- load_repository ---


def test_load_repository_locations_dir():
    locations = load_repository(paths.LOCATIONS_DIR)
    # There's at least one YAML in the locations dir
    assert len(locations) >= 1


def test_load_repository_default():
    """Loading the built-in data directory."""
    locations = load_repository()
    assert len(locations) > 0
    # Each location should be a graph
    for L in locations:
        assert isinstance(L, nx.Graph)
        assert 'T' in L.graph
        assert 'R' in L.graph
