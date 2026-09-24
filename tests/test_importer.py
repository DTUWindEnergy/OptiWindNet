# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

from collections import Counter
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from optiwindnet.importer import (
    L_from_pbf,
    L_from_yaml,
    _get_entries,
    _parser_planar,
    _translate_latlonstr,
    _turbine_power_from_spec,
    _turbine_power_from_tags,
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
    label, lat, lon = result[0]
    assert label is None
    assert lat == pytest.approx(55.5)
    assert lon == pytest.approx(7.5)


def test_translate_latlonstr_decimal_deg():
    entries = '55.5 7.5'
    result = _translate_latlonstr(entries)
    assert len(result) == 1
    label, lat, lon = result[0]
    assert label is None
    assert lat == pytest.approx(55.5)
    assert lon == pytest.approx(7.5)


def test_translate_latlonstr_no_minsec_leak():
    # latitude carries seconds, longitude is degrees-only: the longitude must
    # not inherit the latitude's leftover minutes/seconds
    entries = '11°0\'30"N 44.5°E'
    result = _translate_latlonstr(entries)
    _label, lat, lon = result[0]
    assert lat == pytest.approx(11 + 0.5 / 60)
    assert lon == pytest.approx(44.5)


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


# --- generator:output:electricity ---


def test_L_from_pbf_homogeneous_generator_output():
    """Equal outputs ride on the graph alone; the terminals stay bare."""
    L = L_from_pbf(paths.DATA_DIR / 'Hornsea 2.osm.pbf')

    assert L.graph['power_per_inflow'] == Fraction(8)
    assert L.graph['power_unit'] == 'MW'
    for t in range(L.graph['T']):
        assert 'inflow' not in L.nodes[t]
        assert 'power' not in L.nodes[t]


def test_L_from_pbf_unequal_generator_output():
    """Unequal outputs are declared as they are; their quantization awaits a solve."""
    L = L_from_pbf(paths.DATA_DIR / 'Trianel Windpark Borkum.osm.pbf')
    T = L.graph['T']

    assert L.graph['power_unit'] == 'MW'
    assert L.graph['powers_set'] == (Fraction(5), Fraction(127, 20))
    assert {L.nodes[t]['power'] for t in range(T)} == {Fraction(5), Fraction(127, 20)}
    assert 'power_per_inflow' not in L.graph
    assert all('inflow' not in L.nodes[t] for t in range(T))


def test_importers_skip_the_power_unless_asked():
    for L in (
        L_from_pbf(
            paths.DATA_DIR / 'Trianel Windpark Borkum.osm.pbf', read_powers=False
        ),
        L_from_yaml(paths.DATA_DIR / 'Walney Extension.yaml', read_powers=False),
    ):
        assert not {'power_unit', 'powers_set', 'power_per_inflow'} & L.graph.keys()
        assert all('power' not in L.nodes[t] for t in range(L.graph['T']))


def test_L_from_pbf_without_usable_generator_output():
    """A location whose generators declare only 'yes' carries no power."""
    L = L_from_pbf(paths.DATA_DIR / 'Norther.osm.pbf')

    assert 'power_per_inflow' not in L.graph
    assert 'power_unit' not in L.graph
    for t in range(L.graph['T']):
        assert 'inflow' not in L.nodes[t]


@pytest.mark.parametrize(
    ('tag_values', 'expected'),
    (
        (['8 MW', '8MW'], ([8, 8], 'MW')),
        (['5 MW', '6.35 MW'], ([Fraction(5), Fraction(127, 20)], 'MW')),
        (['3000000', '1500000'], ([3000000, 1500000], '')),
        ([None, None], None),
        (['yes', 'yes'], None),
        (['8 MW', None], None),
        (['8 MW', '0 MW'], None),
    ),
)
def test_turbine_power_from_tags(tag_values, expected):
    assert _turbine_power_from_tags(tag_values, 'test') == expected


def test_turbine_power_from_tags_rejects_inconsistent_units():
    with pytest.raises(ValueError, match='inconsistent units'):
        _turbine_power_from_tags(['8 MW', '8000 kW'], 'test')


# --- the TURBINE section ---


@pytest.mark.parametrize(
    ('turbine', 'T', 'expected'),
    (
        # one mapping: the single turbine model the whole site is built of
        ({'power_MW': 3.6}, 3, [Fraction(18, 5)] * 3),
        ({'power_MW': 8}, 2, [8, 8]),
        # a list: one entry per model, applied to the terminals in blocks
        (
            [{'power_MW': 7, 'qty': 2}, {'power_MW': 8.25, 'qty': 3}],
            5,
            [7, 7, 8.25, 8.25, 8.25],
        ),
        ([{'power_MW': 3.6, 'qty': 2}], 2, [Fraction(18, 5)] * 2),
        # a list of prefixes: each entry claims the labels starting with its own
        (
            [{'power_MW': 7, 'prefix': '4-'}, {'power_MW': 8.25, 'prefix': '3-'}],
            3,
            [7, 8.25, 7],
        ),
        # two entries, claimed by qty
        (
            [{'power_MW': 8, 'qty': 1}, {'power_MW': 9.5, 'qty': 2}],
            3,
            [8, 9.5, 9.5],
        ),
        # a list of makes, models and quantities declares no power
        ([{'make': 'Vestas', 'qty': 2}], 2, None),
        # ... nor does one where only some entries declare theirs
        ([{'power_MW': 7, 'qty': 1}, {'make': 'Vestas', 'qty': 1}], 2, None),
        ({'power_MW': None}, 2, None),
        ({'make': 'Vestas'}, 2, None),
        (None, 2, None),
    ),
)
def test_turbine_power_from_spec(turbine, T, expected):
    labels = ('4-A01', '3-A02', '4-A03')[:T]
    assert _turbine_power_from_spec(turbine, T, 'test', labels) == expected


@pytest.mark.parametrize('qty', (None, 0, -2, 1.5, True))
def test_turbine_power_from_spec_requires_a_quantity(qty):
    with pytest.raises(ValueError, match='positive integer'):
        _turbine_power_from_spec([{'power_MW': 7, 'qty': qty}], 1, 'test')


def test_turbine_power_from_spec_requires_the_quantities_to_add_up():
    turbine = [{'power_MW': 7, 'qty': 2}, {'power_MW': 8.25, 'qty': 3}]
    with pytest.raises(ValueError, match='add up to 5 turbines, but .* has 6'):
        _turbine_power_from_spec(turbine, 6, 'test')


_LABELS = ('3-A01', '4-A02', '4-A03')


def test_turbine_power_by_prefix_is_independent_of_the_listing_order():
    """A prefix claims its turbines wherever they sit among the terminals."""
    turbine = [
        {'power_MW': 8.25, 'qty': 1, 'prefix': '3-'},
        {'power_MW': 7, 'qty': 2, 'prefix': '4-'},
    ]
    assert _turbine_power_from_spec(turbine, 3, 'test', _LABELS) == [8.25, 7, 7]


def test_turbine_power_by_prefix_is_all_or_nothing():
    turbine = [{'power_MW': 8.25, 'prefix': '3-'}, {'power_MW': 7, 'qty': 2}]
    with pytest.raises(ValueError, match='every TURBINE entry claims its turbines'):
        _turbine_power_from_spec(turbine, 3, 'test', _LABELS)


@pytest.mark.parametrize(
    ('prefix', 'match'),
    (
        ('', 'non-empty string'),
        (5, 'non-empty string'),
        # an absent prefix is not a malformed one: it breaks the all-or-nothing
        (None, 'every TURBINE entry claims its turbines'),
    ),
)
def test_turbine_power_by_prefix_rejects_a_non_prefix(prefix, match):
    turbine = [{'power_MW': 7, 'prefix': '4-'}, {'power_MW': 8.25, 'prefix': prefix}]
    with pytest.raises(ValueError, match=match):
        _turbine_power_from_spec(turbine, 3, 'test', _LABELS)


def test_turbine_power_by_prefix_rejects_one_that_matches_nothing():
    turbine = [{'power_MW': 7, 'prefix': '4-'}, {'power_MW': 8.25, 'prefix': '9-'}]
    with pytest.raises(ValueError, match="prefix '9-' matches no turbine"):
        _turbine_power_from_spec(turbine, 3, 'test', _LABELS)


def test_turbine_power_by_prefix_rejects_a_turbine_claimed_twice():
    turbine = [{'power_MW': 7, 'prefix': '4-'}, {'power_MW': 8.25, 'prefix': '4-A02'}]
    with pytest.raises(ValueError, match="'4-A02' is claimed by both"):
        _turbine_power_from_spec(turbine, 3, 'test', _LABELS)


def test_turbine_power_by_prefix_rejects_an_unclaimed_turbine():
    turbine = [{'power_MW': 7, 'prefix': '4-'}]
    with pytest.raises(
        ValueError, match="claims 1 of the 3 turbines, starting with '3-A01'"
    ):
        _turbine_power_from_spec(turbine, 3, 'test', _LABELS)


def test_turbine_power_by_prefix_cross_checks_the_quantity():
    turbine = [
        {'power_MW': 8.25, 'qty': 1, 'prefix': '3-'},
        {'power_MW': 7, 'qty': 3, 'prefix': '4-'},
    ]
    with pytest.raises(ValueError, match="prefix '4-' matches 2 turbines, but its qty"):
        _turbine_power_from_spec(turbine, 3, 'test', _LABELS)


def test_turbine_power_by_prefix_needs_labelled_turbines():
    turbine = [{'power_MW': 7, 'prefix': '4-'}]
    with pytest.raises(ValueError, match='does not give to all of them'):
        _turbine_power_from_spec(turbine, 3, 'test', ())


@pytest.mark.parametrize('power_MW', (0, -3, 'yes', True, float('inf'), [8, 9.5]))
def test_turbine_power_from_spec_rejects_non_power(power_MW):
    """`power_MW` is a positive finite number: the spec rejects everything else."""
    with pytest.raises(ValueError, match='power_MW must be a positive finite number'):
        _turbine_power_from_spec({'power_MW': power_MW}, 2, 'test')


def test_L_from_yaml_uniform_turbine_power():
    """One turbine model rides on the graph alone; the terminals stay bare."""
    L = L_from_yaml(paths.DATA_DIR / 'Anholt.yaml')

    assert L.graph['power_per_inflow'] == Fraction(18, 5)
    assert L.graph['power_unit'] == 'MW'
    for t in range(L.graph['T']):
        assert 'inflow' not in L.nodes[t]
        assert 'power' not in L.nodes[t]


def test_L_from_yaml_mixed_turbine_power():
    """Two turbine models, claimed by the area prefix of each label."""
    L = L_from_yaml(paths.DATA_DIR / 'Walney Extension.yaml')

    assert L.graph['power_unit'] == 'MW'
    # the chart's areas 3 and 4 interleave in the listing, so the blocks the
    # quantities alone would cut are not the two phases
    power = [L.nodes[t]['power'] for t in range(L.graph['T'])]
    for t in range(L.graph['T']):
        area = L.nodes[t]['label'][:2]
        assert power[t] == (8.25 if area == '3-' else 7.0)
    assert Counter(power) == {8.25: 40, 7.0: 47}
    assert L.graph['powers_set'] == (Fraction(7), Fraction(33, 4))
    assert all('inflow' not in L.nodes[t] for t in range(L.graph['T']))


def test_L_from_yaml_turbine_blocks_follow_the_terminal_order():
    """Each block lands on the sub-area of TURBINES it accounts for."""
    L = L_from_yaml(paths.DATA_DIR / 'Borssele.yaml')

    assert L.graph['power_unit'] == 'MW'
    # Borssele I and II are 8 MW turbines, III, IV and V are 9.5 MW ones; the
    # terminals' labels carry the sub-area they belong to
    for t in range(L.graph['T']):
        sub_area = L.nodes[t]['label'].split('-')[0]
        assert L.nodes[t]['power'] == (8.0 if sub_area in ('T1', 'T2') else 9.5)


def test_L_from_yaml_without_turbine_section():
    """A location with no TURBINE section carries no power."""
    L = L_from_yaml(paths.DATA_DIR / 'Yi-2019.yaml')

    assert 'power_per_inflow' not in L.graph
    assert 'power_unit' not in L.graph
