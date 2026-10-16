from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import pytest

from optiwindnet.constraints import NetworkConstraints, _resolved
from optiwindnet.transforming import as_normalized
from optiwindnet.types import FeederRoute, Topology


def test_defaults_leave_only_the_capacity_minimum():
    nc = NetworkConstraints()
    assert nc.topology is Topology.BRANCHED
    assert nc.feeder_route is FeederRoute.SEGMENTED
    b = nc._resolve(T=20, R=1, capacity=5)
    assert (b.feeders_lb, b.feeders_ub) == (4, None)
    assert not b.has_root_bounds
    assert (b.group_load_lb, b.group_load_ub) == (None, None)


def test_is_immutable_hashable_and_coerces_strings():
    nc = NetworkConstraints(topology='radial', max_substation_feeders=[3, None])
    assert nc.topology is Topology.RADIAL
    assert nc.max_substation_feeders == (3, None)
    assert nc == NetworkConstraints('radial', max_substation_feeders=(3, None))
    assert hash(nc) == hash(
        NetworkConstraints('radial', max_substation_feeders=(3, None))
    )
    with pytest.raises(FrozenInstanceError):
        nc.balanced = True  # pyrefly: ignore[read-only]


@pytest.mark.parametrize(
    'kwargs, bounds',
    [
        ({'feeders': 'min'}, (4, 4)),
        ({'feeders': 6}, (6, 6)),
        ({'feeders': 'min+1'}, (5, 5)),
        ({'max_feeders': 'min+2'}, (4, 6)),
        ({'max_feeders': 7}, (4, 7)),
    ],
)
def test_farm_counts(kwargs, bounds):
    b = NetworkConstraints(**kwargs)._resolve(T=20, R=1, capacity=5)
    assert (b.feeders_lb, b.feeders_ub) == bounds


@pytest.mark.parametrize(
    'kwargs, match',
    [
        ({'feeders': 3}, 'below the minimum'),
        ({'max_feeders': 21}, 'above the maximum'),
    ],
)
def test_farm_counts_out_of_range(kwargs, match):
    with pytest.raises(ValueError, match=match):
        NetworkConstraints(**kwargs)._resolve(T=20, R=1, capacity=5)


@pytest.mark.parametrize(
    'kwargs',
    [
        {'feeders': 4, 'max_feeders': 5},
        {'feeders': 0},
        {'feeders': 'minimum'},
        {'max_feeders': 'min-1'},
        {'max_substation_feeders': -1},
        {'max_substation_loads': (0, None)},
        {'substation_feeders': 2, 'max_substation_feeders': 3},
        {'substation_feeders': (2, None), 'max_substation_feeders': (3, None)},
    ],
)
def test_invalid_forms(kwargs):
    with pytest.raises(ValueError):
        NetworkConstraints(**kwargs)


def test_exact_and_maximum_on_different_substations():
    nc = NetworkConstraints(
        substation_feeders=(2, None), max_substation_feeders=(None, 3)
    )
    b = nc._resolve(T=20, R=2, capacity=5)
    assert b.root_feeders_lb == (2, 0)
    assert b.root_feeders_ub == (2, 3)
    assert (b.feeders_lb, b.feeders_ub) == (4, 5)


@pytest.mark.parametrize('kwargs', [{'feeders': 5}, {'max_feeders': 'min+1'}])
def test_ringed_counts_are_even(kwargs):
    with pytest.raises(ValueError, match='even'):
        NetworkConstraints(topology='ringed', **kwargs)


def test_ringed_minimum_counts_both_feeders_of_a_ring():
    b = NetworkConstraints(topology='ringed', feeders='min+2')._resolve(
        T=20, R=1, capacity=5
    )
    assert b.feeders_per_group == 2
    assert (b.feeders_lb, b.feeders_ub) == (6, 6)
    assert b.groups_lb == 3


def test_substation_values_broadcast_and_need_one_entry_per_substation():
    b = NetworkConstraints(max_substation_feeders=3)._resolve(T=20, R=2, capacity=5)
    assert b.root_feeders_ub == (3, 3)
    assert (b.feeders_lb, b.feeders_ub) == (4, 6)
    with pytest.raises(ValueError, match='3 substations'):
        NetworkConstraints(max_substation_feeders=(3, 3))._resolve(
            T=20, R=3, capacity=5
        )


def test_single_substation_limits_bound_the_farm():
    b = NetworkConstraints(max_substation_feeders=5)._resolve(T=20, R=1, capacity=5)
    assert (b.feeders_lb, b.feeders_ub) == (4, 5)


def test_substation_counts_narrow_the_farm_range():
    b = NetworkConstraints(substation_feeders=(3, 3), max_feeders=8)._resolve(
        T=20, R=2, capacity=5
    )
    assert (b.feeders_lb, b.feeders_ub) == (6, 6)


@pytest.mark.parametrize(
    'kwargs, match',
    [
        ({'substation_feeders': (3, 3), 'feeders': 5}, 'outside the range'),
        ({'substation_feeders': (1, 1), 'max_feeders': 8}, 'hold a load'),
        ({'max_substation_feeders': (1, 1)}, 'hold a load'),
        ({'max_substation_loads': (5, 5)}, 'hold a load'),
        (
            {'substation_feeders': (3, None), 'max_substation_loads': (2, None)},
            'Substation -2 needs 3',
        ),
    ],
)
def test_incompatible_substation_limits(kwargs, match):
    with pytest.raises(ValueError, match=match):
        NetworkConstraints(**kwargs)._resolve(T=20, R=2, capacity=5)


def test_substation_loads_in_inflow_and_in_power():
    nc = NetworkConstraints(max_substation_loads=(12, None))
    assert nc._resolve(T=20, R=2, capacity=5).root_load_ub == (12, None)
    nc = NetworkConstraints(max_substation_loads=(40.0, None))
    b = nc._resolve(T=20, R=2, capacity=5, power_per_inflow=Fraction(3))
    assert b.root_load_ub == (13, None)
    with pytest.raises(ValueError, match='integers'):
        NetworkConstraints(max_substation_loads=(12.5, None))._resolve(
            T=20, R=2, capacity=5
        )


def test_relative_counts_need_equal_turbine_power():
    inflow = (2,) * 6 + (1,) * 14  # 26 in all
    with pytest.raises(ValueError, match='relative'):
        NetworkConstraints(feeders='min')._resolve(T=20, R=1, capacity=5, inflow=inflow)
    b = NetworkConstraints(max_feeders=8)._resolve(T=20, R=1, capacity=5, inflow=inflow)
    assert (b.feeders_lb, b.feeders_ub) == (6, 8)
    assert (b.inflow_total, b.groups_min) == (26, 6)
    assert b.nonunit_inflow == dict.fromkeys(range(6), 2)


def test_balanced_needs_a_fixed_count():
    with pytest.raises(ValueError, match='fixed feeder count'):
        NetworkConstraints(balanced=True)._resolve(T=22, R=1, capacity=6)
    b = NetworkConstraints(feeders='min', balanced=True)._resolve(T=22, R=1, capacity=6)
    assert (b.group_load_lb, b.group_load_ub) == (5, 6)
    # the substation counts fix the farm count
    b = NetworkConstraints(substation_feeders=(2, 2), balanced=True)._resolve(
        T=20, R=2, capacity=5
    )
    assert (b.feeders_lb, b.feeders_ub) == (4, 4)


def test_balanced_subtrees_must_fit_the_substation_load():
    nc = NetworkConstraints(
        feeders=4, substation_feeders=(3, None), max_substation_loads=(14, None),
        balanced=True,
    )  # fmt: skip
    with pytest.raises(ValueError, match='Substation -2 cannot hold'):
        nc._resolve(T=20, R=2, capacity=5)


def test_resolved_reads_the_instance_from_the_graph():
    from .sitecache import get_bundle

    A = get_bundle('toy').A
    constraints = NetworkConstraints(feeders='min', balanced=True, capacity=5)
    assert _resolved(A, constraints) == replace(
        constraints._resolve(T=A.graph['T'], R=A.graph['R'], capacity=5),
        linkset_id=A.graph['_linkset_id'],
    )


def test_capacity_is_given_once_and_validated():
    assert NetworkConstraints(capacity_nominal=7.5).capacity_nominal == Fraction(15, 2)
    with pytest.raises(ValueError, match='either capacity or capacity_nominal'):
        NetworkConstraints(capacity=5, capacity_nominal=5)
    for capacity in (0, 2.5, True):
        with pytest.raises(ValueError, match='positive integer'):
            NetworkConstraints(capacity=capacity)  # pyrefly: ignore[bad-argument-type]


def test_resolved_requires_a_capacity():
    from .sitecache import get_bundle

    with pytest.raises(ValueError, match='need a cable capacity'):
        _resolved(get_bundle('toy').A, NetworkConstraints())


def test_bounds_derive_the_group_quantities():
    b = NetworkConstraints(topology='ringed', feeders='min+2')._resolve(
        T=20, R=1, capacity=3
    )
    assert (b.group_capacity, b.groups_min, b.groups_lb) == (6, 4, 5)
    assert b.fixed_above_min
    assert (b.inflow_total, b.inflow_, b.nonunit_inflow) == (20, (1,) * 20, {})
    assert b.power_attrs == {}


def test_resolved_takes_bounds_for_the_same_graph_only():
    from .sitecache import get_bundle

    A = get_bundle('toy').A
    bounds = _resolved(A, NetworkConstraints(capacity=5))
    assert _resolved(A, bounds) is bounds
    with pytest.raises(TypeError, match='power_rtol'):
        _resolved(A, bounds, power_rtol=0.1)
    other = get_bundle('example_location').A
    with pytest.raises(ValueError, match='another graph'):
        _resolved(other, bounds)


def test_resolved_bounds_follow_the_scale_of_the_graph():
    from .sitecache import get_bundle

    A = get_bundle('toy').A
    bounds = _resolved(A, NetworkConstraints(capacity=5, max_link_length=1e6))
    A_norm = as_normalized(A)
    rescaled = _resolved(A_norm, bounds)
    assert rescaled.link_length_ub == pytest.approx(1e6 * A_norm.graph['norm_scale'])
    assert rescaled.length_scale == A_norm.graph['norm_scale']
