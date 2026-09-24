import math
from fractions import Fraction
from itertools import combinations, product

import numpy as np
import pytest

from optiwindnet import loads
from optiwindnet.api import WindFarmNetwork
from optiwindnet.loads import (
    _inflow_of,
    _plainest_near,
    _rationalize,
    _simplest_between,
    quantize_for_capacity,
    set_turbine_powers,
    validate_terminal_power,
)

_SITE = {
    'turbinesC': np.array([[0.0, 0.0], [1.0, 0.0]]),
    'substationsC': np.array([[0.0, 1.0]]),
}


def _assert_packing_preserved(powers, capacity, rtol, result):
    """Check tolerance, rounding and packing of a quantization by brute force."""
    inflow_by_power, power_per_inflow, integer_capacity = result
    unique = sorted({p if isinstance(p, Fraction) else _rationalize(p) for p in powers})
    capacity = capacity if isinstance(capacity, Fraction) else _rationalize(capacity)
    tolerance = _rationalize(rtol)
    assert sorted(inflow_by_power) == unique
    for power, inflow in inflow_by_power.items():
        assert type(inflow) is int and 1 <= inflow <= loads._MAX_INFLOW
        assert abs(inflow * power_per_inflow - power) <= tolerance * power
        assert _inflow_of(power, power_per_inflow) == inflow
    assert integer_capacity == capacity // power_per_inflow >= 1
    # every multiset of turbines a cable could hold, plus one more turbine
    most = int(capacity // unique[0]) + 1
    for counts in product(range(most + 1), repeat=len(unique)):
        if sum(counts) > most:
            continue
        nominal = sum(c * p for c, p in zip(counts, unique))
        integral = sum(c * inflow_by_power[p] for c, p in zip(counts, unique))
        assert (nominal <= capacity) == (integral <= integer_capacity)


# --- quantization math ---


@pytest.mark.parametrize(
    ('lo_open', 'hi_open'), tuple(product((False, True), repeat=2))
)
def test_plainest_near_is_the_simplest_nearest_rational(lo_open, hi_open):
    endpoints = sorted({Fraction(n, d) for n in range(9) for d in range(1, 5)})
    for lo, hi in combinations(endpoints, 2):
        simplest = _simplest_between(lo, hi, lo_open=lo_open, hi_open=hi_open)
        for target in (lo - 1, (lo + hi) / 2, hi + 1):
            result = _plainest_near(lo, hi, target, lo_open=lo_open, hi_open=hi_open)
            candidates = [
                Fraction(n, d)
                for d in range(1, result.denominator + 1)
                for n in range(math.floor(lo * d), math.ceil(hi * d) + 1)
                if (lo < Fraction(n, d) if lo_open else lo <= Fraction(n, d))
                and (Fraction(n, d) < hi if hi_open else Fraction(n, d) <= hi)
            ]
            assert result in candidates
            assert (result.denominator, abs(result - target)) == min(
                (p.denominator, abs(p - target)) for p in candidates
            )
            assert simplest.denominator == result.denominator
    assert _plainest_near(Fraction(3, 7), Fraction(3, 7), Fraction(1)) == Fraction(3, 7)


def test_rationalize_recovers_the_written_number():
    assert _rationalize(6.6) == Fraction(33, 5)
    assert _rationalize(1 / 3) == Fraction(1, 3)


@pytest.mark.parametrize(
    ('powers', 'capacity'),
    (
        ([8.0, 9.5], 40.0),
        ([8.0, 9.5], 28.0),
        ([8.0, 9.5], 19.0),
        ([7.0, 8.25], 24.0),
        ([5.0, 6.35], 32.0),
        ([2.0, 3.3, 5.1], 12.0),
        ([1.0, 2.0000001], 4),
        ([1.0, 1.9999999], 3),
        ([Fraction(1), Fraction(5, 4)], 2.75),
        ([Fraction(1, 2), Fraction(2, 3), Fraction(1)], 2),
        ([Fraction(10**30 + 1, 10**30), Fraction(2 * (10**30 + 1), 10**30)], 3),
    ),
)
@pytest.mark.parametrize('rtol', (0.01, 0.1))
def test_quantization_honours_tolerance_rounding_and_packing(powers, capacity, rtol):
    result = quantize_for_capacity(powers, capacity, rtol)
    _assert_packing_preserved(powers, capacity, rtol, result)


def test_quantization_sweep_agrees_with_rounding_and_exact_fit():
    """Small rationals: no result is coarser in inflow than the exact common unit."""
    values = sorted({Fraction(n, d) for n in range(1, 5) for d in (1, 2, 3)})
    for powers in combinations(values, 2):
        exact_unit = Fraction(
            math.gcd(*(p.numerator for p in powers)),
            math.lcm(*(p.denominator for p in powers)),
        )
        top = max(powers)
        for rtol in (0, 0.01, 0.1, 0.4):
            for capacity in (math.ceil(top) - Fraction(1, 2), math.ceil(2 * top)):
                if capacity < top:
                    continue
                result = quantize_for_capacity(powers, capacity, rtol)
                assert all(k <= p / exact_unit for p, k in result[0].items())
                _assert_packing_preserved(powers, capacity, rtol, result)


def test_quantization_trades_tolerance_for_smaller_inflow():
    largest = math.inf
    for rtol in (0, 0.0001, 0.001, 0.01, 0.05):
        inflow_by_power, _, _ = quantize_for_capacity([5.0, 6.35], 100, rtol)
        assert max(inflow_by_power.values()) <= largest
        largest = max(inflow_by_power.values())
    assert quantize_for_capacity([5.0, 6.35], 100, 0) == (
        {Fraction(5): 100, Fraction(127, 20): 127},
        Fraction(1, 20),
        2000,
    )
    # the reference site's quantization
    assert quantize_for_capacity([9.5, 8.0, 8.0], 40) == (
        {Fraction(8): 5, Fraction(19, 2): 6},
        Fraction(27, 17),
        25,
    )


@pytest.mark.parametrize('powers', ([2.2, 3.3], [3.36, 5.6, 11.2], [0.75, 1.25, 2.5]))
def test_exact_quantization_reproduces_the_declared_floats(powers):
    inflow_by_power, power_per_inflow, _ = quantize_for_capacity(powers, 30, 0)
    assert [
        float(inflow_by_power[_rationalize(p)] * power_per_inflow) for p in powers
    ] == (powers)


def test_quantization_enforces_the_inflow_limit(monkeypatch):
    monkeypatch.setattr(loads, '_MAX_INFLOW', 3)
    loads._quantize_powers_set.cache_clear()
    try:
        assert quantize_for_capacity([2, 3], 3, 0) == (
            {Fraction(2): 2, Fraction(3): 3},
            Fraction(1),
            3,
        )
        monkeypatch.setattr(loads, '_MAX_INFLOW', 2)
        loads._quantize_powers_set.cache_clear()
        with pytest.raises(ValueError, match='at an inflow of at most 2'):
            quantize_for_capacity([2, 3], 3, 0)
    finally:
        loads._quantize_powers_set.cache_clear()


@pytest.mark.parametrize(
    ('powers', 'capacity', 'rtol', 'match'),
    (
        ([1.0, 0.0], 2, 0.01, 'must be positive'),
        ([1.0, float('nan')], 2, 0.01, 'must be finite'),
        ([1.0, 1.5], 0, 0.01, 'must be positive'),
        ([1.0, 1.5], 2, -0.1, r'power_rtol must be a number in \[0, 1\)'),
        ([1.0, 1.5], 2, 1.0, r'power_rtol must be a number in \[0, 1\)'),
        ([1.0, 3.14159265358979], 4, 0, 'raise the tolerance'),
    ),
)
def test_quantization_rejects_invalid_input(powers, capacity, rtol, match):
    with pytest.raises(ValueError, match=match):
        quantize_for_capacity(powers, capacity, rtol)


# --- declaration, validation and per-solve quantization of graphs ---


def _bare_L():
    return WindFarmNetwork(cables=2, **_SITE).L.copy()


def test_set_turbine_powers_declares_unequal_or_uniform_power():
    L = _bare_L()
    L.nodes[0]['inflow'] = 2
    set_turbine_powers(L, [1.0, 1.25], 'MW')
    assert L.graph['powers_set'] == (Fraction(1), Fraction(5, 4))
    assert L.graph['power_unit'] == 'MW'
    assert 'power_per_inflow' not in L.graph
    assert [dict(L.nodes[t]) for t in range(2)] == [
        {'kind': 'wtg', 'power': Fraction(1)},
        {'kind': 'wtg', 'power': Fraction(5, 4)},
    ]

    set_turbine_powers(L, [0.5, 0.5])
    assert L.graph['power_per_inflow'] == Fraction(1, 2)
    assert not {'powers_set', 'power_unit'} & L.graph.keys()
    assert all(dict(L.nodes[t]) == {'kind': 'wtg'} for t in range(2))

    with pytest.raises(ValueError, match='entries but T='):
        set_turbine_powers(L, [1.0])


@pytest.mark.parametrize(
    ('graph_attrs', 'node_attrs', 'match'),
    (
        ({}, {'inflow': 1.0}, "'inflow' must be a positive integer"),
        ({}, {'inflow': True}, "'inflow' must be a positive integer"),
        ({}, {'inflow': 0}, "'inflow' must be a positive integer"),
        ({}, {'power': True}, 'finite numbers'),
        ({}, {'power': float('nan')}, 'finite numbers'),
        ({'power_per_inflow': 0.5}, {}, 'positive rational'),
        ({}, {'power': 2.0}, 'quantizes to a different inflow'),
        ({'powers_set': (1, 2)}, {'power': 1}, "terminal 1 declares no 'power'"),
        ({'powers_set': (1,)}, {'power': 1}, 'at least two distinct'),
        ({'powers_set': (1, 2)}, {'power': 3}, "which 'powers_set' lacks"),
        ({'powers_set': (1, 3)}, {'power': 1, 'inflow': 2}, "declares 'inflow'"),
    ),
)
def test_validate_terminal_power_rejects_broken_conventions(
    graph_attrs, node_attrs, match
):
    L = _bare_L()
    L.graph.update(graph_attrs)
    L.nodes[0].update(node_attrs)
    if 'powers_set' in graph_attrs and match != "terminal 1 declares no 'power'":
        L.nodes[1]['power'] = graph_attrs['powers_set'][-1]
    with pytest.raises(ValueError, match=match):
        validate_terminal_power(L)


def test_validate_terminal_power_normalizes_and_flags_inexact_quantization():
    L = _bare_L()
    L.nodes[0].update(power=1.01)
    L.nodes[1].update(power=2, inflow=2)
    validate_terminal_power(L)
    assert [type(L.nodes[t]['power']) for t in range(2)] == [Fraction, Fraction]
    assert L.nodes[0]['power'] == Fraction(101, 100)
    assert L.graph['power_quantization_inexact'] is True
    L.nodes[0]['power'] = 1
    validate_terminal_power(L)
    assert L.graph['power_quantization_inexact'] is False
