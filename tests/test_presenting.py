# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

from fractions import Fraction

import pytest

from optiwindnet.presenting import (
    _format_inflow_as_power,
    _format_length,
    _format_nominal_power,
    describe_G,
)

from .helpers import tiny_wfn

# ----------
# tests
# ----------


def test_describe_G():
    wfn = tiny_wfn()
    G = wfn.G

    desc = describe_G(G)
    expected = ['κ = 4, T = 4', '(+0) [-1]: 1', 'Σλ = 5.5456\u00a0m', '55\u00a0€']

    assert desc == expected, f'Output mismatch:\nGot: {desc}\nExpected: {expected}'


@pytest.mark.parametrize(
    ('length', 'expected'),
    (
        (5.5456, '5.5456'),
        (1234.5678, '1_234.6'),
        (1234567.8, '1_234_568'),
    ),
)
def test_format_length_keeps_significant_digits(length, expected):
    assert _format_length(length) == expected


def test_format_inflow_as_power_keeps_integers_at_power_per_inflow_one():
    assert _format_inflow_as_power(4, 1) == '4'
    assert _format_inflow_as_power(4, Fraction(1)) == '4'


def test_format_inflow_as_power_scales_by_power_per_inflow():
    assert _format_inflow_as_power(4, Fraction(201, 200)) == '4.02'


def test_format_nominal_power_prefers_the_declared_value():
    power_per_inflow = Fraction(201, 200)
    # the declared power differs from inflow * power_per_inflow within the tolerance
    declared = {'inflow': 4, 'power': Fraction(4)}
    assert _format_nominal_power(declared, power_per_inflow) == '4'
    assert _format_nominal_power({'inflow': 4}, power_per_inflow) == '4.02'
    assert _format_nominal_power({}, power_per_inflow) == '1.005'
