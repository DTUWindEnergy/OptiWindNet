# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

import pytest

from optiwindnet.presenting import _format_length, describe_G

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
