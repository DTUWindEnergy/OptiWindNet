import logging
import math
import re
from fractions import Fraction
from itertools import combinations, product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.markers import MarkerStyle

from optiwindnet import loads
from optiwindnet.api import EWRouter, HGSRouter, MILPRouter, WindFarmNetwork
from optiwindnet.baselines.hgs import hgs_cvrp
from optiwindnet.baselines.lkh import lkh3
from optiwindnet.converting import G_from_S, L_from_G, S_from_G
from optiwindnet.db.storage import pack_G
from optiwindnet.heuristics.constructor import constructor
from optiwindnet.loads import (
    _inflow_of,
    _plainest_near,
    _rationalize,
    _simplest_between,
    quantize_for_capacity,
    quantized,
    set_terminal_power,
    validate_terminal_power,
)
from optiwindnet.mesh import make_planar_embedding
from optiwindnet.MILP import solver_factory
from optiwindnet.plotting import NODESIZE, gplot
from optiwindnet.presenting import _terminal_groups, describe_G
from optiwindnet.svg import _NODE_RADII, svgplot

from .helpers import solver_unavailable

_SITE = {
    'turbinesC': np.array([[0.0, 0.0], [1.0, 0.0]]),
    'substationsC': np.array([[0.0, 1.0]]),
}
# 12 turbines of 8 and 9.5 MW on a 4x3 grid, substation below the grid
_GRID = {
    'turbinesC': np.array(
        [[i * 1000.0, j * 1000.0] for i in range(4) for j in range(3)]
    ),
    'substationsC': np.array([[1500.0, -1500.0]]),
}
_GRID_POWERS = [8.0 if t % 3 else 9.5 for t in range(12)]
_GRID_CABLES = [(20, 100.0), (40, 180.0)]


def _mixed_wfn(**kwargs):
    """Two turbines of 1 and 1.25 MW, quantized to 4 and 5 inflow of 1/4 MW."""
    return WindFarmNetwork(
        cables=[(1, 100.0), (2.5, 150.0)],
        power_unit='MW',
        turbine_powers=[1.0, 1.25],
        **_SITE,
        **kwargs,
    )


def _labels(G, node_tag, backend):
    if backend == 'svg':
        text = svgplot(G, node_tag=node_tag).data
        return sorted(re.findall(r'<text[^>]*>([^<]+)</text>', text))
    fig, ax = plt.subplots(layout='constrained')
    try:
        gplot(G, ax=ax, node_tag=node_tag, infobox=False)
        return sorted(t.get_text() for t in ax.texts if t.get_text())
    finally:
        plt.close(fig)


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


def test_set_terminal_power_declares_unequal_or_uniform_power():
    L = _bare_L()
    L.nodes[0]['inflow'] = 2
    set_terminal_power(L, [1.0, 1.25], 'MW')
    assert L.graph['powers_set'] == (Fraction(1), Fraction(5, 4))
    assert L.graph['power_unit'] == 'MW'
    assert 'power_per_inflow' not in L.graph
    assert [dict(L.nodes[t]) for t in range(2)] == [
        {'kind': 'wtg', 'power': Fraction(1)},
        {'kind': 'wtg', 'power': Fraction(5, 4)},
    ]

    set_terminal_power(L, [0.5, 0.5])
    assert L.graph['power_per_inflow'] == Fraction(1, 2)
    assert not {'powers_set', 'power_unit'} & L.graph.keys()
    assert all(dict(L.nodes[t]) == {'kind': 'wtg'} for t in range(2))

    with pytest.raises(ValueError, match='entries but T='):
        set_terminal_power(L, [1.0])


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


def test_quantized_quantizes_unequal_power_on_a_copy():
    _, A = make_planar_embedding(_mixed_wfn().L)
    before = nx.to_dict_of_dicts(A), dict(A.graph), dict(A.nodes(data=True))

    A_solve, capacity, attrs = quantized(A, capacity_nominal=2.5, power_rtol=0)
    assert A_solve is not A
    assert (nx.to_dict_of_dicts(A), dict(A.graph), dict(A.nodes(data=True))) == before
    assert [A_solve.nodes[t]['inflow'] for t in range(2)] == [4, 5]
    assert A_solve.graph['power_per_inflow'] == Fraction(1, 4)
    assert A_solve.graph['power_quantization_inexact'] is False
    assert capacity == 10
    assert attrs == {
        'power_per_inflow': Fraction(1, 4),
        'capacity_nominal': Fraction(5, 2),
        'power_rtol': 0,
    }


def test_quantized_uses_uniform_power_in_place():
    wfn = WindFarmNetwork(cables=20, power_unit='MW', turbine_powers=[8, 8], **_SITE)
    A = wfn.A
    A_solve, capacity, attrs = quantized(A, capacity_nominal=Fraction(39, 2))
    assert A_solve is A
    assert capacity == 2
    assert attrs['power_per_inflow'] == 8
    # integer capacities need no declared power
    L = _bare_L()
    assert quantized(L, capacity=3) == (L, 3, {})


@pytest.mark.parametrize(
    ('mixed', 'kwargs', 'match'),
    (
        (True, {'capacity': 5}, 'pass capacity_nominal'),
        (True, {}, 'Exactly one of'),
        (True, {'capacity': 5, 'capacity_nominal': 2.5}, 'Exactly one of'),
        (True, {'capacity_nominal': 1.0}, 'below the turbine power'),
        (False, {'capacity_nominal': 2.5}, 'requires A to declare turbine power'),
        (False, {'capacity': 2.0}, 'positive integer'),
    ),
)
def test_quantized_rejects_mismatched_capacity(mixed, kwargs, match):
    A = _mixed_wfn().L if mixed else _bare_L()
    with pytest.raises(ValueError, match=match):
        quantized(A, **kwargs)


# --- WindFarmNetwork input ---


def test_nominal_input_declares_power_on_L_only():
    wfn = _mixed_wfn()
    assert wfn.power_unit == 'MW'
    assert wfn.turbine_powers == [Fraction(1), Fraction(5, 4)]
    assert wfn.cables == [(Fraction(1), 100.0), (Fraction(5, 2), 150.0)]
    assert wfn.cables_capacity == Fraction(5, 2)
    for graph in (wfn.L, wfn.A):
        assert graph.graph['powers_set'] == (Fraction(1), Fraction(5, 4))
        assert graph.graph['power_unit'] == 'MW'
        assert 'power_per_inflow' not in graph.graph
        assert all('inflow' not in graph.nodes[t] for t in range(2))


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    (
        ({'turbine_powers': [1.0]}, 'entries but T='),
        ({'turbine_powers': [1.0, 0.0]}, 'must be positive'),
        ({'turbine_powers': [1.0, float('nan')]}, 'must be finite'),
        ({'turbine_powers': [1.0, 3.0]}, 'exceeds maximum cable capacity'),
        ({'turbine_powers': None}, 'requires turbine_powers'),
    ),
)
def test_nominal_input_is_validated(kwargs, match):
    with pytest.raises(ValueError, match=match):
        WindFarmNetwork(cables=2.5, power_unit='MW', **(kwargs | _SITE))


def test_integer_turbine_powers_are_inflow_without_power_unit():
    wfn = WindFarmNetwork(cables=3, turbine_powers=[1, 2], **_SITE)
    wfn.update_from_terse_links(np.array([-1, 0]))

    assert wfn.power_unit is None
    assert wfn.turbine_powers == [1, 2]
    assert wfn.cables == [(3, 0.0)]
    for graph in (wfn.L, wfn.A, wfn.S, wfn.G, L_from_G(wfn.G), S_from_G(wfn.G)):
        assert [graph.nodes[t].get('inflow', 1) for t in range(2)] == [1, 2]
        assert all('power' not in graph.nodes[t] for t in range(2))
        assert not loads._POWER_GRAPH_ATTRS[:3] & graph.graph.keys()
    assert sorted(wfn.get_network()['load']) == [2, 3]
    assert WindFarmNetwork(cables=3, **_SITE).turbine_powers is None


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    (
        ({'turbine_powers': [1.0, 2]}, 'positive integers'),
        ({'turbine_powers': np.array([1.0, 2.0])}, 'positive integers'),
        ({'turbine_powers': [1, 0]}, 'positive integers'),
        ({'turbine_powers': [True, 1]}, 'positive integers'),
        ({'turbine_powers': [1]}, 'entries but T='),
        ({'cables': 2.0}, 'not a positive integer'),
        ({'cables': [(2.5, 1.0)]}, 'not a positive integer'),
        ({'cables': 0}, 'not a positive integer'),
    ),
)
def test_non_integers_are_rejected_without_power_unit(kwargs, match):
    with pytest.raises(ValueError, match=match):
        WindFarmNetwork(**({'cables': 2} | kwargs), **_SITE)


def test_location_declaration_meets_power_unit():
    L = _mixed_wfn().L
    with pytest.raises(ValueError, match='differs from the power_unit'):
        WindFarmNetwork(L=L.copy(), cables=2.5, power_unit='kW')

    wfn = WindFarmNetwork(L=L.copy(), cables=2.5, power_unit='MW')
    assert wfn.turbine_powers == [1, Fraction(5, 4)]

    wfn = WindFarmNetwork(
        L=L.copy(), cables=2.5, power_unit='MW', turbine_powers=[1.0, 1.0]
    )
    assert wfn.turbine_powers == [1, 1]
    assert wfn.L.graph['power_per_inflow'] == 1
    assert 'powers_set' not in wfn.L.graph
    assert all('power' not in wfn.L.nodes[t] for t in range(2))


@pytest.mark.parametrize(
    ('powers', 'message', 'level'),
    (
        ([1.0, 1.25], 'unequal turbine power', logging.WARNING),
        ([0.5, 0.5], 'uniform turbine power', logging.INFO),
    ),
)
@pytest.mark.parametrize('turbine_powers', (None, [1, 2]))
def test_location_power_is_stripped_without_power_unit(
    caplog, powers, message, level, turbine_powers
):
    L = WindFarmNetwork(
        cables=2.5, power_unit='MW', turbine_powers=powers, **_SITE
    ).L.copy()

    with caplog.at_level(logging.INFO, logger='optiwindnet.api'):
        wfn = WindFarmNetwork(L=L, cables=3, turbine_powers=turbine_powers)

    (record,) = [r for r in caplog.records if message in r.message]
    assert record.levelno == level
    assert not loads._POWER_GRAPH_ATTRS[:3] & wfn.L.graph.keys()
    assert all('power' not in wfn.L.nodes[t] for t in range(2))
    assert wfn.turbine_powers == turbine_powers
    wfn.update_from_terse_links(np.array([-1, 0]))
    assert sorted(wfn.get_network()['load']) == sorted(
        [turbine_powers[1], sum(turbine_powers)] if turbine_powers else [1, 2]
    )


# --- producers and routers ---


def test_reference_site_routes_with_hgs():
    wfn = WindFarmNetwork(
        cables=_GRID_CABLES,
        power_unit='MW',
        turbine_powers=_GRID_POWERS,
        router=HGSRouter(time_limit=1, seed=1),
        **_GRID,
    )
    wfn.optimize()
    _assert_reference_solution(wfn)


def _assert_reference_solution(wfn):
    S, G = wfn.S, wfn.G
    for graph in (S, G):
        assert graph.graph['power_per_inflow'] == Fraction(27, 17)
        assert graph.graph['capacity'] == 25
        assert graph.graph['capacity_nominal'] == 40
        assert graph.graph['power_rtol'] == loads.DEFAULT_POWER_RTOL
        assert {graph.nodes[t]['inflow'] for t in range(12)} == {5, 6}
        assert graph.graph['max_load'] <= 25
    assert all('power' not in S.nodes[t] for t in range(12))
    assert [G.nodes[t]['power'] for t in range(12)] == _GRID_POWERS
    assert G.graph['power_quantization_inexact'] is True
    # declaration only on L and A
    assert 'inflow' not in wfn.L.nodes[0] and 'inflow' not in wfn.A.nodes[0]
    network = wfn.get_network()
    feeders = network[network['tgt'] < 0]
    assert feeders['load'].sum() == pytest.approx(sum(_GRID_POWERS), abs=1e-9)
    assert network['load'].max() <= 40

    # an encoding stores no inflow: re-quantizing restores the same solution
    length = G.size(weight='length')
    wfn.update_from_terse_links(wfn.terse_links())
    assert wfn.G.graph['power_per_inflow'] == Fraction(27, 17)
    assert wfn.G.size(weight='length') == pytest.approx(length)


def test_reference_site_routes_with_milp():
    try:
        wfn = WindFarmNetwork(
            cables=_GRID_CABLES,
            power_unit='MW',
            turbine_powers=_GRID_POWERS,
            router=MILPRouter('highs', time_limit=5, mip_gap=0.01),
            **_GRID,
        )
        wfn.optimize()
    except BaseException as exc:
        if solver_unavailable(exc):
            pytest.skip('highs not available')
        raise
    _assert_reference_solution(wfn)


def test_ew_router_rejects_unequal_power_and_accepts_uniform_power():
    with pytest.raises(NotImplementedError, match='fills a subtree up to `capacity`'):
        _mixed_wfn(router=EWRouter()).optimize()

    wfn = WindFarmNetwork(
        cables=16.0, power_unit='MW', turbine_powers=[8.0, 8.0], **_SITE
    )
    wfn.optimize()
    assert wfn.G.graph['capacity'] == 2
    assert wfn.G.graph['power_per_inflow'] == 8
    assert all(
        dict(wfn.S.nodes[t]).keys() <= {'kind', 'load', 'subtree'} for t in range(2)
    )
    assert sorted(wfn.get_network()['load']) == [8.0, 16.0]


def test_producers_record_the_quantization_on_S():
    A = WindFarmNetwork(
        cables=_GRID_CABLES, power_unit='MW', turbine_powers=_GRID_POWERS, **_GRID
    ).A
    S = hgs_cvrp(A, capacity_nominal=40, time_limit=0.5, seed=2)
    assert {
        k: S.graph[k] for k in ('capacity', 'capacity_nominal', 'power_per_inflow')
    } == {
        'capacity': 25,
        'capacity_nominal': 40,
        'power_per_inflow': Fraction(27, 17),
    }
    assert S.graph['power_rtol'] == loads.DEFAULT_POWER_RTOL

    uniform = WindFarmNetwork(
        cables=40, power_unit='MW', turbine_powers=[8.0] * 12, **_GRID
    ).A
    S = constructor(uniform, capacity_nominal=40, power_rtol=0)
    assert (S.graph['capacity'], S.graph['power_per_inflow']) == (5, 8)
    assert S.graph['capacity_nominal'] == 40 and S.graph['power_rtol'] == 0


@pytest.mark.parametrize(
    ('producer', 'kwargs'),
    (
        (constructor, {}),
        (hgs_cvrp, {'ringed': True, 'time_limit': 0.1}),
        (hgs_cvrp, {'balanced': True, 'time_limit': 0.1}),
        (lkh3, {'ringed': True, 'time_limit': 0.1}),
        (lkh3, {'balanced': True, 'time_limit': 0.1}),
    ),
)
def test_heuristics_reject_unsupported_modes_with_unequal_power(producer, kwargs):
    A = _mixed_wfn().A
    with pytest.raises(NotImplementedError, match="'inflow' other than 1"):
        producer(A, capacity_nominal=2.5, **kwargs)
    with pytest.raises(ValueError, match='pass capacity_nominal'):
        producer(A, capacity=5, **kwargs)


@pytest.mark.parametrize(
    ('model_options', 'error', 'match'),
    (
        ({'topology': 'ringed'}, NotImplementedError, 'RINGED model'),
        ({'feeder_limit': 'minimum'}, ValueError, 'nonunitary terminal inflow'),
        ({'feeder_limit': 'min_plus1'}, ValueError, 'nonunitary terminal inflow'),
    ),
)
def test_milp_rejects_unsupported_models_with_unequal_power(
    model_options, error, match
):
    try:
        solver = solver_factory('highs')
    except BaseException as exc:
        if solver_unavailable(exc):
            pytest.skip('highs not available')
        raise
    wfn = _mixed_wfn()
    with pytest.raises(error, match=match):
        solver.set_problem(
            wfn.P, wfn.A, model_options=model_options, capacity_nominal=2.5
        )


def test_milp_warmstart_is_built_only_for_plain_radial_solves():
    wfn = _mixed_wfn()
    A = wfn.A
    capacity_kwargs = {'capacity_nominal': Fraction(5, 2), 'power_rtol': 0.01}
    A_solve, capacity, _ = quantized(A, **capacity_kwargs)

    def warmstart(model_options):
        router = MILPRouter(
            'highs', time_limit=1, mip_gap=0.01, model_options=model_options
        )
        return router._make_warmstart(A, capacity_kwargs, A_solve, capacity)

    S_warm = warmstart({'topology': 'radial'})
    assert S_warm is not None and S_warm.graph['max_load'] <= capacity
    assert warmstart({'topology': 'ringed'}) is None
    assert (
        warmstart({'balanced': True, 'feeder_limit': 'exactly', 'max_feeders': 2})
        is None
    )


# --- cable reassignment ---


def test_new_cables_keep_the_solution_unless_its_load_exceeds_them():
    wfn = WindFarmNetwork(
        cables=_GRID_CABLES,
        power_unit='MW',
        turbine_powers=_GRID_POWERS,
        router=HGSRouter(time_limit=0.5, seed=1),
        **_GRID,
    )
    wfn.optimize()
    L, P, A = wfn.L, wfn.P, wfn.A
    links = set(map(frozenset, wfn.G.edges))
    largest = max(wfn.get_network()['load'])
    assert 19 < largest <= 40

    wfn.cables = [(19, 100.0), (60, 200.0)]
    assert not wfn._is_stale_SG
    assert set(map(frozenset, wfn.G.edges)) == links
    assert {c for *_, c in wfn.G.edges(data='cable')} == {0, 1}
    assert wfn.G.graph['cables'] == wfn.cables

    wfn.cables = [(9.5, 100.0), (19, 180.0)]
    assert wfn._is_stale_SG
    assert wfn.L is L and wfn.P is P and wfn.A is A
    assert 'power_per_inflow' not in A.graph
    assert all('inflow' not in A.nodes[t] for t in range(12))


def test_integer_capacity_below_the_load_invalidates_the_solution():
    wfn = WindFarmNetwork(cables=2, **_SITE)
    wfn.update_from_terse_links(np.array([-1, 0]))
    wfn.cables = [(1, 1.0), (3, 2.0)]
    assert not wfn._is_stale_SG
    assert sorted(c for *_, c in wfn.G.edges(data='cable')) == [0, 1]
    wfn.cables = 1
    assert wfn._is_stale_SG


def test_nominal_cables_are_assigned_by_nominal_load():
    wfn = _mixed_wfn()
    wfn.update_from_terse_links(np.array([-1, -1]))
    # 1 MW fits the 1 MW cable; 1.25 MW needs the 2.5 MW one
    assert {v: d['cable'] for _, v, d in wfn.G.edges(data=True)} == {0: 0, 1: 1}


# --- round trips and storage ---


def test_unequal_power_round_trips_through_S_and_L():
    wfn = _mixed_wfn()
    wfn.update_from_terse_links(np.array([-1, -1]))
    G = wfn.G
    quantization = {k: G.graph[k] for k in ('power_per_inflow', 'capacity_nominal')}
    assert quantization == {
        'power_per_inflow': Fraction(1, 4),
        'capacity_nominal': Fraction(5, 2),
    }

    S = S_from_G(G)
    assert [S.nodes[t]['inflow'] for t in range(2)] == [4, 5]
    assert all('power' not in S.nodes[t] for t in range(2))
    assert not {'powers_set', 'power_unit'} & S.graph.keys()
    assert {k: S.graph[k] for k in quantization} == quantization

    G_back = G_from_S(S, wfn.A)
    assert [G_back.nodes[t]['power'] for t in range(2)] == [1, Fraction(5, 4)]
    assert {k: G_back.graph[k] for k in quantization} == quantization
    assert G_back.graph['powers_set'] == G.graph['powers_set']

    L = L_from_G(G)
    assert [dict(L.nodes[t]).get('power') for t in range(2)] == [1, Fraction(5, 4)]
    assert all('inflow' not in L.nodes[t] for t in range(2))
    assert not {'power_per_inflow', 'capacity_nominal'} & L.graph.keys()
    restored = WindFarmNetwork(L=L, cables=2.5, power_unit='MW')
    assert restored.turbine_powers == wfn.turbine_powers


def test_uniform_power_stays_off_the_terminals_and_round_trips():
    wfn = WindFarmNetwork(cables=1, power_unit='MW', turbine_powers=[0.5, 0.5], **_SITE)
    wfn.optimize()
    for graph in (wfn.L, wfn.A, wfn.S, wfn.G):
        assert all(not {'inflow', 'power'} & graph.nodes[t].keys() for t in range(2))
        assert graph.graph['power_per_inflow'] == Fraction(1, 2)
    assert sorted(wfn.get_network()['load']) == [0.5, 1.0]

    L = L_from_G(wfn.G)
    assert L.graph['power_per_inflow'] == Fraction(1, 2)
    restored = WindFarmNetwork(L=L, cables=1, power_unit='MW')
    assert restored.turbine_powers == [Fraction(1, 2)] * 2
    restored.update_from_terse_links(wfn.terse_links())
    assert sorted(restored.get_network()['load']) == [0.5, 1.0]


def test_huge_fraction_power_survives_exactly():
    power = Fraction(10**30 + 1, 10**30)
    wfn = WindFarmNetwork(
        cables=4,
        power_unit='MW',
        turbine_powers=[power, 2 * power],
        router=EWRouter(power_rtol=0),
        **_SITE,
    )
    wfn.update_from_terse_links(np.array([-1, 0]))
    assert wfn.G.graph['power_per_inflow'] == power
    assert wfn.G.graph['power_quantization_inexact'] is False
    assert [wfn.G.nodes[t]['power'] for t in range(2)] == [power, 2 * power]
    assert 'load_nominal' not in wfn.G[-1][0]
    assert sorted(wfn.get_network()['load']) == [float(2 * power), float(3 * power)]
    assert WindFarmNetwork(
        L=L_from_G(wfn.G), cables=4, power_unit='MW'
    ).turbine_powers == [power, 2 * power]


def test_pack_G_refuses_unequal_power():
    wfn = _mixed_wfn()
    wfn.update_from_terse_links(np.array([-1, -1]))
    with pytest.raises(NotImplementedError, match='unequal power'):
        pack_G(wfn.G)


# --- rendering ---


def test_describe_G_reports_inflow():
    wfn = _mixed_wfn()
    wfn.update_from_terse_links(np.array([-1, -1]))
    desc = describe_G(wfn.G)
    assert desc[:2] == [f'κ = {wfn.G.graph["capacity"]}, T = 2', 'ι ∈ {4, 5}']


def test_topology_terse_links_record_the_quantized_capacity():
    wfn = _mixed_wfn()
    wfn.update_from_terse_links(np.array([-1, -1]))
    assert wfn.G.graph['capacity'] == 10


@pytest.mark.parametrize('backend', ('svg', 'matplotlib'))
@pytest.mark.parametrize(('rtol', 'inexact'), ((0, False), (0.01, True)))
def test_load_and_power_labels(backend, rtol, inexact):
    wfn = WindFarmNetwork(
        cables=20,
        power_unit='MW',
        turbine_powers=[5.0, 6.35],
        router=EWRouter(power_rtol=rtol),
        **_SITE,
    )
    wfn.update_from_terse_links(np.array([-1, 0]))
    G = wfn.G
    assert G.graph['power_quantization_inexact'] is inexact

    assert _labels(G, 'load_nominal', backend) == ['11.35', '11.35', '6.35']
    # only inexact quantization traverses the network and writes to the graph
    assert ('load_nominal' in G.nodes[0]) is inexact
    assert _labels(G, 'load', backend) == sorted(
        str(G.nodes[n]['load']) for n in (-1, 0, 1)
    )
    assert _labels(G, 'power', backend) == ['5', '6.35']
    assert _labels(wfn.L, 'power', backend) == ['5', '6.35']


def test_power_labels_fall_back_to_inflow_times_power_per_inflow():
    wfn = WindFarmNetwork(cables=2, turbine_powers=[1, 2], **_SITE)
    wfn.update_from_terse_links(np.array([-1, -1]))
    G = wfn.G
    G.graph['power_per_inflow'] = Fraction(1, 2)
    assert _labels(G, 'power', 'svg') == ['0.5', '1']
    assert _labels(G, 'power', 'matplotlib') == ['0.5', '1']


@pytest.mark.parametrize('node_tag', (None, 'load', 'power'))
def test_explicit_unitary_inflow_preserves_svg(node_tag):
    implicit = WindFarmNetwork(cables=2, **_SITE)
    explicit = WindFarmNetwork(cables=2, turbine_powers=[1, 1], **_SITE)
    for wfn in (implicit, explicit):
        wfn.update_from_terse_links(np.array([-1, -1]))
    expected = svgplot(implicit.G, node_tag=node_tag)
    actual = svgplot(explicit.G, node_tag=node_tag)
    assert actual.data == expected.data
    assert actual.metadata == expected.metadata


# --- terminal symbols per power level ---

# seven power levels, so that the shape list cycles; terminal 6 has the lowest power
_LEVELS_POWERS = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]
_LEVELS_SIDES = (0, 8, 7, 6, 5, 0, 8)
_LEVELS_LABELS = [f'{p:g} MW' for p in sorted(_LEVELS_POWERS)]


def _levels_G():
    wfn = WindFarmNetwork(
        cables=[(100, 1.0)],
        turbinesC=np.array([[i * 1000.0, 0.0] for i in range(7)]),
        substationsC=np.array([[3000.0, -1000.0]]),
        power_unit='MW',
        turbine_powers=_LEVELS_POWERS,
    )
    wfn.update_from_terse_links(np.full(7, -1))
    return wfn.G


def test_terminal_groups_follow_powers_set_order():
    G = _levels_G()
    groups = _terminal_groups(G)
    assert [(sides, label, terminals) for sides, _, label, terminals in groups] == [
        (sides, label, [6 - i])
        for i, (sides, label) in enumerate(zip(_LEVELS_SIDES, _LEVELS_LABELS))
    ]
    for sides, scale, _, _ in groups:
        assert scale == (
            1.0
            if sides == 0
            else pytest.approx(
                math.sqrt(2 * math.pi / (sides * math.sin(2 * math.pi / sides))),
                abs=5e-5,
            )
        )
    del G.graph['power_unit']
    assert _terminal_groups(G)[0] == (0, 1.0, 'WTG 3', [6])
    G = WindFarmNetwork(cables=2, **_SITE).L
    assert _terminal_groups(G) == [(0, 1.0, 'WTG', [0, 1])]


def _polygon_area(vertices):
    x, y = np.asarray(vertices).T
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def test_svgplot_terminal_shapes_per_power_level():
    data = svgplot(_levels_G(), legend=True).data
    hrefs = ['#wtg' if s == 0 else f'#wtg{s}' for s in _LEVELS_SIDES]
    defs = data.split('<defs>', 1)[1].split('</defs>', 1)[0]
    assert re.findall(r'<circle[^>]* id="wtg"', defs)
    for sides in (8, 7, 6, 5):
        (points,) = re.findall(rf'<polygon[^>]* id="wtg{sides}" points="([^"]+)"', defs)
        assert len(points.split()) == 2 * sides
        assert '-0.0' not in points
        vertices = np.array(points.split(), dtype=float).reshape(-1, 2)
        radius = _NODE_RADII[0]
        # vertices are rounded to 0.1
        assert _polygon_area(vertices) == pytest.approx(math.pi * radius**2, rel=0.01)
    terminals = data.split('id="WTGgrp"', 1)[1].split('id="OSSgrp"', 1)[0]
    assert sorted(re.findall(r'href="([^"]+)"', terminals)) == sorted(hrefs)
    legend = data.split('id="legend"', 1)[1].split('<defs', 1)[0]
    assert re.findall(r'href="(#wtg\d*)"', legend) == hrefs
    assert re.findall(r'<text[^>]*>([^<]+)</text>', legend)[:7] == _LEVELS_LABELS


def test_gplot_terminal_shapes_per_power_level():
    fig, ax = plt.subplots(layout='constrained')
    try:
        gplot(_levels_G(), ax=ax, legend=True)
        terminals = [
            c
            for c in ax.collections
            if isinstance(c, PathCollection) and c.get_label() in _LEVELS_LABELS
        ]
        assert [c.get_label() for c in terminals] == _LEVELS_LABELS
        circle = MarkerStyle('o')
        circle_path = circle.get_path().transformed(circle.get_transform())
        circle_area = math.pi * 0.5**2 * NODESIZE
        for collection, sides in zip(terminals, _LEVELS_SIDES):
            (path,) = collection.get_paths()
            (size,) = collection.get_sizes()
            if sides:
                # closed regular polygon: one vertex per side plus the closing one
                assert len(path.vertices) == sides + 1
                area = _polygon_area(path.vertices[:-1]) * size
                assert area == pytest.approx(circle_area, rel=1e-4)
            else:
                np.testing.assert_array_equal(path.vertices, circle_path.vertices)
                assert size == NODESIZE
    finally:
        plt.close(fig)
