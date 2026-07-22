import re

import numpy as np
import pytest

from optiwindnet.api import EWRouter, HGSRouter, MILPRouter, WindFarmNetwork
from optiwindnet.interarraylib import L_from_G, S_from_G

from .helpers import solver_unavailable


_SITE = dict(
    turbinesC=np.array([[0.0, 0.0], [1.0, 0.0]]),
    substationsC=np.array([[0.0, 1.0]]),
)


def _weighted_wfn(**kwargs):
    return WindFarmNetwork(
        cables=[(1, 100.0), (2, 150.0)],
        turbine_power=[1.0, 1.25],
        turbine_power_decimals=2,
        **_SITE,
        **kwargs,
    )


def _solve_weighted(solver_name):
    wfn = WindFarmNetwork(
        cables=2,
        turbine_power=[1.0, 1.5],
        router=MILPRouter(solver_name, time_limit=5, mip_gap=0.01),
        **_SITE,
    )
    wfn.optimize()
    return wfn.S, wfn.G


def test_weighted_power_input():
    wfn = WindFarmNetwork(
        cables=2,
        turbine_power=[1.001, 1.249],
        turbine_power_decimals=2,
        **_SITE,
    )
    assert wfn.turbine_power == [1.0, 1.25]
    assert wfn.power_scale == 4

    invalid = (
        ([1.0], 2, 'entries but T='),
        ([1.0, 0.0], 2, 'must be positive'),
        ([1.0, float('nan')], 2, 'must be finite'),
        ([1.0, 1.5], -1, 'non-negative integer'),
    )
    for powers, decimals, message in invalid:
        with pytest.raises(ValueError, match=message):
            WindFarmNetwork(
                cables=2,
                turbine_power=powers,
                turbine_power_decimals=decimals,
                **_SITE,
            )


def test_weighted_power_round_trip():
    wfn = _weighted_wfn()
    wfn.update_from_terse_links(np.array([-1, -1]))

    assert wfn.turbine_power == [1.0, 1.25]
    assert wfn.power_scale == 4
    assert sorted(wfn.get_network()['load']) == [1.0, 1.25]
    assert [wfn.G.nodes[t]['power'] for t in range(2)] == [4, 5]
    assert sorted(edge['load'] for *_, edge in wfn.G.edges(data=True)) == [4, 5]
    assert sorted(edge['cable'] for *_, edge in wfn.G.edges(data=True)) == [0, 1]

    converted = (L_from_G(wfn.G), S_from_G(wfn.G))
    for graph in (wfn.A, wfn.S, wfn.G, *converted):
        assert graph.graph['power_scale'] == 4
        assert [graph.nodes[t]['power'] for t in range(2)] == [4, 5]


@pytest.mark.parametrize('router', [EWRouter(), HGSRouter(time_limit=0.1)])
def test_weighted_power_rejects_ew_hgs_routers(router):
    with pytest.raises(TypeError, match='does not support non-uniform turbine_power'):
        _weighted_wfn(router=router).optimize()


def test_weighted_power_milp():
    try:
        S, G = _solve_weighted('highs')
    except BaseException as exc:
        if solver_unavailable(exc):
            pytest.skip('highs not available')
        raise

    assert sum(S.nodes[n]['load'] for n in S.neighbors(-1)) == 5
    assert S.graph['max_load'] <= 4
    assert [S.nodes[t]['power'] for t in range(2)] == [2, 3]
    assert G.graph['power_scale'] == 2


def test_weighted_power_rendering():
    wfn = _weighted_wfn()
    wfn.update_from_terse_links(np.array([-1, -1]))

    plot = wfn.plot(node_tag='power')
    assert '1.25' in re.findall(r'<text[^>]*>([^<]+)</text>', plot.data)
