import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

from optiwindnet.api import EWRouter, HGSRouter, MILPRouter, WindFarmNetwork
from optiwindnet.interarraylib import L_from_G, S_from_G, assign_cables, calcload
from optiwindnet.plotting import gplot
from optiwindnet.svg import svgplot
from optiwindnet.types import Topology

from .helpers import solver_unavailable, tiny_wfn


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
        router=MILPRouter(solver_name, time_limit=5, mip_gap=0.1),
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
        ([1.0, 1.00000001], 8, 'maximum supported'),
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
    restored = WindFarmNetwork(cables=wfn.cables, L=wfn.L)
    restored.update_from_terse_links(np.array([-1, -1]))

    assert restored.turbine_power == [1.0, 1.25]
    assert restored.power_scale == 4
    assert sorted(restored.get_network()['load']) == [1.0, 1.25]
    assert [restored.G.nodes[t]['power'] for t in range(2)] == [4, 5]
    assert sorted(edge['load'] for *_, edge in restored.G.edges(data=True)) == [4, 5]
    assert sorted(edge['cable'] for *_, edge in restored.G.edges(data=True)) == [0, 1]

    converted = (L_from_G(restored.G), S_from_G(restored.G))
    for graph in (restored.A, restored.S, restored.G, *converted):
        assert graph.graph['power_scale'] == 4
        assert [graph.nodes[t]['power'] for t in range(2)] == [4, 5]


@pytest.mark.parametrize('router', [EWRouter(), HGSRouter(time_limit=0.1)])
def test_weighted_power_rejects_heuristic_routers(router):
    with pytest.raises(TypeError, match='does not support non-uniform turbine_power'):
        _weighted_wfn(router=router).optimize()


@pytest.mark.parametrize('solver_name', ['ortools.cp_sat', 'highs'])
def test_weighted_power_milp(solver_name, ortools_worker):
    if solver_name.startswith('ortools'):
        result = ortools_worker.run(_solve_weighted, (solver_name,), 30)
    else:
        try:
            result = _solve_weighted(solver_name)
        except BaseException as exc:
            result = exc

    if isinstance(result, BaseException) and solver_unavailable(result):
        pytest.skip(f'{solver_name} not available')
    if isinstance(result, BaseException):
        raise result

    S, G = result
    assert sum(S.nodes[n]['load'] for n in S.neighbors(-1)) == 5
    assert S.graph['max_load'] <= 4
    assert [S.nodes[t]['power'] for t in range(2)] == [2, 3]
    assert G.graph['power_scale'] == 2


def test_weighted_load_and_cable_assignment():
    G = nx.Graph(R=1, T=2, max_load=0, topology=Topology.BRANCHED)
    G.add_node(0, power=2)
    G.add_node(1, power=3)
    G.add_edge(-1, 0, length=2.0)
    G.add_edge(-1, 1, length=3.0)

    calcload(G)
    assign_cables(G, [(2, 4.0), (5, 5.0)])

    assert G.nodes[-1]['load'] == 5
    assert G[-1][0]['cable'] == 0
    assert G[-1][1]['cable'] == 1
    assert G[-1][0]['cost'] == 8.0
    assert G[-1][1]['cost'] == 15.0


def test_weighted_power_rendering():
    G = tiny_wfn().G.copy()
    G.graph.update(power_scale=2, capacity=8)
    nx.set_node_attributes(G, 2, 'power')
    G.nodes[0].update(power=3, load=3)

    power_ax = gplot(G, node_tag='power')
    load_ax = gplot(G, node_tag='load', infobox=False)
    assert '1.5' in {text.get_text() for text in power_ax.texts}
    assert '1.5' in {text.get_text() for text in load_ax.texts}
    assert 'κ = 4, T =' in power_ax.get_legend().get_title().get_text()
    plt.close('all')

    power_svg = svgplot(G, node_tag='power')
    load_svg = svgplot(G, node_tag='load')
    assert '1.5' in re.findall(r'<text[^>]*>([^<]+)</text>', power_svg.data)
    assert '1.5' in re.findall(r'<text[^>]*>([^<]+)</text>', load_svg.data)
    assert power_svg.metadata['capacity'] == 4
