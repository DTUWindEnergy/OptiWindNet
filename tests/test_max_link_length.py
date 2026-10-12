"""Coverage for the maximum link length (NetworkConstraints.max_link_length)."""

import shutil

import networkx as nx
import numpy as np
import pytest

from optiwindnet.api import (
    EWRouter,
    HGSRouter,
    MILPRouter,
    NetworkConstraints,
    WindFarmNetwork,
)
from optiwindnet.baselines.hgs import _length_matrix, hgs_cvrp
from optiwindnet.baselines.lkh import _build_weight_matrix, lkh3
from optiwindnet.heuristics import constructor
from optiwindnet.interarraylib import _links_over_limit, links_over_length
from optiwindnet.loads import calcload
from optiwindnet.MILP import OWNWarmupFailed, solver_factory
from optiwindnet.MILP._core import links_to_fix, nonclosest_feeders
from optiwindnet.svg import SvgRepr
from optiwindnet.transforming import as_normalized

from .helpers import solver_unavailable, tiny_wfn
from .sitecache import get_bundle

# 'toy' has 12 terminals within 387..994 of its single root and links up to 1220,
# so this limit puts both links and feeders out of reach.
_TOY_LIMIT = 900.0
_CAPACITY = 5


def _scip():
    try:
        return solver_factory('scip')
    except Exception as exc:
        if solver_unavailable(exc):
            pytest.skip('scip unavailable')
        raise


def _limit(**kwargs):
    return NetworkConstraints(max_link_length=_TOY_LIMIT, **kwargs)


def _toy_wfn(router=None):
    return WindFarmNetwork(
        cables=_CAPACITY, L=get_bundle('toy').L, router=router, constraints=_limit()
    )


def _section_length(A, u, v):
    if min(u, v) < 0:
        return A.graph['d2roots'][max(u, v), min(u, v)]
    return A[u][v]['length']


def _over_limit_links(A, ub):
    return {
        (u, v)
        for u, v, length in A.edges(data='length')
        if u >= 0 and v >= 0 and length > ub
    }


def _over_limit_feeders(A, ub):
    T, R = A.graph['T'], A.graph['R']
    return {
        (t, r) for t in range(T) for r in range(-R, 0) if A.graph['d2roots'][t, r] > ub
    }


# -------- NetworkConstraints --------


@pytest.mark.parametrize('bad', [0, -1.0, float('inf'), float('nan'), '900'])
def test_constraints_reject_an_invalid_limit(bad):
    with pytest.raises(ValueError, match='max_link_length'):
        NetworkConstraints(max_link_length=bad)


def test_resolve_puts_the_limit_in_the_bounds_in_the_scale_of_A():
    A = get_bundle('toy').A
    assert NetworkConstraints().resolve(A, capacity=_CAPACITY).link_length_ub is None
    assert _limit().resolve(A, capacity=_CAPACITY).link_length_ub == _TOY_LIMIT
    normalized = as_normalized(A)
    assert _limit().resolve(normalized, capacity=_CAPACITY).link_length_ub == (
        pytest.approx(_TOY_LIMIT * A.graph['norm_scale'])
    )


def test_resolve_refuses_a_limit_that_isolates_turbines():
    with pytest.raises(ValueError, match='without any link or feeder'):
        NetworkConstraints(max_link_length=1e-3).resolve(
            get_bundle('toy').A, capacity=_CAPACITY
        )


def test_resolve_refuses_a_limit_with_too_few_feeders():
    # at 560, every toy turbine has a link within the limit, but only 4 have a
    # feeder within it
    A = get_bundle('toy').A
    NetworkConstraints(max_link_length=560.0, feeders=4).resolve(A, capacity=_CAPACITY)
    with pytest.raises(ValueError, match='fewer than the 5 subtrees'):
        NetworkConstraints(max_link_length=560.0, feeders=5).resolve(
            A, capacity=_CAPACITY
        )


def test_numeric_resolve_requires_the_limit_in_the_scale_of_A():
    with pytest.raises(ValueError, match='link_length_ub'):
        _limit()._resolve(T=12, R=1, capacity=_CAPACITY)
    with pytest.raises(ValueError, match='link_length_ub'):
        NetworkConstraints()._resolve(
            T=12, R=1, capacity=_CAPACITY, link_length_ub=_TOY_LIMIT
        )


def test_the_limit_leaves_A_unchanged():
    A = get_bundle('toy').A
    edges, graph_keys = set(A.edges), set(A.graph)
    _limit().resolve(A, capacity=_CAPACITY)
    hgs_cvrp(
        as_normalized(A),
        capacity=_CAPACITY,
        time_limit=0.2,
        constraints=_limit(),
        seed=0,
    )
    assert set(A.edges) == edges
    assert set(A.graph) == graph_keys


# -------- MILP --------


@pytest.mark.parametrize('topology', ['branched', 'ringed'])
def test_models_fix_the_over_limit_links_and_feeders(topology):
    A = get_bundle('toy').A
    bounds = _limit(topology=topology).resolve(A, capacity=_CAPACITY)
    links = _over_limit_links(A, _TOY_LIMIT)
    feeders = _over_limit_feeders(A, _TOY_LIMIT)
    assert links and feeders, 'the limit is meant to bind both'
    fixed = list(links_to_fix(A, bounds))
    assert len(fixed) == len(set(fixed))
    expected = links | {(v, u) for u, v in links} | feeders
    if topology == 'ringed':
        expected |= {(r, t) for t, r in feeders}
    assert set(fixed) == expected


def test_multi_root_fixed_links_are_not_repeated():
    A = get_bundle('morayeast').A
    bounds = NetworkConstraints(max_link_length=6000.0).resolve(A, capacity=_CAPACITY)
    fixed = list(links_to_fix(A, bounds))
    assert len(fixed) == len(set(fixed))
    assert set(nonclosest_feeders(A)) <= set(fixed)
    assert _over_limit_feeders(A, 6000.0) <= set(fixed)


def _check_solver_omits_flow_variables(
    solver_name: str, limit: float, capacity: int
) -> None:
    from optiwindnet.constraints import NetworkConstraints
    from optiwindnet.MILP import solver_factory
    from optiwindnet.MILP._core import links_to_fix

    from .sitecache import get_bundle

    bundle = get_bundle('toy')
    A = bundle.A
    constraints = NetworkConstraints(max_link_length=limit)
    bounds = constraints.resolve(A, capacity=capacity)
    fixed = set(links_to_fix(A, bounds))
    assert fixed
    solver = solver_factory(solver_name)
    solver.set_problem(bundle.P, A, capacity=capacity, constraints=constraints)
    for link in fixed:
        assert link in solver.metadata.link_
        assert link not in solver.metadata.flow_


@pytest.mark.parametrize(
    'solver_name',
    ['ortools.cp_sat', 'scip', 'highs', 'pyomo.highs'],
    ids=['ortools', 'scip', 'highs', 'pyomo'],
)
def test_models_omit_flow_variables_for_fixed_links(solver_name, run_isolated):
    result = run_isolated(
        solver_name,
        _check_solver_omits_flow_variables,
        (solver_name, _TOY_LIMIT, _CAPACITY),
        60,
    )
    if isinstance(result, BaseException):
        if solver_unavailable(result):
            pytest.skip(f'{solver_name} unavailable: {result}')
        raise result


def test_model_rejects_a_warm_start_over_the_limit():
    bundle = get_bundle('toy')
    A = bundle.A
    S = constructor(A, capacity=_CAPACITY)
    # re-root one subtree at a terminal beyond the limit: the topology stays a
    # tree, so only the limit can reject it
    forbidden = int(np.flatnonzero(A.graph['d2roots'][:, -1] > _TOY_LIMIT)[0])
    subtree = S.nodes[forbidden]['subtree']
    (hook,) = (n for n in S[-1] if S.nodes[n]['subtree'] == subtree)
    S.remove_edge(-1, hook)
    S.add_edge(-1, forbidden)
    calcload(S)
    assert _links_over_limit(S, A, _TOY_LIMIT)

    with pytest.raises(OWNWarmupFailed, match='link length limit'):
        _scip().set_problem(
            bundle.P, A, capacity=_CAPACITY, constraints=_limit(), warmstart=S
        )
    # the very same topology is a valid warm start without the limit
    _scip().set_problem(bundle.P, A, capacity=_CAPACITY, warmstart=S)


def test_milp_router_enforces_the_limit_on_every_link():
    # albatros at capacity 4 wants a 2337-long feeder, so this limit binds
    limit, capacity = 2300.0, 4
    _scip()
    wfn = WindFarmNetwork(
        cables=capacity,
        L=get_bundle('albatros').L,
        router=MILPRouter(solver_name='scip', time_limit=30, mip_gap=0.005),
        constraints=NetworkConstraints(max_link_length=limit),
    )
    wfn.optimize()
    assert _over_limit_feeders(wfn.A, limit), 'the limit is meant to bind'
    for u, v in wfn.S.edges:
        assert _section_length(wfn.A, u, v) <= limit


# -------- HGS-CVRP and LKH-3 --------


@pytest.mark.parametrize('complete', [False, True])
def test_hgs_matrix_prices_over_limit_links_as_absent_ones(complete):
    A = get_bundle('toy').A
    T = A.graph['T']
    n_from_i = np.array([-1, *range(T)])
    W = _length_matrix(A, -1, 0, n_from_i, complete=complete)
    W_limited = _length_matrix(
        A, -1, 0, n_from_i, complete=complete, link_length_ub=_TOY_LIMIT
    )
    # the penalty, i.e. the clip value
    clip = W_limited.max()
    assert (W_limited[W > _TOY_LIMIT] == clip).all()
    assert (W_limited[W <= _TOY_LIMIT] == W[W <= _TOY_LIMIT]).all()


@pytest.mark.parametrize('complete', [False, True])
def test_lkh_matrix_cuts_at_the_unrounded_lengths(complete):
    # 100.6 is within the limit of 100.7 but rounds to 101, above it; 100.8 is
    # over it. Link (0, 1) is stored in A, pair (0, 2) is only Euclidean.
    VertexC = np.array([(100.6, 0.0), (0.0, 100.8), (201.2, 0.0), (0.0, 0.0)])
    A = nx.Graph(T=3, R=1, VertexC=VertexC)
    A.graph['d2roots'] = np.hypot(*VertexC.T)[:, None]
    A.add_edge(0, 1, length=100.6)
    w_clip = 999
    L = _build_weight_matrix(
        A,
        [0, 1, 2],
        -1,
        scale=1.0,
        complete=complete,
        w_clip=w_clip,
        link_length_ub=100.7,
    )
    assert L[0, 1] == 101
    assert L[0, 2] == (101 if complete else w_clip)
    assert L[1, 2] == w_clip
    assert L[:-1, -1].tolist() == [101, w_clip, w_clip]


def test_hgs_router_keeps_every_link_within_the_limit():
    wfn = _toy_wfn(HGSRouter(time_limit=1, seed=0))
    wfn.optimize()
    for u, v in wfn.S.edges:
        assert _section_length(wfn.A, u, v) <= _TOY_LIMIT


def test_hgs_raises_on_a_solution_over_the_limit():
    # 560 passes the checks of resolve() (4 turbines with a feeder within it, for
    # 3 subtrees), but HGS-CVRP finds no toy network within it
    with pytest.raises(ValueError, match='within the link length limit'):
        hgs_cvrp(
            as_normalized(get_bundle('toy').A),
            capacity=_CAPACITY,
            time_limit=0.2,
            constraints=NetworkConstraints(max_link_length=560.0),
            seed=0,
        )


def test_lkh_keeps_every_link_within_the_limit():
    if shutil.which('LKH') is None:
        pytest.skip('LKH executable not on PATH')
    A = get_bundle('toy').A
    S = lkh3(
        as_normalized(A),
        capacity=_CAPACITY,
        time_limit=1,
        constraints=_limit(),
        seed=1,
    )
    for u, v in S.edges:
        assert _section_length(A, u, v) <= _TOY_LIMIT


# -------- WindFarmNetwork / routers --------


def test_ew_router_refuses_a_limit():
    wfn = _toy_wfn()
    with pytest.raises(NotImplementedError, match='link length'):
        wfn.optimize()
    # it is the limit that makes EW unusable, not EW itself
    wfn.constraints = None
    wfn.optimize(router=EWRouter())
    assert wfn.length() > 0


def _drawn_links(wfn):
    plot = wfn.plot_available_links()
    assert isinstance(plot, SvgRepr)
    return plot.data.count('<line')


def test_plot_available_links_leaves_out_the_links_over_the_limit():
    wfn = WindFarmNetwork(cables=_CAPACITY, L=get_bundle('toy').L)
    drawn = _drawn_links(wfn)
    assert drawn == wfn.A.number_of_edges()
    wfn.constraints = _limit()
    over_limit = sum(
        1 for *_, length in wfn.A.edges(data='length') if length > _TOY_LIMIT
    )
    assert over_limit, 'the limit is meant to bind'
    assert _drawn_links(wfn) == drawn - over_limit
    assert wfn.A.number_of_edges() == drawn


def test_links_over_length_reads_the_limit_off_the_routeset():
    wfn = _toy_wfn(HGSRouter(time_limit=1, seed=0))
    wfn.optimize()
    assert wfn.G.graph['max_link_length'] == _TOY_LIMIT
    assert links_over_length(wfn.G) == links_over_length(wfn.G, _TOY_LIMIT)
    assert links_over_length(wfn.G, 1e12) == []
    over_limit = links_over_length(wfn.G, 1.0)
    assert len(over_limit) == wfn.S.number_of_edges()
    assert [link[2] for link in over_limit] == sorted(
        (link[2] for link in over_limit), reverse=True
    )


# -------- links_over_length() --------


@pytest.mark.parametrize('handle', ['example_location', 'morayeast'])
def test_sections_sum_every_segment_including_detours(handle):
    wfn = WindFarmNetwork(cables=_CAPACITY, L=get_bundle(handle).L)
    wfn.optimize()
    G = wfn.G
    assert G.graph.get('C', 0) + G.graph.get('D', 0), 'meant to have clones'
    sections = links_over_length(G, 0.0)
    assert len(sections) == wfn.S.number_of_edges()
    assert sum(length for *_, length in sections) == pytest.approx(
        G.size(weight='length')
    )


def test_sections_of_a_contoured_routeset():
    wfn = tiny_wfn(optimize=True)
    G = wfn.G
    assert G.graph.get('C', 0), 'meant to have contours'
    sections = links_over_length(G, 0.0)
    assert {(min(u, v), max(u, v)) for u, v, _ in sections} == {
        (min(u, v), max(u, v)) for u, v in wfn.S.edges
    }
    assert sum(length for *_, length in sections) == pytest.approx(
        G.size(weight='length')
    )
