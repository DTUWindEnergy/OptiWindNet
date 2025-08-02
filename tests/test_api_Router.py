import pytest
import numpy as np
from optiwindnet.api import EWRouter, HGSRouter, MILPRouter, ModelOptions
from optiwindnet.mesh import make_planar_embedding
from optiwindnet.importer import L_from_site


# ========== Fixtures ==========

@pytest.fixture
def dummy_coords():
    # 3 turbines, 1 substation
    turbines = np.array([[0, 0], [1, 0], [2, 0]])
    substations = np.array([[1, 1]])
    return turbines, substations

@pytest.fixture
def dummy_cables():
    return [(10, 100), (20, 150)]

@pytest.fixture
def planar_graph(dummy_coords):
    turbines, substations = dummy_coords
    T, R = turbines.shape[0], substations.shape[0]
    coords = np.vstack((turbines, substations))
    border = np.array([])  # no borders

    L = L_from_site(
        T=T,
        R=R,
        VertexC=coords,
        border=border,
        name='test',
        handle='test'
    )
    P, A = make_planar_embedding(L)
    return L, P, A

# ========== EWRouter Tests ==========

def test_ewrouter_output_type(planar_graph, dummy_cables):
    L, P, A = planar_graph
    router = EWRouter()
    S, G = router.optimize(L=L, A=A, P=P, cables=dummy_cables, cables_capacity=10)

    assert S is not None and G is not None
    assert hasattr(S, 'edges')
    assert hasattr(G, 'edges')
    assert isinstance(G.graph['VertexC'], np.ndarray)

# ========== HGSRouter Tests ==========

def test_hgsrouter_output_type(planar_graph, dummy_cables):
    L, P, A = planar_graph
    router = HGSRouter(time_limit=5, seed=42)
    S, G = router.optimize(A=A, P=P, cables=dummy_cables, cables_capacity=10)

    assert S is not None and G is not None
    assert hasattr(S, 'edges')
    assert hasattr(G, 'edges')
    assert G.number_of_edges() > 0

def test_hgsrouter_balanced_flag(planar_graph, dummy_cables):
    L, P, A = planar_graph
    router = HGSRouter(time_limit=5, balanced=True, seed=123)
    S, G = router.optimize(A=A, P=P, cables=dummy_cables, cables_capacity=10)

    assert G is not None
    assert G.number_of_edges() > 0

# ========== MILPRouter Tests ==========

def test_milprouter_output_type(planar_graph, dummy_cables):
    L, P, A = planar_graph
    router = MILPRouter(
        solver_name="cbc",  # or "glpk" or "ortools" — must be installed
        time_limit=30,
        mip_gap=0.1,
        model_options=ModelOptions()
    )

    S, G = router.optimize(P=P, A=A, cables=dummy_cables, cables_capacity=10)

    assert S is not None and G is not None
    assert hasattr(S, 'edges')
    assert hasattr(G, 'edges')
    assert isinstance(G.graph['VertexC'], np.ndarray)

import logging

def test_milprouter_invalid_solver_logs_error(caplog):
    with caplog.at_level(logging.ERROR):
        router = MILPRouter(
            solver_name="nonexistent_solver",
            time_limit=10,
            mip_gap=0.1
        )
    assert "Unsupported solver" in caplog.text

# ========== Shared Checks ==========

@pytest.mark.parametrize("router_class, router_args", [
    (EWRouter, {}),
    (HGSRouter, {"time_limit": 5, "seed": 1}),
    (MILPRouter, {
        "solver_name": "cbc",
        "time_limit": 10,
        "mip_gap": 0.1,
        "model_options": ModelOptions()
    })
])
def test_router_output_consistency(router_class, router_args, planar_graph, dummy_cables):
    L, P, A = planar_graph
    router = router_class(**router_args)

    if isinstance(router, EWRouter):
        S, G = router.optimize(L=L, A=A, P=P, cables=dummy_cables, cables_capacity=10)
    elif isinstance(router, HGSRouter):
        S, G = router.optimize(A=A, P=P, cables=dummy_cables, cables_capacity=10)
    elif isinstance(router, MILPRouter):
        S, G = router.optimize(P=P, A=A, cables=dummy_cables, cables_capacity=10)

    assert G.number_of_nodes() >= 3
    assert G.number_of_edges() > 0
