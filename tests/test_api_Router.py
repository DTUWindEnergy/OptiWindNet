# tests/test_api_Router_slim.py
import types
import numpy as np
import networkx as nx
import pytest

from optiwindnet.api import (
    EWRouter, HGSRouter, MILPRouter, WindFarmNetwork, ModelOptions,
    OWNWarmupFailed, OWNSolutionNotFound,
)

# ---- MILP modules
import optiwindnet.MILP.gurobi as gurobi_mod
import optiwindnet.MILP.pyomo as pyomo_mod
import optiwindnet.MILP.ortools as ortools_mod
from optiwindnet.MILP._core import FeederRoute, Topology
from optiwindnet.MILP.pyomo import SolverPyomo  # reused to stub super().solve

# =============================
# Shared tiny helpers & stubs
# =============================

def tiny_A(R=1, T=2):
    A = nx.Graph(R=R, T=T)
    for i in range(T + R): A.add_node(i)
    if T >= 2: A.add_edge(0, 1)
    # minimal attrs used by solvers
    A.graph['d2roots'] = {(0, -1): 1.0}
    A.graph['diagonals'] = []
    return A

def tiny_P():
    P = nx.Graph()
    P.graph['triangles'] = []
    return P

def fake_G_from_S(S, A):
    G = nx.Graph(R=A.graph["R"], T=A.graph["T"], VertexC=np.zeros((A.graph["T"]+A.graph["R"], 2)))
    G.add_edge(0, 1, length=1.0, cable=0, cost=1.0)
    G.graph['cables'] = [(1, 0.0)]
    G.graph['method_options'] = {}
    return G

class FakePF:
    def __init__(self, G, P, A, branched): self.G = G
    def create_detours(self):
        self.G.graph['pf'] = True
        return self.G

# =============================
# Smoke tests (unchanged idea)
# =============================

@pytest.mark.parametrize(
    "router_ctor, router_kwargs",
    [
        (None, {}),
        (EWRouter, {}),
        (EWRouter, {"feeder_route": "straight"}),
        (HGSRouter, {"time_limit": 0.2, "seed": 0}),
        (HGSRouter, {"time_limit": 0.2, "feeder_limit": 0, "seed": 0}),
        pytest.param(
            MILPRouter, {"solver_name": "ortools", "time_limit": 0.2, "mip_gap": 0.2},
            marks=pytest.mark.xfail(raises=Exception, reason="Optional solver deps"),
        ),
    ],
)
def test_router_smoke_tiny_site(router_ctor, router_kwargs):
    wfn = WindFarmNetwork(
        cables=3,
        turbinesC=np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0]]),
        substationsC=np.array([[0.0, 0.0]]),
    )
    router = None if router_ctor is None else router_ctor(**router_kwargs)
    wfn.optimize(router=router)
    G = wfn.G
    assert isinstance(G, nx.Graph)
    assert all(d.get("length", 0) > 0 for *_, d in G.edges(data=True))

# =============================
# MILPRouter control-flow unit tests (compact)
# =============================

def test_ewrouter_invalid_feeder_route():
    wfn = WindFarmNetwork(
        cables=2,
        turbinesC=np.array([[1., 0.], [2., 0.]]),
        substationsC=np.array([[0., 0.]]),
        router=EWRouter(feeder_route='nope'),
    )
    with pytest.raises(ValueError, match="valid feeder_route values"):
        wfn.optimize()

def test_milprouter_retry_and_warmstart_paths(monkeypatch):
    # set up dummy solver that fails once then succeeds
    from optiwindnet import api as api_mod
    calls = {"solve": 0}
    class DummySolver:
        options = {}
        def set_problem(self, *a, **k): pass
        def solve(self, *a, **k):
            calls["solve"] += 1
            if calls["solve"] == 1: raise OWNSolutionNotFound("first fail")
        def get_solution(self):  # tiny graphs
            S = nx.Graph(R=1, T=2); S.add_edge(0, -1, load=1, reverse=True)
            G = fake_G_from_S(S, tiny_A())
            return S, G

    monkeypatch.setattr(api_mod, "solver_factory", lambda name: DummySolver())
    monkeypatch.setattr(api_mod, "is_warmstart_eligible", lambda **k: None)
    monkeypatch.setattr(api_mod, "assign_cables", lambda G, cables: None)

    wfn = WindFarmNetwork(
        cables=2,
        turbinesC=np.array([[1., 0.], [2., 0.]]),
        substationsC=np.array([[0., 0.]]),
        router=MILPRouter(solver_name="ortools", time_limit=0.1, mip_gap=0.2),
    )
    wfn.optimize()
    assert calls["solve"] == 2  # failed then succeeded

# =============================
# G U R O B I (single end-to-end test)
# =============================

class _DummyVar:  # shared with cplex/pyomo fake maps
    def __init__(self, name): self.name, self._val = name, None
    def set_value(self, v, *, skip_validation=False): self._val = v

class _GurVar:  # for topology_from_mip_pool rounding path
    def __init__(self, xn): self.Xn = xn

class _GurobiInnerModel:
    def __init__(self, objectives): self._objectives, self._idx = objectives, 0
    def getAttr(self, name):
        if name == 'SolCount':   return len(self._objectives)
        if name == 'PoolObjVal': return float(self._objectives[self._idx])
        raise KeyError(name)
    def setParam(self, name, val):
        if name == 'SolutionNumber': self._idx = int(val)

class _FakeGurobiSolver:
    def __init__(self, options):
        self.options = options
        x, y = _DummyVar('x'), _DummyVar('y')
        self._pyomo_var_to_solver_var_map = {x: _GurVar(1.0), y: _GurVar(0.0)}
        self._solver_model = _GurobiInnerModel([5.0, 7.0])
    def solve(self, model, **kw):
        return {
            'Solver':  [{'Termination condition': types.SimpleNamespace(name='OPTIMAL'),
                         'Wallclock time': 0.02}],
            'Problem': [{'Upper bound': 10.0, 'Lower bound': 9.0}],
        }
    def close(self): pass

def test_gurobi_full_flow(monkeypatch):
    # factory → fake solver; adapters → light stubs
    monkeypatch.setattr(gurobi_mod.pyo, "SolverFactory",
                        lambda name, **kw: _FakeGurobiSolver(kw.get('options', {})))
    monkeypatch.setattr(gurobi_mod, "G_from_S", fake_G_from_S)
    monkeypatch.setattr(gurobi_mod, "PathFinder", FakePF)
    # topology_from_mip_pool must return S with 'creator'
    monkeypatch.setattr(gurobi_mod, "topology_from_mip_sol",
                        lambda model: nx.Graph([(0, 1)], creator='MILP.stub'))

    s = gurobi_mod.SolverGurobi()
    s.P, s.A = tiny_P(), tiny_A()
    s.model = object()
    s.model_options = {'feeder_route': FeederRoute.STRAIGHT, 'topology': Topology.BRANCHED}
    s.solve_kwargs = {}

    info = s.solve(time_limit=0.1, mip_gap=0.05, options={'mipfocus': 1})
    assert info.objective == pytest.approx(10.0)
    assert s.objective_at(1) == pytest.approx(7.0)  # pool swap tested

    # STRAIGHT branch
    
    S, G = s.get_solution()
    assert G.graph['pf'] and 'solver_details' in G.graph

    # SEGMENTED branch (flip once to cover investigate_pool)
    monkeypatch.setattr(gurobi_mod, "investigate_pool",
                        lambda P, A, self_ref: (nx.Graph(), nx.Graph()))
    s.model_options['feeder_route'] = FeederRoute.SEGMENTED
    S2, G2 = s.get_solution()
    assert isinstance(S2, nx.Graph) and isinstance(G2, nx.Graph)

# =============================
# P Y O M O (single end-to-end test)
# =============================

class _FakePyomoResults(dict):
    def __init__(self):
        super().__init__({
            'Solver':  [{'Termination condition': types.SimpleNamespace(name='OPTIMAL'),
                         'Wallclock time': 0.01}],
            'Problem': [{'Upper bound': 4.0, 'Lower bound': 3.5}],
        })
        self.solution = [1]
        self.solver = types.SimpleNamespace(status=None)

class _FakePyomoSolver:
    def __init__(self): self.options = {}
    def warm_start_capable(self): return True
    def solve(self, model, **kw): return _FakePyomoResults()

def test_pyomo_full_flow(monkeypatch):
    monkeypatch.setattr(pyomo_mod.pyo, "SolverFactory",
                        lambda name, **kw: _FakePyomoSolver())
    monkeypatch.setattr(pyomo_mod, "G_from_S", fake_G_from_S)
    monkeypatch.setattr(pyomo_mod, "PathFinder", FakePF)

    sp = pyomo_mod.SolverPyomo(name="cbc")
    sp.set_problem(P=tiny_P(), A=tiny_A(), capacity=2,
                   model_options=ModelOptions(topology="branched", feeder_route="segmented"),
                   warmstart=None)

    info = sp.solve(time_limit=0.1, mip_gap=0.05, options={})
    assert info.objective == pytest.approx(4.0)

    # bypass pyomo solution loader
    sp.model = types.SimpleNamespace(solutions=types.SimpleNamespace(load_from=lambda _res: None))

    # return an S with required attrs
    def _stub_topology_from_mip_sol(model=None, **kw):
        S = nx.Graph()
        S.add_edge(0, 1, reverse=True, load=1)
        S.graph.update(creator='MILP.stub', capacity=2, has_loads=True, solver_details={})
        return S
    monkeypatch.setattr(pyomo_mod, "topology_from_mip_sol", _stub_topology_from_mip_sol)

    S, G = sp.get_solution(A=tiny_A())
    assert G.graph['pf']
    assert isinstance(sp.topology_from_mip_sol(), nx.Graph)

# =============================
# O R T O O L S (single end-to-end test)
# =============================

def test_ortools_pool_and_get_solution(monkeypatch):
    so = ortools_mod.SolverORTools()

    # lean set_problem: attach tiny metadata
    def fake_set_problem(self, P, A, capacity, model_options, warmstart=None):
        self.P, self.A, self.capacity = P, A, capacity
        self.model_options = model_options
        self.metadata = types.SimpleNamespace(R=A.graph['R'], T=A.graph['T'],
                                              capacity=capacity, linkset=[], link_={}, flow_={},
                                              model_options=model_options)
    monkeypatch.setattr(ortools_mod.SolverORTools, "set_problem", fake_set_problem)

    # slim solve: fill pool + proper dataclass SolutionInfo
    def fake_solve(self, time_limit, mip_gap, options={}, verbose=False):
        self.solution_pool = [(12.0, {0: True}), (10.0, {0: False})]
        self._value_map = self.solution_pool[1][1]
        self.num_solutions = 2
        self.solution_info = ortools_mod.SolutionInfo(
            runtime=0.01, bound=9.5, objective=10.0, relgap=0.05, termination="OPTIMAL"
        )
        self.applied_options = {}
        return self.solution_info
    monkeypatch.setattr(ortools_mod.SolverORTools, "solve", fake_solve)

    # CpSolver.solution_info() returns a dict used for G.graph['solver_details'].update(...)
    monkeypatch.setattr(so.solver, "solution_info", lambda: {"summary": "fake"}, raising=False)

    monkeypatch.setattr(ortools_mod, "G_from_S", fake_G_from_S)
    monkeypatch.setattr(ortools_mod, "PathFinder", FakePF)
    monkeypatch.setattr(ortools_mod, "topology_from_mip_sol",
                        lambda metadata=None, solver=None, **kw: nx.Graph([(0, 1)]))

    A, P = tiny_A(), tiny_P()
    opts = ModelOptions(topology="branched", feeder_route="straight")
    so.set_problem(P=P, A=A, capacity=2, model_options=opts)
    info = so.solve(0.1, 0.05)
    assert info.objective == pytest.approx(10.0)

    # STRAIGHT
    S, G = so.get_solution(A=A)
    assert 'solver_details' in G.graph and G.graph['pf']
    # objective_at swaps map
    assert so.objective_at(0) == pytest.approx(12.0)

    # SEGMENTED → investigate_pool
    monkeypatch.setattr(ortools_mod, "investigate_pool", lambda P, A, self_ref: (nx.Graph(), nx.Graph()))
    so.model_options['feeder_route'] = FeederRoute.SEGMENTED
    S2, G2 = so.get_solution()
    assert isinstance(S2, nx.Graph) and isinstance(G2, nx.Graph)
