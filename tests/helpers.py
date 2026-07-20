import copy
import pickle
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import networkx as nx
import numpy as np

from optiwindnet.api import EWRouter, HGSRouter, MILPRouter, WindFarmNetwork
from optiwindnet.api_utils import parse_cables_input
from optiwindnet.geometric import is_crossing
from optiwindnet.interarraylib import assign_cables, terse_links_from_S
from optiwindnet.mesh import make_planar_embedding
from optiwindnet.MILP import ModelOptions, solver_factory


def load_instances(path: Path) -> Any:
    """Load a pickled instance file; raise FileNotFoundError with
    regeneration hint if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f'Missing expected test data file: {path}\n\n'
            'To (re)generate run: python -m tests.update_expected_values\n'
            'Or run pytest with --regen-expected.'
        )
    with path.open('rb') as fh:
        return pickle.load(fh)


def router_factory(spec: Optional[Dict[str, Any]]):
    """Create an instantiated router from a spec dict
    (same semantics as your generators)."""
    if spec is None:
        return None
    clsname = spec.get('class')
    params = dict(spec.get('params', {}))
    # Expand ModelOptions dict when present
    if clsname == 'MILPRouter' and isinstance(params.get('model_options'), dict):
        params['model_options'] = ModelOptions(**params['model_options'])
    if clsname is None:
        return None
    if clsname == 'EWRouter':
        return EWRouter(**params)
    if clsname == 'HGSRouter':
        return HGSRouter(**params)
    if clsname == 'MILPRouter':
        return MILPRouter(**params)
    raise ValueError(f'Unknown router class: {clsname!r}')


def needs_process_isolation(router_spec: Dict[str, Any]) -> bool:
    """'ortools*' solvers bundle a copy of HiGHS/SCIP that collides with the
    standalone highspy/pyscipopt packages if both load into the same process.
    Other solvers (cplex, gurobi, 'highs', ...) never touch OR-Tools' native
    libraries and can run directly in this process."""
    solver_name = router_spec['params'].get('solver_name', '')
    return router_spec['class'] == 'MILPRouter' and solver_name.startswith('ortools')


def solve_milp_low_level(router_spec: Dict[str, Any], L: nx.Graph):
    """Solve a `MILPRouter` routed_instance spec via the low-level MILP API,
    bypassing `MILPRouter`/`WindFarmNetwork`.

    Replicates `MILPRouter.route()`'s non-warmstart sequence directly (see
    optiwindnet/api.py: `MILPRouter.route`, `WindFarmNetwork.optimize`) so
    callers don't need a `WindFarmNetwork` -- solver execution (the part that
    needs process isolation for 'ortools*' solvers) stays free of any
    high-level API dependency.

    Returns ``(terse_links, canonical_edges)`` for comparison against a
    routed_instance fixture's expected values.
    """
    params = router_spec['params']
    # cables_capacity and assign_cables() both need the *parsed* cables list,
    # not the raw int/list stored in router_spec['cables'].
    cables = parse_cables_input(router_spec['cables'])
    cables_capacity = max(cables)[0]
    model_options = ModelOptions(**params.get('model_options', {}))

    P, A = make_planar_embedding(L)
    solver = solver_factory(params['solver_name'])
    solver.set_problem(P, A, capacity=cables_capacity, model_options=model_options)
    solver.solve(
        time_limit=params['time_limit'],
        mip_gap=params['mip_gap'],
        options=params.get('solver_options', {}),
        verbose=params.get('verbose', False),
    )
    S, G = solver.get_solution()
    assign_cables(G, cables)
    return tuple(terse_links_from_S(S).tolist()), canonical_edges(G)


def solver_unavailable(exc: BaseException) -> bool:
    """Whether ``exc`` means a MILP backend is missing or unlicensed (=> skip).

    Open-source backends (highs/scip/cbc/fscip) raise import/binary errors when
    absent; commercial ones (gurobi/cplex) raise a license message on machines
    without a valid license (e.g. the size-limited pip license on a large model).
    """
    if isinstance(exc, (FileNotFoundError, ModuleNotFoundError)):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in ('not licensed', 'license', 'token.gurobi.com', 'gurobi model')
    )


def solve_milp_property_metrics(router_spec: Dict[str, Any], L: nx.Graph):
    """Solve a MILPRouter spec and reduce it to picklable property metrics.

    Worker-safe counterpart of ``solve_milp_low_level`` for the property-based
    end-to-end tests: it returns ``(metrics, termination, length)`` instead of an
    exact-topology snapshot, so the caller can assert structural invariants and a
    gap-tolerant length regression rather than an exact edge set.
    """
    params = router_spec['params']
    cables = parse_cables_input(router_spec['cables'])
    cables_capacity = max(cables)[0]
    model_options_dict = params.get('model_options', {})
    model_options = ModelOptions(**model_options_dict)

    P, A = make_planar_embedding(L)
    solver = solver_factory(params['solver_name'])
    solver.set_problem(P, A, capacity=cables_capacity, model_options=model_options)
    info = solver.solve(
        time_limit=params['time_limit'],
        mip_gap=params['mip_gap'],
        options=params.get('solver_options', {}),
        verbose=params.get('verbose', False),
    )
    S, G = solver.get_solution()
    assign_cables(G, cables)
    metrics = solution_property_metrics(S, G, model_options_dict, cables_capacity)
    return metrics, str(info.termination), metrics['length']


def has_cycle(S: nx.Graph) -> bool:
    """Whether ``S`` carries a cycle once the substations are interconnected.

    The graphs OptiWindNet works with do not represent the substation-to-
    substation connections of the physical network, so a ring bridging two
    substations reads as a path between two roots. Linking the roots in a path
    supplies those connections, so that every ring -- bridging or not -- closes
    a cycle.
    """
    Sx = S.copy()
    Sx.add_edges_from(pairwise(range(-S.graph['R'], 0)))
    return not nx.is_forest(Sx)


def solution_property_metrics(
    S: nx.Graph, G: nx.Graph, model_options: Optional[dict], capacity: int
) -> Dict[str, Any]:
    """Reduce a solved (S, G) pair to picklable property metrics.

    Returns only primitives so it can run inside the ``ortools_worker``
    subprocess and be shipped back. ``assert_solution_properties`` consumes the
    result. Keeping the reduction here (rather than in the test) means the exact
    same checks apply whether the solve ran in-process or in the worker.
    """
    import math

    from optiwindnet.crossings import validate_routeset
    from optiwindnet.interarraylib import validate_topology

    R, T = S.graph['R'], S.graph['T']
    topology = (model_options or {}).get('topology', 'branched')

    edge_loads = [d['load'] for _, _, d in S.edges(data=True)]
    feeder_edges = [(u, v) for u, v in S.edges if u < 0 or v < 0]
    feeder_loads = sorted(S[u][v]['load'] for u, v in feeder_edges)

    # a list of strings is picklable, so the whole check crosses the worker
    # boundary intact
    S.graph.setdefault('topology', topology)
    topology = S.graph['topology']
    topology_violations = validate_topology(S, capacity)

    return dict(
        valid_findings=len(validate_routeset(G)),
        length=float(G.size(weight='length')),
        max_edge_load=max(edge_loads) if edge_loads else 0,
        sum_root_load=sum(S.nodes[r]['load'] for r in range(-R, 0)),
        T=T,
        R=R,
        has_cycle=has_cycle(S),
        num_feeders=len(feeder_edges),
        feeder_loads=feeder_loads,
        min_feeders=math.ceil(T / capacity),
        topology_violations=topology_violations,
        topology=str(topology),
    )


def assert_solution_properties(
    metrics: Dict[str, Any], spec: Dict[str, Any], capacity: int
) -> None:
    """Assert the invariants a valid solution must satisfy for its options.

    Covers validity, capacity, full connectivity, topology shape (radial /
    branched / ringed), and -- on single-substation instances where they are
    unambiguous -- the feeder-count and balanced-load constraints implied by the
    model options. Objective-length regression is checked separately by the
    caller against a stored reference.
    """
    mo = (
        spec['params'].get('model_options', {}) if spec['class'] == 'MILPRouter' else {}
    )
    topology = metrics['topology']
    T = metrics['T']

    # --- universal invariants -------------------------------------------------
    assert metrics['valid_findings'] == 0, 'validate_routeset reported findings'
    assert metrics['max_edge_load'] <= capacity, 'a cable exceeds capacity'
    assert metrics['sum_root_load'] == T, 'not every terminal is connected'

    # --- topology shape -------------------------------------------------------
    # the shape invariants themselves belong to the library (`validate_topology`);
    # they were reduced to violation strings on the worker side
    assert metrics['topology_violations'] == []
    if topology == 'ringed' and T > 1:
        assert metrics['has_cycle'], 'a ring closes a cycle'

    # --- feeder-count / balance (single-root, non-ringed: well-defined) -------
    single_root = metrics['R'] == 1
    feeder_limit = mo.get('feeder_limit', 'unlimited')
    min_feeders = metrics['min_feeders']
    if single_root and topology in ('radial', 'branched'):
        nf = metrics['num_feeders']
        if feeder_limit == 'minimum':
            assert nf == min_feeders
        elif feeder_limit == 'min_plus1':
            assert min_feeders <= nf <= min_feeders + 1
        elif feeder_limit == 'exactly':
            assert nf == mo['max_feeders']
        elif feeder_limit == 'specified':
            assert min_feeders <= nf <= mo['max_feeders']
        else:  # unlimited
            assert nf >= min_feeders
        if mo.get('balanced'):
            # pinned feeder count => subtree (feeder) loads differ by at most one
            loads = metrics['feeder_loads']
            assert max(loads) - min(loads) <= 1, 'balanced subtrees must differ by <=1'


def terminal_terminal_edges(S: nx.Graph) -> list[tuple[int, int]]:
    """Edges connecting two terminals (i.e. excluding root feeders)."""
    return [(u, v) for u, v in S.edges if u >= 0 and v >= 0]


def terminal_terminal_crossings(
    S: nx.Graph, VertexC: np.ndarray
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Crossings among terminal-terminal edges (feeders excluded).

    The constructive heuristics guarantee no terminal-terminal crossings (the
    core feature of `constructor`, ringed or not); straight feeders may still
    cross and are resolved later by PathFinder, so they are excluded here.

    Brute-force O(n^2) on purpose: it is independent of `A`/`diagonals`, so it
    checks the heuristic's own guarantee rather than re-deriving it from the
    same combinatorial data the heuristic used.
    """
    edges = terminal_terminal_edges(S)
    crossings = []
    for i, (u, v) in enumerate(edges):
        for s, t in edges[i + 1 :]:
            if len({u, v, s, t}) < 4:
                # shared endpoint: not a crossing
                continue
            if is_crossing(
                VertexC[u], VertexC[v], VertexC[s], VertexC[t], touch_is_cross=False
            ):
                crossings.append(((u, v), (s, t)))
    return crossings


def assert_graph_equal(
    G1: nx.Graph,
    G2: nx.Graph,
    ignored_graph_keys: Optional[Iterable[str]] = None,
    *,
    ignored_node_keys: Optional[Iterable[str]] = None,
    rtol: float = 1e-7,
    atol: float = 1e-10,
    max_show: int = 50,
    verbose: bool = False,
) -> None:
    """
    Compare two NetworkX graphs with tolerant numeric checks and simple diffs.
    Raises AssertionError on any mismatch.

    - `ignored_graph_keys` can contain dotted paths like
      "method_options.fun_fingerprint.funhash"
      or top-level keys like "runtime". Dotted paths are removed from both graphs
      before comparison.
    """

    # --- helpers ----------------------------------------------------------------
    def _pop_nested(d: dict, path: str) -> None:
        """Remove a nested key described by a dotted path from dict d (in-place)."""
        cur = d
        parts = path.split('.')
        for p in parts[:-1]:
            cur = cur.get(p)
            if not isinstance(cur, dict):
                return
        cur.pop(parts[-1], None)

    def _deep_clean(G: nx.Graph, dotted_paths: Iterable[str]) -> nx.Graph:
        """Return a deep copy with specified dotted paths removed
        from graph attr dict."""
        H = copy.deepcopy(G)
        for p in dotted_paths:
            _pop_nested(H.graph, p)
        return H

    def _norm_edges(G: nx.Graph):
        return set(G.edges) if G.is_directed() else {tuple(sorted(e)) for e in G.edges}

    def _preview(seq):
        s = sorted(seq)
        return (
            s
            if verbose or len(s) <= max_show
            else s[:max_show] + [f'...(+{len(s) - max_show} more)']
        )

    def _eq(a, b) -> bool:
        """Tolerant equality for scalars, arrays, lists, dicts."""
        # dict
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                return False
            return all(_eq(a[k], b[k]) for k in a.keys())
        # numpy arrays
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.dtype.kind == 'f' or b.dtype.kind == 'f':
                return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
            return np.array_equal(a, b)
        # sequences
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False
            return all(_eq(x, y) for x, y in zip(a, b))
        # floats
        if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
            return np.isclose(float(a), float(b), rtol=rtol, atol=atol, equal_nan=True)
        # ints (exclude bools)
        if (
            isinstance(a, (int, np.integer))
            and not isinstance(a, (bool, np.bool_))
            and isinstance(b, (int, np.integer))
            and not isinstance(b, (bool, np.bool_))
        ):
            return int(a) == int(b)
        return a == b

    # --- prepare ignored paths --------------------------------------------------
    default_ignored = {
        'bound',
        'relgap',
        'solver_details',
        'method_options.fun_fingerprint.funfile',
        'method_options.fun_fingerprint.funhash',
    }
    ignored_all = set(default_ignored)
    if ignored_graph_keys:
        ignored_all |= set(ignored_graph_keys)

    # --- clean graphs (remove nested ignored fields) -----------------------------
    G1c = _deep_clean(G1, ignored_all)
    G2c = _deep_clean(G2, ignored_all)

    # --- compare node sets ------------------------------------------------------
    nodes1, nodes2 = set(G1c.nodes), set(G2c.nodes)
    if nodes1 != nodes2:
        only1, only2 = nodes1 - nodes2, nodes2 - nodes1
        msg = (
            f'Node sets differ.\nOnly in G1 ({len(only1)}): {_preview(only1)}\n'
            f'Only in G2 ({len(only2)}): {_preview(only2)}'
        )
        raise AssertionError(msg)

    # --- compare edges ----------------------------------------------------------
    e1, e2 = _norm_edges(G1c), _norm_edges(G2c)
    if e1 != e2:
        only1, only2 = e1 - e2, e2 - e1
        msg = (
            f'Edge sets differ.\nOnly in G1 ({len(only1)}): {_preview(only1)}\n'
            f'Only in G2 ({len(only2)}): {_preview(only2)}'
        )
        raise AssertionError(msg)

    # --- compare node attributes -------------------------------------------------
    ignored_node_all = {'label'} | set(ignored_node_keys or ())
    for n in sorted(G1c.nodes):
        a1 = {k: v for k, v in G1c.nodes[n].items() if k not in ignored_node_all}
        a2 = {k: v for k, v in G2c.nodes[n].items() if k not in ignored_node_all}
        if a1.keys() != a2.keys():
            diff = sorted(a1.keys() ^ a2.keys())
            raise AssertionError(f'Node {n} attribute keys differ: {diff}')
        for k in a1:
            if not _eq(a1[k], a2[k]):
                raise AssertionError(
                    f"Node {n} attribute '{k}' differs: {a1[k]!r} != {a2[k]!r}"
                )

    # --- compare graph-level attributes ----------------------------------------
    # After removing nested fields above, also ignore the top-level keys referenced
    # by any dotted ignore paths (e.g. "method_options" if present in ignored_all)
    ignore_top = {p.split('.', 1)[0] for p in ignored_all}
    gkeys1 = set(G1c.graph.keys()) - ignore_top
    gkeys2 = set(G2c.graph.keys()) - ignore_top
    if gkeys1 != gkeys2:
        diff = sorted(gkeys1 ^ gkeys2)
        raise AssertionError(
            f'Graph keys differ (ignoring {sorted(ignore_top)}): {diff}'
        )
    for k in sorted(gkeys1):
        if not _eq(G1c.graph[k], G2c.graph[k]):
            raise AssertionError(
                f"Graph['{k}'] differs: {G1c.graph[k]!r} != {G2c.graph[k]!r}"
            )


def canonical_edges(G: nx.Graph) -> Counter:
    """Edge multiset of G where detour clones are replaced by their primes.

    Two route sets with equal canonical edge multisets are topologically
    equivalent even if their detour clones have different numbering.

    Multiplicity matters: two independent feeders touching the same border
    vertex produce two distinct clones with the same prime, so the canonical
    edge appears twice.
    """
    T = G.graph['T']
    B = G.graph.get('B', 0)
    fnT = G.graph.get('fnT')

    def prime(n: int) -> int:
        n = int(n)
        if n < 0 or n < T + B:
            return n
        return int(fnT[n])

    edges: Counter = Counter()
    for u, v in G.edges:
        pu, pv = prime(u), prime(v)
        edges[(pu, pv) if pu < pv else (pv, pu)] += 1
    return edges


def tiny_wfn(
    turbinesC=None,
    substationsC=None,
    borderC=None,
    obstacleC_=None,
    cables=[(4, 10.0)],
    optimize=True,
    router=None,
):
    """
    Build a compact WindFarmNetwork and return it.

    - turbinesC : (N,2) array-like of turbine coordinates (default four turbines).
    - substationsC : (M,2) array-like of substations (default one at left).
    - borderC : (B,2) array-like polygon coordinates for border (default rectangle).
    - obstacleC_ : list of (k,2) arrays (rings) or a single 2D array
      (default one small obstacle).
    - cables : cables argument passed to WindFarmNetwork (default 4).
    - optimize : if True, call wfn.optimize() before returning (default False).
    """
    # defaults
    if turbinesC is None:
        turbinesC = np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 3.0]], float)
    else:
        turbinesC = np.asarray(turbinesC, float)

    if substationsC is None:
        substationsC = np.array([[0.0, 0.0]], float)
    else:
        substationsC = np.asarray(substationsC, float)
        if substationsC.ndim == 1:
            substationsC = substationsC.reshape(1, 2)

    if borderC is None:
        borderC = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 4.0], [-2.0, 4.0]], float)
    else:
        borderC = np.asarray(borderC, float)

    if obstacleC_ is None:
        obstacleC_ = [np.array([[1.2, -0.5], [1.2, 1], [1.8, 0.5], [1.5, -0.5]])]

    wfn = WindFarmNetwork(
        cables=cables,
        turbinesC=turbinesC,
        substationsC=substationsC,
        borderC=borderC,
        obstacleC_=obstacleC_,
        name='tiny_wfn',
        handle='tiny',
    )

    if optimize:
        # run solver only when requested by the test to avoid heavy/fragile work
        if router:
            wfn.optimize(router=router)
        else:
            wfn.optimize()

    return wfn
