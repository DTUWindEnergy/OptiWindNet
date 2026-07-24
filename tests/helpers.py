import copy
import math
import warnings
from collections import Counter
from collections.abc import Mapping
from typing import Any, Iterable, Optional

import networkx as nx
import numpy as np

from optiwindnet.api import WindFarmNetwork
from optiwindnet.geometric import is_crossing
from optiwindnet.MILP import (
    ModelOptions,
    OWNSolutionNotFound,
    SolutionInfo,
    solver_factory,
)


def run_milp_solve_with_retry(
    P: nx.PlanarEmbedding,
    A: nx.Graph,
    *,
    solver_name: str,
    capacity: int,
    model_options: ModelOptions | Mapping[str, Any],
    time_limit: float,
    mip_gap: float,
    warmstart: nx.Graph | None = None,
) -> tuple[SolutionInfo, nx.Graph, str]:
    """Execute solver.solve with retry on OWNSolutionNotFound or non-finite bounds."""

    def _single_solve(limit: float) -> tuple[SolutionInfo, nx.Graph, str]:
        solver = solver_factory(solver_name)
        solver.set_problem(
            P,
            A,
            capacity=capacity,
            model_options=model_options,
            warmstart=warmstart,
        )
        info = solver.solve(time_limit=limit, mip_gap=mip_gap)
        if not math.isfinite(info.bound) and info.termination.lower() != 'optimal':
            raise OWNSolutionNotFound(
                f'Solver {solver_name!r} returned non-finite dual bound '
                f'({info.bound}) within {limit} s'
            )
        S = solver.get_incumbent_topology()
        return info, S, solver.metadata.warmed_by

    try:
        return _single_solve(time_limit)
    except OWNSolutionNotFound:
        fallback_limit = time_limit * 3.0
        warnings.warn(
            f'Solver {solver_name!r} raised OWNSolutionNotFound within '
            f'{time_limit} s (likely due to high CPU load); '
            f'retrying with {fallback_limit} s time limit.',
            UserWarning,
            stacklevel=2,
        )
        return _single_solve(fallback_limit)


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
            return bool(
                np.isclose(float(a), float(b), rtol=rtol, atol=atol, equal_nan=True)
            )
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
        assert fnT is not None
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
