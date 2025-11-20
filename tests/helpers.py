import copy
from pathlib import Path
from typing import Iterable, Any, Dict, Optional
import dill
import numpy as np
import networkx as nx
from optiwindnet.api import WindFarmNetwork
from optiwindnet.api import EWRouter, HGSRouter, MILPRouter
from optiwindnet.MILP import ModelOptions


def load_dill(path: Path) -> Any:
    """Load a dill file; raise FileNotFoundError with regeneration hint if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f'Missing expected test data file: {path}\n\n'
            'To (re)generate this file run the appropriate generator script, e.g.:\n'
            'update_expected_values.py\n'
            'Or run pytest with --regen-expected to attempt regeneration automatically '
            '(only if you really want that behavior).'
        )
    with path.open('rb') as fh:
        return dill.load(fh)


def router_factory(spec: Optional[Dict[str, Any]]):
    """Create an instantiated router from a spec dict (same semantics as your generators)."""
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


def assert_graph_equal(
    G1: nx.Graph,
    G2: nx.Graph,
    ignored_graph_keys: Optional[Iterable[str]] = None,
    *,
    rtol: float = 1e-7,
    atol: float = 1e-10,
    max_show: int = 50,
    verbose: bool = False,
) -> None:
    """
    Compare two NetworkX graphs with tolerant numeric checks and simple diffs.
    Raises AssertionError on any mismatch.

    - `ignored_graph_keys` can contain dotted paths like "method_options.fun_fingerprint.funhash"
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
        """Return a deep copy with specified dotted paths removed from graph attr dict."""
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
    for n in sorted(G1c.nodes):
        a1 = dict(G1c.nodes[n])
        a1.pop('label', None)
        a2 = dict(G2c.nodes[n])
        a2.pop('label', None)
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


def tiny_wfn(
    turbinesC=None,
    substationsC=None,
    borderC=None,
    obstacleC_=None,
    cables=[(4, 10)],
    optimize=True,
    router=None,
):
    """
    Build a compact WindFarmNetwork and return it.

    - turbinesC : (N,2) array-like of turbine coordinates (default four turbines).
    - substationsC : (M,2) array-like of substations (default one at left).
    - borderC : (B,2) array-like polygon coordinates for border (default rectangle).
    - obstacleC_ : list of (k,2) arrays (rings) or a single 2D array (default one small obstacle).
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
