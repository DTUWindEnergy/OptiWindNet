import copy
from typing import Any, Dict, Iterable, Tuple, Optional
import numpy as np
import networkx as nx


def assert_graph_equal(
    G1: nx.Graph,
    G2: nx.Graph,
    ignored_graph_keys: Optional[Iterable[str]] = None,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-12,
    max_show: int = 100,         # how many items to show in diffs
    verbose: bool = False,  # set True to print full node lists on mismatch
) -> None:
    """Compare two graphs with tolerant numeric checks and helpful diffs."""

    def _sorted_preview(items, limit=max_show):
        seq = sorted(items)
        return seq if verbose or len(seq) <= limit else seq[:limit] + [f"...(+{len(seq)-limit} more)"]

    def _node_brief_list(G: nx.Graph, nodes=None):
        """Return a readable summary list: (id, type, pos(x,y), degree)."""
        def _round_pos(v):
            try:
                if isinstance(v, (tuple, list)) and len(v) >= 2:
                    return (round(float(v[0]), 3), round(float(v[1]), 3))
            except Exception:
                pass
            return None
        out = []
        it = G.nodes(data=True) if nodes is None else ((n, G.nodes[n]) for n in nodes)
        for n, d in it:
            t = d.get("node_type") or d.get("type") or d.get("kind") or d.get("role")
            pos = d.get("pos", d.get("xy", d.get("coords")))
            out.append((n, t, _round_pos(pos), G.degree(n)))
        return _sorted_preview(out)

    def _eq_val(a, b) -> bool:
        # --- containers first ---
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                return False
            return all(_eq_val(a[k], b[k]) for k in a.keys())

        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.dtype.kind == "f" or b.dtype.kind == "f":
                return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
            return np.array_equal(a, b)

        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False
            return all(_eq_val(x, y) for x, y in zip(a, b))

        # --- scalars (allow cross-type numeric matches) ---
        # floats (python float or numpy floating)
        if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
            return np.isclose(float(a), float(b), rtol=rtol, atol=atol, equal_nan=True)

        # integers (python int or numpy integer; exclude bools)
        if (isinstance(a, (int, np.integer)) and not isinstance(a, (bool, np.bool_)) and
            isinstance(b, (int, np.integer)) and not isinstance(b, (bool, np.bool_))):
            return int(a) == int(b)

        # fall back to exact equality
        return a == b


    # --- don’t mutate inputs; strip a couple volatile keys on deep copies
    def _clean_graph_attrs(G: nx.Graph) -> nx.Graph:
        H = copy.deepcopy(G)
        H.graph.get("method_options", {}).pop("fun_fingerprint", None)
        H.graph.get("solver_details", {}).pop("strategy", None)
        return H

    G1c = _clean_graph_attrs(G1)
    G2c = _clean_graph_attrs(G2)

    ignored = {"method_options.fun_fingerprint.funfile", "method_options.fun_fingerprint.funhash"}
    if ignored_graph_keys:
        ignored |= set(ignored_graph_keys)

    # --- node sets
    s1, s2 = set(G1c.nodes), set(G2c.nodes)
    if s1 != s2:
        only1, only2 = s1 - s2, s2 - s1
        # Show IDs
        ids_msg = (
            f"Only in G1 IDs ({len(only1)}): {_sorted_preview(only1)}\n"
            f"Only in G2 IDs ({len(only2)}): {_sorted_preview(only2)}"
        )
        # Also show brief per-node info (type/pos/degree) to spot waypoint nodes quickly
        brief1 = _node_brief_list(G1c, only1)
        brief2 = _node_brief_list(G2c, only2)
        raise AssertionError(
            "Node sets differ.\n"
            + ids_msg
            + "\n"
            + f"G1-only node details (id, type, pos, degree): {brief1}\n"
            + f"G2-only node details (id, type, pos, degree): {brief2}\n"
            + ("" if not verbose else
               f"\nFull G1 nodes: {_node_brief_list(G1c)}\nFull G2 nodes: {_node_brief_list(G2c)}")
        )

    # --- edge sets (normalize for undirected graphs)
    def _norm_edges(G: nx.Graph):
        return set(G.edges) if G.is_directed() else {tuple(sorted(e)) for e in G.edges}

    e1, e2 = _norm_edges(G1c), _norm_edges(G2c)
    if e1 != e2:
        only1, only2 = e1 - e2, e2 - e1
        raise AssertionError(
            "Edge sets differ.\n"
            f"Only in G1 ({len(only1)}): {_sorted_preview(only1)}\n"
            f"Only in G2 ({len(only2)}): {_sorted_preview(only2)}"
        )

    # --- node attributes
    for n in G1c.nodes:
        attrs1 = dict(G1c.nodes[n]); attrs1.pop("label", None)
        attrs2 = dict(G2c.nodes[n]); attrs2.pop("label", None)
        if attrs1.keys() != attrs2.keys():
            diff = sorted(attrs1.keys() ^ attrs2.keys())
            raise AssertionError(f"Node {n} attribute keys differ: {diff}")
        for k in attrs1.keys():
            if not _eq_val(attrs1[k], attrs2[k]):
                raise AssertionError(f"Node {n} attribute '{k}' differs: {attrs1[k]!r} != {attrs2[k]!r}")

    # --- graph-level attributes
    k1 = set(G1c.graph.keys()) - set(ignored)
    k2 = set(G2c.graph.keys()) - set(ignored)
    if k1 != k2:
        diff = _sorted_preview(k1 ^ k2)
        raise AssertionError(f"Graph keys mismatch (ignoring {sorted(ignored)}): {diff}")
    for k in k1:
        v1, v2 = G1c.graph[k], G2c.graph[k]
        if not _eq_val(v1, v2):
            kind = "array" if isinstance(v1, np.ndarray) else type(v1).__name__
            raise AssertionError(f"Mismatch in graph['{k}'] ({kind}): {v1!r} != {v2!r}")
