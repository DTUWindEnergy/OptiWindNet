import numpy as np

# ========== Graph Assertion Helpers ==========

def assert_graph_equal(G1, G2, ignored_graph_keys=None):
    if ignored_graph_keys is None:
        ignored_graph_keys = set()

    ignored_graph_keys.add("method_options.fun_fingerprint.funfile")

    assert set(G1.nodes) == set(G2.nodes), 'Node sets differ'
    assert set(G1.edges) == set(G2.edges), 'Edge sets differ'

    for n in G1.nodes:
        attrs1 = G1.nodes[n]
        attrs2 = G2.nodes[n]
        filtered1 = {k: v for k, v in attrs1.items() if k != 'label'}
        filtered2 = {k: v for k, v in attrs2.items() if k != 'label'}
        assert filtered1 == filtered2, (
            f'Node {n} attributes differ: {filtered1} != {filtered2}'
        )

    for G in (G1, G2):
        G.graph.get('method_options', {}).pop('fun_fingerprint', None)
        G.graph.get('solver_details', {}).pop('strategy', None)

    keys1 = set(G1.graph.keys()) - ignored_graph_keys
    keys2 = set(G2.graph.keys()) - ignored_graph_keys
    assert keys1 == keys2, f'Graph keys mismatch: {keys1.symmetric_difference(keys2)}'

    RTOL = 1e-8 # relative tolerance
    ATOL = 1e-12 # absolute tolerance

    for k in keys1:
        v1 = G1.graph[k]
        v2 = G2.graph[k]

        if isinstance(v1, np.ndarray):
            # If it's a float array, compare with tolerance; otherwise exact
            if np.issubdtype(v1.dtype, np.floating):
                assert np.allclose(v1, v2, rtol=RTOL, atol=ATOL), f"Mismatch in graph['{k}'] (float array)"
            else:
                assert np.array_equal(v1, v2), f"Mismatch in graph['{k}'] (array)"
        elif isinstance(v1, (float, np.floating)):
            # Scalar float: tolerant compare
            assert np.isclose(v1, v2, rtol=RTOL, atol=ATOL), f"Mismatch in graph['{k}'] (float)"
        elif isinstance(v1, list):
            assert isinstance(v2, list) and len(v1) == len(v2), (
                f"Mismatch in list length for graph['{k}']"
            )
            for a, b in zip(v1, v2):
                if isinstance(a, np.ndarray):
                    if np.issubdtype(a.dtype, np.floating):
                        assert np.allclose(a, b, rtol=RTOL, atol=ATOL), (
                            f"Mismatch in list of float arrays in graph['{k}']"
                        )
                    else:
                        assert np.array_equal(a, b), (
                            f"Mismatch in list of arrays in graph['{k}']"
                        )
                elif isinstance(a, (float, np.floating)):
                    assert np.isclose(a, b, rtol=RTOL, atol=ATOL), (
                        f"Mismatch in list float values in graph['{k}']"
                    )
                else:
                    assert a == b, f"Mismatch in list values in graph['{k}']"
        else:
            assert v1 == v2, f"Mismatch in graph['{k}']: {v1} != {v2}"
