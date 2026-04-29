"""Validation driver for the chain-sleeve pathfinding feature.

Loads pickled G_db routesets for four RIDs and re-runs PathFinder, checking
that the expected feeder paths and total length are produced.

Usage:
    python _run_chain_tests.py
"""

import math
import pickle
import sys
from pathlib import Path

from optiwindnet.interarraylib import G_from_S, L_from_G, S_from_G
from optiwindnet.mesh import make_planar_embedding
from optiwindnet.pathfinding import PathFinder

EXPECTED = {
    22822: {
        'length': 9.4484,
        'feeders': [[-1, 46, 70, 35], [-1, 46, 70, 26]],
    },
    33303: {
        'length': 9.5741,
        'feeders': [[-1, 46, 70, 35], [-1, 46, 70, 26]],
    },
    33372: {
        'length': 11.1059,
        'feeders': [[-1, 40, 57, 44], [-1, 40, 57, 58, 59, 5]],
    },
    35875: {
        'length': None,
        'feeders': [[-1, 60, 75, 76, 77, 8]],
    },
}


def _feeder_paths(G):
    """Extract every -1->...->terminal path through the routeset, mapped to primes."""
    R = G.graph['R']
    T = G.graph['T']
    fnT = G.graph.get('fnT')

    def prime(n):
        if n < 0:
            return n
        if fnT is not None and n < len(fnT):
            return int(fnT[n])
        return n

    feeders = []
    # walk every root->...->terminal path through detour clones and contour
    # helpers; emit one feeder per terminal-anchored path that starts at root
    for r in range(-R, 0):
        for first in G.neighbors(r):
            # DFS to find paths root → ... → terminal (prime in [0, T))
            stack = [(first, [r, first], {r, first})]
            while stack:
                cur, path, visited = stack.pop()
                # Terminate on actual terminal node (not a clone whose prime is
                # a terminal — those happen along contour helpers).
                if 0 <= cur < T:
                    feeders.append([prime(x) for x in path])
                    continue
                for nxt in G.neighbors(cur):
                    if nxt in visited:
                        continue
                    stack.append((nxt, path + [nxt], visited | {nxt}))
    return feeders


def _run_one(rid):
    p = Path(__file__).parent / 'tests' / 'locations' / f'G_{rid}.pkl'
    with open(p, 'rb') as f:
        G_db = pickle.load(f)

    # reconstruct site → planar+A → topology → tentative → PathFinder
    L = L_from_G(G_db)
    L.graph['border'] = G_db.graph['border']
    L.graph['VertexC'] = G_db.graph['VertexC'][
        : G_db.graph['T'] + G_db.graph['B'] - G_db.graph.get('D', 0)
    ] if False else G_db.graph['VertexC'][
        : G_db.graph['T'] + G_db.graph['B']
    ].copy()
    # Stitch the actual VertexC together: terminals + borders + roots
    import numpy as np
    R, T, B = G_db.graph['R'], G_db.graph['T'], G_db.graph['B']
    VertexC = G_db.graph['VertexC']
    L.graph['VertexC'] = np.vstack((VertexC[: T + B], VertexC[-R:]))
    L.graph['B'] = B
    L.graph['R'] = R
    L.graph['T'] = T

    P, A = make_planar_embedding(L)
    S = S_from_G(G_db)
    G_tent = G_from_S(S, A)
    pf = PathFinder(G_tent, planar=P, A=A)
    G_out = pf.create_detours()
    length = G_out.size(weight='length')
    feeders = _feeder_paths(G_out)
    return length, feeders


def main():
    failures = 0
    for rid, expected in EXPECTED.items():
        try:
            length, feeders = _run_one(rid)
        except Exception as e:
            print(f'  RID {rid}: ERROR {type(e).__name__}: {e}')
            failures += 1
            continue
        ok_len = (
            expected['length'] is None
            or math.isclose(length, expected['length'], abs_tol=1e-3)
        )
        # Match expected feeders as multisets of prime-vertex sequences
        got_set = {tuple(f) for f in feeders}
        want_set = {tuple(f) for f in expected['feeders']}
        missing = want_set - got_set
        ok_feeders = not missing
        status = 'OK' if (ok_len and ok_feeders) else 'FAIL'
        if status == 'FAIL':
            failures += 1
        print(
            f'RID {rid}: {status}  length={length:.4f}'
            + (f' (expected {expected["length"]})' if expected['length'] else '')
        )
        if missing:
            print(f'    missing feeders: {sorted(missing)}')
            print(f'    got feeders:     {sorted(got_set)}')
    return failures


if __name__ == '__main__':
    sys.exit(main())
