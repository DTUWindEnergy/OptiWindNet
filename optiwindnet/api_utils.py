def check_rogues(solution, A):
    rogues = []
    branches = ([n - 1 for n in branch] for branch in solution['routes'])
    for branch in branches:
        for edge in pairwise(branch):
            if edge not in A.edges:
                rogues.append(edge)
    return rogues


def edge_crossings(u: int, v: int, G: nx.Graph, diagonals: bidict) \
        -> list[tuple[int, int]]:
    u, v = (u, v) if u < v else (v, u)
    st = diagonals.get((u, v))
    conflicting = []
    if st is None:
        # ⟨u, v⟩ is a Delaunay edge
        st = diagonals.inv.get((u, v))
        if st is not None and st[0] >= 0:
            conflicting.append(st)
    else:
        # ⟨u, v⟩ is a diagonal of Delanay edge ⟨s, t⟩
        s, t = st
        # crossing with Delaunay edge
        conflicting.append(st)

        # two triangles may contain ⟨s, t⟩, each defined by their non-st vertex
        for hat in (u, v):
            for diag in (diagonals.inv.get((w, y) if w < y else (y, w))
                         for w, y in ((s, hat), (hat, t))):
                if diag is not None and diag[0] >= 0:
                    conflicting.append(diag)
    return [edge for edge in conflicting if edge in G.edges]