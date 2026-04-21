import networkx as nx

import optiwindnet.baselines.lkh as lkh_mod


def _make_routeset(branches: list[list[int]]) -> nx.Graph:
    S = nx.Graph(T=sum(len(branch) for branch in branches), R=1)
    S.add_node(-1, load=S.graph['T'])
    for subtree, branch in enumerate(branches):
        predecessor = -1
        for load, node in zip(range(len(branch), 0, -1), branch):
            S.add_node(node, load=load, subtree=subtree)
            S.add_edge(predecessor, node, load=load)
            predecessor = node
    return S


def test_initial_tour_from_solution_orders_branches_and_backfills_missing_nodes():
    S = _make_routeset([[0, 1], [3]])
    initial_tour = lkh_mod._initial_tour_from_solution(
        S,
        original_dimension=5,
        vehicles=3,
    )
    assert initial_tour == [1, 2, 4, 3, 6, 7, 5]


def test_initial_tour_from_solution_preserves_max_load():
    S = _make_routeset([[0, 1, 2], [3]])
    S.graph['max_load'] = 3
    initial_tour = lkh_mod._initial_tour_from_solution(
        S,
        original_dimension=5,
        vehicles=2,
    )
    assert initial_tour == [1, 2, 3, 4, 6, 5]


def test_iterative_lkh_seeds_retry_from_repaired_solution(monkeypatch):
    A = nx.Graph(T=4, R=1, diagonals={(0, 1): None})
    A.add_nodes_from(range(4))
    A.add_edge(0, 1, length=1.0)
    A.add_edge(2, 3, length=2.0)

    first_repaired = _make_routeset([[0, 1], [2, 3]])
    first_repaired.graph['outstanding_crossings'] = [((0, 1), (2, 3))]
    second_repaired = _make_routeset([[0, 1], [2, 3]])

    lkh_calls = []

    def fake_lkh(A, **kwargs):
        lkh_calls.append(kwargs['initial_tour_nodes'])
        return nx.Graph()

    repaired_solutions = iter((first_repaired, second_repaired))

    monkeypatch.setattr(lkh_mod, 'lkh', fake_lkh)
    monkeypatch.setattr(
        lkh_mod,
        'repair_routeset_path',
        lambda S, A: next(repaired_solutions),
    )
    monkeypatch.setattr(lkh_mod, 'add_link_blockmap', lambda A: None)
    monkeypatch.setattr(lkh_mod, '_prune_bad_links', lambda A, limit: None)

    S = lkh_mod.iterative_lkh(
        A,
        capacity=2,
        time_limit=0.1,
        seed=1,
    )

    assert lkh_calls == [
        None,
        [1, 2, 3, 4, 6, 5],
    ]
    assert S.graph['retries'] == 1


def test_iterative_lkh_retries_on_capacity_violation(monkeypatch):
    A = nx.Graph(T=4, R=1, diagonals={})
    A.add_nodes_from(range(4))

    overloaded = _make_routeset([[0, 1, 2], [3]])
    overloaded.graph['max_load'] = 3
    feasible = _make_routeset([[0, 1], [2, 3]])
    feasible.graph['max_load'] = 2

    lkh_calls = []

    def fake_lkh(A, **kwargs):
        lkh_calls.append(kwargs['initial_tour_nodes'])
        return nx.Graph()

    repaired_solutions = iter((overloaded, feasible))

    monkeypatch.setattr(lkh_mod, 'lkh', fake_lkh)
    monkeypatch.setattr(
        lkh_mod,
        'repair_routeset_path',
        lambda S, A: next(repaired_solutions),
    )
    monkeypatch.setattr(lkh_mod, 'add_link_blockmap', lambda A: None)
    monkeypatch.setattr(lkh_mod, '_prune_bad_links', lambda A, limit: None)

    warnings = []
    monkeypatch.setattr(lkh_mod, 'warn', warnings.append)

    S = lkh_mod.iterative_lkh(
        A,
        capacity=2,
        time_limit=0.1,
        seed=1,
    )

    assert lkh_calls == [
        None,
        [1, 2, 3, 4, 6, 5],
    ]
    assert S.graph['retries'] == 1
    assert warnings == [
        'Capacity violated in LKH solution: max_load (3) > capacity (2). Retrying.'
    ]


def test_iterative_lkh_warns_when_max_retries_reached_with_invalid_solution(
    monkeypatch,
):
    A = nx.Graph(T=4, R=1, diagonals={})
    A.add_nodes_from(range(4))

    first_overloaded = _make_routeset([[0, 1, 2], [3]])
    first_overloaded.graph['max_load'] = 3
    second_overloaded = _make_routeset([[0, 1, 2], [3]])
    second_overloaded.graph['max_load'] = 3

    monkeypatch.setattr(lkh_mod, 'lkh', lambda A, **kwargs: nx.Graph())
    repaired_solutions = iter((first_overloaded, second_overloaded))
    monkeypatch.setattr(
        lkh_mod,
        'repair_routeset_path',
        lambda S, A: next(repaired_solutions),
    )
    monkeypatch.setattr(lkh_mod, 'add_link_blockmap', lambda A: None)
    monkeypatch.setattr(lkh_mod, '_prune_bad_links', lambda A, limit: None)

    warnings = []
    monkeypatch.setattr(lkh_mod, 'warn', warnings.append)

    S = lkh_mod.iterative_lkh(
        A,
        capacity=2,
        time_limit=0.1,
        seed=1,
        max_retries=1,
    )

    assert S.graph['retries'] == 1
    assert warnings == [
        'Capacity violated in LKH solution: max_load (3) > capacity (2). Retrying.',
        'Capacity violated in LKH solution: max_load (3) > capacity (2). Retrying.',
        'Solution remains invalid (max_retries reached)',
    ]
