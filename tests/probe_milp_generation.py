"""Measure the cost of generating one MILP objective reference.

Run from the repository root, for example::

    python -m tests.probe_milp_generation neart 5

The probe builds the planar mesh, constructs a feasible warm start, attaches
the model to the selected backend, solves it, and decodes the best
model-objective topology. It does not route the solution through PathFinder or
write a golden artifact.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from math import ceil
from time import perf_counter
from typing import TypedDict

import networkx as nx

from optiwindnet.heuristics import constructor
from optiwindnet.interarraylib import as_normalized
from optiwindnet.MILP import ModelOptions, OWNWarmupFailed, solver_factory
from optiwindnet.MILP._core import feeder_and_load_bounds
from optiwindnet.terse import TerseLinks
from optiwindnet.types import Topology

from .cases import MILPCase
from .sitecache import get_bundle, SiteBundle
from .topology_assertions import assert_topology


class _HGSOptions(TypedDict):
    capacity: float
    time_limit: float
    vehicles: int | None
    vehicles_exact: bool
    seed: int | None
    repair: bool
    max_retries: int
    balanced: bool
    ringed: bool


def milp_reference_parser(
    *, positional_required: bool = True
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    nargs = None if positional_required else '?'
    parser.add_argument('site', nargs=nargs, help='canonical location handle')
    parser.add_argument('capacity', nargs=nargs, type=int)
    parser.add_argument(
        '--solver-name',
        default='ortools.cp_sat',
        help='MILP backend passed to solver_factory (default: ortools.cp_sat)',
    )
    parser.add_argument(
        '--topology',
        choices=tuple(topology.value for topology in Topology),
        default=Topology.BRANCHED.value,
    )
    parser.add_argument(
        '--feeder-route',
        choices=('segmented', 'straight'),
        default='segmented',
    )
    parser.add_argument(
        '--feeder-limit',
        choices=(
            'unlimited',
            'minimum',
            'min_plus1',
            'min_plus2',
            'min_plus3',
            'exactly',
            'specified',
        ),
        default='unlimited',
    )
    parser.add_argument('--max-feeders', type=int, default=0)
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--time-limit', type=float, default=30.0)
    parser.add_argument('--mip-gap', type=float, default=1e-8)
    parser.add_argument('--cold', action='store_true', help='omit the warm start')
    parser.add_argument(
        '--warmstart-time-limit',
        type=float,
        default=0.3,
        help='HGS time limit per attempt',
    )
    parser.add_argument('--warmstart-seed', type=int, default=0)
    parser.add_argument(
        '--warmstart-max-retries',
        type=int,
        default=2,
        help='maximum HGS crossing-repair retries',
    )
    parser.add_argument('--verbose', action='store_true')
    return parser


def milp_case_from_args(args: argparse.Namespace) -> MILPCase:
    return MILPCase(
        site=args.site,
        solver_name=args.solver_name,
        capacity=args.capacity,
        model_options=ModelOptions(
            topology=args.topology,
            feeder_route=args.feeder_route,
            feeder_limit=args.feeder_limit,
            balanced=args.balanced,
            max_feeders=args.max_feeders,
        ),
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
    )


def _constructor_method(topology: Topology) -> str:
    return {
        Topology.BRANCHED: 'biased_EW',
        Topology.RADIAL: 'radial_EW',
        Topology.RINGED: 'ringed',
    }[topology]


def _hgs_kwargs(
    A: nx.Graph,
    case: MILPCase,
    *,
    warmstart_time_limit: float,
    warmstart_seed: int,
    warmstart_max_retries: int,
) -> _HGSOptions | None:
    """Return a feasible HGS configuration, or ``None`` if HGS cannot express it."""
    capacity = case.capacity
    options = case.model_options
    topology = options['topology']
    ringed = topology is Topology.RINGED
    feeders_per_subtree = 2 if ringed else 1
    solve_capacity = feeders_per_subtree * capacity
    feeders_min = ceil(A.graph['T'] / solve_capacity)
    feeders_lb, feeders_ub, _, _ = feeder_and_load_bounds(
        A.graph['T'],
        solve_capacity,
        options['feeder_limit'],
        options['max_feeders'],
        options['balanced'],
        feeders_per_subtree,
    )

    # HGS can pin a non-minimum vehicle count only by adding balancing slack
    # nodes. It cannot do so for rings or a multi-root problem.
    exact_above_minimum = feeders_lb == feeders_ub and feeders_lb > feeders_min
    if exact_above_minimum and (ringed or A.graph['R'] > 1):
        return None

    if feeders_ub is None:
        vehicles = None
    elif A.graph['R'] > 1 and feeders_lb != feeders_ub:
        # HGS distributes only the global minimum reliably across root clusters.
        vehicles = feeders_lb
    else:
        vehicles = feeders_ub

    vehicles_exact = exact_above_minimum
    balanced = (options['balanced'] and feeders_lb == feeders_ub) or vehicles_exact
    return {
        'capacity': capacity,
        'time_limit': warmstart_time_limit,
        'vehicles': vehicles,
        'vehicles_exact': vehicles_exact,
        'seed': warmstart_seed,
        'repair': True,
        'max_retries': warmstart_max_retries,
        'balanced': balanced,
        'ringed': ringed,
    }


def build_milp_warmstart(
    A: nx.Graph,
    case: MILPCase,
    *,
    warmstart_time_limit: float,
    warmstart_seed: int,
    warmstart_max_retries: int,
) -> tuple[nx.Graph, dict[str, object]]:
    """Choose a ModelOptions-compatible warm-start producer, preferring HGS."""
    capacity = case.capacity
    options = case.model_options
    topology = options['topology']
    hgs_kwargs = _hgs_kwargs(
        A,
        case,
        warmstart_time_limit=warmstart_time_limit,
        warmstart_seed=warmstart_seed,
        warmstart_max_retries=warmstart_max_retries,
    )
    if hgs_kwargs is not None:
        from optiwindnet.baselines.hgs import hgs_cvrp

        S = hgs_cvrp(as_normalized(A), **hgs_kwargs)
        # A radial topology is a valid seed for the less restrictive branched
        # model; warmup_model() explicitly supports this relationship.
        warmstart_topology = (
            Topology.RADIAL if topology is Topology.BRANCHED else topology
        )
        assert_topology(S, warmstart_topology, capacity)
        return S, {'generator': 'hgs', 'options': dict(hgs_kwargs)}

    return _build_constructor_warmstart(A, case)


def _build_constructor_warmstart(
    A: nx.Graph,
    case: MILPCase,
) -> tuple[nx.Graph, dict[str, object]]:
    """Build the topology-appropriate deterministic fallback warm start."""
    capacity = case.capacity
    options = case.model_options
    topology = options['topology']
    method = _constructor_method(topology)
    weigh_detours = options['feeder_route'].value == 'segmented'
    straight_feeder_route = options['feeder_route'].value == 'straight'
    S = constructor(
        A,
        capacity=capacity,
        method=method,
        weigh_detours=weigh_detours,
        straight_feeder_route=straight_feeder_route,
    )
    assert_topology(S, topology, capacity)
    return S, {
        'generator': 'constructor',
        'options': {
            'capacity': capacity,
            'method': method,
            'weigh_detours': weigh_detours,
            'straight_feeder_route': straight_feeder_route,
        },
    }


def solve_milp_reference(
    case: MILPCase,
    *,
    bundle: SiteBundle | None = None,
    cold: bool = False,
    warmstart_time_limit: float = 0.3,
    warmstart_seed: int = 0,
    warmstart_max_retries: int = 2,
    verbose: bool = False,
) -> dict[str, object]:
    """Solve one reference candidate without routing or writing an artifact.

    Passing a :class:`SiteBundle` lets callers reuse the same location, planar
    embedding, and available-links graph across cases. If omitted, the shared
    :func:`tests.sitecache.get_bundle` cache supplies it.
    """
    started = perf_counter()
    topology = case.model_options['topology']
    options = case.model_options

    phase = perf_counter()
    if bundle is None:
        bundle = get_bundle(case.site)
    elif bundle.handle != case.site:
        raise ValueError(
            f'bundle handle {bundle.handle!r} does not match case site {case.site!r}'
        )
    bundle_fetch = perf_counter() - phase
    L, P, A = bundle.L, bundle.P, bundle.A

    warmstart = None
    warmstart_details: dict[str, object] = {'generator': 'none', 'options': {}}
    phase = perf_counter()
    if not cold:
        warmstart, warmstart_details = build_milp_warmstart(
            A,
            case,
            warmstart_time_limit=warmstart_time_limit,
            warmstart_seed=warmstart_seed,
            warmstart_max_retries=warmstart_max_retries,
        )
    warmstart_build = perf_counter() - phase

    phase = perf_counter()
    solver = solver_factory(case.solver_name)
    solver_init = perf_counter() - phase
    model_build = 0.0
    rejected_warmstarts: list[dict[str, str]] = []
    while True:
        phase = perf_counter()
        try:
            solver.set_problem(
                P,
                A,
                capacity=case.capacity,
                model_options=options,
                warmstart=warmstart,
            )
        except OWNWarmupFailed as exc:
            model_build += perf_counter() - phase
            generator = str(warmstart_details['generator'])
            rejected_warmstarts.append({'generator': generator, 'error': str(exc)})
            if generator == 'hgs':
                phase = perf_counter()
                warmstart, warmstart_details = _build_constructor_warmstart(A, case)
                warmstart_build += perf_counter() - phase
            elif generator == 'constructor':
                warmstart = None
                warmstart_details = {'generator': 'none', 'options': {}}
            else:
                raise
            phase = perf_counter()
            solver = solver_factory(case.solver_name)
            solver_init += perf_counter() - phase
            continue
        model_build += perf_counter() - phase
        break
    if rejected_warmstarts:
        warmstart_details['rejected'] = rejected_warmstarts

    model = getattr(solver, 'model', None)
    if model is not None and hasattr(model, 'component_objects'):
        import pyomo.environ as pyo

        constraints = sum(
            len(c) for c in model.component_objects(pyo.Constraint, active=True)
        )
    elif model is not None and hasattr(model, 'linear_constraints'):
        constraints = sum(1 for _ in model.linear_constraints())
    else:
        constraints = 0

    model_stats = {
        'link_variables': len(solver.metadata.link_),
        'flow_variables': len(solver.metadata.flow_),
        'variables': len(solver.metadata.link_) + len(solver.metadata.flow_),
        'constraints': constraints,
    }

    phase = perf_counter()
    info = solver.solve(
        time_limit=case.time_limit,
        mip_gap=case.mip_gap,
        verbose=verbose,
    )
    solve = perf_counter() - phase

    phase = perf_counter()
    S = solver.get_incumbent_topology()
    assert_topology(S, topology, case.capacity)
    decode_and_validate = perf_counter() - phase

    solution = asdict(info)
    solution.update(
        objective=float(info.objective),
        bound=float(info.bound),
        relgap=float(info.relgap),
        topology_edges=S.number_of_edges(),
    )
    return {
        'case': {
            'site': case.site,
            'capacity': case.capacity,
            'solver_name': case.solver_name,
            'model_options': {
                name: value.value if hasattr(value, 'value') else value
                for name, value in options.items()
            },
            'warmstart': warmstart is not None,
            'warmstart_details': warmstart_details,
            'time_limit': case.time_limit,
            'mip_gap': case.mip_gap,
        },
        'site': {
            'T': L.graph['T'],
            'R': L.graph['R'],
            'nodeset_digest': bundle.coordinate_digest.hex(),
        },
        'model': model_stats,
        'solution': solution,
        'solution_topology': TerseLinks.from_topology(
            S,
            nodeset_digest=bundle.coordinate_digest,
        ).to_dict(),
        'timings_s': {
            'bundle_fetch': bundle_fetch,
            'warmstart_build': warmstart_build,
            'solver_init': solver_init,
            'model_build': model_build,
            'solve': solve,
            'decode_and_validate': decode_and_validate,
            'total': perf_counter() - started,
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    """Compatibility wrapper for callers that already have parsed CLI arguments."""
    return solve_milp_reference(
        milp_case_from_args(args),
        cold=args.cold,
        warmstart_time_limit=args.warmstart_time_limit,
        warmstart_seed=args.warmstart_seed,
        warmstart_max_retries=args.warmstart_max_retries,
        verbose=args.verbose,
    )


def main() -> None:
    args = milp_reference_parser().parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
