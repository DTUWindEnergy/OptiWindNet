"""Typed low-level producer cases with attribute-derived names and golden keys."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from optiwindnet.heuristics.constructor import _METHOD_TOPOLOGY
from optiwindnet.MILP import ModelOptions
from optiwindnet.MILP._core import FeederRoute
from optiwindnet.types import Topology


@dataclass(frozen=True, slots=True)
class ConstructorCase:
    site: str
    capacity: int
    method: str
    feeder_route: FeederRoute = FeederRoute.SEGMENTED
    exact_golden: bool = False
    bias_margin: float | None = None


@dataclass(frozen=True, slots=True)
class BaselineCase:
    producer: Literal['hgs', 'lkh']
    site: str
    capacity: int
    ringed: bool = False
    balanced: bool = False
    seed: int = 0
    time_limit: float = 0.2


def expected_topology(case: ConstructorCase | BaselineCase) -> Topology:
    """Derive the expected output topology from producer input options."""
    if isinstance(case, ConstructorCase):
        return _METHOD_TOPOLOGY[case.method]
    return Topology.RINGED if case.ringed else Topology.RADIAL


@dataclass(frozen=True, slots=True)
class MILPCase:
    site: str
    solver_name: str
    capacity: int
    model_options: ModelOptions = field(default_factory=ModelOptions)
    time_limit: float = 10.0
    mip_gap: float = 1e-3
    exact_golden: bool = False


Case = ConstructorCase | BaselineCase | MILPCase


def _token(value: object) -> str:
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return 'none'
    if isinstance(value, float):
        return format(value, '.12g').replace('-', 'm').replace('.', 'p')
    return str(value).lower().replace('_', '-').replace('.', '-')


def case_node_id(case: Case) -> str:
    """Build a concise pytest node name from one case's execution attributes."""
    if isinstance(case, ConstructorCase):
        tokens = [
            case.site,
            _token(case.method),
            expected_topology(case).value,
            case.feeder_route.value,
            f'cap{case.capacity}',
        ]
        if case.bias_margin is not None:
            tokens.append(f'bias{_token(case.bias_margin)}')
        return '-'.join(tokens)

    if isinstance(case, BaselineCase):
        tokens = [
            case.producer,
            case.site,
            'ringed' if case.ringed else 'radial',
            f'cap{case.capacity}',
        ]
        if case.balanced:
            tokens.append('balanced')
        if case.seed:
            tokens.append(f'seed{case.seed}')
        if case.time_limit != 0.2:
            tokens.append(f'tl{_token(case.time_limit)}')
        return '-'.join(tokens)

    options = case.model_options
    tokens = [
        _token(case.solver_name),
        case.site,
        _token(options['topology']),
        _token(options['feeder_route']),
        _token(options['feeder_limit']),
        f'cap{case.capacity}',
    ]
    defaults = ModelOptions()
    aliases = {'max_feeders': 'maxf'}
    for name, value in options.items():
        if name in {'topology', 'feeder_route', 'feeder_limit'}:
            continue
        if value == defaults.get(name):
            continue
        if isinstance(value, bool):
            tokens.append(name.replace('_', '-') if value else f'no-{name}')
        else:
            tokens.append(f'{aliases.get(name, name)}{_token(value)}')
    if case.time_limit != 10.0:
        tokens.append(f'tl{_token(case.time_limit)}')
    if case.mip_gap != 1e-3:
        tokens.append(f'gap{_token(case.mip_gap)}')
    return '-'.join(tokens)


def topology_golden_key(case: ConstructorCase | MILPCase) -> str:
    """Identify an expected topology independently of its producing backend."""
    if isinstance(case, ConstructorCase):
        tokens = [
            'constructor',
            f'site-{case.site}',
            f'capacity-{case.capacity}',
            f'method-{_token(case.method)}',
            f'feeder-route-{case.feeder_route.value}',
            f'bias-margin-{_token(case.bias_margin)}',
        ]
        return '-'.join(tokens)

    tokens = ['milp', f'site-{case.site}', f'capacity-{case.capacity}']
    tokens.extend(
        f'{name.replace("_", "-")}-{_token(value)}'
        for name, value in case.model_options.items()
    )
    return '-'.join(tokens)


CONSTRUCTOR_CASES = (
    ConstructorCase('cazzaro_2022', 3, 'esau_williams'),
    ConstructorCase('cazzaro_2022', 5, 'biased_EW', exact_golden=True),
    ConstructorCase('morayeast', 8, 'rootlust'),
    ConstructorCase('morayeast', 5, 'radial_EW'),
    ConstructorCase('albatros', 3, 'ringed', exact_golden=True),
    ConstructorCase('example_location', 1, 'ringed'),
    ConstructorCase('neart', 5, 'ringed', bias_margin=0.1),
    ConstructorCase('london', 8, 'rootlust'),
    ConstructorCase('london', 10, 'biased_EW'),
    ConstructorCase(
        'cazzaro_2022',
        5,
        'biased_EW',
        feeder_route=FeederRoute.STRAIGHT,
    ),
)

HGS_CASES = (
    BaselineCase('hgs', 'example_location', 3),
    BaselineCase('hgs', 'example_location', 4, balanced=True),
    BaselineCase('hgs', 'example_location', 3, ringed=True),
    BaselineCase('hgs', 'morayeast', 8),
    BaselineCase('hgs', 'london', 10, ringed=True, time_limit=1.0),
    BaselineCase('hgs', 'london', 10, time_limit=1.0),
)

LKH_CASES = (
    BaselineCase('lkh', 'example_location', 3, time_limit=2),
    BaselineCase('lkh', 'example_location', 3, ringed=True, time_limit=2),
)

MILP_FORMULATION_CASES = tuple(
    MILPCase(
        'toy',
        'ortools.cp_sat',
        5,
        ModelOptions(topology=topology, feeder_route=route),
        exact_golden=topology is Topology.BRANCHED and route is FeederRoute.SEGMENTED,
    )
    for topology in (Topology.BRANCHED, Topology.RADIAL, Topology.RINGED)
    for route in (FeederRoute.SEGMENTED, FeederRoute.STRAIGHT)
)

MILP_ADAPTER_CASES = tuple(
    MILPCase('toy', solver, 5, ModelOptions(), exact_golden=True)
    for solver in (
        'ortools.cp_sat',
        'ortools.gscip',
        'ortools.highs',
        'highs',
        'scip',
        'gurobi',
        'cplex',
        'cbc',
        'fscip',
    )
)

MILP_FAMILY_CASES = (
    MILPCase(
        'toy', 'highs', 5, ModelOptions(topology='ringed', feeder_route='straight')
    ),
    MILPCase(
        'toy', 'scip', 5, ModelOptions(topology='ringed', feeder_route='straight')
    ),
    MILPCase(
        'toy', 'highs', 5, ModelOptions(topology='radial', feeder_route='straight')
    ),
    MILPCase(
        'toy', 'scip', 5, ModelOptions(topology='radial', feeder_route='straight')
    ),
)

MILP_BOUNDARY_CASES = (
    MILPCase('toy', 'ortools.cp_sat', 1, ModelOptions()),
    MILPCase('toy', 'ortools.cp_sat', 12, ModelOptions()),
    MILPCase('toy', 'ortools.cp_sat', 5, ModelOptions(feeder_limit='minimum')),
    MILPCase('toy', 'ortools.cp_sat', 5, ModelOptions(feeder_limit='min_plus1')),
    MILPCase(
        'toy',
        'ortools.cp_sat',
        5,
        ModelOptions(feeder_limit='specified', max_feeders=4),
    ),
    MILPCase(
        'toy',
        'ortools.cp_sat',
        5,
        ModelOptions(feeder_limit='exactly', max_feeders=3, balanced=True),
    ),
    MILPCase('toy', 'ortools.cp_sat', 5, ModelOptions(), mip_gap=0.2),
)


def golden_keys() -> frozenset[str]:
    keys = {
        topology_golden_key(case) for case in CONSTRUCTOR_CASES if case.exact_golden
    }
    keys.update(
        topology_golden_key(case)
        for case in (*MILP_FORMULATION_CASES, *MILP_ADAPTER_CASES)
        if case.exact_golden
    )
    return frozenset(keys)
