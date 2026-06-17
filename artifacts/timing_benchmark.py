#!/usr/bin/env python3
"""Timing benchmark for turbine_power runtime drivers.

Runs MILP optimization for 20 sites across multiple power patterns and saves
results to artifacts/timing_benchmark_results.json.

Usage:
    python artifacts/timing_benchmark.py

Results are consumed by docs/notebooks/a11_weighted_turbine_power.ipynb.
"""

import json
import logging
import math
import os
import platform
import random
import hashlib
import re
import shutil
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from optiwindnet.api import MILPRouter, WindFarmNetwork, _normalize_turbine_power
from optiwindnet.importer import L_from_yaml
from optiwindnet.MILP import ModelOptions
from optiwindnet.mesh import make_planar_embedding

DATA_DIR = Path(__file__).parent.parent / 'optiwindnet' / 'data'
OUTPUT = Path(__file__).parent / 'timing_benchmark_results.json'
API_LOGGER = logging.getLogger('optiwindnet.api')
SCHEMA_VERSION = 6
MODEL_VARIANT = 'continuous_power_flow_v1'
SOLVER_NAME = 'ortools.highs'
SOLVER_NAMES = tuple(
    item.strip()
    for item in os.environ.get('OWN_BENCH_SOLVERS', SOLVER_NAME).split(',')
    if item.strip()
)
MODEL_OPTIONS = ModelOptions(continuous_power_flow=True)
TURBINE_POWER_PRECISION = int(os.environ.get('OWN_BENCH_TURBINE_POWER_PRECISION', 10))
PRECISION_SEMANTICS = 'scale_factor_v1'

SITES = [
    'Ormonde',  # T=30
    'Horns Rev 3',  # T=49
    'Cazzaro-2022',  # T=50
    'Walney 1',  # T=51
    'Borkum Riffgrund 2',  # T=52
    'Gode Wind 1',  # T=55
    'Moray West',  # T=60
    'Dudgeon',  # T=67
    'Borkum Riffgrund 1',  # T=78
    'Butendiek',  # T=80
    'Walney Extension',  # T=87
    'Rødsand 2',  # T=90
    'Dogger Bank A',  # T=95
    'Moray East',  # T=100
    'East Anglia ONE',  # T=102
    'Anholt',  # T=111
    'Rampion',  # T=116
    'Taylor-2023',  # T=122
    'Cazzaro-2022G-140',  # T=140
    'Borssele',  # T=173
]

# Each entry is a base pair. Case families below reuse the same pair/multiset in
# different spatial/index arrangements to separate value effects from placement.
PAIRS = [
    (1.0, 1.0),
    (1.0, 2.0),
    (1.0, 3),
    (1.0, 5),
    (1.0, 1.9),
    (1.0, 1.5),
    (1.0, 1.2),
    (1.0, 1.01),
    (1.0, 0.99),
    (1.0, 0.9),
    (1.0, 0.5),
    (1.0, 0.1),
    (1.0, 0.05),
    (1.0, 0.01),
    (1.0, 1.001),
]

CASE_FAMILIES = (
    'block',
    'x_sorted',
    'near_root_heavy',
    'far_root_heavy',
    'random_seed_1',
    'random_seed_2',
)

PLACEMENT_PAIRS = (
    (1.0, 2.0),
    (1.0, 5),
    (1.0, 1.5),
    (1.0, 1.2),
    (1.0, 1.01),
    (1.0, 0.5),
)

SPARSE_CASES = (
    ('single_heavy_near_root', 5.0, 1),
    ('single_heavy_far_root', 5.0, 1),
    ('few_heavy_near_root', 5.0, 3),
    ('few_heavy_far_root', 5.0, 3),
)

TIME_LIMIT = 60
MIP_GAP = 0.02

CABLE_CONFIGS = [
    ('5', [(5, 1.0)]),
    ('10', [(10, 1.0)]),
]


def csv_env(name):
    return tuple(
        item.strip() for item in os.environ.get(name, '').split(',') if item.strip()
    )


def selected_sites():
    wanted = csv_env('OWN_BENCH_SITES')
    if not wanted:
        return SITES
    wanted_set = set(wanted)
    unknown = wanted_set - set(SITES)
    if unknown:
        raise ValueError(f'Unknown OWN_BENCH_SITES entries: {sorted(unknown)}')
    return [site for site in SITES if site in wanted_set]


def selected_cable_configs():
    wanted = csv_env('OWN_BENCH_CABLES')
    if not wanted:
        return CABLE_CONFIGS
    labels = {label for label, _ in CABLE_CONFIGS}
    unknown = set(wanted) - labels
    if unknown:
        raise ValueError(f'Unknown OWN_BENCH_CABLES entries: {sorted(unknown)}')
    return [(label, cables) for label, cables in CABLE_CONFIGS if label in wanted]


def selected_power_cases(L):
    cases = list(power_cases(L))
    pattern = os.environ.get('OWN_BENCH_CASE_REGEX')
    if pattern:
        case_re = re.compile(pattern)
        cases = [(label, powers) for label, powers in cases if case_re.search(label)]
    limit = os.environ.get('OWN_BENCH_CASE_LIMIT')
    if limit:
        cases = cases[: int(limit)]
    return cases


def case_result_key(case_label, solver_name):
    if len(SOLVER_NAMES) > 1 or os.environ.get('OWN_BENCH_INCLUDE_SOLVER_IN_CASE_KEY'):
        return f'{case_label} :: {solver_name}'
    return case_label


def base_case_label(case_label):
    return case_label.split(' :: ', maxsplit=1)[0]


def benchmark_config():
    sites = selected_sites()
    cable_configs = selected_cable_configs()
    return {
        'schema_version': SCHEMA_VERSION,
        'model_variant': MODEL_VARIANT,
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version,
        'platform': platform.platform(),
        'solver': SOLVER_NAMES[0] if len(SOLVER_NAMES) == 1 else None,
        'solvers': SOLVER_NAMES,
        'model_options': dict(MODEL_OPTIONS),
        'turbine_power_precision': TURBINE_POWER_PRECISION,
        'precision_semantics': PRECISION_SEMANTICS,
        'time_limit': TIME_LIMIT,
        'mip_gap': MIP_GAP,
        'sites': sites,
        'all_sites': SITES,
        'cable_configs': [
            {'label': label, 'cables': cables} for label, cables in cable_configs
        ],
        'available_cable_configs': [
            {'label': label, 'cables': cables} for label, cables in CABLE_CONFIGS
        ],
        'pairs': PAIRS,
        'placement_pairs': PLACEMENT_PAIRS,
        'case_families': CASE_FAMILIES,
        'sparse_cases': SPARSE_CASES,
        'case_regex': os.environ.get('OWN_BENCH_CASE_REGEX'),
    }


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def checksum(values):
    payload = json.dumps(jsonable(values), separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def _nearest_root_distances(L):
    T, R = L.graph['T'], L.graph['R']
    vertexC = L.graph['VertexC']
    turbinesC = vertexC[:T]
    rootsC = vertexC[-R:]
    return np.linalg.norm(turbinesC[:, None, :] - rootsC[None, :, :], axis=2).min(
        axis=1
    )


def make_available_graph(L):
    _, A = make_planar_embedding(L)
    return A


def site_metrics(L, A=None):
    T = L.graph['T']
    R = L.graph['R']
    vertexC = L.graph['VertexC']
    turbinesC = vertexC[:T]
    distances = _nearest_root_distances(L)

    try:
        if A is None:
            embedding_t0 = time.perf_counter()
            A = make_available_graph(L)
            embedding_time_s = round(time.perf_counter() - embedding_t0, 4)
        else:
            embedding_time_s = None
        available_nodes = A.number_of_nodes()
        available_edges = A.number_of_edges()
        available_terminal_edges = sum(1 for u, v in A.edges if u >= 0 and v >= 0)
    except Exception as exc:
        available_nodes = None
        available_edges = None
        available_terminal_edges = None
        embedding_time_s = None
        embedding_error = f'{type(exc).__name__}: {exc}'
    else:
        embedding_error = ''

    bbox_min = turbinesC.min(axis=0)
    bbox_max = turbinesC.max(axis=0)
    return {
        'schema_version': SCHEMA_VERSION,
        'R': R,
        'T': T,
        'B': L.graph.get('B', 0),
        'border_vertices': len(L.graph.get('border', [])),
        'obstacle_count': len(L.graph.get('obstacles', [])),
        'obstacle_vertices': sum(len(obs) for obs in L.graph.get('obstacles', [])),
        'bbox_width': float(bbox_max[0] - bbox_min[0]),
        'bbox_height': float(bbox_max[1] - bbox_min[1]),
        'mean_root_distance': float(distances.mean()),
        'median_root_distance': float(np.median(distances)),
        'min_root_distance': float(distances.min()),
        'max_root_distance': float(distances.max()),
        'std_root_distance': float(distances.std()),
        'available_nodes': available_nodes,
        'available_edges': available_edges,
        'available_terminal_edges': available_terminal_edges,
        'available_edge_density': (
            available_terminal_edges / T
            if available_terminal_edges is not None
            else None
        ),
        'embedding_time_s': embedding_time_s,
        'embedding_error': embedding_error,
    }


def available_link_power_metrics(A, int_powers):
    powers = np.asarray(int_powers, dtype=np.float64)
    T = A.graph['T']
    edge_rows = [
        (u, v, A[u][v].get('length', 1.0))
        for u, v in A.edges
        if 0 <= u < T and 0 <= v < T
    ]
    if not edge_rows:
        return {
            'available_power_edge_count': 0,
            'available_edge_power_absdiff_mean': 0.0,
            'available_edge_power_absdiff_max': 0.0,
            'available_edge_power_absdiff_weighted_mean': 0.0,
            'available_edge_same_power_share': 0.0,
            'available_edge_high_high_share': 0.0,
            'available_edge_low_low_share': 0.0,
            'available_edge_high_low_share': 0.0,
            'available_edge_power_product_mean': 0.0,
        }

    u_, v_, length_ = (np.asarray(col, dtype=np.float64) for col in zip(*edge_rows))
    u_ = u_.astype(np.int_)
    v_ = v_.astype(np.int_)
    absdiff = np.abs(powers[u_] - powers[v_])
    same_power = powers[u_] == powers[v_]
    median_power = np.median(powers)
    high_u = powers[u_] > median_power
    high_v = powers[v_] > median_power
    low_u = powers[u_] < median_power
    low_v = powers[v_] < median_power
    return {
        'available_power_edge_count': int(len(edge_rows)),
        'available_edge_power_absdiff_mean': float(absdiff.mean()),
        'available_edge_power_absdiff_max': float(absdiff.max()),
        'available_edge_power_absdiff_weighted_mean': float(
            np.average(absdiff, weights=length_)
        ),
        'available_edge_same_power_share': float(same_power.mean()),
        'available_edge_high_high_share': float((high_u & high_v).mean()),
        'available_edge_low_low_share': float((low_u & low_v).mean()),
        'available_edge_high_low_share': float(
            ((high_u & low_v) | (low_u & high_v)).mean()
        ),
        'available_edge_power_product_mean': float((powers[u_] * powers[v_]).mean()),
    }


@contextmanager
def api_logging_disabled():
    disabled = API_LOGGER.disabled
    API_LOGGER.disabled = True
    try:
        yield
    finally:
        API_LOGGER.disabled = disabled


def _pair_values(T, a, b):
    half = T // 2
    return [a] * half + [b] * (T - half)


def _heavy_first_values(T, a, b):
    half = T // 2
    count_a = half
    count_b = T - half
    if a >= b:
        return [a] * count_a + [b] * count_b
    return [b] * count_b + [a] * count_a


def make_power_case(L, family, a, b):
    T = L.graph['T']
    values = _pair_values(T, a, b)

    if family == 'alternating':
        return [a if i % 2 == 0 else b for i in range(T)]
    if family == 'block':
        return values
    if family == 'x_sorted':
        order = np.argsort(L.graph['VertexC'][:T, 0], kind='stable')
    elif family == 'near_root_heavy':
        order = np.argsort(_nearest_root_distances(L), kind='stable')
        values = _heavy_first_values(T, a, b)
    elif family == 'far_root_heavy':
        order = np.argsort(-_nearest_root_distances(L), kind='stable')
        values = _heavy_first_values(T, a, b)
    elif family.startswith('random_seed_'):
        seed = int(family.rsplit('_', maxsplit=1)[-1])
        order = list(range(T))
        random.Random(seed).shuffle(order)
    else:
        raise ValueError(f'Unknown case family: {family}')

    powers = [None] * T
    for turbine, power in zip(order, values, strict=True):
        powers[turbine] = power
    return powers


def make_sparse_case(L, family, heavy, count):
    T = L.graph['T']
    powers = [1.0] * T
    distances = _nearest_root_distances(L)
    if family.endswith('near_root'):
        order = np.argsort(distances, kind='stable')
    elif family.endswith('far_root'):
        order = np.argsort(-distances, kind='stable')
    else:
        raise ValueError(f'Unknown sparse case family: {family}')
    for turbine in order[: min(count, T)]:
        powers[turbine] = heavy
    return powers


def power_cases(L):
    for a, b in PAIRS:
        label = f'alternating [{a}, {b}]'
        yield label, make_power_case(L, 'alternating', a, b)

    for a, b in PLACEMENT_PAIRS:
        for family in CASE_FAMILIES:
            label = f'{family} [{a}, {b}]'
            yield label, make_power_case(L, family, a, b)

    for family, heavy, count in SPARSE_CASES:
        yield (
            f'{family} heavy={heavy:g} count={count}',
            make_sparse_case(L, family, heavy, count),
        )


def case_family(case_label):
    if case_label.startswith('alternating'):
        return 'alternating'
    if case_label.startswith('single_heavy'):
        return 'single_heavy'
    if case_label.startswith('few_heavy'):
        return 'few_heavy'
    return case_label.split(' [', maxsplit=1)[0]


def _safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.isclose(a.std(), 0.0) or np.isclose(b.std(), 0.0):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _gini(values):
    values = np.sort(np.asarray(values, dtype=np.float64))
    if np.isclose(values.sum(), 0.0):
        return 0.0
    index = np.arange(1, values.size + 1)
    return float(
        (2 * np.sum(index * values) / values.sum() - values.size - 1) / values.size
    )


def placement_metrics(L, int_powers):
    powers = np.asarray(int_powers, dtype=np.float64)
    distances = _nearest_root_distances(L)
    distance_order = np.argsort(distances, kind='stable')
    q = max(1, powers.size // 4)
    near_power = float(powers[distance_order[:q]].sum() / powers.sum())
    far_power = float(powers[distance_order[-q:]].sum() / powers.sum())
    mean_distance = float(distances.mean())
    weighted_mean_distance = float(np.average(distances, weights=powers))
    return {
        'power_gini': _gini(powers),
        'power_mean': float(powers.mean()),
        'power_std': float(powers.std()),
        'power_cv': float(powers.std() / powers.mean()) if powers.mean() else 0.0,
        'power_distance_corr': _safe_corr(powers, distances),
        'mean_root_distance': mean_distance,
        'weighted_mean_root_distance': weighted_mean_distance,
        'weighted_distance_ratio': (
            weighted_mean_distance / mean_distance if mean_distance else 0.0
        ),
        'near_quartile_power_share': near_power,
        'far_quartile_power_share': far_power,
    }


def load_results():
    if not OUTPUT.exists():
        return []
    if OUTPUT.stat().st_size == 0:
        return []
    with open(OUTPUT) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            corrupt = OUTPUT.with_suffix(
                OUTPUT.suffix
                + f'.corrupt-{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}'
            )
            os.replace(OUTPUT, corrupt)
            print(f'Moved invalid results JSON to {corrupt}; starting fresh.')
            return []


def save_results(results):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(OUTPUT.suffix + f'.{os.getpid()}.tmp')
    with open(tmp, 'w') as f:
        json.dump(jsonable(results), f, indent=2)
        f.write('\n')
    if OUTPUT.exists() and OUTPUT.stat().st_size > 0:
        shutil.copy2(OUTPUT, OUTPUT.with_suffix(OUTPUT.suffix + '.bak'))
    os.replace(tmp, OUTPUT)


def site_result_by_name(results, site_name, T):
    for site_result in results:
        if site_result['site'] == site_name:
            site_result['T'] = T
            site_result['benchmark_config'] = benchmark_config()
            return site_result
    site_result = {
        'site': site_name,
        'T': T,
        'benchmark_config': benchmark_config(),
        'cables': {},
    }
    results.append(site_result)
    return site_result


def power_metrics(powers, int_powers, nominal_capacity, scale):
    nominal_total = sum(powers)
    nominal_capacity_remainder = math.fmod(nominal_total, nominal_capacity)
    if math.isclose(
        nominal_capacity_remainder, nominal_capacity, rel_tol=1e-9, abs_tol=1e-9
    ):
        nominal_capacity_remainder = 0.0
    integer_total = sum(int_powers)
    integer_max_power = max(int_powers)
    scaled_capacity = nominal_capacity * scale
    nominal_feeder_lower_bound = math.ceil(nominal_total / nominal_capacity)
    integer_feeder_lower_bound = math.ceil(integer_total / scaled_capacity)
    integer_feeder_lower_bound_by_count = math.ceil(len(int_powers) / scaled_capacity)
    return {
        'nominal_total_power': nominal_total,
        'nominal_min_power': min(powers),
        'nominal_max_power': max(powers),
        'nominal_mean_power': float(np.mean(powers)),
        'nominal_distinct_power_count': len(set(powers)),
        'nominal_feeder_lower_bound': nominal_feeder_lower_bound,
        'nominal_capacity_remainder': nominal_capacity_remainder,
        'nominal_over_capacity_turbines': sum(p > nominal_capacity for p in powers),
        'integer_scaling_factor': scale,
        'integer_total_power': integer_total,
        'integer_min_power': min(int_powers),
        'integer_max_power': integer_max_power,
        'integer_distinct_power_count': len(set(int_powers)),
        'scaled_capacity': scaled_capacity,
        'integer_feeder_lower_bound_by_power': integer_feeder_lower_bound,
        'integer_feeder_lower_bound_by_scaled_count': (
            integer_feeder_lower_bound_by_count
        ),
        'integer_capacity_remainder': integer_total % scaled_capacity,
        'integer_over_capacity_turbines': sum(p > scaled_capacity for p in int_powers),
        # Legacy aliases retained for older ad-hoc analysis scripts.
        'total_power': integer_total,
        'max_power': integer_max_power,
        'min_power': min(int_powers),
        'distinct_power_count': len(set(int_powers)),
        'feeder_lower_bound_by_power': nominal_feeder_lower_bound,
        'feeder_lower_bound_by_count_nominal': math.ceil(
            len(int_powers) / nominal_capacity
        ),
        'feeder_lower_bound_by_scaled_count': integer_feeder_lower_bound_by_count,
        'feeder_lower_bound_used_by_model': nominal_feeder_lower_bound,
        'feeder_lower_bound_gap': nominal_feeder_lower_bound
        - integer_feeder_lower_bound_by_count,
        'capacity_remainder': nominal_capacity_remainder,
        'over_capacity_turbines': sum(p > nominal_capacity for p in powers),
    }


def diagnostic_metrics(A, L, case_label, powers, int_powers, nominal_capacity, scale):
    return {
        'schema_version': SCHEMA_VERSION,
        'model_variant': MODEL_VARIANT,
        'case_label': case_label,
        'case_family': case_family(case_label),
        'nominal_powers': jsonable(powers),
        'integer_powers': jsonable(int_powers),
        'nominal_power_checksum': checksum(powers),
        'integer_power_checksum': checksum(int_powers),
        'nominal_capacity': nominal_capacity,
        **power_metrics(powers, int_powers, nominal_capacity, scale),
        **placement_metrics(L, powers),
        **available_link_power_metrics(A, powers),
    }


def solution_metrics(wfn):
    info = {}
    try:
        info.update(wfn.solution_info())
    except Exception as exc:
        info['solution_info_error'] = f'{type(exc).__name__}: {exc}'

    try:
        G = wfn.G
        S = wfn.S
        R = G.graph['R']
        roots = range(-R, 0)
        info.update(
            {
                'solution_length': wfn.length(),
                'solution_cost': wfn.cost(),
                'selected_edge_count': S.number_of_edges(),
                'route_edge_count': G.number_of_edges(),
                'feeder_count': sum(G.degree[root] for root in roots),
                'max_load': G.graph.get('max_load'),
                'capacity': G.graph.get('capacity'),
                'max_load_minus_capacity': (
                    G.graph.get('max_load') - G.graph.get('capacity')
                    if G.graph.get('max_load') is not None
                    and G.graph.get('capacity') is not None
                    else None
                ),
                'route_clone_count': G.graph.get('C', 0) + G.graph.get('D', 0),
                'contour_clone_count': G.graph.get('C', 0),
                'detour_clone_count': G.graph.get('D', 0),
            }
        )
    except Exception as exc:
        info['graph_summary_error'] = f'{type(exc).__name__}: {exc}'
    return jsonable(info)


def solver_runtime(solution):
    if not isinstance(solution, dict):
        return None
    runtime = solution.get('runtime')
    return float(runtime) if runtime is not None else None


def prompt_continue(site_name, next_site_name):
    if os.environ.get('OWN_BENCH_NO_PROMPT'):
        return True
    if not sys.stdin.isatty():
        print(
            f'\nFinished {site_name}. Non-interactive input detected; '
            f'continuing with {next_site_name}.'
        )
        return True

    try:
        answer = input(
            f'\nFinished {site_name}. Continue with {next_site_name}? [Y/n] '
        ).strip()
    except EOFError:
        print(f'\nNo input available; continuing with {next_site_name}.')
        return True

    return answer.lower() not in {'n', 'no', 'q', 'quit', 's', 'stop'}


def should_skip_existing_case(existing_case, solver_name):
    if existing_case is None:
        return False
    if existing_case.get('model_variant') != MODEL_VARIANT:
        return False
    if existing_case.get('solver') != solver_name:
        return False
    if existing_case.get('model_options') != dict(MODEL_OPTIONS):
        return False
    if existing_case.get('turbine_power_precision') != TURBINE_POWER_PRECISION:
        return False
    if existing_case.get('precision_semantics') != PRECISION_SEMANTICS:
        return False
    status = existing_case.get('status', 'completed')
    if status == 'completed':
        return True
    if status == 'error' and not os.environ.get('OWN_BENCH_RETRY_ERRORS'):
        return True
    return False


def enrich_existing_results():
    results = load_results()
    if not results:
        print(f'No results found at {OUTPUT}')
        return

    for site_result in results:
        site_name = site_result['site']
        site_path = DATA_DIR / f'{site_name}.yaml'
        L = L_from_yaml(site_path)
        A = make_available_graph(L)
        T = L.graph['T']
        site_result['benchmark_config'] = benchmark_config()
        site_result['site_metrics'] = site_metrics(L, A)
        cases = dict(power_cases(L))

        for cable_label, cables in selected_cable_configs():
            cable_result = site_result.get('cables', {}).get(cable_label)
            if cable_result is None:
                continue
            nominal_capacity = max(capacity for capacity, _ in cables)

            for case_label, case_result in cable_result.get('cases', {}).items():
                powers = cases[base_case_label(case_label)]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    with api_logging_disabled():
                        int_powers, scale = _normalize_turbine_power(
                            powers,
                            T,
                            TURBINE_POWER_PRECISION,
                        )
                case_result.update(
                    {
                        'scale': scale,
                        'case_result_key': case_label,
                        'status': case_result.get('status', 'completed'),
                        **diagnostic_metrics(
                            A,
                            L,
                            case_label,
                            powers,
                            int_powers,
                            nominal_capacity,
                            scale,
                        ),
                    }
                )

    save_results(results)
    print(f'Enriched existing results in {OUTPUT}')


def run_benchmark():
    results = load_results()
    sites = selected_sites()

    for site_index, site_name in enumerate(sites):
        site_path = DATA_DIR / f'{site_name}.yaml'
        L = L_from_yaml(site_path)
        A = make_available_graph(L)
        T = L.graph['T']
        print(f'\n{site_name} (T={T})')

        site_result = site_result_by_name(results, site_name, T)
        site_result['site_metrics'] = site_metrics(L, A)
        cases = selected_power_cases(L)

        for cable_label, cables in selected_cable_configs():
            print(f'  cables={cable_label}')
            nominal_capacity = max(capacity for capacity, _ in cables)
            cable_result = site_result['cables'].setdefault(cable_label, {'cases': {}})

            for case_label, powers in cases:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    with api_logging_disabled():
                        int_powers, scale = _normalize_turbine_power(
                            powers,
                            T,
                            TURBINE_POWER_PRECISION,
                        )

                metrics = diagnostic_metrics(
                    A, L, case_label, powers, int_powers, nominal_capacity, scale
                )

                for solver_name in SOLVER_NAMES:
                    result_key = case_result_key(case_label, solver_name)
                    existing_case = cable_result['cases'].get(result_key)
                    if should_skip_existing_case(existing_case, solver_name):
                        existing_case.update(
                            {
                                'scale': scale,
                                'case_result_key': result_key,
                                'status': existing_case.get('status', 'completed'),
                                'turbine_power_precision': TURBINE_POWER_PRECISION,
                                'precision_semantics': PRECISION_SEMANTICS,
                                **metrics,
                            }
                        )
                        save_results(results)
                        status = existing_case.get('status', 'completed')
                        print(f'    {result_key:<55} skipped ({status})')
                        continue

                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        with api_logging_disabled():
                            setup_t0 = time.perf_counter()
                            wfn = WindFarmNetwork(
                                cables=cables,
                                L=L_from_yaml(site_path),
                                turbine_power=powers,
                                turbine_power_precision=TURBINE_POWER_PRECISION,
                                router=MILPRouter(
                                    solver_name,
                                    time_limit=TIME_LIMIT,
                                    mip_gap=MIP_GAP,
                                    model_options=MODEL_OPTIONS,
                                ),
                            )
                            setup_time = round(time.perf_counter() - setup_t0, 4)

                    started_at = datetime.now(timezone.utc).isoformat()
                    t0 = time.perf_counter()
                    try:
                        wfn.optimize()
                    except Exception as exc:
                        elapsed = round(time.perf_counter() - t0, 2)
                        cable_result['cases'][result_key] = {
                            'scale': scale,
                            'case_result_key': result_key,
                            'status': 'error',
                            'model_variant': MODEL_VARIANT,
                            'solver': solver_name,
                            'model_options': dict(MODEL_OPTIONS),
                            'turbine_power_precision': TURBINE_POWER_PRECISION,
                            'precision_semantics': PRECISION_SEMANTICS,
                            'started_at_utc': started_at,
                            'ended_at_utc': datetime.now(timezone.utc).isoformat(),
                            'time_s': elapsed,
                            'hit_limit': elapsed >= TIME_LIMIT - 1,
                            'error_type': type(exc).__name__,
                            'error_message': str(exc),
                            **metrics,
                        }
                        save_results(results)
                        print(f'    {result_key:<55} ERROR {type(exc).__name__}: {exc}')
                        continue

                    elapsed = round(time.perf_counter() - t0, 2)
                    hit_limit = elapsed >= TIME_LIMIT - 1
                    solution = solution_metrics(wfn)
                    solver_time = solver_runtime(solution)

                    cable_result['cases'][result_key] = {
                        'scale': scale,
                        'case_result_key': result_key,
                        'status': 'completed',
                        'started_at_utc': started_at,
                        'ended_at_utc': datetime.now(timezone.utc).isoformat(),
                        'time_s': elapsed,
                        'optimize_time_s': elapsed,
                        'setup_time_s': setup_time,
                        'model_variant': MODEL_VARIANT,
                        'solver': solver_name,
                        'model_options': dict(MODEL_OPTIONS),
                        'turbine_power_precision': TURBINE_POWER_PRECISION,
                        'precision_semantics': PRECISION_SEMANTICS,
                        'solver_time_s': solver_time,
                        'non_solver_time_s': (
                            round(elapsed - solver_time, 4)
                            if solver_time is not None
                            else None
                        ),
                        'hit_limit': hit_limit,
                        'solution': solution,
                        **metrics,
                    }
                    save_results(results)
                    marker = ' (limit)' if hit_limit else ''
                    print(
                        f'    {result_key:<55} '
                        f'nom_lb={metrics["nominal_feeder_lower_bound"]:<3} '
                        f'nom_rem={metrics["nominal_capacity_remainder"]:<6g} '
                        f'int_scale={scale:<4} '
                        f'{elapsed:.2f}s{marker}'
                    )
        save_results(results)
        if site_index + 1 < len(sites):
            next_site_name = sites[site_index + 1]
            if not prompt_continue(site_name, next_site_name):
                print(f'\nStopped after {site_name}. Results saved to {OUTPUT}')
                return
    print(f'\nResults saved to {OUTPUT}')


if __name__ == '__main__':
    if os.environ.get('OWN_BENCH_ENRICH_ONLY'):
        enrich_existing_results()
    else:
        run_benchmark()
