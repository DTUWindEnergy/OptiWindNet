"""
Generate expected graphs for specified sites-routers.
"""

from __future__ import annotations

import gc
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import dill

from optiwindnet.api import EWRouter, HGSRouter, MILPRouter, WindFarmNetwork
from optiwindnet.importer import load_repository
from optiwindnet.MILP import ModelOptions

# =============================================================================
# Config
# =============================================================================

REPO_ROOT = (
    Path(__file__).resolve().parents[1]
    if (Path(__file__).parent.name == 'scripts')
    else Path.cwd()
)
DEFAULT_OUTPUT = REPO_ROOT / 'tests' / 'test_files' / 'expected_end_to_end.dill'


def r_spec(
    cls: Optional[str], params: Optional[Dict[str, Any]] = None, cables: int = 1
) -> Dict[str, Any]:
    return {'class': cls, 'params': (params or {}), 'cables': int(cables)}


# === Sites as plain names ===
SITES_1: Sequence[str] = (
    'example_location',
    #"albatros",
)

model_options_strict = {
    'topology': 'radial',
    'feeder_limit': 'minimum',
    'feeder_route': 'straight',
}
ROUTERS_1: Dict[str, Dict[str, Any]] = {
    'EWRouter_cap1': r_spec('EWRouter', cables=1),
    'EWRouter_cap3': r_spec('EWRouter', cables=3),
    'EWRouter_cap10': r_spec('EWRouter', cables=10),
    'EWRouter_straight_cap1': r_spec(
        'EWRouter', {'feeder_route': 'straight'}, cables=1
    ),
    'EWRouter_straight_cap4': r_spec(
        'EWRouter', {'feeder_route': 'straight'}, cables=4
    ),
    'EWRouter_straight_cap10': r_spec(
        'EWRouter', {'feeder_route': 'straight'}, cables=10
    ),
    'HGSRouter_cap1': r_spec('HGSRouter', {'time_limit': 0.5, 'seed': 0}, cables=1),
    'HGSRouter_cap3': r_spec('HGSRouter', {'time_limit': 0.5, 'seed': 0}, cables=3),
    'HGSRouter_cap10': r_spec('HGSRouter', {'time_limit': 0.5, 'seed': 0}, cables=10),
    'HGSRouter_feeder_limit_cap1': r_spec(
        'HGSRouter', {'time_limit': 0.5, 'feeder_limit': 0, 'seed': 0}, cables=1
    ),
    'HGSRouter_feeder_limit_cap4': r_spec(
        'HGSRouter', {'time_limit': 0.5, 'feeder_limit': 0, 'seed': 0}, cables=4
    ),
    'HGSRouter_feeder_limit_cap10': r_spec(
        'HGSRouter', {'time_limit': 0.5, 'feeder_limit': 0, 'seed': 0}, cables=10
    ),
    # Enable MILP if solver available:
    'MILPRouter_ortools_cap5': r_spec(
        'MILPRouter',
        {'solver_name': 'ortools', 'time_limit': 5, 'mip_gap': 1e-3},
        cables=5,
    ),
    'MILPRouter_gurobi_cap4': r_spec(
        'MILPRouter',
        {'solver_name': 'gurobi', 'time_limit': 5, 'mip_gap': 1e-3},
        cables=4,
    ),
    'MILPRouter_highs_cap3': r_spec(
        'MILPRouter',
        {'solver_name': 'highs', 'time_limit': 5, 'mip_gap': 1e-3},
        cables=3,
    ),
    'MILPRouter_cplex_cap2': r_spec(
        'MILPRouter',
        {'solver_name': 'cplex', 'time_limit': 5, 'mip_gap': 1e-3},
        cables=2,
    ),
    'MILPRouter_ortools_cap10_modeloptions': r_spec(
        'MILPRouter',
        {
            'solver_name': 'ortools',
            'time_limit': 5,
            'mip_gap': 1e-3,
            'model_options': model_options_strict,
        },
        cables=10,
    ),
    'MILPRouter_gurobi_cap9_modeloptions': r_spec(
        'MILPRouter',
        {
            'solver_name': 'gurobi',
            'time_limit': 5,
            'mip_gap': 1e-3,
            'model_options': model_options_strict,
        },
        cables=9,
    ),
    'MILPRouter_highs_cap8_modeloptions': r_spec(
        'MILPRouter',
        {
            'solver_name': 'highs',
            'time_limit': 5,
            'mip_gap': 1e-3,
            'model_options': model_options_strict,
        },
        cables=8,
    ),
    'MILPRouter_cplex_cap7_modeloptions': r_spec(
        'MILPRouter',
        {
            'solver_name': 'cplex',
            'time_limit': 5,
            'mip_gap': 1e-3,
            'model_options': model_options_strict,
        },
        cables=7,
    ),
}

# add more groups
SITES_2: Sequence[str] = ('hornsea', 'london', 'taylor_2023', 'yi_2019')
ROUTERS_2: Dict[str, Dict[str, Any]] = {
    'EWRouter_cap5': r_spec('EWRouter', cables=5),
    'EWRouter_cap10': r_spec('EWRouter', cables=10),
    'EWRouter_cap100': r_spec('EWRouter', cables=100),
    'EWRouter_straight_cap7': r_spec(
        'EWRouter', {'feeder_route': 'straight'}, cables=7
    ),
    'EWRouter_straight_cap15': r_spec(
        'EWRouter', {'feeder_route': 'straight'}, cables=15
    ),
    'EWRouter_straight_cap300': r_spec(
        'EWRouter', {'feeder_route': 'straight'}, cables=300
    ),
    'HGSRouter_cap2': r_spec('HGSRouter', {'time_limit': 2, 'seed': 0}, cables=2),
    #'HGSRouter_cap20': r_spec('HGSRouter', {'time_limit': 5, 'seed': 0}, cables=20),
    # ...
}

SITES_3: Sequence[str] = ()
ROUTERS_3: Dict[str, Dict[str, Any]] = {}

# =============================================================================
# Helpers
# =============================================================================


def make_model_options_from_spec(spec: Dict[str, Any]) -> ModelOptions:
    return ModelOptions(**spec)


def make_router_from_spec(spec: Optional[Dict[str, Any]]):
    if spec is None:
        return None
    clsname = spec['class']
    params = dict(spec.get('params', {}))
    if clsname == 'MILPRouter' and isinstance(params.get('model_options'), dict):
        params['model_options'] = make_model_options_from_spec(params['model_options'])
    if clsname is None:
        return None
    if clsname == 'EWRouter':
        return EWRouter(**params)
    if clsname == 'HGSRouter':
        return HGSRouter(**params)
    if clsname == 'MILPRouter':
        return MILPRouter(**params)
    raise ValueError(f'Unknown router class: {clsname!r}')


def merge_router_specs(
    *spec_maps: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for m in spec_maps:
        out.update(m)
    return out


def print_header(title: str) -> None:
    print('\n' + '=' * 90)
    print(title)
    print('=' * 90)


def environment_meta() -> Dict[str, Any]:
    meta = {
        'generated_at_utc': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'platform': platform.platform(),
        'python': sys.version.split()[0],
        'executable': sys.executable,
    }
    for name in ('networkx', 'numpy', 'ortools', 'pyomo', 'gurobipy'):
        try:
            mod = __import__(name)
            ver = getattr(mod, '__version__', None)
            if ver:
                meta.setdefault('package_versions', {})[name] = ver
        except Exception:
            pass
    return meta


# =============================================================================
# Core runner
# =============================================================================


def generate_expected(output_path: Path) -> None:
    """
    Output:
      {
        "Sites":         (site_name, ...),
        "Routers":       {router_name: spec, ...},
        "Cases":         [{"key": ..., "site": ..., "router": ...}, ...],
        "RouterGraphs":  {key: networkx.Graph, ...},
        "Meta":          {...}
      }
    """
    print_header('Generating expected graphs')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.unlink(missing_ok=True)
        print(f'Removed (if existed): {output_path}')
    except Exception as e:
        print(f'Error removing old file: {e}')

    sites_union: List[str] = list(SITES_1) + list(SITES_2) + list(SITES_3)
    routers_union = merge_router_specs(ROUTERS_1, ROUTERS_2, ROUTERS_3)

    cases: List[Dict[str, str]] = []
    router_graphs: Dict[str, Any] = {}

    # load once; assume success and membership
    locations = load_repository(path=r'tests\test_files\sites')

    def run_batch(
        batch_sites: Sequence[str], batch_routers: Dict[str, Dict[str, Any]], label: str
    ) -> None:
        if not batch_sites or not batch_routers:
            return
        print_header(
            f'Running {label} ({len(batch_sites)} sites x {len(batch_routers)} routers)'
        )
        for si, site_name in enumerate(batch_sites, 1):
            L = getattr(locations, site_name)  # no validation
            for ri, (router_name, spec) in enumerate(batch_routers.items(), 1):
                key = f'{site_name}_{router_name}'
                cases.append({'key': key, 'site': site_name, 'router': router_name})
                router = make_router_from_spec(spec)
                cables = int(spec['cables'])
                print(
                    f'[{si}/{len(batch_sites)}] [{ri}/{len(batch_routers)}]: {key} (cables={cables})'
                )

                wfn = WindFarmNetwork(L=L, cables=cables)
                if router is None:
                    wfn.optimize()
                else:
                    wfn.optimize(router=router)
                router_graphs[key] = wfn.G.copy()

                del wfn, router
                gc.collect()

    run_plan = [
        ('sites_1 x routers_1', SITES_1, ROUTERS_1),
        ('sites_2 x routers_2', SITES_2, ROUTERS_2),
        ('sites_3 x routers_3', SITES_3, ROUTERS_3),
    ]
    for label, s, r in run_plan:
        run_batch(s, r, label)

    expected = {
        'Sites': tuple(sites_union),
        'Routers': routers_union,
        'Cases': cases,
        'RouterGraphs': router_graphs,
        'Meta': environment_meta(),
    }

    with output_path.open('wb') as f:
        dill.dump(expected, f, protocol=dill.HIGHEST_PROTOCOL)

    print_header('Completed')
    print(f'Saved expected values to: {output_path}')
    print(f'Cases stored: {len(cases)}; Graphs stored: {len(router_graphs)}')


# =============================================================================
# CLI
# =============================================================================


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out = Path(argv[0]).resolve() if argv else DEFAULT_OUTPUT
    generate_expected(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
