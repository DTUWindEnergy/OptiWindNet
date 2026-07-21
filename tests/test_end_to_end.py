"""End-to-end router coverage.

Two complementary strategies, selected per case by the ``determinism`` tag in
``matrix.py`` (see that module for the rationale):

* ``test_snapshot_matches`` -- deterministic backends (EWRouter). The exact
  ``terse_links`` recorded in ``solutions.pkl`` are compared (with a canonical
  edge-set fallback for direction flips).

* ``test_milp_solution_properties`` / ``test_hgs_solution_properties`` --
  backends whose exact output is not reproducible across solver versions, seeds
  or hardware. These assert the solution's *properties* (validity, topology
  shape, capacity, feeder/balance constraints) and, for MILP, that the objective
  length is within the solver's gap tolerance of a stored reference optimum when
  one is available. No exact snapshot is stored for these.

ortools.math_opt must never share a process with standalone highspy/pyscipopt
(colliding native libraries), so only 'ortools*' cases are pushed into the shared
``ortools_worker`` subprocess (see tests/conftest.py); every other solver runs
directly in this process.
"""

import numpy as np
import pytest

from optiwindnet.api import WindFarmNetwork
from optiwindnet.api_utils import parse_cables_input
from optiwindnet.MILP import ModelOptions
from optiwindnet.types import Topology

from . import matrix
from .helpers import (
    assert_solution_properties,
    canonical_edges,
    load_instances,
    needs_process_isolation,
    router_factory,
    solution_property_metrics,
    solve_milp_property_metrics,
    solver_unavailable,
)
from .paths import SOLUTIONS_FILE


# --------------------------------------------------------------------------- #
# Golden data: deterministic snapshots + optional MILP length references
# --------------------------------------------------------------------------- #
def _load_expected():
    if not SOLUTIONS_FILE.exists():
        return None
    data = load_instances(SOLUTIONS_FILE)
    # new-schema file only; an old flat file is treated as missing (regenerate)
    if not isinstance(data, dict) or 'snapshots' not in data:
        return None
    return data


_EXPECTED = _load_expected()
_REF_LENGTHS = (_EXPECTED or {}).get('ref_lengths', {})


def _capacity(spec):
    return max(parse_cables_input(spec['cables']))[0]


def _cases(determinism):
    """Ordered ``[(key, (site, spec)), ...]`` for one determinism class."""
    return [
        (k, sitespec)
        for k, sitespec in matrix.matrix_by_key().items()
        if sitespec[1].get('determinism', 'snapshot') == determinism
    ]


def pytest_generate_tests(metafunc):
    if 'snapshot_case' in metafunc.fixturenames:
        if _EXPECTED is None:
            metafunc.parametrize(
                'snapshot_case',
                [
                    pytest.param(
                        None,
                        marks=pytest.mark.skip(
                            reason=(
                                f'Missing expected test data: {SOLUTIONS_FILE}\n'
                                'To (re)generate run:\n'
                                'python -m tests.update_expected_values\n'
                                'Or run pytest with --regen-expected.'
                            )
                        ),
                    )
                ],
            )
            return
        snaps = _EXPECTED['snapshots']
        keys = sorted(snaps)
        metafunc.parametrize('snapshot_case', [snaps[k] for k in keys], ids=keys)
    elif 'property_case' in metafunc.fixturenames:
        cases = _cases('property')
        metafunc.parametrize(
            'property_case',
            [sitespec for _, sitespec in cases],
            ids=[k for k, _ in cases],
        )


# --------------------------------------------------------------------------- #
# Deterministic snapshot comparison (EWRouter)
# --------------------------------------------------------------------------- #
def test_snapshot_matches(snapshot_case, locations):
    router_spec = snapshot_case['router_spec']
    topology_ref, terse_ref = snapshot_case['terse_links']
    L = getattr(locations, snapshot_case['location'])
    cables = router_spec['cables']

    router = router_factory(router_spec)
    wfn = WindFarmNetwork(L=L, cables=cables)
    wfn.optimize(router=router)
    terse_obt = tuple(wfn.terse_links().tolist())
    if terse_obt == terse_ref:
        return
    # terse_links can differ for topologically equivalent route sets (a chain's
    # direction flips). Compare the canonical edge sets (detour clones replaced
    # by their primes) instead.
    wfn_ref = WindFarmNetwork(L=L, cables=cables)
    wfn_ref.update_from_terse_links(
        np.asarray(terse_ref, dtype=np.int64), topology=topology_ref
    )
    assert canonical_edges(wfn.G) == canonical_edges(wfn_ref.G), (
        f'terse_links differ and canonical edges differ.\n'
        f'  obtained: {terse_obt}\n  expected: {terse_ref}'
    )


# --------------------------------------------------------------------------- #
# Property-based validation (MILP)
# --------------------------------------------------------------------------- #
def _assert_length_regression(spec, site, termination, length):
    ref = _REF_LENGTHS.get(matrix.problem_key(site, spec))
    if ref is None:
        return  # no recorded optimum for this problem -- properties only
    mip_gap = spec['params']['mip_gap']
    if termination == 'OPTIMAL':
        assert length == pytest.approx(ref, rel=max(mip_gap, 1e-3))
    else:
        # stopped at the gap: incumbent no worse than the tolerance above the
        # optimum, and never below it
        assert ref * (1 - 1e-3) <= length <= ref * (1 + mip_gap)


def _expected_topology(spec) -> Topology:
    cls = spec['class']
    if cls == 'MILPRouter':
        return Topology(
            spec['params'].get('model_options', {}).get('topology', 'branched')
        )
    if cls == 'HGSRouter':
        return Topology.RINGED if spec['params'].get('ringed') else Topology.RADIAL
    # EWRouter: only ringed reaches the property path (forest methods snapshot);
    # radial_EW is a path forest, the branched methods are trees.
    method = spec['params'].get('method', 'biased_EW')
    if method == 'ringed':
        return Topology.RINGED
    return Topology.RADIAL if method == 'radial_EW' else Topology.BRANCHED


def test_solution_properties(property_case, locations, ortools_worker):
    """Validate the *properties* of a solution rather than an exact snapshot.

    Covers every non-deterministic / ring-topology case: MILP (all backends),
    HGS, and ringed EWRouter. ortools MILP runs in the isolated worker; every
    other backend runs in-process.
    """
    site, spec = property_case
    L = getattr(locations, site)
    capacity = _capacity(spec)
    cls = spec['class']

    if cls == 'MILPRouter':
        if needs_process_isolation(spec):
            timeout = spec['params'].get('time_limit', 10) + 30
            result = ortools_worker.run(solve_milp_property_metrics, (spec, L), timeout)
        else:
            try:
                result = solve_milp_property_metrics(spec, L)
            except BaseException as exc:  # noqa: BLE001 -- classified below
                result = exc
        if isinstance(result, BaseException):
            if solver_unavailable(result):
                pytest.skip(f'{spec["params"]["solver_name"]} unavailable: {result}')
            raise result
        metrics, termination, length = result
        assert_solution_properties(metrics, spec, capacity)
        _assert_length_regression(spec, site, termination, length)
        return

    # in-process heuristics (ringed EWRouter, HGS)
    if cls == 'HGSRouter':
        pytest.importorskip('hybgensea')
    wfn = WindFarmNetwork(L=L, cables=spec['cables'])
    wfn.optimize(router=router_factory(spec))
    model_options = ModelOptions(topology=_expected_topology(spec))
    metrics = solution_property_metrics(wfn.S, wfn.G, model_options, capacity)
    assert_solution_properties(metrics, spec, capacity)


# --------------------------------------------------------------------------- #
# ortools warmstart behaviour (unchanged)
# --------------------------------------------------------------------------- #
def _run_ortools_warmstart():
    from optiwindnet.api import EWRouter, MILPRouter

    from .helpers import tiny_wfn

    router_ortools = MILPRouter(
        solver_name='ortools.cp_sat', time_limit=2, mip_gap=0.005, verbose=True
    )
    wfn = tiny_wfn()
    wfn.optimize(router=EWRouter())
    results = [list(wfn.optimize(router=router_ortools))]

    # invalid warmstart
    wfn.G.add_edge(-1, 11)
    router_ortools = MILPRouter(
        solver_name='ortools.cp_sat', time_limit=2, mip_gap=0.005, verbose=True
    )
    results.append(list(wfn.optimize(router=router_ortools)))

    # --- with detours
    wfn = tiny_wfn(cables=1)
    wfn.optimize(router=EWRouter())
    router_ortools = MILPRouter(
        solver_name='ortools.cp_sat', time_limit=2, mip_gap=0.005, verbose=True
    )
    results.append(list(wfn.optimize(router=router_ortools)))

    # invalid warmstart
    wfn.G.add_edge(0, 12)
    wfn.G.add_edge(12, 13)
    wfn.G.remove_edge(0, -1)
    router_ortools = MILPRouter(
        solver_name='ortools.cp_sat', time_limit=2, mip_gap=0.005, verbose=True
    )
    results.append(list(wfn.optimize(router=router_ortools)))
    return results


def test_ortools_with_warmstart(ortools_worker):
    result = ortools_worker.run(_run_ortools_warmstart, (), timeout=30)
    if isinstance(result, (FileNotFoundError, ModuleNotFoundError)):
        pytest.skip('ortools.cp_sat not available')
    if isinstance(result, BaseException):
        raise result

    assert result[0] == [-1, 0, 1, 2]
    assert result[1] == [-1, 0, 1, 2]
    assert result[2] == [-1, -1, -1, -1]
    assert result[3] == [-1, -1, -1, -1]
