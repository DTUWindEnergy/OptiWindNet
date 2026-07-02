# test end to end
import numpy as np
import pytest

from optiwindnet.api import WindFarmNetwork

from .helpers import (
    canonical_edges,
    load_instances,
    needs_process_isolation,
    router_factory,
    solve_milp_low_level,
)
from .paths import SOLUTIONS_FILE

# ortools.math_opt must never share a process with standalone highspy/pyscipopt
# (they bundle colliding copies of the same native libraries under the same
# soname). Only 'ortools*' cases are pushed into the shared `ortools_worker`
# subprocess (see tests/conftest.py); every other solver is safe to run
# directly in this process.


def pytest_generate_tests(metafunc):
    if 'routed_instance' in metafunc.fixturenames:
        if not SOLUTIONS_FILE.exists():
            metafunc.parametrize(
                'routed_instance',
                [
                    pytest.param(
                        None,
                        marks=pytest.mark.skip(
                            reason=(
                                f'Missing expected test data: {SOLUTIONS_FILE}\n'
                                'To (re)generate run:'
                                ' python update_expected_values.py\n'
                                'Or run pytest with --regen-expected.'
                            ),
                        ),
                    )
                ],
            )
            return
        all_instances = load_instances(SOLUTIONS_FILE)
        routed_instances = []
        ids = []
        for key in sorted(all_instances):
            routed_instances.append(all_instances[key])
            ids.append(key)
        metafunc.parametrize('routed_instance', routed_instances, ids=ids)


def test_expected_router_graphs_match(routed_instance, locations, ortools_worker):
    router_spec = routed_instance['router_spec']
    terse_ref = routed_instance['terse_links']
    L = getattr(locations, routed_instance['location'])
    cables = router_spec['cables']

    if router_spec['class'] == 'MILPRouter':
        if needs_process_isolation(router_spec):
            timeout = router_spec['params'].get('time_limit', 10) + 10
            result = ortools_worker.run(solve_milp_low_level, (router_spec, L), timeout)
        else:
            try:
                result = solve_milp_low_level(router_spec, L)
            except BaseException as exc:
                result = exc
        if isinstance(result, (FileNotFoundError, ModuleNotFoundError)):
            pytest.skip(f'{router_spec["params"]["solver_name"]} not available')
        if isinstance(result, BaseException):
            raise result
        terse_obt, edges_obt = result
    else:
        try:
            router = router_factory(router_spec)
        except (FileNotFoundError, ModuleNotFoundError):
            pytest.skip(f'{router_spec["params"]["solver_name"]} not available')
        wfn = WindFarmNetwork(L=L, cables=cables)
        wfn.optimize(router=router)
        terse_obt = tuple(wfn.terse_links().tolist())
        edges_obt = canonical_edges(wfn.G)

    if terse_obt == terse_ref:
        return
    # Fallback: terse_links can differ for topologically equivalent route sets
    # (e.g. a chain's direction flips). Compare the canonical edge sets
    # (detour clones replaced by their primes) of G instead.
    wfn_ref = WindFarmNetwork(L=L, cables=cables)
    wfn_ref.update_from_terse_links(np.asarray(terse_ref, dtype=np.int64))
    assert edges_obt == canonical_edges(wfn_ref.G), (
        f'terse_links differ and canonical edges differ.\n'
        f'  obtained: {terse_obt}\n  expected: {terse_ref}'
    )


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
