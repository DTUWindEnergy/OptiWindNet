"""Meta-tests for the typed low-level producer matrix."""

from dataclasses import replace

from optiwindnet.MILP import ModelOptions
from optiwindnet.MILP._core import FeederRoute
from optiwindnet.types import Topology

from .cases import (
    CONSTRUCTOR_CASES,
    HGS_CASES,
    LKH_CASES,
    MILP_ADAPTER_CASES,
    MILP_BOUNDARY_CASES,
    MILP_FAMILY_CASES,
    MILP_FORMULATION_CASES,
    case_node_id,
    expected_topology,
    topology_golden_key,
)
from .sitecache import (
    SELECTED_HANDLES,
    get_bundle,
    get_location,
    location_repository,
)


def test_selected_site_monikers_are_canonical_handles():
    for handle in SELECTED_HANDLES:
        assert get_location(handle).graph['handle'] == handle


def test_single_root_site_handles_describe_the_transformation():
    for handle in SELECTED_HANDLES:
        L = get_location(handle)
        L_single = get_location(handle, single_root=True)
        suffix = '_1' if L.graph['R'] > 1 else ''

        assert L_single.graph['handle'] == f'{handle}{suffix}'
        assert L_single.graph['R'] == 1


def test_repository_and_bundle_share_location_objects():
    locations = location_repository()
    bundle = get_bundle('london')

    assert bundle is get_bundle('london')
    assert bundle.L is locations.london
    assert bundle.handle == bundle.A.graph['handle'] == 'london'


def test_bundle_keys_follow_actual_single_root_handle():
    cazzaro = get_bundle('cazzaro_2022')
    cazzaro_single = get_bundle('cazzaro_2022', single_root=True)
    yi = get_bundle('yi_2019')
    yi_single = get_bundle('yi_2019', single_root=True)

    assert cazzaro_single is cazzaro
    assert yi_single is not yi
    assert yi_single.handle == yi_single.A.graph['handle'] == 'yi_2019_1'


def test_bundle_copy_isolates_mutable_graph_metadata():
    original = get_bundle('example_location')
    copied = get_bundle('example_location', copy=True)
    original_coordinate = original.L.graph['VertexC'][0, 0]

    copied.L.graph['VertexC'][0, 0] += 1
    copied.A.graph['VertexC'][0, 0] += 1

    assert original.L.graph['VertexC'][0, 0] == original_coordinate
    assert original.A.graph['VertexC'][0, 0] == original_coordinate


def test_case_node_ids_are_unique_within_each_matrix():
    matrices = (
        CONSTRUCTOR_CASES,
        HGS_CASES,
        LKH_CASES,
        MILP_FORMULATION_CASES,
        MILP_ADAPTER_CASES,
        MILP_FAMILY_CASES,
        MILP_BOUNDARY_CASES,
    )
    for cases in matrices:
        node_ids = tuple(map(case_node_id, cases))
        assert len(node_ids) == len(set(node_ids))


def test_case_node_id_is_derived_from_execution_attributes():
    case = CONSTRUCTOR_CASES[1]
    node_id = case_node_id(case)

    assert case.site in node_id
    assert case.method.lower().replace('_', '-') in node_id
    assert f'cap{case.capacity}' in node_id
    assert case_node_id(replace(case, capacity=case.capacity + 1)) != node_id
    assert case_node_id(replace(case, exact_golden=False)) == node_id
    assert topology_golden_key(replace(case, exact_golden=False)) == (
        topology_golden_key(case)
    )


def test_milp_golden_key_ignores_backend_and_solve_controls():
    case = MILP_ADAPTER_CASES[0]
    equivalent = replace(
        case,
        solver_name='gurobi',
        time_limit=2.0,
        mip_gap=0.2,
    )

    assert case_node_id(equivalent) != case_node_id(case)
    assert topology_golden_key(equivalent) == topology_golden_key(case)
    assert len({topology_golden_key(item) for item in MILP_ADAPTER_CASES}) == 1


def test_milp_golden_key_tracks_model_options():
    case = MILP_ADAPTER_CASES[0]
    different_problem = replace(
        case,
        model_options=ModelOptions(feeder_limit='minimum'),
    )

    assert topology_golden_key(different_problem) != topology_golden_key(case)


def test_constructor_matrix_covers_required_axes():
    assert {case.method for case in CONSTRUCTOR_CASES} == {
        'esau_williams',
        'biased_EW',
        'rootlust',
        'radial_EW',
        'ringed',
    }
    expected = {
        'esau_williams': Topology.BRANCHED,
        'biased_EW': Topology.BRANCHED,
        'rootlust': Topology.BRANCHED,
        'radial_EW': Topology.RADIAL,
        'ringed': Topology.RINGED,
    }
    actual = {case.method: expected_topology(case) for case in CONSTRUCTOR_CASES}
    assert expected.items() <= actual.items()
    assert {case.feeder_route for case in CONSTRUCTOR_CASES} == set(FeederRoute)
    assert any(case.site == 'london' for case in CONSTRUCTOR_CASES)
    assert {case.site for case in CONSTRUCTOR_CASES} >= {
        'cazzaro_2022',
        'morayeast',
    }


def test_baseline_matrices_cover_topologies_and_roots():
    assert {case.ringed for case in HGS_CASES} == {False, True}
    assert {case.ringed for case in LKH_CASES} == {False, True}
    assert {expected_topology(case) for case in HGS_CASES} == {
        Topology.RADIAL,
        Topology.RINGED,
    }
    assert {expected_topology(case) for case in LKH_CASES} == {
        Topology.RADIAL,
        Topology.RINGED,
    }
    assert any(case.site == 'morayeast' for case in HGS_CASES)


def test_milp_matrix_separates_formulations_from_adapters():
    assert {case.model_options['topology'] for case in MILP_FORMULATION_CASES} == set(
        Topology
    )
    assert {
        case.model_options['feeder_route'] for case in MILP_FORMULATION_CASES
    } == set(FeederRoute)
    assert len({case.solver_name for case in MILP_ADAPTER_CASES}) >= 3
    assert {case.solver_name for case in MILP_FAMILY_CASES} == {'highs', 'scip'}
