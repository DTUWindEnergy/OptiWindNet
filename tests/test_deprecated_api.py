# SPDX-License-Identifier: MIT
# https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/

"""Backward compatibility for public import paths from release 0.3.0."""

import importlib
import sys

import pytest

from optiwindnet import interarraylib

_DEPRECATED_INTERARRAYLIB_EXPORTS = {
    'G_from_S': 'optiwindnet.converting',
    'L_from_G': 'optiwindnet.converting',
    'L_from_site': 'optiwindnet.converting',
    'S_from_G': 'optiwindnet.converting',
    'S_from_terse_links': 'optiwindnet.converting',
    'terse_links_from_S': 'optiwindnet.converting',
    'bfs_subtree_loads': 'optiwindnet.loads',
    'calcload': 'optiwindnet.loads',
    'split_rings_and_calc_loads': 'optiwindnet.loads',
    'TerseLinks': 'optiwindnet.terse',
    'as_hooked_to_head': 'optiwindnet.transforming',
    'as_hooked_to_nearest': 'optiwindnet.transforming',
    'as_normalized': 'optiwindnet.transforming',
    'as_obstacle_free': 'optiwindnet.transforming',
    'as_rescaled': 'optiwindnet.transforming',
    'as_single_root': 'optiwindnet.transforming',
    'as_stratified_vertices': 'optiwindnet.transforming',
    'as_undetoured': 'optiwindnet.transforming',
    'validate_routeset': 'optiwindnet.validating',
    'validate_topology': 'optiwindnet.validating',
}


@pytest.mark.parametrize(
    ('name', 'module_name'), _DEPRECATED_INTERARRAYLIB_EXPORTS.items()
)
def test_interarraylib_deprecated_export_warns_and_aliases(name, module_name):
    interarraylib.__dict__.pop(name, None)

    with pytest.warns(
        DeprecationWarning,
        match=(
            rf'optiwindnet\.interarraylib\.{name} is deprecated and will be'
            r' removed in v0\.4\.0'
        ),
    ):
        old_export = getattr(interarraylib, name)

    assert old_export is getattr(importlib.import_module(module_name), name)


def test_interarraylib_deprecated_exports_remain_in_all():
    assert _DEPRECATED_INTERARRAYLIB_EXPORTS.keys() <= set(interarraylib.__all__)


@pytest.mark.parametrize('name', ('add_ring_to_S', 'rings_from_S'))
def test_interarraylib_does_not_reexport_private_ring_helpers(name):
    assert name not in interarraylib.__all__
    with pytest.raises(AttributeError):
        getattr(interarraylib, name)


def test_fingerprint_module_warns_and_aliases_identity_functions():
    sys.modules.pop('optiwindnet.fingerprint', None)

    with pytest.warns(
        DeprecationWarning,
        match=(
            r'optiwindnet\.fingerprint is deprecated and will be removed'
            r' in v0\.4\.0'
        ),
    ):
        fingerprint = importlib.import_module('optiwindnet.fingerprint')

    identity = importlib.import_module('optiwindnet.identity')
    assert fingerprint.fingerprint_coordinates is identity.fingerprint_coordinates
    assert fingerprint.fingerprint_function is identity.fingerprint_function
