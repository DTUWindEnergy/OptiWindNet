"""Process-local cache of reusable test locations and navigation meshes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import cache

import networkx as nx

from optiwindnet.fingerprint import fingerprint_coordinates
from optiwindnet.importer import L_from_yaml, LocationsRepository, load_repository
from optiwindnet.interarraylib import as_single_root
from optiwindnet.mesh import make_planar_embedding
from optiwindnet.synthetic import toyfarm

from . import paths


@dataclass(frozen=True, slots=True)
class SiteBundle:
    """Cached location, planar embedding, and available-links graph."""

    handle: str
    coordinate_digest: bytes
    L: nx.Graph
    P: nx.PlanarEmbedding
    A: nx.Graph


SELECTED_HANDLES = (
    'toy',
    'example_location',
    'cazzaro_2022',
    'morayeast',
    'albatros',
    'neart',
    'london',
    'taylor_2023',
    'yi_2019',
    'borkum2',
)

# The PathFinder artifact stores coordinate fingerprints rather than handles.
# These are deliberately explicit: adding a golden site must update this table.
PATHFINDER_HANDLES_BY_DIGEST = {
    bytes.fromhex(
        '03b54af1ae046ed6bd37f02b12cfe03e56c90c29817d755af72340b058cf403f'
    ): 'cazzaro_2022',
    bytes.fromhex(
        '6fcdffd7e2da097f6d8f61a247ca11c4de765eea085cf3f062107fe17f14172a'
    ): 'yi_2019',
    bytes.fromhex(
        '1b74440a745d55b905f2fdd300bcf1e5f3c145414878f0812b0916f3fef5bd38'
    ): 'taylor_2023',
}


@cache
def location_repository() -> LocationsRepository:
    """Load and validate the bundled location repository exactly once."""
    locations = load_repository()
    fields = locations._fields  # pyrefly: ignore[missing-attribute]
    for field, L in zip(fields, locations, strict=True):
        handle = L.graph.get('handle')
        if handle is None:
            raise ValueError(f'repository location {field!r} has no handle')
        if handle != field:
            raise ValueError(f'repository field {field!r} contains location {handle!r}')
    return locations


@cache
def _repository_by_handle() -> dict[str, nx.Graph]:
    locations = location_repository()
    fields = locations._fields  # pyrefly: ignore[missing-attribute]
    return dict(zip(fields, locations, strict=True))


@cache
def _base_location(handle: str) -> nx.Graph:
    if handle == 'toy':
        L = toyfarm()
    elif handle == 'example_location':
        L = L_from_yaml(paths.LOCATIONS_DIR / 'example_location.yaml')
    else:
        try:
            L = _repository_by_handle()[handle]
        except KeyError as exc:
            raise KeyError(f'unknown cached location handle: {handle!r}') from exc
    actual = L.graph.get('handle')
    if actual is None:
        raise ValueError(f'cached location {handle!r} has no handle')
    if actual != handle:
        raise ValueError(f'cached location {handle!r} has handle {actual!r}')
    return L


@cache
def get_location(handle: str, *, single_root: bool = False) -> nx.Graph:
    """Return the shared location graph for one canonical handle variant."""
    L = _base_location(handle)
    expected_handle = handle
    if single_root and L.graph['R'] > 1:
        expected_handle += '_1'
        L = as_single_root(L)
    if L.graph.get('handle') != expected_handle:
        raise ValueError(
            f'location derived from {handle!r} has handle '
            f'{L.graph.get("handle")!r}; expected {expected_handle!r}'
        )
    return L


@cache
def _bundle_cached(
    coordinate_digest: bytes, derived_handle: str, base_handle: str
) -> SiteBundle:
    # ``coordinate_digest`` makes the cache contract explicit and prevents a
    # handle-stable but coordinate-changed fixture from reusing a stale mesh.
    L = get_location(base_handle, single_root=derived_handle != base_handle)
    actual = fingerprint_coordinates(L.graph['VertexC'])[0]
    if actual != coordinate_digest:
        raise AssertionError(f'coordinate fingerprint changed for {derived_handle}')
    P, A = make_planar_embedding(L)
    if A.graph.get('handle') != derived_handle:
        raise ValueError(
            f'available-links graph has handle {A.graph.get("handle")!r}; '
            f'expected {derived_handle!r}'
        )
    return SiteBundle(derived_handle, coordinate_digest, L, P, A)


def get_bundle(
    handle: str, *, single_root: bool = False, copy: bool = False
) -> SiteBundle:
    """Return the shared ``L/P/A`` bundle, or a mutation-safe deep copy."""
    L = get_location(handle, single_root=single_root)
    digest = fingerprint_coordinates(L.graph['VertexC'])[0]
    bundle = _bundle_cached(digest, L.graph['handle'], handle)
    return deepcopy(bundle) if copy else bundle


def pathfinder_site(digest: bytes) -> SiteBundle:
    """Return the explicitly mapped single-root bundle for a PathFinder golden."""
    try:
        handle = PATHFINDER_HANDLES_BY_DIGEST[digest]
    except KeyError as exc:
        raise KeyError(
            f'no selected PathFinder site for digest {digest.hex()}'
        ) from exc
    bundle = get_bundle(handle, single_root=True)
    actual = fingerprint_coordinates(bundle.L.graph['VertexC'])[0]
    if actual != digest:
        raise AssertionError(
            f'PathFinder site {bundle.handle} has digest {actual.hex()}, '
            f'expected {digest.hex()}'
        )
    return bundle
