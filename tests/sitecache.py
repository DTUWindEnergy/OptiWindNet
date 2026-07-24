"""Process-local cache of reusable test locations and navigation meshes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import cache
import pickle
from pathlib import Path

import networkx as nx

from optiwindnet.fingerprint import fingerprint_coordinates
from optiwindnet.importer import (
    L_from_pbf,
    L_from_yaml,
    LocationsRepository,
    load_repository,
)
from optiwindnet.interarraylib import as_single_root
from optiwindnet.mesh import make_planar_embedding
from optiwindnet.synthetic import toyfarm

from . import paths
from .nodeset_digest_aliases import (
    NODESET_DIGEST_ALIASES,
    nodeset_digest_alias_coordinates,
)


@dataclass(frozen=True, slots=True)
class SiteBundle:
    """Cached location, planar embedding, and available-links graph."""

    handle: str
    coordinate_digest: bytes
    L: nx.Graph
    P: nx.PlanarEmbedding
    A: nx.Graph


def _derive_selected_handles() -> tuple[str, ...]:
    from .cases import ALL_TEST_MATRICES

    handles = ['toy', 'example_location']
    for matrix in ALL_TEST_MATRICES:
        for case in matrix:
            if case.site not in handles:
                handles.append(case.site)
    return tuple(handles)


SELECTED_HANDLES = _derive_selected_handles()

NODESET_DIGEST_LOCATION_MAP_FILE = Path(__file__).with_name(
    'nodeset_digest-location-map.pkl'
)


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


def bundled_location_path(name: str) -> Path:
    """Resolve a bundled location name to its uniquely named data file."""
    candidates = (
        paths.DATA_DIR / f'{name}.yaml',
        paths.DATA_DIR / f'{name}.osm.pbf',
    )
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ValueError(
            f'location {name!r} must match exactly one bundled data file; '
            f'found {matches}'
        )
    return matches[0]


@cache
def _base_location_by_name(name: str) -> nx.Graph:
    path = bundled_location_path(name)
    L = L_from_pbf(path) if path.suffix == '.pbf' else L_from_yaml(path)
    if L.graph.get('name') != name:
        raise ValueError(f'location file {path} contains name {L.graph.get("name")!r}')
    return L


@cache
def nodeset_digest_location_map() -> dict[bytes, str]:
    """Load the mapping from coordinate digests to bundled location names."""
    if not NODESET_DIGEST_LOCATION_MAP_FILE.exists():
        raise FileNotFoundError(
            f'Missing nodeset digest map: {NODESET_DIGEST_LOCATION_MAP_FILE}\n'
            'Regenerate it with: python -m tests.update_nodeset_digest_location_map'
        )
    with NODESET_DIGEST_LOCATION_MAP_FILE.open('rb') as file:
        mapping = pickle.load(file)
    if not isinstance(mapping, dict) or not all(
        isinstance(digest, bytes)
        and len(digest) == 32
        and isinstance(name, str)
        and name
        for digest, name in mapping.items()
    ):
        raise TypeError(
            'nodeset digest map must be a dict from 32-byte digests to names'
        )
    for name in set(mapping.values()):
        bundled_location_path(name)
    return mapping


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


@cache
def get_bundle_from_nodeset_digest(digest: bytes) -> SiteBundle:
    """Return the location variant identified by a coordinate digest."""
    mapping = nodeset_digest_location_map()
    try:
        name = mapping[digest]
    except KeyError as exc:
        raise KeyError(f'no location for nodeset digest {digest.hex()}') from exc

    base = _base_location_by_name(name)
    variants = (False, True) if base.graph['R'] > 1 else (False,)
    for single_root in variants:
        L = as_single_root(base) if single_root else base
        actual = fingerprint_coordinates(L.graph['VertexC'])[0]
        if actual == digest:
            P, A = make_planar_embedding(L)
            return SiteBundle(L.graph['handle'], digest, L, P, A)

    try:
        canonical_digest = NODESET_DIGEST_ALIASES[digest]
    except KeyError as exc:
        raise AssertionError(
            f'location {name!r} does not match mapped digest {digest.hex()}'
        ) from exc
    if mapping.get(canonical_digest) != name:
        raise AssertionError(
            f'alias target does not map to location {name!r}: {canonical_digest.hex()}'
        )
    for single_root in variants:
        L = as_single_root(base) if single_root else base
        actual = fingerprint_coordinates(L.graph['VertexC'])[0]
        if actual == canonical_digest:
            historical_vertexc = nodeset_digest_alias_coordinates()[digest]
            if historical_vertexc.shape != L.graph['VertexC'].shape:
                raise AssertionError(
                    f'location {name!r} shape does not match alias coordinates'
                )
            L = L.copy()
            L.graph['VertexC'] = historical_vertexc
            P, A = make_planar_embedding(L)
            return SiteBundle(L.graph['handle'], digest, L, P, A)
    raise AssertionError(
        f'location {name!r} does not match alias target {canonical_digest.hex()}'
    )
