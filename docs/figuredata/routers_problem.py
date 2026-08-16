"""Define the problem used by the Routers figure.

This module defines the problem shared by all Routers scripts and data files. To
change it, update the configuration below and regenerate the data.

Generated results are stored in two small files that identify their problem.
"""

import json
import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from optiwindnet.fingerprint import fingerprint_coordinates

HERE = Path(__file__).resolve().parent

# --- the problem ------------------------------------------------------------

#: Name of the location attribute as provided by ``load_repository()``.
LOCATION = 'doggerB'

#: Maximum number of terminals per feeder.
CAPACITY = 8

#: MILP model options used for the optimum proof and MILP solver run.
MODEL_OPTIONS: dict[str, Any] = {
    'topology': 'branched',
    'feeder_route': 'segmented',
    'feeder_limit': 'min_plus1',
    'balanced': False,
}

#: Meta-heuristic time limits sampled for the figure, in seconds.
HGS_TIME_LIMITS = (0.1, 0.5, 2.5, 12.5)

#: Constructive heuristic variants sampled for the figure.
EW_METHODS = ('biased_EW', 'rootlust')

#: RNG seed used for each meta-heuristic run.
HGS_SEED = 0

#: The deliberately early HGS result used to start the MILP time series.
MILP_TIMESERIES_WARMSTART = 'hgs_cvrp:0.1'

#: Proven optimality gap at which the MILP solver run stops.
TARGET_GAP = 0.001

# --- files ------------------------------------------------------------------

SUMMARY = HERE / 'routers_summary.pkl'
TIMESERIES = HERE / 'routers_timeseries.npz'

# --- helpers ----------------------------------------------------------------


def load_location():
    """Return the configured location graph from the packaged data."""
    from optiwindnet.importer import load_repository

    return getattr(load_repository(), LOCATION)


def fingerprint(L=None) -> dict[str, Any]:
    """Return the fingerprint that identifies this problem."""
    L = load_location() if L is None else L
    return {
        'location': LOCATION,
        'nodeset_digest': nodeset_digest(L),
        'capacity': CAPACITY,
        'model_options': MODEL_OPTIONS,
        'ew_methods': EW_METHODS,
        'hgs_time_limits': HGS_TIME_LIMITS,
        'hgs_seed': HGS_SEED,
        'milp_timeseries_warmstart': MILP_TIMESERIES_WARMSTART,
        'target_gap': TARGET_GAP,
    }


def nodeset_digest(L=None) -> bytes:
    """Return the exact configured location geometry's digest."""
    L = load_location() if L is None else L
    return fingerprint_coordinates(L.graph['VertexC'])[0]


def problem_id() -> str:
    """Return a JSON identity suitable for the NumPy archive."""
    return json.dumps(
        fingerprint(),
        default=lambda value: value.hex() if isinstance(value, bytes) else value,
        sort_keys=True,
        separators=(',', ':'),
    )


def describe(L) -> str:
    g = L.graph
    return (
        f'{L.name} [{LOCATION}]: T={g["T"]} R={g["R"]} B={g["B"]} '
        f'capacity={CAPACITY}, model={MODEL_OPTIONS}'
    )


def setup():
    """Load and mesh the configured problem, then return `L`, `P`, and `A`."""
    from optiwindnet.mesh import make_planar_embedding

    L = load_location()
    print(describe(L))
    P, A = make_planar_embedding(L)
    print(f'mesh: |A| = {A.number_of_edges()} links')
    return L, P, A


def load_summary() -> dict[str, Any]:
    """Load the routers summary and reject a different problem."""
    with SUMMARY.open('rb') as file:
        data = pickle.load(file)
    if data.get('problem') != problem_id():
        raise ValueError(f'{SUMMARY.name} does not match the configured problem')
    return data


def save_summary(data: dict[str, Any]) -> None:
    """Replace the routers summary and invalidate the dependent time series."""
    temporary = SUMMARY.with_suffix('.pkl.tmp')
    with temporary.open('wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(SUMMARY)
    TIMESERIES.unlink(missing_ok=True)


def save_timeseries(
    bound: Iterable[tuple[float, float]], incumbent: Iterable[tuple[float, float]]
) -> None:
    """Store the MILP bound and incumbent series in a NumPy archive.

    The two series are independent: the solver reports each quantity on its own
    occasions, so each is stored as ``<name>_time`` and ``<name>`` arrays holding
    the times at which that quantity was reported and the metres reported.

    Args:
      bound: (time, length) pairs of the dual bound.
      incumbent: (time, length) pairs of the incumbent, one per improvement.
    """
    arrays = {}
    for name, pairs in (('bound', bound), ('incumbent', incumbent)):
        points = np.array(list(pairs), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f'MILP {name} series entries must be (time, length)')
        arrays[name + '_time'] = points[:, 0]
        arrays[name] = points[:, 1]
    temporary = TIMESERIES.with_suffix('.npz.tmp')
    with temporary.open('wb') as file:
        np.savez_compressed(file, problem=np.array(problem_id()), **arrays)
    temporary.replace(TIMESERIES)


def load_timeseries() -> dict[str, Any]:
    """Load the MILP bound and incumbent series and check their problem."""
    with np.load(TIMESERIES, allow_pickle=False) as stored:
        if stored['problem'].item() != problem_id():
            raise ValueError(f'{TIMESERIES.name} does not match the configured problem')
        series = {
            name + suffix: stored[name + suffix].copy()
            for name in ('bound', 'incumbent')
            for suffix in ('_time', '')
        }
    for name in ('bound', 'incumbent'):
        if series[name + '_time'].shape != series[name].shape:
            raise ValueError(f'MILP {name} series arrays have different shapes')
    return series
