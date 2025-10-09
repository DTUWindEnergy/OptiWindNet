# tests/conftest.py
"""
Central pytest fixtures for optiwindnet tests.

Responsibilities:
 - Ensure deterministic test environment (disable numba JIT).
 - Resolve repository/test-files paths.
 - Load expected dill blobs with helpful messages.
 - Provide factory fixtures (router construction, L/G loader, site extractor).
 - Optionally regenerate expected data when `--regen-expected` is passed.
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import dill
import numpy as np
import pytest

# import shared path constants
from . import paths
REPO_ROOT = paths.REPO_ROOT
UNITTESTS_DILL = paths.UNITTESTS_DILL
END_TO_END_DILL = paths.END_TO_END_DILL
TEST_FILES_DIR = paths.TEST_FILES_DIR
SITES_DIR = paths.SITES_DIR
GEN_UNITS_SCRIPT = paths.GEN_UNITS_SCRIPT
GEN_END2END_SCRIPT = paths.GEN_END2END_SCRIPT

# Ensure Numba JIT is disabled for tests (both env var and runtime)
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
try:
    import numba

    numba.config.DISABLE_JIT = True
except Exception:
    # If numba is not available, that's fine; tests that require it will import as needed.
    pass

# -----------------------
# Utility helpers
# -----------------------
def _load_dill(path: Path) -> Any:
    """Load a dill file; raise FileNotFoundError with regeneration hint if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing expected test data file: {path}\n\n"
            "To (re)generate this file run the appropriate generator script, e.g.:\n"
            f"  python {GEN_UNITS_SCRIPT}    # smaller unit test blob\n"
            f"  python {GEN_END2END_SCRIPT}  # end-to-end blob (may be slow)\n\n"
            "Or run pytest with --regen-expected to attempt regeneration automatically "
            "(only if you really want that behavior)."
        )
    with path.open("rb") as fh:
        return dill.load(fh)


def _maybe_run_generator(script_path: Path) -> None:
    """Run a generator script via subprocess (fresh Python interpreter)."""
    if not script_path.exists():
        raise FileNotFoundError(f"Generator script not found: {script_path}")
    # Use the same python interpreter
    proc = subprocess.run([sys.executable, str(script_path)], check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Generator script failed: {script_path} (rc={proc.returncode})")


# -----------------------
# Pytest CLI option (optional regeneration)
# -----------------------
def pytest_addoption(parser):
    group = parser.getgroup("optiwindnet", "optiwindnet test helpers")
    group.addoption(
        "--regen-expected",
        action="store_true",
        default=False,
        help=(
            "If set, pytest will attempt to regenerate missing expected dill files "
            "by running the repository generator scripts. Use with care (generators "
            "may be slow or require external solvers)."
        ),
    )


def pytest_sessionstart(session):
    """If user passed --regen-expected and files are missing, try regenerate them."""
    regen = session.config.getoption("--regen-expected")
    if not regen:
        return

    # Attempt to regenerate missing expected files (best-effort; fail loudly if generator fails)
    if not UNITTESTS_DILL.exists() and GEN_UNITS_SCRIPT.exists():
        session.config.warn("optiwindnet", f"Regenerating {UNITTESTS_DILL} via {GEN_UNITS_SCRIPT}")
        _maybe_run_generator(GEN_UNITS_SCRIPT)
    if not END_TO_END_DILL.exists() and GEN_END2END_SCRIPT.exists():
        session.config.warn("optiwindnet", f"Regenerating {END_TO_END_DILL} via {GEN_END2END_SCRIPT}")
        _maybe_run_generator(GEN_END2END_SCRIPT)


# -----------------------
# Simple path fixtures
# -----------------------
@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def test_files_dir(repo_root: Path) -> Path:
    return TEST_FILES_DIR


@pytest.fixture(scope="session")
def sites_dir(test_files_dir: Path) -> Path:
    return SITES_DIR


# -----------------------
# Expected data loaders (session-scoped)
# -----------------------
@pytest.fixture(scope="session")
def expected_unittests() -> Dict[str, Any]:
    """Load the small per-unit test expected blob."""
    return _load_dill(UNITTESTS_DILL)


@pytest.fixture(scope="session")
def expected_end_to_end() -> Dict[str, Any]:
    """Load the larger end-to-end expected blob."""
    return _load_dill(END_TO_END_DILL)


# Backwards compatibility: tests using "expected" expect the unittests blob
@pytest.fixture(scope="session")
def expected(expected_unittests: Dict[str, Any]) -> Dict[str, Any]:
    return expected_unittests


# Convenience 'db' fixture exposing RouterGraphs from the unittests blob
@pytest.fixture(scope="module")
def db(expected_unittests: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    if "RouterGraphs" not in expected_unittests:
        pytest.skip("expected_unittests.dill does not contain 'RouterGraphs'")
    yield expected_unittests["RouterGraphs"]


# -----------------------
# Lazy import helpers for router construction
# -----------------------
def _import_router_symbols():
    try:
        from optiwindnet.api import EWRouter, HGSRouter, MILPRouter  # type: ignore
        from optiwindnet.MILP import ModelOptions  # type: ignore
    except Exception as exc:
        raise RuntimeError("Failed to import router classes from optiwindnet") from exc
    return EWRouter, HGSRouter, MILPRouter, ModelOptions


def make_router_from_spec(spec: Optional[Dict[str, Any]]):
    """Create an instantiated router from a spec dict (same semantics as your generators)."""
    if spec is None:
        return None
    EWRouter, HGSRouter, MILPRouter, ModelOptions = _import_router_symbols()
    clsname = spec.get("class")
    params = dict(spec.get("params", {}))
    # Expand ModelOptions dict when present
    if clsname == "MILPRouter" and isinstance(params.get("model_options"), dict):
        params["model_options"] = ModelOptions(**params["model_options"])
    if clsname is None:
        return None
    if clsname == "EWRouter":
        return EWRouter(**params)
    if clsname == "HGSRouter":
        return HGSRouter(**params)
    if clsname == "MILPRouter":
        return MILPRouter(**params)
    raise ValueError(f"Unknown router class: {clsname!r}")


@pytest.fixture
def router_factory() -> Callable[[Optional[Dict[str, Any]]], Any]:
    """Factory fixture that builds router instances from router specs."""
    return make_router_from_spec


# -----------------------
# L/G / site factories (from stored graphs)
# -----------------------
@pytest.fixture
def LG_from_database(db: Dict[str, Any]):
    """Return a factory that reconstructs (L, G) from stored RouterGraphs labels."""
    from optiwindnet.interarraylib import L_from_G  # local import

    def _factory(label: str):
        if label not in db:
            raise KeyError(f"Label {label!r} not found in test database")
        G = db[label]
        L = L_from_G(G)
        return L, G

    return _factory


@pytest.fixture
def site_from_database(db: Dict[str, Any]):
    """Factory that extracts coordinate-based site components from a stored graph."""
    def _factory(label: str):
        if label not in db:
            raise KeyError(f"Label {label!r} not found in test database")
        G = db[label]
        VertexC = G.graph["VertexC"]
        T = G.graph["T"]
        R = G.graph["R"]

        return {
            "turbinesC": VertexC[:T],
            "substationsC": VertexC[-R:] if R > 0 else np.empty((0, 2)),
            "borderC": VertexC[G.graph.get("border", [])] if "border" in G.graph else np.empty((0, 2)),
            "obstaclesC": [VertexC[o] for o in G.graph.get("obstacles", [])],
            "handle": G.graph.get("handle"),
            "name": G.graph.get("name"),
            "landscape_angle": G.graph.get("landscape_angle"),
        }

    return _factory


# -----------------------
# Lazy-loaded repository locations fixture
# -----------------------
@pytest.fixture(scope="session")
def locations(sites_dir: Path):
    """Load repository-backed sites (the same loader used by generator scripts)."""
    try:
        from optiwindnet.importer import load_repository  # type: ignore
    except Exception as exc:
        raise RuntimeError("Failed to import load_repository from optiwindnet.importer") from exc
    return load_repository(path=str(sites_dir))
