# Contributing to OptiWindNet

## Prerequisites

- Python 3.11 or later
- Git

## Development setup

Clone the repository and install the package in editable mode with the test dependencies:

```bash
git clone https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet.git
cd OptiWindNet
pip install -e ".[test]"
```

> If you need to test against commercial solvers (Gurobi, CPLEX, SCIP, HiGHS), install the solver extras instead:
> ```bash
> pip install -e ".[test-solvers]"
> ```

## Running the tests

```bash
pytest
```

Tests are scoped to the `tests/` directory. Parallel execution is enabled by default in CI via `pytest-xdist`; add `-n auto` locally if you want the same:

```bash
pytest -n auto
```

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting. The easiest way to enforce it automatically is via [pre-commit](https://pre-commit.com/):

```bash
pip install pre-commit
pre-commit install
```

After that, ruff runs automatically on every commit. To run it manually:

```bash
pre-commit run --all-files
```

## Submitting changes

1. Create a branch from `main` with a descriptive name (e.g. `fix/some-issue` or `feat/new-feature`).
2. Make your changes and ensure the test suite passes.
3. Open a merge request on GitLab against `main`.

For questions or discussion, open an issue on the [issue tracker](https://github.com/DTUWindEnergy/OptiWindNet/issues).
