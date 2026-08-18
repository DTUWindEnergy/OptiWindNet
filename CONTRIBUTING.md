# Contributing to OptiWindNet

OptiWindNet's primary development repository and CI are hosted on a private GitLab instance belonging to the [Department of Wind and Energy Systems](https://wind.dtu.dk/) at the [Technical University of Denmark (DTU)](https://www.dtu.dk/english). Contributions are also welcome through the public GitHub mirror; see [Repository access and contribution paths](#repository-access-and-contribution-paths) for the two paths.

## Development setup

OptiWindNet requires Python 3.11 or newer. In an activated development environment, install the package and the dependencies used by the checks:

```sh
git clone https://github.com/DTUWindEnergy/OptiWindNet.git
# With GitLab access, use instead:
# git clone https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet.git
cd OptiWindNet
pip install -e '.[test,docs]'
prek install
```

The `test` extra includes pytest, the pip-installable MILP backends, prek, Pyrefly, and the type stubs. The `docs` extra is included because Pyrefly checks the Python files under `docs/` as well as the library and tests. Ruff, the project's linter and formatter, is supplied through prek from the version pinned in `.pre-commit-config.yaml`.

`prek install` adds the commit hook. The same configuration can also be run with `pre-commit` if that is already part of your workflow.

## Repository access and contribution paths

For most external contributors, the practical route is to open a pull request against the [public GitHub mirror](https://github.com/DTUWindEnergy/OptiWindNet). The maintainers will coordinate moving the contribution to GitLab for CI and integration. This route is new and may require some additional coordination, but GitLab access is not required to propose a change.

Contributors who already have access to the [DTU Wind Energy GitLab project](https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet) can instead open a merge request against `main` there.

[GitLab access can be requested](https://gitlab.windenergy.dtu.dk/users/sign_up), with different procedures for DTU and external contributors:

- DTU employees, students, and other holders of a `dtu.dk` account should not register a new account; sign in through the DTU login using the existing DTU credentials.
- External users must first register with a valid academic or recognized-company email address. After a site administrator approves the account, contact the project's main developer to request access to the private repository.

Questions and bug reports can be opened on the [GitHub issue tracker](https://github.com/DTUWindEnergy/OptiWindNet/issues).

## Checks and tests

The commit hook runs Ruff on the relevant Python, stub, and notebook files, then type-checks the project with Pyrefly. Ruff may fix or reformat files in place; if that aborts a commit, review the changes, stage them, and commit again.

Targeted tests are usually the quickest development loop:

```sh
pytest tests/test_mesh.py -k concavity
```

For the most complete local validation, run the checks across the repository and the full test suite:

```sh
prek run --all-files
pytest
```

These commands are useful when preparing a particularly complete contribution, but they are not a prerequisite for opening a draft or focused pull request. State which checks you ran and which validation remains so that reviewers and CI can pick up from there.

Some project-specific details:

- Ruff uses single-quoted strings and an 88-column line length. Its hooks cover `.py`, `.pyi`, and `.ipynb` files.
- Pyrefly targets Python 3.11 and skips `docs/notebooks/`. If changing Pyrefly or the pinned `types-*` packages, run the all-files checks and update any suppressions that the new versions make obsolete.
- Prefer a narrow, explained suppression when a checker is wrong instead of loosening the project-wide configuration.
- Solver-dependent tests skip when a backend, executable, or licence is unavailable. The `solvers` extra covers the pip-installable backends; CBC and FiberSCIP also require external executables.
- `tests/test_milp_references.py` is sensitive to heavy parallel load. If one of its short solver runs warns or fails, rerun that file serially before treating the result as a regression.

## Documentation and notebooks

If a change affects the documentation, the thorough local checks are:

```sh
make -C docs html
make -C docs check
```

On Linux and macOS, the Sphinx build needs a separately installed `pandoc` executable; on Windows, the `docs` extra supplies one. `make -C docs check` covers documentation rules that Sphinx cannot check and, when Prettier is available, verifies the Markdown format. See the [documentation maintainer notes](docs/README.md) for the source formats, cross-linking rules, figures, and notebook conventions.

Sphinx does not execute notebooks: their outputs are committed. Refresh changed notebooks with a suitable installed kernel:

```sh
python docs/run_notebooks.py --kernel <kernel-name> --changed
```

MILP notebooks are skipped by default; add `--milp` when the relevant solvers are available.

## Making a reviewable contribution

Focused changes are generally easier to review, though related cleanup is welcome when its connection to the contribution is explained.

Where practical, add focused tests for changed behavior or bug fixes. Update the documentation or examples when users would interact with the change differently.

A few areas deserve additional context in the pull request:

- Graph transformations should preserve graph, node, and edge attributes unless changing them is intentional. Several later routing, storage, and plotting stages consume those attributes, so a regression test is especially helpful here.
- If a change affects the public API, graph conventions, or stored-data compatibility, describe the compatibility impact and any migration path.
- `optiwindnet/data/` and `tests/locations/` contain curated reference data. For changes there, explain the source of the new data and how it was produced.
