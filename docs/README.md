# Maintaining the OptiWindNet documentation

Notes for editing the docs. **This file is not part of the built documentation** — it is excluded in `conf.py` so it renders on GitLab/GitHub without becoming a page.

## Layout

```
docs/
├── conf.py               Sphinx configuration
├── Makefile              `make html`, `make check`
├── check_docs.py         checks the Sphinx build cannot do (see below)
├── run_notebooks.py      re-executes notebooks and normalises their JSON
├── figures.py            builds the SVG figures the prose pages embed
├── index.md              landing page; holds every toctree
│
├── install.md            ┐
├── apis.md               ┘ Start
├── problem.md            ┐ Concepts
├── routers.md            ┘
├── reference/            Task Index · Glossary · MILP Solvers and Formulation · Validation
├── high_level_api.md     Network/Router API — toctrees over the hi* notebooks
├── low_level_api.md      Advanced API — toctrees over the lo* notebooks
├── paper.md              framework article + toctree over the p0* notebooks
├── dataset.md            dataset article + the OptiWindNet RouteSets database
│
├── notebooks/            all .ipynb sources
│   └── README.md         folder overview for the GitLab/GitHub web UI (not a page)
├── figuredata/           precomputed router results read by figures.py,
│                         plus generation scripts (not pages)
├── autoapi/              GENERATED from optiwindnet/ — never edit, gitignored
├── milp_formulation/     LaTeX source, independent build rules and MathML fragment
├── _static/              logos, fonts, styles, and the fig_*.svg written by figures.py
└── build/                build output, gitignored
```

`index.md` is the only file with toctrees for the top-level sections; the two API pages each hold the toctrees for their own notebooks.

`milp_formulation/problem_formulation.html` is an insertion-ready MathML fragment generated from the LaTeX source alongside it. Sphinx includes the fragment directly and copies the scoped CSS and subsetted fonts into the built site's `_static/milp_formulation/` directory. The normal documentation build must not run `milp_formulation/Makefile`: its Pandoc, FontTools, XeLaTeX and font-subsetting toolchain is intentionally outside _OptiWindNet_'s documentation dependencies.

## Building and checking

```sh
make -C docs html      # build; fails on any warning
make -C docs check     # the checks Sphinx cannot do
make -C docs figures   # regenerate _static/fig_*.svg (outputs are committed)
```

Both invoke `$(PYTHON)`, which defaults to bare `python`. Where only the virtualenv has one, point it there — on the command line or from the environment, both work:

```sh
make -C docs html PYTHON=.venv/bin/python
export PYTHON=$PWD/.venv/bin/python
```

Dependencies are declared in the `docs` extra in `pyproject.toml`; install with `pip install -e .[docs]`.

Graphviz is _not_ required, despite `sphinx.ext.inheritance_diagram` being enabled — no page uses the directive, so `dot` is never invoked.

### Windows

There is no `make.bat`. Either install make — `conda install -c conda-forge make` — or invoke Sphinx directly with the same flags the Makefile uses:

```bat
cd docs
python -m sphinx -M html . build -W --keep-going
python check_docs.py
```

Do not drop `-W --keep-going`: without it a broken cross-reference builds green locally and fails in CI.

`SPHINXOPTS` is set to `-W --keep-going`, so **any warning fails the build** — a broken cross-reference is an error, not a line that scrolls past. Read the Docs bypasses the Makefile, so it gets the same behaviour from `sphinx.fail_on_warning` in `.readthedocs.yaml`. Both must be changed together.

Nitpicky mode (`-n`) is deliberately off; the reasoning, with numbers, is in `Makefile`.

## The two source formats behave differently

This is the main thing to know. `.md` pages are parsed by **MyST**; markdown cells in `.ipynb` notebooks are converted by **nbsphinx**, which does not understand MyST syntax.

|  | `.md` page | `.ipynb` markdown cell |
| --- | --- | --- |
| `{doc}`, `{ref}`, `{term}`, `{py:class}` roles | ✅ | ❌ **renders as literal text, no warning** |
| `[text](page.md)` | ✅ | ✅ |
| `[text](page.md#heading-slug)` | ✅ | ✅ |
| `[](page.md#heading-slug)` — text filled in from the heading | ✅ | ❌ **empty link text** |
| `(name)=` targets | ✅ defines, but not used here (see Pitfall 2) | ❌ |
| ` ```{admonition} ` and other directives | ✅ | ❌ |
| ``[`code`](page.md)`` | ✅ | ❌ **the link is lost** (Pitfall 4) |

### Pitfall 1 — MyST roles in a notebook are silently wrong

Writing `{doc}`/routers`` in a notebook markdown cell produces the visible text `{doc}`/routers`` on the page. Sphinx emits **no warning**. Notebooks must use relative markdown links:

```markdown
[Routers](../routers.md) ✅ {doc}`/routers` ❌ silently renders as literal text
```

`make check` catches this (`roles`), because the build cannot.

A _raw_ reST cell does support `:ref:`/`:doc:`, but it renders as unformatted text in JupyterLab, and these notebooks are meant to be read and run there — so don't.

### Pitfall 2 — heading text is a public API

Notebook links resolve `#anchor` against **heading slugs only**, so **rewording a heading in `problem.md`, `routers.md` or `reference/` breaks every link into it.** `-W` turns that into a build failure, which is the intended safety net — but expect to fix links whenever you retitle a section.

Pages link headings the same way, with the text left empty so MyST fills it in from the heading — which keeps the link text from drifting away from the title it names:

```markdown
[](/routers.md#exact-optimization) ✅ renders "Exact optimization" [](#optimization-approaches) ✅ same page {ref}`routers-exact` ❌ needs an explicit target
```

Explicit `(name)=` targets above headings were dropped for this: they were a second name for a place that already had one, reachable only from `.md` pages, and they left the two halves of the documentation linking in different styles. `make check` catches new ones (`targets`). Whole pages are still linked with `{doc}`, and `(name)=` above a _directive_ — the glossary is the only case — is unaffected.

### Pitfall 3 — don't guess a slug, read it

MyST collapses runs of separators, and the standalone slugify libraries disagree with it:

```
## Borders, Obstacles & Buffering   ->  #borders-obstacles-buffering
                          (not)         #borders-obstacles--buffering
```

Take the anchor from the built HTML (`grep -o 'id="[^"]*"' docs/build/html/…`), not from intuition.

### Pitfall 4 — a notebook link cannot contain a code span

nbsphinx converts markdown cells with pandoc, and reStructuredText has no nested inline markup. ``[`EWRouter`](../autoapi/optiwindnet/api/index.rst#optiwindnet.api.EWRouter)`` comes out of pandoc as `` `​``EWRouter``<…>`__``, which docutils reads as one literal holding the raw URL: **the link disappears and the URL becomes visible text**, with no warning. `<code>` tags in the link text fare no better — pandoc drops them.

So a reference into the API Reference is written as a plain-text link:

```markdown
[EWRouter](../autoapi/optiwindnet/api/index.rst#optiwindnet.api.EWRouter) ✅ [`EWRouter`](../autoapi/optiwindnet/api/index.rst#optiwindnet.api.EWRouter) ❌
```

`_static/apilinks.css` gives those links back the monospace they had to give up, by matching `a.reference[href*="#optiwindnet."]` — object anchors only, so the page-level links the sidebar and toctrees make to the same pages are untouched.

## Formatting the prose

Markdown here is formatted by Prettier, configured at the repository root in `.prettierrc` (`proseWrap: never`). `make -C docs check` runs `prettier --check` next to `check_docs.py`, so a page wrapped at a column fails the build the same way a broken cross-reference does.

```sh
prettier --write "**/*.md"                     # the pages
python docs/format_notebook_prose.py --write   # the notebooks' markdown cells
```

`never` unwraps paragraphs, which is the point: a contribution hard-wrapped at 80 columns normalises to the shape every other page has, and editing a paragraph no longer reflows its neighbours. The cost is that a paragraph is one long line — break it where the topic changes, not at a column.

Prettier has no `.ipynb` parser, so the notebooks are invisible to it. That is what `docs/format_notebook_prose.py` is for: it lifts the markdown cells out, formats them as Markdown, and puts them back, leaving code cells and outputs alone.

Two things need protecting from `never`, both with Prettier's own fences:

- **Definition lists** — it joins a term and its `: definition` onto one line, and MyST then renders the pair as a paragraph, silently. They sit between `<!-- prettier-ignore-start -->` and `<!-- prettier-ignore-end -->`, spanning the whole list, indented continuations included.
- **Deliberate line breaks**, such as the badge block in the root `README.md`, which takes a single `<!-- prettier-ignore -->` above it.

## Notebooks

### Outputs are committed

`nbsphinx_execute = 'never'`: the published pages show **the outputs stored in the `.ipynb` files**. A stale output stays stale until the notebook is re-run.

```sh
.venv/bin/python docs/run_notebooks.py --kernel <name> --changed
```

It executes, then normalises the JSON so diffs stay small (drops transient metadata, renumbers execution counts, prettifies SVG, marks long-output cells `scrolled`). MILP notebooks are skipped unless `--milp` or explicit paths are given, because they are slow.

### Editing notebook JSON by hand

If you script an edit: the files are `json.dump(..., indent=1)` with a trailing newline, and `source` is a **list of lines**, not one string. Preserve both or every notebook shows up as fully rewritten in the diff.

### Naming

`<api><band><n>_<topic>` — `hi` (Network/Router API), `lo` (Advanced API), `p` (paper).

| Band | Contents                  |
| ---- | ------------------------- |
| `0`  | quickstart                |
| `1`  | input data and inspection |
| `2`  | routers                   |
| `3`  | shaping the solution      |
| `4`  | worked examples           |
| `5`  | integration               |
| `9`  | appendix                  |

Paired notebooks share band, index and topic stem (`hi21_hgs` ↔ `lo21_hgs`), share an emoji in their title, and link to each other from the `**Concepts:**` header line. Gaps are deliberate — `hi22` is empty because `lo22_lkh` has no high-level counterpart.

### Adding a notebook

1. Name it per the scheme above.
2. Give it a short title and a `**Concepts:**` header line as its first markdown cell.
3. **Add it to the toctree** in `high_level_api.md` / `low_level_api.md` / `paper.md`. These list notebooks explicitly rather than by `:glob:`, precisely so a mis-numbered file fails the build instead of silently disappearing from the sidebar.
4. Mention it in `notebooks/README.md` (see below).
5. `make -C docs html && make -C docs check`.

## Two READMEs that are not pages

- **`docs/README.md`** — this file.
- **`docs/notebooks/README.md`** — gives `docs/notebooks/` a readable overview in the GitLab/GitHub web interface. It duplicates documentation content **on purpose**, for readers who never reach the built site, so it must be kept in sync by hand. Being outside the build it can use only plain relative links — no `{doc}`/`{ref}`/`{term}`.

`make check` verifies that every notebook it names exists and that every notebook is named, since nothing in the Sphinx build looks at it.

## What `make check` covers

Only the failure modes the Sphinx build cannot see. Broken references of every kind are already caught by `-W`, and are deliberately not duplicated.

| Check | Catches |
| --- | --- |
| `roles` | MyST roles in notebook markdown cells (Pitfall 1) |
| `readme` | `notebooks/README.md` drifting from the actual files |
| `targets` | explicit `(name)=` targets reintroduced above a heading (Pitfall 2) |
| `apilinks` | API names left as bare code spans in notebooks, and API links whose anchor does not belong to the page they name |
| `crosslinks` | opt-in (`--require-crosslinks`): notebooks with no link to any prose page |

`apilinks` reads the library's `__all__` literals with `ast` — the same list that decides what gets a page — so it needs neither a built `autoapi/` nor an import of optiwindnet. A code span that only looks like an API name, such as `` `MILP` `` naming the optimization approach rather than the package, is recorded in `NOT_A_REFERENCE`.

## The API Reference

`autoapi/` is generated from `optiwindnet/` and never edited by hand, but what it contains is decided by `__all__` in the library and by a handful of handlers in `conf.py`.

### What gets a page, and where

Every object is documented in exactly one place, chosen by `__all__`:

| The module declares | Effect |
| --- | --- |
| nothing | only what it defines is documented; every _imported_ name is hidden |
| `__all__ = ()` | the module gets no page at all (`skip_empty_all_submodules`) |
| a non-empty `__all__` | it owns the names it lists, wherever else they are imported |

`__all__` is all-or-nothing: once a module declares one, anything missing from it disappears from the page. Adding a single name means listing everything else the page should keep.

When a name one module lists is defined by another that has its own page, `skip_reexports_of_documented_modules` drops the second copy and leaves a stub linking to the definition. The stub carries `:no-index-entry:`, so it joins the Module Contents outline without a second entry in the general index — but it **is** a cross-reference target. A relative reference such as ``:class:`.ModelOptions``` can then match more than one object and fail the build. Write the path out.

### Type paths naming a module with no page

autoapi annotates with the module that _defines_ an object, not the one that exports it. Where that module is hidden — `optiwindnet.db.model`, `optiwindnet.MILP._core` — the path it writes leads nowhere, so `_REEXPORTED_PATHS` rewrites those pages to the exported path the reader can follow and does import. `[source]` links are untouched and still point at the defining file.

### A colon in a docstring's first line becomes a type

Napoleon splits the first line of a `Returns:` or `Yields:` block — and of an attribute docstring — on its first colon, taking everything left of it as the **type**:

```
Returns:
  list of interferences, where each is:
    ((u, v, s, t), touching vertex)
```

reaches the reader as

```
Return type:  list of interferences, where each is
Returns:      ((u, v, s, t), touching vertex)
```

with the prose shredded into cross-references on commas, brackets, `|` and the bare words `or` and `of`. Neither `literals` nor roles shield the line — only the absence of a colon does. A colon on any later line is fine.

### Cross-references into other projects

- intersphinx matches `objects.inv` **exactly**, and that a name is importable upstream says nothing about how it is indexed: networkx exports `PlanarEmbedding` at the top level but indexes it only under `networkx.algorithms.planarity`, and peewee indexes its classes with no package prefix at all. `_INVENTORY_RENAMES` maps such names.
- intersphinx also filters by **object type**, where local resolution ignores it. Annotations are emitted as `py:class` references, so an upstream alias published as `py:data` or `py:attribute` (`numpy.typing.NDArray`, `numpy.float64`) stays invisible until retried as `py:obj`; `retarget_upstream_types` does both fixups, ahead of intersphinx's own handler. A local `py:data` target, such as the `geometric` array aliases, resolves from a `py:class` reference untouched and needs none of this.
- `tuple[int, ...]` renders as `tuple[int, Ellipsis]`, and `Ellipsis` has no target, so it sits in `nitpick_ignore`.

### Two autoapi settings that are not optional

- **`autoapi_add_toctree_entry = True`** is the only thing that writes `autoapi/index.rst`, which the Reference toctree in `index.md` points at. Off, that page is never generated and a build from a clean tree fails on the dangling reference. It appends nothing to the master toctree regardless, because autoapi skips that when a toctree already names an autoapi document — `index.md` does.
- **`autoapi_keep_files = True`** is needed for notebook links (see below), but autoapi never deletes a page it has stopped generating. After changing what is documented, `rm -rf docs/autoapi` before rebuilding, or a module that should have vanished keeps building from a leftover file — and any warning count is measured against it.

## Other things worth knowing

- **`autoapi/` is generated** from `optiwindnet/` on every build and is gitignored. Never edit it. `autoapi_keep_files = True` keeps the `.rst` on disk because nbsphinx only resolves a link whose target exists as a source file — that is what lets a notebook link into the API Reference.
- **intersphinx hosts are build dependencies.** With `-W`, an unreachable inventory fails the build, and the warning has no subtype so it cannot be suppressed selectively. The mapping list in `conf.py` is kept to hosts that measurably resolve something.
- **The Binder and "Edit on GitLab" buttons** on each notebook page are generated by `nbsphinx_prolog` from the document path, so renaming a notebook silently repoints them. They are only correct once the rename is pushed.
- **`conf.py` has a per-file `E501` ignore** in `pyproject.toml`: it embeds HTML and Jinja in raw strings, where wrapping a line would change the emitted markup.
