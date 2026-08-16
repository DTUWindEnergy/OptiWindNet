"""Check documentation invariants that the Sphinx build cannot see.

The Sphinx build already catches broken cross-references, and does so for every
link style used in this project: ``{doc}``/``{ref}``/``{term}`` roles in ``.md``
pages, and relative ``[text](page.md#anchor)`` links in both ``.md`` pages and
notebook *markdown* cells. Run it with ``-W`` and those become build failures.
This script deliberately does **not** duplicate that work.

Four failure modes are invisible to the build, and are what this script covers:

``roles``
  A MyST role written in a notebook *markdown* cell -- ``{doc}``, ``{ref}``,
  ``{term}``, ``{py:class}`` -- renders as literal text. nbsphinx passes markdown
  cells through its own converter, which does not know MyST roles, so the reader
  sees the raw ``{doc}`` markup and Sphinx emits no warning. Notebooks must use
  relative markdown links instead. (A *raw* reST cell does support ``:ref:`` and
  ``:doc:``, but renders as unformatted text in JupyterLab, so it is unsuitable
  for notebooks meant to be read in Jupyter.)

``readme``
  ``notebooks/README.md`` exists to give ``docs/notebooks/`` a readable overview
  in the GitLab/GitHub web interface, and is excluded from the Sphinx build
  (``*/README.md`` in ``exclude_patterns``). It duplicates documentation content
  on purpose, for an audience that never reaches the built site, so nothing in
  the build notices when it goes stale. This check asserts that every notebook it
  names exists, and that every notebook is named.

``targets``
  Every heading already carries an anchor -- ``myst_heading_anchors`` gives it one --
  and that anchor is the only kind a *notebook* can link to. An explicit ``(name)=``
  target above a heading is therefore a second name for the same place, reachable
  only through ``{ref}`` from another ``.md`` page, and it splits the two link styles
  apart for no gain. Sphinx never complains about one, so this check does. Pages link
  headings by slug -- ``[](/routers.md#exact-optimization)``, whose empty text
  MyST fills in from the heading -- and whole pages with ``{doc}``.

``apilinks``
  A notebook naming a public API object in a ``code span`` reaches the reader as
  monospaced text and nothing more: nothing in the build knows the name means the
  object, so nothing warns that it could have been a link into the API Reference.
  This check requires every name in an ``__all__`` of the library to be written as
  a link where a notebook mentions it, and it re-reads the links already there --
  the page a link names, and whether its anchor belongs to that page. The build
  validates the document half of ``page.rst#anchor`` and passes the fragment
  through untouched, so a wrong anchor is a dead link that builds clean.

  A code span that merely *looks* like an API name -- ``MILP`` naming the
  optimization approach rather than the package -- goes in ``NOT_A_REFERENCE`` below.

``crosslinks`` (opt-in, ``--require-crosslinks``)
  Every notebook links to at least one prose page. Currently failing by design --
  it is the progress meter for the cross-linking pass, not a gate.
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB_DIR = HERE / 'notebooks'
README = NB_DIR / 'README.md'

# MyST roles render as literal text inside notebook markdown cells.
ROLE_PATTERN = re.compile(r'\{(?:doc|ref|term|numref|eval-rst|py:[a-z]+)\}`')

# A notebook stem or an abbreviated reference to one: hi00_quickstart, hi10, p03.
TOKEN_PATTERN = re.compile(r'\b([a-z]{1,2})(\d{2})((?:_[a-z0-9]+)*)\b')

# `hi10`-`hi14` written with an en dash, hyphen, or the word "to".
RANGE_PATTERN = re.compile(
    r'`?([a-z]{1,2})(\d{2})(?:_[a-z0-9_]+)?`?\s*(?:[-‐-―]|to)\s*'
    r'`?([a-z]{1,2})(\d{2})(?:_[a-z0-9_]+)?`?'
)


def markdown_text(path: Path) -> str:
    """Concatenated source of every markdown cell in a notebook."""
    nb = json.loads(path.read_text(encoding='utf8'))
    return '\n'.join(
        ''.join(cell.get('source', ''))
        for cell in nb.get('cells', ())
        if cell.get('cell_type') == 'markdown'
    )


def notebook_paths() -> list[Path]:
    return sorted(NB_DIR.glob('*.ipynb'))


def check_roles(paths: list[Path]) -> list[str]:
    """MyST roles in notebook markdown cells render as literal text."""
    problems = []
    for path in paths:
        for lineno, line in enumerate(markdown_text(path).splitlines(), 1):
            for match in ROLE_PATTERN.finditer(line):
                problems.append(
                    f'{path.name}: markdown cell line {lineno}: '
                    f'MyST role {match.group(0)}...` renders as literal text; '
                    f'use a relative markdown link instead'
                )
    return problems


def check_readme(paths: list[Path]) -> list[str]:
    """notebooks/README.md must name existing notebooks, and name them all."""
    if not README.is_file():
        return [f'{README.name}: missing (it serves the GitLab/GitHub folder view)']

    text = README.read_text(encoding='utf8')
    stems = {path.stem for path in paths}
    problems = []

    # Every token in the README must resolve to an existing notebook, either
    # exactly (hi00_quickstart) or as an abbreviation of one (hi10, p03).
    referenced: set[str] = set()
    for match in TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        matched = {stem for stem in stems if stem == token or stem.startswith(token)}
        if matched:
            referenced |= matched
        else:
            problems.append(f'{README.name}: names `{token}`, which does not exist')

    # Ranges (hi10-hi14) mention only their endpoints; expand them.
    for lo_alpha, lo_num, hi_alpha, hi_num in RANGE_PATTERN.findall(text):
        if lo_alpha != hi_alpha:
            continue
        span = range(int(lo_num), int(hi_num) + 1)
        referenced |= {
            stem
            for stem in stems
            if stem.startswith(lo_alpha)
            and stem[len(lo_alpha) : len(lo_alpha) + 2].isdigit()
            and int(stem[len(lo_alpha) : len(lo_alpha) + 2]) in span
        }

    for stem in sorted(stems - referenced):
        problems.append(f'{README.name}: does not mention `{stem}`')
    return problems


HEADING_PATTERN = re.compile(r'^#{1,6} (.+)$', re.MULTILINE)
TARGET_PATTERN = re.compile(r'\(([a-z0-9-]+)\)=')


def heading_slug(heading: str) -> str:
    """Approximate MyST's heading-anchor slug.

    MyST strips punctuation and collapses runs of separators, so
    ``Borders, Obstacles & Buffering`` becomes ``borders-obstacles-buffering``
    (not ``...--buffering``). Validated against the ids in the built HTML.
    """
    slug = re.sub(r'[^\w\s-]', '', heading.strip().lower(), flags=re.UNICODE)
    return re.sub(r'[\s_-]+', '-', slug).strip('-')


# Directories excluded from prose-page checks: notebooks/ is checked separately,
# build/ contains generated output, and figuredata/ contains figure inputs.
NON_PAGE_DIRS = frozenset(('notebooks', 'build', 'figuredata'))


def prose_pages() -> list[Path]:
    root = HERE
    return sorted(p for p in root.glob('**/*.md') if NON_PAGE_DIRS.isdisjoint(p.parts))


def check_targets(paths: list[Path]) -> list[str]:
    """A heading carries its own anchor; an explicit target above it is redundant."""
    problems = []
    for page in prose_pages():
        lines = page.read_text(encoding='utf8').splitlines()
        for lineno, line in enumerate(lines[:-1], 1):
            target = TARGET_PATTERN.fullmatch(line)
            if target and lines[lineno].startswith('#'):
                slug = heading_slug(lines[lineno].lstrip('# '))
                problems.append(
                    f'{page.relative_to(HERE)}:{lineno}: explicit target '
                    f'({target.group(1)})= above a heading; the heading already '
                    f'answers to page.md#{slug}, which notebooks can link to and '
                    f'{{ref}} cannot. Drop the target and link the slug'
                )
    return problems


PACKAGE = HERE.parent / 'optiwindnet'

# A code span holding exactly an API name, with or without a call suffix, and
# optionally qualified by the instance or class it is reached through: `EWRouter`,
# `clusterize()`, `wfn.plot()`, `PathFinder(...).create_detours()`.
API_SPAN = re.compile(r'`([A-Za-z_][\w.]*(?:\(\.\.\.\))?\.)?([A-Za-z_]\w*)(\(\))?`')

# Links into the API Reference: ../autoapi/<module path>/index.rst#<dotted name>.
API_LINK = re.compile(r'\[([^\]]*)\]\(\.\./autoapi/([^)#]+)#([^)]+)\)')

FENCE_LINE = re.compile(r'^\s{0,3}(?:```|~~~)')
HEADING_LINE = re.compile(r'^\s{0,3}#')

# Code spans that match a public name without referring to the object.
NOT_A_REFERENCE = {
    # "With `MILP`, we observe ..." names the optimization approach, not the package.
    ('hi31_options.ipynb', 'MILP'),
}


def public_names() -> set[str]:
    """Every name the library exports, read from the ``__all__`` literals.

    ``__all__`` is what decides whether an object gets a page at all (see the API
    Reference section of docs/README.md), so it is also the list of names a notebook
    can link to. Parsed rather than imported: importing optiwindnet pulls in Numba.
    """
    names: set[str] = set()
    for source in sorted(PACKAGE.rglob('*.py')):
        for node in ast.parse(source.read_text(encoding='utf8')).body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(getattr(t, 'id', None) == '__all__' for t in node.targets):
                continue
            try:
                names.update(ast.literal_eval(node.value))
            except ValueError:
                pass  # not a literal; nothing to read
    return {n for n in names if not n.startswith('_')}


def prose_lines(path: Path) -> list[tuple[int, str]]:
    """Markdown-cell lines a reader sees as prose: no headings, no code blocks.

    A link in a heading buys the reader nothing and puts the section anchors that
    other notebooks link to at risk, so headings are left out of this check.
    """
    lines, in_fence = [], False
    for lineno, line in enumerate(markdown_text(path).splitlines(), 1):
        if FENCE_LINE.match(line):
            in_fence = not in_fence
        elif not in_fence and not HEADING_LINE.match(line):
            lines.append((lineno, line))
    return lines


def check_apilinks(paths: list[Path]) -> list[str]:
    """API names in notebook prose must be links, and those links must be right."""
    exported = public_names()
    problems = []
    for path in paths:
        for lineno, line in prose_lines(path):
            where = f'{path.name}: markdown cell line {lineno}'
            for _text, page, anchor in API_LINK.findall(line):
                module = page.rsplit('/', 1)[0].replace('/', '.')
                if not anchor.startswith(f'{module}.'):
                    problems.append(
                        f'{where}: {page} does not describe {anchor}; '
                        f'the anchor has to name an object of {module}'
                    )
            # Spans inside a link are its text, not an unlinked mention.
            for match in API_SPAN.finditer(API_LINK.sub('', line)):
                name = match.group(2)
                if name in exported and (path.name, name) not in NOT_A_REFERENCE:
                    span = match.group(0)[1:-1]
                    problems.append(
                        f'{where}: `{span}` names the exported {name} but does not '
                        f'link to it; write it as '
                        f'[{span}](../autoapi/.../index.rst#...), or record it in '
                        f'NOT_A_REFERENCE if it is not a reference'
                    )
    return problems


def check_crosslinks(paths: list[Path]) -> list[str]:
    """Every notebook should link to at least one prose page."""
    problems = []
    for path in paths:
        if '](../' not in markdown_text(path):
            problems.append(f'{path.name}: no link to any prose page')
    return problems


CHECKS = {
    'roles': check_roles,
    'readme': check_readme,
    'targets': check_targets,
    'apilinks': check_apilinks,
    'crosslinks': check_crosslinks,
}
DEFAULT_CHECKS = ('roles', 'readme', 'targets', 'apilinks')


def run(args: argparse.Namespace) -> int:
    paths = notebook_paths()
    if not paths:
        sys.exit(f'no notebooks found in {NB_DIR}')

    selected = list(DEFAULT_CHECKS)
    if args.require_crosslinks:
        selected.append('crosslinks')
    if args.only:
        selected = args.only

    failed = 0
    for name in selected:
        problems = CHECKS[name](paths)
        status = f'{len(problems)} problem(s)' if problems else 'ok'
        print(f'[{name}] {status}')
        for problem in problems:
            print(f'  {problem}')
        failed += len(problems)

    print(f'\n{len(paths)} notebooks checked, {failed} problem(s)')
    return 1 if failed else 0


def argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--require-crosslinks',
        action='store_true',
        help='also require every notebook to link to a prose page',
    )
    p.add_argument(
        '--only',
        nargs='+',
        choices=sorted(CHECKS),
        help='run only the named checks',
    )
    return p


def main() -> int:
    return run(argument_parser().parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
