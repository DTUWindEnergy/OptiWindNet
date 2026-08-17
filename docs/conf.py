#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import glob
import importlib
import os
import re
import shutil
from importlib.metadata import version as get_version
from pathlib import Path

# nbsphinx converts notebook markdown cells by shelling out to a `pandoc` executable
# (via nbconvert), so pandoc must be on PATH. pypandoc-binary ships one inside the
# package, but pypandoc only exposes it to its own subprocess calls - it never touches
# the ambient environment - so put it on PATH here for nbconvert's benefit.
if shutil.which('pandoc') is None:
    try:
        import pypandoc
    except ImportError:
        pass
    else:
        _bundled = os.path.join(os.path.dirname(pypandoc.__file__), 'files')
        if glob.glob(os.path.join(_bundled, 'pandoc*')):
            os.environ['PATH'] += os.pathsep + _bundled

release: str = get_version('optiwindnet')
# for example take major/minor
version: str = '.'.join(release.split('.')[:2])

# -- Project information -----------------------------------------------------

project = 'OptiWindNet'
copyright = '2026, DTU Wind Energy'
author = 'DTU Wind Energy'

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',
    'nbsphinx',
    #  'myst_nb',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    #  'sphinx.ext.autosectionlabel',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.doctest',
    'autoapi.extension',
    # 'sphinx.ext.imgconverter',
]

# `deflist` renders the `term` / `: definition` blocks on the Task Index, Validation,
# Results and Input pages; without it they reach the reader as paragraphs opening with
# a stray colon.
myst_enable_extensions = ['deflist', 'html_image']
myst_heading_anchors = 3

# Resolve type annotations in the autoapi-generated pages against upstream docs.
# Without these, nitpicky mode (-n) reports several hundred unresolvable references,
# dominated by networkx.Graph and numpy.ndarray.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),  # 597 links
    'networkx': ('https://networkx.org/documentation/stable/', None),  # 122
    'numpy': ('https://numpy.org/doc/stable/', None),  # 41
    'matplotlib': ('https://matplotlib.org/stable/', None),  # 4
    'bidict': ('https://bidict.readthedocs.io/en/main/', None),  # 4
    'peewee': ('https://docs.peewee-orm.com/en/latest/', None),  # 1
    'shapely': ('https://shapely.readthedocs.io/en/stable/', None),  # 2
}
# Every host here is a build dependency: with -W, an unreachable inventory fails the
# build, and the warning carries no subtype so it cannot be suppressed selectively.
# scipy and pyomo were dropped after measuring - they resolved nothing.
intersphinx_timeout = 15

# autoapi renders `tuple[int, ...]` as `tuple[int, Ellipsis]`, and `Ellipsis` has no
# py:class target. The annotations are correct; only their rendering is not.
#
# optiwindnet.db.model.BaseModel only carries the Meta that binds the tables to
# database_proxy. optiwindnet.db does not re-export it, so it has no page, but
# 'show-inheritance' still names it on each table class.
nitpick_ignore = [
    ('py:class', 'Ellipsis'),
    ('py:obj', 'BaseModel'),
]

# Add any paths that contain templates here, relative to this directory.
#  templates_path = ['_templates']

# Auto API conf.:
autoapi_dirs = ['../optiwindnet']
#  autoapi_options = [ 'members', 'undoc-members', 'private-members', 'show-inheritance', 'show-module-summary', 'special-members', 'imported-members', ]
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'special-members',
    'imported-members',
]
autoapi_python_class_content = 'both'
# Named explicitly: the handlers below derive page names from it.
autoapi_root = 'autoapi'
# This also controls whether autoapi writes autoapi/index.rst at all, which index.md
# points its Reference toctree at, so it cannot be turned off: with it False the page is
# never generated and a build from a clean tree fails on the dangling reference. The
# appending it is named after does not happen anyway - autoapi skips that when a toctree
# already names an autoapi document, which index.md does - so placement stays explicit.
autoapi_add_toctree_entry = True
# keep the generated .rst in the source tree so notebooks can link to it
# (nbsphinx only resolves links whose target exists as a source file)
autoapi_keep_files = True

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    # Maintainer notes, not pages. '*/README.md' does not match a file at the source
    # root, so docs/README.md needs its own entry; without it the build would fail
    # under -W with "document isn't included in any toctree".
    'README.md',
    '*/README.md',
    # Slow notebook:
    # 'notebooks/neural_network_with_tfds_data.ipynb',
    # ipynb checkpoints
    'notebooks/.ipynb_checkpoints/*.ipynb',
    'notebooks/figure_notebooks/*.ipynb',
    'build/*',
    # figuredata/ contains figure inputs and generation scripts, not documentation pages.
    # See figuredata/README.md.
    'figuredata/*',
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'default'
pygments_dark_style = 'monokai'


autosummary_generate = True
napolean_use_rtype = False

# -- Options for nbsphinx -----------------------------------------------------

# Execute notebooks before conversion: 'always', 'never', 'auto' (default)
# We execute all notebooks, exclude the slow ones using 'exclude_patterns'
nbsphinx_execute = 'never'

# Use this kernel instead of the one stored in the notebook metadata:
# nbsphinx_kernel_name = 'python3'

# List of arguments to be passed to the kernel that executes the notebooks:
# nbsphinx_execute_arguments = []

# If True, the build process is continued even if an exception occurs:
# nbsphinx_allow_errors = True


# Controls when a cell will time out (defaults to 30; use -1 for no timeout):
nbsphinx_timeout = 180

# Default Pygments lexer for syntax highlighting in code cells:
nbsphinx_codecell_lexer = 'python3'

# Width of input/output prompts used in CSS:
# nbsphinx_prompt_width = '8ex'

# If window is narrower than this, input/output prompts are on separate lines:
# nbsphinx_responsive_width = '700px'

# This is processed by Jinja2 and inserted before each notebook

nbsphinx_prolog = r"""
{% set doc_path = 'docs/' + env.doc2path(env.docname, base=None) | string | replace("\\", "/") %}
{% set doc_path_quoted = 'doc/tree/' + doc_path | urlencode %}


.. only:: html
    
    .. raw:: html

        <style>
        .stderr { background: transparent !important; border: dashed red; }
        </style>

    .. nbinfo::

        .. raw:: html

            <a href="https://mybinder.org/v2/gh/DTUWindEnergy/OptiWindNet/mybinder?urlpath={{ doc_path_quoted }}" target="_blank">
            <img alt="Launch with Binder" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom">
            </a>
            <a href="https://gitlab.windenergy.dtu.dk/TOPFARM/OptiWindNet/-/tree/main/{{ doc_path }}">
            <img alt="Edit on Gitlab" src="https://img.shields.io/badge/Edit%20on-Gitlab-blue?style=flat&logo=gitlab" style="vertical-align:text-bottom">
            </a>

"""
# Alternatively, the binder badge may link to the DTU Wind Energy GitLab:
# <a href="https://mybinder.org/v2/git/https%3A%2F%2Fgitlab.windenergy.dtu.dk%2FTOPFARM%2FOptiWindNet/mybinder?urlpath={{ doc_path_quoted }}" target="_blank">

# This is processed by Jinja2 and inserted after each notebook
# nbsphinx_epilog = r"""
# """

# Input prompt for code cells. "%s" is replaced by the execution count.
# nbsphinx_input_prompt = 'In [%s]:'

# Output prompt for code cells. "%s" is replaced by the execution count.
# nbsphinx_output_prompt = 'Out[%s]:'

# Specify conversion functions for custom notebook formats:
# import jupytext
# nbsphinx_custom_formats = {
#    '.Rmd': lambda s: jupytext.reads(s, '.Rmd'),
# }

# Link or path to require.js, set to empty string to disable
# nbsphinx_requirejs_path = ''

# Options for loading require.js
# nbsphinx_requirejs_options = {'async': 'async'}

mathjax3_config = {
    'tex': {'tags': {'autoNumber': 'ams', 'useLabelIds': True}},
}

# Additional files needed for generating LaTeX/PDF output:
# latex_additional_files = ['references.bib']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#  html_theme_options = {
# TOC options
# 'navigation_depth': 2,  # only show 2 levels on left sidebar
#  'collapse_navigation': False,  # don't allow sidebar to collapse
#  }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'apilinks.css',
    'milp_formulation/problem_formulation.css',
]
html_theme_options = {
    'light_logo': 'OptiWindNet_boxed.svg',
    'dark_logo': 'OptiWindNet.svg',
}
html_title = f'version {release}'

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'OptiWindNetdoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        'OptiWindNet.tex',
        'OptiWindNet Documentation',
        'DTU Wind Energy',
        'manual',
    ),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, 'optiwindnet', 'OptiWindNet Documentation', [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        'OptiWindNet',
        'OptiWindNet Documentation',
        author,
        'OptiWindNet',
        'One line description of project.',
        'Miscellaneous',
    ),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']


# -- Extension configuration -------------------------------------------------


def skip_empty_all_submodules(app, what, name, obj, skip, options):
    # Only consider modules
    if what == 'module' and obj.all is not None and len(obj.all) == 0:
        # __all__ is empty -> do not document this module
        return True
    return None  # Use default behavior otherwise


# Objects re-exported by a package are documented under the package, but autoapi writes
# annotations with the path of the module that defines them. Users import from the
# package - the notebooks all do `from optiwindnet.db import RouteSet` - so the module
# path names something nobody types and, being hidden, has no page to link to either.
# Rewriting the generated pages rather than the resolved reference is deliberate: it
# corrects the path the reader sees, which a missing-reference handler cannot do.
_REEXPORTED_PATHS = (
    ('optiwindnet.db.model.', 'optiwindnet.db.'),
    # optiwindnet.MILP._core is private and undocumented, while every type it defines is
    # exported and documented by optiwindnet.MILP. autoapi spells the private module
    # qualified on the backend pages and relative to the package on the package's own,
    # so both renderings are rewritten.
    ('optiwindnet.MILP._core.', 'optiwindnet.MILP.'),
    ('_core.Solver', 'Solver'),
)


def use_reexported_paths(app, docname, source):
    """Name re-exported objects by the path they are documented under and imported by."""
    if not docname.startswith(f'{autoapi_root}/'):
        return
    text = source[0]
    for defining, documented in _REEXPORTED_PATHS:
        text = text.replace(defining, documented)
    entries = _REEXPORTS.get(docname)
    if entries:
        # autoapi has already rendered every page by the time source-read fires, so the
        # skips recorded above are complete. Placing the stubs directly under the
        # heading puts them at the head of the outline rather than after the members.
        stubs = _reexport_stubs(entries)
        text, substituted = _MODULE_CONTENTS_RE.subn(
            lambda m: m.group(1) + '\n' + stubs, text, count=1
        )
        if not substituted and stubs:
            # A module whose every member is re-exported gets no member section from
            # autoapi, leaving a page with nothing on it but its title.
            text = f'{text.rstrip()}\n\nModule Contents\n---------------\n\n{stubs}'
    source[0] = text


def skip_reexports_of_documented_modules(app, what, name, obj, skip, options):
    """Describe a re-exported object only on the page of the module that defines it.

    A module with a non-empty __all__ is documented by skip_empty_all_submodules
    above, so its members already have a page. Without this, 'imported-members' would
    describe them a second time wherever a package re-exports them - one class as both
    optiwindnet.db.model.NodeSet and optiwindnet.db.NodeSet - which is two pages to
    keep in step and two names to choose between when linking. Keep the definition and
    drop the copy, so every reference resolves to the same place.
    """
    if skip or what == 'module' or not getattr(obj, 'imported', False):
        # A member autoapi already drops is not this handler's to announce: it stays
        # hidden either way, and claiming it would stub names with no page anywhere.
        return None
    origin = obj.obj.get('original_path', '')
    if not origin.startswith('optiwindnet.'):
        # Third-party imports are documented in their own project, not in this tree.
        return None
    defining_name, _, origin_name = origin.rpartition('.')
    try:
        defining = importlib.import_module(defining_name)
    except ImportError:
        return None
    exported = getattr(defining, '__all__', ())
    if not exported:
        return None
    if origin_name in exported:
        # Only then does the definition get a page for the stub below to point at.
        parent, _, short_name = name.rpartition('.')
        docname = f'{autoapi_root}/{parent.replace(".", "/")}/index'
        _REEXPORTS.setdefault(docname, []).append((what, short_name, origin))
    return True


# Suppressing the copy silently would leave the name off the page a reader is on, even
# though it is importable from there. Each one instead gets a stub carrying a link to
# the entry that documents it. ':no-index-entry:' is what makes the stub a full member
# of the Module Contents outline while keeping the general index down to one entry per
# object; ':no-index:' would drop it from the outline too, which defeats the purpose.
_REEXPORTS: dict[str, list[tuple[str, str, str]]] = {}

_STUB_DIRECTIVE = {
    'class': 'py:class',
    'data': 'py:data',
    'exception': 'py:exception',
    'function': 'py:function',
}

_MODULE_CONTENTS_RE = re.compile(r'^(Module Contents\n-+\n)', re.MULTILINE)


def _reexport_stubs(entries):
    lines = ['.. rubric:: Re-exported', '']
    for what, short_name, origin in entries:
        directive = _STUB_DIRECTIVE.get(what)
        if directive is None:
            continue
        lines += [
            f'.. {directive}:: {short_name}',
            '   :no-index-entry:',
            '',
            f'   Importable from this module. Documented at :py:obj:`{origin}`.',
            '',
        ]
    return '\n'.join(lines) + '\n' if len(lines) > 2 else ''


# Upstream projects whose inventories are mapped above and whose names may need the
# fixups below. Extend this together with intersphinx_mapping.
_INTERSPHINX_ROOTS = ('numpy', 'networkx', 'matplotlib', 'bidict', 'peewee', 'shapely')

# networkx re-exports PlanarEmbedding at the top level but documents it under the
# module that defines it, so that is the only name in its objects.inv. peewee is the
# other way round: it documents its classes unqualified, with no package prefix at all.
# Runtime importability does not enter into it: intersphinx matches the target exactly.
_INVENTORY_RENAMES = {
    'networkx.PlanarEmbedding': 'networkx.algorithms.planarity.PlanarEmbedding',
    'peewee.DatabaseProxy': 'DatabaseProxy',
    'peewee.Model': 'Model',
    'peewee.SqliteDatabase': 'SqliteDatabase',
}


def retarget_upstream_types(app, env, node, contnode):
    """Fix up unresolved upstream annotations, leaving the lookup to intersphinx.

    Annotations are emitted as py:class references, which resolve only class,
    exception and type objects. numpy publishes numpy.typing.NDArray as py:data and
    numpy.float64 as py:attribute, so they are in the inventory but invisible to a
    class-only lookup; py:obj resolves every objtype and finds them.
    """
    target = node.get('reftarget', '')
    if node.get('refdomain') != 'py' or node.get('reftype') != 'class':
        return
    if target.partition('.')[0] not in _INTERSPHINX_ROOTS:
        # Leave project-local misses reported as py:class.
        return
    node['reftarget'] = _INVENTORY_RENAMES.get(target, target)
    node['reftype'] = 'obj'
    return  # intersphinx, at priority 500, resolves the rewritten node


def copy_milp_formulation_assets(app, exception):
    """Copy the formulation CSS and fonts without invoking its Makefile."""
    if exception is not None or app.builder.format != 'html':
        return
    source = Path(__file__).parent / 'milp_formulation'
    target = Path(app.outdir) / '_static' / 'milp_formulation'
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / 'problem_formulation.css', target)
    shutil.copytree(source / 'fonts', target / 'fonts', dirs_exist_ok=True)


def setup(app):
    app.connect('autoapi-skip-member', skip_empty_all_submodules)
    app.connect('autoapi-skip-member', skip_reexports_of_documented_modules)
    app.connect('source-read', use_reexported_paths)
    # Ahead of intersphinx's own missing-reference handler, which runs at 500.
    app.connect('missing-reference', retarget_upstream_types, priority=400)
    app.connect('build-finished', copy_milp_formulation_assets)
