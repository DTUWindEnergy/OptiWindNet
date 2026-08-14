# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
# -W turns warnings into errors so that a broken cross-reference fails the build
# instead of scrolling past; --keep-going reports all of them, not just the first.
#
# Nitpicky mode (-n) stays OFF, though little is left for it to report: 12 warnings, of
# which 9 name solver backends that publish no intersphinx inventory (pyscipopt, pyomo,
# ortools) and 3 name _Link, a private alias in MILP._core carried by public
# ModelMetadata fields. Both groups would need nitpick_ignore entries rather than fixes,
# and an ignore list is what makes -n hide the breakage it exists to catch. To measure,
# clear docs/autoapi first (see README) and build with -n:
#   python -m sphinx -b html -n -w /tmp/nitpick.txt . /tmp/nb
SPHINXOPTS    = -W --keep-going
# One knob for the interpreter, honoured by every target. `?=` so that an exported
# PYTHON works too: a plain `=` in a makefile overrides the environment, and only a
# command-line assignment would get through.
PYTHON       ?= python
SPHINXBUILD   = $(PYTHON) -msphinx
SPHINXPROJ    = OptiWindNet
SOURCEDIR     = .
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Documentation invariants the Sphinx build cannot see; see check_docs.py, plus
# the Markdown format itself. Prettier runs from the repository root, where its
# .prettierrc is, and the glob covers every Markdown page in the project.
# It needs node, which the docs extra does not install, so a missing prettier
# says so and is skipped rather than failing a Python-only checkout.
check:
	@$(PYTHON) check_docs.py
	@cd .. && if command -v prettier >/dev/null 2>&1; then \
		prettier --check "**/*.md"; \
	else \
		echo 'prettier not found: skipping the Markdown format check'; \
		echo '  install it with `npm install --global prettier`'; \
	fi

# Rebuild the SVG figures embedded in the prose pages; see figures.py. Their
# outputs are committed, like the notebooks', so this is not part of `make html`.
figures:
	@cd .. && $(PYTHON) -m docs.figures

.PHONY: help check figures Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
