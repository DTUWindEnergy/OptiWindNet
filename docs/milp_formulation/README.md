# Optimization problem formulation

This directory contains a LaTeX optimization-problem definition and build rules for producing PDF, SVG, an insertion-ready MathML HTML fragment, and an optional standalone HTML rendering.

## Build

The build environment needs:

- GNU Make
- Pandoc
- FontTools with Brotli/WOFF2 support
- curl
- XeLaTeX with the STIX Two OpenType fonts installed
- `pdftocairo` for SVG generation

Available targets:

```sh
make html
make html-standalone
make pdf
make svg
make fonts
```

`make html` regenerates only the insertion-ready MathML fragment, without an `<html>`, `<head>` or `<body>` wrapper.

`make html-standalone` additionally produces `problem_formulation_standalone.html`, linked to the same CSS and fonts. The standalone output is for local inspection and is not distributed with _OptiWindNet_.

`make fonts` writes content-specific font subsets to `fonts/`. It downloads the pinned Fontsource WOFF2 inputs only when they are missing and caches them in `fonts-full/`. The downloaded text inputs are the Latin slices required by the current English formulation. Additional scripts require corresponding Fontsource slices in the Makefile and CSS.

`make pdf` compiles the TeX source with XeLaTeX. `make svg` first updates the PDF and then converts it with `pdftocairo`.

## Buildable source package

To distribute the source in a form that can regenerate every output, include:

```text
problem_formulation.tex
problem_formulation.css
Makefile
README.md
```

The generated outputs, `fonts/`, and the `fonts-full/` download cache can be omitted. Running `make html` recreates the HTML fonts and downloads their SIL Open Font License. The PDF build still expects STIX Two to be installed in the TeX environment.

## Documentation package

The _OptiWindNet_ documentation distribution includes:

```text
problem_formulation.tex
problem_formulation.html
problem_formulation.css
Makefile
README.md
fonts/*.woff2
fonts/OFL-1.1.txt
```

`problem_formulation.html` is the fragment incorporated into the Sphinx page. The standalone HTML output is not distributed. Preserve the CSS and `fonts/` relative paths. Do not include `fonts-full/`; it is only a build cache. `fonts/README.md` may be included as additional provenance documentation but is not required for rendering.

The HTML uses native MathML and locally hosted, content-specific STIX Two WOFF2 subsets. The font subsets remain covered by the SIL Open Font License, Version 1.1, supplied in `fonts/OFL-1.1.txt`.
