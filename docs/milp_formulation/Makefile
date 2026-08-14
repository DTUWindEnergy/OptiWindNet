PANDOC ?= pandoc
FONTTOOLS ?= fonttools
XELATEX ?= xelatex
PDFTOCAIRO ?= pdftocairo
CURL ?= curl

TEX := problem_formulation.tex
CSS := problem_formulation.css
HTML := problem_formulation.html
HTML_STANDALONE := problem_formulation_standalone.html
PDF := problem_formulation.pdf
SVG := problem_formulation.svg
FONTSOURCE_VERSION := 5.3.0
FONTSOURCE_TEXT_URL := https://cdn.jsdelivr.net/npm/@fontsource/stix-two-text@$(FONTSOURCE_VERSION)
FONTSOURCE_MATH_URL := https://cdn.jsdelivr.net/npm/@fontsource/stix-two-math@$(FONTSOURCE_VERSION)
FULL_FONT_DIR := fonts-full
SUBSET_FONT_DIR := fonts
FONT_FILES := \
	stix-two-math-latin-400-normal.woff2 \
	stix-two-text-latin-400-normal.woff2 \
	stix-two-text-latin-400-italic.woff2 \
	stix-two-text-latin-600-normal.woff2 \
	stix-two-text-latin-600-italic.woff2
FULL_FONTS := $(addprefix $(FULL_FONT_DIR)/,$(FONT_FILES))
SUBSET_FONTS := $(addprefix $(SUBSET_FONT_DIR)/,$(FONT_FILES))
FONT_LICENSE := $(SUBSET_FONT_DIR)/OFL-1.1.txt
DOWNLOAD_FLAGS := --fail --location --retry 3 --silent --show-error

# Preserve broad font compatibility while allowing FontTools to close the
# glyph set over OpenType layout features and the mathematical MATH table.
SUBSET_FLAGS := \
	--text-file=$(HTML) \
	--flavor=woff2 \
	--layout-features='*' \
	--layout-scripts='*' \
	--glyph-names \
	--symbol-cmap \
	--legacy-cmap \
	--notdef-glyph \
	--notdef-outline \
	--recommended-glyphs \
	--name-IDs='*' \
	--name-legacy \
	--name-languages='*' \
	--hinting \
	--no-recalc-timestamp

.DELETE_ON_ERROR:
.SECONDARY: $(FULL_FONTS)

.PHONY: html html-standalone fonts pdf svg
html: $(HTML)

html-standalone: $(HTML_STANDALONE)

fonts: $(SUBSET_FONTS) $(FONT_LICENSE)

pdf: $(PDF)

svg: $(SVG)

$(HTML): $(TEX) Makefile
	$(PANDOC) $(TEX) \
		--from=latex \
		--to=html5 \
		--mathml \
		--output=$(HTML)

$(HTML_STANDALONE): $(TEX) $(CSS) Makefile
	$(PANDOC) $(TEX) \
		--from=latex \
		--to=html5 \
		--standalone \
		--mathml \
		--css=$(CSS) \
		--metadata title="Optimization problem formulation" \
		--output=$(HTML_STANDALONE)

$(PDF): $(TEX) Makefile
	$(XELATEX) -interaction=nonstopmode -halt-on-error $(TEX)

$(SVG): $(PDF)
	$(PDFTOCAIRO) -svg $(PDF) $(SVG)

$(SUBSET_FONT_DIR)/%.woff2: $(FULL_FONT_DIR)/%.woff2 $(HTML) Makefile | $(FONT_LICENSE)
	$(FONTTOOLS) subset $< \
		$(SUBSET_FLAGS) \
		--output-file=$@

$(FULL_FONT_DIR)/stix-two-text-%.woff2: | $(FULL_FONT_DIR)
	$(CURL) $(DOWNLOAD_FLAGS) \
		--output $@.tmp \
		$(FONTSOURCE_TEXT_URL)/files/$(@F)
	mv $@.tmp $@

$(FULL_FONT_DIR)/stix-two-math-latin-400-normal.woff2: | $(FULL_FONT_DIR)
	$(CURL) $(DOWNLOAD_FLAGS) \
		--output $@.tmp \
		$(FONTSOURCE_MATH_URL)/files/$(@F)
	mv $@.tmp $@

$(FONT_LICENSE):
	mkdir -p $(@D)
	$(CURL) $(DOWNLOAD_FLAGS) \
		--output $@.tmp \
		$(FONTSOURCE_TEXT_URL)/LICENSE
	mv $@.tmp $@

$(FULL_FONT_DIR):
	mkdir -p $@
