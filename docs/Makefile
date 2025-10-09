SPHINXOPTS ?=
SPHINXBUILD ?= sphinx-build
SOURCEDIR = source
BUILDDIR = _build

.PHONY: html html-fast clean serve api

api:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" -W --keep-going

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" -W --keep-going

html-fast:
	$(SPHINXBUILD) -b html -D nb_execution_mode=off "$(SOURCEDIR)" "$(BUILDDIR)/html"

serve:
	python -m http.server --directory $(BUILDDIR)/html 8000

livehtml:
	sphinx-autobuild -b html --host 127.0.0.1 --port 8000 \
	--watch ../afnio -D nb_execution_mode=off $(SOURCEDIR) $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR)
