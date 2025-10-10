# === Config ===
PY312          := python3.12
DOCS_VENV_DIR  := .venv-docs
PY             := $(DOCS_VENV_DIR)/bin/python
PIP            := $(DOCS_VENV_DIR)/bin/pip
SPHINXBUILD    := $(PY) -m sphinx
AUTOBUILD      := $(PY) -m sphinx_autobuild
SOURCEDIR = source
BUILDDIR = _build

# If you keep a pinned requirements file for docs, set it here (optional)
DOCS_REQ       := requirements.txt

# Default Sphinx opts; override at CLI: `make html SPHINXOPTS="-W --keep-going"`
SPHINXOPTS     ?= -D nb_execution_mode=off

# === Helpers ===
.PHONY: html livehtml clean nuke pip-freeze check-py312

check-py312:
	@command -v $(PY312) >/dev/null 2>&1 || \
	 { echo "ERROR: $(PY312) not found. Install python 3.12 (e.g., 'sudo apt install python3.12 python3.12-venv')."; exit 1; }

# Create or re-create venv if missing or wrong version
.ensure-venv: check-py312
	@if [ ! -x "$(PY)" ]; then \
	  echo "==> Creating docs venv ($(DOCS_VENV_DIR)) with $(PY312)"; \
	  $(PY312) -m venv $(DOCS_VENV_DIR); \
	fi; \
	$(PY) -c 'import sys; \
v= sys.version_info[:2]; \
assert v >= (3,12), f"Docs venv must be 3.12+, got {v}"' || { \
	  echo "==> Recreating venv with $(PY312)"; \
	  rm -rf $(DOCS_VENV_DIR); \
	  $(PY312) -m venv $(DOCS_VENV_DIR); \
	}; \
	$(PIP) install -U pip >/dev/null

# Install/upgrade docs deps (idempotent)
.ensure-deps: .ensure-venv
	@test -f "$(DOCS_REQ)" || { \
	  echo "ERROR: $(DOCS_REQ) not found. Create it and rerun."; \
	  exit 1; \
	}
	@echo "==> Installing docs requirements from $(DOCS_REQ)"
	@$(PIP) install -r "$(DOCS_REQ)"
	@echo "==> Installing afnio and its dependencies in docs venv"
	@$(PIP) install -e ..

# Build once (fast local: no notebook execution by default)
html: .ensure-deps
	@echo "==> Building HTML docs to $(BUILDDIR)/html"
	$(SPHINXBUILD) -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html
	@echo "Open: file://$(abspath $(BUILDDIR))/html/index.html"

# Live reload (watches docs/ and your package at ../afnio)
livehtml: .ensure-deps
	@echo "==> Live rebuilding docs on http://127.0.0.1:8000"
	$(AUTOBUILD) -b html $(SPHINXOPTS) \
	  --host 127.0.0.1 --port 8000 \
	  --watch ../afnio \
	  $(SOURCEDIR) $(BUILDDIR)/html

clean:
	@echo "==> Cleaning build output"
	rm -rf $(BUILDDIR)

# Blow away the venv and build; next 'make html' will recreate everything
nuke:
	@echo "==> Removing docs venv and build artifacts"
	rm -rf $(DOCS_VENV_DIR) $(BUILDDIR)

pip-freeze: .ensure-venv
	$(PIP) freeze
