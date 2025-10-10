# Afnio Documentation

This folder contains the Sphinx project for the **Afnio** docs.

---

## Quick start

> Requires **Python ≥ 3.11** for the docs toolchain.

From the repo root:

```bash
cd docs
make html        # build once → docs/_build/html
# or
make livehtml    # live reload at http://127.0.0.1:8000
```

Open `_build/html/index.html` in your browser (or use the `livehtml` URL).

---

## Why a second Python virtual environment?

Our main development/runtime environment may be on Python **3.9** (or another version), but the modern docs stack—especially **sphinx-autobuild** (live reload) and friends—requires **Python ≥ 3.11**. Mixing them in a single venv leads to dependency and asyncio loop errors (e.g., _“Future attached to a different loop”_).

To keep the experience reliable and simple:

- The **project venv** (e.g., `.venv`) stays on your app’s Python (e.g., 3.9).
- The **docs venv** lives in `docs/.venv-docs` and uses **Python 3.12**.
- The `Makefile` auto-creates and verifies the docs venv so you don’t have to manage it manually.

This separation avoids conflicts and makes builds reproducible.

---

## One-time prerequisite

Install Python **3.12** on your machine.

- **Ubuntu/Debian**

  ```bash
  sudo apt-get update
  sudo apt-get install -y python3.12 python3.12-venv
  ```

- **macOS (Homebrew)**

  ```bash
  brew install python@3.12
  ```

- **Windows (PowerShell)**

  ```powershell
  py -3.12 -m venv .venv-docs
  ```

> You do **not** need extra tools like direnv/pyenv. The `Makefile` handles the docs venv.

---

## Makefile targets

Run these from `docs/`:

- `make html`

  - Builds the site into `_build/html`.
  - Ensures `python3.12` is present (`check-py312`).
  - Creates/repairs `docs/.venv-docs` as 3.11+ (`.ensure-venv`).
  - Installs deps from `docs/requirements.txt` (`.ensure-deps`).
  - Runs `sphinx-build`.

- `make livehtml`

  - Live rebuild with auto-refresh at `http://127.0.0.1:8000`.
  - Watches both `docs/` and `../afnio`.
  - Defaults to fast builds (notebook execution off via `nb_execution_mode=off`).

- `make clean`

  - Removes `_build/` only.

- `make nuke`

  - Removes `_build/` **and** the docs venv `docs/.venv-docs`. Use when dependencies get messy; the next `make html` recreates everything.

- `make pip-freeze`

  - Prints the exact package versions installed in the docs venv (useful for updating pins).

---

## Dependencies

We pin the docs toolchain in `requirements-docs.txt`. The Makefile installs **only** from this file. Example pins:

```
sphinx==7.4.7
pydata-sphinx-theme==0.16.1
myst-parser==3.0.1
myst-nb==1.1.1
sphinx-autodoc-typehints==2.5.0
sphinx-copybutton==0.5.2
sphinx-design==0.6.1
sphinx-autobuild==2024.10.3
```

To refresh pins from the current venv:

```bash
cd docs
make pip-freeze > requirements-docs.txt
```

---

## Importing Afnio during builds

If autodoc needs to import Afnio:

- Option A (recommended): install Afnio **editable** into the docs venv

  ```bash
  # from repo root
  docs/.venv-docs/bin/pip install -e .
  ```

- Option B: ensure `conf.py` adds the repo root to `sys.path` (already configured in our template).

---

## Speed tips

- Keep `nb_execution_mode="off"` (Makefile passes this for local builds). Enable execution only in CI if needed.
- Use `make livehtml` while writing docs; it rebuilds on save and refreshes the browser.

---

## Troubleshooting

- **“python3.12 not found”** → Install Python 3.12 (see prerequisites) and rerun.
- **Live reload ASGI crash** → Ensure you’re using the docs venv via `make livehtml` from `docs/`.
- **Weird dependency issues** → `make nuke && make html` to fully reset the docs environment.
