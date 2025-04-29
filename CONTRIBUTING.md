# Contribution Guidelines

Welcome! We're excited to have your contributions to the Afnio project.

---

## 🚀 Getting Started

Follow these steps to set up your environment and run the tests:

### 1. Clone the Repository

```bash
git clone https://github.com/Tellurio-AI/afnio.git
cd afnio
```

### 2. Create a Virtual Environment

Set up a virtual environment to isolate dependencies:

```bash
python3.9 -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

### 3. Install Dependencies

Install the required dependencies:

```bash
pip install .
```

### 4. Configure Environment Variables (Optional)

Afnio uses environment variables to configure various settings, such as the backend URL.
By default, it uses the production backend (`https://platform.tellurio.ai`). If you need
to customize any settings (e.g., for local testing), you can copy the `.env.sample`
file to `.env`:

```bash
cp .env.sample .env
```

Then, edit the `.env` file and update the values as needed. For example, to use a local
backend:

```python
export TELLURIO_BASE_URL="http://localhost:8000"
```

> **Note**: This step is only required if you need a custom configuration. Otherwise,
> the default settings will be used.

### 5. Run the Tests

Run the test suite to ensure everything is working:

```bash
pytest tests/
```

### 6. Updating Requirements

To add or update dependencies, modify the `[project.dependencies]` section in `pyproject.toml`. For example:

```toml
dependencies = [
    "httpx>=0.28.1",
    "click>=8.1.8"
]
```

After updating `pyproject.toml`, reinstall the dependencies:

```bash
pip install .
```

---

## 🧠 Code Style Guide

### 🐍 Python

We follow [PEP 8](https://peps.python.org/pep-0008/) with a few project-specific conventions:

- **Max line length**: 88 characters
- **Imports**: Grouped in the following order, with one empty line between groups:
  1. Standard library modules
  2. Third-party packages
  3. Project-local modules
- **Docstrings**: Follow [Google-style](https://google.github.io/styleguide/pyguide.html#381-docstrings). Example:

```python
def wrap_file(path: str, field_storage: FileStorage, temporary: bool) -> BufferedFileStorage:
    """Wraps a file for buffered writing.

    Args:
        path (str): The path of the file to wrap
        field_storage (FileStorage): The :class:`FileStorage` instance to wrap
        temporary (bool): Whether or not to delete the file when the File
        instance is destructed

    Returns:
        BufferedFileStorage: A buffered writable file descriptor
    """
```

---

## 🧹 Linting & Formatting Setup

### ✅ VSCode Setup (Recommended)

Install the following extensions:

- Python:
  - [Flake8](https://marketplace.visualstudio.com/items?itemName=ms-python.flake8)
  - [Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)

If you're using the repo's `.vscode/settings.json`, all formatter and linting rules will be automatically applied on save.

### 🧪 Manual Commands

For consistency or CI, you can also run the formatters manually:

```bash
# Format Python code
black .

# Lint Python code
flake8 .
```

---

## 🗂️ Project Structure

```
/
├── afnio/                  # Afnio framework
│   └── tellurio/           # Tellurio integration
├── tests/                  # Automated testing
├── .vscode/                # VSCode config
├── .github/workflows/      # GitHub actions
├── setup.py                # Package setup script
├── README.md
└── CONTRIBUTING.md
```

---

Thank you for contributing to Afnio! 💙
