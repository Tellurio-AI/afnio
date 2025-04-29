# Contribution Guidelines

Welcome! We're excited to have your contributions to the Afnio project.

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
├── README.md
└── CONTRIBUTING.md
```

---

Thank you for contributing to Afnio! 💙
