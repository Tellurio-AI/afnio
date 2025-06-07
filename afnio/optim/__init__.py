# afnio/optim/__init__.py

from .tgd import TGD, tgd

__all__ = [
    "TGD",
    "tgd",
]

# Please keep this list sorted
assert __all__ == sorted(__all__)
