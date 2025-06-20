# Define the global default active Run instances

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from afnio.tellurio.run import Run

_active_run = None


def set_active_run(run: "Run"):
    """
    Sets the active run globally.

    Args:
        run (Run): The Run instance to set as active.
    """
    global _active_run
    _active_run = run


def get_active_run() -> "Run":
    """
    Gets the active run.
    If no active run is set, it raises an exception.

    Returns:
        Run: The currently active Run instance.
    """
    global _active_run
    if _active_run is None:
        raise ValueError("No active run is set.")
    return _active_run
