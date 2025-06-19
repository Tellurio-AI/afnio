# Define the global default active Run instances
_active_run_uuid = None


def set_active_run_uuid(uuid: str):
    """
    Sets the active run UUID globally.

    Args:
        uuid (str): The UUID of the active run.
    """
    global _active_run_uuid
    _active_run_uuid = uuid


def get_active_run_uuid() -> str:
    """
    Gets the active run UUID.
    If no active run UUID is set, it raises an exception.

    Returns:
        str: The UUID of the active run.
    """
    global _active_run_uuid
    if _active_run_uuid is None:
        raise ValueError("No active run UUID is set.")
    return _active_run_uuid
