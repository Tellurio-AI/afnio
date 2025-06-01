from typing import Callable, Dict

CALLABLE_REGISTRY: Dict[str, Callable] = {}


def register_callable(callable_id: str, fn: Callable):
    """
    Register a callable function in the local registry.

    Args:
        callable_id (str): The unique identifier for the callable.
        fn (Callable): The function to register.
    """
    CALLABLE_REGISTRY[callable_id] = fn


def get_callable(callable_id: str) -> Callable:
    """
    Retrieve a callable function from the registry by its callable_id.

    Args:
        callable_id (str): The unique identifier for teh callable.

    Returns:
        Callable or None: The callable function if found, else None.
    """
    return CALLABLE_REGISTRY.get(callable_id)


def run_callable(data: dict):
    """
    Run a callable registered in the registry by its ID.

    Args:
        callable_id (str): The unique identifier of the callable.
        *args: Positional arguments to pass to the callable.
        **kwargs: Keyword arguments to pass to the callable.

    Returns:
        The result of the callable execution.
    """
    callable_id = data["callable_id"]
    args = data.get("args", [])
    kwargs = data.get("kwargs", {})
    fn = get_callable(callable_id)
    if fn is None:
        raise ValueError(f"Callable with ID '{callable_id}' not found in registry.")
    return fn(*args, **kwargs)
