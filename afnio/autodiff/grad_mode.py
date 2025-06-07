import threading
from contextlib import contextmanager

# Thread-local flag to control gradient tracking
_grad_enabled = threading.local()
_grad_enabled.enabled = True  # By default, gradients are enabled


def is_grad_enabled() -> bool:
    """Check whether gradients are currently enabled."""
    return getattr(_grad_enabled, "enabled", True)


def set_grad_enabled(mode: bool):
    """Set the global state of gradient tracking."""
    _grad_enabled.enabled = mode


@contextmanager
def no_grad():
    """
    Context manager that disables gradient calculation. All operations within this block
    will not track gradients, making them more memory-efficient.

    Disabling gradient calculation is useful for inference, when you are sure
    that you will not call :meth:`Variable.backward()`. It will reduce memory
    consumption for computations that would otherwise have `requires_grad=True`.

    In this mode, the result of every computation will have
    `requires_grad=False`, even when the inputs have `requires_grad=True`.
    There is an exception! All factory functions, or functions that create
    a new Variable and take a requires_grad kwarg, will NOT be affected by
    this mode.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator.

    Example::
        >>> x = hf.Variable("abc", role="variable", requires_grad=True)
        >>> with hf.no_grad():
        ...     y = x + x
        >>> y.requires_grad
        False
        >>> @hf.no_grad()
        ... def doubler(x):
        ...     return x + x
        >>> z = doubler(x)
        >>> z.requires_grad
        False
        >>> @hf.no_grad
        ... def tripler(x):
        ...     return x + x + x
        >>> z = tripler(x)
        >>> z.requires_grad
        False
        >>> # factory function exception
        >>> with hf.no_grad():
        ...     a = hf.cognitive.Parameter("xyz")
        >>> a.requires_grad
        True
    """
    previous_state = is_grad_enabled()  # Store the current state
    set_grad_enabled(False)  # Disable gradients
    try:
        yield  # Execute the block
    finally:
        set_grad_enabled(previous_state)  # Restore the original state
