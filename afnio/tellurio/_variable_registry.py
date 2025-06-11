from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, List, Optional

from afnio.tellurio._node_registry import get_node

if TYPE_CHECKING:
    from afnio import Variable


# _SUPPRESS_NOTIFICATIONS is a module-level flag that controls whether variable change
# notifications should be suppressed globally. This is used to prevent notification
# loops during server-initiated updates.
_SUPPRESS_NOTIFICATIONS = False


# VARIABLE_REGISTRY is a mapping from variable_id (str) to `afnio.Variable`` instances.
# It is used to look up and update registered Variable objects by their ID,
# typically when processing server-initiated updates.
VARIABLE_REGISTRY: Dict[str, "Variable"] = {}


# PENDING_GRAD_FN_ASSIGNMENTS is a mapping from grad_fn node_id (str) to a list of
# Variable instances that are waiting for their grad_fn node to be registered.
# When deserializing a Function's output Variabe, if its grad_fn node is not yet
# available in the node registry, the Variable is added to this mapping. After the node
# is registered, all Variables in the list for that node_id will have their grad_fn
# set accordingly.
PENDING_GRAD_FN_ASSIGNMENTS: Dict[str, list] = {}


def register_variable(var: "Variable"):
    """
    Register a Variable instance in the local registry.

    Args:
        var (Variable): The Variable instance to register.
    """
    if var.variable_id:
        VARIABLE_REGISTRY[var.variable_id] = var


def get_variable(variable_id: str) -> Optional["Variable"]:
    """
    Retrieve a Variable instance from the registry by its variable_id.

    Args:
        variable_id (str): The unique identifier of the Variable.

    Returns:
        Variable or None: The Variable instance if found, else None.
    """
    return VARIABLE_REGISTRY.get(variable_id)


def update_local_variable_field(variable_id: str, field: str, value):
    """
    Update a specific field of a registered Variable instance as a consequence of a
    notification from the server.

    This function is typically called when the server notifies the client of a change
    to a Variable's attribute. It updates the local Variable instance in the registry
    with the new value for the specified field.

    Args:
        variable_id (str): The unique identifier of the Variable.
        field (str): The field name to update.
        value: The new value to set for the field.
    """
    var = get_variable(variable_id)
    try:
        if field == "_grad":
            from afnio._variable import Variable

            var.grad = [Variable(**g) for g in value] if value else []
        elif field == "_output_nr":
            var.output_nr = value
        elif field == "_grad_fn":
            node = get_node(value)
            if node is None:
                raise ValueError(
                    f"Node with id '{value}' not found in registry "
                    f"when updating _grad_fn for variable '{variable_id}'."
                )
            var._grad_fn = node
        else:
            setattr(var, field, value)
    except RuntimeError:
        raise RuntimeError(
            f"Failed to update field '{field}' for variable with ID '{variable_id}'."
        )


def append_grad_local(variable_id: str, grad: dict):
    """
    Append a gradient to the local Variable instance identified by variable_id.

    This function is typically called in response to a server notification (e.g., via
    an RPC method) indicating that a new gradient should be appended to a Variable's
    grad list. It reconstructs the gradient Variable from the provided dictionary,
    retrieves the target Variable from the local registry, and appends the gradient
    using the Variable's append_grad() method.

    Args:
        variable_id (str): The unique identifier of the Variable to update.
        grad (dict): A dictionary representing the serialized gradient Variable.

    Raises:
        RuntimeError: If the Variable cannot be found in the registry or if appending
            the gradient fails.
    """
    from afnio._variable import Variable

    var = get_variable(variable_id)

    try:
        gradient = Variable(**grad)
        var.append_grad(gradient)
    except RuntimeError:
        raise RuntimeError(
            f"Failed to append gradient for variable with ID '{variable_id}'."
        )


def clear_pending_grad(variable_ids: Optional[List[str]] = []):
    """
    Clear the `_pending_grad` flag for specified Variable instances.

    This function is used to reset the `_pending_grad` flag for Variables that are
    waiting for their gradients to be computed during a backward pass on the server.

    Args:
        variable_ids (Optional[List[str]]): List of variable IDs to clear.
    """
    for var_id in variable_ids:
        var = get_variable(var_id)
        if var is None:
            raise RuntimeError(f"Variable with id '{var_id}' not found in registry.")
        var._pending_grad = False


def clear_pending_data(variable_ids: Optional[List[str]] = []):
    """
    Clear the `_pending_data` flag for specified Variable instances.

    This function is used to reset the `_pending_data` flag for Variables that are
    waiting for their data to be computed during an optimization step on the server.

    Args:
        variable_ids (Optional[List[str]]): List of variable IDs to clear.
    """
    for var_id in variable_ids:
        var = get_variable(var_id)
        if var is None:
            raise RuntimeError(f"Variable with id '{var_id}' not found in registry.")
        var._pending_data = False


@contextmanager
def suppress_variable_notifications():
    """
    Context manager to temporarily suppress variable change notifications.

    When this context manager is active, any attribute changes to afnio.Variable
    instances will not trigger `_on_variable_change` notifications. This is useful
    for internal/client-initiated updates where you do not want to broadcast changes
    back to the server.
    """
    global _SUPPRESS_NOTIFICATIONS
    token = _SUPPRESS_NOTIFICATIONS
    _SUPPRESS_NOTIFICATIONS = True
    try:
        yield
    finally:
        _SUPPRESS_NOTIFICATIONS = token


def is_variable_notify_suppressed():
    """
    Returns True if variable change notifications are currently suppressed.

    This function checks the suppression flag used by the context manager
    `suppress_variable_notifications()`. When True, changes to Variable
    attributes will not trigger notifications to the server.

    Returns:
        bool: True if notifications are suppressed, False otherwise.
    """
    return _SUPPRESS_NOTIFICATIONS
