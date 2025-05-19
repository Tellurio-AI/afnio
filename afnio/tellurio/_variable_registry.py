from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from afnio import Variable

# Local registry of variable_id → Variable instance
VARIABLE_REGISTRY: Dict[str, "Variable"] = {}


def register_variable(var: "Variable"):
    """
    Register a Variable instance in the local registry.

    Args:
        var (Variable): The Variable instance to register.
    """
    if var.variable_id:
        VARIABLE_REGISTRY[var.variable_id] = var


def get_variable(variable_id: str) -> "Variable":
    """
    Retrieve a Variable instance from the registry by its variable_id.

    Args:
        variable_id (str): The unique identifier of the Variable.

    Returns:
        Variable or None: The Variable instance if found, else None.
    """
    return VARIABLE_REGISTRY.get(variable_id)


# TODO: Handle all attributes accordingly
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
    if not var:
        return
    if field == "_grad":
        var._grad = [Variable(**g) for g in value]
    else:
        setattr(var, field, value)
