from typing import Any, Sequence, Union

from afnio._variable import Variable, _allow_grad_fn_assignment
from afnio.tellurio._node_registry import get_node
from afnio.tellurio._variable_registry import (
    PENDING_GRAD_FN_ASSIGNMENTS,
    register_variable,
    suppress_variable_notifications,
)

from .graph import GradientEdge

_VariableOrVariables = Union[Variable, Sequence[Variable]]
_VariableOrVariablesOrGradEdge = Union[
    Variable,
    Sequence[Variable],
    GradientEdge,
    Sequence[GradientEdge],
]


def _deserialize_fn_output(obj: Any) -> Any:
    """
    Recursively deserialize a `Function.forward` response object
    returned from the server.

    Handles:
    - Variable: dict with variable_id and data, creates and registers a Variable.
    - List: deserializes each element and returns a tuple of Variables.
    - Only supports Variable or tuple/list of Variables as output.

    Raises:
        TypeError: If the object is not a Variable or a list/tuple of Variables.
    """

    if isinstance(obj, dict) and "variable_id" in obj and "data" in obj:
        with suppress_variable_notifications():
            var = Variable(
                data=obj["data"], role=obj["role"], requires_grad=obj["requires_grad"]
            )
            var._retain_grad = obj["_retain_grad"]
            var.grad = obj["_grad"]
            var.output_nr = obj["_output_nr"]

            # Assign grad_fun if the Node is already registered,
            # otherwise register for later
            grad_fn_node = get_node(obj["_grad_fn"])
            if grad_fn_node is not None:
                with _allow_grad_fn_assignment():
                    var.grad_fn = grad_fn_node
                var._pending_grad_fn_id = None
            else:
                # Register for later assignment
                var._pending_grad_fn_id = obj["_grad_fn"]
                PENDING_GRAD_FN_ASSIGNMENTS.setdefault(obj["_grad_fn"], []).append(var)

            var.is_leaf = obj["is_leaf"]

        # When Variable is created on the server
        # we must handle local Variable registration manually
        var.variable_id = obj["variable_id"]
        var._initialized = True
        register_variable(var)
        return var
    elif isinstance(obj, list):
        variables = tuple(_deserialize_fn_output(a) for a in obj)
        return variables
    else:
        raise TypeError(
            f"Deserialization only supports Variable or Tuple[Variable], "
            f"but got: {type(obj)}"
        )
