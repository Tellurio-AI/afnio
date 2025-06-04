import uuid
from typing import Any, Optional, Sequence, Union

from afnio._variable import Variable, _allow_grad_fn_assignment
from afnio.models import ChatCompletionModel
from afnio.tellurio._callable_registry import register_callable
from afnio.tellurio._node_registry import get_node
from afnio.tellurio._variable_registry import (
    PENDING_GRAD_FN_ASSIGNMENTS,
    register_variable,
    suppress_variable_notifications,
)

from .graph import GradientEdge

_OptionalVariable = Optional[Variable]
_VariableOrVariables = Union[Variable, Sequence[Variable]]
_VariableOrVariablesOrGradEdge = Union[
    Variable,
    Sequence[Variable],
    GradientEdge,
    Sequence[GradientEdge],
]


def serialize_arg(arg: Any) -> Any:
    """
    Recursively serialize an argument for RPC transmission.

    Handles:
    - Variable: serializes as a dict with a type tag and variable_id.
    - ChatCompletionModel: serializes as a dict with a type tag and model_id.
    - Callable: registers the callable and serializes as a dict with a type tag
      and callable_id.
    - list/tuple: recursively serializes each element.
    - dict: recursively serializes each value.
    - Primitives (str, int, float, bool, None): returned as-is.

    Callables are not currently supported and will raise if encountered.
    """
    if isinstance(arg, Variable):
        return {
            "__variable__": True,
            "variable_id": arg.variable_id,
        }
    elif isinstance(arg, ChatCompletionModel):
        return {
            "__model_client__": True,
            "model_id": arg.model_id,
        }
    elif callable(arg):
        # Register the callable and generate a unique ID
        callable_id = str(uuid.uuid4())
        register_callable(callable_id, arg)
        return {
            "__callable__": True,
            "callable_id": callable_id,
        }
    elif isinstance(arg, list):
        return [serialize_arg(a) for a in arg]
    elif isinstance(arg, tuple):
        return tuple(serialize_arg(a) for a in arg)
    elif isinstance(arg, dict):
        return {k: serialize_arg(v) for k, v in arg.items()}
    elif isinstance(arg, (str, int, float, bool)) or arg is None:
        return arg
    else:
        raise TypeError(
            f"Cannot serialize object of type {type(arg).__name__}: {arg!r}"
        )


def deserialize_output(obj: Any) -> Any:
    """
    Recursively deserialize an object returned from the server.

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
        variables = tuple(deserialize_output(a) for a in obj)
        return variables
    else:
        raise TypeError(
            f"Deserialization only supports Variable or Tuple[Variable], "
            f"but got: {type(obj)}"
        )
