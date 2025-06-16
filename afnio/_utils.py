import uuid
from typing import Any, Dict, List

from afnio._variable import Variable
from afnio.models import ChatCompletionModel
from afnio.models.model import BaseModel
from afnio.tellurio._callable_registry import register_callable
from afnio.tellurio._model_registry import get_model
from afnio.tellurio._variable_registry import get_variable

MultiTurnMessages = List[Dict[str, List[Variable]]]


def _serialize_arg(arg: Any) -> Any:
    """
    Recursively serialize an argument for RPC transmission.

    Handles:
    - Parameter: serializes as a dict with a type tag and variable_id.
    - Variable: serializes as a dict with a type tag and variable_id.
    - ChatCompletionModel: serializes as a dict with a type tag and model_id.
    - Callable: registers the callable and serializes as a dict with a type tag
      and callable_id.
    - list/tuple: recursively serializes each element.
    - dict: recursively serializes each value.
    - Primitives (str, int, float, bool, None): returned as-is.

    Callables are not currently supported and will raise if encountered.
    """
    from afnio.cognitive.parameter import Parameter

    if isinstance(arg, Parameter):
        return {
            "__parameter__": True,
            "variable_id": arg.variable_id,
        }
    elif isinstance(arg, Variable):
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
        return [_serialize_arg(a) for a in arg]
    elif isinstance(arg, tuple):
        return tuple(_serialize_arg(a) for a in arg)
    elif isinstance(arg, dict):
        return {k: _serialize_arg(v) for k, v in arg.items()}
    elif isinstance(arg, (str, int, float, bool)) or arg is None:
        return arg
    else:
        raise TypeError(
            f"Cannot serialize object of type {type(arg).__name__}: {arg!r}"
        )


def _deserialize_output(obj: Any) -> Any:
    """
    Recursively deserialize objects received from the client.

    This function converts serialized representations of Variables, Parameters and model
    clients (as sent by the server) back into their corresponding client-side objects
    using the session registries. It handles lists, tuples, and dictionaries
    recursively, and leaves primitive types unchanged.

    Args:
        obj: The serialized argument to deserialize. Can be a dict, list, tuple,
            or primitive.

    Returns:
        The deserialized object, with Variables, Parameters amd LM models resolved from
        the session registries.

    Raises:
        ValueError: If a referenced variable or model cannot be found in the context.
        TypeError: If the input type is not supported.
    """
    from afnio.cognitive.parameter import Parameter

    if isinstance(obj, dict):
        if obj.get("__variable__") and "variable_id" in obj:
            variable_id = obj["variable_id"]
            try:
                variable = get_variable(variable_id)
                if not variable and not isinstance(variable, Variable):
                    raise ValueError(
                        f"Variable with variable_id {variable_id!r} not found"
                    )
            except KeyError:
                raise ValueError(f"Unknown variable_id: {variable_id!r}")
            return variable
        elif obj.get("__parameter__") and "variable_id" in obj:
            variable_id = obj["variable_id"]
            try:
                parameter = get_variable(variable_id)
                if not parameter and not isinstance(parameter, Parameter):
                    raise ValueError(
                        f"Parameter with variable_id {variable_id!r} not found"
                    )
            except KeyError:
                raise ValueError(f"Unknown variable_id: {variable_id!r}")
            return parameter
        elif obj.get("__model_client__") and "model_id" in obj:
            model_id = obj["model_id"]
            try:
                model = get_model(model_id)
                if not model and not isinstance(model, BaseModel):
                    raise ValueError(f"Model with model_id {model_id!r} not found")
            except KeyError:
                raise ValueError(f"Unknown model_id: {model_id!r}")
            return model
        else:
            return {k: _deserialize_output(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deserialize_output(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_deserialize_output(v) for v in obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        raise TypeError(f"Cannot deserialize object of type {type(obj)}")
