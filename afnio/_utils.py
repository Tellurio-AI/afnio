import uuid
from typing import Any, Dict, List

from afnio._variable import Variable
from afnio.models import ChatCompletionModel
from afnio.tellurio._callable_registry import register_callable

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
