import inspect
import pickle
import types
import uuid
from typing import Any, Callable, Dict, List, Optional

from afnio._variable import Variable
from afnio.models import ChatCompletionModel
from afnio.models.model import BaseModel
from afnio.tellurio._callable_registry import register_callable
from afnio.tellurio._model_registry import get_model
from afnio.tellurio._variable_registry import get_variable

MultiTurnMessages = List[Dict[str, List[Variable]]]


def _validate_typed_sequence(
    input_value, item_type, allow_tuple=False, allow_single=False, input_name=None
):
    """
    Validates that the input is a list (or optionally a tuple or single instance) of a
    specified type.

    Args:
        input_value: The value to validate.
        item_type: The expected type for each element.
        allow_tuple: If True, allows a tuple instead of a list.
        allow_single: If True, allows a single instance instead of a list.
        input_name: The name of the variable being validated (used for error messages).
            If `_validate_typed_sequence()` is not nested within other functions this
            argument is not necessary as it gets automatically derived. Pass this
            argument when `_validate_typed_sequence()` is nested one or multiple times.

    Raises:
        TypeError: If the input is not a list of the specified type (or a single
            instance if allowed).
    """
    if input_name is None:
        caller_locals = inspect.currentframe().f_back.f_locals
        variable_name = [
            name for name, value in caller_locals.items() if value is input_value
        ]
        input_name = variable_name[0] if variable_name else "input"

    if isinstance(input_value, item_type) and allow_single:
        return

    valid_types = (list, tuple) if allow_tuple else (list,)
    type_names = ", ".join(t.__name__ for t in valid_types)
    if not isinstance(input_value, valid_types):
        raise TypeError(
            f"`{input_name}` expects a ({type_names}) of type {item_type.__name__}, "
            f"but got {type(input_value).__name__}."
        )
    if not all(isinstance(item, item_type) for item in input_value):
        raise TypeError(
            f"All elements in `{input_name}` must be of type {item_type.__name__}, "
            f"but got {[type(item).__name__ for item in input_value]}."
        )


def _validate_multi_turn_messages(messages: MultiTurnMessages):
    """
    Validates the structure and content of the `messages` parameter for
    multi-turn composition.

    Args:
        messages (MultiTurnMessages): A list of dictionaries, each representing a chat
            turn with a 'role' and a list of `Variable` objects in the 'content'.

    Raises:
        ValueError: If any message lacks 'role' or 'content' keys.
        TypeError: If 'role' in any message is not a string.
        TypeError: If 'content' in any message is not a list of `Variable` objects.
    """
    _validate_typed_sequence(messages, dict)

    for msg in messages:
        if "role" not in msg or "content" not in msg:
            raise ValueError(
                f"Each message dictionary must contain 'role' and 'content' keys, "
                f"but got {msg.keys()}."
            )
        if not isinstance(msg["role"], str):
            raise TypeError(
                f"'role' must be a string, but got {type(msg['role']).__name__}."
            )
        _validate_typed_sequence(msg["content"], Variable, input_name='msg["content"]')


def is_multi_turn_messages(input_value: MultiTurnMessages):
    """
    Determines whether the given input conforms to the structure of multi-turn messages.

    This function validates if the `input_value` adheres to the expected format for
    multi-turn chat messages, which should be a list of dictionaries where each
    dictionary contains a 'role' (a string) and a 'content' (a list of `Variable`
    objects).

    Args:
        input_value (MultiTurnMessages): The input value to validate.

    Returns:
        bool: Returns `True` if the input is valid multi-turn messages,
            otherwise `False`.
    """
    try:
        _validate_multi_turn_messages(input_value)
        return True
    except TypeError:
        return False


def _validate_function(func: Callable[..., Any]) -> None:
    """
    Validate whether `func` is a valid standalone function that can be safely pickled.

    Allowed functions:
    - Standalone `def` functions (global or module-level)
    - Built-in functions
    - `staticmethod`s (if extracted correctly)

    The function must:
    - Not be a lambda
    - Not be a method (bound or unbound)
    - Not be a callable object (e.g., a class instance with `__call__`)
    - Not be a nested function or closure
    - Be picklable

    Args:
        func (Callable[..., Any]): The function to validate.

    Raises:
        TypeError: If `func` is not a valid function.
        ValueError: If `func` cannot be safely pickled.
    """

    # Reject methods (bound or unbound)
    if isinstance(func, types.MethodType):
        raise TypeError(
            f"Invalid function: '{func.__name__}' is a method (bound or unbound). "
            f"Methods are tied to a class or instance, which prevents pickling. "
            f"If this function does not require 'self' or 'cls', "
            f"consider making it a @staticmethod."
        )

    # Ensure it's a function (user-defined or built-in)
    if not isinstance(func, (types.FunctionType, types.BuiltinFunctionType)):
        raise TypeError(
            f"Invalid function: '{func}'. Expected a standalone function, "
            f"but got a {type(func).__name__}."
        )

    # Reject lambdas
    if func.__name__ == "<lambda>":
        raise TypeError(
            "Invalid function: Lambda functions cannot be pickled and are not allowed."
        )

    # Reject nested functions and closures (but only check __closure__ if it exists)
    if hasattr(func, "__closure__") and func.__closure__:
        raise TypeError(
            f"Invalid function: '{func.__name__}' is a closure or a nested function. "
            f"Closures capture variables from their enclosing scope, making them "
            f"unpicklable. Consider rewriting it as a standalone function "
            f"or a callable class."
        )

    # Test picklability
    try:
        pickle.dumps(func)
    except (pickle.PickleError, AttributeError) as e:
        raise ValueError(
            f"Invalid function: '{func.__name__}' cannot be pickled. Error: {e}"
        )


def _is_valid_function(func: Optional[Callable[..., Any]]) -> bool:
    """
    Check if `func` is a valid standalone function that can be safely pickled.

    Allowed functions:
    - Standalone `def` functions (global or module-level)
    - Built-in functions
    - `staticmethod`s (if extracted correctly)

    The function must:
    - Not be a lambda
    - Not be a method (bound or unbound)
    - Not be a callable object (e.g., a class instance with `__call__`)
    - Not be a nested function or closure
    - Be picklable

    Args:
        func (Callable[..., Any] or None): The function to validate.
            `None` is considered valid.

    Returns:
        bool: True if the function is valid, False otherwise.
    """
    if func is None:
        return True  # `None` is considered valid

    try:
        _validate_function(func)
        return True
    except (TypeError, ValueError):
        return False


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
