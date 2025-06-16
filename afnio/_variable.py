import logging
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, List, Optional, Union

from afnio.logging_config import configure_logging
from afnio.tellurio._eventloop import run_in_background_loop
from afnio.tellurio._variable_registry import (
    is_variable_notify_suppressed,
    register_variable,
)
from afnio.tellurio.client import get_default_client

# Import `Node` only for type hints to avoid runtime circular imports; `TYPE_CHECKING`
# ensures it's available for static analysis (e.g., mypy) without executing at runtime.
if TYPE_CHECKING:
    from afnio.autodiff.graph import Node

from copy import deepcopy

import afnio

# Thread-local flag to control the assignment of `grad_fn` attributes
_grad_fn_assignment_allowed = threading.local()
_grad_fn_assignment_allowed.value = False

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


class Variable:
    """
    A class to represent generic data, such as textual inputs, outputs, or numeric data.

    Attributes:
        data (str | int | float | List[Union[str, int, float]]): The raw data, which can
            be a single string or numeric value or a list of single string or numeric
            values.
        requires_grad (bool): Whether to track operations for automatic differentiation.
        role (str): A specific description of the role of the variable in the model.
        grad (Optional[float]): Stores the gradient of the variable, if `requires_grad`
            is set to True and backpropagation has been performed.
    """

    # Using forward references for `Variable` and `Node` defined later
    requires_grad: bool
    _grad: List["Variable"]

    # TODO: Consider having `VariableMeta` class with `.grad_fn` and `.output_nr`
    #       as attributes
    _output_nr: Optional[int]
    _grad_fn: Optional["Node"]

    _retain_grad: bool
    is_leaf: bool
    r"""All Variables that have :attr:`requires_grad` which is ``False`` will be leaf
    Variables by convention.

    For Variables that have :attr:`requires_grad` which is ``True``, they will be leaf
    Variables if they were created by the user. This means that they are not the result
    of an operation and so :attr:`grad_fn` is None.

    Only leaf Variables will have their :attr:`grad` populated during a call to
    :func:`backward`. To get :attr:`grad` populated for non-leaf Variables, you can use
    :func:`retain_grad`.

    Example::

        >>> a = hf.Variable("abc", requires_grad=True)
        >>> a.is_leaf
        True
        >>> b = hf.Variable("abc", requires_grad=True).upper()
        >>> b.is_leaf
        False
        # b was created by the operation that converts all string characters to uppercase
        >>> c = hf.Variable("abc", requires_grad=True) + "def"
        >>> c.is_leaf
        False
        # c was created by the addition operation
        >>> d = hf.Variable("abc").upper()
        >>> d.is_leaf
        True
        # d does not require gradients and so has no operation creating it (that is tracked by the autodiff engine)
        >>> e = hf.Variable("abc").upper().requires_grad_()
        >>> e.is_leaf
        True
        # e requires gradients and has no operations creating it
    """  # noqa: E501
    variable_id: Optional[str]
    _initialized: bool
    _pending_grad_fn_id: Optional[str]
    _pending_grad: Optional[bool]
    _pending_data: Optional[bool]

    def __init__(
        self,
        data: Optional[Union[str, int, float, List[Union[str, int, float]]]] = "",
        role: str = "",
        requires_grad: bool = False,
    ):
        if not isinstance(data, (str, int, float, list, tuple)):
            raise TypeError(
                "`data` must be a single value (str, int, float) or a list/tuple of "
                "such values."
            )

        if isinstance(data, (list, tuple)):
            # Check if the list/tuple is homogeneous (all strings or all numbers)
            all_strings = all(isinstance(d, str) for d in data)
            all_numbers = all(isinstance(d, (int, float)) for d in data)

            if not (all_strings or all_numbers):
                raise TypeError(
                    f"When `data` is a {type(data).__name__}, it must be either "
                    f"all strings or all numbers (int, float)."
                )

            if all_numbers:
                # Check for mixed int and float types
                contains_int = any(isinstance(d, int) for d in data)
                contains_float = any(isinstance(d, float) for d in data)

                if contains_int and contains_float:
                    data = [float(d) for d in data]

            if isinstance(data, tuple):
                data = list(data)

        # Websocket attributes
        self.variable_id = None
        self._initialized = False  # Falgs variable is ready to send websocket updates
        self._pending_grad_fn_id = None  # Flags grad_fn is being set (fwd pass running)
        self._pending_grad = False  # Flags grad is being set (bwd pass running)
        self._pending_data = False  # Flags data is being set (optim step running)
        # Internal attributes
        self._data = data
        self.role = role
        self.requires_grad = requires_grad
        self._retain_grad = False
        self._grad = []
        self._output_nr = 0
        self._grad_fn = None
        self.is_leaf = not requires_grad or self.grad_fn is None

        # Share the variable with the websocket server
        if not is_variable_notify_suppressed():
            try:
                from afnio.cognitive.parameter import Parameter

                # Get the singleton websocket client
                _, ws_client = get_default_client()

                payload = {
                    "data": self.data,
                    "role": self.role,
                    "requires_grad": self.requires_grad,
                    "obj_type": (
                        "__parameter__"
                        if isinstance(self, Parameter)
                        else "__variable__"
                    ),
                }
                response = run_in_background_loop(
                    ws_client.call("create_variable", payload)
                )
                logger.debug(f"Variable created and shared with the server: {self!r}")
                variable_id = response.get("result", {}).get("variable_id")
                if not variable_id:
                    logger.error(
                        f"Server did not return a variable_id "
                        f"for payload: {payload!r}, response: {response!r}"
                    )
                    raise RuntimeError(
                        "Failed to create Variable: server did not return a variable_id."  # noqa: E501
                    )
                self.variable_id = variable_id
                self._initialized = True
                register_variable(self)
            except Exception as e:
                logger.error(f"Failed to share Variable with the server: {e}")
                raise

    # TODO: pretty print data lists
    def __repr__(self):
        if self._grad_fn:
            return f"Variable(data={self.data}, role={self.role}, grad_fn={self._grad_fn.name()})"  # noqa: E501
        return f"Variable(data={self.data}, role={self.role}, requires_grad={self.requires_grad})"  # noqa: E501

    # TODO: pretty print data lists
    def __str__(self):

        # Helper function to truncate a string if it's longer than 40 characters
        def truncate_str(s):
            if isinstance(s, (int, float)):
                return str(s)
            if len(s) > 40:
                return f"{s[:20]}...{s[-20:]}"
            return s

        # Helper function to show the first and last three elements if it is long
        def format_list(data_list):
            if len(data_list) > 6:
                truncated = [
                    truncate_str(d) for d in (data_list[:3] + ["..."] + data_list[-3:])
                ]
                return f"[{', '.join(truncated)}]"
            return f"[{', '.join(truncate_str(d) for d in data_list)}]"

        if isinstance(self.data, list):
            data_repr = format_list(self.data)
        else:
            data_repr = truncate_str(self.data)

        if self._grad_fn:
            return f"variable({data_repr}, role={truncate_str(self.role)}, grad_fn={self._grad_fn.name()})"  # noqa: E501
        return f"variable({data_repr}, role={truncate_str(self.role)}, requires_grad={self.requires_grad})"  # noqa: E501

    def __add__(self, other) -> "Variable":
        if not isinstance(other, Variable):
            raise TypeError("Only Variables can be added to each other.")

        from afnio.autodiff.basic_ops import Add

        return Add.apply(self, other)

    def __iadd__(self, other) -> "Variable":
        if not isinstance(other, Variable):
            raise TypeError("Only Variables can be added to each other.")

        from afnio.autodiff.basic_ops import Add

        result = Add.apply(self, other)

        self.data = result.data
        self.role = result.role
        self.requires_grad = result.requires_grad

        # Update the grad function in case `other` also has `requires_grad`
        if result.requires_grad:
            with _allow_grad_fn_assignment():
                self.grad_fn = result.grad_fn

        return self

    def backward(
        self, gradient=None, retain_graph=None, create_graph=False, inputs=None
    ) -> None:
        r"""Computes the gradient of current variable wrt graph leaves.

        The graph is differentiated using the chain rule. If the variable is non-scalar
        (i.e. its data has more than one element) and requires gradient, the function
        additionally requires specifying a ``gradient``. It should be a variable with
        data of matching type and shape, that represents the gradient of the
        differentiated function w.r.t. ``self``.

        This function accumulates gradients in the leaves - you might need to zero
        ``.grad`` attributes or set them to ``None`` before calling it.

        .. note::

            When ``inputs`` are provided, each input must be a leaf variable. If any
            input is not a leaf, a ``RuntimeError`` is raised.

        Args:
            gradient (Variable, optional): The gradient of the function
                being differentiated w.r.t. ``self``.
                This argument can be omitted if ``self`` is a scalar.
            retain_graph (bool, optional): If ``False``, the graph used to compute
                the grads will be freed. Setting this to ``True`` retains the graph,
                allowing for additional backward calls on the same graph, useful for
                example for multi-task learning where you have multiple losses.
                However, retaining the graph is not needed in nearly all cases
                and can be worked around in a much more
                efficient way. Defaults to the value of ``create_graph``.
            create_graph (bool, optional): If ``True``, graph of the derivative will
                be constructed, allowing to compute higher order derivative
                products. Defaults to ``False``.
            inputs (sequence of Variable, optional): Inputs w.r.t. which the gradient
                will be accumulated into ``.grad``. All other variables will be ignored.
                If not provided, the gradient is accumulated into all the leaf Variables
                that were used to compute the :attr:`variables`.
        """

        if self.is_leaf:
            raise RuntimeError(
                "Variable does not require grad or does not have a grad_fn."
            )

        afnio.autodiff.backward(
            self, gradient, retain_graph, create_graph, inputs=inputs
        )

    def requires_grad_(self, mode: bool = True) -> "Variable":
        r"""
        requires_grad_(requires_grad=True) -> Variable

        Change if autodiff should record operations on this variable: sets this
        variable's :attr:`requires_grad` attribute in-place. Returns this variable.

        :func:`requires_grad_`'s main use case is to tell autodiff to begin recording
        operations on a Variable ``variable``. If ``variable`` has
        ``requires_grad=False`` (because it was obtained through a DataLoader, or
        required preprocessing or initialization), ``variable.requires_grad_()`` makes
        it so that autodiff will begin to record operations on ``variable``.

        Args:
            requires_grad (bool): If autodiff should record operations on this variable.
                Default: ``True``.

        Example:

            >>> # Initialize with requires_grad=False for data preprocessing
            >>> x = hf.Variable(data="abc", role="input")
            >>> x = preprocess(x)  # Preprocess without gradient tracking
            >>> x
            variable(abc, role=input, requires_grad=False)

            >>> # Now enable requires_grad for backpropagation
            >>> x.requires_grad_()
            >>> output = model(x)
            >>> output.backward()  # Backpropagation through `x`
            >>> x.grad
            variable(ABC, role=input, requires_grad=True)
        """
        self.requires_grad = mode
        self.is_leaf = not self.requires_grad or self.grad_fn is None
        return self

    @property
    def data(self):
        self._wait_for_pending(
            "_pending_data"
        )  # Wait until the pending flag is cleared
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def output_nr(self) -> int:
        return self._output_nr

    @output_nr.setter
    def output_nr(self, n: int):
        if not isinstance(n, int) or not (n >= 0):
            raise TypeError(
                f"`output_nr` can only be an int greater or equal to 0, "
                f"but {n} is of type {type(n).__name__}"
            )
        self._output_nr = n

    @property
    def grad_fn(self) -> Optional["Node"]:
        self._wait_for_pending(
            "_pending_grad_fn_id"
        )  # Wait until the pending flag is cleared
        return self._grad_fn

    @grad_fn.setter
    def grad_fn(self, fn: Callable):
        """
        Sets the ``grad_fn`` that will be called by the engine to produce the actual
        gradient for this variable.
        """
        if not getattr(_grad_fn_assignment_allowed, "value", False):
            raise AttributeError(
                "Direct assignment to `grad_fn` is not allowed. "
                "Use Function.apply() to construct Variables with a grad_fn."
            )
        if not self.requires_grad:
            raise RuntimeError(
                "Cannot set `grad_fn` on a variable that does not require gradients. "
                "To enable gradient tracking for this variable, call "
                "`.requires_grad_()` before setting `grad_fn`. Only variables with "
                "`requires_grad=True` can have a gradient function (`grad_fn`)."
            )
        self._grad_fn = fn
        self.is_leaf = not self.requires_grad or self.grad_fn is None

    @property
    def grad(self) -> Optional["Variable"]:
        self._wait_for_pending(
            "_pending_grad"
        )  # Wait until the pending flag is cleared
        if self.is_leaf or self._retain_grad:
            return self._grad
        else:
            # Throwing a `UserWarning`` instead of `RuntimeError` could do here, like
            # in Pytorch, but for now I cannot think of any use case for not throwing
            # the error
            raise RuntimeError(
                "Attempted to access .grad for a non-leaf Variable without retain_grad "
                "enabled. Non-leaf Variables do not have their gradients retained by "
                "default in autodiff. To retain gradients for this Variable, call "
                "``.retain_grad()`` before performing the backward pass."
            )

    @grad.setter
    def grad(self, gradient: List["Variable"]):
        """
        Sets the ``.grad`` for this variable if it is a leaf or has ``.retain_grad``
        enabled.
        """
        if not isinstance(gradient, list) or not all(
            isinstance(g, Variable) for g in gradient
        ):
            raise TypeError(
                f"`.grad` expects a list of Variables for the gradient to accumulate, "
                f"but got {type(gradient).__name__}."
            )

        if self.is_leaf or self._retain_grad:
            self._grad = gradient
        else:
            # Throwing a `UserWarning`` instead of `RuntimeError` could do here, like
            # in Pytorch, but for now I cannot think of any use case for not throwing
            # the error
            raise RuntimeError(
                "Attempted to set .grad for a non-leaf Variable without retain_grad "
                "enabled. Non-leaf Variables do not have their gradients retained by "
                "default in autodiff. To retain gradients for this Variable, call "
                "``.retain_grad()`` before performing the backward pass."
            )

    def append_grad(self, gradient: "Variable"):
        """
        Appends a gradient value to the list ``.grad`` for this variable.
        """
        if self.is_leaf or self._retain_grad:
            self._on_append_grad(gradient)
            self._grad.append(gradient)
        else:
            # Throwing a `UserWarning`` instead of `RuntimeError` could do here, like
            # in Pytorch, but for now I cannot think of any use case for not throwing
            # the error
            raise RuntimeError(
                "Attempted to append to .grad for a non-leaf Variable without "
                "retain_grad enabled. Non-leaf Variables do not have their gradients "
                "retained by default in autodiff. To retain gradients for this "
                "Variable, call ``.retain_grad()`` before performing the backward pass."
            )

    def retain_grad(self):
        """Enable gradient retention for non-leaf variables."""
        if not self.is_leaf:
            self._retain_grad = True
        else:
            raise RuntimeError("Cannot call retain_grad on a leaf variable")

    def detach(self) -> "Variable":
        """
        Returns a new Variable, detached from the computation graph.
        This new Variable will not have a `grad_fn` and will not track gradients.
        """
        return Variable(self.data, role=self.role, requires_grad=False)

    # def clone(self):
    #     """
    #     Create a copy of this Variable, preserving the data.
    #     """
    #     return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        if not self.is_leaf:
            raise RuntimeError(
                "Only Variables created explicitly by the user "
                "(graph leaves) support the deepcopy protocol at the moment."
            )
        if id(self) in memo:
            return memo[id(self)]

        with afnio.no_grad():
            new_variable = Variable(
                data=deepcopy(self.data, memo),
                role=self.role,
                requires_grad=self.requires_grad,
            )

            new_variable._retain_grad = self._retain_grad
            new_variable._output_nr = self._output_nr

            if self.grad_fn:
                with _allow_grad_fn_assignment():
                    new_variable.grad_fn = deepcopy(
                        self.grad_fn, memo
                    )  # Also sets `.is_leaf`
            if self.grad != []:
                new_variable.grad = deepcopy(self.grad, memo)

            new_variable.__dict__ = deepcopy(self.__dict__, memo)

            memo[id(self)] = new_variable

            return new_variable

    def copy_(self, src: "Variable") -> "Variable":
        """
        Copies the data from the source Variable into this Variable.

        Args:
            src (Variable): The source Variable to copy from.

        Returns:
            self: The current Variable with updated data, role and requires_grad.

        Raises:
            TypeError: If the source is not a Variable.
            ValueError: If the source data type does not match the target data type.
        """
        if not is_variable(src):
            raise TypeError(
                f"Expected `src` to be a Variable, but got {type(src).__name__}."
            )

        is_scalar_self = is_scalar_variable(self)
        is_scalar_src = is_scalar_variable(src)

        if is_scalar_self and is_scalar_src:
            self.data = src.data
        elif not is_scalar_self and not is_scalar_src:
            if len(self.data) != len(src.data):
                raise ValueError(
                    f"Cannot copy list `.data` fields of different lengths: "
                    f"{len(self.data)} vs {len(src.data)}."
                )
            self.data = src.data.copy()
        else:
            raise ValueError(
                f"Cannot copy data from {type(src.data).__name__} "
                f"to {type(self.data).__name__}."
            )

        self.role = src.role
        self.requires_grad = src.requires_grad
        return self

    def is_floating_point(self) -> bool:
        """
        Checks if the Variable's data contains floating-point values.

        Returns:
            bool: True if the data is a floating-point type (either scalar or
                all elements in a list/tuple are floating-point).
        """
        if isinstance(self.data, float):
            return True

        if isinstance(self.data, (list, tuple)) and self.data:
            return all(isinstance(d, float) for d in self.data)

        return False

    def to(self, dtype=None) -> "Variable":
        """
        Cast the data of the Variable to the specified dtype.

        Args:
            dtype (Optional[type]): The target type to cast the data
                (e.g., float, int, str).

        Returns:
            Variable: A new Variable with data cast to the target dtype.
        """
        if dtype is not None:
            if not is_scalar_variable(self):
                # Cast each element in the list to the target dtype
                new_data = [dtype(d) for d in self.data]
            else:
                # Cast scalar data to the target dtype
                new_data = dtype(self.data)
        else:
            # No dtype casting
            new_data = self.data

        # Return a new Variable with the same role and requires_grad, but updated data
        return Variable(data=new_data, role=self.role, requires_grad=self.requires_grad)

    def _on_variable_change(self, field: str, value):
        """
        Notify the server of a change in the variable's attributes.
        This method is called whenever an attribute of the variable is set.
        It sends a notification to the server with the updated field and value.

        Args:
            field (str): The name of the field that changed.
            value: The new value of the field.

        Raises:
            RuntimeError: If the variable is not registered with the server or if the
                server response does not match the request.
            TypeError: If the provided value is of an unexpected type for the field.
        """
        from afnio._utils import _serialize_arg

        if is_variable_notify_suppressed():
            return  # Do not notify server

        if self.variable_id is None:
            logger.error(
                f"Cannot notify server: "
                f"variable_id=None, field='{field}', value={value!r}"
            )
            raise RuntimeError("Cannot notify server: variable_id is None.")

        if field in {
            "output_nr",
            "grad_fn",
            "grad",
            "_initialized",
            "_pending_grad_fn_id",
            "_pending_grad",
            "_pending_data",
        }:
            # Do not notify for the property setter, as we already notify
            # for all the changes made inside the property setter.
            # Also do not notify for `_initialized` and pending states
            return
        elif field == "_data":
            field = "data"  # `data` is a property only on the client
            end_value = value
        elif field == "_grad":
            if not isinstance(value, list):
                raise TypeError(
                    f"Expected `value` to be a list for field '{field}', "
                    f"but got {type(value).__name__}."
                )
            end_value = [_serialize_arg(g) for g in value]
        elif field == "_grad_fn":
            raise RuntimeError(
                "Setting `grad_fn` is only allowed on the server by the autodiff "
                "engine. Do not use `_allow_grad_fn_assignment()` on the client."
            )
        else:
            end_value = value

        payload = {
            "variable_id": self.variable_id,
            "field": field,
            "value": end_value,
        }

        try:
            _, ws_client = get_default_client()
            response = run_in_background_loop(
                ws_client.call("update_variable", payload)
            )
            # Check server response
            if (
                response["result"]["variable_id"] != self.variable_id
                or response["result"]["field"] != field
                or response["result"]["value"] != end_value
            ):
                logger.error(
                    f"Server response mismatch: {response['result']!r} "
                    f"(expected variable_id={self.variable_id!r}, field={field!r}, value={end_value!r})"  # noqa: E501
                )
                raise RuntimeError(
                    "Failed to notify server of variable change: "
                    "server response does not match the update sent."
                )
            logger.debug(
                f"Variable change notified to server and confirmed: "
                f"variable_id={self.variable_id!r}, field='{field}', value={end_value!r}"  # noqa: E501
            )

        except Exception:
            logger.exception(
                f"Failed to notify server of variable change: "
                f"variable_id={self.variable_id!r}, field='{field}', value={end_value!r}"  # noqa: E501
            )
            raise

    def _on_append_grad(self, gradient: "Variable"):
        """
        Notify the server that a new gradient has been appended to this variable.

        This method is called before the gradient is added to the local `.grad` list.
        It sends an 'append_grad' RPC request to the server, including the variable's
        ID and the serialized gradient. The method blocks until the server acknowledges
        the append operation, ensuring synchronization between client and server.

        Args:
            gradient (Variable): The gradient variable to append.

        Raises:
            RuntimeError: If the variable is not registered with the server or if the
                server response does not match the request.
            TypeError: If the provided gradient is not a Variable.
        """
        from afnio._utils import _serialize_arg

        if is_variable_notify_suppressed():
            return  # Do not notify server

        if self.variable_id is None:
            logger.error(
                f"Cannot notify server: variable_id=None, gradient={gradient!r}"
            )
            raise RuntimeError("Cannot notify server: variable_id is None.")

        if not isinstance(gradient, Variable):
            raise TypeError(
                f"Expected `value` to be a Variable, but got {type(gradient).__name__}."
            )

        ser_grad = _serialize_arg(gradient)

        payload = {
            "variable_id": self.variable_id,
            "gradient": ser_grad,
        }

        try:
            _, ws_client = get_default_client()
            response = run_in_background_loop(ws_client.call("append_grad", payload))
            # Check server response
            if (
                response["result"]["variable_id"] != self.variable_id
                or response["result"]["gradient_id"] != gradient.variable_id
            ):
                logger.error(
                    f"Server response mismatch: {response['result']!r} "
                    f"(expected variable_id={self.variable_id!r}, gradient={ser_grad!r}"
                )
                raise RuntimeError(
                    "Failed to notify server of gradient append: "
                    "server response does not match the update sent."
                )
            logger.debug(
                f"Gradient append notified to server and confirmed: "
                f"variable_id={self.variable_id!r}, gradient={ser_grad!r}"
            )

        except Exception:
            logger.exception(
                f"Failed to notify server of gradient append: "
                f"variable_id={self.variable_id!r}, gradient={ser_grad!r}"
            )
            raise

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if getattr(self, "_initialized", False):
            self._on_variable_change(name, value)
        # TODO: Should we handle the else condition and throw an error?

    def _wait_for_pending(
        self, attr_name: str, timeout: float = 3, interval: float = 0.01
    ) -> None:
        """
        Wait until the attribute specified by `attr_name` is no longer truthy.
        Uses time.monotonic() for more reliable timeout measurement.

        Args:
            attr_name (str): Name of the attribute to wait on.
            timeout (float): Maximum time to wait, in seconds.
            interval (float): How frequently to check the attribute, in seconds.

        Raises:
            RuntimeError: If the attribute remains truthy after the timeout.
        """
        end_time = time.monotonic() + timeout
        while getattr(self, attr_name):
            if time.monotonic() > end_time:
                raise RuntimeError(
                    f"Timeout waiting for {attr_name} to be cleared "
                    f"for variable_id={self.variable_id}"
                )
            time.sleep(interval)


def is_variable(obj):
    r"""Returns True if `obj` is an Afnio variable.

    Note that this function is simply doing ``isinstance(obj, hf.Variable)``.
    Using that ``isinstance`` check is better for typechecking with mypy,
    and more explicit - so it's recommended to use that instead of
    ``is_variable``. Use ``is_variable`` for example when importing ``Variable``
    creates circular dependencies.

    Args:
        obj (Object): Object to test
    Example::

        >>> x = hf.Variable("abc")
        >>> hf.is_variable(x)
        True

    """
    return isinstance(obj, Variable)


def is_scalar_variable(obj):
    """
    Check if an object is a Variable and its `.data` is a scalar
    (of type str, int, or float).

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is a scalar Variable, False otherwise.
    """
    if not is_variable(obj):
        return False

    data = getattr(obj, "data", None)
    return isinstance(data, (str, int, float))


@contextmanager
def _allow_grad_fn_assignment():
    """
    Context manager that allows assignment to the `grad_fn` attribute of Variables.
    This is useful for internal operations where you need to set the `grad_fn`
    directly, bypassing the usual restrictions.

    .. note::
        This context manager should only be used by the autodiff engine,
        as it allows direct manipulation of the `grad_fn` attribute, which is
        typically managed internally. Manual use is strongly discouraged.
    """
    previous_state = getattr(_grad_fn_assignment_allowed, "value", False)
    _grad_fn_assignment_allowed.value = True  # Allow grad_fn assignment
    try:
        yield  # Execute the block
    finally:
        _grad_fn_assignment_allowed.value = previous_state  # Restore the original state
