import logging
from typing import Any, Tuple

from afnio._utils import _serialize_arg
from afnio.autodiff.grad_mode import is_grad_enabled
from afnio.autodiff.utils import _deserialize_fn_output
from afnio.logging_config import configure_logging
from afnio.tellurio._eventloop import run_in_background_loop
from afnio.tellurio.client import get_default_client

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


class Function:
    r"""Base class to create custom `autodiff.Function`.

    To create a custom `autodiff.Function`, subclass this class and implement
    the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
    op in the forward pass, call the class method ``apply``. Do not call
    :meth:`forward` directly.

    Example::

        >>> class Func(Function):
        >>>     @staticmethod
        >>>     def forward(ctx, x: hf.Variable):
        >>>         reverse = x.data[::-1]
        >>>         out = hf.Variable(data=reverse, role=x.role, requires_grad=True)
        >>>         ctx.save_for_backward(x, reverse, out)
        >>>         return out
        >>>
        >>>     @staticmethod
        >>>     def backward(ctx, grad_out):
        >>>         x, reverse, out = ctx.saved_variables
        >>>         grad = f"Here is the feedback for {x.role} (reversed): {grad_out.grad}"
        >>>         role = f"Feedback to {x.role}"
        >>>         x.grad = hf.Variable(data=grad, role=role)
        >>>         return x.grad
        >>>
        >>> a = hf.Variable(data="This is a string", role="Input string", requires_grad=True)
        >>> c = Func.apply(a)
    """  # noqa: E501

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            f"{self.__class__} should not be instantiated. Methods on autodiff "
            "functions are all static, so you should invoke them on the class itself. "
            "Instantiating an autodiff function is not allowed."
        )

    @staticmethod
    def forward(*args: Any, **kwargs: Any) -> Any:
        r"""Define the forward of the custom autodiff Function.

        This function is to be overridden by all subclasses.
        There are two ways to define forward:

        Usage 1 (Combined forward and ctx)::

            @staticmethod
            def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
                pass

        - It must accept a context ctx as the first argument, followed by any
          number of arguments (variables or other types).

        Usage 2 (Separate forward and ctx)::

            @staticmethod
            def forward(*args: Any, **kwargs: Any) -> Any:
                pass

            @staticmethod
            def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
                pass

        - The forward no longer accepts a ctx argument.
        - Instead, you must also override the :meth:`afnio.autodiff.Function.setup_context`
          staticmethod to handle setting up the ``ctx`` object.
          ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
          to the forward.

        The context can be used to store arbitrary data that can be then
        retrieved during the backward pass. Variables should not be stored
        directly on `ctx`. Instead, variables should be saved either with
        :func:`ctx.save_for_backward` if they are intended to be used in
        ``backward``.
        """  # noqa: E501
        raise NotImplementedError(
            "You must implement the forward function for custom autodiff.Function."
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> Any:
        r"""There are two ways to define the forward pass of an autodiff.Function.

        Either:

        1. Override forward with the signature ``forward(ctx, *args, **kwargs)``.
           ``setup_context`` is not overridden. Setting up the ctx for backward
           happens inside the ``forward``.
        2. Override forward with the signature ``forward(*args, **kwargs)`` and
           override ``setup_context``. Setting up the ctx for backward happens
           inside ``setup_context`` (as opposed to inside the ``forward``)
        """
        raise NotImplementedError("setup_context is not implemented.")

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        r"""Define a formula for differentiating the operation with backward mode
        automatic differentiation.

        This function is to be overridden by all subclasses.

        It must accept a context :attr:`ctx` as the first argument, followed by
        as many outputs as the :func:`forward` returned (None will be passed in
        for non variable outputs of the forward function),
        and it should return as many variables, as there were inputs to
        :func:`forward`. Each argument is the gradient w.r.t the given output,
        and each returned value should be the gradient w.r.t. the
        corresponding input. If an input is not a Variable or is a Variable not
        requiring grads, you can just pass None as a gradient for that input.

        The context can be used to retrieve variables saved during the forward
        pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
        of booleans representing whether each input needs gradient. E.g.,
        :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
        first input to :func:`forward` needs gradient computed w.r.t. the
        output.
        """
        raise NotImplementedError(
            "You must implement the backward for your custom autodiff.Function "
            "to use it with backward mode AD (automatic differentiation)."
        )

    @classmethod
    def apply(cls, *args, **kwargs):
        """Applies the forward function of the custom Function class.

        This method handles cases where `setup_context` is defined to set up the `ctx`
        (context) object separately or within the `forward` method itself.
        """

        # Serialize the function and arguments
        function_name = cls.__name__

        serialized_args = [_serialize_arg(a) for a in args]
        serialized_kwargs = {k: _serialize_arg(v) for k, v in kwargs.items()}

        # Send the RPC call to the server
        try:
            # Get the singleton websocket client
            _, ws_client = get_default_client()

            payload = {
                "function_name": function_name,
                "grad_enabled": is_grad_enabled(),
                "args": serialized_args,
                "kwargs": serialized_kwargs,
            }
            response = run_in_background_loop(ws_client.call("run_function", payload))
            logger.debug(f"Function instantiated and shared with the server: {cls!r}")

            # Deserialize the result
            result_data = response.get("result", {}).get("data")
            if not result_data:
                logger.error(
                    f"Server did not return any data for Function.apply pass: "
                    f"payload={payload!r}, response={response!r}"
                )
                raise RuntimeError(
                    "Failed to run function forward pass: server did not return data."
                )

            return _deserialize_fn_output(result_data)

        except Exception as e:
            logger.error(f"Failed to share function with the server: {e}")
            raise
