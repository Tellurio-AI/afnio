# afnio/autodiff/__init__.py
import logging
from typing import Optional

from afnio._utils import _serialize_arg
from afnio.logging_config import configure_logging
from afnio.tellurio._eventloop import run_in_background_loop
from afnio.tellurio._variable_registry import get_variable
from afnio.tellurio.client import get_default_client

from .grad_mode import is_grad_enabled, no_grad, set_grad_enabled
from .utils import (
    _VariableOrVariables,
    _VariableOrVariablesOrGradEdge,
)

__all__ = [
    "backward",
    "is_grad_enabled",
    "no_grad",
    "set_grad_enabled",
]

# Please keep this list sorted
assert __all__ == sorted(__all__)

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


def backward(
    variables: _VariableOrVariables,
    grad_variables: Optional[_VariableOrVariables] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    inputs: Optional[_VariableOrVariablesOrGradEdge] = None,
) -> None:
    r"""Computes the sum of gradients of given variables with respect to graph
    leaves.

    The graph is differentiated using the chain rule. If any of ``variables``
    are non-scalar (i.e. their data has more than one element) and require
    gradient, then the Jacobian-vector product would be computed, in this
    case the function additionally requires specifying ``grad_variables``.
    It should be a sequence of matching length, that contains the "vector"
    in the Jacobian-vector product, usually the gradient of the differentiated
    function w.r.t. corresponding variables (``None`` is an acceptable value for
    all variables that don't need gradient variables).

    This function accumulates gradients in the leaves - you might need to zero
    ``.grad`` attributes or set them to ``None`` before calling it.

    .. note::

        Using this method with ``create_graph=True`` will create a reference cycle
        between the parameter and its gradient which can cause a memory leak.
        We recommend using ``autodiff.grad`` when creating the graph to avoid this.
        If you have to use this function, make sure to reset the ``.grad`` fields of
        your parameters to ``None`` after use to break the cycle and avoid the leak.

    .. note::

        When ``inputs`` are provided, each input must be a leaf variable. If any
        input is not a leaf, a ``RuntimeError`` is raised.

    Args:
        variables (Sequence[Variables] or Variable): Variables of which the derivative
            will be computed.
        grad_variables (Sequence[Variable or None] or Variable, optional): The "vector"
            in the Jacobian-vector product, usually gradients w.r.t. each element of
            corresponding variables. None values can be specified for scalar Variables
            or ones that don't require grad. If a None value would be acceptable for all
            grad_variables, then this argument is optional.
        retain_graph (bool, optional): If ``False``, the graph used to compute the grads
            will be freed. Setting this to ``True`` retains the graph, allowing for
            additional backward calls on the same graph, useful for example for
            multi-task learning where you have multiple losses. However, retaining the
            graph is not needed in nearly all cases and can be worked around in a much
            more efficient way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Defaults to ``False``.
        inputs (Sequence[Variable] or Variable or Sequence[GradientEdge], optional):
            Inputs w.r.t. which the gradient will be accumulated into ``.grad``. All
            other Variables will be ignored. If not provided, the gradient is
            accumulated into all the leaf Variables that were used to compute
            the :attr:`variables`.
    """

    # Serialize the arguments
    serialized_variables = _serialize_arg(variables)
    serialized_grad_variables = _serialize_arg(grad_variables)
    serialized_retain_graph = _serialize_arg(retain_graph)
    serialized_create_graph = _serialize_arg(create_graph)
    serialized_inputs = _serialize_arg(inputs)

    # Send the RPC call to the server
    try:
        # Get the singleton websocket client
        _, ws_client = get_default_client()

        # Fetch all Variables which gradients will be computed during backpropagation
        # and mark them as pending for grad update
        payload = {"variables": serialized_variables}
        response_ids = run_in_background_loop(
            ws_client.call("get_backprop_ids", payload)
        )
        logger.debug(
            f"Fetched backpropagation variable IDs from the server: {response_ids!r}"
        )

        result_ids = response_ids.get("result", {})
        backprop_variable_ids = result_ids.get("variable_ids", [])
        if backprop_variable_ids:
            for var_id in backprop_variable_ids:
                var = get_variable(var_id)
                if var is not None:
                    var._pending_grad = True
                    logger.debug(
                        f"Marked variable {var_id!r} as pending for grad update."
                    )
                else:
                    logger.warning(
                        f"Variable id {var_id!r} returned for backward, "
                        "but not found in VARIABLE_REGISTRY."
                    )

        # Run backward pass
        payload = {
            "variables": serialized_variables,
            "grad_variables": serialized_grad_variables,
            "retain_graph": serialized_retain_graph,
            "create_graph": serialized_create_graph,
            "inputs": serialized_inputs,
        }
        response_bwd = run_in_background_loop(ws_client.call("run_backward", payload))
        logger.debug(
            f"Backward pass instantiated and shared with the server: {variables!r}"
        )

        result_message = response_bwd.get("result", {}).get("message")
        if result_message != "Backward pass executed successfully.":
            logger.error(
                f"Server did not return any data for backward pass: "
                f"payload={payload!r}, response={response_bwd!r}"
            )
            raise RuntimeError(
                "Failed to run backward pass: server did not return data."
            )

        logger.debug(
            f"Backward pass executed successfully with variables: {variables!r}"
        )

    except Exception as e:
        logger.error(f"Failed to share backward pass with the server: {e}")
        raise
