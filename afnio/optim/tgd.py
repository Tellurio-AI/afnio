import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

from afnio._utils import (
    MultiTurnMessages,
    _deserialize_output,
    _serialize_arg,
)
from afnio._variable import Variable
from afnio.logging_config import configure_logging
from afnio.models import ChatCompletionModel
from afnio.tellurio._eventloop import run_in_background_loop
from afnio.tellurio._variable_registry import (
    suppress_variable_notifications,
)
from afnio.tellurio.client import get_default_client

from .optimizer import Optimizer, ParamsT, _extract_variable_ids, _wait_for_variable

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


# Suppress notifications for variable changes during modules initialization
# as the WebSocket connection is not established yet
with suppress_variable_notifications():
    # TGD_MESSAGES is only a placeholder that will be replaced on the server side by the
    # actual system prompt and user instruction
    TGD_MESSAGES = [
        {
            "role": "system",
            "content": [
                Variable(
                    data="Placeholder for Textual Gradient Descent optimizer system prompt",  # noqa: E501
                    role="Textual Gradient Descent optimizer system prompt",
                )
            ],
        },
        {
            "role": "user",
            "content": [
                Variable(
                    data="Placeholder for Textual Gradient Descent optimizer user prompt",  # noqa: E501
                    role="Textual Gradient Descent optimizer user prompt",
                )
            ],
        },
    ]


class TGD(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        model_client: Optional[ChatCompletionModel],
        messages: MultiTurnMessages = TGD_MESSAGES,
        inputs: Optional[Dict[str, Union[str, Variable]]] = None,
        constraints: Optional[List[Union[str, Variable]]] = None,
        momentum: int = 0,
        **completion_args,
    ):
        """
        Textual Gradient Descent (TGD) optimizer.

        Args:
            params (ParamsT): Iterable of parameters to optimize or dicts defining
                parameter groups.
            model_client (Optional[ChatCompletionModel]): LM model client used
                for optimization.
            messages (MultiTurnMessages): Messages for multi-turn interactions. It
                typically defines the optimizer system prompt and user instruction.
                In-context examples (shots) can be added as well.
            inputs (Optional[Dict[str, Union[str, Variable]]]): Dynamic values to fill
                placeholders within message templates
            constraints (Optional[List[Union[str, Variable]]]): A list of natural
                language constraints for optimization.
            momentum (int, optional): Momentum window size. Tracks the last `momentum`
                gradients, which helps accelerate updates in the right direction and
                dampen oscillations. Defaults to 0.
            completion_args (Dict[str, Any], optional): Additional arguments to pass to
                the model client when generating text completions. Defaults to an
                empty dictionary.
        """
        # Workaround to trigger TGD_MESSAGES registration with the server
        # and store related variable_ids on the client side
        if messages is TGD_MESSAGES:
            messages = [
                {
                    "role": "system",
                    "content": [
                        Variable(
                            data="Placeholder for Textual Gradient Descent optimizer system prompt",  # noqa: E501
                            role="Textual Gradient Descent optimizer system prompt",
                        )
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        Variable(
                            data="Placeholder for Textual Gradient Descent optimizer user prompt",  # noqa: E501
                            role="Textual Gradient Descent optimizer user prompt",
                        )
                    ],
                },
            ]

        defaults = dict(
            model_client=model_client,
            messages=messages,
            inputs=inputs or {},
            constraints=constraints or [],
            momentum=momentum,
            completion_args=completion_args,
        )
        super().__init__(params, defaults)

    def step(
        self, closure: Optional[Callable] = None
    ) -> Optional[Tuple[Variable, Variable]]:
        """Performs a single optimization step.

        Args:
            closure (Optional[Callable]): A closure that reevaluates the model
                and returns the loss.

        Returns:
            Optional[Tuple[Variable, Variable]]: The loss if `closure` is provided,
                otherwise None. The loss should return a numerical or textual score and
                a textual explanation, both wrapped as `Variable` objects
        """
        loss = closure() if closure else (None, None)
        super().step()
        return loss

    def _extract_variable_ids_from_state(self, state):
        """
        Extract only the variable_ids of deepcopied parameters (i.e., those generated
        on the server) from the optimizer state.

        Args:
            state (list): The serialized optimizer state as returned by the server.

        Returns:
            Set[str]: Set of variable_ids for deepcopied parameters.
        """
        var_ids = set()
        for entry in state:
            momentum_buffer = entry.get("value", {}).get("momentum_buffer", [])
            for buf_entry in momentum_buffer:
                if (
                    isinstance(buf_entry, list)
                    and len(buf_entry) > 0
                    and isinstance(buf_entry[0], dict)
                    and "variable_id" in buf_entry[0]
                ):
                    var_ids.add(buf_entry[0]["variable_id"])
        return var_ids


# TODO: Fix error when passing str constraints
def tgd(
    params: List[Variable],
    grads: List[List[Variable]],
    momentum_buffer_list: List[Optional[List[Variable]]],
    model_client: Optional[ChatCompletionModel],
    messages: MultiTurnMessages,
    inputs: Optional[Dict[str, Union[str, Variable]]],
    constraints: Optional[List[Union[str, Variable]]],
    momentum: int,
    **completion_args,
):
    r"""Functional API that performs TGD (Textual Gradient Descent) algorithm
    computation.

    See :class:`~afnio.optim.SGD` for details.
    """
    # Set `_pending_data` for all parameters that will be optimized
    for p in params:
        p._pending_data = True
        logger.debug(f"Marked variable {p.variable_id!r} as pending for data update.")

    try:
        _, ws_client = get_default_client()

        payload = {
            "params": _serialize_arg(params),
            "grads": _serialize_arg(grads),
            "momentum_buffer_list": _serialize_arg(momentum_buffer_list),
            "model_client": _serialize_arg(model_client),
            "messages": _serialize_arg(messages),
            "inputs": _serialize_arg(inputs),
            "constraints": _serialize_arg(constraints),
            "momentum": momentum,
            "completion_args": _serialize_arg(completion_args),
        }

        response = run_in_background_loop(ws_client.call("run_optimizer_tgd", payload))
        logger.debug(f"TGD optimization request sent: {payload!r}")

        result = response.get("result", {})
        result_message = result.get("message")
        result_momentum_buffer_list = result.get("momentum_buffer_list", [])

        # Extract all variable_ids from the result_momentum_buffer_list
        # and wait for them to be registered in VARIABLE_REGISTRY
        all_var_ids = _extract_variable_ids(result_momentum_buffer_list)
        for var_id in all_var_ids:
            _wait_for_variable(var_id)

        des_momentum_buffer_list = _deserialize_output(result_momentum_buffer_list)

        # Convert [param, grads] lists to (param, grads) tuples
        for i, buffer in enumerate(des_momentum_buffer_list):
            des_momentum_buffer_list[i] = [
                tuple(pair) if isinstance(pair, list) and len(pair) == 2 else pair
                for pair in buffer
            ]

        if result_message != "Functional TGD optimization step executed successfully.":
            logger.error(
                f"Server did not return any data for functional TGD optimization: "
                f"payload={payload!r}, response={response!r}"
            )
            raise RuntimeError(
                "Failed to run functional TGD optimization: server did not return data."
            )

        # Update the momentum_buffer_list with the deserialized buffers
        momentum_buffer_list.clear()
        momentum_buffer_list.extend(des_momentum_buffer_list)

        logger.debug("Functional TGD optimization executed successfully")
    except Exception as e:
        logger.error(f"Failed to run functional TGD optimization: {e!r}")
        raise
