from typing import Callable, Dict, List, Optional, Tuple, Union

from afnio._utils import (
    MultiTurnMessages,
)
from afnio._variable import Variable
from afnio.models import ChatCompletionModel
from afnio.tellurio._variable_registry import (
    suppress_variable_notifications,
)

from .optimizer import Optimizer, ParamsT

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
    # TODO: implement funcitonal API for TGD
    raise NotImplementedError(
        "tgd is implemented on the server. Client-side execution is not supported."
    )
