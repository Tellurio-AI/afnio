from typing import Dict, List, Optional, Tuple, Union

from afnio._utils import (
    MultiTurnMessages,
)
from afnio._variable import Variable
from afnio.autodiff.function import Function
from afnio.models import ChatCompletionModel


class ChatCompletion(Function):
    r"""
    Implements a chat completion operation using the specified lanuage model within
    the `afnio` framework, supporting automatic differentiation.

    This class inherits from `Function` and requires both the `forward` and
    `backward` methods to be defined.

    ### Features:
    - **Mini-Batching**: Processes multiple input dictionaries simultaneously to improve
        throughput.
    - **Asynchronous Execution**: Both the forward and backward passes are optimized to
        run asynchronous calls for each mini-batch, reducing latency.
    - **Gradient Computation**: Supports automatic differentiation for all `Variables`
        in `messages` and `inputs` arguments, maintaining the order of gradients.

    The `ChatCompletion` function generates a `Variable` responses by passing a
    composite prompt, built from a list of `messages` and optional `inputs`, to the
    `forward_model_client`. Each message is a dictionary with a 'role' (e.g., 'system',
    'user') and a list of `Variable` objects as 'content'. `inputs` is a dictionary
    containing strings, list of strings or `Variable`s providing dynamic values to fill
    placeholders within message templates. If `inputs` contain lists of strings or
    `Variable`s which `.data` field is a list, the response's `.data` field will be a
    list, corresponding to the batched results. Otherwise, the `.data` field will be a
    scalar string. Additional behavior, such as temperature or token limits, can be
    customized through `completion_args`.

    Example with scalar inputs:
        >>> system = Variable(
        ...     "You are a helpful assistant.",
        ...     role="system instruction",
        ...     requires_grad=True
        ... )
        >>> user = Variable("Translate 'Hello' to {language}.", role="user query")
        >>> messages = [
        ...     {"role": "system", "content": [system]},
        ...     {"role": "user", "content": [user]},
        ... ]
        >>> inputs = {"language": Variable("Italian", role="language")}
        >>> response = ChatCompletion.apply(
        ...     model_client,
        ...     messages,
        ...     inputs=inputs,
        ...     temperature=0.7
        ... )
        >>> print(response.data)
        'Ciao'
        'Hola'
        >>> feedback = Variable("Use only capital letters.", role="feedback")
        >>> response.backward(feedback)
        >>> system.grad[0].data
        'The system instruction should enforce the use of capital letters only.'

    Example with batched inputs:
        >>> system = Variable(
        ...     "You are a helpful assistant.",
        ...     role="system instruction",
        ...     requires_grad=True
        ... )
        >>> user = Variable("Translate 'Hello' to {language}.", role="user query")
        >>> messages = [
        ...     {"role": "system", "content": [system]},
        ...     {"role": "user", "content": [user]},
        ... ]
        >>> inputs = {
        ...     "language": [
        ...         Variable("Italian", role="language"),
        ...         Variable("Spanish", role="language")
        ...     ]
        ... }
        >>> response = ChatCompletion.apply(
        ...     model_client,
        ...     messages,
        ...     inputs=inputs,
        ...     temperature=0.7
        ... )
        >>> print(response.data)
        ['Ciao', 'Hola']
    """

    @staticmethod
    def forward(
        ctx,
        forward_model_client: Optional[ChatCompletionModel],
        messages: MultiTurnMessages,
        inputs: Optional[Dict[str, Union[str, List[str], Variable]]] = None,
        **completion_args,
    ) -> Variable:
        raise NotImplementedError(
            "ChatCompletion.forward is implemented on the server. "
            "Client-side execution is not supported."
        )

    @staticmethod
    def backward(ctx, grad_output: Variable) -> Tuple[Optional[Variable], ...]:
        raise NotImplementedError(
            "ChatCompletion.backward is implemented on the server. "
            "Client-side execution is not supported."
        )
