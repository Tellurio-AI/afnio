from typing import Any, Dict, Optional, Union

from afnio._utils import MultiTurnMessages
from afnio._variable import Variable
from afnio.autodiff.lm_ops import ChatCompletion as ChatCompletionOp
from afnio.models import ChatCompletionModel

from .module import Module


class ChatCompletion(Module):
    """
    Generates a chat-based completion using a language model.

    This module leverages the `ChatCompletion` operation from
    `afnio.autodiff.lm_ops` to perform model inference. The `forward` method
    accepts a list of `messages` representing the conversation history, with optional
    dynamic `inputs` for filling placeholders within the messages. The
    `forward_model_client` is responsible for interfacing with the language model
    (e.g., GPT), while `completion_args` allows customization of generation parameters
    such as temperature, maximum tokens, and seed.

    Example:
        >>> import afnio as hf
        >>> from afnio import cognitive as cog
        >>> from afnio.models.openai import OpenAI
        >>> from afnio import set_backward_model_client
        >>> fwd_model_client = OpenAI()
        >>> fwd_model_args = {"model": "gpt-4o", "temperature": 0.7}
        >>> set_backward_model_client("openai/gpt-4o")
        >>> class Assistant(cog.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.chat = cog.ChatCompletion()
        ...     def forward(self, fwd_model, messages, inputs, **completion_args):
        ...         return self.chat(fwd_model, messages, inputs, **completion_args)
        >>> system = Variable(
        ...     "You are a helpful assistant.",
        ...     role="system instruction",
        ...     requires_grad=True
        ... )
        >>> user = Variable("Translate 'Hello' to {language}.", role="user query")
        >>> language = hf.Variable("Italian", role="language")
        >>> messages = [
        ...     {"role": "system", "content": [system]},
        ...     {"role": "user", "content": [user]},
        ... ]
        >>> model = Assistant()
        >>> response = model(
        ...     fwd_model_client,
        ...     messages,
        ...     inputs={"language": language},
        ...     **fwd_model_args
        ... )
        >>> print(response.data)
        'Ciao'
        >>> feedback = Variable("Use only capital letters.", role="feedback")
        >>> response.backward(feedback)
        >>> system.grad[0].data
        'The system instruction should enforce the use of capital letters only.'

    See Also:
        :class:`afnio.autodiff.lm_ops.ChatCompletion` for the underlying operation.
    """

    forward_model_client: Optional[ChatCompletionModel]
    messages: MultiTurnMessages
    completion_args: Dict[str, Any]

    def __init__(self):
        super().__init__()

        self.register_model("forward_model_client", None)
        self.register_chat("messages", None)
        self.register_completion_config("completion_args", None)

    def forward(
        self,
        forward_model_client: Optional[ChatCompletionModel],
        messages: MultiTurnMessages,
        inputs: Optional[Dict[str, Union[str, Variable]]] = None,
        **completion_args,
    ) -> Variable:
        self.forward_model_client = forward_model_client
        self.messages = messages
        self.completion_args = completion_args
        return ChatCompletionOp.apply(
            self.forward_model_client,
            self.messages,
            inputs,
            **self.completion_args,
        )
