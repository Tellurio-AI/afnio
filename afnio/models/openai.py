import copy
import json
import os
from typing import Dict, Iterable, List, Mapping, Optional, Union

import httpx
from openai import DEFAULT_MAX_RETRIES, NOT_GIVEN, NotGiven
from openai import AsyncOpenAI as AsyncOpenAICli
from openai import OpenAI as OpenAICli
from openai.types.chat import (
    ChatCompletionReasoningEffort,
    completion_create_params,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_prediction_content_param import (
    ChatCompletionPredictionContentParam,
)
from openai.types.chat.chat_completion_stream_options_param import (
    ChatCompletionStreamOptionsParam,
)
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat_model import ChatModel
from openai.types.completion_usage import CompletionUsage
from typing_extensions import Literal

from .model import ChatCompletionModel, EmbeddingModel, TextCompletionModel

PROVIDER = "openai"
INITIAL_USAGE = {
    "completion_tokens": 0,
    "prompt_tokens": 0,
    "total_tokens": 0,
    "prompt_tokens_details": {
        "cached_tokens": 0,
        "audio_tokens": 0,
    },
    "completion_tokens_details": {
        "reasoning_tokens": 0,
        "audio_tokens": 0,
        "accepted_prediction_tokens": 0,
        "rejected_prediction_tokens": 0,
    },
}


class Omit:
    """In certain situations you need to be able to represent a case where a default value has
    to be explicitly removed and `None` is not an appropriate substitute, for example:

    ```py
    # as the default `Content-Type` header is `application/json` that will be sent
    client.post("/upload/files", files={"file": b"my raw file content"})

    # you can't explicitly override the header as it has to be dynamically generated
    # to look something like: 'multipart/form-data; boundary=0d8382fcf5f8c3be01ca2e11002d2983'
    client.post(..., headers={"Content-Type": "multipart/form-data"})

    # instead you can remove the default `application/json` header by passing Omit
    client.post(..., headers={"Content-Type": Omit()})
    ```
    """  # noqa: E501

    def __bool__(self) -> Literal[False]:
        return False


Headers = Mapping[str, Union[str, Omit]]
Query = Mapping[str, object]
Body = object


class OpenAI(
    TextCompletionModel,
    ChatCompletionModel,
    EmbeddingModel,
    OpenAICli,
):
    """
    OpenAI synchronous client to perform multiple language model operations.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        base_url: Optional[Union[str, httpx.URL]] = None,
        websocket_base_url: Optional[Union[str, httpx.URL]] = None,
        timeout: Union[float, httpx.Timeout, None, NotGiven] = NOT_GIVEN,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Optional[Mapping[str, str]] = None,
        default_query: Optional[Mapping[str, object]] = None,
        http_client: Optional[httpx.Client] = None,
    ):
        usage = copy.deepcopy(INITIAL_USAGE)  # Ensure a fresh copy

        # Validate and build config
        config = {
            "api_key": api_key or os.getenv("OPENAI_API_KEY"),
            "organization": organization,
            "project": project,
            "base_url": base_url,
            "websocket_base_url": websocket_base_url,
            "timeout": timeout,
            "max_retries": max_retries,
            "default_headers": default_headers,
            "default_query": default_query,
        }
        # Remove None and NOT_GIVEN values
        config = {
            k: v for k, v in config.items() if v is not None and v is not NOT_GIVEN
        }
        # Validate serializability
        for k, v in config.items():
            _validate_config_param(k, v)

        self._client = OpenAICli(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            organization=organization,
            project=project,
            base_url=base_url,
            websocket_base_url=websocket_base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
        )
        super().__init__(provider=PROVIDER, config=config, usage=usage)

    # TODO: Finalize implementation
    def complete(self, prompt: str) -> str:
        """
        Synchronous method to generate a completion for the given prompt.

        Args:
            prompt (str): The input text for which the model should generate
              a completion.

        Returns:
            str: A string containing the generated completion.
        """
        raise NotImplementedError

    def chat(
        self,
        *,
        messages: Iterable[ChatCompletionMessageParam],
        model: Union[str, ChatModel],
        frequency_penalty: Union[Optional[float], NotGiven] = NOT_GIVEN,
        logit_bias: Union[Optional[Dict[str, int]], NotGiven] = NOT_GIVEN,
        logprobs: Union[Optional[bool], NotGiven] = NOT_GIVEN,
        max_completion_tokens: Union[Optional[int], NotGiven] = NOT_GIVEN,
        metadata: Union[Optional[Dict[str, str]], NotGiven] = NOT_GIVEN,
        n: Union[Optional[int], NotGiven] = NOT_GIVEN,
        parallel_tool_calls: Union[bool, NotGiven] = NOT_GIVEN,
        prediction: Union[
            Optional[ChatCompletionPredictionContentParam], NotGiven
        ] = NOT_GIVEN,
        presence_penalty: Union[Optional[float], NotGiven] = NOT_GIVEN,
        reasoning_effort: Union[ChatCompletionReasoningEffort, NotGiven] = NOT_GIVEN,
        response_format: Union[
            completion_create_params.ResponseFormat, NotGiven
        ] = NOT_GIVEN,
        seed: Union[Optional[int], NotGiven] = NOT_GIVEN,
        service_tier: Union[Optional[Literal["auto", "default"]], NotGiven] = NOT_GIVEN,
        stop: Union[Union[Optional[str], List[str]], NotGiven] = NOT_GIVEN,
        store: Union[Optional[bool], NotGiven] = NOT_GIVEN,
        # TODO: `stream` can be useful during inference, but forbid during training or backpropagation
        stream: Union[Optional[Literal[False]], NotGiven] = NOT_GIVEN,
        stream_options: Union[
            Optional[ChatCompletionStreamOptionsParam], NotGiven
        ] = NOT_GIVEN,
        temperature: Union[Optional[float], NotGiven] = NOT_GIVEN,
        tool_choice: Union[ChatCompletionToolChoiceOptionParam, NotGiven] = NOT_GIVEN,
        tools: Union[Iterable[ChatCompletionToolParam], NotGiven] = NOT_GIVEN,
        top_logprobs: Union[Optional[int], NotGiven] = NOT_GIVEN,
        top_p: Union[Optional[float], NotGiven] = NOT_GIVEN,
        user: Union[str, NotGiven] = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Optional[Headers] = None,
        extra_query: Optional[Query] = None,
        extra_body: Optional[Body] = None,
        timeout: Optional[Union[float, httpx.Timeout, NotGiven]] = NOT_GIVEN,
    ) -> str:
        """Synchronously creates a model response for the given chat conversation.

        Learn more in the
        [text generation](https://platform.openai.com/docs/guides/text-generation)
        and [chat](https://platform.openai.com/docs/api-reference/chat) guides.

        Parameter support can differ depending on the model used to generate the
        response, particularly for newer reasoning models.

        Args:
          messages: A list of messages comprising the conversation so far. Depending on
            the [model](https://platform.openai.com/docs/models) you use, different
            message types (modalities) are supported. Refer to
            [text](https://platform.openai.com/docs/guides/text-generation).

          model: ID of the model to use. See the
            [model endpoint compatibility](https://platform.openai.com/docs/models#model-endpoint-compatibility)
            table for details on which models work with the Chat API.

          frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new
            tokens based on their existing frequency in the text so far, decreasing the
            model's likelihood to repeat the same line verbatim.

          logit_bias: Modify the likelihood of specified tokens appearing in the
            completion.

            Accepts a JSON object that maps tokens (specified by their token ID in the
            tokenizer) to an associated bias value from -100 to 100. Mathematically, the
            bias is added to the logits generated by the model prior to sampling. The
            exact effect will vary per model, but values between -1 and 1 should
            decrease or increase likelihood of selection; values like -100 or 100 should
            result in a ban or exclusive selection of the relevant token.

          logprobs: Whether to return log probabilities of the output tokens or not. If
            true, returns the log probabilities of each output token returned in the
            `content` of `message`.

          max_completion_tokens: An upper bound for the number of tokens that can be
            generated for a completion, including visible output tokens and
            [reasoning tokens](https://platform.openai.com/docs/guides/reasoning).

          metadata: Developer-defined tags and values used for filtering completions in
            the [dashboard](https://platform.openai.com/chat-completions).

          n: How many chat completion choices to generate for each input message. Note
            that you will be charged based on the number of generated tokens across all
            of the choices. Keep `n` as `1` to minimize costs.

          parallel_tool_calls: Whether to enable
            [parallel function calling](https://platform.openai.com/docs/guides/function-calling#configuring-parallel-function-calling)
            during tool use.

          prediction: Static predicted output content, such as the content of a text
            file that is being regenerated.

          presence_penalty: Number between -2.0 and 2.0. Positive values penalize new
            tokens based on whether they appear in the text so far, increasing the
            model's likelihood to talk about new topics.

          reasoning_effort: **o1 models only**

            Constrains effort on reasoning for
            [reasoning models](https://platform.openai.com/docs/guides/reasoning).
            Currently supported values are `low`, `medium`, and `high`. Reducing
            reasoning effort can result in faster responses and fewer tokens used on
            reasoning in a response.

          response_format: An object specifying the format that the model must output.

            Setting to `{ "type": "json_schema", "json_schema": {...} }` enables
            Structured Outputs which ensures the model will match your supplied
            JSON schema. Learn more in the
            [Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs).

            Setting to `{ "type": "json_object" }` enables JSON mode, which ensures the
            message the model generates is valid JSON.

            **Important:** when using JSON mode, you **must** also instruct the model to
            produce JSON yourself via a system or user message. Without this, the model
            may generate an unending stream of whitespace until the generation reaches
            the token limit, resulting in a long-running and seemingly "stuck" request.
            Also note that the message content may be partially cut off if
            `finish_reason="length"`, which indicates the generation exceeded
            `max_tokens` or the conversation exceeded the max context length.

          seed: This feature is in Beta. If specified, our system will make a best
            effort to sample deterministically, such that repeated requests with the
            same `seed` and parameters should return the same result. Determinism is not
            guaranteed, and you should refer to the `system_fingerprint` response
            parameter to monitor changes in the backend.

          service_tier: Specifies the latency tier to use for processing the request.
            This parameter is relevant for customers subscribed to the scale tier
            service:

            - If set to 'auto', and the Project is Scale tier enabled, the system will
              utilize scale tier credits until they are exhausted.
            - If set to 'auto', and the Project is not Scale tier enabled, the request
              will be processed using the default service tier with a lower uptime SLA
              and no latency guarentee.
            - If set to 'default', the request will be processed using the default
              service tier with a lower uptime SLA and no latency guarentee.
            - When not set, the default behavior is 'auto'.

            When this parameter is set, the response body will include the
            `service_tier` utilized.

          stop: Up to 4 sequences where the API will stop generating further tokens.

          store: Whether or not to store the output of this chat completion request for
            use in our
            [model distillation](https://platform.openai.com/docs/guides/distillation)
            or [evals](https://platform.openai.com/docs/guides/evals) products.

          stream: If set, partial message deltas will be sent, like in ChatGPT. Tokens
            will be sent as data-only
            [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
            as they become available, with the stream terminated by a `data: [DONE]`
            message.
            [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).

          stream_options: Options for streaming response. Only set this when you set
            `stream: true`.

          temperature: What sampling temperature to use, between 0 and 2. Higher values
            like 0.8 will make the output more random, while lower values like 0.2 will
            make it more focused and deterministic. We generally recommend altering this
            or `top_p` but not both.

          tool_choice: Controls which (if any) tool is called by the model. `none` means
            the model will not call any tool and instead generates a message. `auto`
            means the model can pick between generating a message or calling one or more
            tools. `required` means the model must call one or more tools. Specifying a
            particular tool via
            `{"type": "function", "function": {"name": "my_function"}}` forces the model
            to call that tool.

            `none` is the default when no tools are present. `auto` is the default if
            tools are present.

          tools: A list of tools the model may call. Currently, only functions are
            supported as a tool. Use this to provide a list of functions the model may
            generate JSON inputs for. A max of 128 functions are supported.

          top_logprobs: An integer between 0 and 20 specifying the number of most likely
            tokens to return at each token position, each with an associated log
            probability. `logprobs` must be set to `true` if this parameter is used.

          top_p: An alternative to sampling with temperature, called nucleus sampling,
            where the model considers the results of the tokens with top_p probability
            mass. So 0.1 means only the tokens comprising the top 10% probability mass
            are considered.

            We generally recommend altering this or `temperature` but not both.

          user: A unique identifier representing your end-user, which can help OpenAI to
            monitor and detect abuse.
            [Learn more](https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in
            seconds
        """  # noqa: E501
        # TODO: handle `n > 1`` that returns multiple completions
        if isinstance(n, int) and n > 1:
            raise ValueError("n > 1 is not supported for chat completions.")

        response = self._client.chat.completions.create(
            messages=messages,
            model=model,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_completion_tokens=max_completion_tokens,
            metadata=metadata,
            n=n,
            parallel_tool_calls=parallel_tool_calls,
            prediction=prediction,
            presence_penalty=presence_penalty,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
            seed=seed,
            service_tier=service_tier,
            stop=stop,
            store=store,
            stream=stream,
            stream_options=stream_options,
            temperature=temperature,
            tool_choice=tool_choice,
            tools=tools,
            top_logprobs=top_logprobs,
            top_p=top_p,
            user=user,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )

        # Update usage statistics
        if hasattr(response, "usage") and response.usage:
            self.update_usage(response.usage)

        return response.choices[0].message.content

    # TODO: Finalize implementation
    def embed(self, input: List[str]) -> List[List[float]]:
        """
        Synchronous method to generate embeddings for the given input texts.

        Args:
            input (List[str]): A list of input strings for which embeddings
              should be generated.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is represented
              as a list of floats corresponding to the input strings.
        """
        raise NotImplementedError

    def update_usage(self, usage: CompletionUsage) -> None:
        """Updates the internal usage counters with values from a new API response.

        Args:
            usage (CompletionUsage): The usage object returned by the OpenAI API.
        """
        if not hasattr(self, "_usage"):
            self._usage = copy.deepcopy(INITIAL_USAGE)  # Ensure a fresh copy

        # Ensure we convert CompletionUsage to dict properly
        if isinstance(usage, CompletionUsage):
            usage = usage.model_dump()

        # Update core token usage fields
        self._usage["completion_tokens"] += usage.get("completion_tokens", 0)
        self._usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        self._usage["total_tokens"] += usage.get("total_tokens", 0)

        # Update prompt tokens details
        prompt_tokens_details = usage.get("prompt_tokens_details", {})
        self._usage["prompt_tokens_details"][
            "cached_tokens"
        ] += prompt_tokens_details.get("cached_tokens", 0)
        self._usage["prompt_tokens_details"][
            "audio_tokens"
        ] += prompt_tokens_details.get("audio_tokens", 0)

        # Update completion tokens details
        completion_tokens_details = usage.get("completion_tokens_details", {})
        self._usage["completion_tokens_details"][
            "reasoning_tokens"
        ] += completion_tokens_details.get("reasoning_tokens", 0)
        self._usage["completion_tokens_details"][
            "audio_tokens"
        ] += completion_tokens_details.get("audio_tokens", 0)
        self._usage["completion_tokens_details"][
            "accepted_prediction_tokens"
        ] += completion_tokens_details.get("accepted_prediction_tokens", 0)
        self._usage["completion_tokens_details"][
            "rejected_prediction_tokens"
        ] += completion_tokens_details.get("rejected_prediction_tokens", 0)


class AsyncOpenAI(
    TextCompletionModel,
    ChatCompletionModel,
    EmbeddingModel,
    AsyncOpenAICli,
):
    """
    OpenAI asynchronous client to perform multiple language model operations.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        base_url: Optional[Union[str, httpx.URL]] = None,
        websocket_base_url: Optional[Union[str, httpx.URL]] = None,
        timeout: Union[float, httpx.Timeout, None, NotGiven] = NOT_GIVEN,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Optional[Mapping[str, str]] = None,
        default_query: Optional[Mapping[str, object]] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        usage = copy.deepcopy(INITIAL_USAGE)  # Ensure a fresh copy

        # Validate and build config
        config = {
            "api_key": api_key or os.getenv("OPENAI_API_KEY"),
            "organization": organization,
            "project": project,
            "base_url": base_url,
            "websocket_base_url": websocket_base_url,
            "timeout": timeout,
            "max_retries": max_retries,
            "default_headers": default_headers,
            "default_query": default_query,
        }
        # Remove None and NOT_GIVEN values
        config = {
            k: v for k, v in config.items() if v is not None and v is not NOT_GIVEN
        }
        # Validate serializability
        for k, v in config.items():
            _validate_config_param(k, v)

        self._aclient = AsyncOpenAICli(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            organization=organization,
            project=project,
            base_url=base_url,
            websocket_base_url=websocket_base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
        )
        super().__init__(provider=PROVIDER, config=config, usage=usage)

    # TODO: Finalize implementation
    async def acomplete(self, prompt: str) -> str:
        """
        Asynchronous method to generate a completion for the given prompt.

        Args:
            prompt (str): The input text for which the model should generate
              a completion.

        Returns:
            str: A string containing the generated completion.
        """
        raise NotImplementedError

    async def achat(
        self,
        *,
        messages: Iterable[ChatCompletionMessageParam],
        model: Union[str, ChatModel],
        frequency_penalty: Union[Optional[float], NotGiven] = NOT_GIVEN,
        logit_bias: Union[Optional[Dict[str, int]], NotGiven] = NOT_GIVEN,
        logprobs: Union[Optional[bool], NotGiven] = NOT_GIVEN,
        max_completion_tokens: Union[Optional[int], NotGiven] = NOT_GIVEN,
        metadata: Union[Optional[Dict[str, str]], NotGiven] = NOT_GIVEN,
        n: Union[Optional[int], NotGiven] = NOT_GIVEN,
        parallel_tool_calls: Union[bool, NotGiven] = NOT_GIVEN,
        prediction: Union[
            Optional[ChatCompletionPredictionContentParam], NotGiven
        ] = NOT_GIVEN,
        presence_penalty: Union[Optional[float], NotGiven] = NOT_GIVEN,
        reasoning_effort: Union[ChatCompletionReasoningEffort, NotGiven] = NOT_GIVEN,
        response_format: Union[
            completion_create_params.ResponseFormat, NotGiven
        ] = NOT_GIVEN,
        seed: Union[Optional[int], NotGiven] = NOT_GIVEN,
        service_tier: Union[Optional[Literal["auto", "default"]], NotGiven] = NOT_GIVEN,
        stop: Union[Union[Optional[str], List[str]], NotGiven] = NOT_GIVEN,
        store: Union[Optional[bool], NotGiven] = NOT_GIVEN,
        # TODO: `stream` can be useful during inference, but forbid during training or backpropagation
        stream: Union[Optional[Literal[False]], NotGiven] = NOT_GIVEN,
        stream_options: Union[
            Optional[ChatCompletionStreamOptionsParam], NotGiven
        ] = NOT_GIVEN,
        temperature: Union[Optional[float], NotGiven] = NOT_GIVEN,
        tool_choice: Union[ChatCompletionToolChoiceOptionParam, NotGiven] = NOT_GIVEN,
        tools: Union[Iterable[ChatCompletionToolParam], NotGiven] = NOT_GIVEN,
        top_logprobs: Union[Optional[int], NotGiven] = NOT_GIVEN,
        top_p: Union[Optional[float], NotGiven] = NOT_GIVEN,
        user: Union[str, NotGiven] = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Optional[Headers] = None,
        extra_query: Optional[Query] = None,
        extra_body: Optional[Body] = None,
        timeout: Optional[Union[float, httpx.Timeout, NotGiven]] = NOT_GIVEN,
    ) -> str:
        """Asynchronously creates a model response for the given chat conversation.

        Learn more in the
        [text generation](https://platform.openai.com/docs/guides/text-generation)
        and [chat](https://platform.openai.com/docs/api-reference/chat) guides.

        Parameter support can differ depending on the model used to generate the
        response, particularly for newer reasoning models.

        Args:
          messages: A list of messages comprising the conversation so far. Depending on
            the [model](https://platform.openai.com/docs/models) you use, different
            message types (modalities) are supported. Refer to
            [text](https://platform.openai.com/docs/guides/text-generation).

          model: ID of the model to use. See the
            [model endpoint compatibility](https://platform.openai.com/docs/models#model-endpoint-compatibility)
            table for details on which models work with the Chat API.

          frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new
            tokens based on their existing frequency in the text so far, decreasing the
            model's likelihood to repeat the same line verbatim.

          logit_bias: Modify the likelihood of specified tokens appearing in the
            completion.

            Accepts a JSON object that maps tokens (specified by their token ID in the
            tokenizer) to an associated bias value from -100 to 100. Mathematically, the
            bias is added to the logits generated by the model prior to sampling. The
            exact effect will vary per model, but values between -1 and 1 should
            decrease or increase likelihood of selection; values like -100 or 100 should
            result in a ban or exclusive selection of the relevant token.

          logprobs: Whether to return log probabilities of the output tokens or not. If
            true, returns the log probabilities of each output token returned in the
            `content` of `message`.

          max_completion_tokens: An upper bound for the number of tokens that can be
            generated for a completion, including visible output tokens and
            [reasoning tokens](https://platform.openai.com/docs/guides/reasoning).

          metadata: Developer-defined tags and values used for filtering completions in
            the [dashboard](https://platform.openai.com/chat-completions).

          n: How many chat completion choices to generate for each input message. Note
            that you will be charged based on the number of generated tokens across all
            of the choices. Keep `n` as `1` to minimize costs.

          parallel_tool_calls: Whether to enable
            [parallel function calling](https://platform.openai.com/docs/guides/function-calling#configuring-parallel-function-calling)
            during tool use.

          prediction: Static predicted output content, such as the content of a text
            file that is being regenerated.

          presence_penalty: Number between -2.0 and 2.0. Positive values penalize new
            tokens based on whether they appear in the text so far, increasing the
            model's likelihood to talk about new topics.

          reasoning_effort: **o1 models only**

            Constrains effort on reasoning for
            [reasoning models](https://platform.openai.com/docs/guides/reasoning).
            Currently supported values are `low`, `medium`, and `high`. Reducing
            reasoning effort can result in faster responses and fewer tokens used on
            reasoning in a response.

          response_format: An object specifying the format that the model must output.

            Setting to `{ "type": "json_schema", "json_schema": {...} }` enables
            Structured Outputs which ensures the model will match your supplied
            JSON schema. Learn more in the
            [Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs).

            Setting to `{ "type": "json_object" }` enables JSON mode, which ensures the
            message the model generates is valid JSON.

            **Important:** when using JSON mode, you **must** also instruct the model to
            produce JSON yourself via a system or user message. Without this, the model
            may generate an unending stream of whitespace until the generation reaches
            the token limit, resulting in a long-running and seemingly "stuck" request.
            Also note that the message content may be partially cut off if
            `finish_reason="length"`, which indicates the generation exceeded
            `max_tokens` or the conversation exceeded the max context length.

          seed: This feature is in Beta. If specified, our system will make a best
            effort to sample deterministically, such that repeated requests with the
            same `seed` and parameters should return the same result. Determinism is not
            guaranteed, and you should refer to the `system_fingerprint` response
            parameter to monitor changes in the backend.

          service_tier: Specifies the latency tier to use for processing the request.
            This parameter is relevant for customers subscribed to the scale tier
            service:

            - If set to 'auto', and the Project is Scale tier enabled, the system will
              utilize scale tier credits until they are exhausted.
            - If set to 'auto', and the Project is not Scale tier enabled, the request
              will be processed using the default service tier with a lower uptime SLA
              and no latency guarentee.
            - If set to 'default', the request will be processed using the default
              service tier with a lower uptime SLA and no latency guarentee.
            - When not set, the default behavior is 'auto'.

            When this parameter is set, the response body will include the
            `service_tier` utilized.

          stop: Up to 4 sequences where the API will stop generating further tokens.

          store: Whether or not to store the output of this chat completion request for
            use in our
            [model distillation](https://platform.openai.com/docs/guides/distillation)
            or [evals](https://platform.openai.com/docs/guides/evals) products.

          stream: If set, partial message deltas will be sent, like in ChatGPT. Tokens
            will be sent as data-only
            [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
            as they become available, with the stream terminated by a `data: [DONE]`
            message.
            [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).

          stream_options: Options for streaming response. Only set this when you set
            `stream: true`.

          temperature: What sampling temperature to use, between 0 and 2. Higher values
            like 0.8 will make the output more random, while lower values like 0.2 will
            make it more focused and deterministic. We generally recommend altering this
            or `top_p` but not both.

          tool_choice: Controls which (if any) tool is called by the model. `none` means
            the model will not call any tool and instead generates a message. `auto`
            means the model can pick between generating a message or calling one or more
            tools. `required` means the model must call one or more tools. Specifying a
            particular tool via
            `{"type": "function", "function": {"name": "my_function"}}` forces the model
            to call that tool.

            `none` is the default when no tools are present. `auto` is the default if
            tools are present.

          tools: A list of tools the model may call. Currently, only functions are
            supported as a tool. Use this to provide a list of functions the model may
            generate JSON inputs for. A max of 128 functions are supported.

          top_logprobs: An integer between 0 and 20 specifying the number of most likely
            tokens to return at each token position, each with an associated log
            probability. `logprobs` must be set to `true` if this parameter is used.

          top_p: An alternative to sampling with temperature, called nucleus sampling,
            where the model considers the results of the tokens with top_p probability
            mass. So 0.1 means only the tokens comprising the top 10% probability mass
            are considered.

            We generally recommend altering this or `temperature` but not both.

          user: A unique identifier representing your end-user, which can help OpenAI to
            monitor and detect abuse.
            [Learn more](https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in
            seconds
        """  # noqa: E501
        # TODO: handle `n > 1`` that returns multiple completions
        if isinstance(n, int) and n > 1:
            raise ValueError("n > 1 is not supported for async chat completions.")

        response = await self._aclient.chat.completions.create(
            messages=messages,
            model=model,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_completion_tokens=max_completion_tokens,
            metadata=metadata,
            n=n,
            parallel_tool_calls=parallel_tool_calls,
            prediction=prediction,
            presence_penalty=presence_penalty,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
            seed=seed,
            service_tier=service_tier,
            stop=stop,
            store=store,
            stream=stream,
            stream_options=stream_options,
            temperature=temperature,
            tool_choice=tool_choice,
            tools=tools,
            top_logprobs=top_logprobs,
            top_p=top_p,
            user=user,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )

        # Update usage statistics
        if hasattr(response, "usage") and response.usage:
            self.update_usage(response.usage)

        return response.choices[0].message.content

    # TODO: Finalize implementation
    async def aembed(self, input: List[str]) -> List[List[float]]:
        """
        Asynchronous method to generate embeddings for the given input texts.

        Args:
            input (List[str]): A list of input strings for which embeddings
              should be generated.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is represented
              as a list of floats corresponding to the input strings.
        """
        raise NotImplementedError

    def update_usage(self, usage: CompletionUsage) -> None:
        """Updates the internal usage counters with values from a new API response.

        Args:
            usage (CompletionUsage): The usage object returned by the OpenAI API.
        """
        if not hasattr(self, "_usage"):
            self._usage = copy.deepcopy(INITIAL_USAGE)  # Ensure a fresh copy

        # Ensure we convert CompletionUsage to dict properly
        if isinstance(usage, CompletionUsage):
            usage = usage.model_dump()

        # Update core token usage fields
        self._usage["completion_tokens"] += usage.get("completion_tokens", 0)
        self._usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        self._usage["total_tokens"] += usage.get("total_tokens", 0)

        # Update prompt tokens details
        prompt_tokens_details = usage.get("prompt_tokens_details", {})
        self._usage["prompt_tokens_details"][
            "cached_tokens"
        ] += prompt_tokens_details.get("cached_tokens", 0)
        self._usage["prompt_tokens_details"][
            "audio_tokens"
        ] += prompt_tokens_details.get("audio_tokens", 0)

        # Update completion tokens details
        completion_tokens_details = usage.get("completion_tokens_details", {})
        self._usage["completion_tokens_details"][
            "reasoning_tokens"
        ] += completion_tokens_details.get("reasoning_tokens", 0)
        self._usage["completion_tokens_details"][
            "audio_tokens"
        ] += completion_tokens_details.get("audio_tokens", 0)
        self._usage["completion_tokens_details"][
            "accepted_prediction_tokens"
        ] += completion_tokens_details.get("accepted_prediction_tokens", 0)
        self._usage["completion_tokens_details"][
            "rejected_prediction_tokens"
        ] += completion_tokens_details.get("rejected_prediction_tokens", 0)


def _validate_config_param(name, value):
    # Accept basic JSON types
    if isinstance(value, (str, int, float, bool, type(None))):
        return
    # Recursively check mappings (dict, Mapping)
    if isinstance(value, Mapping):
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"Config parameter '{name}' "
                    f"has a non-string key '{k}' of type {type(k).__name__}."
                )
            _validate_config_param(f"{name}.{k}", v)
        return
    # Recursively check sequences (list, tuple, set) but not str/bytes
    if isinstance(value, (list, tuple, set)):
        for idx, item in enumerate(value):
            _validate_config_param(f"{name}[{idx}]", item)
        return
    # Explicitly reject bytes and bytearray
    if isinstance(value, (bytes, bytearray)):
        raise TypeError(
            f"Config parameter '{name}' "
            f"is of type {type(value).__name__}, which is not JSON serializable."
        )
    # Fallback: try json serialization
    try:
        json.dumps(value)
    except Exception:
        raise TypeError(
            f"Config parameter '{name}' "
            f"with value '{value}' of type {type(value).__name__} is not serializable."
        )
