import copy
import json
import os
from typing import Dict, Iterable, List, Mapping, Optional, Union

import httpx
from openai import DEFAULT_MAX_RETRIES, NOT_GIVEN, NotGiven
from openai import AsyncOpenAI as AsyncOpenAICli
from openai import OpenAI as OpenAICli
from openai._types import SequenceNotStr
from openai.types.chat import (
    ChatCompletionAudioParam,
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
from openai.types.chat.chat_completion_tool_union_param import (
    ChatCompletionToolUnionParam,
)
from openai.types.chat_model import ChatModel
from openai.types.completion_usage import CompletionUsage
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.metadata import Metadata
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
        audio: Union[Optional[ChatCompletionAudioParam], NotGiven] = NOT_GIVEN,
        frequency_penalty: Union[Optional[float], NotGiven] = NOT_GIVEN,
        function_call: Union[
            completion_create_params.FunctionCall, NotGiven
        ] = NOT_GIVEN,
        functions: Union[
            Iterable[completion_create_params.Function], NotGiven
        ] = NOT_GIVEN,
        logit_bias: Union[Optional[Dict[str, int]], NotGiven] = NOT_GIVEN,
        logprobs: Union[Optional[bool], NotGiven] = NOT_GIVEN,
        max_completion_tokens: Union[Optional[int], NotGiven] = NOT_GIVEN,
        max_tokens: Union[Optional[int], NotGiven] = NOT_GIVEN,
        metadata: Union[Optional[Metadata], NotGiven] = NOT_GIVEN,
        modalities: Union[
            Optional[List[Literal["text", "audio"]]], NotGiven
        ] = NOT_GIVEN,
        n: Union[Optional[int], NotGiven] = NOT_GIVEN,
        parallel_tool_calls: Union[bool, NotGiven] = NOT_GIVEN,
        prediction: Union[
            Optional[ChatCompletionPredictionContentParam], NotGiven
        ] = NOT_GIVEN,
        presence_penalty: Union[Optional[float], NotGiven] = NOT_GIVEN,
        prompt_cache_key: Union[str, NotGiven] = NOT_GIVEN,
        reasoning_effort: Union[ReasoningEffort, NotGiven] = NOT_GIVEN,
        response_format: Union[
            completion_create_params.ResponseFormat, NotGiven
        ] = NOT_GIVEN,
        safety_identifier: Union[str, NotGiven] = NOT_GIVEN,
        seed: Union[Optional[int], NotGiven] = NOT_GIVEN,
        service_tier: Union[
            Optional[Literal["auto", "default", "flex", "scale", "priority"]], NotGiven
        ] = NOT_GIVEN,
        stop: Union[
            Union[Optional[str], SequenceNotStr[str], None], NotGiven
        ] = NOT_GIVEN,
        store: Union[Optional[bool], NotGiven] = NOT_GIVEN,
        # TODO: `stream` can be useful during inference, but forbid during training or backpropagation
        stream: Union[Optional[Literal[False]], NotGiven] = NOT_GIVEN,
        stream_options: Union[
            Optional[ChatCompletionStreamOptionsParam], NotGiven
        ] = NOT_GIVEN,
        temperature: Union[Optional[float], NotGiven] = NOT_GIVEN,
        tool_choice: Union[ChatCompletionToolChoiceOptionParam, NotGiven] = NOT_GIVEN,
        tools: Union[Iterable[ChatCompletionToolUnionParam], NotGiven] = NOT_GIVEN,
        top_logprobs: Union[Optional[int], NotGiven] = NOT_GIVEN,
        top_p: Union[Optional[float], NotGiven] = NOT_GIVEN,
        user: Union[str, NotGiven] = NOT_GIVEN,
        verbosity: Union[
            Optional[Literal["low", "medium", "high"]], NotGiven
        ] = NOT_GIVEN,
        web_search_options: Union[
            completion_create_params.WebSearchOptions, NotGiven
        ] = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Optional[Headers] = None,
        extra_query: Optional[Query] = None,
        extra_body: Optional[Body] = None,
        timeout: Optional[Union[float, httpx.Timeout, NotGiven]] = NOT_GIVEN,
    ) -> str:
        """Synchronously creates a model response for the given chat conversation.
        Learn more in the
        [text generation](https://platform.openai.com/docs/guides/text-generation),
        [vision](https://platform.openai.com/docs/guides/vision), and
        [audio](https://platform.openai.com/docs/guides/audio) guides.

        Parameter support can differ depending on the model used to generate the
        response, particularly for newer reasoning models. Parameters that are only
        supported for reasoning models are noted below. For the current state of
        unsupported parameters in reasoning models,
        [refer to the reasoning guide](https://platform.openai.com/docs/guides/reasoning).

        Args:
          messages: A list of messages comprising the conversation so far. Depending on the
              [model](https://platform.openai.com/docs/models) you use, different message
              types (modalities) are supported, like
              [text](https://platform.openai.com/docs/guides/text-generation),
              [images](https://platform.openai.com/docs/guides/vision), and
              [audio](https://platform.openai.com/docs/guides/audio).

          model: Model ID used to generate the response, like `gpt-4o` or `o3`. OpenAI offers a
              wide range of models with different capabilities, performance characteristics,
              and price points. Refer to the
              [model guide](https://platform.openai.com/docs/models) to browse and compare
              available models.

          audio: Parameters for audio output. Required when audio output is requested with
              `modalities: ["audio"]`.
              [Learn more](https://platform.openai.com/docs/guides/audio).

          frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their
              existing frequency in the text so far, decreasing the model's likelihood to
              repeat the same line verbatim.

          function_call: Deprecated in favor of `tool_choice`.

              Controls which (if any) function is called by the model.

              `none` means the model will not call a function and instead generates a message.

              `auto` means the model can pick between generating a message or calling a
              function.

              Specifying a particular function via `{"name": "my_function"}` forces the model
              to call that function.

              `none` is the default when no functions are present. `auto` is the default if
              functions are present.

          functions: Deprecated in favor of `tools`.

              A list of functions the model may generate JSON inputs for.

          logit_bias: Modify the likelihood of specified tokens appearing in the completion.

              Accepts a JSON object that maps tokens (specified by their token ID in the
              tokenizer) to an associated bias value from -100 to 100. Mathematically, the
              bias is added to the logits generated by the model prior to sampling. The exact
              effect will vary per model, but values between -1 and 1 should decrease or
              increase likelihood of selection; values like -100 or 100 should result in a ban
              or exclusive selection of the relevant token.

          logprobs: Whether to return log probabilities of the output tokens or not. If true,
              returns the log probabilities of each output token returned in the `content` of
              `message`.

          max_completion_tokens: An upper bound for the number of tokens that can be generated for a completion,
              including visible output tokens and
              [reasoning tokens](https://platform.openai.com/docs/guides/reasoning).

          max_tokens: The maximum number of [tokens](/tokenizer) that can be generated in the chat
              completion. This value can be used to control
              [costs](https://openai.com/api/pricing/) for text generated via API.

              This value is now deprecated in favor of `max_completion_tokens`, and is not
              compatible with
              [o-series models](https://platform.openai.com/docs/guides/reasoning).

          metadata: Set of 16 key-value pairs that can be attached to an object. This can be useful
              for storing additional information about the object in a structured format, and
              querying for objects via API or the dashboard.

              Keys are strings with a maximum length of 64 characters. Values are strings with
              a maximum length of 512 characters.

          modalities: Output types that you would like the model to generate. Most models are capable
              of generating text, which is the default:

              `["text"]`

              The `gpt-4o-audio-preview` model can also be used to
              [generate audio](https://platform.openai.com/docs/guides/audio). To request that
              this model generate both text and audio responses, you can use:

              `["text", "audio"]`

          n: How many chat completion choices to generate for each input message. Note that
              you will be charged based on the number of generated tokens across all of the
              choices. Keep `n` as `1` to minimize costs.

          parallel_tool_calls: Whether to enable
              [parallel function calling](https://platform.openai.com/docs/guides/function-calling#configuring-parallel-function-calling)
              during tool use.

          prediction: Static predicted output content, such as the content of a text file that is
              being regenerated.

          presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on
              whether they appear in the text so far, increasing the model's likelihood to
              talk about new topics.

          prompt_cache_key: Used by OpenAI to cache responses for similar requests to optimize your cache
              hit rates. Replaces the `user` field.
              [Learn more](https://platform.openai.com/docs/guides/prompt-caching).

          reasoning_effort: Constrains effort on reasoning for
              [reasoning models](https://platform.openai.com/docs/guides/reasoning). Currently
              supported values are `minimal`, `low`, `medium`, and `high`. Reducing reasoning
              effort can result in faster responses and fewer tokens used on reasoning in a
              response.

          response_format: An object specifying the format that the model must output.

              Setting to `{ "type": "json_schema", "json_schema": {...} }` enables Structured
              Outputs which ensures the model will match your supplied JSON schema. Learn more
              in the
              [Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs).

              Setting to `{ "type": "json_object" }` enables the older JSON mode, which
              ensures the message the model generates is valid JSON. Using `json_schema` is
              preferred for models that support it.

          safety_identifier: A stable identifier used to help detect users of your application that may be
              violating OpenAI's usage policies. The IDs should be a string that uniquely
              identifies each user. We recommend hashing their username or email address, in
              order to avoid sending us any identifying information.
              [Learn more](https://platform.openai.com/docs/guides/safety-best-practices#safety-identifiers).

          seed: This feature is in Beta. If specified, our system will make a best effort to
              sample deterministically, such that repeated requests with the same `seed` and
              parameters should return the same result. Determinism is not guaranteed, and you
              should refer to the `system_fingerprint` response parameter to monitor changes
              in the backend.

          service_tier: Specifies the processing type used for serving the request.

              - If set to 'auto', then the request will be processed with the service tier
                configured in the Project settings. Unless otherwise configured, the Project
                will use 'default'.
              - If set to 'default', then the request will be processed with the standard
                pricing and performance for the selected model.
              - If set to '[flex](https://platform.openai.com/docs/guides/flex-processing)' or
                '[priority](https://openai.com/api-priority-processing/)', then the request
                will be processed with the corresponding service tier.
              - When not set, the default behavior is 'auto'.

              When the `service_tier` parameter is set, the response body will include the
              `service_tier` value based on the processing mode actually used to serve the
              request. This response value may be different from the value set in the
              parameter.

          stop: Not supported with latest reasoning models `o3` and `o4-mini`.

              Up to 4 sequences where the API will stop generating further tokens. The
              returned text will not contain the stop sequence.

          store: Whether or not to store the output of this chat completion request for use in
              our [model distillation](https://platform.openai.com/docs/guides/distillation)
              or [evals](https://platform.openai.com/docs/guides/evals) products.

              Supports text and image inputs. Note: image inputs over 8MB will be dropped.

          stream: If set to true, the model response data will be streamed to the client as it is
              generated using
              [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format).
              See the
              [Streaming section below](https://platform.openai.com/docs/api-reference/chat/streaming)
              for more information, along with the
              [streaming responses](https://platform.openai.com/docs/guides/streaming-responses)
              guide for more information on how to handle the streaming events.

          stream_options: Options for streaming response. Only set this when you set `stream: true`.

          temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
              make the output more random, while lower values like 0.2 will make it more
              focused and deterministic. We generally recommend altering this or `top_p` but
              not both.

          tool_choice: Controls which (if any) tool is called by the model. `none` means the model will
              not call any tool and instead generates a message. `auto` means the model can
              pick between generating a message or calling one or more tools. `required` means
              the model must call one or more tools. Specifying a particular tool via
              `{"type": "function", "function": {"name": "my_function"}}` forces the model to
              call that tool.

              `none` is the default when no tools are present. `auto` is the default if tools
              are present.

          tools: A list of tools the model may call. You can provide either
              [custom tools](https://platform.openai.com/docs/guides/function-calling#custom-tools)
              or [function tools](https://platform.openai.com/docs/guides/function-calling).

          top_logprobs: An integer between 0 and 20 specifying the number of most likely tokens to
              return at each token position, each with an associated log probability.
              `logprobs` must be set to `true` if this parameter is used.

          top_p: An alternative to sampling with temperature, called nucleus sampling, where the
              model considers the results of the tokens with top_p probability mass. So 0.1
              means only the tokens comprising the top 10% probability mass are considered.

              We generally recommend altering this or `temperature` but not both.

          user: This field is being replaced by `safety_identifier` and `prompt_cache_key`. Use
              `prompt_cache_key` instead to maintain caching optimizations. A stable
              identifier for your end-users. Used to boost cache hit rates by better bucketing
              similar requests and to help OpenAI detect and prevent abuse.
              [Learn more](https://platform.openai.com/docs/guides/safety-best-practices#safety-identifiers).

          verbosity: Constrains the verbosity of the model's response. Lower values will result in
              more concise responses, while higher values will result in more verbose
              responses. Currently supported values are `low`, `medium`, and `high`.

          web_search_options: This tool searches the web for relevant results to use in a response. Learn more
              about the
              [web search tool](https://platform.openai.com/docs/guides/tools-web-search?api-mode=chat).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """  # noqa: E501
        # TODO: handle `n > 1`` that returns multiple completions
        if isinstance(n, int) and n > 1:
            raise ValueError("n > 1 is not supported for chat completions.")

        response = self._client.chat.completions.create(
            messages=messages,
            model=model,
            audio=audio,
            frequency_penalty=frequency_penalty,
            function_call=function_call,
            functions=functions,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            metadata=metadata,
            modalities=modalities,
            n=n,
            parallel_tool_calls=parallel_tool_calls,
            prediction=prediction,
            presence_penalty=presence_penalty,
            prompt_cache_key=prompt_cache_key,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
            safety_identifier=safety_identifier,
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
            verbosity=verbosity,
            web_search_options=web_search_options,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )

        # Update usage statistics
        if hasattr(response, "usage") and response.usage:
            self.update_usage(response.usage, model)

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

    def update_usage(self, usage: CompletionUsage, model_name: str = None) -> None:
        """Updates the internal usage counters with values from a new API response.

        Args:
            usage (CompletionUsage): The usage object returned by the OpenAI API.
            model_name (str, optional): The name of the model for which the usage
                is being updated. If None, cost is copied from usage if available.
        """
        if not hasattr(self, "_usage"):
            self._usage.update(copy.deepcopy(INITIAL_USAGE))  # Ensure a fresh copy

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

        # Update cost
        if model_name is not None:
            pricing = _get_pricing_for_model(self.provider, model_name)
            cost = _calculate_cost(usage, pricing)
            self._usage["cost"]["amount"] += cost
        else:
            # If cost is present in usage, copy it directly
            if "cost" in usage and "amount" in usage["cost"]:
                self._usage["cost"]["amount"] = usage["cost"]["amount"]


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
        audio: Union[Optional[ChatCompletionAudioParam], NotGiven] = NOT_GIVEN,
        frequency_penalty: Union[Optional[float], NotGiven] = NOT_GIVEN,
        function_call: Union[
            completion_create_params.FunctionCall, NotGiven
        ] = NOT_GIVEN,
        functions: Union[
            Iterable[completion_create_params.Function], NotGiven
        ] = NOT_GIVEN,
        logit_bias: Union[Optional[Dict[str, int]], NotGiven] = NOT_GIVEN,
        logprobs: Union[Optional[bool], NotGiven] = NOT_GIVEN,
        max_completion_tokens: Union[Optional[int], NotGiven] = NOT_GIVEN,
        max_tokens: Union[Optional[int], NotGiven] = NOT_GIVEN,
        metadata: Union[Optional[Metadata], NotGiven] = NOT_GIVEN,
        modalities: Union[
            Optional[List[Literal["text", "audio"]]], NotGiven
        ] = NOT_GIVEN,
        n: Union[Optional[int], NotGiven] = NOT_GIVEN,
        parallel_tool_calls: Union[bool, NotGiven] = NOT_GIVEN,
        prediction: Union[
            Optional[ChatCompletionPredictionContentParam], NotGiven
        ] = NOT_GIVEN,
        presence_penalty: Union[Optional[float], NotGiven] = NOT_GIVEN,
        prompt_cache_key: Union[str, NotGiven] = NOT_GIVEN,
        reasoning_effort: Union[ReasoningEffort, NotGiven] = NOT_GIVEN,
        response_format: Union[
            completion_create_params.ResponseFormat, NotGiven
        ] = NOT_GIVEN,
        safety_identifier: Union[str, NotGiven] = NOT_GIVEN,
        seed: Union[Optional[int], NotGiven] = NOT_GIVEN,
        service_tier: Union[
            Optional[Literal["auto", "default", "flex", "scale", "priority"]], NotGiven
        ] = NOT_GIVEN,
        stop: Union[
            Union[Optional[str], SequenceNotStr[str], None], NotGiven
        ] = NOT_GIVEN,
        store: Union[Optional[bool], NotGiven] = NOT_GIVEN,
        # TODO: `stream` can be useful during inference, but forbid during training or backpropagation
        stream: Union[Optional[Literal[False]], NotGiven] = NOT_GIVEN,
        stream_options: Union[
            Optional[ChatCompletionStreamOptionsParam], NotGiven
        ] = NOT_GIVEN,
        temperature: Union[Optional[float], NotGiven] = NOT_GIVEN,
        tool_choice: Union[ChatCompletionToolChoiceOptionParam, NotGiven] = NOT_GIVEN,
        tools: Union[Iterable[ChatCompletionToolUnionParam], NotGiven] = NOT_GIVEN,
        top_logprobs: Union[Optional[int], NotGiven] = NOT_GIVEN,
        top_p: Union[Optional[float], NotGiven] = NOT_GIVEN,
        user: Union[str, NotGiven] = NOT_GIVEN,
        verbosity: Union[
            Optional[Literal["low", "medium", "high"]], NotGiven
        ] = NOT_GIVEN,
        web_search_options: Union[
            completion_create_params.WebSearchOptions, NotGiven
        ] = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Optional[Headers] = None,
        extra_query: Optional[Query] = None,
        extra_body: Optional[Body] = None,
        timeout: Optional[Union[float, httpx.Timeout, NotGiven]] = NOT_GIVEN,
    ) -> str:
        """Asynchronously creates a model response for the given chat conversation.
        Learn more in the
        [text generation](https://platform.openai.com/docs/guides/text-generation),
        [vision](https://platform.openai.com/docs/guides/vision), and
        [audio](https://platform.openai.com/docs/guides/audio) guides.

        Parameter support can differ depending on the model used to generate the
        response, particularly for newer reasoning models. Parameters that are only
        supported for reasoning models are noted below. For the current state of
        unsupported parameters in reasoning models,
        [refer to the reasoning guide](https://platform.openai.com/docs/guides/reasoning).

        Args:
          messages: A list of messages comprising the conversation so far. Depending on the
              [model](https://platform.openai.com/docs/models) you use, different message
              types (modalities) are supported, like
              [text](https://platform.openai.com/docs/guides/text-generation),
              [images](https://platform.openai.com/docs/guides/vision), and
              [audio](https://platform.openai.com/docs/guides/audio).

          model: Model ID used to generate the response, like `gpt-4o` or `o3`. OpenAI offers a
              wide range of models with different capabilities, performance characteristics,
              and price points. Refer to the
              [model guide](https://platform.openai.com/docs/models) to browse and compare
              available models.

          audio: Parameters for audio output. Required when audio output is requested with
              `modalities: ["audio"]`.
              [Learn more](https://platform.openai.com/docs/guides/audio).

          frequency_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on their
              existing frequency in the text so far, decreasing the model's likelihood to
              repeat the same line verbatim.

          function_call: Deprecated in favor of `tool_choice`.

              Controls which (if any) function is called by the model.

              `none` means the model will not call a function and instead generates a message.

              `auto` means the model can pick between generating a message or calling a
              function.

              Specifying a particular function via `{"name": "my_function"}` forces the model
              to call that function.

              `none` is the default when no functions are present. `auto` is the default if
              functions are present.

          functions: Deprecated in favor of `tools`.

              A list of functions the model may generate JSON inputs for.

          logit_bias: Modify the likelihood of specified tokens appearing in the completion.

              Accepts a JSON object that maps tokens (specified by their token ID in the
              tokenizer) to an associated bias value from -100 to 100. Mathematically, the
              bias is added to the logits generated by the model prior to sampling. The exact
              effect will vary per model, but values between -1 and 1 should decrease or
              increase likelihood of selection; values like -100 or 100 should result in a ban
              or exclusive selection of the relevant token.

          logprobs: Whether to return log probabilities of the output tokens or not. If true,
              returns the log probabilities of each output token returned in the `content` of
              `message`.

          max_completion_tokens: An upper bound for the number of tokens that can be generated for a completion,
              including visible output tokens and
              [reasoning tokens](https://platform.openai.com/docs/guides/reasoning).

          max_tokens: The maximum number of [tokens](/tokenizer) that can be generated in the chat
              completion. This value can be used to control
              [costs](https://openai.com/api/pricing/) for text generated via API.

              This value is now deprecated in favor of `max_completion_tokens`, and is not
              compatible with
              [o-series models](https://platform.openai.com/docs/guides/reasoning).

          metadata: Set of 16 key-value pairs that can be attached to an object. This can be useful
              for storing additional information about the object in a structured format, and
              querying for objects via API or the dashboard.

              Keys are strings with a maximum length of 64 characters. Values are strings with
              a maximum length of 512 characters.

          modalities: Output types that you would like the model to generate. Most models are capable
              of generating text, which is the default:

              `["text"]`

              The `gpt-4o-audio-preview` model can also be used to
              [generate audio](https://platform.openai.com/docs/guides/audio). To request that
              this model generate both text and audio responses, you can use:

              `["text", "audio"]`

          n: How many chat completion choices to generate for each input message. Note that
              you will be charged based on the number of generated tokens across all of the
              choices. Keep `n` as `1` to minimize costs.

          parallel_tool_calls: Whether to enable
              [parallel function calling](https://platform.openai.com/docs/guides/function-calling#configuring-parallel-function-calling)
              during tool use.

          prediction: Static predicted output content, such as the content of a text file that is
              being regenerated.

          presence_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens based on
              whether they appear in the text so far, increasing the model's likelihood to
              talk about new topics.

          prompt_cache_key: Used by OpenAI to cache responses for similar requests to optimize your cache
              hit rates. Replaces the `user` field.
              [Learn more](https://platform.openai.com/docs/guides/prompt-caching).

          reasoning_effort: Constrains effort on reasoning for
              [reasoning models](https://platform.openai.com/docs/guides/reasoning). Currently
              supported values are `minimal`, `low`, `medium`, and `high`. Reducing reasoning
              effort can result in faster responses and fewer tokens used on reasoning in a
              response.

          response_format: An object specifying the format that the model must output.

              Setting to `{ "type": "json_schema", "json_schema": {...} }` enables Structured
              Outputs which ensures the model will match your supplied JSON schema. Learn more
              in the
              [Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs).

              Setting to `{ "type": "json_object" }` enables the older JSON mode, which
              ensures the message the model generates is valid JSON. Using `json_schema` is
              preferred for models that support it.

          safety_identifier: A stable identifier used to help detect users of your application that may be
              violating OpenAI's usage policies. The IDs should be a string that uniquely
              identifies each user. We recommend hashing their username or email address, in
              order to avoid sending us any identifying information.
              [Learn more](https://platform.openai.com/docs/guides/safety-best-practices#safety-identifiers).

          seed: This feature is in Beta. If specified, our system will make a best effort to
              sample deterministically, such that repeated requests with the same `seed` and
              parameters should return the same result. Determinism is not guaranteed, and you
              should refer to the `system_fingerprint` response parameter to monitor changes
              in the backend.

          service_tier: Specifies the processing type used for serving the request.

              - If set to 'auto', then the request will be processed with the service tier
                configured in the Project settings. Unless otherwise configured, the Project
                will use 'default'.
              - If set to 'default', then the request will be processed with the standard
                pricing and performance for the selected model.
              - If set to '[flex](https://platform.openai.com/docs/guides/flex-processing)' or
                '[priority](https://openai.com/api-priority-processing/)', then the request
                will be processed with the corresponding service tier.
              - When not set, the default behavior is 'auto'.

              When the `service_tier` parameter is set, the response body will include the
              `service_tier` value based on the processing mode actually used to serve the
              request. This response value may be different from the value set in the
              parameter.

          stop: Not supported with latest reasoning models `o3` and `o4-mini`.

              Up to 4 sequences where the API will stop generating further tokens. The
              returned text will not contain the stop sequence.

          store: Whether or not to store the output of this chat completion request for use in
              our [model distillation](https://platform.openai.com/docs/guides/distillation)
              or [evals](https://platform.openai.com/docs/guides/evals) products.

              Supports text and image inputs. Note: image inputs over 8MB will be dropped.

          stream: If set to true, the model response data will be streamed to the client as it is
              generated using
              [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format).
              See the
              [Streaming section below](https://platform.openai.com/docs/api-reference/chat/streaming)
              for more information, along with the
              [streaming responses](https://platform.openai.com/docs/guides/streaming-responses)
              guide for more information on how to handle the streaming events.

          stream_options: Options for streaming response. Only set this when you set `stream: true`.

          temperature: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
              make the output more random, while lower values like 0.2 will make it more
              focused and deterministic. We generally recommend altering this or `top_p` but
              not both.

          tool_choice: Controls which (if any) tool is called by the model. `none` means the model will
              not call any tool and instead generates a message. `auto` means the model can
              pick between generating a message or calling one or more tools. `required` means
              the model must call one or more tools. Specifying a particular tool via
              `{"type": "function", "function": {"name": "my_function"}}` forces the model to
              call that tool.

              `none` is the default when no tools are present. `auto` is the default if tools
              are present.

          tools: A list of tools the model may call. You can provide either
              [custom tools](https://platform.openai.com/docs/guides/function-calling#custom-tools)
              or [function tools](https://platform.openai.com/docs/guides/function-calling).

          top_logprobs: An integer between 0 and 20 specifying the number of most likely tokens to
              return at each token position, each with an associated log probability.
              `logprobs` must be set to `true` if this parameter is used.

          top_p: An alternative to sampling with temperature, called nucleus sampling, where the
              model considers the results of the tokens with top_p probability mass. So 0.1
              means only the tokens comprising the top 10% probability mass are considered.

              We generally recommend altering this or `temperature` but not both.

          user: This field is being replaced by `safety_identifier` and `prompt_cache_key`. Use
              `prompt_cache_key` instead to maintain caching optimizations. A stable
              identifier for your end-users. Used to boost cache hit rates by better bucketing
              similar requests and to help OpenAI detect and prevent abuse.
              [Learn more](https://platform.openai.com/docs/guides/safety-best-practices#safety-identifiers).

          verbosity: Constrains the verbosity of the model's response. Lower values will result in
              more concise responses, while higher values will result in more verbose
              responses. Currently supported values are `low`, `medium`, and `high`.

          web_search_options: This tool searches the web for relevant results to use in a response. Learn more
              about the
              [web search tool](https://platform.openai.com/docs/guides/tools-web-search?api-mode=chat).

          extra_headers: Send extra headers

          extra_query: Add additional query parameters to the request

          extra_body: Add additional JSON properties to the request

          timeout: Override the client-level default timeout for this request, in seconds
        """  # noqa: E501
        # TODO: handle `n > 1`` that returns multiple completions
        if isinstance(n, int) and n > 1:
            raise ValueError("n > 1 is not supported for async chat completions.")

        response = await self._aclient.chat.completions.create(
            messages=messages,
            model=model,
            audio=audio,
            frequency_penalty=frequency_penalty,
            function_call=function_call,
            functions=functions,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            metadata=metadata,
            modalities=modalities,
            n=n,
            parallel_tool_calls=parallel_tool_calls,
            prediction=prediction,
            presence_penalty=presence_penalty,
            prompt_cache_key=prompt_cache_key,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
            safety_identifier=safety_identifier,
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
            verbosity=verbosity,
            web_search_options=web_search_options,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
        )

        # Update usage statistics
        if hasattr(response, "usage") and response.usage:
            self.update_usage(response.usage, model)

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

    def update_usage(self, usage: CompletionUsage, model_name: str = None) -> None:
        """Updates the internal usage counters with values from a new API response.

        Args:
            usage (CompletionUsage): The usage object returned by the OpenAI API.
            model_name (str, optional): The name of the model for which the usage
                is being updated. If None, cost is copied from usage if available.
        """
        if not hasattr(self, "_usage"):
            self._usage.update(copy.deepcopy(INITIAL_USAGE))  # Ensure a fresh copy

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

        # Update cost
        if model_name is not None:
            pricing = _get_pricing_for_model(self.provider, model_name)
            cost = _calculate_cost(usage, pricing)
            self._usage["cost"]["amount"] += cost
        else:
            # If cost is present in usage, copy it directly
            if "cost" in usage and "amount" in usage["cost"]:
                self._usage["cost"]["amount"] = usage["cost"]["amount"]


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


def _get_pricing_for_model(provider: str, model_name: str) -> dict:
    # Load pricing data (cache this in production)
    prices_path = os.path.join(os.path.dirname(__file__), "model_prices.json")
    with open(prices_path, "r") as f:
        prices = json.load(f)
    provider_data = prices.get(provider, {})
    models_map = provider_data.get("models", {})
    pricing_map = provider_data.get("pricing", {})
    pricing_key = models_map.get(model_name, model_name)
    return pricing_map.get(pricing_key, {})


def _calculate_cost(usage: dict, pricing: dict) -> float:
    input_tokens = usage.get("prompt_tokens", 0)
    cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    uncached_tokens = input_tokens - cached_tokens
    output_tokens = usage.get("completion_tokens", 0)
    cost = 0.0
    if "input" in pricing and pricing["input"] is not None:
        cost += (uncached_tokens * pricing["input"]) / 1_000_000
    if "cached" in pricing and pricing["cached"] is not None:
        cost += (cached_tokens * pricing["cached"]) / 1_000_000
    if "output" in pricing and pricing["output"] is not None:
        cost += (output_tokens * pricing["output"]) / 1_000_000
    return cost
