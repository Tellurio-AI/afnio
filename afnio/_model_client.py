import logging
from typing import Any, Dict, List, Optional

from afnio.logging_config import configure_logging
from afnio.models import ChatCompletionModel
from afnio.models.openai import OpenAI
from afnio.tellurio._eventloop import run_in_background_loop
from afnio.tellurio.client import get_default_client

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


class ModelClientSingleton:
    _instance = None
    _client: ChatCompletionModel = None
    _provider: str = None
    _model: str = None
    _client_args: Dict[str, Any] = {}
    _completion_args: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelClientSingleton, cls).__new__(cls)
        return cls._instance

    def _initialize(
        self,
        provider: str,
        model: str,
        client_args: Optional[Dict[str, Any]] = None,
        completion_args: Optional[Dict[str, Any]] = None,
    ):
        # Re-initialize only if provider or model changes
        if (
            self._client is None
            or provider != self._provider
            or model != self._model
            or client_args != self._client_args
            or completion_args != self._completion_args
        ):
            self._provider = provider
            self._model = model
            self._client_args = client_args or {}
            self._completion_args = completion_args or {}
            self._client = self._create_client(
                provider, model, self._client_args, self._completion_args
            )

    def _create_client(
        self,
        provider: str,
        model_name: str = None,
        client_args: Optional[Dict[str, Any]] = None,
        completion_args: Optional[Dict[str, Any]] = None,
    ) -> ChatCompletionModel:
        if provider == "openai":
            # TODO: Define default values for each provider. Could use configuration
            #       file(s) with settings for each provider and model
            model = OpenAI(**client_args)
            self._rpc_set_model_singleton(
                model_id=model.model_id,
                model_name=model_name,
                client_args=client_args,
                completion_args=completion_args,
            )
            return model
        # Future providers can be added here
        else:
            raise ValueError(f"Unsupported provider: {provider}.")

    def _rpc_set_model_singleton(
        self,
        model_id: str = None,
        model_name: str = None,
        client_args: Optional[Dict[str, Any]] = None,
        completion_args: Optional[Dict[str, Any]] = None,
    ):
        try:
            # Get the singleton websocket client
            _, ws_client = get_default_client()

            payload = {
                "model_id": model_id,
                "model_name": model_name,
                "client_args": client_args,
                "completion_args": completion_args,
            }
            response = run_in_background_loop(
                ws_client.call("set_model_singleton", payload)
            )
            # Check server response
            if (
                response["result"]["model_id"] != model_id
                or response["result"]["completion_args"] != completion_args
            ):
                logger.error(
                    f"Server response mismatch: {response['result']!r} "
                    f"(expected model_id={model_id!r}, "
                    f"completion_args={completion_args!r})"
                )
                raise RuntimeError(
                    "Faild to set model singleton: "
                    "server response does not match the request sent."
                )
            logger.debug(
                f"Model singleton set on server and confirmed: "
                f"model_id={model_id!r}, completion_args={completion_args!r}"
            )
        except Exception:
            logger.exception(
                f"Failed to set model singleton on server: "
                f"model_id={model_id!r}, completion_args={completion_args!r}"
            )
            raise

    def chat(self, messages: List[Dict[str, str]], **override_kwargs):
        if self._client is None:
            raise RuntimeError(
                "Model client is not set. Call `set_backward_model_client` first."
            )

        # Merge completion_args with overrides
        kwargs = {**self._completion_args, **override_kwargs}
        model = kwargs.pop("model", self._model)

        return self._client.chat(messages=messages, model=model, **kwargs)

    def get_provider(self) -> Optional[str]:
        """Returns the model provider name."""
        return self._client.get_provider()

    def get_usage(self) -> Dict[str, int]:
        """
        Retrieves the current token usage statistics.

        Returns:
            Dict[str, int]: A dictionary containing cumulative token usage
                statistics since the model instance was initialized.

        Example:
            >>> model = get_backward_model_client()
            >>> model.get_usage()
            {'prompt_tokens': 1500, 'completion_tokens': 1200, 'total_tokens': 2700}
        """
        return self._client.get_usage()

    def clear_usage(self) -> None:
        """
        Clears the token usage statistics.

        This resets all numerical values in the usage dictionary to zero (including
        nested values), while preserving the dictionary structure.
        """
        self._client.clear_usage()


# Global singleton instance
_model_singleton = ModelClientSingleton()


def set_backward_model_client(
    model_path: str = "openai/gpt-4o",
    client_args: Optional[Dict[str, Any]] = None,
    completion_args: Optional[Dict[str, Any]] = None,
):
    """
    Set the global model client for backward operations.

    Args:
        model_path (str): Path in the format "provider/model_name"
          (e.g., "openai/gpt-4o"). Default: ``"openai/gpt-4o"``.
        client_args (Dict): Arguments to initialize the model client such as:
          - api_key (str): The client API key.
          - organization (str): The organization to bill.
          - base_url (str): The model base endpoint URL (useful when models are
            behind a proxy).
          - etc.
        completion_args (Dict): Arguments to pass to `achat()` during usage such as:
          - model (str): The model to use (e.g., "gpt-4o").
          - temperature (float): Amount of randomness injected into the response.
          - max_completion_tokens (int): Maximum number of tokens to generate.
          - etc.

    .. note::

            For a complete list of supported `client_args` and `completion_args` for
            each model, refer to the respective API documentation.
    """
    try:
        provider, model = model_path.split("/", 1)
    except ValueError:
        raise ValueError("`model_path` must be in the format 'provider/model'")

    _model_singleton._initialize(provider, model, client_args, completion_args)


def get_backward_model_client() -> ModelClientSingleton:
    """
    Retrieve the global model client singleton.

    Raises:
        RuntimeError: If no model client is set globally.

    Returns:
        ModelClientSingleton: The global model client.
    """
    if _model_singleton._client is None:
        raise RuntimeError(
            "No global model client set for backward pass. "
            "Call `set_backward_model_client` to define one."
        )
    return _model_singleton
