import os

import pytest

from afnio._model_client import (
    _model_singleton,
    get_backward_model_client,
    set_backward_model_client,
)
from afnio.models.openai import OpenAI
from afnio.tellurio import login
from afnio.tellurio._model_registry import MODEL_REGISTRY
from afnio.tellurio.run import init


@pytest.fixture(autouse=True)
def login_and_ensure_default_run():
    """
    Test the login function with real HTTP and WebSocket connections and
    ensure a default Run exists and is set as active before tests.
    """
    # Log in to the Tellurio service using the API key
    api_key = os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key")
    login(api_key=api_key)

    # Use your test org/project names from env or defaults
    namespace_slug = os.getenv("TEST_ORG_SLUG", "tellurio-test")
    project_display_name = os.getenv("TEST_PROJECT", "Test Project")
    run = init(namespace_slug, project_display_name)
    return run


class TestBackwardModelClientSync:

    def test_set_and_get_backward_model_client(self, monkeypatch):
        """
        Test setting and retrieving the backward model client and its registration.
        """
        # Forcing consent to sharing API keys
        monkeypatch.setenv("ALLOW_API_KEY_SHARING", "true")

        # Setting and getting the backward model client
        client_args = {"api_key": "test-key", "organization": "test-org"}
        completion_args = {"model": "gpt-4o", "temperature": 0.5}
        set_backward_model_client(
            model_path="openai/gpt-4o",
            client_args=client_args,
            completion_args=completion_args,
        )
        client = get_backward_model_client()
        assert client._provider == "openai"
        assert client._model == "gpt-4o"
        assert client._client_args == client_args
        assert client._completion_args == completion_args
        # The model should be an OpenAI instance and registered
        assert isinstance(client._client, OpenAI)
        assert client._client.model_id in MODEL_REGISTRY
        assert MODEL_REGISTRY[client._client.model_id] is client._client

    def test_get_backward_model_client_before_set_raises(self):
        """
        Test that getting the backward model client before setting it raises an error.
        """
        # Ensure singleton is reset
        _model_singleton._client = None
        with pytest.raises(
            RuntimeError,
            match=(
                "No global model client set for backward pass. "
                "Call `set_backward_model_client` to define one."
            ),
        ):
            get_backward_model_client()

    def test_set_backward_model_client_invalid_model_path(self):
        """
        Test that an invalid model_path format raises a ValueError.
        """
        with pytest.raises(
            ValueError, match="`model_path` must be in the format 'provider/model'"
        ):
            set_backward_model_client(model_path="openai-gpt-4o")

    def test_set_backward_model_client_invalid_provider(self):
        """
        Test that an unsupported provider raises a ValueError.
        """
        with pytest.raises(ValueError, match="Unsupported provider: notvalid."):
            set_backward_model_client(model_path="notvalid/gpt-4o")

    def test_set_backward_model_client_twice_reinitializes(self, monkeypatch):
        """
        Test that setting the backward model client twice reinitializes the singleton.
        """
        # Forcing consent to sharing API keys
        monkeypatch.setenv("ALLOW_API_KEY_SHARING", "true")

        # Setting backward model client
        set_backward_model_client(
            model_path="openai/gpt-4o",
            client_args={"api_key": "key1"},
            completion_args={"model": "gpt-4o"},
        )
        first_client = get_backward_model_client()._client
        set_backward_model_client(
            model_path="openai/gpt-3.5-turbo",
            client_args={"api_key": "key2"},
            completion_args={"model": "gpt-3.5-turbo"},
        )
        client = get_backward_model_client()
        assert client._provider == "openai"
        assert client._model == "gpt-3.5-turbo"
        assert client._client_args["api_key"] == "key2"
        assert client._completion_args["model"] == "gpt-3.5-turbo"
        # The model should be a new OpenAI instance
        assert isinstance(client._client, OpenAI)
        assert client._client is not first_client
        assert client._client.model_id in MODEL_REGISTRY
        assert MODEL_REGISTRY[client._client.model_id] is client._client
