import asyncio
import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from afnio.models.openai import AsyncOpenAI
from afnio.tellurio import login
from afnio.tellurio._model_registry import MODEL_REGISTRY
from afnio.tellurio.run import init
from afnio.tellurio.websocket_client import TellurioWebSocketClient


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


@pytest.fixture
def model(monkeypatch):
    """
    Fixture to create an LM model instance.
    """
    # Forcing consent to sharing API keys
    monkeypatch.setenv("ALLOW_API_KEY_SHARING", "true")

    # Create OpenAI model client
    model = AsyncOpenAI(api_key="1234567890", organization="test-org")
    # Ensure model is registered
    assert model.model_id in MODEL_REGISTRY
    return model


class TestClientToServerModelSync:
    def test_create_model(self, model):
        """
        Test that creating a model registers it in the MODEL_REGISTRY
        and assigns a model_id.
        """
        assert model.model_id is not None
        assert model.model_id in MODEL_REGISTRY
        assert MODEL_REGISTRY[model.model_id] is model

    def test_missing_api_key_raises(self, monkeypatch):
        """
        Test that not passing an api_key and not setting the OPENAI_API_KEY env variable
        raises an exception.
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(Exception):
            AsyncOpenAI()


class TestServerToClientModelSync:

    @pytest.fixture
    def mock_server_update_model_request(self):
        """
        Fixture to simulate receiving an 'update_model' RPC call from the server.

        Usage:
            mock_server_update_model_request(model_id, field, value)
        """

        def _mock(model_id, field, value):

            # Compose the JSON-RPC message
            message = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "update_model",
                    "params": {
                        "model_id": model_id,
                        "field": field,
                        "value": value,
                    },
                    "id": "test-id-123",
                }
            )

            # Patch the connection to use a fake async send
            class FakeConnection:
                def __init__(self):
                    self.sent_messages = []
                    self._closed = False

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if hasattr(self, "_sent"):
                        self._closed = True
                        raise StopAsyncIteration
                    self._sent = True
                    return message

                async def send(self, msg):
                    self.sent_messages.append(msg)

                @property
                def closed(self):
                    return self._closed

            # Create a fresh client instance (no singleton)
            ws_client = TellurioWebSocketClient()
            ws_client.connection = FakeConnection()

            # Patch send to track calls
            with patch.object(
                ws_client.connection, "send", new_callable=AsyncMock
            ) as mock_send:
                loop = asyncio.get_event_loop()
                listener_task = loop.create_task(ws_client._listener())
                loop.run_until_complete(asyncio.sleep(0.1))
                listener_task.cancel()
                try:
                    loop.run_until_complete(listener_task)
                except asyncio.CancelledError:
                    pass

                # Return the mock so the test can assert on it
                return mock_send

        return _mock

    @staticmethod
    def assert_valid_update_model_response(send_mock):
        """
        Assert that the client sent a valid JSON-RPC response
        to an update_model request.
        """
        send_mock.assert_awaited()
        sent_msg = send_mock.call_args[0][0]
        response = json.loads(sent_msg)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-id-123"
        assert response["result"]["message"] == "Ok"

    def test_set_update_usage_reflected_in_client(
        self, model, mock_server_update_model_request
    ):
        """
        Test that a server's update to a model's _usage attribute
        is reflected in the client.
        """
        # Simulate server updating the _usage field
        send_mock = mock_server_update_model_request(
            model.model_id, "_usage", {"prompt_tokens": 42, "completion_tokens": 7}
        )

        # Assert that the model was updated locally
        assert hasattr(model, "_usage")
        assert model._usage == {"prompt_tokens": 42, "completion_tokens": 7}

        # Assert that the client sends the correct response to the server
        self.assert_valid_update_model_response(send_mock)

    def test_multiple_set_fields_reflected_in_client(
        self, model, mock_server_update_model_request
    ):
        """
        Test that multiple server updates to a model's attribute
        are propagated to the client in the correct order.
        """
        # Simulate a sequence of server updates to the _usage field
        usages = [
            {"prompt_tokens": 10, "completion_tokens": 5},
            {"prompt_tokens": 20, "completion_tokens": 10},
            {"prompt_tokens": 30, "completion_tokens": 15},
            {"prompt_tokens": 40, "completion_tokens": 20},
            {"prompt_tokens": 50, "completion_tokens": 25},
        ]
        for usage in usages:
            send_mock = mock_server_update_model_request(
                model.model_id, "_usage", usage
            )
            # Assert that the model was updated locally
            assert hasattr(model, "_usage")
            assert model._usage == usage
            # Assert that the client sends the correct response to the server
            self.assert_valid_update_model_response(send_mock)

    def test_update_any_field_reflected_in_client(
        self, model, mock_server_update_model_request
    ):
        """
        Test that a server's update to any model attribute is reflected in the client.
        """
        # Simulate server updating a custom field
        send_mock = mock_server_update_model_request(
            model.model_id, "custom_field", 12345
        )

        # Assert that the model was updated locally
        assert hasattr(model, "custom_field")
        assert getattr(model, "custom_field") == 12345

        # Assert that the client sends the correct response to the server
        self.assert_valid_update_model_response(send_mock)

    def test_update_model_raises_if_model_missing(
        self, mock_server_update_model_request
    ):
        """
        Test that updating a model that does not exist in the registry raises an error.
        """
        non_existent_model_id = "not-in-registry"
        send_mock = mock_server_update_model_request(
            non_existent_model_id, "data", "new-value"
        )
        send_mock.assert_awaited()
        sent_msg = send_mock.call_args[0][0]
        response = json.loads(sent_msg)
        assert "error" in response
        assert response["error"]["message"] == "Internal error"
        assert response["error"]["data"]["exception"] == (
            "Failed to update model with ID 'not-in-registry': "
            "Model with id 'not-in-registry' not found in registry."
        )
