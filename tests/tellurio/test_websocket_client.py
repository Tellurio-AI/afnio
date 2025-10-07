import asyncio
import os

import pytest
import pytest_asyncio

import afnio.cognitive.functional as F
from afnio._variable import Variable
from afnio.tellurio import login
from afnio.tellurio.run import init
from afnio.tellurio.websocket_client import TellurioWebSocketClient


@pytest.fixture(scope="module", autouse=True)
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


@pytest_asyncio.fixture
async def connected_ws_client():
    """
    Fixture to create and connect a TellurioWebSocketClient instance for testing.
    Ensures the connection is properly closed after the test.
    """
    client = TellurioWebSocketClient(
        base_url=os.getenv(
            "TELLURIO_BACKEND_WS_BASE_URL", "wss://platform.tellurio.ai"
        ),
        port=int(os.getenv("TELLURIO_BACKEND_WS_PORT", 443)),
        default_timeout=10,
    )
    await client.connect(api_key=os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key"))
    try:
        yield client
    finally:
        await client.close()


@pytest.mark.asyncio
class TestTellurioWebSocketClient:

    async def test_connect_success(self, connected_ws_client):
        """
        Test that the WebSocket client successfully connects to the server.
        """
        # Assert that the connection is established
        assert connected_ws_client.connection is not None

        # Assert that the listener task is running
        assert connected_ws_client.listener_task is not None

        # Assert that there are no pending requests initially
        assert len(connected_ws_client.pending) == 0

    async def test_close(self, connected_ws_client: TellurioWebSocketClient):
        """
        Test that the WebSocket client closes the connection and cleans up resources.
        """
        await connected_ws_client.close()

        assert connected_ws_client.connection is None
        assert connected_ws_client.listener_task is None
        assert len(connected_ws_client.pending) == 0

    async def test_connect_invalid_api_key(self):
        """
        Test that the WebSocket client raises an error
        when connecting with an invalid API key.
        """
        client = TellurioWebSocketClient(
            base_url=os.getenv(
                "TELLURIO_BACKEND_WS_BASE_URL", "wss://platform.tellurio.ai"
            ),
            port=int(os.getenv("TELLURIO_BACKEND_WS_PORT", 443)),
            default_timeout=10,
        )

        with pytest.raises(
            RuntimeError,
            match="Failed to connect to WebSocket after multiple attempts.",
        ):
            await client.connect(api_key="invalid_api_key")

    async def test_call_success(self, connected_ws_client: TellurioWebSocketClient):
        """
        Test that the WebSocket client successfully sends a request
        and receives a response.
        """
        response = await connected_ws_client.call(
            "create_variable",
            {
                "data": ["Tellurio", "is", "great!"],
                "role": "text varibale",
                "requires_grad": True,
            },
        )
        assert "variable_id" in response["result"]
        assert response["result"]["message"] == "Variable created successfully."

    async def test_call_invalid_method(
        self, connected_ws_client: TellurioWebSocketClient
    ):
        """
        Test that the WebSocket client handles an invalid method error from the server.
        """
        response = await connected_ws_client.call("invalid_method", {"param": "value"})
        assert response["error"]["code"] == -32601
        assert response["error"]["message"] == "Method 'invalid_method' not found."
        assert response["error"]["data"] == {"method": "invalid_method"}

    async def test_heartbeat_received_during_long_chat_completion(
        self, monkeypatch, model
    ):
        """
        Test that heartbeats are received by the client during a long-running
        chat completion before the final response arrives.
        """
        # Track received heartbeats
        received_heartbeats = []

        # Save the original rpc_heartbeat method
        original_rpc_heartbeat = TellurioWebSocketClient.rpc_heartbeat

        async def counting_rpc_heartbeat(self, params):
            received_heartbeats.append(params)
            await original_rpc_heartbeat(self, params)

        # Monkeypatch the heartbeat handler to count heartbeats
        # but still run original logic
        monkeypatch.setattr(
            TellurioWebSocketClient, "rpc_heartbeat", counting_rpc_heartbeat
        )

        # Prepare chat completion input
        system = Variable(
            data="You are a helpful assistant.",
            role="system instruction",
            requires_grad=True,
        )
        query = Variable(
            data="Tell me a long story.",
            role="user query",
            requires_grad=False,
        )
        messages = [
            {"role": "system", "content": [system]},
            {"role": "user", "content": [query]},
        ]

        # Run chat completion (this should trigger heartbeats)
        result = F.chat_completion(
            model,
            messages,
            inputs={},
            model="gpt-5",
            seed=42,
            reasoning_effort="low",
        )

        # Wait for the result
        if asyncio.iscoroutine(result):
            result = await result

        # Assert that at least one heartbeat was received before the response
        assert len(received_heartbeats) > 0
        # Check the content of the heartbeat
        for hb in received_heartbeats:
            assert "id" in hb

    # TODO: Finalize this method once backend has Celery long running tasks implemented
    # async def test_call_timeout(self, connected_ws_client: TellurioWebSocketClient):
    #     """
    #     Test that the WebSocket client raises a TimeoutError
    #     if no response is received within the timeout period.
    #     """
    #     with pytest.raises(asyncio.TimeoutError):
    #         await connected_ws_client.call("long_running_operation", {}, timeout=0.0)
