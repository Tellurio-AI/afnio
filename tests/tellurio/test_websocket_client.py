import os

import pytest
import pytest_asyncio

from afnio.tellurio import login
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

    # TODO: Finalize this method once backend has Celery long running tasks implemented
    # async def test_call_timeout(self, connected_ws_client: TellurioWebSocketClient):
    #     """
    #     Test that the WebSocket client raises a TimeoutError
    #     if no response is received within the timeout period.
    #     """
    #     with pytest.raises(asyncio.TimeoutError):
    #         await connected_ws_client.call("long_running_operation", {}, timeout=0.0)
