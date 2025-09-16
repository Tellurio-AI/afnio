import logging
import os

from afnio.logging_config import configure_logging
from afnio.tellurio.websocket_client import TellurioWebSocketClient

from .client import InvalidAPIKeyError, TellurioClient

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


async def _close_ws_connection(ws_client: TellurioWebSocketClient, reason: str):
    """
    Closes the WebSocket connection and logs the reason.

    Args:
        ws_client (TellurioWebSocketClient): The WebSocket client instance.
        reason (str): The reason for closing the connection.
    """
    if ws_client.connection:
        await ws_client.close()
        logger.info(f"WebSocket connection closed due to {reason}.")


async def _login(
    client: TellurioClient,
    ws_client: TellurioWebSocketClient,
    api_key: str = None,
    relogin=False,
):
    """
    Asynchronously logs in the user using the provided HTTP and WebSocket clients.

    This internal coroutine performs both HTTP and WebSocket authentication. It uses
    the provided API key (or retrieves one from the client if not supplied), verifies
    the key with the backend, and establishes a WebSocket connection for real-time
    communication. If authentication fails at any step, the WebSocket connection is
    closed and an appropriate error is raised.

    Args:
        client (TellurioClient): The HTTP client instance to use for authentication.
        ws_client (TellurioWebSocketClient): The WebSocket client instance to connect.
        api_key (str, optional): The user's API key. If not provided, the client
            should already have a valid API key set.
        relogin (bool): If True, forces a re-login and requires a new API key.

    Returns:
        dict: A dictionary containing the user's email, username, and session ID
        for the WebSocket connection.
        Example:
            {
                "email": "user@example.com",
                "username": "user123",
                "session_id": "abc123xyz"
            }

    Raises:
        ValueError: If the API key is not provided during first login or re-login.
        InvalidAPIKeyError: If the backend rejects the API key.
        RuntimeError: If the WebSocket connection fails.
        Exception: For any other unexpected errors during login.
    """
    try:
        # Perform HTTP login
        login_info = client.login(api_key=api_key, relogin=relogin)
        logger.debug(f"HTTP login successful for user '{login_info['username']}'.")

        # Perform WebSocket login
        ws_info = await ws_client.connect(api_key=client.api_key)
        logger.debug(
            f"WebSocket connection established "
            f"with session ID '{ws_info['session_id']}'."
        )

        base_url = os.getenv(
            "TELLURIO_BACKEND_HTTP_BASE_URL", "https://platform.tellurio.ai"
        )
        logger.info(
            "Currently logged in as %r to %r. "
            "Use `afnio login --relogin` to force relogin.",
            login_info["username"],
            base_url,
            extra={"colors": {0: "yellow", 1: "green"}},
        )

        return {
            "email": login_info.get("email"),
            "username": login_info.get("username"),
            "session_id": ws_info.get("session_id"),
        }
    except ValueError as e:
        logger.error(f"HTTP login failed: {e}")
        await _close_ws_connection(ws_client, "missing API key")
        raise
    except InvalidAPIKeyError as e:
        logger.error(f"HTTP login failed: {e}")
        await _close_ws_connection(ws_client, "invalid API key")
        raise
    except RuntimeError as e:
        logger.error(f"WebSocket connection error: {e}")
        await _close_ws_connection(ws_client, "runtime error")
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        await _close_ws_connection(ws_client, "an unexpected error")
        raise
