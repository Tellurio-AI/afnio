import atexit
import logging
import os
from typing import Any, Optional

from afnio.logging_config import configure_logging
from afnio.tellurio._eventloop import _event_loop_thread, run_in_background_loop
from afnio.tellurio.run_context import get_active_run
from afnio.tellurio.websocket_client import TellurioWebSocketClient

from .client import InvalidAPIKeyError, TellurioClient, get_default_client
from .run import init

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


def login(api_key: str = None, relogin=False):
    """
    Logs in the user using an API key and verifies its validity.

    This method allows the user to provide an API key or retrieve a stored API key
    from the system. It verifies the API key by calling the backend and securely
    stores it using the `keyring` library if valid. It also establishes a WebSocket
    connection for further communication.

    Args:
        api_key (str, optional): The user's API key. If not provided, the method
            attempts to retrieve a stored API key from the local system.
        relogin (bool): If True, forces a re-login and requires the user to provide
            a new API key.

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
        ValueError: If the API key is invalid or not provided during re-login.
    """

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

    async def _login():
        # Get the default HTTP and WebSocket clients
        client, ws_client = get_default_client()

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

    return run_in_background_loop(_login())  # Handle both sync and async contexts


def log(
    name: str,
    value: Any,
    step: Optional[int] = None,
    client: Optional[TellurioClient] = None,
):
    """
    Log a metric to the active run.

    Args:
            name (str): Name of the metric.
            value (Any): Value of the metric. Can be any type that is JSON serializable.
            step (int, optional): Step number. If not provided, the backend will
                auto-compute it.
            client (TellurioClient, optional): The client to use for the request.
    """
    run = get_active_run()
    run.log(name=name, value=value, step=step, client=client)


def _close_singleton_ws_client():
    """
    Closes the singleton WebSocket client if it exists.
    This function is registered to be called at interpreter shutdown to ensure
    that the WebSocket connection is properly closed and resources are cleaned up.
    """
    try:
        _, ws_client = get_default_client()
        if ws_client and ws_client.connection:
            run_in_background_loop(ws_client.close())
    except Exception as e:
        logger.error(f"Error closing WebSocket client: {e}")
        pass  # Avoid raising errors at interpreter shutdown


def _shutdown():
    """
    Closes the singleton WebSocket client and shuts down the event loop thread.
    Registered with atexit to ensure proper cleanup at interpreter shutdown.
    """
    try:
        _close_singleton_ws_client()
    finally:
        _event_loop_thread.shutdown()


atexit.register(_shutdown)


__all__ = ["configure_logging", "init", "log", "login"]

# Please keep this list sorted
assert __all__ == sorted(__all__)
