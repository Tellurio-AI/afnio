import atexit
import logging
from typing import Any, Optional

from afnio.logging_config import configure_logging
from afnio.tellurio._auth import _login
from afnio.tellurio._client_manager import get_default_clients
from afnio.tellurio._eventloop import _event_loop_thread, run_in_background_loop
from afnio.tellurio._variable_registry import suppress_variable_notifications
from afnio.tellurio.client import TellurioClient
from afnio.tellurio.run import init
from afnio.tellurio.run_context import get_active_run

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


def login(api_key: str = None, relogin=False):
    """
    Logs in the user and establishes both HTTP and WebSocket connections to the
    Tellurio Studio backend.

    This function authenticates the user using an API key, which can be provided
    directly, via the `TELLURIO_API_KEY` environment variable, or retrieved from the
    local keyring. On successful authentication, it establishes a WebSocket connection
    for real-time communication and stores the API key securely for future use.

    Args:
        api_key (str, optional): The user's API key. If not provided, the function will
            attempt to use the `TELLURIO_API_KEY` environment variable. If that is not
            set, it will look for a stored API key in the local keyring.
        relogin (bool): If True, forces a re-login and requires the user to provide a
            new API key (either directly or via the environment variable).

    Returns:
        dict: A dictionary containing the user's email, username, and session ID for the
        WebSocket connection. Example:
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

    Notes:
        - On first login, the API key is stored securely in the local keyring for
            future use.
        - If relogin is True, a new API key must be provided (directly or via
            environment variable).
        - This function is synchronous and can be called from both scripts and
            interactive environments.
    """
    # Get the default HTTP and WebSocket clients
    # Login in a separate step below to pass parameters
    client, ws_client = get_default_clients(login=False)

    return run_in_background_loop(
        _login(client=client, ws_client=ws_client, api_key=api_key, relogin=relogin)
    )  # Handle both sync and async contexts


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
        _, ws_client = get_default_clients(login=False)
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


__all__ = [
    "configure_logging",
    "init",
    "log",
    "login",
    "suppress_variable_notifications",
]

# Please keep this list sorted
assert __all__ == sorted(__all__)
