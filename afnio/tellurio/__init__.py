import asyncio
import logging

from afnio.logging_config import configure_logging
from afnio.tellurio.websocket_client import TellurioWebSocketClient

from .client import InvalidAPIKeyError, get_default_client
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
        dict: A dictionary containing the email used for login and the session ID
            for the WebSocket connection.
            Example: {"email": "user@example.com", "session_id": "1234567890"}

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
            logger.info(f"HTTP login successful for user '{login_info['username']}'.")

            # Perform WebSocket login
            ws_info = await ws_client.connect(api_key=client.api_key)
            logger.info(
                f"WebSocket connection established "
                f"with session ID '{ws_info['session_id']}'."
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

    return _run_in_context(_login())  # Handle both sync and async contexts


def _run_in_context(coro):
    """
    Runs a coroutine in the appropriate context (sync or async).
    Handles event loop issues (e.g., Jupyter or nested loops) automatically.

    Args:
        coro (coroutine): The coroutine to execute.

    Returns:
        The gathered responses from the coroutine.
    """
    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        pass  # No event loop running

    if loop and loop.is_running():
        # Running inside an existing event loop (e.g., Jupyter Notebook)
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        # No event loop is running, start a new one
        return asyncio.run(coro)


__all__ = ["init", "login"]

# Please keep this list sorted
assert __all__ == sorted(__all__)
