from afnio.tellurio._auth import _login
from afnio.tellurio._eventloop import run_in_background_loop
from afnio.tellurio.client import TellurioClient
from afnio.tellurio.websocket_client import TellurioWebSocketClient

# Define the global default client instances
_default_client = None
_default_ws_client = None


def get_default_clients(login=True) -> tuple[TellurioClient, TellurioWebSocketClient]:
    """
    Returns the default TellurioClient and TellurioWebSocketClient instances.

    This function initializes a global instance of the TellurioClient and a global
    instance of TellurioWebSocketClient if they don't already exist and returns them.
    The global instances can be used as a singleton for interacting with the backend.

    Args:
        login (bool): If True, attempts to log in the client automatically using either
            the API key stored in the `TELLURIO_API_KEY` environment variable (if set),
            or the API key stored in the local keyring. If neither is available,
            login will not be performed and API calls may fail until authenticated.

    Returns:
        tuple: A tuple containing the TellurioClient and TellurioWebSocketClient
        instances.
    """
    global _default_client, _default_ws_client

    if _default_client is None:
        # Initialize the default HTTP client
        _default_client = TellurioClient()

    if _default_ws_client is None:
        # Initialize the default WebSocket client
        _default_ws_client = TellurioWebSocketClient()

    # If both are newly created and login is requested, attempt login using stored key
    if (
        login
        and _default_client.api_key is None
        and _default_ws_client.connection is None
    ):
        run_in_background_loop(_login(_default_client, _default_ws_client))

    return _default_client, _default_ws_client
