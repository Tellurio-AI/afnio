from .client import get_default_client
from .run import init


def login(api_key: str = None, relogin=False):
    """
    Logs in the user using an API key and verifies its validity.

    This method allows the user to provide an API key or retrieve a stored API key
    from the system. It verifies the API key by calling the backend and securely
    stores it using the `keyring` library if valid.

    Args:
        api_key (str, optional): The user's API key. If not provided, the method
            attempts to retrieve a stored API key from the local system.
        relogin (bool): If True, forces a re-login and requires the user to provide
            a new API key.

    Returns:
        str: A confirmation message if the API key is valid.

    Raises:
        ValueError: If the API key is invalid or not provided during re-login.
    """
    client = get_default_client()  # Use the global accessor for _default_client
    return client.login(api_key=api_key, relogin=relogin)


__all__ = ["init", "login"]

# Please keep this list sorted
assert __all__ == sorted(__all__)
