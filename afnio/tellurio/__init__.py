from .client import get_default_client
from .run import init


def login(api_key=None, relogin=False):
    """
    Logs in the user using the default TellurioClient instance.

    Args:
        api_key (str, optional): The user's API key. If not provided, it will be
            read from the local system.
        relogin (bool): If True, forces a re-login and requests a new API key.

    Returns:
        str: A confirmation message if the API key is valid.
    """
    client = get_default_client()  # Use the global accessor for _default_client
    return client.login(api_key=api_key, relogin=relogin)


__all__ = ["init", "login"]

# Please keep this list sorted
assert __all__ == sorted(__all__)
