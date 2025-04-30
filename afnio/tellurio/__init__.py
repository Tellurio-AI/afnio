from .client import TellurioClient

# Create a default client instance
_default_client = TellurioClient()


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
    return _default_client.login(api_key=api_key, relogin=relogin)
