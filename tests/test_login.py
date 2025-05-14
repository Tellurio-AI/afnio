import os

import pytest

from afnio.tellurio import _close_singleton_ws_client, login
from afnio.tellurio.client import InvalidAPIKeyError, get_default_client


@pytest.mark.asyncio
async def test_login_success():
    """
    Test the login function with real HTTP and WebSocket connections.
    """
    api_key = os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key")

    # Call the login function
    result = login(api_key=api_key)

    # Assert the result contains the expected keys
    assert "email" in result
    assert "username" in result
    assert "session_id" in result


@pytest.mark.asyncio
async def test_login_invalid_api_key():
    """
    Test the login function with an invalid API key.
    This should raise a ValueError.
    """
    # Use an invalid API key for testing
    api_key = "invalid_api_key"

    # Call the login function and assert it raises a ValueError
    with pytest.raises(
        InvalidAPIKeyError, match="Login failed due to invalid API key."
    ):
        login(api_key=api_key)


def test_close_singleton_ws_client_direct():
    """
    Test that the singleton WebSocket client is closed directly.
    """
    _, ws_client = get_default_client()
    assert ws_client is not None

    api_key = os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key")
    login(api_key=api_key)
    assert ws_client.connection is not None

    # Close the singleton WebSocket client
    _close_singleton_ws_client()
    assert ws_client.connection is None
