import os

import pytest

from afnio.tellurio import login
from afnio.tellurio.client import InvalidAPIKeyError


@pytest.mark.asyncio
async def test_login_success(close_ws_client):
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
