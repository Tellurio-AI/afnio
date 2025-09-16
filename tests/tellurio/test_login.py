import multiprocessing
import os

import pytest

from afnio.tellurio import _close_singleton_ws_client, login
from afnio.tellurio._client_manager import get_default_clients
from afnio.tellurio.client import InvalidAPIKeyError
from afnio.tellurio.run import init


@pytest.mark.asyncio
async def test_login_valid_api_key(setup_keyring):
    """
    Test the login function with a valid `api_key` parameter.
    """
    api_key = os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key")

    # Call the login function
    result = login(api_key=api_key)

    # Assert the result contains the expected keys
    assert "email" in result
    assert "username" in result
    assert "session_id" in result


@pytest.mark.asyncio
async def test_login_invalid_api_key(setup_keyring):
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


def test_login_with_valid_env_var(monkeypatch, setup_keyring):
    """
    Test that login works when the API key is provided via the TELLURIO_API_KEY
    environment variable.
    """
    api_key = os.environ["TEST_ACCOUNT_API_KEY"]
    monkeypatch.setenv("TELLURIO_API_KEY", api_key)
    # Do not pass api_key argument, so only env var is used
    result = login()
    assert "email" in result
    assert "username" in result
    assert "session_id" in result


def test_api_key_login_stores_keyring(setup_keyring):
    """
    Test that a successful login via api_key parameter stores the key in the keyring.
    """
    api_key = os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key")
    service = os.getenv("KEYRING_SERVICE_NAME", "Tellurio")

    result = login(api_key=api_key)
    # Now the key should be stored in the in-memory keyring
    username = result["username"]
    stored_api_key = setup_keyring.get_password(service, username)
    assert stored_api_key == api_key


def test_env_var_login_stores_keyring(monkeypatch, setup_keyring):
    """
    Test that a successful login via TELLURIO_API_KEY stores the key in the keyring.
    """
    api_key = os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key")
    service = os.getenv("KEYRING_SERVICE_NAME", "Tellurio")

    api_key = os.environ["TEST_ACCOUNT_API_KEY"]
    monkeypatch.setenv("TELLURIO_API_KEY", api_key)
    result = login()
    # Now the key should be stored in the in-memory keyring
    username = result["username"]
    stored_api_key = setup_keyring.get_password(service, username)
    assert stored_api_key == api_key


def test_close_singleton_ws_client_direct(setup_keyring):
    """
    Test that the singleton WebSocket client is closed directly.
    """
    _, ws_client = get_default_clients()
    assert ws_client is not None

    api_key = os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key")
    login(api_key=api_key)
    assert ws_client.connection is not None

    # Close the singleton WebSocket client
    _close_singleton_ws_client()
    assert ws_client.connection is None


def test_implicit_login_with_stored_key(setup_keyring):
    """
    Test that after first login, a new process can create a Run
    and logs in implicitly using the stored API key in the keyring.
    """
    api_key = os.environ["TEST_ACCOUNT_API_KEY"]
    namespace = os.getenv("TEST_ORG_SLUG", "tellurio-test")
    project = os.getenv("TEST_PROJECT", "Test-Project")

    # First login: store the API key in the keyring
    login(api_key=api_key, relogin=True)

    # Function to run in a new process
    def run_in_subprocess(queue):
        try:
            # Do NOT call login() here!
            run = init(namespace, project)
            # If we get here, implicit login worked
            queue.put(run is not None)
        except Exception as e:
            queue.put(str(e))

    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=run_in_subprocess, args=(queue,))
    p.start()
    p.join(timeout=30)
    assert p.exitcode == 0, "Subprocess did not exit cleanly"
    result = queue.get(timeout=5)
    assert result is True, f"Implicit login failed or run creation failed: {result}"
