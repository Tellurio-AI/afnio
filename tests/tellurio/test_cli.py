import os

import pytest
from click.testing import CliRunner
from keyring import set_keyring

from afnio.tellurio.cli import cli
from tests.utils import InMemoryKeyring


@pytest.fixture(autouse=True)
def setup_keyring():
    """
    Fixture to use the in-memory keyring backend for tests.
    Ensures tests do not interact with the real keyring.
    """
    test_keyring = InMemoryKeyring()
    set_keyring(test_keyring)
    yield test_keyring


@pytest.mark.asyncio
async def test_login_success(setup_keyring):
    """
    Test the CLI login command with a valid API key.
    """
    api_key = os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key")
    username = os.getenv("TEST_USER_USERNAME", "TestUser")
    service = os.getenv("KEYRING_SERVICE_NAME", "Tellurio")

    runner = CliRunner()
    result = runner.invoke(cli, ["login"], input=f"{api_key}\n")
    assert result.exit_code == 0
    assert f"Currently logged in as {username!r}" in result.output
    assert "API key provided and stored securely in local keyring." in result.output

    # Verify the API key is stored in the in-memory keyring
    stored_api_key = setup_keyring.get_password(service, username)
    assert stored_api_key == api_key


def test_login_invalid_api_key(setup_keyring):
    """
    Test the CLI login command with an invalid API key.
    """
    username = os.getenv("TEST_USER_USERNAME", "TestUser")
    service = os.getenv("KEYRING_SERVICE_NAME", "Tellurio")

    runner = CliRunner()
    result = runner.invoke(cli, ["login"], input="invalid_api_key\n")
    assert result.exit_code == 0
    assert "Login failed: Invalid API key. Please try again." in result.output

    # Verify the invalid API key is not stored in the in-memory keyring
    stored_api_key = setup_keyring.get_password(service, username)
    assert stored_api_key is None


def test_login_relogin(setup_keyring):
    """
    Test the CLI login command with the --relogin option.
    """
    username = os.getenv("TEST_USER_USERNAME", "TestUser")
    service = os.getenv("KEYRING_SERVICE_NAME", "Tellurio")

    # Simulate a stored API key in the in-memory keyring
    setup_keyring.set_password("tellurio", "api_key", "old_api_key")

    api_key = os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key")

    runner = CliRunner()
    result = runner.invoke(cli, ["login", "--relogin"], input=f"{api_key}\n")
    assert result.exit_code == 0
    assert "API key provided and stored securely in local keyring." in result.output
    assert f"Currently logged in as {username!r}" in result.output

    # Verify the new API key is stored in the in-memory keyring
    stored_api_key = setup_keyring.get_password(service, username)
    assert stored_api_key == api_key
