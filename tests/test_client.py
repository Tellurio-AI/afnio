import os

import pytest
from dotenv import load_dotenv
from keyring import set_keyring
from keyring.backend import KeyringBackend

from afnio.tellurio.client import TellurioClient

# Load environment variables from .env
load_dotenv()


class InMemoryKeyring(KeyringBackend):
    """
    A simple in-memory keyring backend for testing purposes.
    """

    priority = 1  # High priority to ensure it is used during tests

    def __init__(self):
        self.store = {}

    def get_password(self, service, username):
        return self.store.get((service, username))

    def set_password(self, service, username, password):
        self.store[(service, username)] = password

    def delete_password(self, service, username):
        if (service, username) in self.store:
            del self.store[(service, username)]
        else:
            raise KeyError("No such password")


class TestTellurioClient:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """
        Fixture to use the in-memory keyring backend for tests.
        Ensures tests do not interact with the real keyring.
        """
        self.test_keyring = InMemoryKeyring()
        set_keyring(self.test_keyring)
        yield

    def test_login_success(self):
        """
        Test that a valid API key logs in successfully and is stored in the keyring.
        """
        api_key = os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key")
        client = TellurioClient()
        response = client.login(api_key=api_key)

        # Assert that the response is a success message
        assert response == "API key is valid for user 'test@tellurio.ai'."

        # Assert that the API key was stored in the test keyring
        stored_api_key = self.test_keyring.get_password("tellurio", "api_key")
        assert stored_api_key == api_key

    def test_login_invalid_api_key(self):
        """
        Test that an invalid API key returns None and does not log in.
        """
        client = TellurioClient()
        response = client.login(api_key="invalid_api_key")

        # Assert that the response is None for an invalid API key
        assert response is None

    def test_login_stored_api_key(self):
        """
        Test that a stored API key in the keyring is used for login.
        """
        api_key = os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key")

        # Simulate a stored API key in the test keyring
        self.test_keyring.set_password("tellurio", "api_key", api_key)

        # Use the temporary path for testing
        client = TellurioClient()
        response = client.login()

        # Assert that the response is a success message
        assert response == "API key is valid for user 'test@tellurio.ai'."

    def test_login_relogin_with_new_api_key(self):
        """
        Test that re-login with a new API key replaces the old key in the keyring.
        """
        old_api_key = "old_api_key"
        new_api_key = os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key")

        # Simulate a stored old API key in the test keyring
        self.test_keyring.set_password("tellurio", "api_key", old_api_key)

        client = TellurioClient()
        response = client.login(api_key=new_api_key, relogin=True)

        # Assert that the response is a success message
        assert response == "API key is valid for user 'test@tellurio.ai'."

        # Assert that the new API key was stored in the test keyring
        stored_api_key = self.test_keyring.get_password("tellurio", "api_key")
        assert stored_api_key == new_api_key

    def test_login_relogin_with_missing_api_key(self):
        """
        Test that re-login without providing a new API key raises a ValueError.
        """
        # Simulate a stored old API key in the test keyring
        self.test_keyring.set_password("tellurio", "api_key", "old_api_key")

        client = TellurioClient()

        # Attempt re-login without providing a new API key
        with pytest.raises(ValueError, match="API key is required for re-login."):
            client.login(relogin=True)
