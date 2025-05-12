import os

import pytest
from keyring import set_keyring

from afnio.tellurio.client import InvalidAPIKeyError, TellurioClient
from tests.utils import InMemoryKeyring


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
        assert response == "test@tellurio.ai"

        # Assert that the API key was stored in the test keyring
        stored_api_key = self.test_keyring.get_password("tellurio", "api_key")
        assert stored_api_key == api_key

    def test_login_invalid_api_key(self):
        """
        Test that an invalid API key returns None and does not log in.
        """
        client = TellurioClient()

        with pytest.raises(
            InvalidAPIKeyError, match="Login failed due to invalid API key."
        ):
            client.login(api_key="invalid_api_key")

        with pytest.raises(
            InvalidAPIKeyError, match="Re-login failed due to invalid API key."
        ):
            client.login(api_key="invalid_api_key", relogin=True)

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
        assert response == "test@tellurio.ai"

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
        assert response == "test@tellurio.ai"

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

    def test_invalid_api_key_not_stored(self):
        """
        Test that an invalid API key is not stored in the keyring.
        """
        client = TellurioClient()

        # Attempt to log in with an invalid API key
        with pytest.raises(
            InvalidAPIKeyError, match="Login failed due to invalid API key."
        ):
            client.login(api_key="invalid_api_key")

        # Assert that the invalid API key was not stored in the keyring
        stored_api_key = self.test_keyring.get_password("tellurio", "api_key")
        assert stored_api_key is None
