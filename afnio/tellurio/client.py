import logging
import os

import httpx
import keyring

from afnio.logging_config import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


class TellurioClient:
    def __init__(self, base_url=None):
        # Use the base_url from environment variables, or default to production
        self.base_url = base_url or os.getenv(
            "TELLURIO_BASE_URL", "https://platform.tellurio.ai"
        )
        self.service_name = "tellurio"  # Service name for keyring
        self.api_key = None

    def login(self, api_key=None, relogin=False):
        """
        Logs in the user using an API key and verifies its validity.

        Args:
            api_key (str, optional): The user's API key. If not provided, it will be
                read from the local system.
            relogin (bool): If True, forces a re-login and requests a new API key.

        Returns:
            str: A confirmation message if the API key is valid.
        """
        # Use the provided API key if passed, otherwise check for stored key
        if api_key:
            self.api_key = api_key
            # Save the API key securely using keyring
            keyring.set_password(self.service_name, "api_key", self.api_key)
            logger.info("API key provided and stored securely.")
        elif not relogin:
            self.api_key = keyring.get_password(self.service_name, "api_key")
            if self.api_key:
                logger.info("Using stored API key from keyring.")
            else:
                raise ValueError("API key is required for the first login.")
        else:
            raise ValueError("API key is required for re-login.")

        # Verify the API key
        response_data = self._verify_api_key()
        if response_data:
            email = response_data.get("email", "unknown user")
            logger.info(f"API key is valid for user '{email}'.")
            return f"API key is valid for user '{email}'."
        else:
            logger.warning("Invalid API key. Please provide a valid API key.")
            if relogin:
                raise ValueError("Re-login failed due to invalid API key.")
            return None

    def _verify_api_key(self):
        """
        Verifies the validity of the API key
        by calling the /api/v0/verify-api-key/ endpoint.

        Returns:
            dict: A dictionary containing the email and message if the API key is valid,
                None otherwise.
        """
        endpoint = f"{self.base_url}/api/v0/verify-api-key/"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "*/*",
        }
        with httpx.Client() as client:
            response = client.get(endpoint, headers=headers)

        if response.status_code == 200:
            try:
                data = response.json()
                logger.info(f"API key verification successful: {data}")
                return data
            except ValueError:
                logger.error("Failed to parse JSON response from backend.")
                return None
        elif response.status_code == 401:
            logger.warning("API key is invalid or missing.")
        else:
            logger.error(f"Error: {response.status_code} - {response.text}")

        return None
