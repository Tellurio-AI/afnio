import json
import logging
import os

import httpx
import keyring
from dotenv import load_dotenv

from afnio.logging_config import configure_logging
from afnio.tellurio.utils import get_config_path
from afnio.tellurio.websocket_client import TellurioWebSocketClient

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

# Define the global default client instances
_default_client = None
_default_ws_client = None


def save_username(username):
    """
    Saves the username to a JSON configuration file.
    If the username is different from the one already stored, this function
    updates the 'username' field and clears all other preferences.
    Otherwise, it preserves all existing values.
    """
    config_path = get_config_path()
    # Load existing config if present, otherwise start with empty dict
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError:
                config = {}
    else:
        config = {}

    # If username is different, clear all preferences except username
    if config.get("username") != username:
        config = {"username": username}
    else:
        config["username"] = username

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def load_username():
    """
    Loads the username from a JSON configuration file.
    This function reads the JSON file at the specified path and retrieves the username
    stored in it. If the file does not exist, it returns None.
    """
    config_path = get_config_path()
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            try:
                return json.load(f).get("username")
            except json.JSONDecodeError:
                logger.debug("Failed to decode JSON from the configuration file.")
                return None
    return None


class InvalidAPIKeyError(Exception):
    """Exception raised when the API key is invalid."""

    pass


class TellurioClient:
    """
    A client for interacting with the Tellurio backend.

    This client provides methods for authenticating with the backend, making HTTP
    requests (GET, POST, DELETE), and verifying API keys. It is designed to simplify
    communication with the Tellurio platform.
    """

    def __init__(self, base_url: str = None, port: int = None):
        """
        Initializes the TellurioClient instance.

        Args:
            base_url (str, optional): The base URL of the Tellurio backend. If not
                provided, it defaults to the value of the
                `TELLURIO_BACKEND_HTTP_BASE_URL` environment variable
                or "https://platform.tellurio.ai".
            port (int, optional): The port number for the backend. If not provided,
                it defaults to the value of the `TELLURIO_BACKEND_HTTP_PORT`
                environment variable or 443.
        """
        self.base_url = base_url or os.getenv(
            "TELLURIO_BACKEND_HTTP_BASE_URL", "https://platform.tellurio.ai"
        )
        self.port = port or os.getenv("TELLURIO_BACKEND_HTTP_PORT", 443)
        self.url = f"{self.base_url}:{self.port}"
        self.service_name = os.getenv(
            "KEYRING_SERVICE_NAME", "Tellurio"
        )  # Service name for keyring
        self.api_key = None

    def login(self, api_key: str = None, relogin: bool = False):
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
            dict: A dictionary containing the user's email and username.

        Raises:
            ValueError: If the API key is invalid or not provided during re-login.
        """
        # Use the provided API key if passed, otherwise check for stored key
        if api_key:
            self.api_key = api_key
        elif not relogin:
            username = load_username()
            self.api_key = keyring.get_password(self.service_name, username)
            if self.api_key:
                logger.info("Using stored API key from local keyring.")
            else:
                logger.error("No API key found in local keyring.")
                raise ValueError("API key is required for the first login.")
        else:
            logger.error("No API key provided for re-login.")
            raise ValueError("API key is required for re-login.")

        # Verify the API key
        response_data = self._verify_api_key()
        if response_data:
            email = response_data.get("email", "unknown user")
            username = response_data.get("username", "unknown user")
            logger.debug(f"API key is valid for user '{username}'.")

            # Save the API key securely only if it was provided and is valid
            if api_key:
                keyring.set_password(
                    self.service_name, response_data["username"], self.api_key
                )
                logger.info("API key provided and stored securely in local keyring.")
                save_username(response_data["username"])

            return {
                "email": email,
                "username": username,
            }
        else:
            logger.warning("Invalid API key. Please provide a valid API key.")
            if relogin:
                raise InvalidAPIKeyError("Re-login failed due to invalid API key.")
            raise InvalidAPIKeyError("Login failed due to invalid API key.")

    def get(self, endpoint: str) -> httpx.Response:
        """
        Makes a GET request to the specified endpoint.

        Args:
            endpoint (str): The API endpoint (relative to the base URL).

        Returns:
            httpx.Response: The HTTP response object.
        """
        url = f"{self.url}{endpoint}"
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Accept": "*/*",
        }

        try:
            with httpx.Client() as client:
                response = client.get(url, headers=headers)
            return response
        except httpx.RequestError as e:
            logger.error(f"Network error occurred while making GET request: {e}")
            raise ValueError("Network error occurred. Please check your connection.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise ValueError("An unexpected error occurred. Please try again later.")

    def post(self, endpoint: str, json: dict) -> httpx.Response:
        """
        Makes a POST request to the specified endpoint.

        Args:
            endpoint (str): The API endpoint (relative to the base URL).
            json (dict): The JSON payload to send in the request.

        Returns:
            httpx.Response: The HTTP response object.
        """
        url = f"{self.url}{endpoint}"
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client() as client:
                response = client.post(url, headers=headers, json=json)
            return response
        except httpx.RequestError as e:
            logger.error(f"Network error occurred while making POST request: {e}")
            raise ValueError("Network error occurred. Please check your connection.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise ValueError("An unexpected error occurred. Please try again later.")

    def patch(self, endpoint: str, json: dict) -> httpx.Response:
        """
        Makes a PATCH request to the specified endpoint.

        Args:
            endpoint (str): The API endpoint (relative to the base URL).
            json (dict): The JSON payload to send in the request.

        Returns:
            httpx.Response: The HTTP response object.
        """
        url = f"{self.url}{endpoint}"
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client() as client:
                response = client.patch(url, headers=headers, json=json)
            return response
        except httpx.RequestError as e:
            logger.error(f"Network error occurred while making PATCH request: {e}")
            raise ValueError("Network error occurred. Please check your connection.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise ValueError("An unexpected error occurred. Please try again later.")

    def delete(self, endpoint: str) -> httpx.Response:
        """
        Makes a DELETE request to the specified endpoint.

        Args:
            endpoint (str): The API endpoint (relative to the base URL).

        Returns:
            httpx.Response: The HTTP response object.
        """
        url = f"{self.url}{endpoint}"
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Accept": "*/*",
        }

        try:
            with httpx.Client() as client:
                response = client.delete(url, headers=headers)
            return response
        except httpx.RequestError as e:
            logger.error(f"Network error occurred while making DELETE request: {e}")
            raise ValueError("Network error occurred. Please check your connection.")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise ValueError("An unexpected error occurred. Please try again later.")

    def _verify_api_key(self) -> dict:
        """
        Verifies the validity of the API key
        by calling the /api/v0/verify-api-key/ endpoint.

        Returns:
            dict: A dictionary containing the user's email, username and a message
            indicating if the API key is valid, None otherwise.
        """
        endpoint = "/api/v0/verify-api-key/"
        try:
            response = self.get(endpoint)

            if response.status_code == 200:
                try:
                    data = response.json()
                    logger.debug(f"API key verification successful: {data}")
                    return data
                except ValueError:
                    logger.error("Failed to parse JSON response from backend.")
                    return None
            elif response.status_code == 401:
                logger.warning("API key is invalid or missing.")
            else:
                logger.error(f"Error: {response.status_code} - {response.text}")
        except ValueError as e:
            logger.error(f"Error during API key verification: {e}")
            raise

        return None


def get_default_client() -> tuple[TellurioClient, TellurioWebSocketClient]:
    """
    Returns the global default TellurioClient instance.

    This function initializes a global instance of the TellurioClient if it does not
    already exist and returns it. The global instance can be used as a singleton for
    interacting with the backend.

    Returns:
        TellurioClient: The global default TellurioClient instance.
    """
    global _default_client, _default_ws_client

    if _default_client is None:
        # Initialize the default HTTP client
        _default_client = TellurioClient()

    if _default_ws_client is None:
        # Initialize default WebSocket client
        _default_ws_client = TellurioWebSocketClient()

    return _default_client, _default_ws_client
