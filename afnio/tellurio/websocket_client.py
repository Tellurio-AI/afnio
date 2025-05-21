import asyncio
import json
import logging
import os
import uuid

import websockets
from websockets.exceptions import ConnectionClosed

from afnio.logging_config import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


class TellurioWebSocketClient:
    """
    A WebSocket client for interacting with the Tellurio backend.

    This client establishes a WebSocket connection to the backend, sends requests,
    listens for responses, and handles reconnections. It supports JSON-RPC-style
    communication and is designed to work with asynchronous workflows.
    """

    def __init__(
        self,
        base_url: str = None,
        port: int = None,
        default_timeout: int = 60,
    ):
        """
        Initializes the WebSocket client.

        Args:
            base_url (str): The base URL of the Tellurio backend
              (e.g., "https://platform.tellurio.ai").
            default_timeout (int): The default timeout (in seconds)
              for WebSocket requests.
        """
        self.base_url = base_url or os.getenv(
            "TELLURIO_BACKEND_WS_BASE_URL", "wss://platform.tellurio.ai"
        )
        self.port = port or os.getenv("TELLURIO_BACKEND_WS_PORT", 443)
        self.api_key = None
        self.default_timeout = default_timeout
        self.ws_url = self._build_ws_url(self.base_url, self.port)
        self.connection: websockets.ClientConnection = None
        self.listener_task = None
        self.pending = {}  # req_id → Future

    def _build_ws_url(self, base_url, port):
        """
        Constructs the WebSocket URL from the base URL and port.

        Args:
            base_url (str): The base URL of the Tellurio backend
              (e.g., "wss://platform.tellurio.ai").
            port (int): The port number for the backend.

        Returns:
            str: The WebSocket URL (e.g., "wss://platform.tellurio.ai/ws/v0/rpc/").
        """
        return f"{base_url}:{port}/ws/v0/rpc/"

    async def connect(self, api_key: str = None, retries: int = 3, delay: int = 5):
        """
        Connects to the WebSocket server with retry logic.

        Attempts to establish a WebSocket connection to the backend. If the connection
        fails, it retries up to the specified number of attempts with a delay between
        each attempt.

        Args:
            api_key (str): The API key for authenticating with the backend.
            retries (int): The number of reconnection attempts (default: 3).
            delay (int): The delay (in seconds) between reconnection attempts
              (default: 5).

        Returns:
            str: The session ID received from the server upon successful connection.

        Raises:
            RuntimeError: If the connection fails after all retry attempts.
        """
        self.api_key = api_key

        headers = {"Authorization": f"Api-Key {self.api_key}"}
        for attempt in range(retries):
            try:
                logger.info(
                    f"Connecting to WebSocket at {self.ws_url} "
                    f"(attempt {attempt + 1}/{retries})"
                )
                self.connection = await websockets.connect(
                    self.ws_url, additional_headers=headers
                )

                # Start the listener task
                self.listener_task = asyncio.create_task(self._listener())
                logger.info("WebSocket connection established.")

                # Example: Retrieve session ID from the server
                response = await self.connection.recv()
                response_data = json.loads(response)
                session_id = response_data.get("result", {}).get("session_id")
                return {"session_id": session_id}
            except Exception as e:
                logger.error(f"Failed to connect to WebSocket: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                else:
                    raise RuntimeError(
                        "Failed to connect to WebSocket after multiple attempts."
                    )

    async def _listener(self):
        """
        Listens for incoming WebSocket messages and resolves pending requests.

        This method runs as a background task and continuously listens for messages
        from the WebSocket server. It matches responses to their corresponding requests
        using the `req_id` and resolves the associated futures.

        Raises:
            ConnectionClosed: If the WebSocket connection is closed.
            Exception: For any unexpected errors during message processing.
        """
        try:
            async for message in self.connection:
                logger.debug(f"Received message: {message}")
                try:
                    data = json.loads(message)
                    req_id = data.get("id")
                    if req_id:
                        future = self.pending.pop(req_id, None)
                        if future:
                            # Handle both success and error responses
                            if "error" in data:
                                future.set_result(data)  # Pass full error response
                            elif "result" in data:
                                future.set_result(data)  # Pass full success response
                            else:
                                logger.warning(f"Unexpected response format: {data}")
                                future.set_exception(
                                    ValueError(f"Unexpected response format: {data}")
                                )
                        else:
                            logger.warning(f"Unexpected message or missing ID: {data}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON message: {e}")
        except ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
            await asyncio.sleep(1)  # Add a delay before reconnecting
            await self.connect()  # Attempt to reconnect
        except Exception as e:
            logger.error(f"Unexpected error in listener: {e}")

    async def call(self, method: str, params: dict, timeout=None) -> dict:
        """
        Sends a request over the WebSocket connection and waits for a response.

        Constructs a JSON-RPC request, sends it to the WebSocket server, and waits
        for the corresponding response. If no response is received within the timeout
        period, a `TimeoutError` is raised.

        Args:
            method (str): The name of the method to call on the backend.
            params (dict): The parameters to pass to the method.
            timeout (int, optional): The timeout (in seconds) for the response.
                If not provided, the default timeout is used.

        Returns:
            dict: The result of the method call.

        Raises:
            RuntimeError: If the WebSocket connection is not established.
            asyncio.TimeoutError: If the response is not received within
              the timeout period.
        """
        timeout = timeout or self.default_timeout  # Use default timeout if not provided

        if not self.connection:
            raise RuntimeError("WebSocket is not connected")

        req_id = str(uuid.uuid4()) if timeout else None
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        if req_id:
            request["id"] = req_id

        # Send request and wait for matching response
        await self.connection.send(json.dumps(request))
        logger.debug(f"Sent RPC request: {request}")

        # If it's a notification (no `id`), return immediately
        if not req_id:
            return None

        # Wait for response
        future = asyncio.get_event_loop().create_future()
        self.pending[req_id] = future
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"Request timed out: {request}")
            self.pending.pop(req_id, None)
            raise

    async def close(self):
        """
        Closes the WebSocket connection and cleans up resources.

        Cancels the listener task, clears pending requests, and closes the WebSocket
        connection.
        """
        if self.listener_task:
            logger.info("Canceling listener task...")
            self.listener_task.cancel()
            try:
                await self.listener_task  # Wait for the listener task to finish
            except asyncio.CancelledError:
                logger.info("Listener task canceled.")
                pass  # Ignore cancellation errors
        self.listener_task = None  # Clean up the listener task

        if self.connection:
            logger.info("Closing WebSocket connection...")
            try:
                await self.connection.close()
            finally:
                self.connection = None

        logger.info("Clearing pending requests...")
        self._cancel_pending_requests()  # Clear pending requests

        logger.info("WebSocket connection closed.")

    async def __aenter__(self):
        """
        Asynchronous context manager entry.

        Establishes the WebSocket connection when entering the context.
        If the connection is already established, it ensures the connection is active.

        Returns:
            TellurioWebSocketClient: The WebSocket client instance.
        """
        if not self.connection or self.connection.closed:
            await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Asynchronous context manager exit.

        Closes the WebSocket connection and cleans up resources
          when exiting the context.
        """
        await self.close()

    def _cancel_pending_requests(self):
        """
        Cancels all pending requests and clears the pending dictionary.
        """
        for req_id, future in self.pending.items():
            if not future.done():
                future.cancel()
        self.pending.clear()
        logger.info("All pending requests have been canceled.")
