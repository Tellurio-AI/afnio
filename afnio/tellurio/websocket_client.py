import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from afnio.logging_config import configure_logging
from afnio.tellurio._callable_registry import run_callable
from afnio.tellurio._model_registry import update_local_model_field
from afnio.tellurio._node_registry import (
    create_and_append_edge,
    create_node,
)
from afnio.tellurio._variable_registry import (
    append_grad_local,
    clear_pending_data,
    clear_pending_grad,
    create_local_variable,
    suppress_variable_notifications,
    update_local_variable_field,
)
from afnio.tellurio.run_context import get_active_run

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


# JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


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
                logger.debug(
                    f"Connecting to WebSocket at {self.ws_url} "
                    f"(attempt {attempt + 1}/{retries})"
                )
                self.connection = await websockets.connect(
                    self.ws_url, additional_headers=headers
                )

                # Start the listener task
                self.listener_task = asyncio.create_task(self._listener())
                logger.debug("WebSocket connection established.")

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
        Continuously listens for and processes incoming WebSocket messages.

        This method runs as a background task and handles all incoming messages from the
        WebSocket server according to the JSON-RPC 2.0 protocol. It supports:

        - Resolving responses to client-initiated requests by matching them with their
          corresponding request IDs and completing the associated futures.
        - Handling server-initiated JSON-RPC requests and notifications by dispatching
          them to the appropriate handler methods (e.g., `rpc_update_variable`).
        - Sending JSON-RPC responses or error messages back to the server as needed.
        - Logging and reporting protocol errors, unexpected messages, or exceptions.
        - Attempting to reconnect if the WebSocket connection is closed unexpectedly.

        This method ensures robust, asynchronous, and standards-compliant communication
        between the client and the Tellurio backend.

        Raises:
            ConnectionClosed: If the WebSocket connection is closed.
            Exception: For any unexpected errors during message processing.
        """
        try:
            async for message in self.connection:
                logger.debug(f"Received message: {message}")
                try:
                    data: Dict[str, Any] = json.loads(message)
                    req_id = data.get("id")

                    # Validate JSON-RPC version
                    jsonrpc_version = data.get("jsonrpc")
                    if jsonrpc_version != "2.0":
                        logger.warning(f"Invalid JSON-RPC version: {jsonrpc_version}")
                        await self._send_error(
                            req_id,
                            INVALID_REQUEST,
                            "Invalid JSON-RPC version. Expected '2.0'.",
                        )
                        continue

                    # Handle JSON-RPC responses to client-initiated requests
                    if req_id and "method" not in data:
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
                            logger.warning(f"Unexpected data or missing ID: {data}")
                        continue

                    # Handle JSON-RPC requests and notifications (must have "method")
                    if "method" not in data:
                        logger.warning("Invalid request. Missing 'method' field.")
                        await self._send_error(
                            req_id,
                            INVALID_REQUEST,
                            "Missing required field: method",
                        )
                        continue

                    # Handle the RPC method
                    method = data.get("method")
                    handler = getattr(self, f"rpc_{method}", None)
                    if not handler:
                        logger.warning(f"RPC method not found: {method}")
                        await self._send_error(
                            req_id,
                            METHOD_NOT_FOUND,
                            f"Method '{method}' not found.",
                            {"method": method},
                        )
                        continue

                    # Handle notifications (no id): do not send a response
                    if req_id is None:
                        logger.debug(
                            f"Received notification for method '{method}' "
                            f"with params: {data.get('params', {})}"
                        )
                        await handler(data.get("params", {}))
                        continue

                    # Handle request (with id): execute RPC method and send response
                    params = data.get("params", {})
                    logger.debug(
                        f"RPC method call: method={method!r} "
                        f"params={params!r} id={req_id!r}"
                    )

                    result = await handler(params)

                    # Send the response
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": result,
                    }
                    logger.debug(
                        f"RPC method executed successfully: method={method!r} "
                        f"result={result!r} id={req_id!r}"
                    )
                    await self.connection.send(json.dumps(response))

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON data: {e}")
                    await self._send_error(
                        req_id, PARSE_ERROR, "Parse error", {"error": str(e)}
                    )
                except KeyError as e:
                    logger.error(f"Missing key in request: {e}")
                    await self._send_error(
                        req_id,
                        INVALID_PARAMS,
                        f"Missing key: {e}",
                        {"missing_key": str(e)},
                    )
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    await self._send_error(
                        req_id, INTERNAL_ERROR, "Internal error", {"exception": str(e)}
                    )

        except ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
            await asyncio.sleep(1)  # Add a delay before reconnecting
            await self.connect()  # Attempt to reconnect
        except Exception as e:
            logger.error(f"Unexpected error in listener: {e}")

    async def rpc_create_variable(self, params):
        """
        Handle the 'create_variable' JSON-RPC method from the server.

        This method creates and registers a new Variable instance in the local registry
        using the provided parameters. It is typically called when the server creates a
        deepcopy of a Variable or Parameter and needs to notify the client.

        Args:
            params (dict): A dictionary with keys:
                - "variable_id": The unique identifier of the Variable.
                - "obj_type": The type of the variable object
                    (e.g., "__variable__" or "__parameter__").
                - "data": The initial data for the variable.
                - "role": The role or description of the variable.
                - "requires_grad": Whether the variable requires gradients.
                - "_retain_grad": Whether to retain gradients for non-leaf variables.
                - "_grad": The initial gradient(s) for the variable.
                - "_output_nr": The output number for the variable in the computation
                    graph.
                - "_grad_fn": The gradient function associated with the variable.
                - "is_leaf": Whether the variable is a leaf node in the computation
                    graph.

        Returns:
            dict: A dictionary with a success message if the variable is created.

        Raises:
            KeyError: If required keys are missing from params.
            RuntimeError: If the variable creation fails for any reason.
        """
        try:
            var = create_local_variable(
                params["variable_id"],
                params["obj_type"],
                params["data"],
                params["role"],
                params["requires_grad"],
                params["_retain_grad"],
                params["_grad"],
                params["_output_nr"],
                params["_grad_fn"],
                params["is_leaf"],
            )
            logger.debug(f"Variable created: variable_id={var.variable_id!r}")
            return {"message": "Ok"}
        except KeyError as e:
            logger.error(f"Missing key in params: {e}")
            raise KeyError(f"Missing key: {e}")
        except RuntimeError as e:
            logger.error(f"Failed to create variable: {e}")
            raise RuntimeError(f"Failed to create variable: {e}")

    async def rpc_update_variable(self, params):
        """
        Handle the 'update_variable' JSON-RPC method from the server.

        This method updates a specific field of a registered Variable instance
        in response to a server notification. It uses the provided parameters
        to identify the variable and the field to update.

        Args:
            params (dict): A dictionary with keys:
                - "variable_id": The unique identifier of the Variable.
                - "field": The field name to update.
                - "value": The new value to set for the field.

        Returns:
            dict: A dictionary with a success message if the update is successful.

        Raises:
            KeyError: If required keys are missing from params.
            RuntimeError: If the update fails for any reason.
        """
        try:
            with suppress_variable_notifications():
                update_local_variable_field(
                    params["variable_id"], params["field"], params["value"]
                )
                logger.debug(
                    f"Variable updated: variable_id={params['variable_id']!r} "
                    f"field={params['field']!r} value={params['value']!r}"
                )
                return {"message": "Ok"}
        except KeyError as e:
            logger.error(f"Missing key in params: {e}")
            raise KeyError(f"Missing key: {e}")
        except RuntimeError as e:
            logger.error(
                f"Failed to update variable with ID {params.get('variable_id')!r}: {e}"
            )
            raise RuntimeError(
                f"Failed to update variable with ID {params.get('variable_id')!r}: {e}"
            )

    async def rpc_append_grad(self, params):
        """
        Handle the 'append_grad' JSON-RPC method from the server.

        This method appends a new gradient variable to the local grad list of the
        specified Variable instance. It is typically called when the server notifies
        the client that a new gradient has been added to a variable during
        the backward pass.

        Args:
            params (dict): A dictionary containing:
                - "variable_id": The unique identifier of the Variable to update.
                - "gradient": The serialized gradient Variable to append.

        Returns:
            dict: A dictionary with a success message if the gradient is appended.

        Raises:
            KeyError: If required keys are missing from params.
            RuntimeError: If appending the gradient fails for any reason.
        """
        try:
            with suppress_variable_notifications():
                append_grad_local(
                    params["variable_id"], params["gradient_id"], params["gradient"]
                )
                logger.debug(
                    f"Gradient appended: variable_id={params['variable_id']!r} "
                    f"gradient={params['gradient']!r}"
                )
                return {"message": "Ok"}
        except KeyError as e:
            logger.error(f"Missing key in params: {e}")
            raise KeyError(f"Missing key: {e}")
        except RuntimeError as e:
            logger.error(
                f"Failed to append gradient for variable with ID "
                f"{params.get('variable_id')!r}: {e}"
            )
            raise RuntimeError(
                f"Failed to append gradient for variable with ID "
                f"{params.get('variable_id')!r}: {e}"
            )

    async def rpc_create_node(self, params):
        """
        Handle the 'create_node' JSON-RPC method from the server.

        This method creates and registers a new Node instance in the local registry
        using the provided parameters. It is typically called when the server notifies
        the client that a new node has been created in the computation graph.

        Args:
            params (dict): A dictionary with keys:
                - "node_id": The unique identifier of the Node.
                - "name": The class name or type of the Node.

        Returns:
            dict: A dictionary with a success message if the node is created.

        Raises:
            KeyError: If required keys are missing from params.
        """
        try:
            create_node(params)
            logger.debug(
                f"Node created: node_id={params['node_id']!r} name={params['name']!r}"
            )
            return {"message": "Ok"}
        except KeyError as e:
            logger.error(f"Missing key in params: {e}")
            raise KeyError(f"Missing key: {e}")

    async def rpc_create_edge(self, params):
        """
        Handle the 'create_edge' JSON-RPC method from the server.

        This method creates a GradientEdge between two nodes in the local registry,
        appending the edge to the from_node's next_functions. It is typically called
        when the server notifies the client that a new edge has been created in the
        computation graph.

        Note:
            The terms 'from_node' and 'to_node' should be interpreted in the context
            of the backward pass (backpropagation): the edge is added to the
            from_node's next_functions and points to the to_node, following the
            direction of gradient flow during backpropagation.

        Args:
            params (dict): A dictionary with keys:
                - "from_node_id": The unique identifier of the source node.
                - "to_node_id": The unique identifier of the destination node.
                - "output_nr": The output number associated with the edge.

        Returns:
            dict: A dictionary with a success message if the edge is created.

        Raises:
            KeyError: If required keys are missing from params.
        """
        try:
            create_and_append_edge(params)
            logger.debug(
                f"Edge created: from_node_id={params['from_node_id']!r} "
                f"to_node_id={params['to_node_id']!r} output_nr={params['output_nr']!r}"
            )
            return {"message": "Ok"}
        except KeyError as e:
            logger.error(f"Missing key in params: {e}")
            raise KeyError(f"Missing key: {e}")

    async def rpc_update_model(self, params):
        """
        Handle the 'update_model' JSON-RPC method from the server.

        This method updates a specific field of a registered LM model instance
        in response to a server notification. It uses the provided parameters
        to identify the LM model and the field to update.

        Args:
            params (dict): A dictionary with keys:
                - "model_id": The unique identifier of the LM model.
                - "field": The field name to update.
                - "value": The new value to set for the field.

        Returns:
            dict: A dictionary with a success message if the update is successful.

        Raises:
            KeyError: If required keys are missing from params.
            RuntimeError: If the update fails for any reason.
        """
        try:
            update_local_model_field(
                params["model_id"], params["field"], params["value"]
            )
            logger.debug(
                f"Model updated: model_id={params['model_id']!r} "
                f"field={params['field']!r} value={params['value']!r}"
            )
            return {"message": "Ok"}
        except KeyError as e:
            logger.error(f"Missing key in params: {e}")
            raise KeyError(f"Missing key: {e}")
        except RuntimeError as e:
            logger.error(
                f"Failed to update model with ID {params.get('model_id')!r}: {e}"
            )
            raise RuntimeError(
                f"Failed to update model with ID {params.get('model_id')!r}: {e}"
            )

    async def rpc_run_callable(self, params):
        """
        Handle the 'run_callable' JSON-RPC method from the server.

        This method is invoked when the server sends a JSON-RPC request with the
        method "run_callable". It extracts callable details from the provided
        parameters, executes the callable from the registry, and returns a response
        containing the result. The response is expected to be JSON-serializable.

        Args:
            params (dict): A dictionary containing:
                - "callable_id": A unique identifier for the callable.
                - "args": Positional arguments (as a list or tuple) for the callable.
                - "kwargs": Keyword arguments for the callable.

        Returns:
            dict: A dictionary with the following structure:
                {
                    "message": "Ok",
                    "data": <result of executing the callable>
                }

        Raises:
            KeyError: If required keys are missing in params.
            TypeError: If the result of the callable is not JSON-serializable.
            ValueError: If the callable execution fails due to invalid parameters.
            RuntimeError: For any other exception encountered during callable execution.
        """
        try:
            result = run_callable(params)

            # Check if result is JSON serializable
            try:
                json.dumps(result)
            except (TypeError, ValueError) as e:
                logger.error(
                    f"Result of callable with ID {params.get('callable_id')!r} "
                    f"is not JSON-serializable: {result!r} ({e})"
                )
                raise TypeError(
                    f"Result of callable with ID {params.get('callable_id')!r} "
                    f"is not JSON-serializable: {result!r} ({e})"
                )

            logger.debug(
                f"Callable executed successfully: "
                f"callable_id={params['callable_id']!r}, "
                f"args={params.get('args', {})!r}, "
                f"kwargs={params.get('kwargs', {})!r}"
            )
            return {"message": "Ok", "data": result}
        except KeyError as e:
            logger.error(f"Missing key in params: {e}")
            raise KeyError(f"Missing key: {e}")
        except ValueError as e:
            logger.error(
                f"Failed to run callable with ID {params.get('callable_id')!r}: {e}"
            )
            raise ValueError(
                f"Failed to run callable with ID {params.get('callable_id')!r}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Exception during execution of callable "
                f"with ID {params.get('callable_id')!r}: {e}"
            )
            raise RuntimeError(
                f"Exception during execution of callable "
                f"with ID {params.get('callable_id')!r}: {e}"
            )

    async def rpc_clear_backward(self, params):
        """
        Handle the 'clear_backward' JSON-RPC method from the server.

        This method clears the `_pending_grad` flag for the specified variables.
        It is called after the server finalizes the backward pass for the entire
        computation graph, indicating that the gradients for its variables have been
        computed and already shared with the client. Once it receives 'clear_backward',
        the client can safely access the values of these gradients without worrying
        about them being modified.

        Args:
            params (dict): A dictionary containing:
                - "variable_ids": A list of variable IDs for which to clear
                    the `_pending_grad` flag.

        Raises:
            KeyError: If required keys are missing from params.
            RuntimeError: If clearing the pending grad fails for any variable.
        """
        try:
            variable_ids = params["variable_ids"]
            clear_pending_grad(variable_ids)

            logger.debug(f"Cleared pending gradients for variables: {variable_ids!r}")
            return {"message": "Ok"}
        except KeyError as e:
            logger.error(f"Missing key in params: {e}")
            raise KeyError(f"Missing key: {e}")
        except RuntimeError as e:
            logger.error(
                f"Failed to update variable with ID {params.get('variable_id')!r}: {e}"
            )
            raise RuntimeError(
                f"Failed to update variable with ID {params.get('variable_id')!r}: {e}"
            )
        except Exception as e:
            logger.error(f"Exception during execution of backward clearing: {e}")
            raise RuntimeError(f"Exception during execution of backward clearing: {e}")

    async def rpc_clear_step(self, params):
        """
        Handle the 'clear_step' JSON-RPC method from the server.

        This method clears the `_pending_data` flag for the specified variables.
        It is called after the server completes an optimizer step and updates
        the data for the relevant variables. Once 'clear_step' is received,
        the client can safely access the updated values of these variables,
        knowing that the data is no longer pending or being modified.

        Args:
            params (dict): A dictionary containing:
                - "variable_ids": A list of variable IDs (str) for which to clear
                  the `_pending_data` flag.

        Raises:
            KeyError: If required keys are missing from params.
            RuntimeError: If clearing the pending data fails for any variable.
        """
        try:
            variable_ids = params["variable_ids"]
            clear_pending_data(variable_ids)

            logger.debug(f"Cleared pending data for variables: {variable_ids!r}")
            return {"message": "Ok"}
        except KeyError as e:
            logger.error(f"Missing key in params: {e}")
            raise KeyError(f"Missing key: {e}")
        except RuntimeError as e:
            logger.error(
                f"Failed to update variable with ID {params.get('variable_id')!r}: {e}"
            )
            raise RuntimeError(
                f"Failed to update variable with ID {params.get('variable_id')!r}: {e}"
            )
        except Exception as e:
            logger.error(f"Exception during execution of backward clearing: {e}")
            raise RuntimeError(f"Exception during execution of backward clearing: {e}")

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

        active_run = get_active_run()
        params["run_uuid"] = active_run.uuid

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
        future = asyncio.get_running_loop().create_future()
        self.pending[req_id] = future
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"Request timed out: {request}")
            raise
        finally:
            self.pending.pop(req_id, None)

    async def close(self):
        """
        Closes the WebSocket connection and cleans up resources.

        Cancels the listener task, clears pending requests, and closes the WebSocket
        connection.
        """
        # Add a delay to allow receiving and replying to remaining server requests
        await asyncio.sleep(1)

        if self.listener_task:
            logger.debug("Canceling listener task...")
            self.listener_task.cancel()
            try:
                await self.listener_task  # Wait for the listener task to finish
            except asyncio.CancelledError:
                logger.debug("Listener task canceled.")
                pass  # Ignore cancellation errors
        self.listener_task = None  # Clean up the listener task

        if self.connection:
            logger.debug("Closing WebSocket connection...")
            try:
                await self.connection.close()
            finally:
                self.connection = None

        logger.debug("Clearing pending requests...")
        self._cancel_pending_requests()  # Clear pending requests

        logger.debug("WebSocket connection closed.")

    async def _send_error(
        self,
        req_id: Optional[str],
        code: int,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ):
        """
        Send an error response to the server.

        Args:
            req_id (Optional[str]): The ID of the request that caused the error.
            code (int): The JSON-RPC error code.
            message (str): A description of the error.
            data (Optional[Dict[str, Any]]): Additional data about the error.
        """
        logger.warning(
            f"Sending error response. ID: {req_id}, Code: {code}, "
            f"Message: {message}, Data: {data}"
        )
        error_response = {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": code,
                "message": message,
            },
        }
        if data:
            error_response["error"]["data"] = data
        await self.connection.send(json.dumps(error_response))

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
        logger.debug("All pending requests have been canceled.")
