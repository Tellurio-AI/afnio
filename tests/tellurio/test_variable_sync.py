import asyncio
import json
import os
import re
from unittest.mock import AsyncMock, patch

import pytest

import afnio as hf
from afnio._variable import _allow_grad_fn_assignment
from afnio.autodiff.graph import Node
from afnio.tellurio import login
from afnio.tellurio._eventloop import run_in_background_loop
from afnio.tellurio._node_registry import register_node
from afnio.tellurio._variable_registry import VARIABLE_REGISTRY
from afnio.tellurio.client import get_default_client
from afnio.tellurio.run import init
from afnio.tellurio.websocket_client import TellurioWebSocketClient


@pytest.fixture(autouse=True)
def login_and_ensure_default_run():
    """
    Test the login function with real HTTP and WebSocket connections and
    ensure a default Run exists and is set as active before tests.
    """
    # Log in to the Tellurio service using the API key
    api_key = os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key")
    login(api_key=api_key)

    # Use your test org/project names from env or defaults
    namespace_slug = os.getenv("TEST_ORG_SLUG", "tellurio-test")
    project_display_name = os.getenv("TEST_PROJECT", "Test Project")
    run = init(namespace_slug, project_display_name)
    return run


@pytest.fixture
def variable():
    """
    Fixture to create a Variable instance.
    """
    var = hf.Variable(data="Tellurio", role="input variable", requires_grad=True)

    # Assert initial state of the variable
    assert var.data == "Tellurio"
    assert var.role == "input variable"
    assert var.requires_grad is True
    assert var._retain_grad is False
    assert var._grad == []
    assert var._output_nr == 0
    assert var._grad_fn is None
    assert var.is_leaf is True
    assert var.variable_id is not None

    return var


class TestClientToServerVariableSync:
    """
    Test the synchronization of variables between the client and server.
    This test suite uses the TellurioWebSocketClient to create and manipulate variables,
    and verifies that changes are reflected on the server.
    """

    def fetch_server_variable(self, variable_id):
        """
        Fetch the variable from the server using its variable_id.
        This function uses the TellurioWebSocketClient to send a request to the server
        and retrieve the variable's data.
        """
        _, ws_client = get_default_client()
        response = run_in_background_loop(
            ws_client.call("get_variable", {"variable_id": variable_id})
        )
        return response["result"]

    def test_create_variable(self, variable):
        """
        Test that creating a Variable triggers a notification to the server.
        """
        assert variable.variable_id is not None
        assert variable.variable_id in VARIABLE_REGISTRY
        assert VARIABLE_REGISTRY[variable.variable_id] is variable

    @pytest.mark.parametrize(
        "field,value",
        [
            ("data", "Tellurio is great!"),
            ("role", "output"),
            ("requires_grad", False),
        ],
    )
    def test_set_field_triggers_notification(self, variable, field, value):
        """
        Test that setting a Variable's attribute triggers a notification.
        """
        # Set the field to a new value
        setattr(variable, field, value)

        # Assert that the variable was updated locally
        assert value == getattr(variable, field)

        # Assert that the variable was updated on the server
        server_var = self.fetch_server_variable(variable.variable_id)
        assert server_var[field] == value

    def test_multiple_set_field_notification_order(self, variable):
        """
        Test that multiple internal Variable changes trigger notifications in order.
        """
        # Set the field to a new value
        changes = [10, 20, 30, 40, 50]
        for val in changes:
            variable.data = val

            # Assert that the variable was updated locally
            assert val == getattr(variable, "data")

            # Assert that the variable was updated on the server
            server_var = self.fetch_server_variable(variable.variable_id)
            assert server_var["data"] == val

    def test_requires_grad_method_triggers_notification(self, variable):
        """
        Test that calling requires_grad_() triggers two notifications
        (requires_grad and is_leaf).
        """
        # Set `_requires_grad` to False
        variable.requires_grad_(False)

        # Assert that the variable was updated locally
        assert variable.requires_grad is False
        assert variable.is_leaf is True

        # Assert that the variable was updated on the server
        server_var = self.fetch_server_variable(variable.variable_id)
        assert server_var["requires_grad"] is False
        assert server_var["is_leaf"] is True

    def test_set_output_nr_triggers_notification(self, variable):
        """
        Test that setting output_nr using the setter triggers a notification.
        """
        # Set `_output_nr` to a new value
        variable.output_nr = 3

        # Assert that the variable was updated locally
        assert variable.output_nr == 3

        # Assert that the variable was updated on the server
        server_var = self.fetch_server_variable(variable.variable_id)
        assert server_var["output_nr"] == 3

    def test_set_grad_fn_raises(self, variable):
        """
        Test that setting grad_fn using the setter raises an error on the client.

        `_allow_grad_fn_assignment()` should never be called on the client.
        """
        # Make sure the variable requires_grad is True
        variable.requires_grad = True
        server_var = self.fetch_server_variable(variable.variable_id)
        assert server_var["requires_grad"] is True

        # Use a dummy callable for grad_fn
        class AddBackward:
            pass

        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "Setting `grad_fn` is only allowed on the server by the autodiff "
                "engine. Do not use `_allow_grad_fn_assignment()` on the client."
            ),
        ):
            with _allow_grad_fn_assignment():
                variable.grad_fn = AddBackward()

    def test_set_grad_triggers_notification(self, variable):
        """Test that setting grad using the setter triggers a notification."""
        grad_1 = hf.Variable(data="gradient", role="grad_1", requires_grad=False)
        grad_2 = hf.Variable(data="gradient", role="grad_2", requires_grad=False)
        variable.grad = [grad_1, grad_2]

        # Assert that the variable was updated locally
        assert variable.grad == [grad_1, grad_2]

        # Assert that the variable was updated on the server
        server_var = self.fetch_server_variable(variable.variable_id)
        assert server_var["grad"] == [
            {"__variable__": True, "variable_id": grad_1.variable_id},
            {"__variable__": True, "variable_id": grad_2.variable_id},
        ]

    def test_append_grad_triggers_notification(self, variable):
        """
        Test that appending gradients to a Variable using append_grad()
        triggers a notification to the client with the correct grad serialization.
        """
        grad_list_id = id(variable.grad)  # Get the ID of the grad list before appending

        # Append first gradient
        grad_1 = hf.Variable(data="gradient", role="grad_1", requires_grad=False)
        variable.append_grad(grad_1)

        # Assert that the variable was updated locally
        # and the grad list ID remains the same
        assert len(variable.grad) == 1
        assert variable.grad[0].variable_id == grad_1.variable_id
        assert grad_list_id == id(variable.grad)

        # Assert that the variable was updated on the server
        server_var = self.fetch_server_variable(variable.variable_id)
        assert server_var["grad"] == [
            {"__variable__": True, "variable_id": grad_1.variable_id}
        ]

        # Append second gradient
        grad_2 = hf.Variable(data="gradient", role="grad_2", requires_grad=False)
        variable.append_grad(grad_2)

        # Assert that the variable was updated locally
        # and the grad list ID remains the same
        assert len(variable.grad) == 2
        assert variable.grad[0].variable_id == grad_1.variable_id
        assert variable.grad[1].variable_id == grad_2.variable_id
        assert grad_list_id == id(variable.grad)

        # Assert that the variable was updated on the server
        server_var = self.fetch_server_variable(variable.variable_id)
        assert server_var["grad"] == [
            {"__variable__": True, "variable_id": grad_1.variable_id},
            {"__variable__": True, "variable_id": grad_2.variable_id},
        ]

    def test_retain_grad_method_triggers_notification(self, variable):
        """Test that calling retain_grad() triggers a notification."""
        # WARNING: Manually setting `is_leaf` should be avoided in production code.
        # This is just for testing purposes to simulate the server's behavior.
        variable.is_leaf = False  # We force the variable to be non-leaf

        # Call retain_grad() on a non-leaf variable
        variable.retain_grad()

        # Assert that the variable was updated locally
        assert variable._retain_grad is True

        # Assert that the variable was updated on the server
        server_var = self.fetch_server_variable(variable.variable_id)
        assert server_var["_retain_grad"] is True

    def test_copy_method_triggers_notifications(self, variable):
        """
        Test that calling copy_() triggers notifications for data, role,
        and requires_grad.
        """
        src = hf.Variable(data=123, role="copied", requires_grad=False)
        variable.copy_(src)

        # Assert that the variable was updated locally
        assert variable.data == 123
        assert variable.role == "copied"
        assert variable.requires_grad is False

        # Assert that the variable was updated on the server
        server_var = self.fetch_server_variable(variable.variable_id)
        assert server_var["data"] == 123
        assert server_var["role"] == "copied"
        assert server_var["requires_grad"] is False


class TestServerToClientVariableSync:
    """
    Test the synchronization of variables from the server to the client.
    This test suite uses the TellurioClient to create and manipulate variables,
    and verifies that changes are reflected on the client.
    """

    @pytest.fixture
    def mock_server_update_variable_request(self):
        """
        Fixture to simulate receiving an 'update_variable' RPC call from the server.

        Usage:
            mock_server_update_variable_request(variable_id, field, value)
        """

        def _mock(variable_id, field, value):

            # Compose the JSON-RPC message
            message = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "update_variable",
                    "params": {
                        "variable_id": variable_id,
                        "field": field,
                        "value": value,
                    },
                    "id": "test-id-123",
                }
            )

            # Patch the connection to use a fake async send
            class FakeConnection:
                def __init__(self):
                    self.sent_messages = []
                    self._closed = False

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if hasattr(self, "_sent"):
                        self._closed = True
                        raise StopAsyncIteration
                    self._sent = True
                    return message

                async def send(self, msg):
                    self.sent_messages.append(msg)

                @property
                def closed(self):
                    return self._closed

            # Create a fresh client instance (no singleton)
            ws_client = TellurioWebSocketClient()
            ws_client.connection = FakeConnection()

            # Patch send to track calls
            with patch.object(
                ws_client.connection, "send", new_callable=AsyncMock
            ) as mock_send:
                loop = asyncio.get_event_loop()
                listener_task = loop.create_task(ws_client._listener())
                loop.run_until_complete(asyncio.sleep(0.1))
                listener_task.cancel()
                try:
                    loop.run_until_complete(listener_task)
                except asyncio.CancelledError:
                    pass

                # Return the mock so the test can assert on it
                return mock_send

        return _mock

    @pytest.fixture
    def mock_server_append_grad_request(self):
        """
        Fixture to simulate receiving an 'append_grad' RPC call from the server.

        Usage:
            mock_server_append_grad_request(variable_id, field, value)
        """

        def _mock(variable_id, gradient_id, gradient):

            # Compose the JSON-RPC message
            message = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "append_grad",
                    "params": {
                        "variable_id": variable_id,
                        "gradient_id": gradient_id,
                        "gradient": gradient,
                    },
                    "id": "test-id-123",
                }
            )

            # Patch the connection to use a fake async send
            class FakeConnection:
                def __init__(self):
                    self.sent_messages = []
                    self._closed = False

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if hasattr(self, "_sent"):
                        self._closed = True
                        raise StopAsyncIteration
                    self._sent = True
                    return message

                async def send(self, msg):
                    self.sent_messages.append(msg)

                @property
                def closed(self):
                    return self._closed

            # Create a fresh client instance (no singleton)
            ws_client = TellurioWebSocketClient()
            ws_client.connection = FakeConnection()

            # Patch send to track calls
            with patch.object(
                ws_client.connection, "send", new_callable=AsyncMock
            ) as mock_send:
                loop = asyncio.get_event_loop()
                listener_task = loop.create_task(ws_client._listener())
                loop.run_until_complete(asyncio.sleep(0.1))
                listener_task.cancel()
                try:
                    loop.run_until_complete(listener_task)
                except asyncio.CancelledError:
                    pass

                # Return the mock so the test can assert on it
                return mock_send

        return _mock

    @pytest.fixture
    def mock_server_create_variable_request(self):
        """
        Fixture to simulate receiving a 'create_variable' RPC call from the server.

        Usage:
            mock_server_create_variable_request(
                variable_id,
                obj_type,
                data,
                role,
                requires_grad,
                _retain_grad,
                _grad,
                _output_nr,
                _grad_fn,
                is_leaf
            )
        """

        def _mock(
            variable_id,
            obj_type,
            data,
            role,
            requires_grad,
            _retain_grad,
            _grad,
            _output_nr,
            _grad_fn,
            is_leaf,
        ):

            # Compose the JSON-RPC message
            message = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "create_variable",
                    "params": {
                        "variable_id": variable_id,
                        "obj_type": obj_type,
                        "data": data,
                        "role": role,
                        "requires_grad": requires_grad,
                        "_retain_grad": _retain_grad,
                        "_grad": _grad,
                        "_output_nr": _output_nr,
                        "_grad_fn": _grad_fn,
                        "is_leaf": is_leaf,
                    },
                    "id": "test-id-123",
                }
            )

            # Patch the connection to use a fake async send
            class FakeConnection:
                def __init__(self):
                    self.sent_messages = []
                    self._closed = False

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if hasattr(self, "_sent"):
                        self._closed = True
                        raise StopAsyncIteration
                    self._sent = True
                    return message

                async def send(self, msg):
                    self.sent_messages.append(msg)

                @property
                def closed(self):
                    return self._closed

            # Create a fresh client instance (no singleton)
            ws_client = TellurioWebSocketClient()
            ws_client.connection = FakeConnection()

            # Patch send to track calls
            with patch.object(
                ws_client.connection, "send", new_callable=AsyncMock
            ) as mock_send:
                loop = asyncio.get_event_loop()
                listener_task = loop.create_task(ws_client._listener())
                loop.run_until_complete(asyncio.sleep(0.1))
                listener_task.cancel()
                try:
                    loop.run_until_complete(listener_task)
                except asyncio.CancelledError:
                    pass

                # Return the mock so the test can assert on it
                return mock_send

        return _mock

    @staticmethod
    def assert_valid_client_response(send_mock):
        """
        Assert that the client sent a valid JSON-RPC response
        to an update_variable or append_grad request.
        """
        send_mock.assert_awaited()
        sent_msg = send_mock.call_args[0][0]
        response = json.loads(sent_msg)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-id-123"
        assert response["result"]["message"] == "Ok"

    @pytest.mark.parametrize(
        "obj_type",
        [
            ("__variable__"),
            ("__parameter__"),
        ],
    )
    def test_create_variable(self, mock_server_create_variable_request, obj_type):
        """
        Test that a server's creation of a Variable is reflected in the client.
        """
        # Server creates a new variable
        send_mock = mock_server_create_variable_request(
            variable_id="var-123",
            obj_type=obj_type,
            data="Tellurio",
            role="input variable",
            requires_grad=True,
            _retain_grad=False,
            _grad=[],
            _output_nr=0,
            _grad_fn=None,
            is_leaf=True,
        )

        # Assert that the variable was updated locally
        var = VARIABLE_REGISTRY.get("var-123")
        assert var.data == "Tellurio"

        # Assert that the client sends the correct response to the server
        self.assert_valid_client_response(send_mock)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("data", "Tellurio is great!"),
            ("role", "output"),
            ("requires_grad", False),
        ],
    )
    def test_set_field_triggers_notification(
        self, variable, field, value, mock_server_update_variable_request
    ):
        """
        Test that a server's update to a Variable's attribute
        is reflected in the client.
        """
        # Server sets the field to a new value
        send_mock = mock_server_update_variable_request(
            variable.variable_id, field, value
        )

        # Assert that the variable was updated locally
        assert value == getattr(variable, field)

        # Assert that the client sends the correct response to the server
        self.assert_valid_client_response(send_mock)

    def test_multiple_set_field_notification_order(
        self, variable, mock_server_update_variable_request
    ):
        """
        Test that multiple server updates to a Variable's attribute
        are propagated to the client in the correct order.
        """
        # Server sets the field to a new value
        changes = [10, 20, 30, 40, 50]
        for val in changes:
            send_mock = mock_server_update_variable_request(
                variable.variable_id, "data", val
            )

            # Assert that the variable was updated locally
            assert val == getattr(variable, "data")

            # Assert that the client sends the correct response to the server
            self.assert_valid_client_response(send_mock)

    def test_requires_grad_method_triggers_notification(
        self, variable, mock_server_update_variable_request
    ):
        """
        Test that the server calling `requires_grad_()` is reflected in the client.
        We emulate the server call to `requires_grad_()` by directly sending the two
        notifications to the client:
          - `requires_grad` is set to False
          - `is_leaf` is set to True
        """
        updates = [("requires_grad", False), ("is_leaf", True)]

        # Server sets `_requires_grad` to False
        for field, value in updates:
            send_mock = mock_server_update_variable_request(
                variable.variable_id, field, value
            )

        for field, value in updates:
            # Assert that the variable was updated locally
            assert getattr(variable, field) == value

            # Assert that the client sends the correct response to the server
            self.assert_valid_client_response(send_mock)

    def test_set_output_nr_triggers_notification(
        self, variable, mock_server_update_variable_request
    ):
        """
        Test that a server's update to a Variable's output_nr
        is reflected in the client.
        """
        # Server sets `_output_nr` to a new value
        send_mock = mock_server_update_variable_request(
            variable.variable_id, "output_nr", 3
        )

        # Assert that the variable was updated locally
        assert variable.output_nr == 3

        # Assert that the client sends the correct response to the server
        self.assert_valid_client_response(send_mock)

    def test_set_grad_fn_triggers_notification(
        self, variable, mock_server_update_variable_request
    ):
        """
        Test that a server's update to a Variable's grad_fn is reflected in the client.
        """
        # Register the node in the registry before sending the update
        node_id = "node-id-123"
        node = Node()
        node._name = "AddBackward"
        node.node_id = node_id
        register_node(node)

        # Server sets `_grad_fn` to a new value
        send_mock = mock_server_update_variable_request(
            variable.variable_id, "_grad_fn", "node-id-123"
        )

        # Assert that the variable was updated locally
        assert variable.grad_fn is node
        assert variable.grad_fn.name() == "AddBackward"

        # Assert that the client sends the correct response to the server
        self.assert_valid_client_response(send_mock)

    def test_set_grad_fn_raises_if_node_missing(
        self, variable, mock_server_update_variable_request
    ):
        """
        Test that updating _grad_fn with a non-existent node_id raises a ValueError.
        """
        non_existent_node_id = "not-in-registry"
        send_mock = mock_server_update_variable_request(
            variable.variable_id, "_grad_fn", non_existent_node_id
        )
        send_mock.assert_awaited()
        sent_msg = send_mock.call_args[0][0]
        response = json.loads(sent_msg)
        assert "error" in response
        assert response["error"]["message"] == "Internal error"
        assert response["error"]["data"]["exception"] == (
            f"Node with id 'not-in-registry' not found in registry "
            f"when updating _grad_fn for variable '{variable.variable_id}'."
        )

    def test_set_grad_triggers_notification(
        self, variable, mock_server_update_variable_request
    ):
        """
        Test that a server's update to a Variable's grad is reflected in the client.
        """
        # Server sets `_grad` to a new value
        value = [
            {"data": "gradient", "role": "grad_1", "requires_grad": False},
            {"data": "gradient", "role": "grad_2", "requires_grad": False},
        ]
        send_mock = mock_server_update_variable_request(
            variable.variable_id, "_grad", value
        )

        # Assert that the variable was updated locally
        assert len(variable.grad) == 2
        assert variable.grad[0].data == "gradient"
        assert variable.grad[0].role == "grad_1"
        assert variable.grad[0].requires_grad is False
        assert variable.grad[1].data == "gradient"
        assert variable.grad[1].role == "grad_2"
        assert variable.grad[1].requires_grad is False

        # Assert that the client sends the correct response to the server
        self.assert_valid_client_response(send_mock)

    def test_append_grad_triggers_notification(
        self, variable, mock_server_append_grad_request
    ):
        """
        Test that the server calling `append_grad()` is reflected in the client. We also
        make sure that the grad list is the same object after appending gradients.
        """
        grad_list_id = id(variable.grad)  # Get the ID of the grad list before appending

        # Server appends first gradient
        value = {"data": "gradient", "role": "grad_1", "requires_grad": False}
        send_mock = mock_server_append_grad_request(
            variable.variable_id, "grad_123", value
        )

        # Assert that the variable was updated locally
        assert len(variable.grad) == 1
        assert grad_list_id == id(variable.grad)
        assert variable.grad[0].data == "gradient"
        assert variable.grad[0].role == "grad_1"
        assert variable.grad[0].requires_grad is False

        # Assert that the client sends the correct response to the server
        self.assert_valid_client_response(send_mock)

        # Server appends second gradient (new entire _grad list is sent)
        value = {"data": "gradient", "role": "grad_2", "requires_grad": False}
        send_mock = mock_server_append_grad_request(
            variable.variable_id, "grad_456", value
        )

        # Assert that the variable was updated locally
        assert len(variable.grad) == 2
        assert grad_list_id == id(variable.grad)
        assert variable.grad[0].data == "gradient"
        assert variable.grad[0].role == "grad_1"
        assert variable.grad[0].requires_grad is False
        assert variable.grad[1].data == "gradient"
        assert variable.grad[1].role == "grad_2"
        assert variable.grad[1].requires_grad is False

        # Assert that the client sends the correct response to the server
        self.assert_valid_client_response(send_mock)

    def test_retain_grad_method_triggers_notification(
        self, variable, mock_server_update_variable_request
    ):
        """
        Test that the server calling `retain_grad()` is reflected in the client.
        We emulate the server call to `retain_grad()` by directly sending the
        notification to the client:
          - `_retain_grad` is set to True
        """
        # WARNING: Manually setting `is_leaf` should be avoided in production code.
        # This is just for testing purposes to simulate the server's behavior.
        variable.is_leaf = False  # We force the variable to be non-leaf

        # Server call retain_grad() on a non-leaf variable
        send_mock = mock_server_update_variable_request(
            variable.variable_id, "_retain_grad", True
        )

        # Assert that the variable was updated locally
        assert variable._retain_grad is True

        # Assert that the client sends the correct response to the server
        self.assert_valid_client_response(send_mock)

    def test_copy_method_triggers_notifications(
        self, variable, mock_server_update_variable_request
    ):
        """
        Test that the server calling `copy_()` is reflected in the client.
        We emulate the server call to `copy_()` by directly sending the
        notification to the client:
          - `data` is set to 123
          - `role` is set to "copied"
          - `requires_grad` is set to False
        """
        # Server calls `copy_()`` on a variable with new values
        send_mock = mock_server_update_variable_request(
            variable.variable_id, "data", 123
        )
        send_mock = mock_server_update_variable_request(
            variable.variable_id, "role", "copied"
        )
        send_mock = mock_server_update_variable_request(
            variable.variable_id, "requires_grad", False
        )

        # Assert that the variable was updated locally
        assert variable.data == 123
        assert variable.role == "copied"
        assert variable.requires_grad is False

        # Assert that the client sends the correct response to the server
        self.assert_valid_client_response(send_mock)

    def test_update_variable_raises_if_variable_missing(
        self, mock_server_update_variable_request
    ):
        """
        Test that updating a variable that does not exist in the registry results
        in an error response.
        """
        non_existent_variable_id = "not-in-registry"
        send_mock = mock_server_update_variable_request(
            non_existent_variable_id, "data", "new-value"
        )
        send_mock.assert_awaited()
        sent_msg = send_mock.call_args[0][0]
        response = json.loads(sent_msg)
        assert "error" in response
        assert response["error"]["message"] == "Internal error"
        assert response["error"]["data"]["exception"] == (
            "'NoneType' object has no attribute 'data'"
        )
