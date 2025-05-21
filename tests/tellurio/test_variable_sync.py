import asyncio
import json
import os
import re
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

import afnio as hf
from afnio._variable import _allow_grad_fn_assignment
from afnio.tellurio import login
from afnio.tellurio._eventloop import run_in_background_loop
from afnio.tellurio.client import get_default_client
from afnio.tellurio.websocket_client import TellurioWebSocketClient


@pytest.fixture
def variable():
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


@pytest_asyncio.fixture(autouse=True)
async def login_fixture():
    """
    Test the login function with real HTTP and WebSocket connections.
    """
    api_key = os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key")
    login(api_key=api_key)
    api_key = api_key


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
        print(response)
        return response["result"]

    def test_create_variable(self, variable):
        """
        Test that creating a Variable triggers a notification to the server.
        """
        assert variable.variable_id is not None

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
            {"data": "gradient", "role": "grad_1", "requires_grad": False},
            {"data": "gradient", "role": "grad_2", "requires_grad": False},
        ]

    def test_append_grad_triggers_notification(self, variable):
        """
        Test that appending gradients to a Variable using append_grad()
        triggers a notification to the client with the correct grad serialization.
        """
        # Append first gradient
        grad_1 = hf.Variable(data="gradient", role="grad_1", requires_grad=False)
        variable.append_grad(grad_1)

        # Assert that the variable was updated locally
        assert variable.grad == [grad_1]

        # Assert that the variable was updated on the server
        server_var = self.fetch_server_variable(variable.variable_id)
        assert server_var["grad"] == [
            {"data": "gradient", "role": "grad_1", "requires_grad": False}
        ]

        # Append second gradient
        grad_2 = hf.Variable(data="gradient", role="grad_2", requires_grad=False)
        variable.append_grad(grad_2)

        # Assert that the variable was updated locally
        assert variable.grad == [grad_1, grad_2]

        # Assert that the variable was updated on the server
        server_var = self.fetch_server_variable(variable.variable_id)
        assert server_var["grad"] == [
            {"data": "gradient", "role": "grad_1", "requires_grad": False},
            {"data": "gradient", "role": "grad_2", "requires_grad": False},
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
    def mock_server_update_notification(self):
        """
        Fixture to simulate receiving an 'update_variable' RPC call from the server.

        Usage:
            mock_server_update_notification(variable_id, field, value)
        """

        def _mock(variable_id, field, value):
            _, ws_client = get_default_client()

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
    def assert_valid_update_variable_response(send_mock):
        """
        Assert that the client sent a valid JSON-RPC response
        to an update_variable request.
        """
        send_mock.assert_awaited()
        sent_msg = send_mock.call_args[0][0]
        response = json.loads(sent_msg)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-id-123"
        assert response["result"]["message"] == "Ok"

    @pytest.mark.parametrize(
        "field,value",
        [
            ("data", "Tellurio is great!"),
            ("role", "output"),
            ("requires_grad", False),
        ],
    )
    def test_set_field_triggers_notification(
        self, variable, field, value, mock_server_update_notification
    ):
        """
        Test that a server's update to a Variable's attribute
        is reflected in the client.
        """
        # Server sets the field to a new value
        send_mock = mock_server_update_notification(variable.variable_id, field, value)

        # Assert that the variable was updated locally
        assert value == getattr(variable, field)

        # Assert that the client sends the correct response to the server
        self.assert_valid_update_variable_response(send_mock)

    def test_multiple_set_field_notification_order(
        self, variable, mock_server_update_notification
    ):
        """
        Test that multiple server updates to a Variable's attribute
        are propagated to the client in the correct order.
        """
        # Server sets the field to a new value
        changes = [10, 20, 30, 40, 50]
        for val in changes:
            send_mock = mock_server_update_notification(
                variable.variable_id, "data", val
            )

            # Assert that the variable was updated locally
            assert val == getattr(variable, "data")

            # Assert that the client sends the correct response to the server
            self.assert_valid_update_variable_response(send_mock)

    def test_requires_grad_method_triggers_notification(
        self, variable, mock_server_update_notification
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
            send_mock = mock_server_update_notification(
                variable.variable_id, field, value
            )

        for field, value in updates:
            # Assert that the variable was updated locally
            assert getattr(variable, field) == value

            # Assert that the client sends the correct response to the server
            self.assert_valid_update_variable_response(send_mock)

    def test_set_output_nr_triggers_notification(
        self, variable, mock_server_update_notification
    ):
        """
        Test that a server's update to a Variable's output_nr
        is reflected in the client.
        """
        # Server sets `_output_nr` to a new value
        send_mock = mock_server_update_notification(
            variable.variable_id, "output_nr", 3
        )

        # Assert that the variable was updated locally
        assert variable.output_nr == 3

        # Assert that the client sends the correct response to the server
        self.assert_valid_update_variable_response(send_mock)

    # TODO: Finalize after implementing NODE_REGISTRY
    def test_set_grad_fn_triggers_notification(
        self, variable, mock_server_update_notification
    ):
        """
        Test that a server's update to a Variable's grad_fn is reflected in the client.
        """
        # Server sets `_grad_fn` to a new value
        send_mock = mock_server_update_notification(
            variable.variable_id, "_grad_fn", "test-id-456"
        )

        # Assert that the variable was updated locally
        assert variable.grad_fn.name() == "AddBackward0"

        # Assert that the client sends the correct response to the server
        self.assert_valid_update_variable_response(send_mock)

    def test_set_grad_triggers_notification(
        self, variable, mock_server_update_notification
    ):
        """
        Test that a server's update to a Variable's grad is reflected in the client.
        """
        # Server sets `_grad` to a new value
        value = [
            {"data": "gradient", "role": "grad_1", "requires_grad": False},
            {"data": "gradient", "role": "grad_2", "requires_grad": False},
        ]
        send_mock = mock_server_update_notification(
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
        self.assert_valid_update_variable_response(send_mock)

    def test_append_grad_triggers_notification(
        self, variable, mock_server_update_notification
    ):
        """
        Test that the server calling `append_grad()` is reflected in the client.
        We emulate the server call to `append_grad()` by directly sending the
        notification to the client:
          - `_grad` is set to a new value
        """
        # Server appends first gradient
        value = [{"data": "gradient", "role": "grad_1", "requires_grad": False}]
        send_mock = mock_server_update_notification(
            variable.variable_id, "_grad", value
        )

        # Assert that the variable was updated locally
        assert len(variable.grad) == 1
        assert variable.grad[0].data == "gradient"
        assert variable.grad[0].role == "grad_1"
        assert variable.grad[0].requires_grad is False

        # Assert that the client sends the correct response to the server
        self.assert_valid_update_variable_response(send_mock)

        # Server appends second gradient (new entire _grad list is sent)
        value = [
            {"data": "gradient", "role": "grad_1", "requires_grad": False},
            {"data": "gradient", "role": "grad_2", "requires_grad": False},
        ]
        send_mock = mock_server_update_notification(
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
        self.assert_valid_update_variable_response(send_mock)

    def test_retain_grad_method_triggers_notification(
        self, variable, mock_server_update_notification
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
        send_mock = mock_server_update_notification(
            variable.variable_id, "_retain_grad", True
        )

        # Assert that the variable was updated locally
        assert variable._retain_grad is True

        # Assert that the client sends the correct response to the server
        self.assert_valid_update_variable_response(send_mock)

    def test_copy_method_triggers_notifications(
        self, variable, mock_server_update_notification
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
        send_mock = mock_server_update_notification(variable.variable_id, "data", 123)
        send_mock = mock_server_update_notification(
            variable.variable_id, "role", "copied"
        )
        send_mock = mock_server_update_notification(
            variable.variable_id, "requires_grad", False
        )

        # Assert that the variable was updated locally
        assert variable.data == 123
        assert variable.role == "copied"
        assert variable.requires_grad is False

        # Assert that the client sends the correct response to the server
        self.assert_valid_update_variable_response(send_mock)
