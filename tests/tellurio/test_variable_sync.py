import os
import re

import pytest
import pytest_asyncio

import afnio as hf
from afnio._variable import _allow_grad_fn_assignment
from afnio.tellurio import login
from afnio.tellurio._eventloop import run_in_background_loop
from afnio.tellurio.client import get_default_client


class TestClientToServerVariableSync:
    """
    Test the synchronization of variables between the client and server.
    This test suite uses the TellurioWebSocketClient to create and manipulate variables,
    and verifies that changes are reflected on the server.
    """

    @pytest_asyncio.fixture(autouse=True)
    async def login(self):
        """
        Test the login function with real HTTP and WebSocket connections.
        """
        api_key = os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key")
        login_info = login(api_key=api_key)
        self.session_id = login_info["session_id"]
        self.api_key = api_key

    @pytest.fixture
    def variable(self):
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

    def fetch_server_variable(self, variable_id):
        _, ws_client = get_default_client()
        response = run_in_background_loop(
            ws_client.call("get_variable", {"variable_id": variable_id})
        )
        print(response)
        return response["result"]

    def test_create_variable(self, variable):
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

    # TODO: Implement `afnio.autodiff.basic_ops` operations
    def test_retain_grad_method_triggers_notification(self, variable):
        """Test that calling retain_grad() triggers a notification."""
        variable += variable

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

    pass
