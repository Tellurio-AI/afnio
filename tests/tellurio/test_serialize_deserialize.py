import os
import re

import pytest

from afnio._utils import _deserialize_output, _serialize_arg
from afnio._variable import Variable
from afnio.autodiff.basic_ops import Add
from afnio.autodiff.utils import _deserialize_fn_output
from afnio.cognitive.parameter import Parameter
from afnio.models.openai import OpenAI
from afnio.tellurio import login
from afnio.tellurio._node_registry import create_node
from afnio.tellurio._variable_registry import PENDING_GRAD_FN_ASSIGNMENTS
from afnio.tellurio.run import init


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
def variables():
    x = Variable(data="abc", role="first input", requires_grad=True)
    y = Variable(data="def", role="second input", requires_grad=False)
    return x, y


class TestSerializeArg:
    """
    Tests for the _serialize_arg function, covering Variable, list, tuple, dict,
    and primitive types.
    """

    def test_serialize_arg_variable(self, variables):
        """
        Test serialization of a single Variable instance.
        """
        x, _ = variables
        serialized = _serialize_arg(x)
        assert isinstance(serialized, dict)
        assert serialized["__variable__"] is True
        assert serialized["variable_id"] == x.variable_id

    def test_serialize_arg_parameter(self):
        """
        Test serialization of a single Parameter instance.
        """
        p = Parameter(data="test", role="weight", requires_grad=True)
        serialized = _serialize_arg(p)
        assert isinstance(serialized, dict)
        assert serialized.get("__parameter__") is True
        assert serialized.get("variable_id") == p.variable_id

    def test_serialize_arg_model_client(self, monkeypatch):
        """
        Test serialization of an LM model client.
        """
        # Forcing consent to sharing API keys
        monkeypatch.setenv("ALLOW_API_KEY_SHARING", "true")

        # Create OpenAI model client
        model = OpenAI(api_key="test_key")
        serialized = _serialize_arg(model)
        assert isinstance(serialized, dict)
        assert serialized["__model_client__"] is True
        assert serialized["model_id"] == model.model_id

    def test_serialize_arg_list(self, variables):
        """
        Test serialization of a list of Variables.
        """
        x, y = variables
        serialized = _serialize_arg([x, y])
        assert isinstance(serialized, list)
        assert all("__variable__" in item for item in serialized)
        assert serialized[0]["variable_id"] == x.variable_id
        assert serialized[1]["variable_id"] == y.variable_id

    def test_serialize_arg_tuple(self, variables):
        """
        Test serialization of a tuple of Variables.
        """
        x, y = variables
        serialized = _serialize_arg((x, y))
        assert isinstance(serialized, tuple)
        assert all("__variable__" in item for item in serialized)
        assert serialized[0]["variable_id"] == x.variable_id
        assert serialized[1]["variable_id"] == y.variable_id

    def test_serialize_arg_dict(self, variables):
        """
        Test serialization of a dictionary with Variables.
        """
        x, y = variables
        serialized = _serialize_arg({"x": x, "y": y})
        assert isinstance(serialized, dict)
        assert "__variable__" in serialized["x"]
        assert "__variable__" in serialized["y"]
        assert serialized["x"]["variable_id"] == x.variable_id
        assert serialized["y"]["variable_id"] == y.variable_id

    def test_serialize_arg_primitives(self):
        """
        Test serialization of primitive types (int, str, float, bool, None).
        """
        assert _serialize_arg(42) == 42
        assert _serialize_arg("hello") == "hello"
        assert _serialize_arg(3.14) == 3.14
        assert _serialize_arg(True) is True
        assert _serialize_arg(None) is None

    def test_serialize_arg_unrecognized_type(self):
        """
        Test that _serialize_arg raises TypeError for unrecognized types.
        """

        class Dummy:
            pass

        dummy = Dummy()
        with pytest.raises(TypeError, match="Cannot serialize object of type Dummy"):
            _serialize_arg(dummy)


class TestDeserializeFnOutput:
    """
    Tests for the _deserialize_fn_output function, covering single Variable,
    list of Variables, and handling of non-Variable types.
    """

    def test_deserialize_fn_output_variable(self, variables):
        """
        Test deserialization of a single Variable instance.
        This test ensures that the deserialized Variable retains all attributes
        and correctly references its grad_fn.
        """
        # Node registration happens before variable deserialization
        node_id = "node-id-123"
        create_node({"name": "AddBackward", "node_id": node_id})

        x, y = variables
        result = Add.apply(x, y)
        assert result.grad_fn is not None

        obj = {
            "variable_id": result.variable_id,
            "data": result.data,
            "role": result.role,
            "requires_grad": result.requires_grad,
            "_retain_grad": result._retain_grad,
            "_grad": result._grad,
            "_output_nr": result.output_nr,
            "_grad_fn": node_id,
            "is_leaf": result.is_leaf,
        }
        var = _deserialize_fn_output(obj)
        assert isinstance(var, Variable)
        assert var.variable_id == result.variable_id
        assert var.data == result.data
        assert var.role == result.role
        assert var.requires_grad is result.requires_grad
        assert var._retain_grad is result._retain_grad
        assert var._grad == result._grad
        assert var.output_nr == result.output_nr
        assert var.grad_fn.node_id == node_id
        assert var.is_leaf is result.is_leaf

    def test_deserialize_fn_output_variable_with_pending_grad_fn(self, variables):
        """
        Test deserialization of a Variable with a pending grad_fn assignment.
        This simulates a scenario where the grad_fn is not immediately available,
        and checks that the deserialization correctly handles the pending assignment.
        """
        x, y = variables
        result = Add.apply(x, y)
        assert result.grad_fn is not None

        node_id = "node-id-456"
        obj = {
            "variable_id": result.variable_id,
            "data": result.data,
            "role": result.role,
            "requires_grad": result.requires_grad,
            "_retain_grad": result._retain_grad,
            "_grad": result._grad,
            "_output_nr": result.output_nr,
            "_grad_fn": node_id,
            "is_leaf": result.is_leaf,
        }
        var = _deserialize_fn_output(obj)
        assert isinstance(var, Variable)
        assert var.variable_id == result.variable_id
        assert var.data == result.data
        assert var.role == result.role
        assert var.requires_grad is result.requires_grad
        assert var._retain_grad is result._retain_grad
        assert var._grad == result._grad
        assert var.output_nr == result.output_nr
        assert var.is_leaf is result.is_leaf

        # Assert PENDING_GRAD_FN_ASSIGNMENTS contains the variable under node_id
        assert node_id in PENDING_GRAD_FN_ASSIGNMENTS
        assert var in PENDING_GRAD_FN_ASSIGNMENTS[node_id]

        with pytest.raises(
            RuntimeError,
            match=re.escape(
                f"Timeout waiting for _pending_grad_fn_id to be cleared "
                f"for variable_id={result.variable_id}"
            ),
        ):
            assert var.grad_fn.node_id == node_id

        # Node registration happens after variable deserialization
        create_node({"name": "AddBackward", "node_id": node_id})

        # After registering, the pending assignment should be cleared
        assert node_id not in PENDING_GRAD_FN_ASSIGNMENTS

        assert var.grad_fn.node_id == node_id

    def test_deserialize_fn_output_list(self, variables):
        """
        Test deserialization of a list of Variable instances.
        This test ensures that the deserialized list retains all Variable attributes
        and correctly references their grad_fns.
        """
        x, y = variables
        obj_list = [
            {
                "variable_id": x.variable_id,
                "data": x.data,
                "role": x.role,
                "requires_grad": x.requires_grad,
                "_retain_grad": False,
                "_grad": [],
                "_output_nr": 0,
                "_grad_fn": None,
                "is_leaf": True,
            },
            {
                "variable_id": y.variable_id,
                "data": y.data,
                "role": y.role,
                "requires_grad": y.requires_grad,
                "_retain_grad": False,
                "_grad": [],
                "_output_nr": 0,
                "_grad_fn": None,
                "is_leaf": True,
            },
        ]
        result = _deserialize_fn_output(obj_list)
        assert isinstance(result, tuple)
        assert all(isinstance(v, Variable) for v in result)

    def test_deserialize_fn_output_invalid_type(self):
        """
        Test that _deserialize_fn_output raises TypeError for unsupported types.
        """
        invalid_obj = 12345  # int is not a supported type for deserialization

        with pytest.raises(
            TypeError,
            match=re.escape(
                "Deserialization only supports Variable or Tuple[Variable], "
                "but got: <class 'int'>"
            ),
        ):
            _deserialize_fn_output(invalid_obj)


class TestDeserializeOutput:
    """
    Tests for the _deserialize_output function, covering Variable, Parameters, list,
    tuple, dict, and primitive types.
    """

    def test_deserialize_variable(self, variables):
        """
        Test deserialization of a variable object.
        """
        x, _ = variables
        obj = {"__variable__": True, "variable_id": x.variable_id}
        assert _deserialize_output(obj) is x

    def test_deserialize_parameter(self, variables):
        """
        Test deserialization of a parameter object.
        """
        param = Parameter(data="test", role="weight", requires_grad=True)
        obj = {"__parameter__": True, "variable_id": param.variable_id}
        assert _deserialize_output(obj) is param

    def test_deserialize_model(self, monkeypatch):
        """
        Test deserialization of a model object.
        """
        # Forcing consent to sharing API keys
        monkeypatch.setenv("ALLOW_API_KEY_SHARING", "true")

        model = OpenAI(api_key="test_key")
        obj = {"__model_client__": True, "model_id": model.model_id}
        assert _deserialize_output(obj) is model

    def test_deserialize_dict(self, variables):
        """
        Test deserialization of a dictionary containing a variable.
        """
        x, _ = variables
        obj = {"a": 1, "b": {"__variable__": True, "variable_id": x.variable_id}}
        result = _deserialize_output(obj)
        assert result == {"a": 1, "b": x}

    def test_deserialize_list(self, variables):
        """
        Test deserialization of a list containing a variable.
        """
        x, _ = variables
        obj = [1, {"__variable__": True, "variable_id": x.variable_id}, 3]
        result = _deserialize_output(obj)
        assert result == [1, x, 3]

    def test_deserialize_tuple(self, monkeypatch):
        """
        Test deserialization of a tuple containing a model.
        """
        # Forcing consent to sharing API keys
        monkeypatch.setenv("ALLOW_API_KEY_SHARING", "true")

        model = OpenAI(api_key="test_key")
        obj = (1, {"__model_client__": True, "model_id": model.model_id})
        result = _deserialize_output(obj)
        assert result == (1, model)

    def test_deserialize_primitive(self):
        """
        Test deserialization of primitive types (None, int, float, str, bool).
        """
        for val in [None, 42, 3.14, "foo", True, False]:
            assert _deserialize_output(val) == val

    def test_deserialize_variable_not_found(self):
        """
        Test deserialization of a variable that does not exist in the context.
        """
        obj = {"__variable__": True, "variable_id": "notfound"}
        with pytest.raises(
            ValueError,
            match=re.escape("Variable with variable_id 'notfound' not found"),
        ):
            _deserialize_output(obj)

    def test_deserialize_parameter_not_found(self):
        """
        Test deserialization of a parameter that does not exist in the context.
        """
        obj = {"__parameter__": True, "variable_id": "notfound"}
        with pytest.raises(
            ValueError,
            match=re.escape("Parameter with variable_id 'notfound' not found"),
        ):
            _deserialize_output(obj)

    def test_deserialize_model_not_found(self):
        """
        Test deserialization of a model that does not exist in the context.
        """
        obj = {"__model_client__": True, "model_id": "notfound"}
        with pytest.raises(
            ValueError,
            match=re.escape("Model with model_id 'notfound' not found"),
        ):
            _deserialize_output(obj)

    def test_deserialize_unexpected_type(self):
        """
        Test deserialization of an unexpected input type (e.g., set).
        Should raise a TypeError.
        """
        obj = {1, 2, 3}
        with pytest.raises(
            TypeError, match=r"Cannot deserialize object of type .*set.*"
        ):
            _deserialize_output(obj)
