import os
import re

import pytest
import pytest_asyncio

from afnio._variable import Variable
from afnio.autodiff.basic_ops import Add, Split
from afnio.autodiff.function import _deserialize_output, _serialize_arg
from afnio.models.openai import OpenAI
from afnio.tellurio import login
from afnio.tellurio._node_registry import create_node
from afnio.tellurio._variable_registry import PENDING_GRAD_FN_ASSIGNMENTS


@pytest_asyncio.fixture(autouse=True)
async def login_fixture():
    """
    Test the login function with real HTTP and WebSocket connections.
    """
    api_key = os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key")
    login(api_key=api_key)
    api_key = api_key


@pytest.fixture
def variables():
    x = Variable(data="abc", role="first input", requires_grad=True)
    y = Variable(data="def", role="second input", requires_grad=False)
    return x, y


class TestFunctionSync:
    """Tests for synchronous function operations in afnio.autodiff."""

    def test_add_apply(self, variables):
        """
        Test the Add function's apply method with two variables.
        """
        x, y = variables
        result = Add.apply(x, y)
        assert isinstance(result, Variable)
        assert result.data == "abcdef"
        assert result.role == "first input and second input"
        assert result.requires_grad is True
        assert "<AddBackward" in str(result.grad_fn)
        assert "<afnio.autodiff.function.AddBackward" in repr(result.grad_fn)
        assert len(result.grad_fn.next_functions) == 2
        assert "<AccumulateGrad" in str(result.grad_fn.next_functions[0].node)
        result.grad_fn.next_functions[0].output_nr == 0
        assert "None" in str(result.grad_fn.next_functions[1].node)
        result.grad_fn.next_functions[1].output_nr == 0

    def test_split_apply(self, variables):
        """
        Test the Split function's apply method with a variable.
        """
        x, _ = variables
        x.data = "a b c"
        result = Split.apply(x, sep=" ")
        assert isinstance(result, tuple)
        assert all(isinstance(v, Variable) for v in result)
        assert [v.data for v in result] == ["a", "b", "c"]
        expected_roles = [f"split part {i} of first input" for i in range(len(result))]
        for i, v in enumerate(result):
            assert v.role == expected_roles[i]
            assert v.requires_grad is True
            assert "<SplitBackward" in str(result[i].grad_fn)
            assert "<afnio.autodiff.function.SplitBackward" in repr(result[i].grad_fn)
            assert len(v.grad_fn.next_functions) == 1
            assert "<AccumulateGrad" in str(v.grad_fn.next_functions[0].node)
            assert v.grad_fn.next_functions[0].output_nr == 0


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

    def test_serialize_arg_model_client(self):
        """
        Test serialization of an LM model client.
        """
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


class TestDeserializeOutput:
    """
    Tests for the _deserialize_output function, covering single Variable,
    list of Variables, and handling of non-Variable types.
    """

    def test_deserialize_output_variable(self, variables):
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
        var = _deserialize_output(obj)
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

    def test_deserialize_output_variable_with_pending_grad_fn(self, variables):
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
        var = _deserialize_output(obj)
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
                f"Timeout waiting for grad_fn node "
                f"for variable_id={result.variable_id} "
                f"(waiting for node_id={node_id})"
            ),
        ):
            assert var.grad_fn.node_id == node_id

        # Node registration happens after variable deserialization
        create_node({"name": "AddBackward", "node_id": node_id})

        # After registering, the pending assignment should be cleared
        assert node_id not in PENDING_GRAD_FN_ASSIGNMENTS

        assert var.grad_fn.node_id == node_id

    def test_deserialize_output_list(self, variables):
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
        result = _deserialize_output(obj_list)
        assert isinstance(result, tuple)
        assert all(isinstance(v, Variable) for v in result)

    def test_deserialize_output_invalid_type(self):
        """
        Test that _deserialize_output raises TypeError for unsupported types.
        """
        invalid_obj = 12345  # int is not a supported type for deserialization

        with pytest.raises(
            TypeError,
            match=re.escape(
                "Deserialization only supports Variable or Tuple[Variable], "
                "but got: <class 'int'>"
            ),
        ):
            _deserialize_output(invalid_obj)
