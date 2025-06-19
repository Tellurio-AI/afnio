import os

import pytest

from afnio._variable import Variable
from afnio.cognitive.parameter import Parameter
from afnio.models.openai import AsyncOpenAI
from afnio.optim import TGD
from afnio.tellurio import login
from afnio.tellurio._optimizer_registry import OPTIMIZER_REGISTRY
from afnio.tellurio._variable_registry import VARIABLE_REGISTRY
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
def parameter():
    """
    Fixture to create a Parameter instance.
    """
    # Create a parameter to optimize
    return Parameter(data="Initial value", role="parameter", requires_grad=True)


@pytest.fixture
def tgd_optimizer(parameter, monkeypatch):
    """
    Fixture to create a TGD Optimizer instance.
    """
    # Forcing consent to sharing API keys
    monkeypatch.setenv("ALLOW_API_KEY_SHARING", "true")

    # Create OpenAI model client
    api_key = os.getenv("OPENAI_API_KEY", "sk-test-1234567890abcdef")
    optim_model_client = AsyncOpenAI(api_key=api_key)

    # Create TGD optimizer
    optimizer = TGD(
        [parameter], model_client=optim_model_client, momentum=3, model="gpt-4o"
    )

    # Assert initial state of the optimizer
    messages = optimizer.defaults["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert (
        messages[0]["content"][0].data
        == "Placeholder for Textual Gradient Descent optimizer system prompt"
    )
    assert (
        messages[1]["content"][0].data
        == "Placeholder for Textual Gradient Descent optimizer user prompt"
    )

    defaults = {
        "model_client": optim_model_client,
        "messages": messages,
        "inputs": {},
        "constraints": [],
        "momentum": 3,
        "completion_args": {"model": "gpt-4o"},
    }
    assert optimizer.state == []
    assert optimizer.defaults == defaults
    defaults["params"] = [parameter]
    assert optimizer.param_groups == [defaults]

    return optimizer


class TestClientToServerOptimizerSync:

    def test_create_optimizer(self, tgd_optimizer):
        """
        Test that creating a TGD optimizer registers it in the OPTIMIZER_REGISTRY
        and assigns an optimizer_id.
        """
        assert tgd_optimizer.optimizer_id is not None
        assert tgd_optimizer.optimizer_id in OPTIMIZER_REGISTRY
        assert OPTIMIZER_REGISTRY[tgd_optimizer.optimizer_id] is tgd_optimizer

    def test_run_step(self, parameter, tgd_optimizer):
        """
        Test that running a step on the TGD optimizer returns the expected loss.
        """

        assert parameter.data == "Initial value"

        gradient = Variable(data="Use only capital letters", role="gradient")
        parameter.append_grad(gradient)

        # Take a snapshot of variable objects before tgd()
        before_vars = set(VARIABLE_REGISTRY.values())

        def closure():
            # Simulate a loss calculation
            return (
                Variable(data=10, role="score"),
                Variable(data="Explanation", role="explanation"),
            )

        loss = tgd_optimizer.step(closure)
        score, explanation = loss
        assert isinstance(loss, tuple)
        assert len(loss) == 2
        assert isinstance(score, Variable)
        assert isinstance(explanation, Variable)
        assert score.data == 10
        assert explanation.data == "Explanation"

        assert parameter.data == "INITIAL VALUE"

        # Take a snapshot after tgd()
        after_vars = set(VARIABLE_REGISTRY.values())
        new_vars = after_vars - before_vars - {score, explanation}
        new_var = next(iter(new_vars))

        assert tgd_optimizer.state == {
            parameter: {
                "momentum_buffer": [
                    (
                        new_var,
                        [gradient],
                    )
                ],
            }
        }

    def test_run_step_clear_pending_data(self, monkeypatch):
        """
        Test that _pending_data is set during optimizer step
        and cleared after the update.

        This uses a single parameter and a TGD optimizer.
        """
        param_1 = Variable(data="Initial value 1", role="parameter", requires_grad=True)
        param_2 = Variable(data="Initial value 2", role="parameter", requires_grad=True)
        param_3 = Variable(data="Initial value 3", role="parameter", requires_grad=True)
        param_4 = Variable(data="Initial value 4", role="parameter", requires_grad=True)
        param_5 = Variable(data="Initial value 5", role="parameter", requires_grad=True)

        gradient = Variable(data="Use only capital letters", role="gradient")
        p_list = [param_1, param_2, param_3, param_4, param_5]
        for param in p_list:
            param.append_grad(gradient)

        # Forcing consent to sharing API keys
        monkeypatch.setenv("ALLOW_API_KEY_SHARING", "true")

        # Create OpenAI model client and TGD optimizer
        api_key = os.getenv("OPENAI_API_KEY", "sk-test-1234567890abcdef")
        optim_model_client = AsyncOpenAI(api_key=api_key)
        optimizer = TGD(
            p_list, model_client=optim_model_client, momentum=3, model="gpt-4o"
        )
        optimizer.step()

        # The parameters should have _pending_data set before clear_step
        for param in p_list:
            assert (
                param._pending_data is True
            )  # This rarely fails (when server is too fast)

        # We are only able to read `param.data` when we exit `_wait_for_pending()`
        # and at that point, _pending_data should be False
        for param in p_list:
            assert isinstance(param.data, str)
            assert param.data == param.data.upper()
            assert param._pending_data is False

    def test_clear_grad(self, parameter, tgd_optimizer):
        """
        Test that clear_grad clears the gradients of the parameters and
        does not create a new grad list object.
        """
        grad_list_id = id(parameter.grad)  # Get the ID of the grad list before clearing

        # Append a gradient to the parameter
        assert len(parameter.grad) == 0
        gradient = Variable(data="Use only capital letters", role="gradient")
        parameter.append_grad(gradient)

        # Check initial gradient
        assert len(parameter.grad) == 1
        assert parameter.grad[0].data == "Use only capital letters"

        # Clear gradients
        tgd_optimizer.clear_grad()

        # Check that gradients are cleared and the grad list object is the same
        assert len(parameter.grad) == 0
        assert grad_list_id == id(parameter.grad)

    def test_add_param_group(self, parameter, tgd_optimizer):
        """
        Test that adding a new parameter group to the optimizer works correctly and
        does not create a new param_groups list object.
        """
        # Get the ID of the param_groups before adding
        param_groups_id = id(tgd_optimizer.param_groups)

        # Check initial state of the optimizer
        assert len(tgd_optimizer.param_groups) == 1
        assert tgd_optimizer.param_groups[0]["params"] == [parameter]

        # Add a new parameter group
        new_param = Variable(data="New parameter", role="parameter", requires_grad=True)
        new_param_group = {"params": [new_param]}
        tgd_optimizer.add_param_group(new_param_group)

        # Check the updated state of the optimizer
        assert len(tgd_optimizer.param_groups) == 2
        assert tgd_optimizer.param_groups[1]["params"] == [new_param]

        # Ensure the param_groups list object is the same
        assert param_groups_id == id(tgd_optimizer.param_groups)
