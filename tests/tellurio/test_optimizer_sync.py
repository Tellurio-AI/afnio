import os

import pytest
import pytest_asyncio

from afnio.cognitive.parameter import Parameter
from afnio.models.openai import AsyncOpenAI
from afnio.optim import TGD
from afnio.tellurio import login
from afnio.tellurio._optimizer_registry import OPTIMIZER_REGISTRY


@pytest.fixture
def tgd_optimizer():
    """
    Fixture to create a TGD Optimizer instance.
    """
    # Create a parameter to optimize
    param = Parameter(data="Initial value", role="parameter", requires_grad=True)

    # Create OpenAI model client
    optim_model_client = AsyncOpenAI(api_key="1234567890", organization="test-org")

    # Create TGD optimizer
    optimizer = TGD(
        [param], model_client=optim_model_client, momentum=3, model="gpt-4o"
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
    defaults["params"] = [param]
    assert optimizer.param_groups == [defaults]

    return optimizer


@pytest_asyncio.fixture(autouse=True)
async def login_fixture():
    """
    Test the login function with real HTTP and WebSocket connections.
    """
    api_key = os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key")
    login(api_key=api_key)
    api_key = api_key


class TestClientToServerOptimizerSync:

    def test_create_optimizer(self, tgd_optimizer):
        """
        Test that creating a TGD optimizer registers it in the OPTIMIZER_REGISTRY
        and assigns an optimizer_id.
        """
        assert tgd_optimizer.optimizer_id is not None
        assert tgd_optimizer.optimizer_id in OPTIMIZER_REGISTRY
        assert OPTIMIZER_REGISTRY[tgd_optimizer.optimizer_id] is tgd_optimizer
