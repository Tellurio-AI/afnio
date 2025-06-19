import os

import pytest

from afnio._variable import Variable
from afnio.models.openai import AsyncOpenAI
from afnio.optim import tgd
from afnio.tellurio import login
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


class TestClientToServerTgdSync:

    def test_run_optimizer_tgd(self, monkeypatch):
        """
        Test that running a functional TGD optimizer step with multiple parameters
        works correctly.
        """
        # Clear VARIABLE_REGISTRY to ensure a clean state
        # when searching for the new deepcopied parameters
        VARIABLE_REGISTRY.clear()

        # Placeholder for Textual Gradient Descent optimizer messages
        TGD_MESSAGES = [
            {
                "role": "system",
                "content": [
                    Variable(
                        data="Placeholder for Textual Gradient Descent optimizer system prompt",  # noqa: E501
                        role="Textual Gradient Descent optimizer system prompt",
                    )
                ],
            },
            {
                "role": "user",
                "content": [
                    Variable(
                        data="Placeholder for Textual Gradient Descent optimizer user prompt",  # noqa: E501
                        role="Textual Gradient Descent optimizer user prompt",
                    )
                ],
            },
        ]

        # Forcing consent to sharing API keys
        monkeypatch.setenv("ALLOW_API_KEY_SHARING", "true")

        # Create OpenAI model client and TGD optimizer
        api_key = os.getenv("OPENAI_API_KEY", "sk-test-1234567890abcdef")
        optim_model_client = AsyncOpenAI(api_key=api_key)

        # Create parameters, gradients, and momentum buffers
        param = Variable(data="Initial value", role="parameter", requires_grad=True)
        params = [param]
        grad_1 = Variable(data="Translate the parameter to Italian", role="gradient")
        grad_2 = Variable(data="Only use capital letters", role="gradient")
        grads = [[grad_1, grad_2]]
        momentum_buffer_list = [[]]

        # Take a snapshot of variable objects before tgd()
        before_vars = set(VARIABLE_REGISTRY.values())

        tgd(
            params=params,
            grads=grads,
            momentum_buffer_list=momentum_buffer_list,
            model_client=optim_model_client,
            messages=TGD_MESSAGES,
            inputs={},
            constraints=[],
            momentum=2,
            model="gpt-4o",
        )

        assert params[0].data == "VALORE INIZIALE"

        # Take a snapshot after tgd()
        after_vars = set(VARIABLE_REGISTRY.values())
        new_vars = after_vars - before_vars
        new_var = next(iter(new_vars))

        assert len(momentum_buffer_list[0]) == 1
        assert momentum_buffer_list == [
            [
                (
                    new_var,
                    [
                        grad_1,
                        grad_2,
                    ],
                ),
            ],
        ]
