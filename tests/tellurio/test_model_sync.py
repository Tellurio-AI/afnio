import os

import pytest
import pytest_asyncio

from afnio.models.openai import AsyncOpenAI
from afnio.tellurio import login
from afnio.tellurio._model_registry import MODEL_REGISTRY


@pytest_asyncio.fixture(autouse=True)
async def login_fixture():
    """
    Test the login function with real HTTP and WebSocket connections.
    """
    api_key = os.getenv("TEST_ACCOUNT_API_KEY", "valid_api_key")
    login(api_key=api_key)
    api_key = api_key


class TestClientToServerModelSync:
    def test_create_model(self):
        """
        Test that creating a model registers it in the MODEL_REGISTRY
        and assigns a model_id.
        """
        model = AsyncOpenAI(api_key="1234567890", organization="test-org")

        assert model.model_id is not None
        assert model.model_id in MODEL_REGISTRY
        assert MODEL_REGISTRY[model.model_id] is model

    def test_missing_api_key_raises(self, monkeypatch):
        """
        Test that not passing an api_key and not setting the OPENAI_API_KEY env variable
        raises an exception.
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(Exception):
            AsyncOpenAI()
