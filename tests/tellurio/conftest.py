import os

import pytest

from afnio.models.openai import AsyncOpenAI
from afnio.tellurio._model_registry import MODEL_REGISTRY


@pytest.fixture
def model(monkeypatch):
    """
    Fixture to create an LM model instance.
    """
    # Forcing consent to sharing API keys
    monkeypatch.setenv("ALLOW_API_KEY_SHARING", "true")

    # Create OpenAI model client
    api_key = os.getenv("OPENAI_API_KEY", "sk-test-1234567890abcdef")
    model = AsyncOpenAI(api_key=api_key)
    # Ensure model is registered
    assert model.model_id in MODEL_REGISTRY
    return model
