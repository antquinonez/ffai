# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ffai.core.client_base import FFAIClientBase

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_mistral_response():
    """Mock response from Mistral API."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "This is a test response."
    response.choices[0].message.tool_calls = None
    response.usage = None
    return response


@pytest.fixture
def mock_mistral_client(mock_mistral_response):
    """Mock Mistral client."""
    client = MagicMock()
    client.chat.complete.return_value = mock_mistral_response
    return client


@pytest.fixture
def mock_openai_response():
    """Mock response from OpenAI-compatible API."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "This is a test response."
    response.choices[0].message.tool_calls = None
    response.usage = None
    return response


@pytest.fixture
def mock_openai_client(mock_openai_response):
    """Mock OpenAI client."""
    client = MagicMock()
    client.chat.completions.create.return_value = mock_openai_response
    return client


@pytest.fixture
def mock_ffmistralsmall(mock_mistral_client):
    """Mock FFMistralSmall instance."""
    from ffai.Clients.FFMistralSmall import FFMistralSmall

    with patch.object(FFMistralSmall, "_initialize_client", return_value=mock_mistral_client):
        client = FFMistralSmall(
            api_key="test-api-key",
            model="mistral-small-2503",
            temperature=0.8,
            max_tokens=128000,
        )
    yield client


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return [
        {
            "sequence": 1,
            "prompt_name": "greeting",
            "prompt": "Hello, how are you?",
            "history": None,
        },
        {
            "sequence": 2,
            "prompt_name": "math",
            "prompt": "What is 2 + 2?",
            "history": None,
        },
        {
            "sequence": 3,
            "prompt_name": "followup",
            "prompt": "What was the answer to my math question?",
            "history": ["math", "greeting"],
        },
    ]


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": "mistral-small-2503",
        "api_key_env": "MISTRALSMALL_KEY",
        "max_retries": 3,
        "temperature": 0.8,
        "max_tokens": 4096,
        "system_instructions": "You are a helpful assistant.",
    }


class ConcreteClient(FFAIClientBase):
    """Minimal concrete implementation of FFAIClientBase for testing."""

    def __init__(self):
        self._history: list[dict[str, Any]] = []
        self.model = "test-model"
        self.system_instructions = ""

    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        return ""

    def clear_conversation(self) -> None:
        self._history = []

    def get_conversation_history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def set_conversation_history(self, history: list[dict[str, Any]]) -> None:
        self._history = list(history)

    def clone(self) -> "ConcreteClient":
        c = ConcreteClient()
        c._history = list(self._history)
        return c


@pytest.fixture
def concrete_client():
    """Provide a ConcreteClient instance for tests that need a real FFAIClientBase."""
    return ConcreteClient()


class FakeEmbeddings:
    """Deterministic fake embeddings for testing Memory without network.

    Maps each unique text to a stable synthetic vector via SHA-256 hash,
    so identical texts produce identical vectors and different texts
    produce uncorrelated vectors. Suitable for predictable ranking
    assertions in unit tests.

    Defined in ``tests/conftest.py`` (not as a fixture-only helper) so
    tests can instantiate it with custom ``dim`` via direct import:
    ``from conftest import FakeEmbeddings``. The ``fake_embeddings``
    fixture below provides a default-dim instance.

    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def _vec(self, text: str) -> list[float]:
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        return [((h[i % len(h)] / 255.0) - 0.5) for i in range(self.dim)]


@pytest.fixture
def fake_embeddings():
    """Default FakeEmbeddings instance (dim=8) for memory tests."""
    return FakeEmbeddings(dim=8)
