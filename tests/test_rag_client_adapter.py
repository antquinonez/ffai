from __future__ import annotations

from unittest.mock import MagicMock

from src.rag.client_adapter import ClientAdapter
from src.rag.types import GenerationResult


class TestClientAdapter:
    def test_calls_generate_response_with_prompt(self):
        client = MagicMock()
        client.generate_response.return_value = "answer"
        client.last_usage = None
        client.last_cost_usd = 0.0
        adapter = ClientAdapter(client)
        result = adapter("test prompt")
        assert isinstance(result, GenerationResult)
        assert result.text == "answer"
        client.generate_response.assert_called_once_with(prompt="test prompt")

    def test_passes_kwargs_to_generate_response(self):
        client = MagicMock()
        client.generate_response.return_value = "answer"
        client.last_usage = None
        client.last_cost_usd = 0.0
        adapter = ClientAdapter(client, model="gpt-4")
        result = adapter("hello")
        assert result.text == "answer"
        client.generate_response.assert_called_once_with(prompt="hello", model="gpt-4")

    def test_is_callable(self):
        adapter = ClientAdapter(MagicMock())
        assert callable(adapter)

    def test_stores_client(self):
        client = MagicMock()
        adapter = ClientAdapter(client)
        assert adapter._client is client

    def test_returns_generation_result_with_usage(self):
        usage = MagicMock()
        client = MagicMock()
        client.generate_response.return_value = "answer"
        client.last_usage = usage
        client.last_cost_usd = 0.003
        client.last_duration_ms = 150.0
        adapter = ClientAdapter(client)
        result = adapter("prompt")
        assert result.text == "answer"
        assert result.usage is usage
        assert result.cost_usd == 0.003
        assert result.duration_ms == 150.0

    def test_reads_usage_after_generate_response(self):
        call_log = []

        def fake_generate(prompt, **kw):
            call_log.append("generate")
            client.last_usage = {"tokens": 42}
            client.last_cost_usd = 0.005
            return "answer"

        client = MagicMock()
        client.last_usage = None
        client.last_cost_usd = 0.0
        client.generate_response.side_effect = fake_generate

        adapter = ClientAdapter(client)
        result = adapter("prompt")
        assert result.text == "answer"
        assert result.usage == {"tokens": 42}
        assert result.cost_usd == 0.005
        assert call_log == ["generate"]

    def test_returns_defaults_when_no_usage(self):
        client = MagicMock(spec=["generate_response"])
        client.generate_response.return_value = "answer"
        adapter = ClientAdapter(client)
        result = adapter("prompt")
        assert result.text == "answer"
        assert result.usage is None
        assert result.cost_usd == 0.0
        assert result.duration_ms is None
