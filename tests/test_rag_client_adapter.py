from __future__ import annotations

from unittest.mock import MagicMock

from src.rag.client_adapter import ClientAdapter


class TestClientAdapter:
    def test_calls_generate_response_with_prompt(self):
        client = MagicMock()
        client.generate_response.return_value = "answer"
        adapter = ClientAdapter(client)
        result = adapter("test prompt")
        assert result == "answer"
        client.generate_response.assert_called_once_with(prompt="test prompt")

    def test_passes_kwargs_to_generate_response(self):
        client = MagicMock()
        client.generate_response.return_value = "answer"
        adapter = ClientAdapter(client, model="gpt-4")
        adapter("hello")
        client.generate_response.assert_called_once_with(prompt="hello", model="gpt-4")

    def test_is_callable(self):
        adapter = ClientAdapter(MagicMock())
        assert callable(adapter)

    def test_stores_client(self):
        client = MagicMock()
        adapter = ClientAdapter(client)
        assert adapter._client is client
