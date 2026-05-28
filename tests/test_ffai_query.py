from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.FFAI import FFAI
from src.rag.types import QueryResult, SearchHit


class TestFFAIQuery:
    def test_raises_without_rag(self, concrete_client):
        ffai = FFAI(client=concrete_client)
        with pytest.raises(ValueError, match="RAG is not configured"):
            ffai.query("test")

    def test_delegates_to_rag_query(self, concrete_client):
        mock_rag = MagicMock()
        expected = QueryResult(
            answer="Paris",
            hits=[SearchHit(content="Paris is the capital.", score=0.9, source="geo.txt")],
            sources=["geo.txt"],
            prompt="ctx: Paris is the capital.\nQ: capital of France?",
        )
        mock_rag.query.return_value = expected
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        result = ffai.query("capital of France?")
        assert result.answer == "Paris"
        assert result.sources == ["geo.txt"]
        mock_rag.query.assert_called_once()

    def test_passes_top_k(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.query.return_value = QueryResult(answer="a", hits=[], sources=[], prompt="p")
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        ffai.query("q?", top_k=10)
        call_kwargs = mock_rag.query.call_args
        assert call_kwargs[1]["top_k"] == 10

    def test_passes_custom_template(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.query.return_value = QueryResult(answer="a", hits=[], sources=[], prompt="p")
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        template = "Custom: {context}\n{question}"
        ffai.query("q?", prompt_template=template)
        call_kwargs = mock_rag.query.call_args
        assert call_kwargs[1]["prompt_template"] == template

    def test_client_adapter_wraps_client(self, concrete_client):
        concrete_client.generate_response = lambda prompt, **kw: f"reply to: {prompt[:10]}"
        mock_rag = MagicMock()

        def capture_query(*args, **kwargs):
            generate_fn = kwargs["generate_fn"]
            answer = generate_fn("test prompt")
            return QueryResult(answer=answer, hits=[], sources=[], prompt="test prompt")

        mock_rag.query.side_effect = capture_query
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        result = ffai.query("q?")
        assert result.answer == "reply to: test promp"

    def test_no_rag_param_stores_none(self, concrete_client):
        ffai = FFAI(client=concrete_client)
        assert ffai.rag is None

    def test_rag_param_stored(self, concrete_client):
        mock_rag = MagicMock()
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        assert ffai.rag is mock_rag

    def test_passes_filters(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.query.return_value = QueryResult(answer="a", hits=[], sources=[], prompt="p")
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        ffai.query("q?", source="doc1")
        call_kwargs = mock_rag.query.call_args
        assert call_kwargs[1]["source"] == "doc1"

    def test_existing_tests_unaffected(self, concrete_client):
        ffai = FFAI(client=concrete_client)
        assert hasattr(ffai, "generate_response")
        assert hasattr(ffai, "history")

    def test_query_returns_query_result_instance(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.query.return_value = QueryResult(answer="a", hits=[], sources=[], prompt="p")
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        result = ffai.query("q?")
        assert isinstance(result, QueryResult)
        assert result.answer == "a"

    def test_query_raises_if_rag_returns_wrong_type(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.query.return_value = "not a QueryResult"
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        with pytest.raises(AssertionError):
            ffai.query("q?")


class TestFFAIAQuery:
    def test_aquery_raises_without_rag(self, concrete_client):
        ffai = FFAI(client=concrete_client)
        with pytest.raises(ValueError, match="RAG is not configured"):
            asyncio.run(ffai.aquery("test"))

    def test_aquery_delegates_to_rag_aquery(self, concrete_client):
        mock_rag = MagicMock()
        expected = QueryResult(
            answer="Paris",
            hits=[SearchHit(content="Paris is the capital.", score=0.9, source="geo.txt")],
            sources=["geo.txt"],
            prompt="ctx: Paris is the capital.\nQ: capital?",
        )
        mock_rag.aquery = AsyncMock(return_value=expected)
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        result = asyncio.run(ffai.aquery("capital of France?"))
        assert result.answer == "Paris"
        assert result.sources == ["geo.txt"]
        mock_rag.aquery.assert_called_once()

    def test_aquery_passes_top_k(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.aquery = AsyncMock(return_value=QueryResult(answer="a", hits=[], sources=[], prompt="p"))
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        asyncio.run(ffai.aquery("q?", top_k=10))
        call_kwargs = mock_rag.aquery.call_args
        assert call_kwargs[1]["top_k"] == 10

    def test_aquery_client_adapter_wraps_client(self, concrete_client):
        concrete_client.generate_response = lambda prompt, **kw: f"reply to: {prompt[:10]}"
        mock_rag = MagicMock()

        async def capture_aquery(*args, **kwargs):
            generate_fn = kwargs["generate_fn"]
            answer = generate_fn("test prompt")
            return QueryResult(answer=answer, hits=[], sources=[], prompt="test prompt")

        mock_rag.aquery = AsyncMock(side_effect=capture_aquery)
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        result = asyncio.run(ffai.aquery("q?"))
        assert result.answer == "reply to: test promp"

    def test_aquery_passes_custom_template(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.aquery = AsyncMock(return_value=QueryResult(answer="a", hits=[], sources=[], prompt="p"))
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        template = "Custom: {context}\n{question}"
        asyncio.run(ffai.aquery("q?", prompt_template=template))
        call_kwargs = mock_rag.aquery.call_args
        assert call_kwargs[1]["prompt_template"] == template

    def test_aquery_returns_query_result_instance(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.aquery = AsyncMock(return_value=QueryResult(answer="a", hits=[], sources=[], prompt="p"))
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        result = asyncio.run(ffai.aquery("q?"))
        assert isinstance(result, QueryResult)
        assert result.answer == "a"
