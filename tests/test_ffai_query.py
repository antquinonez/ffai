from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ffai.FFAI import FFAI
from ffai.rag.types import QueryResult, SearchHit


class TestFFAIQuery:
    def test_raises_without_rag(self, concrete_client):
        ffai = FFAI(client=concrete_client)
        with pytest.raises(AttributeError):
            ffai.rag.query("test")  # type: ignore[union-attr]

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
        result = ffai.rag.query("capital of France?")  # type: ignore[union-attr]
        assert result.answer == "Paris"
        assert result.sources == ["geo.txt"]
        mock_rag.query.assert_called_once()

    def test_passes_top_k(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.query.return_value = QueryResult(answer="a", hits=[], sources=[], prompt="p")
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        ffai.rag.query("q?", top_k=10)  # type: ignore[union-attr]
        call_kwargs = mock_rag.query.call_args
        assert call_kwargs[1]["top_k"] == 10

    def test_passes_custom_template(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.query.return_value = QueryResult(answer="a", hits=[], sources=[], prompt="p")
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        template = "Custom: {context}\n{question}"
        ffai.rag.query("q?", prompt_template=template)  # type: ignore[union-attr]
        call_kwargs = mock_rag.query.call_args
        assert call_kwargs[1]["prompt_template"] == template

    def test_client_adapter_wraps_client(self, concrete_client):
        concrete_client.generate_response = lambda prompt, **kw: f"reply to: {prompt[:10]}"
        mock_rag = MagicMock()
        mock_rag._generate_fn = None
        mock_rag.set_generate_fn.side_effect = lambda fn: setattr(mock_rag, "_generate_fn", fn)

        def capture_query(*args, **kwargs):
            generate_fn = mock_rag._generate_fn
            answer = generate_fn("test prompt")
            return QueryResult(answer=answer.text, hits=[], sources=[], prompt="test prompt")

        mock_rag.query.side_effect = capture_query
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        result = ffai.rag.query("q?")  # type: ignore[union-attr]
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
        ffai.rag.query("q?", source="doc1")  # type: ignore[union-attr]
        call_kwargs = mock_rag.query.call_args
        assert call_kwargs[1]["source"] == "doc1"

    def test_existing_tests_unaffected(self, concrete_client):
        ffai = FFAI(client=concrete_client)
        assert hasattr(ffai, "workflow")
        assert hasattr(ffai, "history")

    def test_query_returns_query_result_instance(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.query.return_value = QueryResult(answer="a", hits=[], sources=[], prompt="p")
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        result = ffai.rag.query("q?")  # type: ignore[union-attr]
        assert isinstance(result, QueryResult)
        assert result.answer == "a"

    def test_query_returns_what_rag_returns(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.query.return_value = "not a QueryResult"
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        result = ffai.rag.query("q?")  # type: ignore[union-attr]
        assert result == "not a QueryResult"  # type: ignore[union-attr]

    def test_sets_generate_fn_on_rag(self, concrete_client):
        mock_rag = MagicMock()
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        mock_rag.set_generate_fn.assert_called_once()
        adapter = mock_rag.set_generate_fn.call_args[0][0]
        assert callable(adapter)

    def test_set_client_rewires_rag_generate_fn(self, concrete_client):
        from ffai.rag import ClientAdapter

        mock_rag = MagicMock()
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        mock_rag.set_generate_fn.reset_mock()

        new_client = MagicMock()
        ffai.set_client(new_client)

        mock_rag.set_generate_fn.assert_called_once()
        adapter = mock_rag.set_generate_fn.call_args[0][0]
        assert isinstance(adapter, ClientAdapter)
        assert adapter._client is new_client

    def test_set_client_without_rag_does_not_error(self, concrete_client):
        ffai = FFAI(client=concrete_client)
        new_client = MagicMock()
        ffai.set_client(new_client)
        assert ffai.client is new_client

    def test_assigning_rag_auto_wires_generate_fn(self, concrete_client):
        from ffai.rag import ClientAdapter

        ffai = FFAI(client=concrete_client)
        mock_rag = MagicMock()
        ffai.rag = mock_rag
        mock_rag.set_generate_fn.assert_called_once()
        adapter = mock_rag.set_generate_fn.call_args[0][0]
        assert isinstance(adapter, ClientAdapter)
        assert adapter._client is concrete_client

    def test_assigning_rag_none_clears(self, concrete_client):
        mock_rag = MagicMock()
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        ffai.rag = None
        assert ffai.rag is None


class TestFFAIAQuery:
    def test_aquery_raises_without_rag(self, concrete_client):
        ffai = FFAI(client=concrete_client)
        with pytest.raises(AttributeError):
            asyncio.run(ffai.rag.aquery("test"))  # type: ignore[union-attr]

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
        result = asyncio.run(ffai.rag.aquery("capital of France?"))  # type: ignore[union-attr]
        assert result.answer == "Paris"
        assert result.sources == ["geo.txt"]
        mock_rag.aquery.assert_called_once()

    def test_aquery_passes_top_k(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.aquery = AsyncMock(return_value=QueryResult(answer="a", hits=[], sources=[], prompt="p"))
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        asyncio.run(ffai.rag.aquery("q?", top_k=10))  # type: ignore[union-attr]
        call_kwargs = mock_rag.aquery.call_args
        assert call_kwargs[1]["top_k"] == 10

    def test_aquery_client_adapter_wraps_client(self, concrete_client):
        concrete_client.generate_response = lambda prompt, **kw: f"reply to: {prompt[:10]}"
        mock_rag = MagicMock()
        mock_rag._generate_fn = None
        mock_rag.set_generate_fn.side_effect = lambda fn: setattr(mock_rag, "_generate_fn", fn)

        async def capture_aquery(*args, **kwargs):
            generate_fn = mock_rag._generate_fn
            answer = generate_fn("test prompt")
            return QueryResult(answer=answer.text, hits=[], sources=[], prompt="test prompt")

        mock_rag.aquery = AsyncMock(side_effect=capture_aquery)
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        result = asyncio.run(ffai.rag.aquery("q?"))  # type: ignore[union-attr]
        assert result.answer == "reply to: test promp"

    def test_aquery_passes_custom_template(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.aquery = AsyncMock(return_value=QueryResult(answer="a", hits=[], sources=[], prompt="p"))
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        template = "Custom: {context}\n{question}"
        asyncio.run(ffai.rag.aquery("q?", prompt_template=template))  # type: ignore[union-attr]
        call_kwargs = mock_rag.aquery.call_args
        assert call_kwargs[1]["prompt_template"] == template

    def test_aquery_returns_query_result_instance(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.aquery = AsyncMock(return_value=QueryResult(answer="a", hits=[], sources=[], prompt="p"))
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        result = asyncio.run(ffai.rag.aquery("q?"))  # type: ignore[union-attr]
        assert isinstance(result, QueryResult)
        assert result.answer == "a"


class TestFFAIFacade:
    def test_index_delegates_to_rag(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.index.return_value = 5
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        result = ffai.rag.index("some text", source="doc")  # type: ignore[union-attr]
        assert result == 5
        mock_rag.index.assert_called_once_with("some text", source="doc")

    def test_index_with_checksum(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.index.return_value = 3
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        result = ffai.rag.index("text", source="doc", checksum="abc123")  # type: ignore[union-attr]
        assert result == 3
        mock_rag.index.assert_called_once_with("text", source="doc", checksum="abc123")

    def test_search_delegates_to_rag(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.search.return_value = [SearchHit(content="hit", score=0.9, source="s1")]
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        hits = ffai.rag.search("query", top_k=3)  # type: ignore[union-attr]
        assert len(hits) == 1
        assert hits[0].content == "hit"
        mock_rag.search.assert_called_once_with("query", top_k=3)

    def test_delete_delegates_to_rag(self, concrete_client):
        mock_rag = MagicMock()
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        ffai.rag.delete("doc1")  # type: ignore[union-attr]
        mock_rag.delete.assert_called_once_with("doc1")

    def test_count_delegates_to_rag(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.count.return_value = 42
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        assert ffai.rag.count() == 42  # type: ignore[union-attr]

    def test_aindex_delegates_to_rag(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.aindex = AsyncMock(return_value=7)
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        result = asyncio.run(ffai.rag.aindex("text", source="doc"))  # type: ignore[union-attr]
        assert result == 7
        mock_rag.aindex.assert_called_once_with("text", source="doc")

    def test_asearch_delegates_to_rag(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.asearch = AsyncMock(return_value=[SearchHit(content="hit", score=0.9, source="s1")])
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        hits = asyncio.run(ffai.rag.asearch("query", top_k=2))  # type: ignore[union-attr]
        assert len(hits) == 1
        mock_rag.asearch.assert_called_once_with("query", top_k=2)

    def test_facade_raises_without_rag(self, concrete_client):
        ffai = FFAI(client=concrete_client)
        with pytest.raises(AttributeError):
            ffai.rag.index("text")  # type: ignore[union-attr]
        with pytest.raises(AttributeError):
            ffai.rag.search("q")  # type: ignore[union-attr]
        with pytest.raises(AttributeError):
            ffai.rag.delete("doc")  # type: ignore[union-attr]
        with pytest.raises(AttributeError):
            ffai.rag.count()  # type: ignore[union-attr]
        with pytest.raises(AttributeError):
            asyncio.run(ffai.rag.aindex("text"))  # type: ignore[union-attr]
        with pytest.raises(AttributeError):
            asyncio.run(ffai.rag.asearch("q"))  # type: ignore[union-attr]

    def test_query_passes_allow_llm_on_empty(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.query.return_value = QueryResult(answer="", hits=[], sources=[], prompt="")
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        ffai.rag.query("q?", allow_llm_on_empty=False)  # type: ignore[union-attr]
        call_kwargs = mock_rag.query.call_args
        assert call_kwargs[1]["allow_llm_on_empty"] is False

    def test_query_passes_generate_timeout(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.query.return_value = QueryResult(answer="a", hits=[], sources=[], prompt="p")
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        ffai.rag.query("q?", generate_timeout=30.0)  # type: ignore[union-attr]
        call_kwargs = mock_rag.query.call_args
        assert call_kwargs[1]["generate_timeout"] == 30.0

    def test_aquery_passes_generate_timeout(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.aquery = AsyncMock(return_value=QueryResult(answer="a", hits=[], sources=[], prompt="p"))
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        asyncio.run(ffai.rag.aquery("q?", generate_timeout=15.0))  # type: ignore[union-attr]
        call_kwargs = mock_rag.aquery.call_args
        assert call_kwargs[1]["generate_timeout"] == 15.0

    def test_aquery_passes_allow_llm_on_empty(self, concrete_client):
        mock_rag = MagicMock()
        mock_rag.aquery = AsyncMock(return_value=QueryResult(answer="", hits=[], sources=[], prompt=""))
        ffai = FFAI(client=concrete_client, rag=mock_rag)
        asyncio.run(ffai.rag.aquery("q?", allow_llm_on_empty=False))  # type: ignore[union-attr]
        call_kwargs = mock_rag.aquery.call_args
        assert call_kwargs[1]["allow_llm_on_empty"] is False
