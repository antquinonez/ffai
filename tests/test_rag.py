from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.rag import RAG
from src.rag.types import SearchHit


def _make_mock_embed():
    mock = MagicMock(spec=["aembed", "aembed_single", "model", "provider", "is_local", "cache_stats", "clear_cache"])
    mock.model = "mistral/mistral-embed"
    mock.aembed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock.aembed_single = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return mock


def _make_mock_store():
    mock = MagicMock()
    mock.count.return_value = 0
    mock.needs_reindex.return_value = True
    mock.asearch = AsyncMock(return_value=[])
    mock.aadd = AsyncMock(return_value=2)
    mock.get_all.return_value = []
    return mock


def _build_rag(**kwargs):
    embed = _make_mock_embed()
    store = _make_mock_store()
    mock_chunker = MagicMock()
    mock_chunker.chunk.return_value = [
        MagicMock(content="chunk 1", chunk_index=0, metadata={"source": "doc1"}),
        MagicMock(content="chunk 2", chunk_index=1, metadata={"source": "doc1"}),
    ]
    with patch("src.rag.rag.get_chunker", return_value=mock_chunker):
        rag = RAG(embed=embed, store=store, **kwargs)
    return rag, embed, store, mock_chunker


class TestRAGInit:
    def test_stores_components(self):
        rag, embed, store, _ = _build_rag()
        assert rag._embed is embed
        assert rag._store is store

    def test_string_embed_creates_instance(self):
        mock_chunker = MagicMock()
        with patch("src.rag.rag.get_chunker", return_value=mock_chunker):
            rag = RAG(embed="mistral/mistral-embed")
            assert rag._embed.model == "mistral/mistral-embed"

    def test_no_store_chunk_only_mode(self):
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = []
        with patch("src.rag.rag.get_chunker", return_value=mock_chunker):
            rag = RAG(embed="mistral/mistral-embed")
            assert rag._store is None
            assert rag.count() == 0


class TestRAGIndex:
    def test_returns_chunk_count(self):
        rag, _, store, _ = _build_rag()
        result = rag.index("some text", source="doc1")
        assert result == 2
        store.aadd.assert_called_once()

    def test_empty_text_returns_zero(self):
        rag, _, store, _ = _build_rag()
        assert rag.index("") == 0
        assert rag.index("   ") == 0
        store.aadd.assert_not_called()

    def test_passes_source_in_metadata(self):
        rag, _, _, mock_chunker = _build_rag()
        rag.index("text", source="my_doc")
        call_meta = mock_chunker.chunk.call_args[1]["metadata"]
        assert call_meta["source"] == "my_doc"


class TestRAGSearch:
    def test_returns_search_hits(self):
        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[
            SearchHit(content="result", score=0.9, source="doc1"),
        ])
        hits = rag.search("query")
        assert len(hits) == 1
        assert hits[0].content == "result"
        assert hits[0].score == 0.9

    def test_returns_empty_without_store_or_bm25(self):
        mock_chunker = MagicMock()
        with patch("src.rag.rag.get_chunker", return_value=mock_chunker):
            rag = RAG(embed=_make_mock_embed())
        assert rag.search("query") == []


class TestRAGDelete:
    def test_deletes_from_store(self):
        rag, _, store, _ = _build_rag()
        rag.delete("doc1")
        store.delete_by_source.assert_called_once_with("doc1")


class TestRAGCount:
    def test_delegates_to_store(self):
        rag, _, store, _ = _build_rag()
        store.count.return_value = 42
        assert rag.count() == 42


class TestRAGChunk:
    def test_returns_chunks_without_storing(self):
        rag, _, store, mock_chunker = _build_rag()
        mock_chunker.chunk.return_value = [MagicMock(content="c1")]
        chunks = rag.chunk("some text")
        assert len(chunks) == 1
        store.aadd.assert_not_called()


class TestRAGBM25:
    def test_bm25_syncs_on_index(self):
        rag, _, store, _ = _build_rag(bm25_alpha=0.6)
        rag.index("text to index", source="doc1")
        assert rag._bm25 is not None
        assert rag._bm25.count() == 2

    def test_bm25_deletes_by_source(self):
        rag, _, store, _ = _build_rag(bm25_alpha=0.6)
        rag.index("text", source="doc1")
        rag.delete("doc1")
        assert rag._bm25 is not None
        assert rag._bm25.count() == 0


class TestRAGReranker:
    def test_reranker_applied_on_search(self):
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {"id": "1", "content": "reranked", "score": 0.95, "metadata": {"source": "a"}},
        ]
        rag, _, store, _ = _build_rag(reranker="diversity")
        rag._reranker = mock_reranker
        store.asearch = AsyncMock(return_value=[
            SearchHit(content="result", score=0.7, source="a"),
        ])
        hits = rag.search("query")
        mock_reranker.rerank.assert_called_once()
        assert len(hits) == 1

    def test_reranker_receives_raw_dicts(self):
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {"id": "1", "content": "reranked", "score": 0.95, "metadata": {"source": "a"}},
        ]
        rag, _, store, _ = _build_rag(reranker="diversity")
        rag._reranker = mock_reranker
        store.asearch = AsyncMock(return_value=[
            SearchHit(id="1", content="result", score=0.7, source="a"),
        ])
        rag.search("query")
        call_args = mock_reranker.rerank.call_args[0][1]
        assert isinstance(call_args[0], dict)
        assert call_args[0]["id"] == "1"

    def test_reranker_preserves_rrf_score(self):
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {"id": "1", "content": "reranked", "score": 0.95, "rrf_score": 0.012, "metadata": {"source": "a"}},
        ]
        rag, _, store, _ = _build_rag(bm25_alpha=0.6, reranker="diversity")
        rag._reranker = mock_reranker
        store.asearch = AsyncMock(return_value=[
            SearchHit(id="v1", content="vector hit", score=0.8, source="a"),
        ])
        hits = rag.search("query")
        assert hits[0].score == pytest.approx(0.012)


class TestRAGBatchIndexing:
    def test_index_many_returns_total_chunks(self):
        rag, _, store, _ = _build_rag()
        docs = [
            {"text": "doc one", "source": "s1"},
            {"text": "doc two", "source": "s2"},
        ]
        total = rag.index_many(docs)
        assert total == 4
        assert store.aadd.call_count == 2

    def test_index_many_empty_list(self):
        rag, _, store, _ = _build_rag()
        assert rag.index_many([]) == 0
        store.aadd.assert_not_called()

    def test_index_many_skips_empty_text(self):
        rag, _, store, _ = _build_rag()
        docs = [
            {"text": "valid text", "source": "s1"},
            {"text": "", "source": "s2"},
        ]
        total = rag.index_many(docs)
        assert total == 2
        assert store.aadd.call_count == 1


class TestRAGDedup:
    def test_skips_index_when_checksum_matches(self):
        rag, _, store, _ = _build_rag()
        store.needs_reindex.return_value = False
        result = rag.index("text", source="doc1", checksum="abc")
        assert result == 0
        store.aadd.assert_not_called()

    def test_indexes_when_checksum_differs(self):
        rag, _, store, _ = _build_rag()
        store.needs_reindex.return_value = True
        result = rag.index("text", source="doc1", checksum="abc")
        assert result == 2
        store.aadd.assert_called_once()

    def test_indexes_when_no_checksum(self):
        rag, _, store, _ = _build_rag()
        result = rag.index("text", source="doc1")
        assert result == 2
        store.needs_reindex.assert_not_called()

    def test_indexes_when_no_source(self):
        rag, _, store, _ = _build_rag()
        result = rag.index("text", checksum="abc")
        assert result == 2
        store.needs_reindex.assert_not_called()

    def test_indexes_when_no_store(self):
        embed = _make_mock_embed()
        chunker = MagicMock()
        chunker.chunk.return_value = [MagicMock(content="c1", metadata={"source": "s"})]
        with patch("src.rag.rag.get_chunker", return_value=chunker):
            rag = RAG(embed=embed, bm25_alpha=0.6)
        result = rag.index("text", source="s1", checksum="abc")
        assert result == 1

    def test_stores_checksum_in_metadata(self):
        rag, _, store, _ = _build_rag()
        rag.index("text", source="doc1", checksum="abc")
        call_args = store.aadd.call_args[0]
        metas = call_args[3]
        assert metas[0]["document_checksum"] == "abc"
        assert metas[0]["chunking_strategy"] == "recursive"


class TestRAGQueryExpansion:
    def test_expansion_issues_multiple_searches(self):
        call_count = 0

        def expanding_fn(q):
            return [q, q + " explained", q + " for beginners"]

        rag, _, store, _ = _build_rag(query_expander=expanding_fn)
        original_asearch = store.asearch
        store.asearch = AsyncMock(return_value=[
            SearchHit(id="1", content="result", score=0.9, source="a"),
        ])
        hits = rag.search("query")
        assert store.asearch.call_count == 3

    def test_expansion_disabled_by_default(self):
        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[])
        rag.search("query")
        assert store.asearch.call_count == 1

    def test_expansion_fallback_on_exception(self):
        def failing_fn(q):
            raise RuntimeError("LLM unavailable")

        rag, _, store, _ = _build_rag(query_expander=failing_fn)
        store.asearch = AsyncMock(return_value=[
            SearchHit(id="1", content="result", score=0.9, source="a"),
        ])
        hits = rag.search("query")
        assert len(hits) == 1
        assert store.asearch.call_count == 1

    def test_expansion_deduplicates_results(self):
        def expanding_fn(q):
            return [q, q + " variant"]

        rag, _, store, _ = _build_rag(query_expander=expanding_fn)
        hit = SearchHit(id="1", content="shared", score=0.9, source="a")
        store.asearch = AsyncMock(return_value=[hit])
        hits = rag.search("query")
        ids = [h.id for h in hits]
        assert len(ids) == len(set(ids))


class TestRAGQuery:
    def test_query_returns_query_result(self):
        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[
            SearchHit(content="Paris is the capital of France.", score=0.9, source="geo.txt"),
        ])
        result = rag.query("What is the capital of France?", generate_fn=lambda p: "Paris")
        assert result.answer == "Paris"
        assert len(result.hits) == 1
        assert result.hits[0].content == "Paris is the capital of France."
        assert result.sources == ["geo.txt"]
        assert "Paris is the capital of France." in result.prompt
        assert "What is the capital of France?" in result.prompt

    def test_query_custom_template(self):
        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[
            SearchHit(content="ctx text", score=0.8, source="s1", metadata={"source": "s1"}),
        ])
        template = "Context: {context}\nAsk: {question}\nReply:"
        result = rag.query("q?", generate_fn=lambda p: "a", prompt_template=template)
        assert "ctx text" in result.prompt
        assert "Ask: q?" in result.prompt
        assert "Reply:" in result.prompt

    def test_query_max_context_chars_truncates(self):
        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[
            SearchHit(content="x" * 500, score=0.9, source="s1", metadata={"source": "s1"}),
            SearchHit(content="y" * 500, score=0.8, source="s1", metadata={"source": "s1"}),
        ])
        result_full = rag.query("q?", generate_fn=lambda p: "a")
        result_trunc = rag.query("q?", generate_fn=lambda p: "a", max_context_chars=100)
        assert len(result_trunc.prompt) < len(result_full.prompt)

    def test_query_max_context_chars_zero_excludes_all(self):
        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[
            SearchHit(content="content here", score=0.9, source="s1", metadata={"source": "s1"}),
        ])
        result = rag.query("q?", generate_fn=lambda p: "a", max_context_chars=1)
        assert "content here" not in result.prompt

    def test_query_empty_results(self):
        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[])
        result = rag.query("q?", generate_fn=lambda p: "no info")
        assert result.answer == "no info"
        assert result.hits == []
        assert result.sources == []

    def test_query_deduplicates_sources(self):
        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[
            SearchHit(content="a", score=0.9, source="s1", metadata={"source": "s1"}),
            SearchHit(content="b", score=0.8, source="s1", metadata={"source": "s1"}),
            SearchHit(content="c", score=0.7, source="s2", metadata={"source": "s2"}),
        ])
        result = rag.query("q?", generate_fn=lambda p: "a")
        assert result.sources == ["s1", "s2"]

    def test_query_passes_top_k(self):
        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[])
        rag.query("q?", generate_fn=lambda p: "a", top_k=3)
        store.asearch.assert_called_once()

    def test_generate_fn_receives_formatted_prompt(self):
        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[
            SearchHit(content="the sky is blue", score=0.9, source="s1", metadata={"source": "s1"}),
        ])
        captured_prompt = []
        rag.query("why sky?", generate_fn=lambda p: (captured_prompt.append(p), "blue")[1])
        assert "the sky is blue" in captured_prompt[0]
        assert "why sky?" in captured_prompt[0]

    def test_query_template_with_unknown_placeholders(self):
        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[
            SearchHit(content="data", score=0.9, source="s1", metadata={"source": "s1"}),
        ])
        template = "Context: {context}\nQuestion: {question}\nRole: {role}\nAnswer:"
        result = rag.query("q?", generate_fn=lambda p: "a", prompt_template=template)
        assert "data" in result.prompt
        assert "Question: q?" in result.prompt
        assert "Role: " in result.prompt
        assert "Answer:" in result.prompt

    def test_allow_llm_on_empty_false_skips_generate(self):
        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[])
        call_count = 0

        def counting_fn(p: str) -> str:
            nonlocal call_count
            call_count += 1
            return "should not be called"

        result = rag.query("q?", generate_fn=counting_fn, allow_llm_on_empty=False)
        assert result.answer == ""
        assert result.hits == []
        assert result.sources == []
        assert result.prompt == ""
        assert call_count == 0

    def test_allow_llm_on_empty_true_calls_generate(self):
        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[])
        result = rag.query("q?", generate_fn=lambda p: "no info", allow_llm_on_empty=True)
        assert result.answer == "no info"

    def test_allow_llm_on_empty_default_is_true(self):
        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[])
        result = rag.query("q?", generate_fn=lambda p: "no info")
        assert result.answer == "no info"

    def test_generate_timeout_raises_on_slow_fn(self):
        import time

        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[
            SearchHit(content="ctx", score=0.9, source="s1", metadata={"source": "s1"}),
        ])

        def slow_fn(p: str) -> str:
            time.sleep(1.0)
            return "too late"

        with pytest.raises(TimeoutError):
            rag.query("q?", generate_fn=slow_fn, generate_timeout=0.01)

    def test_generate_timeout_allows_fast_fn(self):
        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[
            SearchHit(content="ctx", score=0.9, source="s1", metadata={"source": "s1"}),
        ])
        result = rag.query("q?", generate_fn=lambda p: "fast", generate_timeout=30.0)
        assert result.answer == "fast"

    def test_generate_timeout_default_is_none(self):
        import time

        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[
            SearchHit(content="ctx", score=0.9, source="s1", metadata={"source": "s1"}),
        ])

        def slow_fn(p: str) -> str:
            time.sleep(0.1)
            return "done"

        result = rag.query("q?", generate_fn=slow_fn)
        assert result.answer == "done"

    def test_allow_llm_on_empty_false_with_nonempty_hits(self):
        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[
            SearchHit(content="ctx text", score=0.9, source="s1", metadata={"source": "s1"}),
        ])
        result = rag.query("q?", generate_fn=lambda p: "answer", allow_llm_on_empty=False)
        assert result.answer == "answer"
        assert len(result.hits) == 1

    def test_generate_timeout_aquery_async_path(self):
        import asyncio
        import time

        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[
            SearchHit(content="ctx", score=0.9, source="s1", metadata={"source": "s1"}),
        ])

        def slow_fn(p: str) -> str:
            time.sleep(1.0)
            return "too late"

        with pytest.raises(TimeoutError):
            asyncio.run(rag.aquery("q?", generate_fn=slow_fn, generate_timeout=0.01))


class TestRAGBM25OnlySearch:
    def test_bm25_only_search_returns_hits(self):
        embed = _make_mock_embed()
        chunker = MagicMock()
        chunker.chunk.return_value = [
            MagicMock(content="async programming in python", chunk_index=0, metadata={"source": "tutorial"}),
            MagicMock(content="synchronous code blocks", chunk_index=1, metadata={"source": "tutorial"}),
        ]
        with patch("src.rag.rag.get_chunker", return_value=chunker):
            rag = RAG(embed=embed, bm25_alpha=0.6)
        rag.index("some text", source="tutorial")
        hits = rag.search("async programming")
        assert len(hits) >= 1
        assert hits[0].source == "tutorial"

    def test_bm25_only_count_returns_bm25_count(self):
        embed = _make_mock_embed()
        chunker = MagicMock()
        chunker.chunk.return_value = [
            MagicMock(content="chunk text", chunk_index=0, metadata={"source": "doc"}),
        ]
        with patch("src.rag.rag.get_chunker", return_value=chunker):
            rag = RAG(embed=embed, bm25_alpha=0.6)
        rag.index("text", source="doc")
        assert rag.count() == 1

    def test_bm25_only_query_returns_answer(self):
        embed = _make_mock_embed()
        chunker = MagicMock()
        chunker.chunk.return_value = [
            MagicMock(content="Python uses async await for coroutines.", chunk_index=0, metadata={"source": "tutorial"}),
        ]
        with patch("src.rag.rag.get_chunker", return_value=chunker):
            rag = RAG(embed=embed, bm25_alpha=0.6)
        rag.index("text", source="tutorial")
        result = rag.query("What are coroutines?", generate_fn=lambda p: "async functions")
        assert result.answer == "async functions"
        assert len(result.hits) >= 1
        assert result.sources == ["tutorial"]

    def test_bm25_only_delete_clears_index(self):
        embed = _make_mock_embed()
        chunker = MagicMock()
        chunker.chunk.return_value = [
            MagicMock(content="chunk text", chunk_index=0, metadata={"source": "doc1"}),
        ]
        with patch("src.rag.rag.get_chunker", return_value=chunker):
            rag = RAG(embed=embed, bm25_alpha=0.6)
        rag.index("text", source="doc1")
        assert rag.count() == 1
        rag.delete("doc1")
        assert rag.count() == 0

    def test_bm25_only_search_filters_by_metadata(self):
        embed = _make_mock_embed()
        chunker = MagicMock()
        chunker.chunk.side_effect = [
            [MagicMock(content="python async programming", chunk_index=0, metadata={"source": "tutorial"})],
            [MagicMock(content="python async programming", chunk_index=0, metadata={"source": "api_docs"})],
        ]
        with patch("src.rag.rag.get_chunker", return_value=chunker):
            rag = RAG(embed=embed, bm25_alpha=0.6)
        rag.index("text", source="tutorial")
        rag.index("text", source="api_docs")
        hits = rag.search("python async", source="tutorial")
        assert all(h.source == "tutorial" for h in hits)


class TestRAGFromConfig:
    def test_from_config_creates_bm25_only_by_default(self):
        with patch("src.rag.rag.CHROMADB_AVAILABLE", False):
            rag = RAG.from_config()
        assert rag._store is None
        assert rag._bm25 is not None

    def test_from_config_uses_config_values(self):
        with patch("src.rag.rag.CHROMADB_AVAILABLE", False):
            rag = RAG.from_config(bm25_alpha=0.8)
        assert rag._bm25_alpha == 0.8

    def test_from_config_bm25_only_skips_store(self):
        with patch("src.rag.rag.CHROMADB_AVAILABLE", True):
            rag = RAG.from_config(bm25_only=True)
        assert rag._store is None
        assert rag._bm25 is not None

    def test_from_config_overrides_take_precedence(self):
        with patch("src.rag.rag.CHROMADB_AVAILABLE", False):
            rag = RAG.from_config(chunk_size=200, chunk_overlap=20)
        assert rag._chunker_name == "recursive"
        sample = "word " * 300
        chunks = rag._chunker.chunk(sample, metadata={})
        assert len(chunks) > 0
        assert max(len(c.content) for c in chunks) <= 200
