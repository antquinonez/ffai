from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.rag.pipeline import (
    RAGPipeline,
    _PipelineConfig,
    format_results_for_prompt,
    normalize_scores,
)
from src.rag.splitters.base import TextChunk


def _make_mock_embeddings():
    emb = MagicMock()
    emb.model = "mistral/mistral-embed"
    emb.embed.return_value = [[0.1, 0.2, 0.3]]
    emb.embed_single.return_value = [0.1, 0.2, 0.3]
    return emb


def _make_mock_vector_store():
    vs = MagicMock()
    vs.count.return_value = 0
    vs.get_stats.return_value = {"collection_name": "test", "count": 0}
    vs.needs_reindex.return_value = True
    vs.search.return_value = [
        {"id": "id1", "content": "text", "metadata": {"reference_name": "doc1"}, "distance": 0.3},
    ]
    vs.get_all_documents.return_value = []
    vs.add_chunks.return_value = 2
    vs.list_documents.return_value = ["doc1"]
    return vs


def _make_mock_chunker():
    chunker = MagicMock()
    chunker.chunk.return_value = [
        TextChunk(content="chunk 1", chunk_index=0, start_char=0, end_char=7, metadata={}),
        TextChunk(content="chunk 2", chunk_index=1, start_char=7, end_char=14, metadata={}),
    ]
    return chunker


def _build_pipeline(**builder_overrides):
    emb = _make_mock_embeddings()
    mock_vs = _make_mock_vector_store()
    mock_chunker = _make_mock_chunker()

    with (
        patch("src.rag.pipeline.get_chunker", return_value=mock_chunker),
        patch("src.rag.vector_store.FFVectorStore", return_value=mock_vs),
    ):
        builder = RAGPipeline(emb).chunk("recursive").with_vector_store()
        for key, value in builder_overrides.items():
            method = getattr(builder, key)
            if isinstance(value, dict):
                method(**value)
            else:
                method(value)
        pipeline = builder.build()

    return pipeline, mock_vs, emb, mock_chunker


class TestNormalizeScores:
    def test_converts_distance_to_score(self):
        results = [{"distance": 0.3}]
        out = normalize_scores(results)
        assert out[0]["score"] == pytest.approx(0.7)

    def test_uses_rrf_score_when_no_distance(self):
        results = [{"rrf_score": 0.015}]
        out = normalize_scores(results)
        assert out[0]["score"] == pytest.approx(0.015)

    def test_defaults_to_zero(self):
        results = [{}]
        out = normalize_scores(results)
        assert out[0]["score"] == 0.0

    def test_preserves_existing_score(self):
        results = [{"score": 0.9}]
        out = normalize_scores(results)
        assert out[0]["score"] == pytest.approx(0.9)

    def test_distance_takes_precedence_over_score(self):
        results = [{"distance": 0.5, "score": 0.99}]
        out = normalize_scores(results)
        assert out[0]["score"] == pytest.approx(0.5)


class TestFormatResultsForPrompt:
    def test_numbered_sourced_output(self):
        results = [
            {"content": "alpha", "metadata": {"reference_name": "a"}, "score": 0.9},
            {"content": "beta", "metadata": {"reference_name": "b"}, "score": 0.8},
        ]
        out = format_results_for_prompt(results)
        assert "[1]" in out
        assert "[2]" in out
        assert "source: a" in out
        assert "source: b" in out
        assert "alpha" in out
        assert "beta" in out

    def test_max_chars_truncates(self):
        results = [
            {"content": "short", "metadata": {"reference_name": "a"}, "score": 0.9},
            {"content": "x" * 500, "metadata": {"reference_name": "b"}, "score": 0.8},
        ]
        first = "[1] (source: a, relevance: 0.90)\nshort\n"
        out = format_results_for_prompt(results, max_chars=len(first))
        assert "short" in out
        assert "xxxx" not in out

    def test_empty_results_returns_empty_string(self):
        assert format_results_for_prompt([]) == ""

    def test_parent_context_included(self):
        results = [
            {
                "content": "child",
                "metadata": {"reference_name": "a"},
                "score": 0.9,
                "parent_content": "parent text here",
            },
        ]
        out = format_results_for_prompt(results, include_parent_context=True)
        assert "Parent context: parent text here" in out

    def test_parent_context_excluded(self):
        results = [
            {
                "content": "child",
                "metadata": {"reference_name": "a"},
                "score": 0.9,
                "parent_content": "parent text",
            },
        ]
        out = format_results_for_prompt(results, include_parent_context=False)
        assert "Parent context" not in out

    def test_unknown_source_when_no_reference_name(self):
        results = [{"content": "text", "metadata": {}, "score": 0.5}]
        out = format_results_for_prompt(results)
        assert "source: unknown" in out


class TestPipelineConfig:
    def test_defaults(self):
        cfg = _PipelineConfig()
        assert cfg.chunk_strategy == "recursive"
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 200
        assert cfg.bm25_enabled is False
        assert cfg.contextual_headers is True

    def test_independent_instances(self):
        a = _PipelineConfig()
        b = _PipelineConfig()
        a.chunk_size = 999
        assert b.chunk_size == 1000


class TestRAGPipelineBuilder:
    def test_builder_returns_self(self):
        emb = _make_mock_embeddings()
        p = RAGPipeline(emb)
        assert p.chunk("recursive") is p
        assert p.with_vector_store() is p
        assert p.with_bm25() is p
        assert p.with_hierarchical() is p
        assert p.with_reranker() is p
        assert p.with_query_expansion() is p
        assert p.with_summaries() is p
        assert p.with_contextual_headers() is p

    def test_build_idempotent(self):
        emb = _make_mock_embeddings()
        mock_chunker = _make_mock_chunker()
        with patch("src.rag.pipeline.get_chunker", return_value=mock_chunker):
            p = RAGPipeline(emb).chunk("recursive").build()
            assert p.build() is p

    def test_config_propagated(self):
        emb = _make_mock_embeddings()
        p = RAGPipeline(emb)
        p.chunk("markdown", chunk_size=500, chunk_overlap=50)
        p.with_bm25(k1=1.2, b=0.8, alpha=0.7)
        assert p._config.chunk_strategy == "markdown"
        assert p._config.chunk_size == 500
        assert p._config.bm25_k1 == 1.2
        assert p._config.hybrid_alpha == 0.7


class TestRAGPipelineIndex:
    def test_index_returns_chunk_count(self):
        pipeline, mock_vs, _, _ = _build_pipeline()
        result = pipeline.index("some text content", reference_name="doc1")
        assert result == 2
        mock_vs.add_chunks.assert_called_once()

    def test_index_returns_zero_for_empty(self):
        pipeline, _, _, _ = _build_pipeline()
        assert pipeline.index("") == 0
        assert pipeline.index("   ") == 0

    def test_index_without_vector_store_returns_chunk_count(self):
        emb = _make_mock_embeddings()
        mock_chunker = _make_mock_chunker()
        with patch("src.rag.pipeline.get_chunker", return_value=mock_chunker):
            pipeline = RAGPipeline(emb).chunk("recursive").build()
        result = pipeline.index("some text", reference_name="doc1")
        assert result == 2


class TestRAGPipelineSearch:
    def test_search_returns_normalized_results(self):
        pipeline, mock_vs, _, _ = _build_pipeline()
        mock_vs.search.return_value = [
            {"id": "id1", "content": "text", "metadata": {"reference_name": "doc1"}, "distance": 0.3},
        ]
        results = pipeline.search("query")
        assert results[0]["score"] == pytest.approx(0.7)

    def test_search_returns_empty_without_vector_store(self):
        emb = _make_mock_embeddings()
        mock_chunker = _make_mock_chunker()
        with patch("src.rag.pipeline.get_chunker", return_value=mock_chunker):
            pipeline = RAGPipeline(emb).chunk("recursive").build()
        assert pipeline.search("query") == []

    def test_search_with_reranker(self):
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {"id": "id1", "content": "reranked", "metadata": {}, "score": 0.95},
        ]
        pipeline, mock_vs, _, _ = _build_pipeline()
        pipeline._reranker = mock_reranker
        mock_vs.search.return_value = [
            {"id": "id1", "content": "text", "metadata": {}, "distance": 0.3},
        ]
        results = pipeline.search("query", rerank=True)
        mock_reranker.rerank.assert_called_once()

    def test_search_with_query_expansion(self):
        mock_expander = MagicMock()
        mock_expander.expand.return_value = ["query", "variation 1"]
        mock_expander.llm_generate_fn = lambda x: x
        pipeline, mock_vs, _, _ = _build_pipeline(
            with_query_expansion={"n_variations": 2},
        )
        pipeline._query_expander = mock_expander
        mock_vs.search.return_value = [
            {"id": "id1", "content": "text", "metadata": {}, "distance": 0.3},
        ]
        results = pipeline.search("query", query_expansion=True)
        assert mock_vs.search.call_count == 2


class TestRAGPipelineFormat:
    def test_format_delegates_to_function(self):
        pipeline, _, _, _ = _build_pipeline()
        results = [{"content": "text", "metadata": {"reference_name": "a"}, "score": 0.9}]
        out = pipeline.format(results)
        assert "[1]" in out
        assert "source: a" in out


class TestRAGPipelineLifecycle:
    def test_delete_delegates(self):
        pipeline, mock_vs, _, _ = _build_pipeline()
        pipeline.delete("doc1")
        mock_vs.delete_by_reference.assert_called_once_with("doc1")

    def test_count_delegates(self):
        pipeline, mock_vs, _, _ = _build_pipeline()
        mock_vs.count.return_value = 42
        assert pipeline.count() == 42

    def test_clear_delegates(self):
        pipeline, mock_vs, _, _ = _build_pipeline()
        pipeline.clear()
        mock_vs.clear.assert_called_once()

    def test_list_documents_delegates(self):
        pipeline, mock_vs, _, _ = _build_pipeline()
        mock_vs.list_documents.return_value = ["a", "b"]
        assert pipeline.list_documents() == ["a", "b"]

    def test_get_stats_includes_vector_store(self):
        pipeline, mock_vs, _, _ = _build_pipeline()
        mock_vs.get_stats.return_value = {"count": 10}
        stats = pipeline.get_stats()
        assert stats["count"] == 10

    def test_count_returns_zero_without_vector_store(self):
        emb = _make_mock_embeddings()
        mock_chunker = _make_mock_chunker()
        with patch("src.rag.pipeline.get_chunker", return_value=mock_chunker):
            pipeline = RAGPipeline(emb).chunk("recursive").build()
        assert pipeline.count() == 0


class TestRAGPipelineSetLlmFn:
    def test_propagates_to_query_expander(self):
        mock_expander = MagicMock()
        pipeline, _, _, _ = _build_pipeline(
            with_query_expansion={"n_variations": 3},
        )
        pipeline._query_expander = mock_expander
        def fn(p):
            return "ok"
        pipeline.set_llm_fn(fn)
        mock_expander.set_llm_function.assert_called_once_with(fn)


class TestRAGPipelineProperties:
    def test_chunker_returns_built_chunker(self):
        pipeline, _, _, mock_chunker = _build_pipeline()
        assert pipeline.chunker is mock_chunker

    def test_embeddings_returns_init_embeddings(self):
        _, _, mock_emb, _ = _build_pipeline()
        assert _build_pipeline()[2] is not None

    def test_vector_store_returns_built_store(self):
        pipeline, mock_vs, _, _ = _build_pipeline()
        assert pipeline.vector_store is mock_vs

    def test_vector_store_none_without_build(self):
        emb = _make_mock_embeddings()
        p = RAGPipeline(emb)
        assert p.vector_store is None


class TestRAGPipelineSearchAndFormat:
    def test_combines_search_and_format(self):
        pipeline, mock_vs, _, _ = _build_pipeline()
        mock_vs.search.return_value = [
            {"id": "id1", "content": "found text", "metadata": {"reference_name": "doc1"}, "distance": 0.1},
        ]
        out = pipeline.search_and_format("query")
        assert "found text" in out
        assert "source: doc1" in out
