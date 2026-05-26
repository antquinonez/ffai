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

    def test_returns_empty_without_store(self):
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


class TestRAGNoGetConfigImport:
    def test_rag_module_has_no_get_config_import(self):
        import inspect

        import src.rag.rag as rag_module
        source = inspect.getsource(rag_module)
        assert "get_config" not in source


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
