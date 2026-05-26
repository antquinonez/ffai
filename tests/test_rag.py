from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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


class TestRAGNoGetConfigImport:
    def test_rag_module_has_no_get_config_import(self):
        import inspect

        import src.rag.rag as rag_module
        source = inspect.getsource(rag_module)
        assert "get_config" not in source
