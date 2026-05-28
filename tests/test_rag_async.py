from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ffai.rag.rag import RAG
from ffai.rag.types import SearchHit


def _make_mock_embed():
    mock = MagicMock(spec=["aembed", "aembed_single", "model", "provider", "is_local", "cache_stats", "clear_cache"])
    mock.model = "mistral/mistral-embed"
    mock.aembed = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    mock.aembed_single = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return mock


def _make_mock_store():
    mock = MagicMock()
    mock.count.return_value = 0
    mock.asearch = AsyncMock(return_value=[])
    mock.aadd = AsyncMock(return_value=2)
    return mock


def _make_chunker(n_chunks=2):
    mock = MagicMock()
    mock.chunk.return_value = [
        MagicMock(content=f"chunk {i}", chunk_index=i, metadata={"source": "doc1"})
        for i in range(n_chunks)
    ]
    return mock


def _build_rag(**kwargs):
    embed = _make_mock_embed()
    store = _make_mock_store()
    chunker = _make_chunker()
    with patch("ffai.rag.rag.get_chunker", return_value=chunker):
        rag = RAG(embed=embed, store=store, **kwargs)
    return rag, embed, store, chunker


class TestRAGAsyncIndex:
    def test_aindex_returns_chunk_count(self):
        rag, _, store, _ = _build_rag()
        result = asyncio.run(rag.aindex("some text", source="doc1"))
        assert result == 2
        store.aadd.assert_called_once()

    def test_aindex_empty_text(self):
        rag, _, store, _ = _build_rag()
        assert asyncio.run(rag.aindex("")) == 0
        store.aadd.assert_not_called()

    def test_aindex_whitespace_only(self):
        rag, _, store, _ = _build_rag()
        assert asyncio.run(rag.aindex("   \n\t")) == 0

    def test_aindex_no_embeddings_when_no_store(self):
        embed = _make_mock_embed()
        chunker = _make_chunker()
        with patch("ffai.rag.rag.get_chunker", return_value=chunker):
            rag = RAG(embed=embed)
        result = asyncio.run(rag.aindex("text", source="doc1"))
        assert result == 2
        embed.aembed.assert_not_called()

    def test_aindex_with_bm25_only(self):
        embed = _make_mock_embed()
        chunker = _make_chunker()
        with patch("ffai.rag.rag.get_chunker", return_value=chunker):
            rag = RAG(embed=embed, bm25_alpha=0.6)
        result = asyncio.run(rag.aindex("text to index", source="doc1"))
        assert result == 2
        embed.aembed.assert_not_called()
        assert rag._bm25 is not None
        assert rag._bm25.count() == 2

    def test_aindex_passes_metadata(self):
        rag, _, _, chunker = _build_rag()
        asyncio.run(rag.aindex("text", source="doc1", category="test"))
        call_meta = chunker.chunk.call_args[1]["metadata"]
        assert call_meta["source"] == "doc1"
        assert call_meta["category"] == "test"


class TestRAGAsyncSearch:
    def test_asearch_returns_hits(self):
        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[
            SearchHit(id="1", content="result", score=0.9, source="doc1"),
        ])
        hits = asyncio.run(rag.asearch("query"))
        assert len(hits) == 1
        assert hits[0].content == "result"
        assert hits[0].id == "1"

    def test_asearch_no_store_no_bm25_returns_empty(self):
        embed = _make_mock_embed()
        with patch("ffai.rag.rag.get_chunker", return_value=_make_chunker()):
            rag = RAG(embed=embed)
        assert asyncio.run(rag.asearch("query")) == []

    def test_asearch_bm25_only_returns_hits(self):
        embed = _make_mock_embed()
        chunker = MagicMock()
        chunker.chunk.return_value = [
            MagicMock(content="async programming in python", chunk_index=0, metadata={"source": "tutorial"}),
        ]
        with patch("ffai.rag.rag.get_chunker", return_value=chunker):
            rag = RAG(embed=embed, bm25_alpha=0.6)
        asyncio.run(rag.aindex("text", source="tutorial"))
        hits = asyncio.run(rag.asearch("async programming"))
        assert len(hits) >= 1

    def test_asearch_respects_top_k(self):
        rag, _, store, _ = _build_rag()
        all_hits = [SearchHit(id=str(i), content=f"hit {i}", score=0.9 - i * 0.1) for i in range(10)]
        store.asearch = AsyncMock(return_value=all_hits)
        hits = asyncio.run(rag.asearch("query", top_k=3))
        assert len(hits) == 3

    def test_asearch_with_filters(self):
        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[])
        asyncio.run(rag.asearch("query", source="doc1"))
        store.asearch.assert_called_once()
        call_kwargs = store.asearch.call_args
        assert call_kwargs[1]["where"] == {"source": "doc1"}


class TestRAGAsyncHybridSearch:
    def test_ahybrid_search_fuses_results(self):
        rag, embed, store, _ = _build_rag(bm25_alpha=0.6)

        def mock_vector_results():
            return [
                SearchHit(id="v1", content="vector result", score=0.9, source="s1"),
            ]

        store.asearch = AsyncMock(return_value=mock_vector_results())
        asyncio.run(rag.aindex("test document about python", source="s1"))

        raw = asyncio.run(rag._ahybrid_search("python", 5))
        assert len(raw) >= 1
        assert all("rrf_score" in r for r in raw)

    def test_ahybrid_search_empty_bm25(self):
        rag, _, store, _ = _build_rag(bm25_alpha=0.6)
        store.asearch = AsyncMock(return_value=[])
        raw = asyncio.run(rag._ahybrid_search("query", 5))
        assert raw == []

    def test_asearch_hybrid_path(self):
        rag, embed, store, chunker = _build_rag(bm25_alpha=0.6)
        new_chunker = MagicMock()
        new_chunker.chunk.return_value = [
            MagicMock(content="python programming language", chunk_index=0, metadata={"source": "wiki"}),
        ]
        with patch("ffai.rag.rag.get_chunker", return_value=new_chunker):
            rag2, embed2, store2, _ = _build_rag(bm25_alpha=0.6)

        store2.asearch = AsyncMock(return_value=[
            SearchHit(id="wiki_0", content="python programming language", score=0.8, source="wiki"),
        ])
        asyncio.run(rag2.aindex("python programming language guide", source="wiki"))
        hits = asyncio.run(rag2.asearch("python"))
        assert len(hits) >= 1


class TestRAGRawToHits:
    def test_converts_distance_to_score(self):
        rag, _, _, _ = _build_rag()
        raw = [{"id": "1", "content": "test", "distance": 0.3, "metadata": {"source": "s"}}]
        hits = rag._raw_to_hits(raw)
        assert hits[0].score == pytest.approx(0.7)

    def test_converts_rrf_score(self):
        rag, _, _, _ = _build_rag()
        raw = [{"id": "1", "content": "test", "rrf_score": 0.0123, "metadata": {}}]
        hits = rag._raw_to_hits(raw)
        assert hits[0].score == pytest.approx(0.0123)

    def test_falls_back_to_score(self):
        rag, _, _, _ = _build_rag()
        raw = [{"id": "1", "content": "test", "score": 0.5, "metadata": {"source": "s"}}]
        hits = rag._raw_to_hits(raw)
        assert hits[0].score == pytest.approx(0.5)

    def test_handles_missing_metadata(self):
        rag, _, _, _ = _build_rag()
        raw = [{"id": "1", "content": "test", "score": 0.5}]
        hits = rag._raw_to_hits(raw)
        assert hits[0].metadata == {}
        assert hits[0].source == ""

    def test_handles_parent_content(self):
        rag, _, _, _ = _build_rag()
        raw = [{"id": "1", "content": "chunk", "score": 0.5, "parent_content": "full doc"}]
        hits = rag._raw_to_hits(raw)
        assert hits[0].parent_content == "full doc"

    def test_source_from_top_level_when_not_in_metadata(self):
        rag, _, _, _ = _build_rag()
        raw = [{"id": "1", "content": "chunk", "score": 0.5, "source": "my_doc.txt", "metadata": {}}]
        hits = rag._raw_to_hits(raw)
        assert hits[0].source == "my_doc.txt"

    def test_metadata_source_takes_precedence_over_top_level(self):
        rag, _, _, _ = _build_rag()
        raw = [{"id": "1", "content": "chunk", "score": 0.5, "source": "fallback.txt", "metadata": {"source": "primary.txt"}}]
        hits = rag._raw_to_hits(raw)
        assert hits[0].source == "primary.txt"

    def test_source_empty_when_neither_present(self):
        rag, _, _, _ = _build_rag()
        raw = [{"id": "1", "content": "chunk", "score": 0.5, "metadata": {}}]
        hits = rag._raw_to_hits(raw)
        assert hits[0].source == ""


class TestRAGEmbedSync:
    def test_embed_via_embed_class(self):
        embed = _make_mock_embed()
        embed.embed = MagicMock(return_value=[[0.1, 0.2, 0.3]])
        chunker = _make_chunker()
        with patch("ffai.rag.rag.get_chunker", return_value=chunker):
            rag = RAG(embed=embed, store=_make_mock_store())
        result = rag._embed.embed(["hello"])
        embed.embed.assert_called_once_with(["hello"])
        assert result == [[0.1, 0.2, 0.3]]


class TestRAGAQuery:
    def test_aquery_returns_query_result(self):
        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[
            SearchHit(content="Paris is capital.", score=0.9, source="geo.txt", metadata={"source": "geo.txt"}),
        ])
        result = asyncio.run(rag.aquery("capital?", generate_fn=lambda p: "Paris"))
        assert result.answer == "Paris"
        assert len(result.hits) == 1
        assert result.sources == ["geo.txt"]
        assert "Paris is capital." in result.prompt
        assert "capital?" in result.prompt

    def test_aquery_custom_template(self):
        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[
            SearchHit(content="ctx", score=0.8, source="s1", metadata={"source": "s1"}),
        ])
        template = "Info: {context}\nQ: {question}\nA:"
        result = asyncio.run(rag.aquery("q?", generate_fn=lambda p: "a", prompt_template=template))
        assert "ctx" in result.prompt
        assert "Q: q?" in result.prompt
        assert "A:" in result.prompt

    def test_aquery_empty_results(self):
        rag, _, store, _ = _build_rag()
        store.asearch = AsyncMock(return_value=[])
        result = asyncio.run(rag.aquery("q?", generate_fn=lambda p: "no info"))
        assert result.answer == "no info"
        assert result.hits == []
        assert result.sources == []
