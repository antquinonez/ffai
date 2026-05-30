from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from ffai.rag.types import SearchHit


def _make_mock_chromadb():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_settings = MagicMock()
    mock_chromadb = MagicMock()
    mock_chromadb.PersistentClient.return_value = mock_client
    mock_chromadb.config.Settings.return_value = mock_settings
    return mock_chromadb, mock_client, mock_collection


@contextmanager
def _patch_store(mock_chromadb):
    with (
        patch("ffai.rag.stores.chroma.CHROMADB_AVAILABLE", True),
        patch("ffai.rag.stores.chroma.chromadb", mock_chromadb),
        patch("ffai.rag.stores.chroma.Settings", mock_chromadb.config.Settings),
    ):
        yield


class TestVectorStoreInit:
    def test_creates_collection(self):
        from ffai.rag.store import VectorStore
        mock_chromadb, mock_client, _ = _make_mock_chromadb()
        with _patch_store(mock_chromadb):
            store = VectorStore("test_col", dir="/tmp/test")
            mock_client.get_or_create_collection.assert_called_once()
            assert store.collection_name == "test_col"

    def test_raises_when_chromadb_unavailable(self):
        from ffai.rag.store import VectorStore
        with patch("ffai.rag.stores.chroma.CHROMADB_AVAILABLE", False):
            with pytest.raises(ImportError, match="chromadb"):
                VectorStore("test")


class TestVectorStoreSearch:
    def test_asearch_returns_search_hits(self):
        from ffai.rag.store import VectorStore
        mock_chromadb, _, mock_col = _make_mock_chromadb()
        mock_col.query.return_value = {
            "ids": [["id1"]],
            "documents": [["content"]],
            "metadatas": [[{"source": "doc1"}]],
            "distances": [[0.3]],
        }
        with _patch_store(mock_chromadb):
            store = VectorStore("test", dir="/tmp/test")
            hits = asyncio.run(store.asearch([0.1, 0.2], top_k=5))
            assert len(hits) == 1
            assert isinstance(hits[0], SearchHit)
            assert hits[0].content == "content"
            assert hits[0].score == pytest.approx(0.7)
            assert hits[0].source == "doc1"


class TestVectorStoreAdd:
    def test_aadd_stores_chunks(self):
        from ffai.rag.store import VectorStore
        mock_chromadb, _, mock_col = _make_mock_chromadb()
        with _patch_store(mock_chromadb):
            store = VectorStore("test", dir="/tmp/test")
            count = asyncio.run(store.aadd(
                ids=["id1"], texts=["text"],
                embeddings=[[0.1]], metadatas=[{"source": "a"}],
            ))
            assert count == 1
            mock_col.add.assert_called_once()


class TestVectorStoreLifecycle:
    def test_delete_by_source(self):
        from ffai.rag.store import VectorStore
        mock_chromadb, _, mock_col = _make_mock_chromadb()
        with _patch_store(mock_chromadb):
            store = VectorStore("test", dir="/tmp/test")
            store.delete_by_source("doc1")
            mock_col.delete.assert_called_once_with(where={"source": "doc1"})

    def test_count(self):
        from ffai.rag.store import VectorStore
        mock_chromadb, _, mock_col = _make_mock_chromadb()
        mock_col.count.return_value = 42
        with _patch_store(mock_chromadb):
            store = VectorStore("test", dir="/tmp/test")
            assert store.count() == 42

    def test_list_sources(self):
        from ffai.rag.store import VectorStore
        mock_chromadb, _, mock_col = _make_mock_chromadb()
        mock_col.get.return_value = {
            "ids": ["1", "2"],
            "metadatas": [{"source": "b"}, {"source": "a"}],
        }
        with _patch_store(mock_chromadb):
            store = VectorStore("test", dir="/tmp/test")
            assert store.list_sources() == ["a", "b"]

    def test_needs_reindex_true_when_not_found(self):
        from ffai.rag.store import VectorStore
        mock_chromadb, _, mock_col = _make_mock_chromadb()
        mock_col.get.return_value = {"metadatas": []}
        with _patch_store(mock_chromadb):
            store = VectorStore("test", dir="/tmp/test")
            assert store.needs_reindex("doc1", "abc") is True

    def test_needs_reindex_false_when_match(self):
        from ffai.rag.store import VectorStore
        mock_chromadb, _, mock_col = _make_mock_chromadb()
        mock_col.get.return_value = {"metadatas": [{"document_checksum": "abc"}]}
        with _patch_store(mock_chromadb):
            store = VectorStore("test", dir="/tmp/test")
            assert store.needs_reindex("doc1", "abc") is False
