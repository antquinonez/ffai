from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.qdrant


def _make_mock_qdrant():
    mock_client = MagicMock()
    mock_collection_info = MagicMock()
    mock_collection_info.points_count = 42
    mock_client.get_collection.return_value = mock_collection_info
    mock_client.get_collections.return_value = MagicMock(collections=[])
    mock_client.create_collection.return_value = None
    mock_client.scroll.return_value = ([], None)

    mock_async_client = MagicMock()

    mock_models = MagicMock()
    mock_models.Distance = MagicMock()
    mock_models.VectorParams = MagicMock()
    mock_models.FieldCondition = MagicMock()
    mock_models.Filter = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    mock_models.MatchValue = MagicMock()
    mock_models.PointStruct = MagicMock()

    mock_qdrant_client_mod = MagicMock()
    mock_qdrant_client_mod.QdrantClient.return_value = mock_client
    mock_qdrant_client_mod.AsyncQdrantClient.return_value = mock_async_client

    mock_qdrant_mod = MagicMock()
    mock_qdrant_mod.QdrantClient = mock_qdrant_client_mod.QdrantClient
    mock_qdrant_mod.AsyncQdrantClient = mock_qdrant_client_mod.AsyncQdrantClient
    mock_qdrant_mod.models = mock_models

    return mock_qdrant_mod, mock_client, mock_async_client, mock_models


@contextmanager
def _patch_qdrant(mock_qdrant_mod):
    import ffai.rag.stores.qdrant as qdrant_mod

    originals = {
        "QDRANT_AVAILABLE": qdrant_mod.QDRANT_AVAILABLE,
        "QdrantClient": qdrant_mod.QdrantClient,
        "AsyncQdrantClient": qdrant_mod.AsyncQdrantClient,
        "Distance": getattr(qdrant_mod, "Distance", None),
        "VectorParams": getattr(qdrant_mod, "VectorParams", None),
        "FieldCondition": getattr(qdrant_mod, "FieldCondition", None),
        "Filter": getattr(qdrant_mod, "Filter", None),
        "MatchValue": getattr(qdrant_mod, "MatchValue", None),
        "PointStruct": getattr(qdrant_mod, "PointStruct", None),
    }

    qdrant_mod.QDRANT_AVAILABLE = True
    qdrant_mod.QdrantClient = mock_qdrant_mod.QdrantClient
    qdrant_mod.AsyncQdrantClient = mock_qdrant_mod.AsyncQdrantClient
    qdrant_mod.Distance = mock_qdrant_mod.models.Distance
    qdrant_mod.VectorParams = mock_qdrant_mod.models.VectorParams
    qdrant_mod.FieldCondition = mock_qdrant_mod.models.FieldCondition
    qdrant_mod.Filter = mock_qdrant_mod.models.Filter
    qdrant_mod.MatchValue = mock_qdrant_mod.models.MatchValue
    qdrant_mod.PointStruct = mock_qdrant_mod.models.PointStruct

    try:
        yield
    finally:
        for key, val in originals.items():
            setattr(qdrant_mod, key, val)


class TestQdrantVectorStoreInit:
    def test_creates_store(self):
        from ffai.rag.stores.qdrant import QdrantVectorStore
        mock_mod, _, _, _ = _make_mock_qdrant()
        with _patch_qdrant(mock_mod):
            store = QdrantVectorStore(collection_name="test_col", embedding_dim=512)
            assert store.name == "qdrant"

    def test_raises_when_deps_missing(self):
        from ffai.rag.stores.qdrant import QdrantVectorStore
        with patch.object(
            __import__("ffai.rag.stores.qdrant", fromlist=["QDRANT_AVAILABLE"]),
            "QDRANT_AVAILABLE", False,
        ):
            with pytest.raises(ImportError, match="qdrant-client"):
                QdrantVectorStore()

    def test_creates_collection_if_missing(self):
        from ffai.rag.stores.qdrant import QdrantVectorStore
        mock_mod, mock_client, _, _ = _make_mock_qdrant()
        mock_client.get_collections.return_value = MagicMock(collections=[])
        with _patch_qdrant(mock_mod):
            QdrantVectorStore(collection_name="new_col")
        mock_client.create_collection.assert_called_once()


class TestQdrantVectorStoreOperations:
    def _make_store(self):
        from ffai.rag.stores.qdrant import QdrantVectorStore
        mock_mod, mock_client, mock_async, mock_models = _make_mock_qdrant()
        ctx = _patch_qdrant(mock_mod)
        ctx.__enter__()
        store = QdrantVectorStore(collection_name="test_col")
        return store, mock_client, mock_async, ctx

    def test_count_returns_points_count(self):
        store, mock_client, _, ctx = self._make_store()
        try:
            assert store.count() == 42
        finally:
            ctx.__exit__(None, None, None)

    def test_delete_by_source(self):
        store, mock_client, _, ctx = self._make_store()
        try:
            store.delete_by_source("doc1")
            mock_client.delete.assert_called_once()
        finally:
            ctx.__exit__(None, None, None)

    def test_clear_deletes_and_recreates(self):
        store, mock_client, _, ctx = self._make_store()
        try:
            store.clear()
            mock_client.delete_collection.assert_called_once()
            mock_client.create_collection.assert_called()
        finally:
            ctx.__exit__(None, None, None)

    def test_list_sources_returns_sorted(self):
        store, mock_client, _, ctx = self._make_store()
        try:
            point_b = MagicMock(payload={"source": "b", "content": "text"})
            point_a = MagicMock(payload={"source": "a", "content": "text"})
            mock_client.scroll.return_value = ([point_b, point_a], None)
            assert store.list_sources() == ["a", "b"]
        finally:
            ctx.__exit__(None, None, None)

    def test_needs_reindex_true_when_no_results(self):
        store, mock_client, _, ctx = self._make_store()
        try:
            mock_client.scroll.return_value = ([], None)
            assert store.needs_reindex("doc1", "abc") is True
        finally:
            ctx.__exit__(None, None, None)

    def test_needs_reindex_false_when_checksum_matches(self):
        store, mock_client, _, ctx = self._make_store()
        try:
            point = MagicMock(payload={"document_checksum": "abc", "source": "doc1"})
            mock_client.scroll.return_value = ([point], None)
            assert store.needs_reindex("doc1", "abc") is False
        finally:
            ctx.__exit__(None, None, None)


class TestQdrantBuildFilter:
    def _make_store(self):
        from ffai.rag.stores.qdrant import QdrantVectorStore
        mock_mod, _, _, _ = _make_mock_qdrant()
        ctx = _patch_qdrant(mock_mod)
        ctx.__enter__()
        store = QdrantVectorStore()
        return store, ctx

    def test_empty_where_returns_none(self):
        store, ctx = self._make_store()
        try:
            assert store._build_filter(None) is None
        finally:
            ctx.__exit__(None, None, None)

    def test_simple_filter(self):
        store, ctx = self._make_store()
        try:
            f = store._build_filter({"source": "doc1"})
            assert f is not None
        finally:
            ctx.__exit__(None, None, None)

    def test_compound_filter(self):
        store, ctx = self._make_store()
        try:
            f = store._build_filter(
                {"$and": [{"source": "doc1"}, {"chunking_strategy": "recursive"}]}
            )
            assert f is not None
        finally:
            ctx.__exit__(None, None, None)


class TestQdrantEnsureUuid:
    def test_valid_uuid_passthrough(self):
        import uuid

        from ffai.rag.stores.qdrant import QdrantVectorStore
        uid = str(uuid.uuid4())
        result = QdrantVectorStore._ensure_uuid(uid)
        assert str(result) == uid

    def test_non_uuid_gets_deterministic_uuid5(self):
        from ffai.rag.stores.qdrant import QdrantVectorStore
        r1 = QdrantVectorStore._ensure_uuid("chunk_0")
        r2 = QdrantVectorStore._ensure_uuid("chunk_0")
        assert r1 == r2
        assert str(r1) != "chunk_0"

    def test_different_ids_produce_different_uuids(self):
        from ffai.rag.stores.qdrant import QdrantVectorStore
        r1 = QdrantVectorStore._ensure_uuid("chunk_0")
        r2 = QdrantVectorStore._ensure_uuid("chunk_1")
        assert r1 != r2


class TestQdrantLocalMode:
    def test_local_mode_uses_sync_client_for_aadd(self):
        from ffai.rag.stores.qdrant import QdrantVectorStore
        mock_mod, mock_client, _, _ = _make_mock_qdrant()
        with _patch_qdrant(mock_mod):
            store = QdrantVectorStore(
                path="/tmp/test_qdrant_local",
                embedding_dim=8,
                collection_name="test",
            )
            assert store._is_local is True

    def test_server_mode_is_not_local(self):
        from ffai.rag.stores.qdrant import QdrantVectorStore
        mock_mod, mock_client, _, _ = _make_mock_qdrant()
        with _patch_qdrant(mock_mod):
            store = QdrantVectorStore(
                url="http://localhost:6333",
                collection_name="test",
            )
            assert store._is_local is False
