from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from ffai.rag.types import SearchHit

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


class TestQdrantVectorStoreAdd:
    def _make_store(self):
        from ffai.rag.stores.qdrant import QdrantVectorStore
        mock_mod, mock_client, _, _ = _make_mock_qdrant()
        ctx = _patch_qdrant(mock_mod)
        ctx.__enter__()
        store = QdrantVectorStore(collection_name="test_col", embedding_dim=4)
        return store, mock_client, ctx

    def test_aadd_returns_count(self):
        store, mock_client, ctx = self._make_store()
        try:
            count = asyncio.run(store.aadd(
                ids=["id1", "id2"],
                texts=["text1", "text2"],
                embeddings=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                metadatas=[{"source": "a"}, {"source": "b"}],
            ))
            assert count == 2
        finally:
            ctx.__exit__(None, None, None)

    def test_aadd_calls_upsert(self):
        store, mock_client, ctx = self._make_store()
        try:
            asyncio.run(store.aadd(
                ids=["id1"],
                texts=["text1"],
                embeddings=[[0.1, 0.2, 0.3, 0.4]],
                metadatas=[{"source": "a"}],
            ))
            mock_client.upsert.assert_called_once()
            call_kwargs = mock_client.upsert.call_args
            assert call_kwargs.kwargs["collection_name"] == "test_col"
            assert len(call_kwargs.kwargs["points"]) == 1
        finally:
            ctx.__exit__(None, None, None)

    def test_aadd_converts_non_uuid_ids(self):
        store, mock_client, ctx = self._make_store()
        try:
            asyncio.run(store.aadd(
                ids=["not-a-uuid"],
                texts=["text"],
                embeddings=[[0.1, 0.2, 0.3, 0.4]],
                metadatas=[{"source": "a"}],
            ))
            points = mock_client.upsert.call_args.kwargs["points"]
            point_id = points[0].id
            assert str(point_id) != "not-a-uuid"
        finally:
            ctx.__exit__(None, None, None)

    def test_aadd_passes_uuid_ids_unchanged(self):
        store, mock_client, ctx = self._make_store()
        try:
            from ffai.rag.stores.qdrant import QdrantVectorStore
            uid_str = "550e8400-e29b-41d4-a716-446655440000"
            uid = QdrantVectorStore._ensure_uuid(uid_str)
            assert str(uid) == uid_str
        finally:
            ctx.__exit__(None, None, None)

    def test_aadd_builds_correct_point_count(self):
        store, mock_client, ctx = self._make_store()
        try:
            asyncio.run(store.aadd(
                ids=["id1", "id2", "id3"],
                texts=["t1", "t2", "t3"],
                embeddings=[[0.1] * 4, [0.2] * 4, [0.3] * 4],
                metadatas=[{"source": "a"}, {"source": "b"}, {"source": "c"}],
            ))
            points = mock_client.upsert.call_args.kwargs["points"]
            assert len(points) == 3
        finally:
            ctx.__exit__(None, None, None)


class TestQdrantVectorStoreSearch:
    def _make_store(self):
        from ffai.rag.stores.qdrant import QdrantVectorStore
        mock_mod, mock_client, _, _ = _make_mock_qdrant()
        ctx = _patch_qdrant(mock_mod)
        ctx.__enter__()
        store = QdrantVectorStore(collection_name="test_col", embedding_dim=4)
        return store, mock_client, ctx

    def _mock_query_result(self, points):
        result = MagicMock()
        result.points = points
        return result

    def test_asearch_returns_search_hits(self):
        store, mock_client, ctx = self._make_store()
        try:
            mock_point = MagicMock()
            mock_point.id = "abc-123"
            mock_point.score = 0.85
            mock_point.payload = {"content": "Rust memory safety", "source": "rust.md"}
            mock_client.query_points.return_value = self._mock_query_result([mock_point])

            hits = asyncio.run(store.asearch([0.1, 0.2, 0.3, 0.4], top_k=5))
            assert len(hits) == 1
            assert isinstance(hits[0], SearchHit)
            assert hits[0].content == "Rust memory safety"
            assert hits[0].score == pytest.approx(0.85)
            assert hits[0].source == "rust.md"
        finally:
            ctx.__exit__(None, None, None)

    def test_asearch_converts_score_none_to_zero(self):
        store, mock_client, ctx = self._make_store()
        try:
            mock_point = MagicMock()
            mock_point.id = "abc-123"
            mock_point.score = None
            mock_point.payload = {"content": "text", "source": "doc"}
            mock_client.query_points.return_value = self._mock_query_result([mock_point])

            hits = asyncio.run(store.asearch([0.1, 0.2, 0.3, 0.4]))
            assert hits[0].score == 0.0
        finally:
            ctx.__exit__(None, None, None)

    def test_asearch_returns_empty_when_no_results(self):
        store, mock_client, ctx = self._make_store()
        try:
            mock_client.query_points.return_value = self._mock_query_result([])
            hits = asyncio.run(store.asearch([0.1, 0.2, 0.3, 0.4]))
            assert hits == []
        finally:
            ctx.__exit__(None, None, None)

    def test_asearch_passes_top_k_and_filter(self):
        store, mock_client, ctx = self._make_store()
        try:
            mock_client.query_points.return_value = self._mock_query_result([])
            asyncio.run(store.asearch(
                [0.1, 0.2, 0.3, 0.4],
                top_k=3,
                where={"source": "doc1"},
            ))
            call_kwargs = mock_client.query_points.call_args.kwargs
            assert call_kwargs["limit"] == 3
            assert call_kwargs["query_filter"] is not None
        finally:
            ctx.__exit__(None, None, None)

    def test_asearch_without_filter_passes_none(self):
        store, mock_client, ctx = self._make_store()
        try:
            mock_client.query_points.return_value = self._mock_query_result([])
            asyncio.run(store.asearch([0.1, 0.2, 0.3, 0.4]))
            call_kwargs = mock_client.query_points.call_args.kwargs
            assert call_kwargs["query_filter"] is None
        finally:
            ctx.__exit__(None, None, None)

    def test_asearch_handles_missing_content_in_payload(self):
        store, mock_client, ctx = self._make_store()
        try:
            mock_point = MagicMock()
            mock_point.id = "abc-123"
            mock_point.score = 0.5
            mock_point.payload = {"source": "doc"}
            mock_client.query_points.return_value = self._mock_query_result([mock_point])

            hits = asyncio.run(store.asearch([0.1, 0.2, 0.3, 0.4]))
            assert hits[0].content == ""
        finally:
            ctx.__exit__(None, None, None)

    def test_asearch_preserves_extra_metadata(self):
        store, mock_client, ctx = self._make_store()
        try:
            mock_point = MagicMock()
            mock_point.id = "abc-123"
            mock_point.score = 0.9
            mock_point.payload = {
                "content": "text",
                "source": "doc",
                "chunking_strategy": "recursive",
                "document_checksum": "sha256:abc",
            }
            mock_client.query_points.return_value = self._mock_query_result([mock_point])

            hits = asyncio.run(store.asearch([0.1, 0.2, 0.3, 0.4]))
            assert hits[0].metadata["chunking_strategy"] == "recursive"
            assert hits[0].metadata["document_checksum"] == "sha256:abc"
            assert "content" not in hits[0].metadata
        finally:
            ctx.__exit__(None, None, None)

    def test_asearch_multiple_results_ranked(self):
        store, mock_client, ctx = self._make_store()
        try:
            p1 = MagicMock(id="1", score=0.9, payload={"content": "first", "source": "a"})
            p2 = MagicMock(id="2", score=0.7, payload={"content": "second", "source": "b"})
            p3 = MagicMock(id="3", score=0.5, payload={"content": "third", "source": "c"})
            mock_client.query_points.return_value = self._mock_query_result([p1, p2, p3])

            hits = asyncio.run(store.asearch([0.1, 0.2, 0.3, 0.4], top_k=3))
            assert len(hits) == 3
            assert hits[0].score > hits[1].score > hits[2].score
            assert hits[0].content == "first"
            assert hits[2].content == "third"
        finally:
            ctx.__exit__(None, None, None)
