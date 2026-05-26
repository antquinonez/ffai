from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.rag.splitters.base import TextChunk


def _make_chunk(content="hello world", chunk_index=0, start_char=0, end_char=11, metadata=None):
    return TextChunk(content=content, chunk_index=chunk_index, start_char=start_char, end_char=end_char, metadata=metadata)


def _make_mock_client_and_collection():
    collection = MagicMock()
    collection.count.return_value = 0
    collection.get.return_value = {"ids": [], "metadatas": [], "documents": []}
    collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    client.delete_collection.return_value = None
    return client, collection


def _make_mock_embeddings():
    embeddings = MagicMock()
    embeddings.model = "mistral/mistral-embed"
    embeddings.embed.return_value = [[0.1, 0.2, 0.3]]
    embeddings.embed_single.return_value = [0.1, 0.2, 0.3]
    return embeddings


class TestCHROMADB_AVAILABLE:
    def test_constant_exists_in_module(self):
        from src.rag.vector_store import CHROMADB_AVAILABLE
        assert isinstance(CHROMADB_AVAILABLE, bool)


class TestFFVectorStoreInit:
    def test_default_embedding_model_created(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, _ = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                assert store._embeddings is mock_emb
                assert store.collection_name == "test_col"
                assert store.persist_dir == "./chroma_db"

    def test_ffembeddings_instance_passed_through(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, _ = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
            store = FFVectorStore(collection_name="test_col", client=mock_client, embedding_model=mock_emb)
            assert store._embeddings is mock_emb

    def test_string_embedding_model_creates_ffembeddings(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, _ = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb) as mock_cls:
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client, embedding_model="openai/text-embedding-ada-002")
                mock_cls.assert_called_once_with(model="openai/text-embedding-ada-002")
                assert store._embeddings is mock_emb

    def test_raises_importerror_when_chromadb_unavailable(self):
        from src.rag.vector_store import FFVectorStore
        with patch("src.rag.vector_store.CHROMADB_AVAILABLE", False):
            with pytest.raises(ImportError, match="chromadb is not installed"):
                FFVectorStore(collection_name="test_col")

    def test_creates_persistent_client_when_no_client_provided(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, _ = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        mock_settings = MagicMock()
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                with patch("src.rag.vector_store.chromadb") as mock_chromadb:
                    with patch("src.rag.vector_store.Settings", mock_settings):
                        mock_chromadb.PersistentClient.return_value = mock_client
                        store = FFVectorStore(collection_name="test_col", persist_dir="/tmp/test_db")
                        mock_chromadb.PersistentClient.assert_called_once()

    def test_collection_created_with_cosine_space(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, _ = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                FFVectorStore(collection_name="test_col", client=mock_client)
                mock_client.get_or_create_collection.assert_called_once_with(
                    name="test_col",
                    metadata={"hnsw:space": "cosine"},
                )


class TestFFVectorStoreAddChunks:
    def test_embeds_texts_and_stores_with_metadata(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        mock_emb.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                chunks = [
                    _make_chunk(content="chunk A", chunk_index=0, start_char=0, end_char=7, metadata={"reference_name": "doc1"}),
                    _make_chunk(content="chunk B", chunk_index=1, start_char=7, end_char=14),
                ]
                result = store.add_chunks(chunks, chunking_strategy="recursive", document_checksum="abc123")
                assert result == 2
                mock_emb.embed.assert_called_once_with(["chunk A", "chunk B"])
                mock_collection.add.assert_called_once()
                call_kwargs = mock_collection.add.call_args[1]
                assert len(call_kwargs["ids"]) == 2
                assert call_kwargs["documents"] == ["chunk A", "chunk B"]
                assert len(call_kwargs["embeddings"]) == 2
                assert len(call_kwargs["metadatas"]) == 2
                assert call_kwargs["metadatas"][0]["_chunk_index"] == 0
                assert call_kwargs["metadatas"][0]["chunking_strategy"] == "recursive"
                assert call_kwargs["metadatas"][0]["document_checksum"] == "abc123"
                assert "indexed_at" in call_kwargs["metadatas"][0]

    def test_returns_zero_for_empty_chunks(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                result = store.add_chunks([])
                assert result == 0
                mock_collection.add.assert_not_called()

    def test_uses_texts_for_embedding_when_provided(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        mock_emb.embed.return_value = [[0.1, 0.2]]
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                chunks = [_make_chunk(content="plain text")]
                result = store.add_chunks(chunks, texts_for_embedding=["contextual: plain text"])
                assert result == 1
                mock_emb.embed.assert_called_once_with(["contextual: plain text"])
                call_docs = mock_collection.add.call_args[1]["documents"]
                assert call_docs == ["plain text"]

    def test_custom_ids_used_when_provided(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                chunks = [_make_chunk()]
                store.add_chunks(chunks, ids=["custom_id_1"])
                call_ids = mock_collection.add.call_args[1]["ids"]
                assert call_ids == ["custom_id_1"]


class TestFFVectorStoreDedup:
    def test_dedup_mode_filters_duplicates(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        mock_emb.embed.return_value = [[0.1, 0.2], [0.1, 0.2]]
        mock_dedup = MagicMock()
        mock_dedup.filter_duplicates.return_value = ([_make_chunk(content="unique")], [[0.1, 0.2]])
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                chunks = [_make_chunk(content="dup"), _make_chunk(content="dup")]
                with patch("src.rag.indexing.deduplication.ChunkDeduplicator", return_value=mock_dedup) as mock_cls:
                    result = store.add_chunks(chunks, dedup=True, dedup_mode="exact")
                    mock_cls.assert_called_once_with(mode="exact")
                    assert result == 1


class TestFFVectorStoreSearch:
    def test_embeds_query_and_returns_formatted_results(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["doc text 1", "doc text 2"]],
            "metadatas": [[{"reference_name": "r1"}, {"reference_name": "r2"}]],
            "distances": [[0.1, 0.5]],
        }
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                results = store.search("test query", n_results=2)
                mock_emb.embed_single.assert_called_once_with("test query")
                assert len(results) == 2
                assert results[0]["id"] == "id1"
                assert results[0]["content"] == "doc text 1"
                assert results[0]["metadata"] == {"reference_name": "r1"}
                assert results[0]["distance"] == 0.1
                assert results[1]["id"] == "id2"
                assert results[1]["distance"] == 0.5

    def test_returns_empty_when_no_results(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                results = store.search("empty query")
                assert results == []

    def test_passes_where_filter_to_collection(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        mock_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                store.search("q", where={"reference_name": "doc1"})
                call_kwargs = mock_collection.query.call_args[1]
                assert call_kwargs["where"] == {"reference_name": "doc1"}


class TestFFVectorStoreDeleteByReference:
    def test_calls_collection_delete_with_metadata_filter(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                store.delete_by_reference("my_doc")
                mock_collection.delete.assert_called_once_with(where={"reference_name": "my_doc"})


class TestFFVectorStoreDeleteByReferenceAndStrategy:
    def test_calls_collection_delete_with_combined_filter(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                result = store.delete_by_reference_and_strategy("my_doc", "recursive")
                assert result == 0
                mock_collection.delete.assert_called_once_with(
                    where={"$and": [{"reference_name": "my_doc"}, {"chunking_strategy": "recursive"}]}
                )


class TestFFVectorStoreNeedsReindex:
    def test_returns_true_when_no_existing_chunks(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        mock_collection.get.return_value = {"metadatas": []}
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                assert store.needs_reindex("doc1", "abc", "recursive") is True

    def test_returns_true_when_checksum_changed(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        mock_collection.get.return_value = {"metadatas": [{"document_checksum": "old_checksum"}]}
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                assert store.needs_reindex("doc1", "new_checksum", "recursive") is True

    def test_returns_false_when_checksum_unchanged(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        mock_collection.get.return_value = {"metadatas": [{"document_checksum": "same_checksum"}]}
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                assert store.needs_reindex("doc1", "same_checksum", "recursive") is False

    def test_returns_true_for_empty_metadata_key(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        mock_collection.get.return_value = {"metadatas": [{}]}
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                assert store.needs_reindex("doc1", "abc", "recursive") is True


class TestFFVectorStoreGetIndexedDocuments:
    def test_returns_deduplicated_document_list(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        mock_collection.get.return_value = {
            "metadatas": [
                {"reference_name": "doc1", "chunking_strategy": "recursive", "document_checksum": "c1", "indexed_at": "2025-01-01T00:00:00", "tags": ""},
                {"reference_name": "doc1", "chunking_strategy": "recursive", "document_checksum": "c1", "indexed_at": "2025-01-02T00:00:00", "tags": ""},
                {"reference_name": "doc2", "chunking_strategy": "markdown", "document_checksum": "c2", "indexed_at": "2025-01-01T00:00:00", "tags": "tag1"},
            ]
        }
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                result = store.get_indexed_documents()
                assert len(result) == 2
                doc1_entries = [d for d in result if d["reference_name"] == "doc1" and d["chunking_strategy"] == "recursive"]
                assert len(doc1_entries) == 1
                assert doc1_entries[0]["indexed_at"] == "2025-01-02T00:00:00"
                doc2_entries = [d for d in result if d["reference_name"] == "doc2"]
                assert doc2_entries[0]["tags"] == "tag1"

    def test_filters_by_chunking_strategy(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        mock_collection.get.return_value = {"metadatas": []}
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                store.get_indexed_documents(chunking_strategy="recursive")
                mock_collection.get.assert_called_with(
                    where={"chunking_strategy": "recursive"},
                    include=["metadatas"],
                )

    def test_returns_empty_for_no_metadatas(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        mock_collection.get.return_value = {"metadatas": None}
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                assert store.get_indexed_documents() == []


class TestFFVectorStoreListDocuments:
    def test_returns_sorted_reference_names(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        mock_collection.get.return_value = {
            "metadatas": [
                {"reference_name": "charlie"},
                {"reference_name": "alpha"},
                {"reference_name": "bravo"},
                {"reference_name": "alpha"},
            ]
        }
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                result = store.list_documents()
                assert result == ["alpha", "bravo", "charlie"]

    def test_returns_empty_when_no_documents(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        mock_collection.get.return_value = {"metadatas": []}
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                assert store.list_documents() == []


class TestFFVectorStoreCount:
    def test_returns_collection_count(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        mock_collection.count.return_value = 42
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                assert store.count() == 42


class TestFFVectorStoreGetStats:
    def test_returns_dict_with_collection_info(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        mock_collection.count.return_value = 15
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", persist_dir="/tmp/test_db", client=mock_client)
                stats = store.get_stats()
                assert stats == {
                    "collection_name": "test_col",
                    "count": 15,
                    "persist_dir": "/tmp/test_db",
                    "embedding_model": "mistral/mistral-embed",
                }


class TestFFVectorStoreClear:
    def test_deletes_and_recreates_collection(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                store.clear()
                mock_client.delete_collection.assert_called_once_with("test_col")
                assert mock_client.get_or_create_collection.call_count == 2
                second_call = mock_client.get_or_create_collection.call_args_list[1]
                assert second_call[1] == {"name": "test_col", "metadata": {"hnsw:space": "cosine"}}


class TestFFVectorStoreDelete:
    def test_delete_with_ids(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                store.delete(ids=["id1", "id2"])
                mock_collection.delete.assert_called_once_with(ids=["id1", "id2"], where=None)

    def test_delete_with_where_filter(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                store.delete(where={"reference_name": "doc1"})
                mock_collection.delete.assert_called_once_with(ids=None, where={"reference_name": "doc1"})

    def test_delete_raises_without_ids_or_where(self):
        from src.rag.vector_store import FFVectorStore
        mock_client, mock_collection = _make_mock_client_and_collection()
        mock_emb = _make_mock_embeddings()
        with patch("src.rag.vector_store.FFEmbeddings", return_value=mock_emb):
            with patch("src.rag.vector_store.CHROMADB_AVAILABLE", True):
                store = FFVectorStore(collection_name="test_col", client=mock_client)
                with pytest.raises(ValueError, match="Must provide either ids or where filter"):
                    store.delete()
