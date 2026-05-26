from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.rag.splitters.base import TextChunk


def _make_mock_embeddings():
    from src.rag.embeddings import FFEmbeddings
    emb = MagicMock(spec=FFEmbeddings)
    emb.model = "mistral/mistral-embed"
    emb.embed.return_value = [[0.1, 0.2, 0.3]]
    emb.embed_single.return_value = [0.1, 0.2, 0.3]
    return emb


def _make_mock_vector_store():
    vs = MagicMock()
    vs.count.return_value = 0
    vs.get_stats.return_value = {
        "collection_name": "ffai_kb",
        "count": 0,
        "persist_dir": "./chroma_db",
        "embedding_model": "mistral/mistral-embed",
    }
    vs.needs_reindex.return_value = True
    vs.search.return_value = [
        {"id": "id1", "content": "result text", "metadata": {"reference_name": "doc1"}, "distance": 0.2},
    ]
    vs.list_documents.return_value = ["doc1"]
    vs.get_indexed_documents.return_value = []
    vs.get_all_documents.return_value = []
    vs.add_chunks.return_value = 3
    return vs


def _make_mock_chunker():
    chunker = MagicMock()
    chunker.chunk.return_value = [
        TextChunk(content="chunk text 1", chunk_index=0, start_char=0, end_char=12, metadata={"reference_name": "doc1"}),
        TextChunk(content="chunk text 2", chunk_index=1, start_char=12, end_char=24, metadata={"reference_name": "doc1"}),
        TextChunk(content="chunk text 3", chunk_index=2, start_char=24, end_char=36, metadata={"reference_name": "doc1"}),
    ]
    return chunker


def _build_client(**kwargs):
    from src.rag.client import RAGClient

    mock_vs = _make_mock_vector_store()
    mock_emb = _make_mock_embeddings()
    mock_chunker = _make_mock_chunker()

    if "embedding_model" not in kwargs:
        kwargs["embedding_model"] = mock_emb

    with (
        patch("src.rag.pipeline.get_chunker", return_value=mock_chunker),
        patch("src.rag.pipeline.FFEmbeddings", return_value=mock_emb),
        patch("src.rag.vector_store.FFVectorStore", return_value=mock_vs),
    ):
        client = RAGClient(**kwargs)

    return client, mock_vs, mock_emb, mock_chunker


class TestRAGClientInitDefaults:
    def test_stores_constructor_attributes(self):
        client, _, _, _ = _build_client()
        assert client.chunk_size == 1000
        assert client.chunk_overlap == 200
        assert client.n_results_default == 5
        assert client.chunking_strategy == "recursive"
        assert client.search_mode == "vector"
        assert client.hierarchical_enabled is False

    def test_default_embedding_model_is_mistral_embed(self):
        client, _, mock_emb, _ = _build_client()
        assert client.embeddings is mock_emb


class TestRAGClientInitExplicit:
    def test_custom_parameters_stored(self):
        client, _, _, _ = _build_client(
            chunk_size=500,
            chunk_overlap=50,
            n_results_default=10,
            chunking_strategy="markdown",
            search_mode="hybrid",
        )
        assert client.chunk_size == 500
        assert client.chunk_overlap == 50
        assert client.n_results_default == 10
        assert client.chunking_strategy == "markdown"
        assert client.search_mode == "hybrid"

    def test_ffembeddings_instance_passed_through(self):
        custom_emb = _make_mock_embeddings()
        mock_vs = _make_mock_vector_store()
        mock_chunker = _make_mock_chunker()
        with (
            patch("src.rag.pipeline.get_chunker", return_value=mock_chunker),
            patch("src.rag.vector_store.FFVectorStore", return_value=mock_vs),
        ):
            from src.rag.client import RAGClient
            client = RAGClient(embedding_model=custom_emb)
            assert client.embeddings is custom_emb


class TestRAGClientInitChromaUnavailable:
    def test_raises_importerror_when_chromadb_missing(self):
        from src.rag.client import RAGClient
        mock_emb = _make_mock_embeddings()
        mock_chunker = _make_mock_chunker()
        with (
            patch("src.rag.pipeline.get_chunker", return_value=mock_chunker),
            patch("src.rag.pipeline.FFEmbeddings", return_value=mock_emb),
            patch("src.rag.vector_store.FFVectorStore", side_effect=ImportError("chromadb")),
            pytest.raises(ImportError, match="chromadb"),
        ):
            RAGClient()


class TestRAGClientAddDocument:
    def test_returns_chunk_count_from_pipeline(self):
        client, mock_vs, _, _ = _build_client()
        mock_vs.add_chunks.return_value = 3
        result = client.add_document(
            "Some long document text to chunk",
            metadata={"source": "test"},
            reference_name="doc1",
        )
        assert result == 3

    def test_returns_zero_for_empty_content(self):
        client, _, _, _ = _build_client()
        assert client.add_document("") == 0
        assert client.add_document("   ") == 0


class TestRAGClientSearch:
    def test_search_returns_normalized_scores(self):
        client, mock_vs, _, _ = _build_client()
        mock_vs.search.return_value = [
            {"id": "id1", "content": "text", "metadata": {"reference_name": "doc1"}, "distance": 0.3},
        ]
        results = client.search("test query", n_results=5)
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(0.7)
        assert results[0]["content"] == "text"

    def test_search_uses_default_n_results(self):
        client, mock_vs, _, _ = _build_client(n_results_default=10)
        mock_vs.search.return_value = []
        client.search("query")
        mock_vs.search.assert_called_with("query", n_results=10, where=None)


class TestRAGClientSearchAndFormat:
    def test_returns_formatted_string(self):
        client, mock_vs, _, _ = _build_client()
        mock_vs.search.return_value = [
            {"id": "id1", "content": "chunk one", "metadata": {"reference_name": "ref1"}, "distance": 0.1},
            {"id": "id2", "content": "chunk two", "metadata": {"reference_name": "ref2"}, "distance": 0.2},
        ]
        formatted = client.search_and_format("test query")
        assert "[1]" in formatted
        assert "[2]" in formatted
        assert "ref1" in formatted
        assert "ref2" in formatted
        assert "chunk one" in formatted
        assert "chunk two" in formatted


class TestRAGClientFormatResultsForPrompt:
    def test_produces_numbered_sourced_output(self):
        client, _, _, _ = _build_client()
        results = [
            {"content": "alpha text", "metadata": {"reference_name": "doc_a"}, "score": 0.95},
            {"content": "beta text", "metadata": {"reference_name": "doc_b"}, "score": 0.80},
        ]
        output = client.format_results_for_prompt(results)
        assert "[1]" in output
        assert "[2]" in output
        assert "doc_a" in output
        assert "doc_b" in output
        assert "0.95" in output
        assert "0.80" in output

    def test_respects_max_chars_limit(self):
        client, _, _, _ = _build_client()
        results = [
            {"content": "short", "metadata": {"reference_name": "doc1"}, "score": 0.9},
            {"content": "B" * 500, "metadata": {"reference_name": "doc2"}, "score": 0.8},
        ]
        first_formatted = "[1] (source: doc1, relevance: 0.90)\nshort\n"
        output = client.format_results_for_prompt(results, max_chars=len(first_formatted))
        assert "doc1" in output
        assert "doc2" not in output

    def test_returns_empty_string_for_no_results(self):
        client, _, _, _ = _build_client()
        assert client.format_results_for_prompt([]) == ""

    def test_parent_context_included(self):
        client, _, _, _ = _build_client()
        results = [
            {
                "content": "child text",
                "metadata": {"reference_name": "doc1"},
                "score": 0.9,
                "parent_content": "parent context text",
            },
        ]
        output = client.format_results_for_prompt(results)
        assert "Parent context: parent context text" in output


class TestRAGClientIndexDocument:
    def test_skips_already_indexed_document(self):
        client, mock_vs, _, _ = _build_client()
        mock_vs.needs_reindex.return_value = False
        result = client.index_document(
            content="some text",
            reference_name="doc1",
            common_name="Document One",
            checksum="abc123",
        )
        assert result == 0
        mock_vs.delete_by_reference_and_strategy.assert_not_called()

    def test_indexes_new_document(self):
        client, mock_vs, _, _ = _build_client()
        mock_vs.needs_reindex.return_value = True
        mock_vs.add_chunks.return_value = 5
        result = client.index_document(
            content="new document text",
            reference_name="new_doc",
            common_name="New Doc",
            checksum="def456",
        )
        assert result == 5
        mock_vs.delete_by_reference_and_strategy.assert_called_once_with(
            reference_name="new_doc",
            chunking_strategy="recursive",
        )

    def test_force_reindex_even_when_unchanged(self):
        client, mock_vs, _, _ = _build_client()
        mock_vs.needs_reindex.return_value = False
        mock_vs.add_chunks.return_value = 2
        result = client.index_document(
            content="text",
            reference_name="doc1",
            common_name="Doc",
            checksum="abc",
            force=True,
        )
        assert result == 2
        mock_vs.delete_by_reference_and_strategy.assert_called_once()


class TestRAGClientDeleteByReference:
    def test_deletes_from_vector_store(self):
        client, mock_vs, _, _ = _build_client()
        client.delete_by_reference("my_doc")
        mock_vs.delete_by_reference.assert_called_once_with("my_doc")


class TestRAGClientGetStats:
    def test_returns_comprehensive_stats(self):
        client, mock_vs, _, _ = _build_client()
        mock_vs.get_stats.return_value = {
            "collection_name": "ffai_kb",
            "count": 42,
            "persist_dir": "./chroma_db",
            "embedding_model": "mistral/mistral-embed",
        }
        stats = client.get_stats()
        assert stats["collection_name"] == "ffai_kb"
        assert stats["count"] == 42
        assert stats["chunk_size"] == 1000
        assert stats["chunk_overlap"] == 200
        assert stats["chunking_strategy"] == "recursive"
        assert stats["search_mode"] == "vector"
        assert stats["hierarchical_enabled"] is False


class TestRAGClientSetLLMGenerateFn:
    def test_propagates_to_pipeline(self):
        client, _, _, _ = _build_client()
        mock_pipeline = MagicMock()
        client._pipeline = mock_pipeline
        def fn(prompt):
            return "response"
        client.set_llm_generate_fn(fn)
        mock_pipeline.set_llm_fn.assert_called_once_with(fn)


class TestRAGClientNoGetConfigImport:
    def test_client_module_has_no_get_config_import(self):
        import inspect

        import src.rag.client as client_module
        source = inspect.getsource(client_module)
        assert "get_config" not in source


class TestRAGClientProperties:
    def test_vector_store_property(self):
        client, mock_vs, _, _ = _build_client()
        assert client.vector_store is mock_vs

    def test_embeddings_property(self):
        _, _, mock_emb, _ = _build_client()
        from src.rag.client import RAGClient
        custom_emb = _make_mock_embeddings()
        mock_vs = _make_mock_vector_store()
        mock_chunker = _make_mock_chunker()
        with (
            patch("src.rag.pipeline.get_chunker", return_value=mock_chunker),
            patch("src.rag.vector_store.FFVectorStore", return_value=mock_vs),
        ):
            client = RAGClient(embedding_model=custom_emb)
            assert client.embeddings is custom_emb

    def test_chunker_property(self):
        client, _, _, mock_chunker = _build_client()
        assert client.chunker is mock_chunker

    def test_pipeline_property(self):
        client, _, _, _ = _build_client()
        assert client.pipeline is client._pipeline


class TestRAGClientCount:
    def test_count_delegates_to_vector_store(self):
        client, mock_vs, _, _ = _build_client()
        mock_vs.count.return_value = 99
        assert client.count() == 99


class TestRAGClientListDocuments:
    def test_list_documents_delegates(self):
        client, mock_vs, _, _ = _build_client()
        mock_vs.list_documents.return_value = ["doc_a", "doc_b"]
        assert client.list_documents() == ["doc_a", "doc_b"]


class TestRAGClientClear:
    def test_clear_delegates_to_vector_store(self):
        client, mock_vs, _, _ = _build_client()
        client.clear()
        mock_vs.clear.assert_called_once()


class TestFFRAGClientIsRAGClient:
    def test_alias(self):
        from src.rag.client import FFRAGClient, RAGClient
        assert FFRAGClient is RAGClient


class TestRAGClientAddDocuments:
    def test_loops_over_documents(self):
        client, mock_vs, _, _ = _build_client()
        mock_vs.add_chunks.return_value = 2
        docs = [
            {"content": "text one", "reference_name": "a"},
            {"content": "text two", "reference_name": "b"},
        ]
        total = client.add_documents(docs)
        assert total == 4
        assert mock_vs.add_chunks.call_count == 2
