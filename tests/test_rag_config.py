"""Tests for RAG configuration classes and integration with Config."""

from __future__ import annotations

from src.config import (
    Config,
    RAGChunkingConfig,
    RAGConfig,
    RAGSearchConfig,
    get_config,
    reload_config,
)


class TestRAGConfigDefaults:
    """Test RAGConfig has correct default values."""

    def test_rag_config_disabled_by_default(self):
        rag = RAGConfig()
        assert rag.enabled is False

    def test_rag_config_default_collection_name(self):
        rag = RAGConfig()
        assert rag.collection_name == "ffai_kb"

    def test_rag_config_default_persist_dir(self):
        rag = RAGConfig()
        assert rag.persist_dir == "./chroma_db"

    def test_rag_config_default_embedding_model(self):
        rag = RAGConfig()
        assert rag.embedding_model == "mistral/mistral-embed"

    def test_rag_config_default_embedding_cache_size(self):
        rag = RAGConfig()
        assert rag.embedding_cache_size == 256


class TestRAGChunkingDefaults:
    """Test RAGChunkingConfig has correct default values."""

    def test_strategy_default(self):
        chunking = RAGChunkingConfig()
        assert chunking.strategy == "recursive"

    def test_chunk_size_default(self):
        chunking = RAGChunkingConfig()
        assert chunking.chunk_size == 1000

    def test_chunk_overlap_default(self):
        chunking = RAGChunkingConfig()
        assert chunking.chunk_overlap == 200

    def test_contextual_headers_default(self):
        chunking = RAGChunkingConfig()
        assert chunking.contextual_headers is True


class TestRAGSearchDefaults:
    """Test RAGSearchConfig has correct default values."""

    def test_mode_default(self):
        search = RAGSearchConfig()
        assert search.mode == "vector"

    def test_n_results_default(self):
        search = RAGSearchConfig()
        assert search.n_results_default == 5

    def test_hybrid_alpha_default(self):
        search = RAGSearchConfig()
        assert search.hybrid_alpha == 0.6

    def test_rerank_default(self):
        search = RAGSearchConfig()
        assert search.rerank is False

    def test_rerank_model_default(self):
        search = RAGSearchConfig()
        assert search.rerank_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"


class TestConfigRAGIntegration:
    """Test RAG field integrates with the Config class."""

    def test_config_rag_is_rag_config_instance(self):
        config = Config()
        assert isinstance(config.rag, RAGConfig)

    def test_config_rag_enabled_defaults_false(self):
        config = Config()
        assert config.rag.enabled is False

    def test_config_rag_chunking_accessible(self):
        config = Config()
        assert config.rag.chunking.strategy == "recursive"
        assert config.rag.chunking.chunk_size == 1000

    def test_config_rag_search_accessible(self):
        config = Config()
        assert config.rag.search.mode == "vector"
        assert config.rag.search.hybrid_alpha == 0.6

    def test_config_created_without_yaml_files(self):
        config = Config()
        assert config.rag.enabled is False
        assert config.rag.collection_name == "ffai_kb"

    def test_rag_env_override(self, monkeypatch):
        monkeypatch.setenv("RAG__ENABLED", "true")
        config = reload_config()
        assert config.rag.enabled is True

    def test_rag_nested_env_override(self, monkeypatch):
        monkeypatch.setenv("RAG__CHUNKING__CHUNK_SIZE", "500")
        config = reload_config()
        assert config.rag.chunking.chunk_size == 500

    def test_rag_via_get_config(self):
        config = get_config()
        assert hasattr(config, "rag")
        assert isinstance(config.rag, RAGConfig)
