from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.embed import Embeddings


class TestEmbeddingsInit:
    def test_api_model_stores_config(self):
        emb = Embeddings("mistral/mistral-embed", api_key="test-key")
        assert emb.model == "mistral/mistral-embed"
        assert emb.api_key == "test-key"
        assert emb.is_local is False
        assert emb.provider == "mistral"

    def test_local_model_detected(self):
        with pytest.raises(ImportError, match="sentence-transformers"):
            Embeddings("local/all-MiniLM-L6-v2")

    def test_cache_enabled_by_default(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key")
        stats = emb.cache_stats()
        assert stats["enabled"] is True
        assert stats["max_size"] == 256
        assert stats["entries"] == 0


class TestEmbeddingsAsync:
    def test_aembed_returns_vectors(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key")
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1, 0.2], "index": 0}]
        with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response):
            import asyncio

            vectors = asyncio.run(emb.aembed(["hello"]))
            assert vectors == [[0.1, 0.2]]

    def test_aembed_uses_cache(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key")
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1, 0.2], "index": 0}]
        with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response) as mock_aemb:
            asyncio.run(emb.aembed(["hello"]))
            asyncio.run(emb.aembed(["hello"]))
            assert mock_aemb.call_count == 1
            assert emb.cache_stats()["entries"] == 1

    def test_aembed_empty_returns_empty(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key")
        import asyncio

        assert asyncio.run(emb.aembed([])) == []

    def test_aembed_single(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key")
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.3, 0.4], "index": 0}]
        with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response):
            import asyncio

            vec = asyncio.run(emb.aembed_single("test"))
            assert vec == [0.3, 0.4]


class TestEmbeddingsCosineSimilarity:
    def test_identical_vectors(self):
        assert Embeddings.cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert Embeddings.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert Embeddings.cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert Embeddings.cosine_similarity([0, 0], [1, 1]) == 0.0


class TestEmbeddingsCache:
    def test_clear_cache_returns_count(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key")
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1], "index": 0}]
        with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response):
            import asyncio

            asyncio.run(emb.aembed(["hello"]))
        assert emb.clear_cache() == 1
        assert emb.cache_stats()["entries"] == 0
