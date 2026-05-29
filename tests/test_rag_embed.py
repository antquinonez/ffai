from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ffai.rag.embed import Embeddings


class TestEmbeddingsInit:
    def test_api_model_stores_config(self):
        emb = Embeddings("mistral/mistral-embed", api_key="test-key")
        assert emb.model == "mistral/mistral-embed"
        assert emb.api_key == "test-key"
        assert emb.is_local is False
        assert emb.provider == "mistral"

    def test_local_model_detected(self):
        with patch.dict("sys.modules", {"fastembed": None}):
            with pytest.raises(ImportError, match="No local embedding backend found"):
                Embeddings("local/all-MiniLM-L6-v2")

    def test_cache_enabled_by_default(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key")
        stats = emb.cache_stats()
        assert stats["enabled"] is True
        assert stats["max_size"] == 256
        assert stats["entries"] == 0

    def test_api_base_stored(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key", api_base="https://custom.api/v1")
        assert emb.api_base == "https://custom.api/v1"

    def test_extra_kwargs_stored(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key", timeout=30)
        assert emb._extra_kwargs == {"timeout": 30}

    def test_default_api_key_from_env(self):
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "env-key"}):
            emb = Embeddings("mistral/mistral-embed")
            assert emb.api_key == "env-key"

    def test_default_api_key_openai_provider(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "oai-key"}):
            emb = Embeddings("openai/text-embedding-3-small")
            assert emb.api_key == "oai-key"

    def test_default_api_key_unknown_provider(self):
        with patch.dict("os.environ", {"CUSTOM_API_KEY": "custom-key"}):
            emb = Embeddings("custom/model-x")
            assert emb.api_key == "custom-key"

    def test_default_api_key_no_slash(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "oai-key"}):
            emb = Embeddings("model-no-slash")
            assert emb.api_key == "oai-key"

    def test_no_api_key_available(self):
        with patch.dict("os.environ", {}, clear=True):
            emb = Embeddings("mistral/mistral-embed")
            assert emb.api_key is None

    def test_provider_extracted_from_model_prefix(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key")
        assert emb.provider == "mistral"

    def test_provider_unknown_no_slash(self):
        with patch.dict("os.environ", {}, clear=True):
            emb = Embeddings("noslashmodel")
            assert emb.provider == "unknown"

    def test_explicit_api_key_overrides_env(self):
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "env-key"}):
            emb = Embeddings("mistral/mistral-embed", api_key="explicit-key")
            assert emb.api_key == "explicit-key"

    def test_device_stored(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key", device="cuda")
        assert emb._device == "cuda"


class TestEmbeddingsLocalModel:
    def test_local_init_success(self):
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        mock_model.encode.return_value.tolist = lambda: [[0.1, 0.2, 0.3]]
        with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=MagicMock(return_value=mock_model))}):
            emb = Embeddings("local/test-model")
            assert emb.is_local is True
            assert emb.provider == "local"
            assert emb._local_model is mock_model

    def test_local_embed_caches_result(self):
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=MagicMock(return_value=mock_model))}):
            emb = Embeddings("local/test-model")
            vectors = asyncio.run(emb.aembed(["hello"]))
            assert vectors == [[0.1, 0.2]]
            assert emb.cache_stats()["entries"] == 1

    def test_local_embed_returns_from_cache(self):
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2]])
        with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=MagicMock(return_value=mock_model))}):
            emb = Embeddings("local/test-model")
            asyncio.run(emb.aembed(["hello"]))
            asyncio.run(emb.aembed(["hello"]))
            assert mock_model.encode.call_count == 1
            assert emb.cache_stats()["entries"] == 1

    def test_local_embed_multiple_texts(self):
        import numpy as np

        mock_model = MagicMock()
        arr = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_model.encode.return_value = arr
        with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=MagicMock(return_value=mock_model))}):
            emb = Embeddings("local/test-model")
            vectors = asyncio.run(emb.aembed(["hello", "world"]))
            assert len(vectors) == 2
            assert vectors[0] == [pytest.approx(0.1), pytest.approx(0.2)]
            assert vectors[1] == [pytest.approx(0.3), pytest.approx(0.4)]

    def test_local_embed_no_model_raises(self):
        from collections import OrderedDict

        emb = object.__new__(Embeddings)
        emb._is_local = True
        emb._local_model = None
        emb._local_backend = None
        emb._cache_enabled = True
        emb._cache_size = 256
        emb._cache = OrderedDict()
        emb.model = "local/test"
        with pytest.raises(RuntimeError, match="Local model not initialized"):
            emb._embed_local(["hello"])

    def test_local_embed_cache_eviction(self):
        import numpy as np

        mock_model = MagicMock()
        with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=MagicMock(return_value=mock_model))}):
            emb = Embeddings("local/test-model", cache_size=2)
            responses = [
                np.array([[float(i), float(i) + 0.1]])
                for i in range(4)
            ]
            mock_model.encode.side_effect = responses
            for i in range(4):
                asyncio.run(emb.aembed([f"text-{i}"]))
            assert emb.cache_stats()["entries"] == 2

    def test_local_embed_str_input(self):
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.5, 0.6]])
        with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=MagicMock(return_value=mock_model))}):
            emb = Embeddings("local/test-model")
            vectors = asyncio.run(emb.aembed("hello"))
            assert vectors == [[0.5, 0.6]]


try:
    import fastembed as _fastembed  # noqa: F401

    _fastembed_available = True
except ImportError:
    _fastembed_available = False


@pytest.mark.skipif(not _fastembed_available, reason="fastembed not installed")
class TestEmbeddingsLocalFastembedFallback:
    def test_fastembed_fallback_init(self):
        mock_fe_model = MagicMock()
        with patch("fastembed.TextEmbedding", return_value=mock_fe_model):
            emb = Embeddings("local/all-MiniLM-L6-v2")
            assert emb._local_backend == "fastembed"
            assert emb._local_model is mock_fe_model

    def test_fastembed_model_name_mapping(self):
        mock_fe_model = MagicMock()
        mock_fe_cls = MagicMock(return_value=mock_fe_model)
        with patch("fastembed.TextEmbedding", mock_fe_cls):
            Embeddings("local/all-MiniLM-L6-v2")
            mock_fe_cls.assert_called_with("sentence-transformers/all-MiniLM-L6-v2")

    def test_fastembed_model_name_passthrough(self):
        mock_fe_model = MagicMock()
        mock_fe_cls = MagicMock(return_value=mock_fe_model)
        with patch("fastembed.TextEmbedding", mock_fe_cls):
            Embeddings("local/BAAI/bge-small-en-v1.5")
            mock_fe_cls.assert_called_with("BAAI/bge-small-en-v1.5")

    def test_fastembed_embed_returns_vectors(self):
        mock_fe_model = MagicMock()
        mock_fe_model.embed.return_value = iter([MagicMock(tolist=lambda: [0.1, 0.2, 0.3])])
        with patch("fastembed.TextEmbedding", return_value=mock_fe_model):
            emb = Embeddings("local/test-model")
            vectors = asyncio.run(emb.aembed(["hello"]))
            assert vectors == [[0.1, 0.2, 0.3]]

    def test_fastembed_embed_multiple_texts(self):
        mock_fe_model = MagicMock()
        mock_fe_model.embed.return_value = iter([
            MagicMock(tolist=lambda: [0.1, 0.2]),
            MagicMock(tolist=lambda: [0.3, 0.4]),
        ])
        with patch("fastembed.TextEmbedding", return_value=mock_fe_model):
            emb = Embeddings("local/test-model")
            vectors = asyncio.run(emb.aembed(["hello", "world"]))
            assert vectors == [[0.1, 0.2], [0.3, 0.4]]

    def test_fastembed_embed_caches_result(self):
        mock_fe_model = MagicMock()
        mock_fe_model.embed.return_value = iter([MagicMock(tolist=lambda: [0.5, 0.6])])
        with patch("fastembed.TextEmbedding", return_value=mock_fe_model):
            emb = Embeddings("local/test-model")
            asyncio.run(emb.aembed(["hello"]))
            assert emb.cache_stats()["entries"] == 1

    def test_fastembed_embed_returns_from_cache(self):
        mock_fe_model = MagicMock()
        mock_fe_model.embed.return_value = iter([MagicMock(tolist=lambda: [0.5, 0.6])])
        with patch("fastembed.TextEmbedding", return_value=mock_fe_model):
            emb = Embeddings("local/test-model")
            asyncio.run(emb.aembed(["hello"]))
            asyncio.run(emb.aembed(["hello"]))
            assert mock_fe_model.embed.call_count == 1

    def test_no_backend_available_raises(self):
        with patch.dict("sys.modules", {"fastembed": None}):
            with pytest.raises(ImportError, match="No local embedding backend found"):
                Embeddings("local/all-MiniLM-L6-v2")


class TestEmbeddingsSync:
    def test_embed_returns_vectors(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key")
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1, 0.2], "index": 0}]
        with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response):
            vectors = emb.embed(["hello"])
            assert vectors == [[0.1, 0.2]]

    def test_embed_single_returns_vector(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key")
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.3, 0.4], "index": 0}]
        with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response):
            vec = emb.embed_single("test")
            assert vec == [0.3, 0.4]


class TestEmbeddingsAsync:
    def test_aembed_returns_vectors(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key")
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1, 0.2], "index": 0}]
        with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response):
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
        assert asyncio.run(emb.aembed([])) == []

    def test_aembed_single(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key")
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.3, 0.4], "index": 0}]
        with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response):
            vec = asyncio.run(emb.aembed_single("test"))
            assert vec == [0.3, 0.4]

    def test_aembed_no_api_key_raises(self):
        emb = Embeddings("mistral/mistral-embed")
        emb.api_key = None
        with pytest.raises(ValueError, match="No API key for model"):
            asyncio.run(emb.aembed(["hello"]))

    def test_aembed_str_input_wrapped_in_list(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key")
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.5, 0.6], "index": 0}]
        with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response):
            vectors = asyncio.run(emb.aembed("hello"))
            assert vectors == [[0.5, 0.6]]

    def test_aembed_multiple_texts(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key")
        mock_response = MagicMock()
        mock_response.data = [
            {"embedding": [0.1, 0.2], "index": 0},
            {"embedding": [0.3, 0.4], "index": 1},
        ]
        with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response):
            vectors = asyncio.run(emb.aembed(["hello", "world"]))
            assert vectors == [[0.1, 0.2], [0.3, 0.4]]

    def test_aembed_out_of_order_response_sorted(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key")
        mock_response = MagicMock()
        mock_response.data = [
            {"embedding": [0.3, 0.4], "index": 1},
            {"embedding": [0.1, 0.2], "index": 0},
        ]
        with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response):
            vectors = asyncio.run(emb.aembed(["hello", "world"]))
            assert vectors[0] == [0.1, 0.2]
            assert vectors[1] == [0.3, 0.4]

    def test_aembed_partial_cache_hit(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key")
        first_response = MagicMock()
        first_response.data = [{"embedding": [0.1, 0.2], "index": 0}]
        second_response = MagicMock()
        second_response.data = [{"embedding": [0.3, 0.4], "index": 0}]
        with patch("litellm.aembedding", new_callable=AsyncMock, side_effect=[first_response, second_response]):
            asyncio.run(emb.aembed(["hello"]))
            vectors = asyncio.run(emb.aembed(["hello", "world"]))
            assert vectors == [[0.1, 0.2], [0.3, 0.4]]

    def test_aembed_sends_api_base(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key", api_base="https://custom.api/v1")
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1], "index": 0}]
        with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response) as mock_aemb:
            asyncio.run(emb.aembed(["hello"]))
            assert mock_aemb.call_args.kwargs["api_base"] == "https://custom.api/v1"

    def test_aembed_sends_extra_kwargs(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key", timeout=30)
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1], "index": 0}]
        with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response) as mock_aemb:
            asyncio.run(emb.aembed(["hello"]))
            assert mock_aemb.call_args.kwargs["timeout"] == 30

    def test_aembed_cache_eviction(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key", cache_size=2)
        for i in range(4):
            mock_response = MagicMock()
            mock_response.data = [{"embedding": [float(i)], "index": 0}]
            with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response):
                asyncio.run(emb.aembed([f"text-{i}"]))
        assert emb.cache_stats()["entries"] == 2

    def test_aembed_cache_disabled(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key", cache_enabled=False)
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1, 0.2], "index": 0}]
        with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response):
            asyncio.run(emb.aembed(["hello"]))
        assert emb.cache_stats()["entries"] == 0


class TestEmbeddingsCosineSimilarity:
    def test_identical_vectors(self):
        assert Embeddings.cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert Embeddings.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert Embeddings.cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert Embeddings.cosine_similarity([0, 0], [1, 1]) == 0.0

    def test_both_zero_vectors(self):
        assert Embeddings.cosine_similarity([0, 0], [0, 0]) == 0.0

    def test_known_angle(self):
        assert Embeddings.cosine_similarity([1, 1], [1, 0]) == pytest.approx(0.7071, abs=1e-4)


class TestEmbeddingsCache:
    def test_clear_cache_returns_count(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key")
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1], "index": 0}]
        with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response):
            asyncio.run(emb.aembed(["hello"]))
        assert emb.clear_cache() == 1
        assert emb.cache_stats()["entries"] == 0

    def test_clear_empty_cache(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key")
        assert emb.clear_cache() == 0

    def test_cache_disabled_stats(self):
        emb = Embeddings("mistral/mistral-embed", api_key="key", cache_enabled=False)
        stats = emb.cache_stats()
        assert stats["enabled"] is False
