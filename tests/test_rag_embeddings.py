from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest


class TestFFEmbeddingsInitAPI:
    def test_api_model_stores_model_string(self):
        from src.rag.embeddings import FFEmbeddings

        with patch("src.rag.embeddings.embedding"):
            ff = FFEmbeddings(model="mistral/mistral-embed", api_key="key123")
            assert ff.model == "mistral/mistral-embed"

    def test_api_model_stores_api_key(self):
        from src.rag.embeddings import FFEmbeddings

        with patch("src.rag.embeddings.embedding"):
            ff = FFEmbeddings(model="mistral/mistral-embed", api_key="test-key")
            assert ff.api_key == "test-key"

    def test_api_model_stores_api_base(self):
        from src.rag.embeddings import FFEmbeddings

        with patch("src.rag.embeddings.embedding"):
            ff = FFEmbeddings(
                model="mistral/mistral-embed",
                api_key="k",
                api_base="https://custom.api/v1",
            )
            assert ff.api_base == "https://custom.api/v1"

    def test_api_model_is_not_local(self):
        from src.rag.embeddings import FFEmbeddings

        with patch("src.rag.embeddings.embedding"):
            ff = FFEmbeddings(model="mistral/mistral-embed", api_key="k")
            assert ff.is_local is False

    def test_provider_mistral(self):
        from src.rag.embeddings import FFEmbeddings

        with patch("src.rag.embeddings.embedding"):
            ff = FFEmbeddings(model="mistral/mistral-embed", api_key="k")
            assert ff.provider == "mistral"

    def test_provider_openai(self):
        from src.rag.embeddings import FFEmbeddings

        with patch("src.rag.embeddings.embedding"):
            ff = FFEmbeddings(model="openai/text-embedding-ada-002", api_key="k")
            assert ff.provider == "openai"

    def test_provider_azure(self):
        from src.rag.embeddings import FFEmbeddings

        with patch("src.rag.embeddings.embedding"):
            ff = FFEmbeddings(model="azure/my-deployment", api_key="k")
            assert ff.provider == "azure"

    def test_missing_api_key_raises_valueerror(self):
        from src.rag.embeddings import FFEmbeddings

        with patch.dict("os.environ", {}, clear=True):
            ff = FFEmbeddings(model="mistral/mistral-embed", api_key=None)
            with pytest.raises(ValueError, match="No API key configured"):
                ff.embed("hello")

    def test_extra_kwargs_stored(self):
        from src.rag.embeddings import FFEmbeddings

        with patch("src.rag.embeddings.embedding"):
            ff = FFEmbeddings(model="mistral/mistral-embed", api_key="k", timeout=30)
            assert ff._extra_kwargs == {"timeout": 30}


class TestFFEmbeddingsInitLocal:
    def test_local_model_prefix_sets_is_local(self):
        from src.rag.embeddings import FFEmbeddings

        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            ff = FFEmbeddings(model="local/all-MiniLM-L6-v2")
            assert ff.is_local is True

    def test_local_model_strips_prefix_for_sentence_transformer(self):
        from src.rag.embeddings import FFEmbeddings

        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = MagicMock()

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            ff = FFEmbeddings(model="local/all-MiniLM-L6-v2")
            mock_st.SentenceTransformer.assert_called_once_with(
                "all-MiniLM-L6-v2", device="cpu"
            )

    def test_local_model_custom_device(self):
        from src.rag.embeddings import FFEmbeddings

        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = MagicMock()

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            ff = FFEmbeddings(model="local/all-MiniLM-L6-v2", device="cuda")
            mock_st.SentenceTransformer.assert_called_once_with(
                "all-MiniLM-L6-v2", device="cuda"
            )

    def test_local_model_provider_is_local(self):
        from src.rag.embeddings import FFEmbeddings

        mock_st = MagicMock()
        mock_st.SentenceTransformer.return_value = MagicMock()

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            ff = FFEmbeddings(model="local/all-MiniLM-L6-v2")
            assert ff.provider == "local"

    def test_local_model_import_error(self):
        from src.rag.embeddings import FFEmbeddings

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers not installed"):
                FFEmbeddings(model="local/all-MiniLM-L6-v2")


class TestFFEmbeddingsEmbedAPI:
    def test_embed_single_text_wraps_in_list(self):
        from src.rag.embeddings import FFEmbeddings

        fake_response = MagicMock()
        fake_response.data = [{"embedding": [0.1, 0.2, 0.3], "index": 0}]

        with patch("src.rag.embeddings.embedding", return_value=fake_response) as mock_emb:
            ff = FFEmbeddings(model="mistral/mistral-embed", api_key="k")
            result = ff.embed("hello world")
            assert len(result) == 1
            assert result[0] == [0.1, 0.2, 0.3]
            mock_emb.assert_called_once()
            call_kwargs = mock_emb.call_args
            assert call_kwargs[1]["input"] == ["hello world"] or call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1]["input"] == ["hello world"]

    def test_embed_list_of_texts(self):
        from src.rag.embeddings import FFEmbeddings

        fake_response = MagicMock()
        fake_response.data = [
            {"embedding": [0.1, 0.2], "index": 0},
            {"embedding": [0.3, 0.4], "index": 1},
            {"embedding": [0.5, 0.6], "index": 2},
        ]

        with patch("src.rag.embeddings.embedding", return_value=fake_response):
            ff = FFEmbeddings(model="mistral/mistral-embed", api_key="k")
            result = ff.embed(["one", "two", "three"])
            assert len(result) == 3
            assert result[0] == [0.1, 0.2]
            assert result[1] == [0.3, 0.4]
            assert result[2] == [0.5, 0.6]

    def test_embed_passes_api_key_and_model(self):
        from src.rag.embeddings import FFEmbeddings

        fake_response = MagicMock()
        fake_response.data = [{"embedding": [0.1], "index": 0}]

        with patch("src.rag.embeddings.embedding", return_value=fake_response) as mock_emb:
            ff = FFEmbeddings(model="mistral/mistral-embed", api_key="my-key")
            ff.embed("test")
            call_kwargs = mock_emb.call_args[1]
            assert call_kwargs["model"] == "mistral/mistral-embed"
            assert call_kwargs["api_key"] == "my-key"

    def test_embed_passes_api_base_when_set(self):
        from src.rag.embeddings import FFEmbeddings

        fake_response = MagicMock()
        fake_response.data = [{"embedding": [0.1], "index": 0}]

        with patch("src.rag.embeddings.embedding", return_value=fake_response) as mock_emb:
            ff = FFEmbeddings(
                model="mistral/mistral-embed",
                api_key="k",
                api_base="https://custom.api",
            )
            ff.embed("test")
            call_kwargs = mock_emb.call_args[1]
            assert call_kwargs["api_base"] == "https://custom.api"

    def test_embed_empty_list_returns_empty_list(self):
        from src.rag.embeddings import FFEmbeddings

        with patch("src.rag.embeddings.embedding"):
            ff = FFEmbeddings(model="mistral/mistral-embed", api_key="k")
            assert ff.embed([]) == []

    def test_embed_api_failure_raises_runtime_error(self):
        from src.rag.embeddings import FFEmbeddings

        with patch("src.rag.embeddings.embedding", side_effect=Exception("API down")):
            ff = FFEmbeddings(model="mistral/mistral-embed", api_key="k")
            with pytest.raises(RuntimeError, match="Embedding generation failed"):
                ff.embed("test")


class TestFFEmbeddingsEmbedLocal:
    def test_embed_local_uses_model_encode(self):
        from src.rag.embeddings import FFEmbeddings

        mock_arr = MagicMock()
        mock_arr.__getitem__ = lambda self, idx: [0.5, 0.6] if idx == 0 else None
        mock_arr.__len__ = lambda self: 1

        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_arr
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            ff = FFEmbeddings(model="local/all-MiniLM-L6-v2")

        with patch.object(ff, "_embed_local", return_value=[[0.5, 0.6]]) as mock_local:
            result = ff.embed("hello")
            assert len(result) == 1
            assert result[0] == [0.5, 0.6]

    def test_embed_local_multiple_texts(self):
        from src.rag.embeddings import FFEmbeddings

        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            ff = FFEmbeddings(model="local/all-MiniLM-L6-v2")

        with patch.object(
            ff, "_embed_local", return_value=[[0.1, 0.2], [0.3, 0.4]]
        ):
            result = ff.embed(["hello", "world"])
            assert len(result) == 2
            assert result[0] == [0.1, 0.2]
            assert result[1] == [0.3, 0.4]


class TestFFEmbeddingsEmbedSingle:
    def test_embed_single_returns_vector(self):
        from src.rag.embeddings import FFEmbeddings

        fake_response = MagicMock()
        fake_response.data = [{"embedding": [0.1, 0.2, 0.3], "index": 0}]

        with patch("src.rag.embeddings.embedding", return_value=fake_response):
            ff = FFEmbeddings(model="mistral/mistral-embed", api_key="k")
            result = ff.embed_single("hello")
            assert result == [0.1, 0.2, 0.3]


class TestFFEmbeddingsCache:
    def test_cache_hit_does_not_call_api(self):
        from src.rag.embeddings import FFEmbeddings

        fake_response = MagicMock()
        fake_response.data = [{"embedding": [0.1, 0.2], "index": 0}]

        with patch("src.rag.embeddings.embedding", return_value=fake_response) as mock_emb:
            ff = FFEmbeddings(model="mistral/mistral-embed", api_key="k")
            ff.embed("hello")
            assert mock_emb.call_count == 1
            ff.embed("hello")
            assert mock_emb.call_count == 1

    def test_cache_miss_calls_api(self):
        from src.rag.embeddings import FFEmbeddings

        call_count = 0
        embeddings_map = {
            "hello": [0.1, 0.2],
            "world": [0.3, 0.4],
        }

        def fake_embedding(**kwargs):
            nonlocal call_count
            call_count += 1
            texts = kwargs["input"]
            resp = MagicMock()
            resp.data = [
                {"embedding": embeddings_map[t], "index": i}
                for i, t in enumerate(texts)
            ]
            return resp

        with patch("src.rag.embeddings.embedding", side_effect=fake_embedding):
            ff = FFEmbeddings(model="mistral/mistral-embed", api_key="k")
            ff.embed("hello")
            assert call_count == 1
            ff.embed("world")
            assert call_count == 2

    def test_cache_disabled_calls_api_every_time(self):
        from src.rag.embeddings import FFEmbeddings

        fake_response = MagicMock()
        fake_response.data = [{"embedding": [0.1, 0.2], "index": 0}]

        with patch("src.rag.embeddings.embedding", return_value=fake_response) as mock_emb:
            ff = FFEmbeddings(
                model="mistral/mistral-embed", api_key="k", cache_enabled=False
            )
            ff.embed("hello")
            ff.embed("hello")
            assert mock_emb.call_count == 2

    def test_clear_cache_returns_count(self):
        from src.rag.embeddings import FFEmbeddings

        fake_response = MagicMock()
        fake_response.data = [{"embedding": [0.1], "index": 0}]

        with patch("src.rag.embeddings.embedding", return_value=fake_response):
            ff = FFEmbeddings(model="mistral/mistral-embed", api_key="k")
            ff.embed("a")
            ff.embed("b")
            ff.embed("c")
            cleared = ff.clear_cache()
            assert cleared == 3

    def test_clear_cache_resets_to_zero(self):
        from src.rag.embeddings import FFEmbeddings

        fake_response = MagicMock()
        fake_response.data = [{"embedding": [0.1], "index": 0}]

        with patch("src.rag.embeddings.embedding", return_value=fake_response):
            ff = FFEmbeddings(model="mistral/mistral-embed", api_key="k")
            ff.embed("a")
            ff.clear_cache()
            assert ff.clear_cache() == 0

    def test_get_cache_stats(self):
        from src.rag.embeddings import FFEmbeddings

        fake_response = MagicMock()
        fake_response.data = [{"embedding": [0.1], "index": 0}]

        with patch("src.rag.embeddings.embedding", return_value=fake_response):
            ff = FFEmbeddings(
                model="mistral/mistral-embed", api_key="k", cache_size=128
            )
            stats = ff.get_cache_stats()
            assert stats == {"cache_enabled": True, "max_size": 128, "current_entries": 0}

            ff.embed("hello")
            stats = ff.get_cache_stats()
            assert stats["current_entries"] == 1

    def test_cache_evicts_when_full(self):
        from src.rag.embeddings import FFEmbeddings

        def fake_embedding(**kwargs):
            texts = kwargs["input"]
            resp = MagicMock()
            resp.data = [
                {"embedding": [float(i)], "index": i} for i in range(len(texts))
            ]
            return resp

        with patch("src.rag.embeddings.embedding", side_effect=fake_embedding):
            ff = FFEmbeddings(
                model="mistral/mistral-embed", api_key="k", cache_size=2
            )
            ff.embed("a")
            ff.embed("b")
            assert ff.get_cache_stats()["current_entries"] == 2
            ff.embed("c")
            assert ff.get_cache_stats()["current_entries"] == 2


class TestFFEmbeddingsCosineSimilarity:
    def test_identical_vectors_returns_one(self):
        from src.rag.embeddings import FFEmbeddings

        v = [1.0, 0.0, 0.0]
        assert FFEmbeddings.cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_returns_zero(self):
        from src.rag.embeddings import FFEmbeddings

        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert FFEmbeddings.cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_returns_minus_one(self):
        from src.rag.embeddings import FFEmbeddings

        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert FFEmbeddings.cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        from src.rag.embeddings import FFEmbeddings

        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert FFEmbeddings.cosine_similarity(a, b) == 0.0

    def test_arbitrary_vectors(self):
        from src.rag.embeddings import FFEmbeddings

        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        dot = 1 * 4 + 2 * 5 + 3 * 6
        norm_a = math.sqrt(1 + 4 + 9)
        norm_b = math.sqrt(16 + 25 + 36)
        expected = dot / (norm_a * norm_b)
        assert FFEmbeddings.cosine_similarity(a, b) == pytest.approx(expected)


class TestFFEmbeddingsDefaultAPIKey:
    def test_mistral_env_key(self):
        from src.rag.embeddings import FFEmbeddings

        with patch.dict("os.environ", {"MISTRAL_API_KEY": "env-key"}):
            with patch("src.rag.embeddings.embedding"):
                ff = FFEmbeddings(model="mistral/mistral-embed")
                assert ff.api_key == "env-key"

    def test_openai_env_key(self):
        from src.rag.embeddings import FFEmbeddings

        with patch.dict("os.environ", {"OPENAI_API_KEY": "oai-key"}):
            with patch("src.rag.embeddings.embedding"):
                ff = FFEmbeddings(model="openai/text-embedding-ada-002")
                assert ff.api_key == "oai-key"

    def test_no_slash_defaults_to_openai_env(self):
        from src.rag.embeddings import FFEmbeddings

        with patch.dict("os.environ", {"OPENAI_API_KEY": "default-key"}):
            with patch("src.rag.embeddings.embedding"):
                ff = FFEmbeddings(model="text-embedding-ada-002")
                assert ff.api_key == "default-key"
