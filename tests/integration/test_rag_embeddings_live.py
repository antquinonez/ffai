import os

import pytest

from ffai.rag.embed import Embeddings

pytestmark = pytest.mark.integration


def _get_mistral_api_key():
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        pytest.skip("MISTRAL_API_KEY not set")
    return key


class TestEmbeddingsLiveAPI:
    def test_aembed_returns_vectors(self):
        emb = Embeddings("mistral/mistral-embed", api_key=_get_mistral_api_key())
        import asyncio
        vectors = asyncio.run(emb.aembed(["hello world"]))
        assert len(vectors) == 1
        assert len(vectors[0]) > 0
        assert all(isinstance(v, float) for v in vectors[0])

    def test_aembed_single_returns_vector(self):
        emb = Embeddings("mistral/mistral-embed", api_key=_get_mistral_api_key())
        import asyncio
        vec = asyncio.run(emb.aembed_single("test query"))
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)

    def test_aembed_multiple_texts(self):
        emb = Embeddings("mistral/mistral-embed", api_key=_get_mistral_api_key())
        import asyncio
        vectors = asyncio.run(emb.aembed(["first text", "second text", "third text"]))
        assert len(vectors) == 3
        assert all(len(v) > 0 for v in vectors)

    def test_embedding_dimension_consistent(self):
        emb = Embeddings("mistral/mistral-embed", api_key=_get_mistral_api_key())
        import asyncio
        v1 = asyncio.run(emb.aembed_single("text one"))
        v2 = asyncio.run(emb.aembed_single("text two"))
        assert len(v1) == len(v2)

    def test_similar_texts_high_cosine_similarity(self):
        emb = Embeddings("mistral/mistral-embed", api_key=_get_mistral_api_key())
        import asyncio
        v1, v2 = asyncio.run(emb.aembed(["a cat sat on the mat", "a kitten sat on the rug"]))
        sim = Embeddings.cosine_similarity(v1, v2)
        assert sim > 0.7

    def test_dissimilar_texts_lower_cosine_similarity(self):
        emb = Embeddings("mistral/mistral-embed", api_key=_get_mistral_api_key())
        import asyncio
        v1, v2 = asyncio.run(emb.aembed([
            "machine learning neural networks",
            "cooking recipe for chocolate cake",
        ]))
        sim_similar = Embeddings.cosine_similarity(v1, v2)

        v3, v4 = asyncio.run(emb.aembed([
            "machine learning neural networks",
            "deep learning algorithms",
        ]))
        sim_related = Embeddings.cosine_similarity(v3, v4)

        assert sim_related > sim_similar

    def test_cache_prevents_duplicate_api_calls(self):
        emb = Embeddings("mistral/mistral-embed", api_key=_get_mistral_api_key(), cache_enabled=True)
        import asyncio
        asyncio.run(emb.aembed(["unique text for cache test"]))
        assert emb.cache_stats()["entries"] == 1

        v1 = asyncio.run(emb.aembed_single("unique text for cache test"))
        v2 = asyncio.run(emb.aembed_single("unique text for cache test"))
        assert v1 == v2
        assert emb.cache_stats()["entries"] == 1

    def test_no_api_key_raises_value_error(self):
        emb = Embeddings("mistral/mistral-embed", api_key=None)
        emb.api_key = None
        import asyncio
        with pytest.raises(ValueError, match="No API key"):
            asyncio.run(emb.aembed(["test"]))
