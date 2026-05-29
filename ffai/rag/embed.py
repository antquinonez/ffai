"""Compute text embeddings via local sentence-transformer models or remote API providers."""

from __future__ import annotations

import asyncio
import logging
import math
import os
from collections import OrderedDict
from typing import Any

from ._async import run_sync

logger = logging.getLogger(__name__)


class Embeddings:
    """Compute text embeddings using local or remote models.

    Supports local sentence-transformer models (prefix ``local/``) and
    remote providers via LiteLLM (e.g. ``mistral/mistral-embed``,
    ``openai/text-embedding-3-small``).  Results are optionally cached
    in an LRU-style in-memory cache.

    Args:
        model: Model identifier.  Use ``local/<name>`` for local
            sentence-transformer models, or ``<provider>/<model>`` for
            remote APIs.
        api_key: API key for the remote provider.  If *None*, the key
            is read from a provider-specific environment variable.
        api_base: Optional custom API base URL.
        cache_enabled: Whether to cache embedding results in memory.
        cache_size: Maximum number of entries in the embedding cache.
        device: Torch device for local models (e.g. ``"cpu"``, ``"cuda"``).
        **kwargs: Extra keyword arguments forwarded to the LiteLLM
            embedding call.

    """

    def __init__(
        self,
        model: str = "mistral/mistral-embed",
        api_key: str | None = None,
        api_base: str | None = None,
        cache_enabled: bool = True,
        cache_size: int = 256,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        self.model = model
        self._device = device
        self._extra_kwargs = kwargs
        self._is_local = model.startswith("local/")
        self._local_model: Any = None
        self._local_backend: str | None = None
        self.api_key: str | None = None
        self.api_base: str | None = None

        if self._is_local:
            self._init_local(model)
        else:
            self.api_key = api_key or self._get_default_api_key()
            self.api_base = api_base

        self._cache_enabled = cache_enabled
        self._cache_size = cache_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    _FASTEMBED_MODEL_MAP: dict[str, str] = {
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    }

    def _init_local(self, model: str) -> None:
        model_name = model.replace("local/", "")
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            self._local_model = SentenceTransformer(model_name, device=self._device)
            self._local_backend = "sentence-transformers"
            logger.info(f"Loaded local model (sentence-transformers): {model_name}")
        except ImportError:
            try:
                from fastembed import TextEmbedding  # type: ignore[import-untyped]

                fe_name = self._FASTEMBED_MODEL_MAP.get(model_name, model_name)
                self._local_model = TextEmbedding(fe_name)
                self._local_backend = "fastembed"
                logger.info(f"Loaded local model (fastembed): {fe_name}")
            except ImportError as e:
                raise ImportError(
                    "No local embedding backend found. Install one of:\n"
                    "  pip install sentence-transformers\n"
                    "  pip install fastembed"
                ) from e

    def _get_default_api_key(self) -> str | None:
        provider = self.model.split("/")[0] if "/" in self.model else "openai"
        env_mappings = {
            "mistral": "MISTRAL_API_KEY",
            "openai": "OPENAI_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        env_var = env_mappings.get(provider, f"{provider.upper()}_API_KEY")
        return os.getenv(env_var)

    async def aembed(self, texts: str | list[str]) -> list[list[float]]:
        """Compute embeddings for one or more texts asynchronously.

        Args:
            texts: A single text string or a list of text strings.

        Returns:
            List of embedding vectors, one per input text.

        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []

        if self._is_local:
            return await asyncio.to_thread(self._embed_local, texts)

        return await self._aembed_api(texts)

    async def _aembed_api(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise ValueError(f"No API key for model {self.model}")

        results: list[tuple[int, list[float]]] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            if self._cache_enabled and text in self._cache:
                self._cache.move_to_end(text)
                results.append((i, self._cache[text]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            from litellm import aembedding

            params: dict[str, Any] = {"model": self.model, "input": uncached_texts}
            if self.api_key:
                params["api_key"] = self.api_key
            if self.api_base:
                params["api_base"] = self.api_base
            params.update(self._extra_kwargs)

            response = await aembedding(**params)
            sorted_data = sorted(response.data, key=lambda x: x["index"])
            api_embeddings = [item["embedding"] for item in sorted_data]

            for j, (idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
                emb = api_embeddings[j]
                results.append((idx, emb))
                if self._cache_enabled and text not in self._cache:
                    self._cache[text] = emb
                    self._cache.move_to_end(text)
                    if len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)

        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        if self._local_model is None:
            raise RuntimeError("Local model not initialized")

        results: list[tuple[int, list[float]]] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            if self._cache_enabled and text in self._cache:
                self._cache.move_to_end(text)
                results.append((i, self._cache[text]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            if self._local_backend == "fastembed":
                raw = self._local_model.embed(uncached_texts)
                embeddings_list = [e.tolist() for e in raw]
            else:
                import numpy as np

                embeddings = self._local_model.encode(uncached_texts, convert_to_numpy=True)
                embeddings_list = [row.tolist() for row in np.asarray(embeddings)]
            for j, (idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
                emb = embeddings_list[j]
                results.append((idx, emb))
                if self._cache_enabled:
                    self._cache[text] = emb
                    self._cache.move_to_end(text)
                    if len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)

        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]

    async def aembed_single(self, text: str) -> list[float]:
        """Compute the embedding for a single text asynchronously.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.

        """
        return (await self.aembed(text))[0]

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """Compute embeddings for one or more texts synchronously.

        Args:
            texts: A single text string or a list of text strings.

        Returns:
            List of embedding vectors, one per input text.

        """
        return run_sync(self.aembed(texts))

    def embed_single(self, text: str) -> list[float]:
        """Compute the embedding for a single text synchronously.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.

        """
        return self.embed(text)[0]

    def clear_cache(self) -> int:
        """Clear the embedding cache.

        Returns:
            Number of entries that were in the cache before clearing.

        """
        count = len(self._cache)
        self._cache.clear()
        return count

    def cache_stats(self) -> dict[str, Any]:
        """Return statistics about the embedding cache.

        Returns:
            Dictionary with keys ``enabled``, ``max_size``, and ``entries``.

        """
        return {
            "enabled": self._cache_enabled,
            "max_size": self._cache_size,
            "entries": len(self._cache),
        }

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity score in the range [-1, 1].  Returns 0.0
            if either vector has zero magnitude.

        """
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @property
    def provider(self) -> str:
        if self._is_local:
            return "local"
        return self.model.split("/")[0] if "/" in self.model else "unknown"

    @property
    def is_local(self) -> bool:
        return self._is_local
