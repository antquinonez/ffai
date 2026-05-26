"""High-level RAG client wrapping RAGPipeline for backward compatibility."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .embeddings import FFEmbeddings
from .pipeline import RAGPipeline
from .splitters import ChunkerBase

if TYPE_CHECKING:
    from .vector_store import FFVectorStore

try:
    from .vector_store import FFVectorStore as _FFVectorStore
except ImportError:
    _FFVectorStore = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class RAGClient:
    """Thin facade over RAGPipeline for backward compatibility.

    For new code, prefer using RAGPipeline directly.

    Example:
        >>> client = RAGClient(embedding_model="mistral/mistral-embed")
        >>> client.add_document("Long document...", reference_name="doc1")
        >>> results = client.search("What is the document about?")
        >>> print(client.format_results_for_prompt(results))

    """

    def __init__(
        self,
        collection_name: str = "ffai_kb",
        persist_dir: str = "./chroma_db",
        embedding_model: str | FFEmbeddings | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        n_results_default: int = 5,
        chunking_strategy: str = "recursive",
        search_mode: str = "vector",
        hybrid_alpha: float = 0.6,
        rerank_enabled: bool = False,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        hierarchical_enabled: bool = False,
        parent_context: bool = True,
        parent_chunk_size: int = 1500,
        contextual_headers: bool = True,
        query_expansion_enabled: bool = False,
        query_expansion_variations: int = 3,
        generate_summaries: bool = False,
        summary_boost: float = 1.5,
    ) -> None:
        self.n_results_default = n_results_default
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.search_mode = search_mode
        self.hierarchical_enabled = hierarchical_enabled

        emb = (
            embedding_model
            if isinstance(embedding_model, FFEmbeddings)
            else FFEmbeddings(model=embedding_model or "mistral/mistral-embed")
        )

        builder = (
            RAGPipeline(emb)
            .chunk(
                chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            .with_vector_store(
                collection_name=collection_name,
                persist_dir=persist_dir,
            )
        )

        if search_mode == "hybrid":
            builder = builder.with_bm25(alpha=hybrid_alpha)
        elif search_mode == "bm25":
            builder = builder.with_bm25(alpha=0.0)

        if hierarchical_enabled:
            builder = builder.with_hierarchical(
                parent_chunk_size=parent_chunk_size,
                include_parent_context=parent_context,
            )

        if rerank_enabled:
            builder = builder.with_reranker(
                reranker_type="cross_encoder",
                model_name=rerank_model,
            )

        if query_expansion_enabled:
            builder = builder.with_query_expansion(
                n_variations=query_expansion_variations,
            )

        if generate_summaries:
            builder = builder.with_summaries(boost=summary_boost)

        if contextual_headers:
            builder = builder.with_contextual_headers()

        self._pipeline = builder.build()

    def add_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        reference_name: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        checksum: str | None = None,
        chunking_strategy: str | None = None,
    ) -> int:
        return self._pipeline.index(
            content,
            reference_name=reference_name,
            metadata=metadata,
            checksum=checksum,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=chunking_strategy,
        )

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        text_key: str = "content",
    ) -> int:
        total = 0
        for doc in documents:
            content = doc.get(text_key, "")
            metadata = {k: v for k, v in doc.items() if k != text_key}
            reference_name = metadata.get("reference_name")
            total += self.add_document(
                content, metadata=metadata, reference_name=reference_name,
            )
        return total

    def index_document(
        self,
        content: str,
        reference_name: str,
        common_name: str,
        checksum: str,
        force: bool = False,
        tags: list[str] | None = None,
        chunking_strategy: str | None = None,
    ) -> int:
        return self._pipeline._index_document(
            content,
            reference_name,
            common_name,
            checksum,
            force=force,
            tags=tags,
            chunking_strategy=chunking_strategy,
        )

    def search(
        self,
        query: str,
        n_results: int | None = None,
        where: dict[str, Any] | None = None,
        query_expansion: bool | None = None,
        rerank: bool | None = None,
    ) -> list[dict[str, Any]]:
        n = n_results or self.n_results_default
        return self._pipeline.search(
            query,
            n_results=n,
            where=where,
            query_expansion=query_expansion,
            rerank=rerank,
        )

    async def async_search(
        self,
        query: str,
        n_results: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        n = n_results or self.n_results_default
        return await self._pipeline.async_search(
            query, n_results=n, where=where,
        )

    def format_results_for_prompt(
        self,
        results: list[dict[str, Any]],
        max_chars: int | None = None,
    ) -> str:
        return self._pipeline.format(results, max_chars=max_chars)

    def search_and_format(
        self,
        query: str,
        n_results: int | None = None,
        max_chars: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> str:
        results = self.search(query, n_results=n_results, where=where)
        return self.format_results_for_prompt(results, max_chars=max_chars)

    def delete_by_reference(self, reference_name: str) -> None:
        self._pipeline.delete(reference_name)

    def list_documents(self) -> list[str]:
        return self._pipeline.list_documents()

    def get_indexed_documents(
        self,
        chunking_strategy: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._pipeline.get_indexed_documents(chunking_strategy)

    def needs_reindex(
        self,
        reference_name: str,
        checksum: str,
        chunking_strategy: str | None = None,
    ) -> bool:
        effective = chunking_strategy or self.chunking_strategy
        return self._pipeline.needs_reindex(reference_name, checksum, effective)

    def count(self) -> int:
        return self._pipeline.count()

    def clear(self) -> None:
        self._pipeline.clear()

    def get_stats(self) -> dict[str, Any]:
        stats = self._pipeline.get_stats()
        stats["chunk_size"] = self.chunk_size
        stats["chunk_overlap"] = self.chunk_overlap
        stats["chunking_strategy"] = self.chunking_strategy
        stats["search_mode"] = self.search_mode
        stats["hierarchical_enabled"] = self.hierarchical_enabled
        return stats

    def set_llm_generate_fn(self, fn: Callable[[str], str]) -> None:
        self._pipeline.set_llm_fn(fn)

    def set_query_expansion_llm(self, llm_generate_fn: Callable[[str], str]) -> None:
        self._pipeline.set_llm_fn(llm_generate_fn)

    @property
    def pipeline(self) -> RAGPipeline:
        return self._pipeline

    @property
    def embeddings(self) -> FFEmbeddings:
        return self._pipeline.embeddings

    @property
    def vector_store(self) -> FFVectorStore:
        vs = self._pipeline.vector_store
        if vs is None:
            raise AttributeError("No vector store configured")
        return vs

    @property
    def chunker(self) -> ChunkerBase:
        return self._pipeline.chunker


FFRAGClient = RAGClient
