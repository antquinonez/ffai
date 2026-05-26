"""Composable RAG pipeline with builder pattern and async support."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .embeddings import FFEmbeddings
from .indexing import BM25Index, HierarchicalIndex
from .search import (
    HybridSearch,
    QueryExpander,
    fuse_search_results,
    get_reranker,
)
from .splitters import (
    ChunkerBase,
    HierarchicalTextChunk,
    TextChunk,
    get_chunker,
)

logger = logging.getLogger(__name__)


def normalize_scores(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for result in results:
        if result.get("distance") is not None:
            result["score"] = 1.0 - result["distance"]
        elif result.get("rrf_score") is not None:
            result["score"] = result["rrf_score"]
        elif result.get("score") is None:
            result["score"] = 0.0
    return results


def format_results_for_prompt(
    results: list[dict[str, Any]],
    max_chars: int | None = None,
    include_parent_context: bool = True,
) -> str:
    if not results:
        return ""

    formatted_chunks: list[str] = []
    total_chars = 0

    for i, result in enumerate(results, start=1):
        content = result.get("content", "")
        source = result.get("metadata", {}).get("reference_name", "unknown")
        score = result.get("score", 0.0)

        parent_content = result.get("parent_content")
        if parent_content and include_parent_context:
            context_note = (
                f"\n[Parent context: {parent_content[:200]}...]"
                if len(parent_content) > 200
                else f"\n[Parent context: {parent_content}]"
            )
        else:
            context_note = ""

        chunk_text = (
            f"[{i}] (source: {source}, relevance: {score:.2f})\n{content}{context_note}\n"
        )

        if max_chars and total_chars + len(chunk_text) > max_chars:
            break

        formatted_chunks.append(chunk_text)
        total_chars += len(chunk_text)

    return "".join(formatted_chunks)


SUMMARY_PROMPT = """Summarize the following document in 2-3 sentences.
Focus on the main topic, key concepts, and purpose.

Document:
{content}

Summary:"""


@dataclass
class _PipelineConfig:
    chunk_strategy: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_kwargs: dict[str, Any] = field(default_factory=dict)
    vector_store_enabled: bool = False
    collection_name: str = "ffai_kb"
    persist_dir: str = "./chroma_db"
    bm25_enabled: bool = False
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    hybrid_alpha: float = 0.6
    hierarchical_enabled: bool = False
    parent_chunk_size: int = 1500
    include_parent_context: bool = True
    reranker_type: str | None = None
    reranker_kwargs: dict[str, Any] = field(default_factory=dict)
    query_expansion_enabled: bool = False
    query_expansion_n: int = 3
    summaries_enabled: bool = False
    summary_boost: float = 1.5
    contextual_headers: bool = True


class RAGPipeline:
    def __init__(self, embeddings: FFEmbeddings) -> None:
        self._embeddings = embeddings
        self._config = _PipelineConfig()
        self._built = False
        self._chunker: ChunkerBase | None = None
        self._vector_store: Any = None
        self._bm25_index: BM25Index | None = None
        self._hierarchical_index: HierarchicalIndex | None = None
        self._hybrid_search: HybridSearch | None = None
        self._reranker: Any = None
        self._query_expander: QueryExpander | None = None
        self._llm_fn: Callable[[str], str] | None = None

    def chunk(
        self,
        strategy: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs: Any,
    ) -> RAGPipeline:
        self._config.chunk_strategy = strategy
        self._config.chunk_size = chunk_size
        self._config.chunk_overlap = chunk_overlap
        self._config.chunk_kwargs = kwargs
        return self

    def with_vector_store(
        self,
        collection_name: str = "ffai_kb",
        persist_dir: str = "./chroma_db",
    ) -> RAGPipeline:
        self._config.vector_store_enabled = True
        self._config.collection_name = collection_name
        self._config.persist_dir = persist_dir
        return self

    def with_bm25(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        alpha: float = 0.6,
    ) -> RAGPipeline:
        self._config.bm25_enabled = True
        self._config.bm25_k1 = k1
        self._config.bm25_b = b
        self._config.hybrid_alpha = alpha
        return self

    def with_hierarchical(
        self,
        parent_chunk_size: int = 1500,
        include_parent_context: bool = True,
    ) -> RAGPipeline:
        self._config.hierarchical_enabled = True
        self._config.parent_chunk_size = parent_chunk_size
        self._config.include_parent_context = include_parent_context
        return self

    def with_reranker(
        self,
        reranker_type: str = "diversity",
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        **kwargs: Any,
    ) -> RAGPipeline:
        self._config.reranker_type = reranker_type
        self._config.reranker_kwargs = {"model_name": model_name, **kwargs}
        return self

    def with_query_expansion(
        self,
        n_variations: int = 3,
        llm_fn: Callable[[str], str] | None = None,
    ) -> RAGPipeline:
        self._config.query_expansion_enabled = True
        self._config.query_expansion_n = n_variations
        if llm_fn:
            self._llm_fn = llm_fn
        return self

    def with_summaries(
        self,
        boost: float = 1.5,
        llm_fn: Callable[[str], str] | None = None,
    ) -> RAGPipeline:
        self._config.summaries_enabled = True
        self._config.summary_boost = boost
        if llm_fn:
            self._llm_fn = llm_fn
        return self

    def with_contextual_headers(self, enabled: bool = True) -> RAGPipeline:
        self._config.contextual_headers = enabled
        return self

    def build(self) -> RAGPipeline:
        if self._built:
            return self

        self._chunker = get_chunker(
            strategy=self._config.chunk_strategy,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
            **self._config.chunk_kwargs,
        )

        if self._config.vector_store_enabled:
            from .vector_store import FFVectorStore

            self._vector_store = FFVectorStore(
                collection_name=self._config.collection_name,
                persist_dir=self._config.persist_dir,
                embedding_model=self._embeddings,
            )

        if self._config.bm25_enabled and self._vector_store is not None:
            self._bm25_index = BM25Index(
                k1=self._config.bm25_k1, b=self._config.bm25_b,
            )
            docs = self._vector_store.get_all_documents()
            for doc in docs:
                self._bm25_index.add_document(
                    doc_id=doc["id"],
                    content=doc["content"],
                    metadata=doc.get("metadata"),
                )
            self._hybrid_search = HybridSearch(
                vector_search_fn=self._vector_store.search,
                bm25_search_fn=self._bm25_index.search,
                alpha=self._config.hybrid_alpha,
            )

        if self._config.hierarchical_enabled:
            self._hierarchical_index = HierarchicalIndex(
                include_parent_context=self._config.include_parent_context,
            )

        if self._config.reranker_type:
            self._reranker = get_reranker(
                self._config.reranker_type, **self._config.reranker_kwargs,
            )

        if self._config.query_expansion_enabled:
            self._query_expander = QueryExpander(
                n_variations=self._config.query_expansion_n,
                include_original=True,
            )
            if self._llm_fn:
                self._query_expander.set_llm_function(self._llm_fn)

        self._built = True
        return self

    def set_llm_fn(self, fn: Callable[[str], str]) -> None:
        self._llm_fn = fn
        if self._query_expander:
            self._query_expander.set_llm_function(fn)

    def index(
        self,
        content: str,
        reference_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        checksum: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        chunking_strategy: str | None = None,
    ) -> int:
        if not content or not content.strip():
            return 0

        meta = metadata.copy() if metadata else {}
        if reference_name:
            meta["reference_name"] = reference_name

        effective_strategy = chunking_strategy or self._config.chunk_strategy
        effective_size = chunk_size or self._config.chunk_size
        effective_overlap = chunk_overlap or self._config.chunk_overlap

        if chunk_size or chunk_overlap or chunking_strategy:
            chunker = get_chunker(
                strategy=effective_strategy,
                chunk_size=effective_size,
                chunk_overlap=effective_overlap,
            )
        else:
            chunker = self._chunker

        assert chunker is not None
        chunks = chunker.chunk(content, metadata=meta)

        if not chunks:
            return 0

        if (
            self._config.hierarchical_enabled
            and isinstance(chunks[0], HierarchicalTextChunk)
        ):
            return self._add_hierarchical_chunks(
                chunks,  # type: ignore[arg-type]
                reference_name=reference_name,
                checksum=checksum,
                chunking_strategy=effective_strategy,
            )

        return self._add_regular_chunks(
            chunks,
            reference_name=reference_name,
            checksum=checksum,
            chunking_strategy=effective_strategy,
        )

    async def async_index(
        self,
        content: str,
        reference_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        checksum: str | None = None,
    ) -> int:
        return self.index(
            content,
            reference_name=reference_name,
            metadata=metadata,
            checksum=checksum,
        )

    def _add_regular_chunks(
        self,
        chunks: list[TextChunk],
        reference_name: str | None = None,
        checksum: str | None = None,
        chunking_strategy: str | None = None,
    ) -> int:
        if self._vector_store is None:
            return len(chunks)

        texts_for_embedding: list[str] | None = None

        if self._config.contextual_headers:
            from .indexing.contextual import ContextualEmbeddings

            contextual = ContextualEmbeddings()
            chunks_for_context = [
                {"content": c.content, "metadata": c.metadata} for c in chunks
            ]
            texts_for_embedding = contextual.prepare_chunks_batch(
                chunks_for_context,
                document_title=reference_name,
            )

        count = self._vector_store.add_chunks(
            chunks,
            chunking_strategy=chunking_strategy or self._config.chunk_strategy,
            document_checksum=checksum or "",
            texts_for_embedding=texts_for_embedding,
        )

        if self._bm25_index is not None:
            for chunk in chunks:
                chunk_id = (
                    f"{(chunk.metadata or {}).get('reference_name', 'doc')}_{chunk.chunk_index}"
                )
                self._bm25_index.add_document(
                    doc_id=chunk_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                )

        return count

    def _add_hierarchical_chunks(
        self,
        chunks: list[HierarchicalTextChunk],
        reference_name: str | None = None,
        checksum: str | None = None,
        chunking_strategy: str | None = None,
    ) -> int:
        if self._vector_store is None:
            return len(chunks)

        if self._hierarchical_index is None:
            self._hierarchical_index = HierarchicalIndex(
                include_parent_context=self._config.include_parent_context,
            )

        child_chunks = [c for c in chunks if c.hierarchy_level > 0]
        if not child_chunks:
            child_chunks = chunks  # type: ignore[assignment]

        texts = [c.content for c in child_chunks]
        embeddings = self._embeddings.embed(texts)

        ids: list[str] = []
        for i, chunk in enumerate(child_chunks):
            chunk_id = (
                f"{(chunk.metadata or {}).get('reference_name', 'doc')}_{chunk.chunk_index}_{i}"
            )
            ids.append(chunk_id)

            self._hierarchical_index.add_chunk(
                chunk_id=chunk_id,
                content=chunk.content,
                embedding=embeddings[i] if i < len(embeddings) else None,
                parent_id=chunk.parent_id,
                hierarchy_level=chunk.hierarchy_level,
                metadata=chunk.metadata,
            )

        count = self._vector_store.add_chunks(
            child_chunks,  # type: ignore[arg-type]
            ids=ids,
            chunking_strategy=chunking_strategy or self._config.chunk_strategy,
            document_checksum=checksum or "",
        )

        if self._bm25_index is not None:
            for chunk_id, chunk in zip(ids, child_chunks):
                self._bm25_index.add_document(
                    doc_id=chunk_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                )

        return count

    def _index_document(
        self,
        content: str,
        reference_name: str,
        common_name: str,
        checksum: str,
        force: bool = False,
        tags: list[str] | None = None,
        chunking_strategy: str | None = None,
    ) -> int:
        strategy = chunking_strategy or self._config.chunk_strategy

        if (
            not force
            and self._vector_store is not None
            and not self._vector_store.needs_reindex(
                reference_name=reference_name,
                checksum=checksum,
                chunking_strategy=strategy,
            )
        ):
            logger.debug(f"Document {reference_name} already indexed, skipping")
            return 0

        logger.info(f"Indexing document {reference_name} with strategy {strategy}")

        if self._vector_store is not None:
            self._vector_store.delete_by_reference_and_strategy(
                reference_name=reference_name,
                chunking_strategy=strategy,
            )

        metadata: dict[str, Any] = {"common_name": common_name}
        if tags:
            metadata["tags"] = ",".join(tags)

        chunks_added = self.index(
            content=content,
            reference_name=reference_name,
            metadata=metadata,
            checksum=checksum,
            chunking_strategy=strategy,
        )

        if (
            self._config.summaries_enabled
            and self._llm_fn
            and chunks_added > 0
        ):
            self._generate_and_store_summary(
                content, reference_name, common_name, checksum,
            )

        logger.info(f"Indexed {chunks_added} chunks for {reference_name}")
        return chunks_added

    def _generate_and_store_summary(
        self,
        content: str,
        reference_name: str,
        common_name: str | None = None,
        checksum: str | None = None,
    ) -> bool:
        if not self._llm_fn:
            return False
        if self._vector_store is None:
            return False

        try:
            content_for_summary = content[:3000] if len(content) > 3000 else content
            prompt = SUMMARY_PROMPT.format(content=content_for_summary)
            summary = self._llm_fn(prompt)

            if not summary or not summary.strip():
                return False

            summary_chunk = TextChunk(
                content=f"[DOCUMENT SUMMARY]\n{summary.strip()}",
                chunk_index=-1,
                start_char=0,
                end_char=len(content),
                metadata={
                    "reference_name": reference_name,
                    "common_name": common_name or reference_name,
                    "chunk_type": "summary",
                },
            )

            self._vector_store.add_chunks(
                [summary_chunk],
                chunking_strategy=f"{self._config.chunk_strategy}_summary",
                document_checksum=checksum or "",
            )

            logger.info(f"Generated summary for {reference_name}")
            return True

        except Exception as e:
            logger.warning(f"Failed to generate summary for {reference_name}: {e}")
            return False

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        query_expansion: bool | None = None,
        rerank: bool | None = None,
    ) -> list[dict[str, Any]]:
        if self._vector_store is None:
            return []

        use_expansion = (
            query_expansion
            if query_expansion is not None
            else self._config.query_expansion_enabled
        )
        use_rerank = (
            rerank if rerank is not None else self._config.reranker_type is not None
        )

        if use_expansion:
            self._ensure_query_expander()
            if self._query_expander and self._query_expander.llm_generate_fn:
                return self._search_with_expansion(query, n_results, where, use_rerank)

        return self._search_single(query, n_results, where, use_rerank)

    async def async_search(
        self,
        query: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self._vector_store is None:
            return []

        results = await self._vector_store.async_search(
            query, n_results=n_results, where=where,
        )

        results = normalize_scores(results)

        if self._config.summaries_enabled:
            for result in results:
                if result.get("metadata", {}).get("chunk_type") == "summary":
                    result["score"] = result.get("score", 1.0) * self._config.summary_boost
                    result["is_summary"] = True
            results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        if (
            self._hierarchical_index is not None
            and self._config.include_parent_context
        ):
            results = self._hierarchical_index.enhance_results_with_context(results)

        return results[:n_results]

    def _search_single(
        self,
        query: str,
        n_results: int,
        where: dict[str, Any] | None = None,
        rerank: bool = True,
    ) -> list[dict[str, Any]]:
        fetch_count = (
            n_results * 2 if self._config.summaries_enabled else n_results
        )

        if self._hybrid_search is not None and self._config.bm25_enabled:
            results = self._hybrid_search.search(
                query, n_results=fetch_count, mode="hybrid",
            )
        else:
            results = self._vector_store.search(
                query, n_results=fetch_count, where=where,
            )

        results = normalize_scores(results)

        if self._config.summaries_enabled:
            for result in results:
                if result.get("metadata", {}).get("chunk_type") == "summary":
                    result["score"] = result.get("score", 1.0) * self._config.summary_boost
                    result["is_summary"] = True
            results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        if (
            self._hierarchical_index is not None
            and self._config.include_parent_context
        ):
            results = self._hierarchical_index.enhance_results_with_context(results)

        if rerank:
            self._ensure_reranker()
            if self._reranker is not None and results:
                results = self._reranker.rerank(query, results, n_results=n_results)
                return results

        return results[:n_results]

    def _search_with_expansion(
        self,
        query: str,
        n_results: int,
        where: dict[str, Any] | None = None,
        rerank: bool = True,
    ) -> list[dict[str, Any]]:
        queries = (
            self._query_expander.expand(query) if self._query_expander else [query]
        )
        logger.info(f"Query expansion: {len(queries)} variations generated")

        all_result_lists: list[list[dict[str, Any]]] = []
        for q in queries:
            results = self._vector_store.search(
                q, n_results=n_results * 2, where=where,
            )
            all_result_lists.append(results)

        fused = fuse_search_results(all_result_lists, n_results=n_results * 2)

        fused = normalize_scores(fused)

        if (
            self._hierarchical_index is not None
            and self._config.include_parent_context
        ):
            fused = self._hierarchical_index.enhance_results_with_context(fused)

        if rerank:
            self._ensure_reranker()
            if self._reranker is not None and fused:
                fused = self._reranker.rerank(query, fused, n_results=n_results)
                return fused[:n_results]

        return fused[:n_results]

    def _ensure_query_expander(self) -> None:
        if self._query_expander is None:
            self._query_expander = QueryExpander(
                n_variations=self._config.query_expansion_n,
                include_original=True,
            )
            if self._llm_fn:
                self._query_expander.set_llm_function(self._llm_fn)

    def _ensure_reranker(self) -> None:
        if self._reranker is None and self._config.reranker_type:
            self._reranker = get_reranker(
                self._config.reranker_type, **self._config.reranker_kwargs,
            )

    def format(
        self,
        results: list[dict[str, Any]],
        max_chars: int | None = None,
    ) -> str:
        return format_results_for_prompt(
            results,
            max_chars=max_chars,
            include_parent_context=self._config.include_parent_context,
        )

    def search_and_format(
        self,
        query: str,
        n_results: int = 5,
        max_chars: int | None = None,
    ) -> str:
        results = self.search(query, n_results=n_results)
        return self.format(results, max_chars=max_chars)

    def delete(self, reference_name: str) -> None:
        if self._vector_store is not None:
            self._vector_store.delete_by_reference(reference_name)

        if self._bm25_index is not None:
            self._bm25_index.delete_by_metadata("reference_name", reference_name)

        if self._hierarchical_index is not None:
            self._hierarchical_index.delete_by_reference(reference_name)

    def count(self) -> int:
        if self._vector_store is None:
            return 0
        return self._vector_store.count()

    def clear(self) -> None:
        if self._vector_store is not None:
            self._vector_store.clear()

        if self._bm25_index is not None:
            self._bm25_index.clear()

        if self._hierarchical_index is not None:
            self._hierarchical_index.clear()

    def list_documents(self) -> list[str]:
        if self._vector_store is None:
            return []
        return self._vector_store.list_documents()

    def get_stats(self) -> dict[str, Any]:
        if self._vector_store is None:
            return {"count": 0}

        stats = self._vector_store.get_stats()

        if self._bm25_index is not None:
            stats["bm25_docs"] = self._bm25_index.count()
        if self._hierarchical_index is not None:
            stats["hierarchical"] = self._hierarchical_index.get_stats()

        return stats

    def get_indexed_documents(
        self, chunking_strategy: str | None = None,
    ) -> list[dict[str, Any]]:
        if self._vector_store is None:
            return []
        return self._vector_store.get_indexed_documents(
            chunking_strategy=chunking_strategy,
        )

    def needs_reindex(
        self,
        reference_name: str,
        checksum: str,
        chunking_strategy: str | None = None,
    ) -> bool:
        if self._vector_store is None:
            return True
        effective = chunking_strategy or self._config.chunk_strategy
        return self._vector_store.needs_reindex(reference_name, checksum, effective)

    @property
    def chunker(self) -> ChunkerBase:
        assert self._chunker is not None
        return self._chunker

    @property
    def embeddings(self) -> FFEmbeddings:
        return self._embeddings

    @property
    def vector_store(self) -> Any:
        return self._vector_store
