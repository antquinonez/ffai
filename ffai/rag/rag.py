"""Orchestrate end-to-end RAG pipelines combining embedding, chunking, search, reranking, and generation."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from ._async import run_sync
from .embed import Embeddings
from .format import format_hits
from .indexing import BM25Index
from .prompts import DEFAULT_RAG_PROMPT
from .search import get_reranker
from .search.hybrid import reciprocal_rank_fusion
from .search.query_expansion import fuse_search_results
from .search.rerankers import RerankerBase
from .splitters import HierarchicalTextChunk, TextChunk, get_chunker
from .store import CHROMADB_AVAILABLE, VectorStore
from .types import GenerationResult, QueryResult, SearchHit

logger = logging.getLogger(__name__)


class RAG:
    """End-to-end retrieval-augmented generation pipeline.

    Combines embedding, chunking, vector/BM25 search, reranking,
    query expansion, and LLM generation into a single interface.

    Args:
        embed: Embedding model instance or model name string
            (e.g. ``"mistral/mistral-embed"``).
        store: Vector store for persistent embeddings. If ``None``,
            only BM25 search is available (when ``bm25_alpha`` is set).
        chunker: Name of the chunking strategy (``"recursive"``,
            ``"character"``, ``"markdown"``, ``"code"``,
            ``"hierarchical"``).
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.
        bm25_alpha: If set, enables BM25 alongside vector search with
            this weight for the vector component (0–1). ``None``
            disables BM25.
        reranker: Reranker strategy name (``"cross_encoder"``,
            ``"diversity"``, ``"noop"``). ``None`` disables reranking.
        query_expander: Callable that takes a query string and returns
            a list of expanded query strings.
        generate_fn: Default generation function for ``query()`` and
            ``aquery()``. Takes a prompt string, returns an answer
            string or ``GenerationResult``.

    """

    def __init__(
        self,
        embed: Embeddings | str = "mistral/mistral-embed",
        store: VectorStore | None = None,
        chunker: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        bm25_alpha: float | None = None,
        reranker: str | None = None,
        query_expander: Callable[[str], list[str]] | None = None,
        generate_fn: Callable[[str], str | GenerationResult] | None = None,
    ) -> None:
        if isinstance(embed, str):
            embed = Embeddings(model=embed)

        self._embed = embed
        self._store = store
        self._chunker_name = chunker
        self._chunker = get_chunker(
            strategy=chunker, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )
        self._bm25: BM25Index | None = None
        self._bm25_alpha: float = 0.6
        self._bm25_rrf_k: int = 60
        self._reranker: RerankerBase | None = None
        self._query_expander = query_expander
        self._generate_fn = generate_fn

        if bm25_alpha is not None:
            self._bm25 = BM25Index()
            self._bm25_alpha = bm25_alpha

        if reranker is not None:
            self._reranker = get_reranker(reranker)

        if self._query_expander is not None and self._generate_fn is not None:
            self._wire_query_expander()

    @classmethod
    def from_config(
        cls,
        *,
        bm25_only: bool = False,
        api_key: str | None = None,
        **overrides: Any,
    ) -> RAG:
        """Create a RAG instance from the project configuration file.

        Reads all settings from ``config/main.yaml`` under the ``rag:``
        key.  The embedding model's API key is resolved in order:
        the ``api_key`` parameter, the provider-specific environment
        variable (e.g. ``MISTRAL_API_KEY``), and finally ``None``
        (which will raise at embed time if no key is found).

        Args:
            bm25_only: If True, skip vector store creation and use BM25
                search only.
            api_key: API key for the embedding model provider.  When
                *None*, the key is read from a provider-specific
                environment variable (e.g. ``MISTRAL_API_KEY``).
            **overrides: Override any constructor parameter from config.

        Returns:
            Configured RAG instance.

        """
        from ..config import get_config

        cfg = get_config().rag

        if "embed" not in overrides:
            embed = Embeddings(model=cfg.embedding_model, api_key=api_key)
            overrides["embed"] = embed

        kwargs: dict[str, Any] = {
            "chunker": cfg.chunker,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "bm25_alpha": cfg.bm25_alpha,
            "reranker": cfg.reranker,
        }
        kwargs.update(overrides)

        if "store" not in kwargs and not bm25_only and CHROMADB_AVAILABLE:
            store = VectorStore(
                collection_name=cfg.collection_name,
                dir=cfg.persist_dir,
            )
            kwargs["store"] = store

        if kwargs.get("bm25_alpha") is None and (bm25_only or not CHROMADB_AVAILABLE):
            kwargs["bm25_alpha"] = 0.6

        return cls(**kwargs)

    def set_generate_fn(self, generate_fn: Callable[[str], str | GenerationResult]) -> None:
        """Set or replace the default generation function.

        Also re-wires the query expander if one is configured.

        Args:
            generate_fn: Sync callable that takes a prompt and returns
                an answer string or ``GenerationResult``.

        """
        self._generate_fn = generate_fn
        if self._query_expander is not None:
            self._wire_query_expander()

    def _wire_query_expander(self) -> None:
        from .search.query_expansion import QueryExpander

        if isinstance(self._query_expander, QueryExpander) and self._generate_fn is not None:
            self._query_expander.set_llm_function(self._generate_fn)

    def _enrich_hierarchical_chunks(
        self,
        chunks: list[Any],
    ) -> list[Any]:
        parent_map: dict[str, str] = {}
        for c in chunks:
            if isinstance(c, HierarchicalTextChunk) and c.hierarchy_level == 0:
                parent_map[c.id] = c.content

        children: list[Any] = []
        for c in chunks:
            if not isinstance(c, HierarchicalTextChunk) or c.hierarchy_level == 0:
                continue
            if c.metadata is None:
                c.metadata = {}
            c.metadata["parent_content"] = parent_map.get(c.parent_id or "", "")
            c.metadata["hierarchy_level"] = c.hierarchy_level
            if c.parent_id:
                c.metadata["parent_id"] = c.parent_id
            children.append(c)

        return children if children else chunks

    def index(
        self,
        text: str,
        source: str | None = None,
        checksum: str | None = None,
        **metadata: str,
    ) -> int:
        """Index a document (sync wrapper).

        Safe to call from within a running event loop (e.g. Jupyter).

        Args:
            text: Document text to index.
            source: Source identifier for deduplication and filtering.
            checksum: If provided with ``source``, skip indexing when
                the checksum matches the previously stored value.
            **metadata: Additional metadata key-value pairs attached to
                every chunk.

        Returns:
            Number of chunks created.

        """
        return run_sync(self.aindex(text, source=source, checksum=checksum, **metadata))

    async def aindex(
        self,
        text: str,
        source: str | None = None,
        checksum: str | None = None,
        **metadata: str,
    ) -> int:
        """Index a document into the vector store and/or BM25 index.

        Chunks the text, computes embeddings, and stores them. Skips
        indexing if ``checksum`` matches the previously stored value.

        Args:
            text: Document text to index.
            source: Source identifier for deduplication and filtering.
            checksum: If provided with ``source``, skip indexing when
                the checksum matches the previously stored value.
            **metadata: Additional metadata key-value pairs attached to
                every chunk.

        Returns:
            Number of chunks created (0 if text is empty or checksum
            matches).

        """
        if not text or not text.strip():
            return 0

        if checksum and source and self._store is not None and not self._store.needs_reindex(source, checksum, strategy=self._chunker_name):
            return 0

        meta = dict(metadata)
        if source:
            meta["source"] = source

        chunks = self._chunker.chunk(text, metadata=meta)
        if not chunks:
            return 0

        if chunks and isinstance(chunks[0], HierarchicalTextChunk):
            chunks = self._enrich_hierarchical_chunks(chunks)

        texts = [c.content for c in chunks]
        if self._store is not None:
            embeddings = await self._embed.aembed(texts)
            ids = [f"{source or 'doc'}_{i}" for i in range(len(chunks))]
            metas = [c.metadata or {} for c in chunks]
            if checksum:
                for m in metas:
                    m["document_checksum"] = checksum
                    m["chunking_strategy"] = self._chunker_name
            await self._store.aadd(ids, texts, embeddings, metas)

        if self._bm25 is not None:
            for i, chunk in enumerate(chunks):
                chunk_id = f"{source or 'doc'}_{i}"
                self._bm25.add_document(
                    doc_id=chunk_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                )

        return len(chunks)

    def index_many(self, documents: list[dict[str, Any]]) -> int:
        """Index multiple documents sequentially (sync wrapper).

        Args:
            documents: List of dicts with keys matching ``aindex``
                parameters (``text``, ``source``, ``checksum``, etc.).

        Returns:
            Total number of chunks created across all documents.

        """
        return run_sync(self.aindex_many(documents))

    async def aindex_many(self, documents: list[dict[str, Any]]) -> int:
        """Index multiple documents sequentially.

        Args:
            documents: List of dicts with keys matching ``aindex``
                parameters (``text``, ``source``, ``checksum``, etc.).

        Returns:
            Total number of chunks created across all documents.

        """
        total = 0
        for doc in documents:
            total += await self.aindex(**doc)
        return total

    def chunk(self, text: str, **metadata: str) -> list[TextChunk]:
        """Split text into chunks without indexing.

        Args:
            text: Text to chunk.
            **metadata: Metadata attached to each chunk.

        Returns:
            List of TextChunk instances.

        """
        return self._chunker.chunk(text, metadata=metadata)

    def search(self, query: str, top_k: int = 5, **filters: str) -> list[SearchHit]:
        """Search for relevant chunks (sync wrapper).

        Safe to call from within a running event loop (e.g. Jupyter).

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            **filters: Metadata key-value filters passed to the store.

        Returns:
            Ranked list of search hits.

        """
        return run_sync(self.asearch(query, top_k=top_k, **filters))

    async def asearch(
        self,
        query: str,
        top_k: int = 5,
        **filters: str,
    ) -> list[SearchHit]:
        """Search for relevant chunks using vector, BM25, or hybrid search.

        Applies query expansion and reranking if configured.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            **filters: Metadata key-value filters passed to the store.

        Returns:
            Ranked list of search hits.

        """
        raw = await self._araw_search(query, top_k, filters)

        if self._query_expander is not None:
            raw = await self._expand_query(query, raw, top_k, filters)

        if self._reranker is not None and raw:
            raw = self._reranker.rerank(query, raw, n_results=top_k)

        hits = self._raw_to_hits(raw)
        return hits[:top_k]

    async def _araw_search(
        self, query: str, top_k: int, filters: dict[str, str],
    ) -> list[dict[str, Any]]:
        if self._store is not None and self._bm25 is not None:
            return await self._ahybrid_search(query, top_k)

        if self._store is not None:
            return await self._astore_search(query, top_k, where=filters or None)

        if self._bm25 is not None:
            results = self._bm25.search(query, top_k)
            if filters:
                results = [
                    r for r in results
                    if all(r.get("metadata", {}).get(k) == v for k, v in filters.items())
                ]
            return results

        return []

    async def _expand_query(
        self, query: str, raw: list[dict[str, Any]], top_k: int, filters: dict[str, str],
    ) -> list[dict[str, Any]]:
        try:
            queries = self._query_expander(query)  # type: ignore[misc]
        except Exception:
            logger.warning("Query expansion failed, using original query only")
            return raw
        if len(queries) <= 1:
            return raw
        all_raws = [raw]
        for extra_query in queries[1:]:
            extra_raw = await self._araw_search(extra_query, top_k, filters)
            all_raws.append(extra_raw)
        return fuse_search_results(all_raws, n_results=top_k)

    def query(
        self,
        question: str,
        generate_fn: Callable[[str], str | GenerationResult] | None = None,
        top_k: int = 5,
        prompt_template: str | None = None,
        max_context_chars: int | None = None,
        allow_llm_on_empty: bool = True,
        generate_timeout: float | None = None,
        **filters: str,
    ) -> QueryResult:
        """Retrieve context and generate an answer (sync wrapper).

        Safe to call from within a running event loop (e.g. Jupyter).
        """
        return run_sync(self.aquery(
            question, generate_fn=generate_fn, top_k=top_k,
            prompt_template=prompt_template,
            max_context_chars=max_context_chars,
            allow_llm_on_empty=allow_llm_on_empty,
            generate_timeout=generate_timeout,
            **filters,
        ))

    async def aquery(
        self,
        question: str,
        generate_fn: Callable[[str], str | GenerationResult] | None = None,
        top_k: int = 5,
        prompt_template: str | None = None,
        max_context_chars: int | None = None,
        allow_llm_on_empty: bool = True,
        generate_timeout: float | None = None,
        **filters: str,
    ) -> QueryResult:
        """Retrieve context and generate an answer.

        1. Searches for relevant chunks via ``asearch``.
        2. Formats hits into a context string (``format_hits``).
        3. Fills the prompt template with ``{context}`` and ``{question}``.
        4. Calls ``generate_fn`` in a thread (``asyncio.to_thread``).

        Args:
            question: The user question.
            generate_fn: A sync callable that takes the formatted prompt
                and returns an answer string or ``GenerationResult``.
                When ``None``, uses the default set on the RAG instance.
            top_k: Number of search results to retrieve.
            prompt_template: Must contain ``{context}`` and ``{question}``
                placeholders. Unknown placeholders resolve to empty string.
                Defaults to :data:`DEFAULT_RAG_PROMPT`.
            max_context_chars: Truncates the formatted context to this
                many characters. Note that each hit includes a header
                line (source, relevance) that counts toward the budget,
                so small values may exclude all content. ``None`` means
                no limit.
            allow_llm_on_empty: When ``False`` and no search hits are
                found, skip the LLM call and return an empty
                ``QueryResult``. Defaults to ``True`` (backward compat).
            generate_timeout: Maximum seconds to wait for ``generate_fn``
                to complete. Raises ``TimeoutError`` if exceeded. The
                underlying LLM request continues running in its thread
                (cannot be cancelled), so the API cost may still be
                incurred. Defaults to ``None`` (no timeout).
            **filters: Passed through to the vector store ``where`` clause.

        Returns:
            A ``QueryResult`` with the answer, search hits, deduplicated
            sources, the full prompt sent to ``generate_fn``, and
            generation metadata (usage, cost, duration).
        """
        fn = generate_fn or self._generate_fn
        if fn is None:
            raise ValueError(
                "generate_fn not provided and no default set on RAG. "
                "Pass generate_fn= or use FFAI.query()."
            )
        hits = await self.asearch(question, top_k=top_k, **filters)
        if not hits and not allow_llm_on_empty:
            return QueryResult(
                answer="", hits=[], sources=[], prompt="",
                usage=None, cost_usd=0.0, duration_ms=None,
            )
        context = format_hits(hits, max_chars=max_context_chars)
        template = prompt_template or DEFAULT_RAG_PROMPT
        prompt = template.format_map(defaultdict(str, {"context": context, "question": question}))
        coro = asyncio.to_thread(fn, prompt)
        raw = await (asyncio.wait_for(coro, timeout=generate_timeout) if generate_timeout is not None else coro)
        if isinstance(raw, GenerationResult):
            answer = raw.text
            gen_usage = raw.usage
            gen_cost = raw.cost_usd
            gen_duration = raw.duration_ms
        else:
            answer = raw
            gen_usage = None
            gen_cost = 0.0
            gen_duration = None
        sources = list(dict.fromkeys(h.source for h in hits if h.source))
        return QueryResult(
            answer=answer, hits=hits, sources=sources, prompt=prompt,
            usage=gen_usage, cost_usd=gen_cost, duration_ms=gen_duration,
        )

    def delete(self, source: str) -> None:
        """Delete all chunks associated with a source from the store and BM25 index.

        Args:
            source: Source identifier to delete.

        """
        if self._store is not None:
            self._store.delete_by_source(source)
        if self._bm25 is not None:
            self._bm25.delete_by_metadata("source", source)

    def count(self) -> int:
        """Return the total number of indexed chunks.

        Checks the vector store first, then falls back to BM25.

        Returns:
            Chunk count, or 0 if neither store is configured.

        """
        if self._store is not None:
            return self._store.count()
        if self._bm25 is not None:
            return self._bm25.count()
        return 0

    async def _astore_search(
        self, query: str, n_results: int, where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self._store is None:
            return []
        emb = await self._embed.aembed_single(query)
        hits = await self._store.asearch(emb, top_k=n_results, where=where)
        return [
            {"id": h.id, "content": h.content, "score": h.score,
             "metadata": h.metadata, "source": h.source}
            for h in hits
        ]

    async def _ahybrid_search(
        self, query: str, n_results: int,
    ) -> list[dict[str, Any]]:
        fetch_count = min(n_results * 3, 50)
        vector_results = await self._astore_search(query, fetch_count)
        for r in vector_results:
            r["search_type"] = "vector"
        bm25_results = self._bm25.search(query, fetch_count)  # type: ignore[union-attr]
        for r in bm25_results:
            r["search_type"] = "bm25"
        return reciprocal_rank_fusion(
            [vector_results, bm25_results],
            k=self._bm25_rrf_k,
            weights=[self._bm25_alpha, 1 - self._bm25_alpha],
        )[:n_results]

    def _raw_to_hits(self, raw: list[dict[str, Any]]) -> list[SearchHit]:
        hits = []
        for r in raw:
            dist = r.get("distance")
            if dist is not None:
                score = 1.0 - dist
            elif r.get("rrf_score") is not None:
                score = r["rrf_score"]
            else:
                score = r.get("score", 0.0)
            meta = r.get("metadata") or {}
            hits.append(SearchHit(
                id=r.get("id", ""),
                content=r.get("content", ""),
                score=score,
                source=meta.get("source", "") or r.get("source", ""),
                metadata=meta,
                parent_content=r.get("parent_content") or meta.get("parent_content"),
            ))
        return hits
