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
from .splitters import TextChunk, get_chunker
from .store import CHROMADB_AVAILABLE, VectorStore
from .types import QueryResult, SearchHit

logger = logging.getLogger(__name__)


class RAG:
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

        if bm25_alpha is not None:
            self._bm25 = BM25Index()
            self._bm25_alpha = bm25_alpha

        if reranker is not None:
            self._reranker = get_reranker(reranker)

    @classmethod
    def from_config(
        cls,
        *,
        bm25_only: bool = False,
        **overrides: Any,
    ) -> RAG:
        from ..config import get_config

        cfg = get_config().rag
        kwargs: dict[str, Any] = {
            "embed": cfg.embedding_model,
            "chunker": cfg.chunker,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "bm25_alpha": cfg.bm25_alpha,
            "reranker": cfg.reranker,
        }
        kwargs.update(overrides)

        if not bm25_only and CHROMADB_AVAILABLE:
            store = VectorStore(
                collection_name=cfg.collection_name,
                dir=cfg.persist_dir,
            )
            kwargs["store"] = store

        if kwargs.get("bm25_alpha") is None and (bm25_only or not CHROMADB_AVAILABLE):
            kwargs["bm25_alpha"] = 0.6

        return cls(**kwargs)

    def index(
        self,
        text: str,
        source: str | None = None,
        checksum: str | None = None,
        **metadata: str,
    ) -> int:
        return run_sync(self.aindex(text, source=source, checksum=checksum, **metadata))

    async def aindex(
        self,
        text: str,
        source: str | None = None,
        checksum: str | None = None,
        **metadata: str,
    ) -> int:
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
        return run_sync(self.aindex_many(documents))

    async def aindex_many(self, documents: list[dict[str, Any]]) -> int:
        total = 0
        for doc in documents:
            total += await self.aindex(**doc)
        return total

    def chunk(self, text: str, **metadata: str) -> list[TextChunk]:
        return self._chunker.chunk(text, metadata=metadata)

    def search(self, query: str, top_k: int = 5, **filters: str) -> list[SearchHit]:
        return run_sync(self.asearch(query, top_k=top_k, **filters))

    async def asearch(
        self,
        query: str,
        top_k: int = 5,
        **filters: str,
    ) -> list[SearchHit]:
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
        generate_fn: Callable[[str], str],
        top_k: int = 5,
        prompt_template: str | None = None,
        max_context_chars: int | None = None,
        **filters: str,
    ) -> QueryResult:
        """Retrieve context and generate an answer (sync wrapper).

        Safe to call from within a running event loop (e.g. Jupyter).
        """
        return run_sync(self.aquery(
            question, generate_fn=generate_fn, top_k=top_k,
            prompt_template=prompt_template,
            max_context_chars=max_context_chars, **filters,
        ))

    async def aquery(
        self,
        question: str,
        generate_fn: Callable[[str], str],
        top_k: int = 5,
        prompt_template: str | None = None,
        max_context_chars: int | None = None,
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
                and returns an answer string.
            top_k: Number of search results to retrieve.
            prompt_template: Must contain ``{context}`` and ``{question}``
                placeholders. Unknown placeholders resolve to empty string.
                Defaults to :data:`DEFAULT_RAG_PROMPT`.
            max_context_chars: Truncates the formatted context to this
                many characters. Note that each hit includes a header
                line (source, relevance) that counts toward the budget,
                so small values may exclude all content. ``None`` means
                no limit.
            **filters: Passed through to the vector store ``where`` clause.

        Returns:
            A ``QueryResult`` with the answer, search hits, deduplicated
            sources, and the full prompt sent to ``generate_fn``.
        """
        hits = await self.asearch(question, top_k=top_k, **filters)
        context = format_hits(hits, max_chars=max_context_chars)
        template = prompt_template or DEFAULT_RAG_PROMPT
        prompt = template.format_map(defaultdict(str, {"context": context, "question": question}))
        answer = await asyncio.to_thread(generate_fn, prompt)
        sources = list(dict.fromkeys(h.source for h in hits if h.source))
        return QueryResult(answer=answer, hits=hits, sources=sources, prompt=prompt)

    def delete(self, source: str) -> None:
        if self._store is not None:
            self._store.delete_by_source(source)
        if self._bm25 is not None:
            self._bm25.delete_by_metadata("source", source)

    def count(self) -> int:
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
                parent_content=r.get("parent_content"),
            ))
        return hits
