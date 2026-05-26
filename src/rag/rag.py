from __future__ import annotations

import asyncio
import logging
from typing import Any

from .embed import Embeddings
from .indexing import BM25Index
from .search import get_reranker
from .search.hybrid import reciprocal_rank_fusion
from .search.rerankers import RerankerBase
from .splitters import TextChunk, get_chunker
from .store import VectorStore
from .types import SearchHit

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
    ) -> None:
        if isinstance(embed, str):
            embed = Embeddings(model=embed)

        self._embed = embed
        self._store = store
        self._chunker = get_chunker(
            strategy=chunker, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )
        self._bm25: BM25Index | None = None
        self._bm25_alpha: float = 0.6
        self._bm25_rrf_k: int = 60
        self._reranker: RerankerBase | None = None

        if bm25_alpha is not None:
            self._bm25 = BM25Index()
            self._bm25_alpha = bm25_alpha

        if reranker is not None:
            self._reranker = get_reranker(reranker)

    def index(
        self,
        text: str,
        source: str | None = None,
        **metadata: str,
    ) -> int:
        return asyncio.run(self.aindex(text, source=source, **metadata))

    async def aindex(
        self,
        text: str,
        source: str | None = None,
        **metadata: str,
    ) -> int:
        if not text or not text.strip():
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

    def chunk(self, text: str, **metadata: str) -> list[TextChunk]:
        return self._chunker.chunk(text, metadata=metadata)

    def search(self, query: str, top_k: int = 5, **filters: str) -> list[SearchHit]:
        return asyncio.run(self.asearch(query, top_k=top_k, **filters))

    async def asearch(
        self,
        query: str,
        top_k: int = 5,
        **filters: str,
    ) -> list[SearchHit]:
        if self._store is None:
            return []

        if self._bm25 is not None and self._store is not None:
            raw = await self._ahybrid_search(query, top_k)
        else:
            query_emb = await self._embed.aembed_single(query)
            store_hits = await self._store.asearch(
                query_emb, top_k=top_k, where=filters or None,
            )
            raw = [
                {"id": h.id, "content": h.content, "score": h.score,
                 "metadata": h.metadata, "source": h.source}
                for h in store_hits
            ]

        if self._reranker is not None and raw:
            raw = self._reranker.rerank(query, raw, n_results=top_k)

        hits = self._raw_to_hits(raw)

        return hits[:top_k]

    def delete(self, source: str) -> None:
        if self._store is not None:
            self._store.delete_by_source(source)
        if self._bm25 is not None:
            self._bm25.delete_by_metadata("source", source)

    def count(self) -> int:
        if self._store is None:
            return 0
        return self._store.count()

    async def _astore_search(
        self, query: str, n_results: int,
    ) -> list[dict[str, Any]]:
        if self._store is None:
            return []
        emb = await self._embed.aembed_single(query)
        hits = await self._store.asearch(emb, top_k=n_results)
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
                source=meta.get("source", ""),
                metadata=meta,
                parent_content=r.get("parent_content"),
            ))
        return hits
