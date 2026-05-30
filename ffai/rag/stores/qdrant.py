"""Qdrant vector store backend."""

from __future__ import annotations

import logging
from typing import Any

from .base import VectorStoreBase
from ffai.rag.types import SearchHit

logger = logging.getLogger(__name__)

try:
    from qdrant_client import AsyncQdrantClient, QdrantClient  # type: ignore[reportMissingImports]
    from qdrant_client.models import (  # type: ignore[reportMissingImports]
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QdrantClient = None  # type: ignore[assignment]
    AsyncQdrantClient = None  # type: ignore[assignment]
    QDRANT_AVAILABLE = False


def get_store_class() -> type[VectorStoreBase]:
    if not QDRANT_AVAILABLE:
        raise ImportError(
            "Qdrant backend requires qdrant-client. "
            "Install with: pip install qdrant-client"
        )
    return QdrantVectorStore


class QdrantVectorStore(VectorStoreBase):
    """Qdrant vector store backend.

    Stores embeddings in a Qdrant collection using cosine distance.
    Supports server mode, local mode, and in-memory mode.

    Args:
        collection_name: Qdrant collection name.
        embedding_dim: Dimensionality of embedding vectors.
        host: Qdrant server host (server mode).
        port: Qdrant server port.
        path: Local storage path (local mode). When set, host/port
            are ignored.
        api_key: Qdrant API key (cloud mode).
        url: Qdrant URL (cloud mode).
    """

    def __init__(
        self,
        *,
        collection_name: str = "ffai_kb",
        embedding_dim: int = 1024,
        host: str = "localhost",
        port: int = 6333,
        path: str | None = None,
        api_key: str | None = None,
        url: str | None = None,
    ) -> None:
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "Qdrant backend requires qdrant-client. "
                "Install with: pip install qdrant-client"
            )

        self._collection_name = collection_name
        self._embedding_dim = embedding_dim

        client_kwargs: dict[str, Any] = {}
        if url:
            client_kwargs["url"] = url
            client_kwargs["api_key"] = api_key
        elif path:
            client_kwargs["path"] = path
        else:
            client_kwargs["host"] = host
            client_kwargs["port"] = port

        self._client: Any = QdrantClient(**client_kwargs)
        self._async_client: Any = AsyncQdrantClient(**client_kwargs)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self._client.get_collections().collections
        names = [c.name for c in collections]
        if self._collection_name not in names:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    @property
    def name(self) -> str:
        return "qdrant"

    def _build_filter(self, where: dict[str, Any] | None) -> Any:
        if not where:
            return None
        conditions = []
        if "$and" in where:
            for cond in where["$and"]:
                for k, v in cond.items():
                    conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
        else:
            for k, v in where.items():
                conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
        return Filter(must=conditions)

    async def aadd(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> int:
        points = [
            PointStruct(id=id_, vector=emb, payload={"content": text, **meta})
            for id_, text, emb, meta in zip(ids, texts, embeddings, metadatas)
        ]
        await self._async_client.upsert(
            collection_name=self._collection_name,
            points=points,
        )
        logger.info(f"Added {len(ids)} chunks to {self._collection_name}")
        return len(ids)

    async def asearch(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        results = await self._async_client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=self._build_filter(where),
            with_payload=True,
        )
        hits = []
        for point in results.points:
            payload = point.payload or {}
            hits.append(SearchHit(
                id=str(point.id),
                content=payload.pop("content", ""),
                score=point.score or 0.0,
                source=payload.get("source", ""),
                metadata=payload,
            ))
        return hits

    def delete_by_source(self, source: str) -> None:
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))],
            ),
        )

    def delete_by_source_and_strategy(self, source: str, strategy: str) -> None:
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value=source)),
                    FieldCondition(key="chunking_strategy", match=MatchValue(value=strategy)),
                ],
            ),
        )

    def count(self) -> int:
        info = self._client.get_collection(self._collection_name)
        return info.points_count or 0

    def clear(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._ensure_collection()

    def list_sources(self) -> list[str]:
        results = self._client.scroll(
            collection_name=self._collection_name,
            limit=10000,
            with_payload=True,
        )[0]
        sources = set()
        for point in results:
            payload = point.payload or {}
            if "source" in payload:
                sources.add(payload["source"])
        return sorted(sources)

    def get_all(self) -> list[dict[str, Any]]:
        results = self._client.scroll(
            collection_name=self._collection_name,
            limit=10000,
            with_payload=True,
        )[0]
        return [
            {
                "id": str(p.id),
                "content": (p.payload or {}).pop("content", ""),
                "metadata": {k: v for k, v in (p.payload or {}).items() if k != "content"},
            }
            for p in results
        ]

    def needs_reindex(self, source: str, checksum: str, strategy: str = "default") -> bool:
        results = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value=source)),
                    FieldCondition(key="chunking_strategy", match=MatchValue(value=strategy)),
                ],
            ),
            limit=1,
            with_payload=True,
        )[0]
        if not results:
            return True
        return (results[0].payload or {}).get("document_checksum", "") != checksum
