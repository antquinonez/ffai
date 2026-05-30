"""Qdrant vector store backend."""

from __future__ import annotations

import asyncio
import logging
import uuid as _uuid
from typing import Any

from ffai.rag.types import SearchHit

from .base import VectorStoreBase

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
    Supports server mode, local mode, in-memory mode, and cloud mode.

    Args:
        collection_name: Qdrant collection name.
        embedding_dim: Dimensionality of embedding vectors.
        host: Qdrant server host (server mode).
        port: Qdrant server port.
        path: Local storage path (local mode). When set, host/port
            are ignored.
        location: ``":memory:"`` for in-memory mode (ephemeral).
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
        location: str | None = None,
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
        elif location:
            client_kwargs["location"] = location
        else:
            client_kwargs["host"] = host
            client_kwargs["port"] = port

        self._client: Any = QdrantClient(**client_kwargs)  # type: ignore[union-attr]
        self._async_client: Any = None
        self._client_kwargs = client_kwargs
        self._is_local = (path is not None or location is not None) and url is None
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
            self._create_payload_indexes()

    def _create_payload_indexes(self) -> None:
        import warnings

        for field in ("source", "chunking_strategy"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._client.create_payload_index(
                        collection_name=self._collection_name,
                        field_name=field,
                        field_schema="keyword",
                    )
            except Exception:
                pass

    @property
    def name(self) -> str:
        return "qdrant"

    async def _get_async_client(self) -> Any:
        if self._async_client is None:
            self._async_client = AsyncQdrantClient(**self._client_kwargs)  # type: ignore[union-attr]
        return self._async_client

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

    @staticmethod
    def _ensure_uuid(id_str: str) -> str | _uuid.UUID:
        try:
            return _uuid.UUID(id_str)
        except ValueError:
            return _uuid.uuid5(_uuid.NAMESPACE_URL, id_str)

    async def aadd(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> int:
        points = [
            PointStruct(id=self._ensure_uuid(id_), vector=emb, payload={"content": text, **meta})
            for id_, text, emb, meta in zip(ids, texts, embeddings, metadatas)
        ]
        await asyncio.to_thread(
            self._client.upsert,
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
        results = await asyncio.to_thread(
            self._client.query_points,
            collection_name=self._collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=self._build_filter(where),
            with_payload=True,
        )
        hits = []
        for point in results.points:
            payload = dict(point.payload or {})
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
                must=[FieldCondition(key="source", match=MatchValue(value=source))],  # type: ignore[arg-type]
            ),
        )

    def delete_by_source_and_strategy(self, source: str, strategy: str) -> None:
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value=source)),
                    FieldCondition(key="chunking_strategy", match=MatchValue(value=strategy)),
                ],  # type: ignore[arg-type]
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
                "content": (dict(p.payload or {})).pop("content", ""),
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
                ],  # type: ignore[arg-type]
            ),
            limit=1,
            with_payload=True,
        )[0]
        if not results:
            return True
        return (results[0].payload or {}).get("document_checksum", "") != checksum
