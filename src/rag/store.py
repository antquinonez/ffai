from __future__ import annotations

import logging
from typing import Any

from .types import SearchHit

try:
    import chromadb  # type: ignore[reportMissingImports]
    from chromadb.config import Settings  # type: ignore[reportMissingImports]

    CHROMADB_AVAILABLE = True
except Exception:
    chromadb = None  # type: ignore[assignment]
    Settings = None  # type: ignore[assignment]
    CHROMADB_AVAILABLE = False

from .embed import Embeddings

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        collection_name: str = "ffai_kb",
        dir: str = "./chroma_db",
        embed: Embeddings | None = None,
    ) -> None:
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb is not installed. Install with: pip install chromadb")

        self.collection_name = collection_name
        self.dir = dir

        self._client = chromadb.PersistentClient(  # type: ignore[union-attr]
            path=dir,
            settings=Settings(anonymized_telemetry=False),  # type: ignore[operator]
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embed = embed

    async def aadd(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> int:
        self._collection.add(
            ids=ids,
            embeddings=embeddings,  # type: ignore[reportArgumentType]
            documents=texts,
            metadatas=metadatas,  # type: ignore[reportArgumentType]
        )
        logger.info(f"Added {len(ids)} chunks to {self.collection_name}")
        return len(ids)

    async def asearch(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[SearchHit] = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else None
                score = 1.0 - distance if distance is not None else 0.0
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                hits.append(SearchHit(
                    id=doc_id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    score=score,
                    source=str(meta.get("source", "")),
                    metadata=dict(meta),
                ))

        return hits

    def delete_by_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})
        logger.info(f"Deleted chunks for source: {source}")

    def delete_by_source_and_strategy(self, source: str, strategy: str) -> None:
        self._collection.delete(
            where={"$and": [{"source": source}, {"chunking_strategy": strategy}]}
        )

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def list_sources(self) -> list[str]:
        results = self._collection.get(include=["metadatas"])
        sources = set()
        if results["metadatas"]:
            for meta in results["metadatas"]:
                if "source" in meta:
                    sources.add(meta["source"])
        return sorted(sources)

    def get_all(self) -> list[dict[str, Any]]:
        results = self._collection.get(include=["documents", "metadatas"])
        docs = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                docs.append({
                    "id": doc_id,
                    "content": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })
        return docs

    def needs_reindex(self, source: str, checksum: str, strategy: str = "default") -> bool:
        results = self._collection.get(
            where={"$and": [{"source": source}, {"chunking_strategy": strategy}]},
            include=["metadatas"],
            limit=1,
        )
        if not results["metadatas"]:
            return True
        return results["metadatas"][0].get("document_checksum", "") != checksum
