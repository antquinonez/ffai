"""Abstract base class for vector store backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ffai.rag.types import SearchHit


class VectorStoreBase(ABC):
    """Abstract base class for vector store backends.

    All backends must implement the core CRUD + search methods.
    The ``where`` parameter in ``asearch`` accepts a backend-neutral
    filter dict with string keys and values.  Each backend translates
    this to its native filter format internally.

    Example backend-neutral filters::

        {"source": "doc1"}
        {"chunking_strategy": "recursive", "source": "doc1"}

    For compound filters, backends should support an ``$and`` key::

        {"$and": [{"source": "doc1"}, {"chunking_strategy": "recursive"}]}
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g. ``"chroma"``, ``"pgvector"``)."""
        ...

    @abstractmethod
    async def aadd(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> int:
        """Add documents with pre-computed embeddings to the store.

        Args:
            ids: Unique identifiers for each document.
            texts: Document text content.
            embeddings: Pre-computed embedding vectors.
            metadatas: Metadata dicts (must include ``source`` key).

        Returns:
            Number of documents added.
        """
        ...

    @abstractmethod
    async def asearch(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Search for documents by vector similarity.

        Args:
            query_embedding: Query vector.
            top_k: Maximum number of results.
            where: Metadata filter dict (e.g. ``{"source": "doc1"}``).

        Returns:
            Ranked search hits.
        """
        ...

    @abstractmethod
    def delete_by_source(self, source: str) -> None:
        """Delete all chunks matching ``source``."""
        ...

    @abstractmethod
    def delete_by_source_and_strategy(self, source: str, strategy: str) -> None:
        """Delete chunks matching both ``source`` and ``chunking_strategy``."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored chunks."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Delete all stored data and recreate the collection."""
        ...

    @abstractmethod
    def list_sources(self) -> list[str]:
        """Return a sorted list of indexed source names."""
        ...

    @abstractmethod
    def get_all(self) -> list[dict[str, Any]]:
        """Return all stored documents as dicts with ``id``, ``content``, ``metadata`` keys."""
        ...

    @abstractmethod
    def needs_reindex(self, source: str, checksum: str, strategy: str = "default") -> bool:
        """Check whether ``source`` needs re-indexing.

        Args:
            source: Source identifier.
            checksum: Expected document checksum.
            strategy: Chunking strategy name.

        Returns:
            True if the source has changed or is not indexed.
        """
        ...
