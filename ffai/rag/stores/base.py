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
        ...

    @abstractmethod
    async def asearch(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        ...

    @abstractmethod
    def delete_by_source(self, source: str) -> None:
        ...

    @abstractmethod
    def delete_by_source_and_strategy(self, source: str, strategy: str) -> None:
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

    @abstractmethod
    def list_sources(self) -> list[str]:
        ...

    @abstractmethod
    def get_all(self) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def needs_reindex(self, source: str, checksum: str, strategy: str = "default") -> bool:
        ...
