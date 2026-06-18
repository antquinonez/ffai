"""In-memory vector store for memory recall over PermanentHistory turns.

Stores ``(text, embedding, turn, metadata)`` tuples in parallel lists and
provides cosine-similarity search. Designed for Tier 1 scale (<100K turns);
Chroma/Qdrant backends are a future concern for larger workloads.
"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from typing import Any

from ..embeddings import Embeddings
from .types import Entry, TurnHit


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length float vectors.

    Thin wrapper around :meth:`Embeddings.cosine_similarity` so that
    ``turn_store`` consumers don't need to import the ``Embeddings``
    class just to compute similarity. Returns ``0.0`` when either
    vector has zero magnitude.

    Args:
        a: First vector.
        b: Second vector. Must be the same length as *a*.

    Returns:
        Cosine similarity in ``[-1.0, 1.0]``.

    """
    return Embeddings.cosine_similarity(a, b)


class TurnVectorStore:
    """Append-only in-memory store of embedded turns with cosine search.

    Entries are stored in four parallel lists keyed by an integer index.
    ``add()`` returns the index of the appended entry; that index is
    preserved on the corresponding ``TurnHit`` returned by ``search()``.

    Search is O(N) over all stored entries. Adequate for Tier 1 scale;
    a vector-index backend (Chroma/Qdrant) is a future concern.

    """

    def __init__(self) -> None:
        self._texts: list[str] = []
        self._embeddings: list[list[float]] = []
        self._turns: list[dict[str, Any]] = []
        self._metadatas: list[dict[str, Any]] = []

    def add(
        self,
        text: str,
        embedding: list[float],
        turn: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Append an entry and return its integer index.

        Args:
            text: Plain text that was embedded.
            embedding: Float vector returned by the embedding backend.
            turn: Raw turn dict (mirrors PermanentHistory turn shape).
            metadata: Caller metadata. Defaults to an empty dict.

        Returns:
            The integer index of the newly appended entry.

        """
        idx = len(self._texts)
        self._texts.append(text)
        self._embeddings.append(list(embedding))
        self._turns.append(deepcopy(turn))
        self._metadatas.append(deepcopy(metadata) if metadata is not None else {})
        return idx

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list[TurnHit]:
        """Return up to ``top_k`` turns ranked by cosine similarity.

        Args:
            query_embedding: Embedded query vector.
            top_k: Maximum hits to return.
            threshold: Optional minimum cosine similarity. Hits below
                this score are excluded. ``None`` means no floor.

        Returns:
            Hits sorted by ``score`` descending. Empty list if the
            store is empty or all hits fall below ``threshold``.

        """
        if not self._texts or top_k <= 0:
            return []
        scored: list[TurnHit] = []
        for idx in range(len(self._texts)):
            score = cosine_similarity(query_embedding, self._embeddings[idx])
            if threshold is not None and score < threshold:
                continue
            scored.append(
                TurnHit(
                    score=score,
                    turn=self._turns[idx],
                    turn_index=idx,
                    text=self._texts[idx],
                    metadata=self._metadatas[idx],
                )
            )
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    def iter_entries(self) -> Iterator[Entry]:
        """Yield all entries as ``Entry`` tuples in insertion order.

        Used by ``Memory.reindex()`` and ``persist_store()`` to read the
        store's full contents without exposing the underlying lists.

        """
        for idx in range(len(self._texts)):
            yield Entry(
                text=self._texts[idx],
                embedding=list(self._embeddings[idx]),
                turn=deepcopy(self._turns[idx]),
                metadata=deepcopy(self._metadatas[idx]),
            )

    def count(self) -> int:
        """Number of entries currently in the store."""
        return len(self._texts)

    def clear(self) -> None:
        """Remove all entries. Subsequent ``count()`` returns 0."""
        self._texts.clear()
        self._embeddings.clear()
        self._turns.clear()
        self._metadatas.clear()
