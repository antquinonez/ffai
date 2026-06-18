"""High-level facade combining an Embeddings backend with a TurnVectorStore.

Provides synchronous and asynchronous methods for indexing completed turns
(Q+A pairs), semantic search, and re-embedding with a new model. The
L4 integration layer wires this into ``FFAI`` via ``HistoryRecorder``.
"""

from __future__ import annotations

import logging
from typing import Any

from .turn_store import TurnVectorStore
from .types import EmbeddingBackend

logger = logging.getLogger(__name__)


class Memory:
    """Semantic recall over completed conversation turns.

    Wraps an :class:`EmbeddingBackend` and a
    :class:`TurnVectorStore`. Provides two indexing entry points:

    - :meth:`index_turn` / :meth:`aindex_turn` — extract text from a
      structured ``turn`` dict via ``turn["content"][0]["text"]``.
    - :meth:`index_turn_text` / :meth:`aindex_turn_text` — embed an
      arbitrary caller-supplied string while still storing the
      structured ``turn`` dict alongside. Used by ``HistoryRecorder``
      (L4) to embed the Q+A pair (``f"{prompt}\\n{response}"``) rather
      than just the response.

    Args:
        embeddings: Embedding backend (LiteLLM API model or local
            ``local/...`` model). Any object implementing
            :class:`EmbeddingBackend` is accepted.
        store: Optional ``TurnVectorStore``. Defaults to a fresh
            instance. Public and settable so callers can swap in a
            store loaded from Parquet (L3).

    """

    def __init__(
        self,
        embeddings: EmbeddingBackend,
        store: TurnVectorStore | None = None,
    ) -> None:
        self._embeddings = embeddings
        self.store: TurnVectorStore = store if store is not None else TurnVectorStore()

    def index_turn(
        self,
        turn: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Embed ``turn["content"][0]["text"]`` and store the turn.

        Args:
            turn: Turn dict mirroring ``PermanentHistory`` shape. Must
                contain ``content[0]["text"]``.
            metadata: Optional caller metadata. Defaults to empty dict.

        Returns:
            The integer index of the stored entry.

        """
        text = turn["content"][0]["text"]
        embedding = self._embeddings.embed([text])[0]
        return self.store.add(text=text, embedding=embedding, turn=turn, metadata=metadata)

    def index_turn_text(
        self,
        text: str,
        turn: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Embed an arbitrary ``text`` and store the structured ``turn``.

        Use this when the embedded text differs from
        ``turn["content"][0]["text"]`` — e.g., when embedding the Q+A
        pair (``f"{prompt}\\n{response}"``) while still storing the
        response as the canonical turn content.

        Args:
            text: The plain text to embed.
            turn: The structured turn dict to store alongside.
            metadata: Optional caller metadata.

        Returns:
            The integer index of the stored entry.

        """
        embedding = self._embeddings.embed([text])[0]
        return self.store.add(text=text, embedding=embedding, turn=turn, metadata=metadata)

    async def aindex_turn(
        self,
        turn: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Async variant of :meth:`index_turn`."""
        text = turn["content"][0]["text"]
        embedding = (await self._embeddings.aembed([text]))[0]
        return self.store.add(text=text, embedding=embedding, turn=turn, metadata=metadata)

    async def aindex_turn_text(
        self,
        text: str,
        turn: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Async variant of :meth:`index_turn_text`."""
        embedding = (await self._embeddings.aembed([text]))[0]
        return self.store.add(text=text, embedding=embedding, turn=turn, metadata=metadata)

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list:
        """Embed ``query`` and return ranked hits from the store.

        Args:
            query: Plain-text query.
            top_k: Maximum hits to return.
            threshold: Optional minimum cosine similarity.

        Returns:
            List of :class:`TurnHit` sorted by score descending.

        """
        query_embedding = self._embeddings.embed([query])[0]
        return self.store.search(
            query_embedding=query_embedding, top_k=top_k, threshold=threshold
        )

    async def asearch(
        self,
        query: str,
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list:
        """Async variant of :meth:`search`."""
        query_embedding = (await self._embeddings.aembed([query]))[0]
        return self.store.search(
            query_embedding=query_embedding, top_k=top_k, threshold=threshold
        )

    def reindex(self, new_embeddings: EmbeddingBackend) -> None:
        """Re-embed all stored texts with a new embedding model.

        Reads all entries via :meth:`TurnVectorStore.iter_entries`,
        clears the store, and re-adds each entry with the new embedding.
        The turn dicts and metadata are preserved verbatim. After
        reindexing, ``self._embeddings`` is updated so subsequent
        :meth:`search` calls embed queries with the new model.

        Args:
            new_embeddings: The new embedding backend to use.

        """
        entries = list(self.store.iter_entries())
        texts = [entry.text for entry in entries]
        if not texts:
            self._embeddings = new_embeddings
            return
        new_vectors = new_embeddings.embed(texts)
        self.store.clear()
        for entry, embedding in zip(entries, new_vectors, strict=True):
            self.store.add(
                text=entry.text,
                embedding=embedding,
                turn=entry.turn,
                metadata=entry.metadata,
            )
        self._embeddings = new_embeddings

    def count(self) -> int:
        """Number of turns currently indexed."""
        return self.store.count()

    def clear(self) -> None:
        """Remove all indexed turns."""
        self.store.clear()
