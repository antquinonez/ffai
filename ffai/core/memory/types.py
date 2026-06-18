"""Types for the memory vector recall feature.

Defines the result and storage shapes used by ``TurnVectorStore`` and
the higher-level ``Memory`` facade (added in L2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol


class EmbeddingBackend(Protocol):
    """Duck-typed contract for an embedding backend used by ``Memory``.

    ``Embeddings`` satisfies this natively; tests can supply any object
    with matching ``embed`` and ``aembed`` signatures (e.g.,
    ``FakeEmbeddings``).

    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for one or more texts (synchronous)."""
        ...

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for one or more texts (asynchronous)."""
        ...


@dataclass(frozen=True)
class TurnHit:
    """A single ranked result returned by ``TurnVectorStore.search``.

    Attributes:
        score: Cosine similarity in ``[-1.0, 1.0]`` between the query
            embedding and the stored turn's embedding. Sorted descending
            by ``search()``.
        turn: The raw turn dict stored at index time. Mirrors the
            ``PermanentHistory`` turn shape:
            ``{"role": str, "content": [{"type": "text", "text": str}],
            "timestamp": float, "metadata": dict}``.
        turn_index: Position of the entry in ``TurnVectorStore`` at the
            time of the search. Stable within a single ``search()`` call
            for a given store; may shift after ``clear()``.
        text: Pre-extracted plain text (typically the Q+A pair) that was
            embedded. Equal to the string passed to ``add(text=...)``.
        metadata: Caller-provided metadata attached at index time.
            Always present (possibly empty dict). Tier 2 will populate
            ``user_id`` / ``session_id`` / ``agent_id`` here.

    """

    score: float
    turn: dict[str, Any]
    turn_index: int
    text: str
    metadata: dict[str, Any]


class Entry(NamedTuple):
    """Storage tuple yielded by ``TurnVectorStore.iter_entries``.

    Fields mirror what was passed to ``add()`` at index time. Used by
    ``Memory.reindex()`` (L2) and ``persist_store()`` (L3) to read the
    store's full contents without reaching into private state.

    Attributes:
        text: The plain text that was embedded.
        embedding: The float vector returned by the embedding backend.
        turn: The raw turn dict.
        metadata: The caller-provided metadata dict.

    """

    text: str
    embedding: list[float]
    turn: dict[str, Any]
    metadata: dict[str, Any]
