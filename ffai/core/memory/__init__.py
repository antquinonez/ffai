"""Memory vector recall: semantic search over PermanentHistory turns.

Tier 1 primitives:

- :class:`TurnHit` — ranked result returned by search.
- :class:`Entry` — storage tuple yielded by ``TurnVectorStore.iter_entries``.
- :class:`TurnVectorStore` — in-memory append-only vector store with
  cosine-similarity search.

L2 adds:

- :class:`Memory` — facade combining an ``Embeddings`` backend with a
  ``TurnVectorStore`` for high-level ``index_turn`` / ``search`` /
  ``reindex`` operations.

L3 adds:

- :func:`persist_store` / :func:`load_store` — Parquet persistence for
  the in-memory store.

"""

from __future__ import annotations

from .memory import Memory
from .persist import load_store, persist_store
from .turn_store import TurnVectorStore, cosine_similarity
from .types import Entry, TurnHit

__all__ = [
    "Entry",
    "Memory",
    "TurnHit",
    "TurnVectorStore",
    "cosine_similarity",
    "load_store",
    "persist_store",
]
