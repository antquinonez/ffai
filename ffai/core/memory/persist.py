"""Parquet persistence for ``TurnVectorStore``.

Serializes the in-memory store to a single Parquet file so vector-indexed
memory survives process restarts when the user opts in via
``config.memory.persist: true``.

Schema (v1):

- ``text``: ``String`` — the plain text that was embedded.
- ``embedding``: ``List(Float64)`` — the float vector.
- ``turn``: ``String`` (JSON) — the raw turn dict, round-tripped via ``json``.
- ``metadata``: ``String`` (JSON) — the caller metadata dict.
- ``_schema_version``: ``Int8`` — always ``1`` for this version.

Functions are stateless I/O helpers; the store itself remains in-memory.
"""

from __future__ import annotations

import json
import os
from typing import Any

import polars as pl

from .turn_store import TurnVectorStore

SCHEMA_VERSION: int = 1


def persist_store(store: TurnVectorStore, path: str) -> None:
    """Write all entries in *store* to a Parquet file at *path*.

    Overwrites any existing file. Empty stores produce a valid zero-row
    Parquet file.

    Args:
        store: The ``TurnVectorStore`` to serialize.
        path: Destination file path. Parent directories are created.

    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for entry in store.iter_entries():
        rows.append(
            {
                "text": entry.text,
                "embedding": entry.embedding,
                "turn": json.dumps(entry.turn),
                "metadata": json.dumps(entry.metadata),
                "_schema_version": SCHEMA_VERSION,
            }
        )

    df = pl.DataFrame(
        rows,
        schema={
            "text": pl.String,
            "embedding": pl.List(pl.Float64),
            "turn": pl.String,
            "metadata": pl.String,
            "_schema_version": pl.Int8,
        },
    )
    df.write_parquet(path)


def load_store(path: str) -> TurnVectorStore:
    """Load a Parquet file written by :func:`persist_store` into a fresh store.

    Args:
        path: Source file path.

    Returns:
        A new ``TurnVectorStore`` whose entries and ordering match what
        was persisted.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file's ``_schema_version`` is not ``1``.

    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Memory persistence file not found: {path}")

    df = pl.read_parquet(path)
    if df.height == 0:
        return TurnVectorStore()

    if "_schema_version" not in df.columns:
        raise ValueError(
            "Memory persistence file missing '_schema_version' column; "
            "cannot determine format version."
        )

    versions = df.select("_schema_version").unique().to_series().to_list()
    if any(v != SCHEMA_VERSION for v in versions):
        raise ValueError(
            f"Unsupported memory persistence schema version(s): {versions}. "
            f"This build supports version {SCHEMA_VERSION} only."
        )

    store = TurnVectorStore()
    for row in df.iter_rows(named=True):
        store.add(
            text=row["text"],
            embedding=list(row["embedding"]),
            turn=json.loads(row["turn"]),
            metadata=json.loads(row["metadata"]),
        )
    return store
