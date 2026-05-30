"""SQLite + sqlite-vss vector store backend."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from .base import VectorStoreBase
from ffai.rag.types import SearchHit

logger = logging.getLogger(__name__)

try:
    import sqlite_vss  # type: ignore[reportMissingImports]

    SQLITE_VSS_AVAILABLE = True
except ImportError:
    sqlite_vss = None  # type: ignore[assignment]
    SQLITE_VSS_AVAILABLE = False


def get_store_class() -> type[VectorStoreBase]:
    if not SQLITE_VSS_AVAILABLE:
        raise ImportError(
            "sqlite-vss backend requires sqlite-vss. "
            "Install with: pip install sqlite-vss"
        )
    return SQLiteVssStore


class SQLiteVssStore(VectorStoreBase):
    """SQLite + sqlite-vss vector store backend.

    Stores embeddings in a local SQLite database file using the
    sqlite-vss extension for vector similarity search. No server
    required.

    Args:
        db_path: Path to the SQLite database file.
        table_name: Base table name (creates ``{table_name}_vss``
            and ``{table_name}_data``).
        embedding_dim: Dimensionality of embedding vectors.
    """

    def __init__(
        self,
        *,
        db_path: str = "./ffai_vectors.db",
        table_name: str = "ffai_vectors",
        embedding_dim: int = 1024,
    ) -> None:
        if not SQLITE_VSS_AVAILABLE:
            raise ImportError(
                "sqlite-vss backend requires sqlite-vss. "
                "Install with: pip install sqlite-vss"
            )

        self._db_path = db_path
        self._table_name = table_name
        self._embedding_dim = embedding_dim
        self._vss_table = f"{table_name}_vss"
        self._data_table = f"{table_name}_data"

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._setup_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.enable_load_extension(True)
        sqlite_vss.load(conn)
        conn.row_factory = sqlite3.Row
        return conn

    def _setup_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._data_table} (
                    id        TEXT PRIMARY KEY,
                    content   TEXT NOT NULL,
                    metadata  TEXT NOT NULL DEFAULT '{{}}'
                )
            """)
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {self._vss_table}
                USING vss0(embedding({self._embedding_dim}))
            """)
            conn.commit()
        finally:
            conn.close()

    @property
    def name(self) -> str:
        return "sqlite_vss"

    def _sync_add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> int:
        conn = self._connect()
        try:
            for id_, text, emb, meta in zip(ids, texts, embeddings, metadatas):
                conn.execute(
                    f"INSERT OR REPLACE INTO {self._data_table} (id, content, metadata) VALUES (?, ?, ?)",
                    (id_, text, json.dumps(meta)),
                )
                row = conn.execute(
                    f"SELECT rowid FROM {self._data_table} WHERE id = ?", (id_,)
                ).fetchone()
                if row:
                    emb_json = json.dumps(emb)
                    conn.execute(
                        f"INSERT OR REPLACE INTO {self._vss_table} (rowid, embedding) VALUES (?, ?)",
                        (row["rowid"], emb_json),
                    )
            conn.commit()
        finally:
            conn.close()
        logger.info(f"Added {len(ids)} chunks to {self._db_path}")
        return len(ids)

    async def aadd(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> int:
        return await asyncio.to_thread(self._sync_add, ids, texts, embeddings, metadatas)

    def _sync_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchHit]:
        conn = self._connect()
        try:
            emb_json = json.dumps(query_embedding)
            results = conn.execute(
                f"SELECT rowid, distance FROM {self._vss_table} "
                f"WHERE vss_search(embedding, ?) LIMIT ?",
                (emb_json, top_k),
            ).fetchall()

            hits: list[SearchHit] = []
            for row in results:
                data_row = conn.execute(
                    f"SELECT id, content, metadata FROM {self._data_table} WHERE rowid = ?",
                    (row["rowid"],),
                ).fetchone()
                if not data_row:
                    continue
                meta = json.loads(data_row["metadata"])
                distance = row["distance"]
                hits.append(SearchHit(
                    id=data_row["id"],
                    content=data_row["content"],
                    score=1.0 - distance if distance is not None else 0.0,
                    source=meta.get("source", ""),
                    metadata=meta,
                ))
            return hits
        finally:
            conn.close()

    def _matches_filter(self, metadata: dict[str, Any], where: dict[str, Any]) -> bool:
        if "$and" in where:
            return all(
                all(metadata.get(k) == v for k, v in cond.items())
                for cond in where["$and"]
            )
        return all(metadata.get(k) == v for k, v in where.items())

    async def asearch(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        effective_top_k = top_k * 3 if where else top_k
        hits = await asyncio.to_thread(
            self._sync_search, query_embedding, effective_top_k,
        )
        if where:
            hits = [h for h in hits if self._matches_filter(h.metadata, where)]
        return hits[:top_k]

    def delete_by_source(self, source: str) -> None:
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT rowid FROM {self._data_table} "
                f"WHERE json_extract(metadata, '$.source') = ?",
                (source,),
            ).fetchall()
            for row in rows:
                conn.execute(
                    f"DELETE FROM {self._vss_table} WHERE rowid = ?", (row["rowid"],)
                )
            conn.execute(
                f"DELETE FROM {self._data_table} "
                f"WHERE json_extract(metadata, '$.source') = ?",
                (source,),
            )
            conn.commit()
        finally:
            conn.close()

    def delete_by_source_and_strategy(self, source: str, strategy: str) -> None:
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT rowid FROM {self._data_table} "
                f"WHERE json_extract(metadata, '$.source') = ? "
                f"AND json_extract(metadata, '$.chunking_strategy') = ?",
                (source, strategy),
            ).fetchall()
            for row in rows:
                conn.execute(
                    f"DELETE FROM {self._vss_table} WHERE rowid = ?", (row["rowid"],)
                )
            conn.execute(
                f"DELETE FROM {self._data_table} "
                f"WHERE json_extract(metadata, '$.source') = ? "
                f"AND json_extract(metadata, '$.chunking_strategy') = ?",
                (source, strategy),
            )
            conn.commit()
        finally:
            conn.close()

    def count(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {self._data_table}").fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def clear(self) -> None:
        conn = self._connect()
        try:
            conn.execute(f"DELETE FROM {self._vss_table}")
            conn.execute(f"DELETE FROM {self._data_table}")
            conn.commit()
        finally:
            conn.close()

    def list_sources(self) -> list[str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT DISTINCT json_extract(metadata, '$.source') AS source "
                f"FROM {self._data_table} WHERE source IS NOT NULL ORDER BY source"
            ).fetchall()
            return [row[0] for row in rows]
        finally:
            conn.close()

    def get_all(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT id, content, metadata FROM {self._data_table}"
            ).fetchall()
            return [
                {"id": row[0], "content": row[1], "metadata": json.loads(row[2])}
                for row in rows
            ]
        finally:
            conn.close()

    def needs_reindex(self, source: str, checksum: str, strategy: str = "default") -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT json_extract(metadata, '$.document_checksum') "
                f"FROM {self._data_table} "
                f"WHERE json_extract(metadata, '$.source') = ? "
                f"AND json_extract(metadata, '$.chunking_strategy') = ? LIMIT 1",
                (source, strategy),
            ).fetchone()
            if row is None:
                return True
            return row[0] != checksum
        finally:
            conn.close()
