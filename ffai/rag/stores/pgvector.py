"""PostgreSQL + pgvector vector store backend."""

from __future__ import annotations

import json
import logging
from typing import Any

from ffai.rag.types import SearchHit

from .base import VectorStoreBase

logger = logging.getLogger(__name__)

try:
    import asyncpg  # type: ignore[reportMissingImports]
    import psycopg  # type: ignore[reportMissingImports]

    PGVECTOR_AVAILABLE = True
except ImportError:
    asyncpg = None  # type: ignore[assignment]
    psycopg = None  # type: ignore[assignment]
    PGVECTOR_AVAILABLE = False


def get_store_class() -> type[VectorStoreBase]:
    """Return the pgvector store class.

    Raises:
        ImportError: If ``psycopg`` and ``asyncpg`` are not installed.
    """
    if not PGVECTOR_AVAILABLE:
        raise ImportError(
            "pgvector backend requires psycopg and asyncpg. "
            "Install with: pip install psycopg asyncpg"
        )
    return PgVectorStore


class PgVectorStore(VectorStoreBase):
    """PostgreSQL + pgvector vector store backend.

    Stores embeddings in a PostgreSQL table with the pgvector extension.
    Uses psycopg for DDL/setup and asyncpg for async queries.

    Args:
        connection_string: PostgreSQL connection string.
        table_name: Table name for storing vectors.
        embedding_dim: Dimensionality of embedding vectors.
        host: Database host (alternative to connection_string).
        port: Database port.
        database: Database name.
        user: Database user.
        password: Database password.
    """

    def __init__(
        self,
        *,
        connection_string: str | None = None,
        table_name: str = "ffai_vectors",
        embedding_dim: int = 1024,
        host: str = "localhost",
        port: int = 5432,
        database: str = "ffai_test",
        user: str = "ffai",
        password: str = "ffai_dev",
    ) -> None:
        if not PGVECTOR_AVAILABLE:
            raise ImportError(
                "pgvector backend requires psycopg and asyncpg. "
                "Install with: pip install psycopg asyncpg"
            )

        self._connection_string = connection_string or (
            f"postgresql://{user}:{password}@{host}:{port}/{database}"
        )
        self._table_name = table_name
        self._embedding_dim = embedding_dim
        self._pool: Any = None

        self._setup_schema()

    def _setup_schema(self) -> None:
        with psycopg.connect(self._connection_string) as conn:  # type: ignore[union-attr]
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id        TEXT PRIMARY KEY,
                    content   TEXT NOT NULL,
                    embedding vector({self._embedding_dim}),
                    metadata  JSONB NOT NULL DEFAULT '{{}}'
                )
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_source
                ON {self._table_name} ((metadata->>'source'))
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table_name}_strategy
                ON {self._table_name} ((metadata->>'chunking_strategy'))
            """)

    @property
    def name(self) -> str:
        return "pgvector"

    async def _get_pool(self) -> Any:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(  # type: ignore[union-attr]
                self._connection_string, min_size=2, max_size=10,
            )
        return self._pool

    async def aadd(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> int:
        """Add documents via asyncpg with UPSERT semantics (ON CONFLICT DO UPDATE)."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.executemany(
                f"""
                INSERT INTO {self._table_name} (id, content, embedding, metadata)
                VALUES ($1, $2, $3::vector, $4::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                [
                    (id_, text, json.dumps(emb), json.dumps(meta))
                    for id_, text, emb, meta in zip(ids, texts, embeddings, metadatas)
                ],
            )
        logger.info(f"Added {len(ids)} chunks to {self._table_name}")
        return len(ids)

    async def asearch(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Search using pgvector cosine distance operator (``<=>``) with optional metadata filtering."""
        pool = await self._get_pool()
        where_clause, params = self._build_where(where)
        emb_str = json.dumps(query_embedding)
        param_idx = 1 + len(params)
        sql = f"""
            SELECT id, content, metadata,
                   1 - (embedding <=> ${param_idx}::vector) AS score
            FROM {self._table_name}
            {where_clause}
            ORDER BY embedding <=> ${param_idx}::vector
            LIMIT ${param_idx + 1}
        """
        rows = await pool.fetch(sql, *params, emb_str, top_k)
        return [
            SearchHit(
                id=row["id"],
                content=row["content"],
                score=float(row["score"]),
                source=json.loads(row["metadata"]).get("source", ""),
                metadata=json.loads(row["metadata"]),
            )
            for row in rows
        ]

    @staticmethod
    def _safe_key(key: str) -> str:
        if not key.replace("_", "").replace("-", "").replace(".", "").isalnum():
            raise ValueError(f"Invalid metadata key: {key!r}")
        return key

    def _build_where(self, where: dict[str, Any] | None) -> tuple[str, list[str]]:
        if not where:
            return "", []
        if "$and" in where:
            conditions = []
            params: list[str] = []
            for cond in where["$and"]:
                for k, v in cond.items():
                    conditions.append(f"metadata->>'{self._safe_key(k)}' = ${len(params) + 1}")
                    params.append(str(v))
            return f"WHERE {' AND '.join(conditions)}", params
        conditions = []
        params = []
        for k, v in where.items():
            conditions.append(f"metadata->>'{self._safe_key(k)}' = ${len(params) + 1}")
            params.append(str(v))
        return f"WHERE {' AND '.join(conditions)}", params

    def delete_by_source(self, source: str) -> None:
        """Delete all rows matching ``source`` via a synchronous psycopg connection."""
        with psycopg.connect(self._connection_string) as conn:  # type: ignore[union-attr]
            conn.execute(
                f"DELETE FROM {self._table_name} WHERE metadata->>'source' = %s",
                (source,),
            )
        logger.info(f"Deleted chunks for source: {source}")

    def delete_by_source_and_strategy(self, source: str, strategy: str) -> None:
        """Delete rows matching both ``source`` and ``chunking_strategy`` via psycopg."""
        with psycopg.connect(self._connection_string) as conn:  # type: ignore[union-attr]
            conn.execute(
                f"DELETE FROM {self._table_name} "
                f"WHERE metadata->>'source' = %s AND metadata->>'chunking_strategy' = %s",
                (source, strategy),
            )

    def count(self) -> int:
        """Return the total number of rows in the vector table."""
        with psycopg.connect(self._connection_string) as conn:  # type: ignore[union-attr]
            result = conn.execute(f"SELECT COUNT(*) FROM {self._table_name}").fetchone()
            return result[0] if result else 0

    def clear(self) -> None:
        """Truncate all rows from the vector table."""
        with psycopg.connect(self._connection_string) as conn:  # type: ignore[union-attr]
            conn.execute(f"TRUNCATE {self._table_name}")

    def list_sources(self) -> list[str]:
        """Return a sorted list of distinct source names from the metadata JSONB column."""
        with psycopg.connect(self._connection_string) as conn:  # type: ignore[union-attr]
            rows = conn.execute(
                f"SELECT DISTINCT metadata->>'source' AS source "
                f"FROM {self._table_name} ORDER BY source"
            ).fetchall()
            return [row[0] for row in rows if row[0] is not None]

    def get_all(self) -> list[dict[str, Any]]:
        """Return all rows as dicts with ``id``, ``content``, ``metadata`` keys."""
        with psycopg.connect(self._connection_string) as conn:  # type: ignore[union-attr]
            rows = conn.execute(
                f"SELECT id, content, metadata FROM {self._table_name}"
            ).fetchall()
            return [
                {"id": row[0], "content": row[1], "metadata": json.loads(row[2])}
                for row in rows
            ]

    def needs_reindex(self, source: str, checksum: str, strategy: str = "default") -> bool:
        """Check whether ``source`` needs re-indexing by comparing stored checksum in JSONB."""
        with psycopg.connect(self._connection_string) as conn:  # type: ignore[union-attr]
            row = conn.execute(
                f"SELECT metadata->>'document_checksum' FROM {self._table_name} "
                f"WHERE metadata->>'source' = %s AND metadata->>'chunking_strategy' = %s LIMIT 1",
                (source, strategy),
            ).fetchone()
            if row is None:
                return True
            return row[0] != checksum
