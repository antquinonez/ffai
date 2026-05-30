from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_psycopg():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchone.return_value = None
    mock_cursor.fetchall.return_value = []

    mock_psycopg = MagicMock()
    mock_psycopg.connect.return_value = mock_conn

    mock_asyncpg = MagicMock()

    return mock_psycopg, mock_asyncpg, mock_conn


@contextmanager
def _patch_pg(mock_psycopg, mock_asyncpg):
    with (
        patch("ffai.rag.stores.pgvector.PGVECTOR_AVAILABLE", True),
        patch("ffai.rag.stores.pgvector.psycopg", mock_psycopg),
        patch("ffai.rag.stores.pgvector.asyncpg", mock_asyncpg),
    ):
        yield


class TestPgVectorStoreInit:
    def test_creates_store_with_connection_string(self):
        from ffai.rag.stores.pgvector import PgVectorStore
        mock_pg, mock_async, mock_conn = _make_mock_psycopg()
        with _patch_pg(mock_pg, mock_async):
            store = PgVectorStore(
                connection_string="postgresql://u:p@h:5432/db",
                table_name="test_vec",
                embedding_dim=512,
            )
            assert store.name == "pgvector"
            assert store._table_name == "test_vec"

    def test_creates_store_with_individual_params(self):
        from ffai.rag.stores.pgvector import PgVectorStore
        mock_pg, mock_async, mock_conn = _make_mock_psycopg()
        with _patch_pg(mock_pg, mock_async):
            store = PgVectorStore(
                host="localhost",
                port=5432,
                database="testdb",
                user="testuser",
                password="testpass",
            )
            assert "testuser:testpass@localhost:5432/testdb" in store._connection_string

    def test_raises_when_deps_missing(self):
        from ffai.rag.stores.pgvector import PgVectorStore
        with patch("ffai.rag.stores.pgvector.PGVECTOR_AVAILABLE", False):
            with pytest.raises(ImportError, match="psycopg and asyncpg"):
                PgVectorStore(connection_string="postgresql://x")

    def test_schema_creates_extension_and_table(self):
        from ffai.rag.stores.pgvector import PgVectorStore
        mock_pg, mock_async, mock_conn = _make_mock_psycopg()
        with _patch_pg(mock_pg, mock_async):
            PgVectorStore(connection_string="postgresql://x")
        calls = [str(c) for c in mock_conn.execute.call_args_list]
        assert any("CREATE EXTENSION" in c for c in calls)
        assert any("CREATE TABLE" in c for c in calls)


class TestPgVectorStoreOperations:
    def _make_store(self):
        from ffai.rag.stores.pgvector import PgVectorStore
        mock_pg, mock_async, mock_conn = _make_mock_psycopg()
        ctx = _patch_pg(mock_pg, mock_async)
        ctx.__enter__()
        store = PgVectorStore(connection_string="postgresql://x")
        return store, mock_conn, ctx

    def test_count_returns_zero(self):
        store, mock_conn, ctx = self._make_store()
        try:
            mock_conn.execute.return_value.fetchone.return_value = (0,)
            assert store.count() == 0
        finally:
            ctx.__exit__(None, None, None)

    def test_count_returns_value(self):
        store, mock_conn, ctx = self._make_store()
        try:
            mock_conn.execute.return_value.fetchone.return_value = (42,)
            assert store.count() == 42
        finally:
            ctx.__exit__(None, None, None)

    def test_delete_by_source(self):
        store, mock_conn, ctx = self._make_store()
        try:
            store.delete_by_source("doc1")
            mock_conn.execute.assert_called()
            call_args = str(mock_conn.execute.call_args)
            assert "DELETE" in call_args
        finally:
            ctx.__exit__(None, None, None)

    def test_delete_by_source_and_strategy(self):
        store, mock_conn, ctx = self._make_store()
        try:
            store.delete_by_source_and_strategy("doc1", "recursive")
            mock_conn.execute.assert_called()
        finally:
            ctx.__exit__(None, None, None)

    def test_clear_truncates_table(self):
        store, mock_conn, ctx = self._make_store()
        try:
            store.clear()
            call_args = str(mock_conn.execute.call_args)
            assert "TRUNCATE" in call_args
        finally:
            ctx.__exit__(None, None, None)

    def test_list_sources_returns_sorted(self):
        store, mock_conn, ctx = self._make_store()
        try:
            mock_conn.execute.return_value.fetchall.return_value = [("a",), ("b",), ("c",)]
            assert store.list_sources() == ["a", "b", "c"]
        finally:
            ctx.__exit__(None, None, None)

    def test_get_all_returns_dicts(self):
        store, mock_conn, ctx = self._make_store()
        try:
            mock_conn.execute.return_value.fetchall.return_value = [
                ("id1", "text1", '{"source": "doc1"}'),
            ]
            result = store.get_all()
            assert len(result) == 1
            assert result[0]["id"] == "id1"
            assert result[0]["content"] == "text1"
            assert result[0]["metadata"]["source"] == "doc1"
        finally:
            ctx.__exit__(None, None, None)

    def test_needs_reindex_true_when_no_row(self):
        store, mock_conn, ctx = self._make_store()
        try:
            mock_conn.execute.return_value.fetchone.return_value = None
            assert store.needs_reindex("doc1", "abc") is True
        finally:
            ctx.__exit__(None, None, None)

    def test_needs_reindex_false_when_checksum_matches(self):
        store, mock_conn, ctx = self._make_store()
        try:
            mock_conn.execute.return_value.fetchone.return_value = ("abc",)
            assert store.needs_reindex("doc1", "abc") is False
        finally:
            ctx.__exit__(None, None, None)

    def test_needs_reindex_true_when_checksum_differs(self):
        store, mock_conn, ctx = self._make_store()
        try:
            mock_conn.execute.return_value.fetchone.return_value = ("old",)
            assert store.needs_reindex("doc1", "abc") is True
        finally:
            ctx.__exit__(None, None, None)


class TestPgVectorBuildWhere:
    def test_empty_where(self):
        from ffai.rag.stores.pgvector import PgVectorStore
        mock_pg, mock_async, _ = _make_mock_psycopg()
        with _patch_pg(mock_pg, mock_async):
            store = PgVectorStore(connection_string="postgresql://x")
        clause, params = store._build_where(None)
        assert clause == ""
        assert params == []

    def test_simple_filter(self):
        from ffai.rag.stores.pgvector import PgVectorStore
        mock_pg, mock_async, _ = _make_mock_psycopg()
        with _patch_pg(mock_pg, mock_async):
            store = PgVectorStore(connection_string="postgresql://x")
        clause, params = store._build_where({"source": "doc1"})
        assert "WHERE" in clause
        assert "metadata->>'source'" in clause
        assert params == ["doc1"]

    def test_compound_filter(self):
        from ffai.rag.stores.pgvector import PgVectorStore
        mock_pg, mock_async, _ = _make_mock_psycopg()
        with _patch_pg(mock_pg, mock_async):
            store = PgVectorStore(connection_string="postgresql://x")
        clause, params = store._build_where(
            {"$and": [{"source": "doc1"}, {"chunking_strategy": "recursive"}]}
        )
        assert "AND" in clause
        assert len(params) == 2

    def test_sql_injection_in_key_raises(self):
        from ffai.rag.stores.pgvector import PgVectorStore
        mock_pg, mock_async, _ = _make_mock_psycopg()
        with _patch_pg(mock_pg, mock_async):
            store = PgVectorStore(connection_string="postgresql://x")
        with pytest.raises(ValueError, match="Invalid metadata key"):
            store._build_where({"'); DROP TABLE t; --": "value"})

    def test_valid_keys_with_underscores_and_dots(self):
        from ffai.rag.stores.pgvector import PgVectorStore
        mock_pg, mock_async, _ = _make_mock_psycopg()
        with _patch_pg(mock_pg, mock_async):
            store = PgVectorStore(connection_string="postgresql://x")
        clause, params = store._build_where({"chunking_strategy": "recursive"})
        assert "metadata->>'chunking_strategy'" in clause
