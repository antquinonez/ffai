from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_sqlite_vss():
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.execute.return_value = mock_cursor
    mock_conn.commit.return_value = None
    mock_conn.close.return_value = None
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    mock_vss_module = MagicMock()
    mock_vss_module.load.return_value = None

    return mock_vss_module, mock_conn


@contextmanager
def _patch_sqlite(mock_vss, mock_conn):
    with (
        patch("ffai.rag.stores.sqlite_vss.SQLITE_VSS_AVAILABLE", True),
        patch("ffai.rag.stores.sqlite_vss.sqlite_vss", mock_vss),
        patch("sqlite3.connect", return_value=mock_conn),
    ):
        yield


class TestSQLiteVssStoreInit:
    def test_creates_store(self):
        from ffai.rag.stores.sqlite_vss import SQLiteVssStore
        mock_vss, mock_conn = _make_mock_sqlite_vss()
        with _patch_sqlite(mock_vss, mock_conn):
            store = SQLiteVssStore(db_path="/tmp/test_vss.db")
            assert store.name == "sqlite_vss"
            assert store._data_table == "ffai_vectors_data"

    def test_raises_when_deps_missing(self):
        from ffai.rag.stores.sqlite_vss import SQLiteVssStore
        with patch("ffai.rag.stores.sqlite_vss.SQLITE_VSS_AVAILABLE", False):
            with pytest.raises(ImportError, match="sqlite-vss"):
                SQLiteVssStore()


class TestSQLiteVssStoreOperations:
    def _make_store(self):
        from ffai.rag.stores.sqlite_vss import SQLiteVssStore
        mock_vss, mock_conn = _make_mock_sqlite_vss()
        ctx = _patch_sqlite(mock_vss, mock_conn)
        ctx.__enter__()
        store = SQLiteVssStore(db_path="/tmp/test_vss.db")
        return store, mock_conn, ctx

    def test_count_returns_value(self):
        store, mock_conn, ctx = self._make_store()
        try:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = (5,)
            mock_conn.execute.return_value = mock_cursor
            assert store.count() == 5
        finally:
            ctx.__exit__(None, None, None)

    def test_list_sources(self):
        store, mock_conn, ctx = self._make_store()
        try:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [("a",), ("b",)]
            mock_conn.execute.return_value = mock_cursor
            assert store.list_sources() == ["a", "b"]
        finally:
            ctx.__exit__(None, None, None)

    def test_get_all_returns_dicts(self):
        store, mock_conn, ctx = self._make_store()
        try:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [
                ("id1", "text1", json.dumps({"source": "doc1"})),
            ]
            mock_conn.execute.return_value = mock_cursor
            result = store.get_all()
            assert len(result) == 1
            assert result[0]["id"] == "id1"
            assert result[0]["metadata"]["source"] == "doc1"
        finally:
            ctx.__exit__(None, None, None)

    def test_needs_reindex_true_when_no_row(self):
        store, mock_conn, ctx = self._make_store()
        try:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = None
            mock_conn.execute.return_value = mock_cursor
            assert store.needs_reindex("doc1", "abc") is True
        finally:
            ctx.__exit__(None, None, None)

    def test_needs_reindex_false_when_checksum_matches(self):
        store, mock_conn, ctx = self._make_store()
        try:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = ("abc",)
            mock_conn.execute.return_value = mock_cursor
            assert store.needs_reindex("doc1", "abc") is False
        finally:
            ctx.__exit__(None, None, None)

    def test_needs_reindex_true_when_checksum_differs(self):
        store, mock_conn, ctx = self._make_store()
        try:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = ("old",)
            mock_conn.execute.return_value = mock_cursor
            assert store.needs_reindex("doc1", "abc") is True
        finally:
            ctx.__exit__(None, None, None)

    def test_clear_deletes_all(self):
        store, mock_conn, ctx = self._make_store()
        try:
            store.clear()
            execute_calls = [str(c) for c in mock_conn.execute.call_args_list]
            assert any("DELETE" in c for c in execute_calls)
        finally:
            ctx.__exit__(None, None, None)

    def test_delete_by_source(self):
        store, mock_conn, ctx = self._make_store()
        try:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            mock_conn.execute.return_value = mock_cursor
            store.delete_by_source("doc1")
            execute_calls = [str(c) for c in mock_conn.execute.call_args_list]
            assert any("DELETE" in c for c in execute_calls)
        finally:
            ctx.__exit__(None, None, None)


class TestSQLiteVssMatchesFilter:
    def _make_store(self):
        from ffai.rag.stores.sqlite_vss import SQLiteVssStore
        mock_vss, mock_conn = _make_mock_sqlite_vss()
        with _patch_sqlite(mock_vss, mock_conn):
            return SQLiteVssStore(db_path="/tmp/test_vss.db")

    def test_simple_filter_match(self):
        store = self._make_store()
        assert store._matches_filter({"source": "doc1"}, {"source": "doc1"}) is True

    def test_simple_filter_no_match(self):
        store = self._make_store()
        assert store._matches_filter({"source": "doc1"}, {"source": "doc2"}) is False

    def test_compound_filter(self):
        store = self._make_store()
        where = {"$and": [{"source": "doc1"}, {"chunking_strategy": "recursive"}]}
        meta = {"source": "doc1", "chunking_strategy": "recursive"}
        assert store._matches_filter(meta, where) is True

    def test_compound_filter_partial_no_match(self):
        store = self._make_store()
        where = {"$and": [{"source": "doc1"}, {"chunking_strategy": "recursive"}]}
        meta = {"source": "doc1", "chunking_strategy": "character"}
        assert store._matches_filter(meta, where) is False
