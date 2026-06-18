# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os

import polars as pl
import pytest

from ffai.core.memory import TurnVectorStore, load_store, persist_store


def _turn(text: str = "response") -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "timestamp": 1718700000.0,
    }


def _populated_store() -> TurnVectorStore:
    store = TurnVectorStore()
    store.add(
        text="first turn text",
        embedding=[0.1, 0.2, 0.3],
        turn=_turn("first"),
        metadata={"prompt_name": "q1", "session_id": "s1"},
    )
    store.add(
        text="second turn text",
        embedding=[0.4, 0.5, 0.6],
        turn=_turn("second"),
        metadata={"prompt_name": "q2"},
    )
    return store


class TestPersistStore:
    def test_writes_parquet_file(self, tmp_path):
        store = _populated_store()
        path = str(tmp_path / "turns.parquet")
        persist_store(store, path)
        assert os.path.exists(path)

    def test_creates_parent_directories(self, tmp_path):
        store = _populated_store()
        path = str(tmp_path / "nested" / "deep" / "turns.parquet")
        persist_store(store, path)
        assert os.path.exists(path)

    def test_schema_columns_present(self, tmp_path):
        store = _populated_store()
        path = str(tmp_path / "turns.parquet")
        persist_store(store, path)
        df = pl.read_parquet(path)
        assert set(df.columns) == {
            "text",
            "embedding",
            "turn",
            "metadata",
            "_schema_version",
        }

    def test_schema_dtypes(self, tmp_path):
        store = _populated_store()
        path = str(tmp_path / "turns.parquet")
        persist_store(store, path)
        df = pl.read_parquet(path)
        assert df.schema["text"] == pl.String
        assert df.schema["embedding"] == pl.List(pl.Float64)
        assert df.schema["turn"] == pl.String
        assert df.schema["metadata"] == pl.String
        assert df.schema["_schema_version"] == pl.Int8

    def test_row_count_matches_store(self, tmp_path):
        store = _populated_store()
        path = str(tmp_path / "turns.parquet")
        persist_store(store, path)
        df = pl.read_parquet(path)
        assert df.height == store.count()

    def test_schema_version_is_one(self, tmp_path):
        store = _populated_store()
        path = str(tmp_path / "turns.parquet")
        persist_store(store, path)
        df = pl.read_parquet(path)
        versions = df.select("_schema_version").unique().to_series().to_list()
        assert versions == [1]

    def test_empty_store_writes_zero_row_file(self, tmp_path):
        store = TurnVectorStore()
        path = str(tmp_path / "empty.parquet")
        persist_store(store, path)
        df = pl.read_parquet(path)
        assert df.height == 0
        assert set(df.columns) == {
            "text",
            "embedding",
            "turn",
            "metadata",
            "_schema_version",
        }

    def test_overwrites_existing_file(self, tmp_path):
        path = str(tmp_path / "turns.parquet")
        persist_store(_populated_store(), path)
        new_store = TurnVectorStore()
        new_store.add("only one", [1.0], _turn(), {})
        persist_store(new_store, path)
        df = pl.read_parquet(path)
        assert df.height == 1


class TestLoadStore:
    def test_round_trips_all_fields(self, tmp_path):
        original = _populated_store()
        path = str(tmp_path / "turns.parquet")
        persist_store(original, path)

        loaded = load_store(path)
        assert loaded.count() == original.count()

        original_entries = list(original.iter_entries())
        loaded_entries = list(loaded.iter_entries())
        for orig, loaded_entry in zip(original_entries, loaded_entries, strict=True):
            assert loaded_entry.text == orig.text
            assert loaded_entry.embedding == orig.embedding
            assert loaded_entry.turn == orig.turn
            assert loaded_entry.metadata == orig.metadata

    def test_preserves_insertion_order(self, tmp_path):
        original = _populated_store()
        path = str(tmp_path / "turns.parquet")
        persist_store(original, path)
        loaded = load_store(path)

        original_texts = [e.text for e in original.iter_entries()]
        loaded_texts = [e.text for e in loaded.iter_entries()]
        assert loaded_texts == original_texts

    def test_load_empty_store_round_trips(self, tmp_path):
        original = TurnVectorStore()
        path = str(tmp_path / "empty.parquet")
        persist_store(original, path)
        loaded = load_store(path)
        assert loaded.count() == 0
        assert list(loaded.iter_entries()) == []

    def test_missing_file_raises_filenotfounderror(self, tmp_path):
        path = str(tmp_path / "does_not_exist.parquet")
        with pytest.raises(FileNotFoundError, match="Memory persistence file not found"):
            load_store(path)

    def test_unsupported_schema_version_raises_valueerror(self, tmp_path):
        path = str(tmp_path / "bad_version.parquet")
        df = pl.DataFrame(
            {
                "text": ["x"],
                "embedding": [[0.1]],
                "turn": ["{}"],
                "metadata": ["{}"],
                "_schema_version": [99],
            },
            schema={
                "text": pl.String,
                "embedding": pl.List(pl.Float64),
                "turn": pl.String,
                "metadata": pl.String,
                "_schema_version": pl.Int8,
            },
        )
        df.write_parquet(path)
        with pytest.raises(ValueError, match="Unsupported memory persistence schema"):
            load_store(path)

    def test_missing_schema_version_column_raises_valueerror(self, tmp_path):
        path = str(tmp_path / "no_version.parquet")
        df = pl.DataFrame(
            {
                "text": ["x"],
                "embedding": [[0.1]],
                "turn": ["{}"],
                "metadata": ["{}"],
            },
            schema={
                "text": pl.String,
                "embedding": pl.List(pl.Float64),
                "turn": pl.String,
                "metadata": pl.String,
            },
        )
        df.write_parquet(path)
        with pytest.raises(ValueError, match="missing '_schema_version'"):
            load_store(path)


class TestJsonRoundTrip:
    def test_nested_metadata_preserved(self, tmp_path):
        store = TurnVectorStore()
        nested_meta = {"user": {"profile": {"name": "alice"}}, "tags": ["a", "b"]}
        store.add("text", [0.1, 0.2], _turn(), nested_meta)
        path = str(tmp_path / "nested.parquet")
        persist_store(store, path)
        loaded = load_store(path)
        entries = list(loaded.iter_entries())
        assert entries[0].metadata == nested_meta

    def test_nested_turn_content_preserved(self, tmp_path):
        store = TurnVectorStore()
        turn = {
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
            "metadata": {"prompt_name": "greet"},
            "timestamp": 1718700000.0,
        }
        store.add("hello", [0.5], turn, {})
        path = str(tmp_path / "nested_turn.parquet")
        persist_store(store, path)
        loaded = load_store(path)
        entries = list(loaded.iter_entries())
        assert entries[0].turn == turn


class TestEmbeddingPrecision:
    def test_floats_round_trip_within_tolerance(self, tmp_path):
        store = TurnVectorStore()
        embedding = [0.123456789, -0.987654321, 0.5, 1e-10]
        store.add("text", embedding, _turn(), {})
        path = str(tmp_path / "precision.parquet")
        persist_store(store, path)
        loaded = load_store(path)
        entries = list(loaded.iter_entries())
        for orig, loaded_val in zip(embedding, entries[0].embedding, strict=True):
            assert loaded_val == pytest.approx(orig, rel=1e-9, abs=1e-12)


class TestEndToEndWithSearch:
    def test_loaded_store_supports_search(self, tmp_path):
        from conftest import FakeEmbeddings

        from ffai.core.memory import Memory

        original_memory = Memory(FakeEmbeddings(dim=4))
        original_memory.index_turn_text(
            text="alpha", turn=_turn("alpha-response"), metadata={"prompt_name": "a"}
        )
        original_memory.index_turn_text(
            text="beta", turn=_turn("beta-response"), metadata={"prompt_name": "b"}
        )

        path = str(tmp_path / "e2e.parquet")
        persist_store(original_memory.store, path)

        reloaded_memory = Memory(FakeEmbeddings(dim=4), store=load_store(path))
        hits = reloaded_memory.search("alpha")
        assert len(hits) == 2
        assert hits[0].text == "alpha"
        assert hits[0].metadata == {"prompt_name": "a"}

    def test_loaded_store_metadata_json_parses(self, tmp_path):
        store = _populated_store()
        path = str(tmp_path / "meta.parquet")
        persist_store(store, path)
        loaded = load_store(path)
        for entry in loaded.iter_entries():
            assert isinstance(json.loads(json.dumps(entry.metadata)), dict)
