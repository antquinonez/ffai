# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
from conftest import FakeEmbeddings

from ffai.core.memory import Memory, TurnHit, TurnVectorStore


def _turn(text: str = "response text") -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "timestamp": 1718700000.0,
    }


class TestMemoryConstruction:
    def test_default_store_is_fresh_turnvectorstore(self, fake_embeddings):
        memory = Memory(fake_embeddings)
        assert isinstance(memory.store, TurnVectorStore)
        assert memory.store.count() == 0

    def test_explicit_store_is_used(self, fake_embeddings):
        store = TurnVectorStore()
        store.add("preexisting", [0.1] * 8, _turn(), {})
        memory = Memory(fake_embeddings, store=store)
        assert memory.store is store
        assert memory.count() == 1

    def test_store_is_reassignable(self, fake_embeddings):
        memory = Memory(fake_embeddings)
        memory.index_turn_text(text="first", turn=_turn(), metadata={})
        assert memory.count() == 1
        new_store = TurnVectorStore()
        memory.store = new_store
        assert memory.count() == 0
        assert memory.store is new_store


class TestIndexTurn:
    def test_embeds_text_from_turn_content(self, fake_embeddings):
        memory = Memory(fake_embeddings)
        memory.index_turn(_turn("hello world"), metadata={"k": "v"})
        assert memory.count() == 1
        entries = list(memory.store.iter_entries())
        assert entries[0].text == "hello world"

    def test_metadata_attached_to_entry(self, fake_embeddings):
        memory = Memory(fake_embeddings)
        memory.index_turn(_turn("hi"), metadata={"prompt_name": "greet"})
        entries = list(memory.store.iter_entries())
        assert entries[0].metadata == {"prompt_name": "greet"}

    def test_metadata_none_defaults_to_empty_dict(self, fake_embeddings):
        memory = Memory(fake_embeddings)
        memory.index_turn(_turn("hi"), metadata=None)
        entries = list(memory.store.iter_entries())
        assert entries[0].metadata == {}

    def test_returns_incrementing_index(self, fake_embeddings):
        memory = Memory(fake_embeddings)
        assert memory.index_turn(_turn("a")) == 0
        assert memory.index_turn(_turn("b")) == 1


class TestIndexTurnText:
    def test_embeds_supplied_text_not_turn_content(self, fake_embeddings):
        memory = Memory(fake_embeddings)
        memory.index_turn_text(
            text="prompt\nresponse",
            turn=_turn("response"),
            metadata={},
        )
        entries = list(memory.store.iter_entries())
        assert entries[0].text == "prompt\nresponse"
        assert entries[0].turn["content"][0]["text"] == "response"

    def test_metadata_attached(self, fake_embeddings):
        memory = Memory(fake_embeddings)
        memory.index_turn_text(text="x", turn=_turn(), metadata={"prompt_name": "p"})
        entries = list(memory.store.iter_entries())
        assert entries[0].metadata == {"prompt_name": "p"}

    def test_returns_index(self, fake_embeddings):
        memory = Memory(fake_embeddings)
        idx = memory.index_turn_text(text="x", turn=_turn(), metadata={})
        assert idx == 0


class TestSearch:
    def test_empty_memory_returns_empty_list(self, fake_embeddings):
        memory = Memory(fake_embeddings)
        assert memory.search("anything") == []

    def test_returns_turnhit_instances(self, fake_embeddings):
        memory = Memory(fake_embeddings)
        memory.index_turn_text(text="hello", turn=_turn(), metadata={})
        hits = memory.search("hello")
        assert len(hits) == 1
        assert isinstance(hits[0], TurnHit)

    def test_identical_query_ranks_top(self, fake_embeddings):
        memory = Memory(fake_embeddings)
        memory.index_turn_text(text="alpha", turn=_turn(), metadata={})
        memory.index_turn_text(text="beta", turn=_turn(), metadata={})
        hits = memory.search("alpha")
        assert hits[0].text == "alpha"
        assert hits[0].score == pytest.approx(1.0)

    def test_top_k_limits_results(self, fake_embeddings):
        memory = Memory(fake_embeddings)
        for i in range(5):
            memory.index_turn_text(text=f"item_{i}", turn=_turn(), metadata={})
        hits = memory.search("item_0", top_k=2)
        assert len(hits) == 2

    def test_threshold_excludes_low_scores(self, fake_embeddings):
        memory = Memory(fake_embeddings)
        memory.index_turn_text(text="alpha", turn=_turn(), metadata={})
        memory.index_turn_text(text="completely_different", turn=_turn(), metadata={})
        hits = memory.search("alpha", threshold=0.99)
        assert all(h.score >= 0.99 for h in hits)
        assert len(hits) == 1
        assert hits[0].text == "alpha"

    def test_threshold_above_all_scores_returns_empty(self, fake_embeddings):
        memory = Memory(fake_embeddings)
        memory.index_turn_text(text="alpha", turn=_turn(), metadata={})
        assert memory.search("alpha", threshold=2.0) == []


class TestAsyncIndexTurn:
    def test_aindex_turn_matches_sync(self, fake_embeddings):
        async_memory = Memory(FakeEmbeddings(dim=8))
        sync_memory = Memory(FakeEmbeddings(dim=8))

        import asyncio

        asyncio.run(async_memory.aindex_turn(_turn("hello"), metadata={"k": "v"}))
        sync_memory.index_turn(_turn("hello"), metadata={"k": "v"})

        async_entries = list(async_memory.store.iter_entries())
        sync_entries = list(sync_memory.store.iter_entries())
        assert async_entries[0].text == sync_entries[0].text
        assert async_entries[0].embedding == sync_entries[0].embedding
        assert async_entries[0].metadata == sync_entries[0].metadata
        assert async_memory.count() == sync_memory.count()

    def test_aindex_turn_text_matches_sync(self, fake_embeddings):
        async_memory = Memory(FakeEmbeddings(dim=8))
        sync_memory = Memory(FakeEmbeddings(dim=8))

        import asyncio

        text = "prompt\nresponse"
        asyncio.run(
            async_memory.aindex_turn_text(text=text, turn=_turn("response"), metadata={"k": "v"})
        )
        sync_memory.index_turn_text(text=text, turn=_turn("response"), metadata={"k": "v"})

        async_entries = list(async_memory.store.iter_entries())
        sync_entries = list(sync_memory.store.iter_entries())
        assert async_entries[0].text == sync_entries[0].text
        assert async_entries[0].embedding == sync_entries[0].embedding


class TestAsyncSearch:
    def test_asearch_returns_same_results_as_sync(self, fake_embeddings):
        memory = Memory(FakeEmbeddings(dim=8))
        memory.index_turn_text(text="alpha", turn=_turn(), metadata={})
        memory.index_turn_text(text="beta", turn=_turn(), metadata={})

        import asyncio

        sync_hits = memory.search("alpha")
        async_hits = asyncio.run(memory.asearch("alpha"))

        assert len(sync_hits) == len(async_hits)
        for sync_hit, async_hit in zip(sync_hits, async_hits, strict=True):
            assert sync_hit.text == async_hit.text
            assert sync_hit.score == pytest.approx(async_hit.score)


class TestReindex:
    def test_preserves_turns_and_metadata(self, fake_embeddings):
        memory = Memory(fake_embeddings)
        memory.index_turn_text(text="alpha", turn=_turn("alpha-response"), metadata={"k": "v1"})
        memory.index_turn_text(text="beta", turn=_turn("beta-response"), metadata={"k": "v2"})

        original_entries = list(memory.store.iter_entries())
        new_embeddings = FakeEmbeddings(dim=16)
        memory.reindex(new_embeddings)

        reindexed_entries = list(memory.store.iter_entries())
        assert len(reindexed_entries) == 2
        assert reindexed_entries[0].text == original_entries[0].text
        assert reindexed_entries[0].turn == original_entries[0].turn
        assert reindexed_entries[0].metadata == original_entries[0].metadata
        assert reindexed_entries[1].text == original_entries[1].text

    def test_replaces_embeddings_with_new_dim(self, fake_embeddings):
        memory = Memory(FakeEmbeddings(dim=8))
        memory.index_turn_text(text="alpha", turn=_turn(), metadata={})

        original_entries = list(memory.store.iter_entries())
        assert len(original_entries[0].embedding) == 8

        memory.reindex(FakeEmbeddings(dim=16))
        reindexed_entries = list(memory.store.iter_entries())
        assert len(reindexed_entries[0].embedding) == 16

    def test_search_uses_new_model_after_reindex(self, fake_embeddings):
        memory = Memory(FakeEmbeddings(dim=8))
        memory.index_turn_text(text="alpha", turn=_turn(), metadata={})

        memory.reindex(FakeEmbeddings(dim=16))
        hits = memory.search("alpha")
        assert len(hits) == 1
        assert hits[0].score == pytest.approx(1.0)

    def test_reindex_empty_store_is_noop(self, fake_embeddings):
        memory = Memory(fake_embeddings)
        new_embeddings = FakeEmbeddings(dim=16)
        memory.reindex(new_embeddings)
        assert memory.count() == 0


class TestCountAndClear:
    def test_count_reflects_indexing(self, fake_embeddings):
        memory = Memory(fake_embeddings)
        assert memory.count() == 0
        memory.index_turn_text(text="a", turn=_turn(), metadata={})
        assert memory.count() == 1
        memory.index_turn_text(text="b", turn=_turn(), metadata={})
        assert memory.count() == 2

    def test_clear_empties_store(self, fake_embeddings):
        memory = Memory(fake_embeddings)
        memory.index_turn_text(text="a", turn=_turn(), metadata={})
        memory.index_turn_text(text="b", turn=_turn(), metadata={})
        memory.clear()
        assert memory.count() == 0
        assert memory.search("a") == []
