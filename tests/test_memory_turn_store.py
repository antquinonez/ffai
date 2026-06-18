# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from ffai.core.memory import Entry, TurnHit, TurnVectorStore
from ffai.core.memory.turn_store import cosine_similarity


def _turn(text: str, role: str = "assistant") -> dict:
    return {
        "role": role,
        "content": [{"type": "text", "text": text}],
        "timestamp": 1718700000.0,
    }


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_score_negative_one(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero_not_nan(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert cosine_similarity([1.0, 1.0], [0.0, 0.0]) == 0.0

    def test_similarity_is_symmetric(self):
        a = [0.3, 0.8, 0.1]
        b = [0.6, 0.1, 0.9]
        assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))


class TestAdd:
    def test_returns_incrementing_indices(self):
        store = TurnVectorStore()
        assert store.add("a", [0.1], _turn("a"), {}) == 0
        assert store.add("b", [0.2], _turn("b"), {}) == 1
        assert store.add("c", [0.3], _turn("c"), {}) == 2

    def test_metadata_none_defaults_to_empty_dict(self):
        store = TurnVectorStore()
        store.add("a", [0.1], _turn("a"), metadata=None)
        entries = list(store.iter_entries())
        assert entries[0].metadata == {}

    def test_add_copies_turn_dict_defensively(self):
        store = TurnVectorStore()
        original_turn = _turn("original")
        store.add("a", [0.1], original_turn, {})
        original_turn["content"][0]["text"] = "mutated"
        entries = list(store.iter_entries())
        assert entries[0].turn["content"][0]["text"] == "original"

    def test_add_copies_metadata_defensively(self):
        store = TurnVectorStore()
        original_meta = {"nested": {"k": "v"}}
        store.add("a", [0.1], _turn("a"), original_meta)
        original_meta["nested"]["k"] = "mutated"
        entries = list(store.iter_entries())
        assert entries[0].metadata["nested"]["k"] == "v"

    def test_add_copies_embedding_defensively(self):
        store = TurnVectorStore()
        emb = [0.1, 0.2, 0.3]
        store.add("a", emb, _turn("a"), {})
        emb[0] = 99.0
        entries = list(store.iter_entries())
        assert entries[0].embedding == [0.1, 0.2, 0.3]


class TestCount:
    def test_empty_store_has_zero_count(self):
        assert TurnVectorStore().count() == 0

    def test_count_reflects_adds(self):
        store = TurnVectorStore()
        store.add("a", [0.1], _turn("a"), {})
        store.add("b", [0.2], _turn("b"), {})
        assert store.count() == 2

    def test_count_after_clear_is_zero(self):
        store = TurnVectorStore()
        store.add("a", [0.1], _turn("a"), {})
        store.clear()
        assert store.count() == 0


class TestClear:
    def test_clear_empties_store(self):
        store = TurnVectorStore()
        store.add("a", [0.1], _turn("a"), {})
        store.add("b", [0.2], _turn("b"), {})
        store.clear()
        assert store.count() == 0
        assert list(store.iter_entries()) == []
        assert store.search([0.1]) == []


class TestSearchRanking:
    def test_empty_store_returns_empty_list(self):
        store = TurnVectorStore()
        assert store.search([0.1, 0.2]) == []

    def test_single_hit_returns_self(self):
        store = TurnVectorStore()
        store.add("alpha", [1.0, 0.0], _turn("alpha"), {})
        hits = store.search([1.0, 0.0])
        assert len(hits) == 1
        assert hits[0].text == "alpha"
        assert hits[0].score == pytest.approx(1.0)

    def test_results_sorted_by_score_descending(self):
        store = TurnVectorStore()
        store.add("low", [0.0, 1.0], _turn("low"), {})
        store.add("high", [1.0, 0.0], _turn("high"), {})
        store.add("mid", [1.0, 1.0], _turn("mid"), {})
        hits = store.search([1.0, 0.0])
        assert [h.text for h in hits] == ["high", "mid", "low"]
        assert hits[0].score > hits[1].score > hits[2].score

    def test_top_k_limits_result_count(self):
        store = TurnVectorStore()
        for i in range(5):
            store.add(f"item_{i}", [float(i) * 0.1, 0.0], _turn(f"item_{i}"), {})
        hits = store.search([1.0, 0.0], top_k=3)
        assert len(hits) == 3

    def test_top_k_zero_returns_empty(self):
        store = TurnVectorStore()
        store.add("a", [1.0], _turn("a"), {})
        assert store.search([1.0], top_k=0) == []

    def test_top_k_negative_returns_empty(self):
        store = TurnVectorStore()
        store.add("a", [1.0], _turn("a"), {})
        assert store.search([1.0], top_k=-1) == []


class TestSearchThreshold:
    def test_threshold_excludes_low_score_hits(self):
        store = TurnVectorStore()
        store.add("relevant", [1.0, 0.0], _turn("relevant"), {})
        store.add("irrelevant", [0.0, 1.0], _turn("irrelevant"), {})
        hits = store.search([1.0, 0.0], threshold=0.5)
        assert len(hits) == 1
        assert hits[0].text == "relevant"

    def test_threshold_zero_includes_orthogonal(self):
        store = TurnVectorStore()
        store.add("orthogonal", [0.0, 1.0], _turn("orthogonal"), {})
        hits = store.search([1.0, 0.0], threshold=0.0)
        assert len(hits) == 1
        assert hits[0].score == pytest.approx(0.0)

    def test_threshold_above_all_scores_returns_empty(self):
        store = TurnVectorStore()
        store.add("a", [1.0, 0.0], _turn("a"), {})
        store.add("b", [0.5, 0.0], _turn("b"), {})
        assert store.search([1.0, 0.0], threshold=2.0) == []


class TestSearchHitFields:
    def test_hit_carries_turn_index(self):
        store = TurnVectorStore()
        store.add("first", [1.0], _turn("first"), {})
        store.add("second", [1.0], _turn("second"), {})
        hits = store.search([1.0])
        assert {h.turn_index for h in hits} == {0, 1}

    def test_hit_carries_stored_turn(self):
        store = TurnVectorStore()
        turn = _turn("hello")
        store.add("hello", [1.0], turn, {"prompt_name": "greet"})
        hits = store.search([1.0])
        assert hits[0].turn == turn
        assert hits[0].metadata == {"prompt_name": "greet"}

    def test_hit_is_turnhit_instance(self):
        store = TurnVectorStore()
        store.add("a", [1.0], _turn("a"), {})
        hits = store.search([1.0])
        assert isinstance(hits[0], TurnHit)


class TestIterEntries:
    def test_empty_store_yields_nothing(self):
        store = TurnVectorStore()
        assert list(store.iter_entries()) == []

    def test_yields_in_insertion_order(self):
        store = TurnVectorStore()
        store.add("first", [0.1], _turn("first"), {"i": 1})
        store.add("second", [0.2], _turn("second"), {"i": 2})
        store.add("third", [0.3], _turn("third"), {"i": 3})
        entries = list(store.iter_entries())
        assert [e.text for e in entries] == ["first", "second", "third"]

    def test_yields_entry_namedtuples(self):
        store = TurnVectorStore()
        store.add("a", [1.0], _turn("a"), {"k": "v"})
        entries = list(store.iter_entries())
        assert len(entries) == 1
        assert isinstance(entries[0], Entry)

    def test_entry_fields_match_add_arguments(self):
        store = TurnVectorStore()
        text = "the embedded text"
        embedding = [0.4, 0.5, 0.6]
        turn = _turn(text)
        metadata = {"prompt_name": "test", "session_id": "abc"}
        store.add(text, embedding, turn, metadata)
        entries = list(store.iter_entries())
        assert entries[0].text == text
        assert entries[0].embedding == embedding
        assert entries[0].turn == turn
        assert entries[0].metadata == metadata

    def test_iter_entries_after_clear_is_empty(self):
        store = TurnVectorStore()
        store.add("a", [0.1], _turn("a"), {})
        store.clear()
        assert list(store.iter_entries()) == []

    def test_iter_entries_yields_defensive_copies(self):
        store = TurnVectorStore()
        store.add("a", [0.1, 0.2], _turn("a"), {"k": "v"})
        first_pass = list(store.iter_entries())
        first_pass[0].embedding[0] = 99.0
        first_pass[0].metadata["k"] = "mutated"
        second_pass = list(store.iter_entries())
        assert second_pass[0].embedding == [0.1, 0.2]
        assert second_pass[0].metadata == {"k": "v"}
