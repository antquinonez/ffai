from __future__ import annotations

import pytest

from src.rag.search.hybrid import HybridSearch, reciprocal_rank_fusion


def _vector_fn(query: str, n: int) -> list[dict]:
    return [
        {"id": f"v{i}", "content": f"vector result {i}", "score": 0.9 - i * 0.1}
        for i in range(n)
    ]


def _bm25_fn(query: str, n: int) -> list[dict]:
    return [
        {"id": f"b{i}", "content": f"bm25 result {i}", "score": 2.5 - i * 0.5}
        for i in range(n)
    ]


class TestHybridSearchInit:
    def test_stores_fns_and_params(self):
        hs = HybridSearch(vector_search_fn=_vector_fn, bm25_search_fn=_bm25_fn, alpha=0.7, rrf_k=30)
        assert hs.alpha == 0.7
        assert hs.rrf_k == 30
        assert hs.vector_search_fn is _vector_fn
        assert hs.bm25_search_fn is _bm25_fn

    def test_defaults(self):
        hs = HybridSearch()
        assert hs.alpha == 0.6
        assert hs.rrf_k == 60
        assert hs.vector_search_fn is None


class TestHybridSearchModes:
    def test_vector_mode_returns_vector_results(self):
        hs = HybridSearch(vector_search_fn=_vector_fn, bm25_search_fn=_bm25_fn)
        results = hs.search("test", n_results=3, mode="vector")
        assert len(results) == 3
        assert results[0]["id"] == "v0"
        assert results[0]["search_type"] == "vector"

    def test_bm25_mode_returns_bm25_results(self):
        hs = HybridSearch(vector_search_fn=_vector_fn, bm25_search_fn=_bm25_fn)
        results = hs.search("test", n_results=3, mode="bm25")
        assert len(results) == 3
        assert results[0]["id"] == "b0"
        assert results[0]["search_type"] == "bm25"

    def test_hybrid_mode_fuses_results(self):
        hs = HybridSearch(vector_search_fn=_vector_fn, bm25_search_fn=_bm25_fn, alpha=0.6)
        results = hs.search("test", n_results=5, mode="hybrid")
        assert len(results) == 5
        assert all(r["search_type"] == "hybrid" for r in results)
        assert "rrf_score" in results[0]
        assert results[0]["rrf_score"] > results[-1]["rrf_score"]

    def test_invalid_mode_raises(self):
        hs = HybridSearch()
        with pytest.raises(ValueError, match="Unknown search mode"):
            hs.search("test", mode="invalid")

    def test_vector_mode_no_fn_returns_empty(self):
        hs = HybridSearch(bm25_search_fn=_bm25_fn)
        results = hs.search("test", mode="vector")
        assert results == []

    def test_bm25_mode_no_fn_returns_empty(self):
        hs = HybridSearch(vector_search_fn=_vector_fn)
        results = hs.search("test", mode="bm25")
        assert results == []


class TestHybridSearchSetAlpha:
    def test_updates_alpha(self):
        hs = HybridSearch(alpha=0.5)
        hs.set_alpha(0.8)
        assert hs.alpha == 0.8

    def test_rejects_out_of_range(self):
        hs = HybridSearch()
        with pytest.raises(ValueError, match="between 0 and 1"):
            hs.set_alpha(1.5)
        with pytest.raises(ValueError, match="between 0 and 1"):
            hs.set_alpha(-0.1)


class TestHybridSearchSetFunctions:
    def test_updates_search_fns(self):
        hs = HybridSearch()
        hs.set_search_functions(vector_search_fn=_vector_fn, bm25_search_fn=_bm25_fn)
        assert hs.vector_search_fn is _vector_fn
        assert hs.bm25_search_fn is _bm25_fn


class TestHybridSearchFusionProperties:
    def test_overlapping_ids_get_combined_scores(self):
        def overlap_vector(q, n):
            return [{"id": "shared", "content": "from vector", "score": 0.9}]

        def overlap_bm25(q, n):
            return [{"id": "shared", "content": "from bm25", "score": 2.0}]

        hs = HybridSearch(vector_search_fn=overlap_vector, bm25_search_fn=overlap_bm25, alpha=0.6)
        results = hs.search("test", mode="hybrid")
        assert len(results) == 1
        assert results[0]["id"] == "shared"
        assert results[0]["vector_score"] == 0.9
        assert results[0]["bm25_score"] == 2.0
        expected_rrf = 0.6 / (60 + 1) + 0.4 / (60 + 1)
        assert results[0]["rrf_score"] == pytest.approx(expected_rrf)

    def test_results_sorted_by_rrf_descending(self):
        results = hs = HybridSearch(vector_search_fn=_vector_fn, bm25_search_fn=_bm25_fn, alpha=0.6)
        results = hs.search("test", n_results=10, mode="hybrid")
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestReciprocalRankFusion:
    def test_merges_two_lists(self):
        list_a = [{"id": "a1", "content": "a1"}, {"id": "a2", "content": "a2"}]
        list_b = [{"id": "b1", "content": "b1"}, {"id": "a1", "content": "a1 dup"}]
        fused = reciprocal_rank_fusion([list_a, list_b], k=60)
        ids = [r["id"] for r in fused]
        assert "a1" in ids
        assert "a2" in ids
        assert "b1" in ids
        assert len(fused) == 3

    def test_weights_affect_ordering(self):
        list_a = [{"id": "a", "content": "a"}]
        list_b = [{"id": "b", "content": "b"}]
        fused = reciprocal_rank_fusion([list_a, list_b], k=60, weights=[1.0, 0.0])
        assert fused[0]["id"] == "a"

    def test_mismatched_weights_raises(self):
        with pytest.raises(ValueError, match="Number of weights"):
            reciprocal_rank_fusion([[], []], weights=[1.0])

    def test_empty_input_returns_empty(self):
        assert reciprocal_rank_fusion([[]]) == []

    def test_equal_weights_default(self):
        lists = [[{"id": "a", "content": "a"}], [{"id": "b", "content": "b"}]]
        fused = reciprocal_rank_fusion(lists, k=60)
        assert len(fused) == 2
        assert fused[0]["rrf_score"] == pytest.approx(fused[1]["rrf_score"])

    def test_duplicate_id_gets_combined_score(self):
        lists = [
            [{"id": "x", "content": "from a"}],
            [{"id": "x", "content": "from b"}],
        ]
        fused = reciprocal_rank_fusion(lists, k=60, weights=[0.5, 0.5])
        assert len(fused) == 1
        expected = 0.5 / (60 + 1) + 0.5 / (60 + 1)
        assert fused[0]["rrf_score"] == pytest.approx(expected)

    def test_skips_entries_without_id(self):
        lists = [[{"id": "", "content": "no id"}], [{"id": "a", "content": "valid"}]]
        fused = reciprocal_rank_fusion(lists)
        assert len(fused) == 1
        assert fused[0]["id"] == "a"
