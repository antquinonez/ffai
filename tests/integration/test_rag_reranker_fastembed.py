import importlib.util

import pytest

from ffai.rag.search.rerankers import CrossEncoderReranker

pytestmark = pytest.mark.integration

HAS_FASTEMBED = importlib.util.find_spec("fastembed") is not None


def _skip_no_fastembed():
    if not HAS_FASTEMBED:
        pytest.skip("fastembed not installed (pip install ffai[rag])")


class TestCrossEncoderRerankerFastEmbed:
    def test_rerank_produces_scores(self):
        _skip_no_fastembed()
        reranker = CrossEncoderReranker()
        results = [
            {"id": "1", "content": "async await in Python", "score": 0.8},
            {"id": "2", "content": "Rust borrow checker rules", "score": 0.6},
            {"id": "3", "content": "Go channels and goroutines", "score": 0.5},
        ]
        reranked = reranker.rerank("async programming", results)
        assert len(reranked) == 3
        assert all("rerank_score" in r for r in reranked)
        assert all("original_score" in r for r in reranked)

    def test_rerank_reorders_by_relevance(self):
        _skip_no_fastembed()
        reranker = CrossEncoderReranker()
        results = [
            {"id": "1", "content": "Python async await programming", "score": 0.5},
            {"id": "2", "content": "Italian cooking recipes", "score": 0.9},
        ]
        reranked = reranker.rerank("async programming", results)
        assert reranked[0]["id"] == "1"
        assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]

    def test_rerank_n_results_truncates(self):
        _skip_no_fastembed()
        reranker = CrossEncoderReranker()
        results = [
            {"id": "1", "content": "async await", "score": 0.8},
            {"id": "2", "content": "rust borrow", "score": 0.6},
            {"id": "3", "content": "go channels", "score": 0.5},
        ]
        reranked = reranker.rerank("async programming", results, n_results=2)
        assert len(reranked) == 2

    def test_rerank_empty_returns_empty(self):
        _skip_no_fastembed()
        reranker = CrossEncoderReranker()
        assert reranker.rerank("query", []) == []

    def test_backend_is_fastembed(self):
        _skip_no_fastembed()
        reranker = CrossEncoderReranker()
        reranker._load_model()
        assert reranker._backend == "fastembed"

    def test_scores_are_sorted_descending(self):
        _skip_no_fastembed()
        reranker = CrossEncoderReranker()
        results = [
            {"id": str(i), "content": c, "score": 0.5}
            for i, c in enumerate([
                "async programming in Python",
                "baking chocolate chip cookies",
                "concurrent goroutines in Go",
            ])
        ]
        reranked = reranker.rerank("async programming", results)
        scores = [r["rerank_score"] for r in reranked]
        assert scores == sorted(scores, reverse=True)
