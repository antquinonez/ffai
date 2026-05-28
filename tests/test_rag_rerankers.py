from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from ffai.rag.search.rerankers import (
    CrossEncoderReranker,
    DiversityReranker,
    NoopReranker,
    get_reranker,
)


def _make_results(n: int) -> list[dict]:
    return [
        {"id": str(i), "content": f"result {i} about topic {i % 3}", "score": 1.0 - i * 0.1}
        for i in range(n)
    ]


class TestNoopReranker:
    def test_returns_results_unchanged(self):
        results = _make_results(5)
        reranker = NoopReranker()
        assert reranker.rerank("query", results) == results

    def test_truncates_with_n_results(self):
        results = _make_results(5)
        reranker = NoopReranker()
        assert len(reranker.rerank("query", results, n_results=2)) == 2

    def test_empty_input(self):
        reranker = NoopReranker()
        assert reranker.rerank("query", []) == []


class TestDiversityReranker:
    def test_preserves_all_results(self):
        results = _make_results(4)
        reranker = DiversityReranker()
        reranked = reranker.rerank("query", results)
        assert len(reranked) == 4

    def test_respects_n_results(self):
        results = _make_results(5)
        reranker = DiversityReranker()
        reranked = reranker.rerank("query", results, n_results=2)
        assert len(reranked) == 2

    def test_adds_diversity_rank(self):
        results = _make_results(3)
        reranker = DiversityReranker()
        reranked = reranker.rerank("query", results)
        assert reranked[0]["diversity_rank"] == 1
        assert reranked[1]["diversity_rank"] == 2
        assert reranked[2]["diversity_rank"] == 3

    def test_single_result_unchanged(self):
        results = [{"id": "1", "content": "only result", "score": 0.9}]
        reranker = DiversityReranker()
        reranked = reranker.rerank("query", results)
        assert len(reranked) == 1
        assert reranked[0]["id"] == "1"

    def test_empty_returns_empty(self):
        reranker = DiversityReranker()
        assert reranker.rerank("query", []) == []

    def test_lambda_param_controls_diversity(self):
        results = [
            {"id": "1", "content": "cat dog pet", "score": 0.9},
            {"id": "2", "content": "cat dog pet", "score": 0.8},
            {"id": "3", "content": "space rocket mars", "score": 0.3},
        ]
        high_relevance = DiversityReranker(lambda_param=0.99)
        reranked_hr = high_relevance.rerank("query", results, n_results=2)
        ids_hr = [r["id"] for r in reranked_hr]

        high_diversity = DiversityReranker(lambda_param=0.01)
        reranked_hd = high_diversity.rerank("query", results, n_results=2)
        ids_hd = [r["id"] for r in reranked_hd]

        assert "3" in ids_hd

    def test_similarity_metric(self):
        reranker = DiversityReranker()
        assert reranker._simple_similarity("cat dog", "cat dog") == pytest.approx(1.0)
        assert reranker._simple_similarity("cat dog", "cat") == pytest.approx(0.5)
        assert reranker._simple_similarity("", "cat") == 0.0


class TestCrossEncoderReranker:
    def test_rerank_with_mock_model(self):
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.1, 0.5])
        reranker._model = mock_model

        results = [
            {"id": "1", "content": "first", "score": 0.3},
            {"id": "2", "content": "second", "score": 0.7},
            {"id": "3", "content": "third", "score": 0.5},
        ]
        reranked = reranker.rerank("query", results)
        assert len(reranked) == 3
        assert reranked[0]["id"] == "1"
        assert reranked[0]["rerank_score"] == pytest.approx(0.9)
        assert reranked[0]["original_score"] == pytest.approx(0.3)

    def test_empty_returns_empty(self):
        reranker = CrossEncoderReranker()
        assert reranker.rerank("query", []) == []

    def test_n_results_truncates(self):
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.8, 0.2])
        reranker._model = mock_model

        results = [{"id": "1", "content": "a", "score": 0.5}, {"id": "2", "content": "b", "score": 0.5}]
        reranked = reranker.rerank("query", results, n_results=1)
        assert len(reranked) == 1


class TestGetReranker:
    def test_diversity_type(self):
        r = get_reranker("diversity")
        assert isinstance(r, DiversityReranker)

    def test_cross_encoder_type(self):
        r = get_reranker("cross_encoder")
        assert isinstance(r, CrossEncoderReranker)

    def test_unknown_returns_noop(self):
        r = get_reranker("unknown")
        assert isinstance(r, NoopReranker)

    def test_none_returns_noop(self):
        r = get_reranker("none")
        assert isinstance(r, NoopReranker)

    def test_diversity_with_kwargs(self):
        r = get_reranker("diversity", lambda_param=0.3)
        assert isinstance(r, DiversityReranker)
        assert r.lambda_param == 0.3
