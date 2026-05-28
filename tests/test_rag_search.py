from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ffai.rag.search.hybrid import HybridSearch, reciprocal_rank_fusion
from ffai.rag.search.query_expansion import QueryExpander, fuse_search_results
from ffai.rag.search.rerankers import (
    CrossEncoderReranker,
    DiversityReranker,
    NoopReranker,
    get_reranker,
)


def _make_result(doc_id, score, content=""):
    return {"id": doc_id, "score": score, "content": content, "metadata": {}}


class TestHybridSearchVectorOnly:
    def test_returns_vector_results_tagged(self):
        vector_fn = MagicMock(return_value=[
            _make_result("v1", 0.9, "vec content 1"),
            _make_result("v2", 0.7, "vec content 2"),
        ])
        hs = HybridSearch(vector_search_fn=vector_fn)
        results = hs.search("query", mode="vector")
        assert len(results) == 2
        assert all(r["search_type"] == "vector" for r in results)
        assert results[0]["id"] == "v1"

    def test_vector_fn_not_set_returns_empty(self):
        hs = HybridSearch()
        results = hs.search("query", mode="vector")
        assert results == []


class TestHybridSearchBM25Only:
    def test_returns_bm25_results_tagged(self):
        bm25_fn = MagicMock(return_value=[
            _make_result("b1", 2.5, "bm25 content"),
        ])
        hs = HybridSearch(bm25_search_fn=bm25_fn)
        results = hs.search("query", mode="bm25")
        assert len(results) == 1
        assert results[0]["search_type"] == "bm25"
        assert results[0]["score"] == 2.5

    def test_bm25_fn_not_set_returns_empty(self):
        hs = HybridSearch()
        results = hs.search("query", mode="bm25")
        assert results == []


class TestHybridSearchHybridMode:
    def test_fuses_vector_and_bm25(self):
        vector_fn = MagicMock(return_value=[
            _make_result("v1", 0.9, "vec1"),
            _make_result("v2", 0.7, "vec2"),
        ])
        bm25_fn = MagicMock(return_value=[
            _make_result("b1", 2.5, "bm1"),
            _make_result("v1", 1.8, "vec1 overlap"),
        ])
        hs = HybridSearch(vector_search_fn=vector_fn, bm25_search_fn=bm25_fn)
        results = hs.search("query", mode="hybrid", n_results=5)
        assert len(results) == 3
        assert all(r["search_type"] == "hybrid" for r in results)
        ids = {r["id"] for r in results}
        assert ids == {"v1", "v2", "b1"}

    def test_hybrid_respects_n_results(self):
        vector_fn = MagicMock(return_value=[
            _make_result(f"v{i}", 0.9 - i * 0.1) for i in range(5)
        ])
        bm25_fn = MagicMock(return_value=[])
        hs = HybridSearch(vector_search_fn=vector_fn, bm25_search_fn=bm25_fn)
        results = hs.search("query", mode="hybrid", n_results=3)
        assert len(results) == 3

    def test_invalid_mode_raises(self):
        hs = HybridSearch()
        with pytest.raises(ValueError, match="Unknown search mode"):
            hs.search("query", mode="invalid")


class TestHybridSearchRRFScoring:
    def test_shared_doc_gets_combined_score(self):
        vector_fn = MagicMock(return_value=[_make_result("doc1", 0.9)])
        bm25_fn = MagicMock(return_value=[_make_result("doc1", 2.5)])
        hs = HybridSearch(vector_search_fn=vector_fn, bm25_search_fn=bm25_fn, alpha=0.6, rrf_k=60)
        results = hs.search("query", mode="hybrid")
        assert len(results) == 1
        expected_vector = 0.6 / (60 + 1)
        expected_bm25 = 0.4 / (60 + 1)
        assert results[0]["rrf_score"] == pytest.approx(expected_vector + expected_bm25)

    def test_alpha_weighting(self):
        vector_fn = MagicMock(return_value=[_make_result("v1", 0.9)])
        bm25_fn = MagicMock(return_value=[_make_result("b1", 2.5)])
        hs = HybridSearch(vector_search_fn=vector_fn, bm25_search_fn=bm25_fn, alpha=0.9, rrf_k=60)
        results = hs.search("query", mode="hybrid")
        v_score = next(r for r in results if r["id"] == "v1")
        b_score = next(r for r in results if r["id"] == "b1")
        assert v_score["rrf_score"] == pytest.approx(0.9 / 61)
        assert b_score["rrf_score"] == pytest.approx(0.1 / 61)


class TestHybridSearchSetAlpha:
    def test_set_valid_alpha(self):
        hs = HybridSearch()
        hs.set_alpha(0.3)
        assert hs.alpha == 0.3

    def test_set_invalid_alpha_raises(self):
        hs = HybridSearch()
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            hs.set_alpha(1.5)


class TestHybridSearchSetSearchFunctions:
    def test_update_functions(self):
        hs = HybridSearch()
        new_fn = MagicMock(return_value=[_make_result("x", 1.0)])
        hs.set_search_functions(vector_search_fn=new_fn)
        results = hs.search("query", mode="vector")
        assert len(results) == 1
        assert results[0]["id"] == "x"


class TestReciprocalRankFusion:
    def test_fuses_two_lists(self):
        list_a = [_make_result("a1", 0.9), _make_result("a2", 0.7)]
        list_b = [_make_result("b1", 2.5), _make_result("a2", 1.8)]
        results = reciprocal_rank_fusion([list_a, list_b], k=60)
        ids = [r["id"] for r in results]
        assert "a2" in ids
        a2 = next(r for r in results if r["id"] == "a2")
        assert a2["rrf_score"] == pytest.approx(0.5 / 62 + 0.5 / 62)

    def test_with_weights(self):
        list_a = [_make_result("x", 0.9)]
        list_b = [_make_result("x", 2.5)]
        results = reciprocal_rank_fusion([list_a, list_b], k=60, weights=[0.7, 0.3])
        x = results[0]
        assert x["rrf_score"] == pytest.approx(0.7 / 61 + 0.3 / 61)

    def test_mismatched_weights_raises(self):
        list_a = [_make_result("x", 0.9)]
        with pytest.raises(ValueError, match="Number of weights must match"):
            reciprocal_rank_fusion([list_a], weights=[0.5, 0.5])

    def test_empty_lists_raises_zero_division(self):
        with pytest.raises(ZeroDivisionError):
            reciprocal_rank_fusion([], k=60)

    def test_deduplicates_across_lists(self):
        list_a = [_make_result("x", 0.9), _make_result("y", 0.8)]
        list_b = [_make_result("x", 2.5), _make_result("z", 1.0)]
        results = reciprocal_rank_fusion([list_a, list_b], k=60)
        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids))


class TestNoopReranker:
    def test_returns_results_unchanged(self):
        reranker = NoopReranker()
        results = [_make_result("a", 0.9), _make_result("b", 0.7)]
        reranked = reranker.rerank("query", results)
        assert reranked == results

    def test_truncates_when_n_results_set(self):
        reranker = NoopReranker()
        results = [_make_result("a", 0.9), _make_result("b", 0.7), _make_result("c", 0.5)]
        reranked = reranker.rerank("query", results, n_results=2)
        assert len(reranked) == 2
        assert reranked[0]["id"] == "a"

    def test_empty_results(self):
        reranker = NoopReranker()
        assert reranker.rerank("query", []) == []


class TestDiversityReranker:
    def test_single_result_returns_as_is(self):
        reranker = DiversityReranker()
        results = [_make_result("a", 1.0)]
        reranked = reranker.rerank("query", results)
        assert len(reranked) == 1
        assert reranked[0]["id"] == "a"

    def test_promotes_diverse_results(self):
        reranker = DiversityReranker(lambda_param=0.3)
        results = [
            _make_result("a", 1.0, "machine learning neural network deep"),
            _make_result("b", 0.9, "machine learning neural network gradient"),
            _make_result("c", 0.8, "cooking recipe dinner meal"),
        ]
        reranked = reranker.rerank("query", results, n_results=3)
        assert len(reranked) == 3
        assert reranked[0]["id"] == "a"
        last_ids = {r["id"] for r in reranked}
        assert "c" in last_ids

    def test_assigns_diversity_rank(self):
        reranker = DiversityReranker()
        results = [
            _make_result("a", 1.0, "alpha beta"),
            _make_result("b", 0.8, "gamma delta"),
        ]
        reranked = reranker.rerank("query", results)
        assert reranked[0]["diversity_rank"] == 1
        assert reranked[1]["diversity_rank"] == 2

    def test_empty_results(self):
        reranker = DiversityReranker()
        assert reranker.rerank("query", []) == []


class TestCrossEncoderReranker:
    def test_rerank_with_mocked_model(self):
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]
        reranker._model = mock_model

        results = [
            _make_result("a", 0.9, "text a"),
            _make_result("b", 0.7, "text b"),
            _make_result("c", 0.5, "text c"),
        ]
        reranked = reranker.rerank("query", results)
        assert len(reranked) == 3
        assert reranked[0]["id"] == "b"
        assert reranked[0]["score"] == pytest.approx(0.9)
        assert reranked[0]["rerank_score"] == pytest.approx(0.9)
        assert reranked[0]["original_score"] == 0.7

    def test_rerank_truncates_to_n_results(self):
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]
        reranker._model = mock_model

        results = [
            _make_result("a", 0.9, "text a"),
            _make_result("b", 0.7, "text b"),
            _make_result("c", 0.5, "text c"),
        ]
        reranked = reranker.rerank("query", results, n_results=2)
        assert len(reranked) == 2

    def test_rerank_empty_results(self):
        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()
        assert reranker.rerank("query", []) == []

    def test_prediction_failure_returns_original(self):
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("model error")
        reranker._model = mock_model

        results = [_make_result("a", 0.9, "text")]
        reranked = reranker.rerank("query", results)
        assert len(reranked) == 1
        assert reranked[0]["id"] == "a"


class TestGetReranker:
    def test_none_type_returns_noop(self):
        r = get_reranker("none")
        assert isinstance(r, NoopReranker)

    def test_diversity_type_returns_diversity(self):
        r = get_reranker("diversity")
        assert isinstance(r, DiversityReranker)

    def test_cross_encoder_type_returns_cross_encoder(self):
        r = get_reranker("cross_encoder")
        assert isinstance(r, CrossEncoderReranker)

    def test_unknown_type_defaults_to_noop(self):
        r = get_reranker("unknown_type")
        assert isinstance(r, NoopReranker)

    def test_kwargs_passed_through(self):
        r = get_reranker("diversity", lambda_param=0.4)
        assert isinstance(r, DiversityReranker)
        assert r.lambda_param == 0.4


class TestQueryExpanderNoLLM:
    def test_no_llm_returns_original_only(self):
        expander = QueryExpander()
        result = expander.expand("authentication methods")
        assert result == ["authentication methods"]

    def test_no_llm_include_original_true(self):
        expander = QueryExpander(include_original=True)
        result = expander.expand("test query")
        assert result == ["test query"]


class TestQueryExpanderWithLLM:
    def test_expand_with_variations(self):
        response = "1. What is user auth?\n2. How to login?\n3. Identity verification"
        llm_fn = MagicMock(return_value=response)
        expander = QueryExpander(llm_generate_fn=llm_fn, n_variations=3)
        result = expander.expand("authentication")
        assert result[0] == "authentication"
        assert len(result) == 4

    def test_expand_without_original(self):
        response = "1. What is auth?\n2. Login methods"
        llm_fn = MagicMock(return_value=response)
        expander = QueryExpander(
            llm_generate_fn=llm_fn, n_variations=2, include_original=False
        )
        result = expander.expand("authentication")
        assert "authentication" not in result
        assert len(result) == 2

    def test_expand_llm_failure_returns_original(self):
        llm_fn = MagicMock(side_effect=RuntimeError("API error"))
        expander = QueryExpander(llm_generate_fn=llm_fn)
        result = expander.expand("test query")
        assert result == ["test query"]

    def test_expand_empty_response_returns_original(self):
        llm_fn = MagicMock(return_value="")
        expander = QueryExpander(llm_generate_fn=llm_fn)
        result = expander.expand("test query")
        assert result == ["test query"]


class TestQueryExpanderParseResponse:
    def test_parse_numbered_list(self):
        expander = QueryExpander()
        response = "1. First query\n2. Second query\n3. Third query"
        parsed = expander._parse_response(response)
        assert parsed == ["First query", "Second query", "Third query"]

    def test_parse_numbered_with_parentheses(self):
        expander = QueryExpander()
        response = "1) First\n2) Second"
        parsed = expander._parse_response(response)
        assert parsed == ["First", "Second"]

    def test_parse_strips_quotes(self):
        expander = QueryExpander()
        response = '1. "quoted query"\n2. \'single quoted\''
        parsed = expander._parse_response(response)
        assert parsed == ["quoted query", "single quoted"]

    def test_parse_filters_short_lines(self):
        expander = QueryExpander()
        response = "1. Ok\n2. A real query here"
        parsed = expander._parse_response(response)
        assert len(parsed) == 1
        assert parsed[0] == "A real query here"

    def test_parse_skips_empty_lines(self):
        expander = QueryExpander()
        response = "1. First\n\n2. Second\n\n"
        parsed = expander._parse_response(response)
        assert len(parsed) == 2

    def test_deduplication_in_expand(self):
        response = "1. authentication\n2. authentication"
        llm_fn = MagicMock(return_value=response)
        expander = QueryExpander(llm_generate_fn=llm_fn, include_original=True)
        result = expander.expand("authentication")
        assert result.count("authentication") == 1


class TestQueryExpanderSetLLM:
    def test_set_llm_function(self):
        expander = QueryExpander()
        assert expander.expand("test") == ["test"]
        new_fn = MagicMock(return_value="1. new query variant")
        expander.set_llm_function(new_fn)
        result = expander.expand("test")
        assert len(result) == 2


class TestFuseSearchResults:
    def test_fuses_multiple_lists(self):
        list_a = [_make_result("a", 0.9), _make_result("b", 0.7)]
        list_b = [_make_result("c", 2.5), _make_result("b", 1.8)]
        fused = fuse_search_results([list_a, list_b], n_results=5)
        ids = [r["id"] for r in fused]
        assert ids == ["a", "b", "c"]

    def test_deduplicates_by_id(self):
        list_a = [_make_result("x", 0.9)]
        list_b = [_make_result("x", 2.5)]
        fused = fuse_search_results([list_a, list_b], n_results=5)
        assert len(fused) == 1
        assert fused[0]["id"] == "x"

    def test_respects_n_results(self):
        lists = [[_make_result(f"doc{i}", float(i))] for i in range(10)]
        flat = [item for sublist in lists for item in sublist]
        fused = fuse_search_results([flat], n_results=3)
        assert len(fused) == 3

    def test_empty_input(self):
        assert fuse_search_results([]) == []

    def test_preserves_first_occurrence(self):
        list_a = [_make_result("x", 0.9, "from a")]
        list_b = [_make_result("x", 2.5, "from b")]
        fused = fuse_search_results([list_a, list_b])
        assert fused[0]["content"] == "from a"
