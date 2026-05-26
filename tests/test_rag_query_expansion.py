from __future__ import annotations

from src.rag.search.query_expansion import QueryExpander, fuse_search_results


class TestQueryExpander:
    def test_no_llm_returns_original(self):
        expander = QueryExpander()
        result = expander.expand("what is machine learning?")
        assert result == ["what is machine learning?"]

    def test_expands_with_llm(self):
        def mock_llm(prompt: str) -> str:
            return "1. How does ML work?\n2. What are ML algorithms?\n3. Explain machine learning"

        expander = QueryExpander(llm_generate_fn=mock_llm, n_variations=3)
        result = expander.expand("what is machine learning?")
        assert len(result) == 4
        assert result[0] == "what is machine learning?"
        assert "How does ML work?" in result[1]

    def test_excludes_original_when_configured(self):
        def mock_llm(prompt: str) -> str:
            return "1. variant one"

        expander = QueryExpander(llm_generate_fn=mock_llm, include_original=False)
        result = expander.expand("original query")
        assert "original query" not in result

    def test_llm_failure_returns_original(self):
        def failing_llm(prompt: str) -> str:
            raise RuntimeError("API error")

        expander = QueryExpander(llm_generate_fn=failing_llm)
        result = expander.expand("test query")
        assert result == ["test query"]

    def test_deduplicates_results(self):
        def mock_llm(prompt: str) -> str:
            return "1. same query\n2. same query\n3. different query"

        expander = QueryExpander(llm_generate_fn=mock_llm)
        result = expander.expand("same query")
        assert len(result) == len(set(result))

    def test_parse_response_handles_numbered_list(self):
        expander = QueryExpander()
        parsed = expander._parse_response("1. first query\n2. second query\n3. third query")
        assert parsed == ["first query", "second query", "third query"]

    def test_parse_response_handles_parentheses(self):
        expander = QueryExpander()
        parsed = expander._parse_response("1) first\n2) second")
        assert parsed == ["first", "second"]

    def test_parse_response_strips_quotes(self):
        expander = QueryExpander()
        parsed = expander._parse_response('1. "quoted query"')
        assert parsed == ["quoted query"]

    def test_parse_response_filters_short(self):
        expander = QueryExpander()
        parsed = expander._parse_response("1. ok\n2. this is a real query")
        assert len(parsed) == 1
        assert parsed[0] == "this is a real query"

    def test_set_llm_function(self):
        expander = QueryExpander()
        expander.set_llm_function(lambda p: "1. expanded")
        result = expander.expand("test")
        assert len(result) == 2


class TestFuseSearchResults:
    def test_merges_lists(self):
        a = [{"id": "1", "content": "a"}, {"id": "2", "content": "b"}]
        b = [{"id": "3", "content": "c"}]
        fused = fuse_search_results([a, b])
        assert len(fused) == 3

    def test_deduplicates(self):
        a = [{"id": "1", "content": "a"}]
        b = [{"id": "1", "content": "a dup"}, {"id": "2", "content": "b"}]
        fused = fuse_search_results([a, b])
        assert len(fused) == 2
        ids = [r["id"] for r in fused]
        assert ids.count("1") == 1

    def test_respects_n_results(self):
        lists = [[{"id": str(i), "content": f"c{i}"} for i in range(10)]]
        fused = fuse_search_results(lists, n_results=3)
        assert len(fused) == 3

    def test_empty_input(self):
        assert fuse_search_results([]) == []

    def test_empty_inner_list(self):
        assert fuse_search_results([[]]) == []

    def test_preserves_first_occurrence(self):
        a = [{"id": "x", "content": "first"}]
        b = [{"id": "x", "content": "second"}]
        fused = fuse_search_results([a, b])
        assert fused[0]["content"] == "first"
