from __future__ import annotations

from ffai.rag.types import QueryResult, SearchHit


class TestSearchHit:
    def test_defaults(self):
        hit = SearchHit(content="text", score=0.9)
        assert hit.content == "text"
        assert hit.score == 0.9
        assert hit.source == ""
        assert hit.metadata == {}
        assert hit.parent_content is None
        assert hit.id == ""

    def test_all_fields(self):
        hit = SearchHit(
            content="text",
            score=0.8,
            source="doc1",
            metadata={"key": "val"},
            parent_content="parent",
            id="abc123",
        )
        assert hit.source == "doc1"
        assert hit.metadata == {"key": "val"}
        assert hit.parent_content == "parent"
        assert hit.id == "abc123"

    def test_is_dataclass(self):
        hit = SearchHit(content="a", score=0.5)
        assert hasattr(hit, "__dataclass_fields__")


class TestQueryResult:
    def test_all_fields(self):
        hits = [SearchHit(content="c1", score=0.9, source="s1")]
        result = QueryResult(answer="yes", hits=hits, sources=["s1"], prompt="ctx: q?")
        assert result.answer == "yes"
        assert len(result.hits) == 1
        assert result.hits[0].content == "c1"
        assert result.sources == ["s1"]
        assert result.prompt == "ctx: q?"

    def test_empty_hits(self):
        result = QueryResult(answer="idk", hits=[], sources=[], prompt="p")
        assert result.hits == []
        assert result.sources == []
        assert result.answer == "idk"

    def test_sources_deduplicated(self):
        hits = [
            SearchHit(content="a", score=0.9, source="s1"),
            SearchHit(content="b", score=0.8, source="s1"),
            SearchHit(content="c", score=0.7, source="s2"),
        ]
        result = QueryResult(answer="ans", hits=hits, sources=["s1", "s2"], prompt="p")
        assert result.sources == ["s1", "s2"]

    def test_is_dataclass(self):
        result = QueryResult(answer="a", hits=[], sources=[], prompt="p")
        assert hasattr(result, "__dataclass_fields__")
