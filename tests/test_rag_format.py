from __future__ import annotations

from ffai.rag.format import format_hits
from ffai.rag.types import SearchHit


class TestFormatHits:
    def test_empty_returns_empty_string(self):
        assert format_hits([]) == ""

    def test_single_hit(self):
        hits = [SearchHit(content="alpha text", score=0.95, source="doc_a")]
        out = format_hits(hits)
        assert "[1]" in out
        assert "source: doc_a" in out
        assert "relevance: 0.95" in out
        assert "alpha text" in out

    def test_multiple_hits_numbered(self):
        hits = [
            SearchHit(content="first", score=0.9, source="a"),
            SearchHit(content="second", score=0.8, source="b"),
        ]
        out = format_hits(hits)
        assert "[1]" in out
        assert "[2]" in out
        assert "source: a" in out
        assert "source: b" in out

    def test_max_chars_truncates(self):
        hits = [
            SearchHit(content="short", score=0.9, source="a"),
            SearchHit(content="B" * 500, score=0.8, source="b"),
        ]
        first = "[1] (source: a, relevance: 0.90)\nshort\n"
        out = format_hits(hits, max_chars=len(first))
        assert "short" in out
        assert "BBBB" not in out

    def test_parent_context_included(self):
        hits = [
            SearchHit(content="child", score=0.9, source="a", parent_content="parent text"),
        ]
        out = format_hits(hits, include_parent=True)
        assert "Parent context: parent text" in out

    def test_parent_context_excluded(self):
        hits = [
            SearchHit(content="child", score=0.9, source="a", parent_content="parent text"),
        ]
        out = format_hits(hits, include_parent=False)
        assert "Parent context" not in out

    def test_parent_context_truncated_at_200(self):
        hits = [
            SearchHit(content="child", score=0.9, source="a", parent_content="x" * 300),
        ]
        out = format_hits(hits)
        assert "..." in out
        assert out.count("x" * 200) == 1

    def test_no_source_shows_empty(self):
        hits = [SearchHit(content="text", score=0.5)]
        out = format_hits(hits)
        assert "source: " in out
