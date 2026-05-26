from __future__ import annotations

from src.rag.types import SearchHit


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
