from __future__ import annotations

import pytest

from ffai.rag.splitters.base import HierarchicalTextChunk, TextChunk
from ffai.rag.splitters.character import CharacterChunker
from ffai.rag.splitters.code import CodeChunker
from ffai.rag.splitters.factory import CHUNKER_REGISTRY, chunk_text, get_chunker, list_chunkers
from ffai.rag.splitters.hierarchical import HierarchicalChunker
from ffai.rag.splitters.markdown import MarkdownChunker
from ffai.rag.splitters.recursive import RecursiveChunker


class TestTextChunk:
    def test_fields_assigned(self):
        chunk = TextChunk(content="hello", chunk_index=0, start_char=0, end_char=5)
        assert chunk.content == "hello"
        assert chunk.chunk_index == 0
        assert chunk.start_char == 0
        assert chunk.end_char == 5

    def test_metadata_defaults_to_none(self):
        chunk = TextChunk(content="x", chunk_index=0, start_char=0, end_char=1)
        assert chunk.metadata is None

    def test_metadata_stored(self):
        chunk = TextChunk(
            content="x", chunk_index=0, start_char=0, end_char=1,
            metadata={"source": "test", "page": 1},
        )
        assert chunk.metadata == {"source": "test", "page": 1}


class TestHierarchicalTextChunk:
    def test_defaults(self):
        chunk = HierarchicalTextChunk(content="x", chunk_index=0, start_char=0, end_char=1)
        assert chunk.id == ""
        assert chunk.parent_id is None
        assert chunk.child_ids == []
        assert chunk.hierarchy_level == 0

    def test_post_init_populates_none_collections(self):
        chunk = HierarchicalTextChunk(
            content="x", chunk_index=0, start_char=0, end_char=1,
            child_ids=None, metadata=None,
        )
        assert chunk.child_ids == []
        assert chunk.metadata == {}

    def test_inherits_text_chunk(self):
        chunk = HierarchicalTextChunk(content="x", chunk_index=0, start_char=0, end_char=1)
        assert isinstance(chunk, TextChunk)

    def test_explicit_values(self):
        chunk = HierarchicalTextChunk(
            content="x", chunk_index=2, start_char=10, end_char=11,
            id="abc-123", parent_id="parent-1", child_ids=["c1", "c2"],
            hierarchy_level=1, metadata={"key": "val"},
        )
        assert chunk.id == "abc-123"
        assert chunk.parent_id == "parent-1"
        assert chunk.child_ids == ["c1", "c2"]
        assert chunk.hierarchy_level == 1


class TestChunkerBaseValidation:
    def test_negative_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            CharacterChunker(chunk_size=-1)

    def test_zero_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            CharacterChunker(chunk_size=0)

    def test_negative_overlap_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            CharacterChunker(chunk_size=100, chunk_overlap=-1)

    def test_overlap_equal_to_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            CharacterChunker(chunk_size=100, chunk_overlap=100)

    def test_overlap_greater_than_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            CharacterChunker(chunk_size=50, chunk_overlap=200)

    def test_valid_params(self):
        chunker = CharacterChunker(chunk_size=100, chunk_overlap=20)
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 20


class TestChunkerNameProperty:
    def test_character(self):
        assert CharacterChunker().name == "character"

    def test_recursive(self):
        assert RecursiveChunker().name == "recursive"

    def test_markdown(self):
        assert MarkdownChunker().name == "markdown"

    def test_code(self):
        assert CodeChunker().name == "code"

    def test_hierarchical(self):
        assert HierarchicalChunker().name == "hierarchical"


class TestChunkerBaseMergeMetadata:
    def test_default_metadata_only(self):
        chunker = CharacterChunker(metadata={"source": "doc.pdf"})
        assert chunker._merge_metadata(None) == {"source": "doc.pdf"}

    def test_call_metadata_overrides_default(self):
        chunker = CharacterChunker(metadata={"source": "doc.pdf", "page": 1})
        merged = chunker._merge_metadata({"page": 5, "section": "intro"})
        assert merged == {"source": "doc.pdf", "page": 5, "section": "intro"}

    def test_no_metadata_returns_empty(self):
        chunker = CharacterChunker()
        assert chunker._merge_metadata(None) == {}

    def test_call_metadata_only(self):
        chunker = CharacterChunker()
        assert chunker._merge_metadata({"key": "val"}) == {"key": "val"}


class TestCharacterChunker:
    def test_empty_string(self):
        assert CharacterChunker().chunk("") == []

    def test_whitespace_only(self):
        assert CharacterChunker().chunk("   \n\t  ") == []

    def test_single_word(self):
        chunks = CharacterChunker(chunk_size=1000).chunk("hello")
        assert len(chunks) == 1
        assert chunks[0].content == "hello"
        assert chunks[0].chunk_index == 0
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == 5

    def test_text_shorter_than_chunk_size(self):
        text = "Hello world, this is a test."
        chunks = CharacterChunker(chunk_size=1000).chunk(text)
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == len(text)

    def test_text_exactly_at_chunk_size_boundary(self):
        text = "a" * 10
        chunks = CharacterChunker(
            chunk_size=10, chunk_overlap=0, respect_word_boundaries=False,
        ).chunk(text)
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == 10

    def test_splits_at_word_boundary(self):
        chunks = CharacterChunker(
            chunk_size=10, chunk_overlap=0, respect_word_boundaries=True,
        ).chunk("hello world")
        assert len(chunks) == 2
        assert chunks[0].content == "hello"
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == 5
        assert chunks[1].content == "world"
        assert chunks[1].start_char == 5
        assert chunks[1].end_char == 11

    def test_splits_without_word_boundary(self):
        chunks = CharacterChunker(
            chunk_size=10, chunk_overlap=0, respect_word_boundaries=False,
        ).chunk("hello world foo bar")
        assert len(chunks) == 2
        assert chunks[0].content == "hello worl"
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == 10
        assert chunks[1].content == "d foo bar"
        assert chunks[1].start_char == 10
        assert chunks[1].end_char == 19

    def test_overlap_creates_repeated_content(self):
        chunks = CharacterChunker(
            chunk_size=10, chunk_overlap=5, respect_word_boundaries=False,
        ).chunk("hello world foo bar")
        assert len(chunks) == 3
        assert chunks[0].content == "hello worl"
        assert "world foo" in chunks[1].content
        assert "d foo bar" in chunks[2].content

    def test_chunk_index_sequential(self):
        text = "word " * 100
        chunks = CharacterChunker(chunk_size=20, chunk_overlap=0).chunk(text)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunks_cover_full_text(self):
        text = "hello world"
        chunks = CharacterChunker(
            chunk_size=10, chunk_overlap=0, respect_word_boundaries=True,
        ).chunk(text)
        assert chunks[0].start_char == 0
        assert chunks[-1].end_char == len(text)

    def test_metadata_attached_from_constructor(self):
        chunks = CharacterChunker(chunk_size=1000, metadata={"doc": "test"}).chunk("hello")
        assert chunks[0].metadata == {"doc": "test"}

    def test_call_metadata_merges_with_default(self):
        chunks = CharacterChunker(
            chunk_size=1000, metadata={"doc": "base"},
        ).chunk("hello", metadata={"page": 1})
        assert chunks[0].metadata == {"doc": "base", "page": 1}

    def test_each_chunk_gets_independent_metadata_copy(self):
        chunks = CharacterChunker(chunk_size=10, chunk_overlap=0).chunk("hello world foo bar")
        assert len(chunks) >= 2
        assert chunks[0].metadata is not chunks[1].metadata


class TestRecursiveChunker:
    def test_empty_string(self):
        assert RecursiveChunker().chunk("") == []

    def test_whitespace_only(self):
        assert RecursiveChunker().chunk("   ") == []

    def test_single_word(self):
        chunks = RecursiveChunker(chunk_size=1000).chunk("hello")
        assert len(chunks) == 1
        assert chunks[0].content == "hello"
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == 5

    def test_text_shorter_than_chunk_size(self):
        text = "Hello world."
        chunks = RecursiveChunker(chunk_size=1000).chunk(text)
        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_separator_hierarchy_splits_paragraphs(self):
        text = "First paragraph here.\n\nSecond paragraph here."
        chunks = RecursiveChunker(chunk_size=25, chunk_overlap=0).chunk(text)
        assert len(chunks) == 2
        assert "First paragraph" in chunks[0].content
        assert "Second paragraph" in chunks[1].content

    def test_custom_separators(self):
        chunks = RecursiveChunker(
            chunk_size=3, chunk_overlap=0, separators=["|", " "], keep_separator=False,
        ).chunk("a|b|c|d|e")
        assert len(chunks) == 2
        assert chunks[0].content == "abc"
        assert chunks[1].content == "de"

    def test_chunk_index_sequential(self):
        text = "Word " * 200
        chunks = RecursiveChunker(chunk_size=50, chunk_overlap=0).chunk(text)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_metadata_propagated(self):
        chunks = RecursiveChunker(
            chunk_size=1000, metadata={"src": "test"},
        ).chunk("hello")
        assert chunks[0].metadata == {"src": "test"}

    def test_large_text_produces_multiple_chunks(self):
        text = "This is a sentence with some words. " * 50
        chunks = RecursiveChunker(chunk_size=100, chunk_overlap=0).chunk(text)
        assert len(chunks) > 1

    def test_fallback_to_character_split_for_no_separators(self):
        text = "a" * 50
        chunks = RecursiveChunker(chunk_size=20, chunk_overlap=0).chunk(text)
        assert len(chunks) == 3
        assert len(chunks[0].content) == 20
        assert len(chunks[1].content) == 20
        assert len(chunks[2].content) == 10

    def test_keep_separator_true(self):
        text = "Hello.\n\nWorld."
        chunks = RecursiveChunker(
            chunk_size=15, chunk_overlap=0, keep_separator=True,
        ).chunk(text)
        joined = "".join(c.content for c in chunks)
        assert "\n\n" in joined

    def test_keep_separator_false(self):
        text = "Hello.\n\nWorld."
        chunks = RecursiveChunker(
            chunk_size=7, chunk_overlap=0, keep_separator=False,
        ).chunk(text)
        joined = "".join(c.content for c in chunks)
        assert "\n\n" not in joined


class TestMarkdownChunker:
    def test_empty_string(self):
        assert MarkdownChunker().chunk("") == []

    def test_whitespace_only(self):
        assert MarkdownChunker().chunk("  \n  ") == []

    def test_single_header_single_chunk(self):
        text = "# Title\nSome content."
        chunks = MarkdownChunker(chunk_size=1000).chunk(text)
        assert len(chunks) == 1
        assert "# Title" in chunks[0].content
        assert "Some content." in chunks[0].content

    def test_splits_at_h1_and_h2(self):
        text = "# Title\nContent A\n\n## Section\nContent B"
        chunks = MarkdownChunker(chunk_size=1000, split_headers=["h1", "h2"]).chunk(text)
        assert len(chunks) == 2
        meta0 = chunks[0].metadata
        meta1 = chunks[1].metadata
        assert meta0 is not None and meta0["header_level"] == "h1"
        assert meta1 is not None and meta1["header_level"] == "h2"

    def test_header_text_in_metadata(self):
        text = "# My Title\nSome text."
        chunks = MarkdownChunker(chunk_size=1000).chunk(text)
        meta = chunks[0].metadata
        assert meta is not None and meta["header"] == "# My Title"

    def test_no_headers_produces_single_chunk(self):
        text = "Just some text without headers."
        chunks = MarkdownChunker(chunk_size=1000).chunk(text)
        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_large_section_splits_with_fallback_enabled(self):
        content = "Word " * 500
        text = f"# Title\n{content}"
        chunks = MarkdownChunker(chunk_size=200, chunk_overlap=50, max_chunk_fallback=True).chunk(text)
        assert len(chunks) > 1

    def test_large_section_no_split_when_fallback_disabled(self):
        content = "Word " * 500
        text = f"# Title\n{content}"
        chunks_with = MarkdownChunker(chunk_size=200, chunk_overlap=50, max_chunk_fallback=True).chunk(text)
        chunks_without = MarkdownChunker(chunk_size=200, chunk_overlap=50, max_chunk_fallback=False).chunk(text)
        assert len(chunks_with) > len(chunks_without)

    def test_preserve_structure_includes_header_in_content(self):
        text = "# Title\nBody text."
        chunks = MarkdownChunker(chunk_size=1000, preserve_structure=True).chunk(text)
        assert chunks[0].content.startswith("# Title")

    def test_chunk_index_sequential(self):
        text = "# A\nContent A\n\n## B\nContent B\n\n### C\nContent C"
        chunks = MarkdownChunker(chunk_size=1000, split_headers=["h1", "h2", "h3"]).chunk(text)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_only_h1_splits(self):
        text = "# H1\nText\n## H2\nText\n### H3\nText"
        chunks = MarkdownChunker(chunk_size=1000, split_headers=["h1"]).chunk(text)
        assert len(chunks) == 1

    def test_h1_h2_h3_all_split(self):
        text = "# A\nText\n## B\nText\n### C\nText"
        chunks = MarkdownChunker(chunk_size=1000, split_headers=["h1", "h2", "h3"]).chunk(text)
        assert len(chunks) == 3

    def test_metadata_merged_with_header_info(self):
        chunks = MarkdownChunker(
            chunk_size=1000, metadata={"source": "doc.md"},
        ).chunk("# Title\nBody.")
        meta = chunks[0].metadata
        assert meta is not None
        assert meta["source"] == "doc.md"
        assert meta["header"] == "# Title"
        assert meta["header_level"] == "h1"


class TestCodeChunker:
    def test_empty_string(self):
        assert CodeChunker().chunk("") == []

    def test_whitespace_only(self):
        assert CodeChunker().chunk("   \n  ") == []

    def test_single_function(self):
        code = "def hello():\n    print('hello')\n"
        chunks = CodeChunker(chunk_size=1000).chunk(code)
        assert len(chunks) >= 1
        assert "def hello():" in chunks[0].content

    def test_function_name_in_metadata(self):
        code = "def my_func():\n    pass\n"
        chunks = CodeChunker(chunk_size=1000).chunk(code)
        meta = chunks[0].metadata
        assert meta is not None
        assert meta["block_name"] == "my_func"
        assert meta["block_type"] == "function"

    def test_language_in_metadata(self):
        code = "def f():\n    pass\n"
        chunks = CodeChunker(chunk_size=1000).chunk(code)
        meta = chunks[0].metadata
        assert meta is not None and meta["language"] == "python"

    def test_multiple_functions_multiple_chunks(self):
        code = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        chunks = CodeChunker(chunk_size=1000).chunk(code)
        assert len(chunks) == 2
        meta0 = chunks[0].metadata
        meta1 = chunks[1].metadata
        assert meta0 is not None and meta0["block_name"] == "foo"
        assert meta1 is not None and meta1["block_name"] == "bar"

    def test_large_function_splits_into_parts(self):
        body = "    x = 1\n" * 100
        code = f"def big():\n{body}"
        chunks = CodeChunker(chunk_size=200, chunk_overlap=50).chunk(code)
        assert len(chunks) > 1

    def test_module_level_code_before_function(self):
        code = "import os\nimport sys\n\ndef main():\n    pass\n"
        chunks = CodeChunker(chunk_size=1000).chunk(code)
        assert len(chunks) == 2
        meta0 = chunks[0].metadata
        meta1 = chunks[1].metadata
        assert meta0 is not None and meta0["block_type"] == "module_level"
        assert meta1 is not None and meta1["block_name"] == "main"

    def test_fallback_chunk_method(self):
        chunker = CodeChunker(chunk_size=15, chunk_overlap=5)
        code = "x = 1\ny = 2\nz = 3\n"
        chunks = chunker._fallback_chunk(code, {"language": "python"})
        assert len(chunks) >= 1
        meta = chunks[0].metadata
        assert meta is not None and meta["block_type"] == "fallback"

    def test_async_function_detection(self):
        code = "async def fetch():\n    await something()\n"
        chunks = CodeChunker(chunk_size=1000).chunk(code)
        assert len(chunks) >= 1
        meta = chunks[0].metadata
        assert meta is not None and meta["block_name"] == "fetch"

    def test_class_detection_with_split_by_class(self):
        code = "class MyClass:\n    def method(self):\n        pass\n"
        chunks = CodeChunker(chunk_size=1000, split_by="class").chunk(code)
        assert len(chunks) >= 2
        assert any(
            c.metadata is not None and c.metadata.get("block_name") == "MyClass"
            for c in chunks
        )

    def test_javascript_language(self):
        code = "function hello() {\n  return 1;\n}\n"
        chunks = CodeChunker(chunk_size=1000, language="javascript").chunk(code)
        assert len(chunks) >= 1
        meta = chunks[0].metadata
        assert meta is not None and meta["language"] == "javascript"

    def test_unknown_language_uses_generic_patterns(self):
        code = "function unknown() {\n  return 1;\n}\n"
        chunks = CodeChunker(chunk_size=1000, language="brainfuck").chunk(code)
        assert len(chunks) >= 1
        meta = chunks[0].metadata
        assert meta is not None and meta["language"] == "brainfuck"

    def test_chunk_index_sequential(self):
        code = "def a():\n    pass\n\ndef b():\n    pass\n\ndef c():\n    pass\n"
        chunks = CodeChunker(chunk_size=1000).chunk(code)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


class TestHierarchicalChunker:
    def test_empty_string(self):
        assert HierarchicalChunker().chunk("") == []

    def test_whitespace_only(self):
        assert HierarchicalChunker().chunk("   ") == []

    def test_produces_parent_and_child(self):
        chunks = HierarchicalChunker(chunk_size=400, parent_chunk_size=1500).chunk("hello")
        parents = [c for c in chunks if c.hierarchy_level == 0]
        children = [c for c in chunks if c.hierarchy_level > 0]
        assert len(parents) == 1
        assert len(children) >= 1

    def test_parent_has_nonempty_uuid(self):
        chunks = HierarchicalChunker().chunk("hello world")
        parent = [c for c in chunks if c.hierarchy_level == 0][0]
        assert len(parent.id) == 36
        assert parent.id.count("-") == 4

    def test_child_parent_id_references_valid_parent(self):
        chunks = HierarchicalChunker(chunk_size=100, chunk_overlap=50, parent_chunk_size=500).chunk(
            "word " * 200,
        )
        parent_ids = {c.id for c in chunks if c.hierarchy_level == 0}
        children = [c for c in chunks if c.hierarchy_level > 0]
        for child in children:
            assert child.parent_id in parent_ids

    def test_parent_child_ids_reference_valid_children(self):
        chunks = HierarchicalChunker(chunk_size=100, chunk_overlap=50, parent_chunk_size=500).chunk(
            "word " * 200,
        )
        parents = [c for c in chunks if c.hierarchy_level == 0]
        all_ids = {c.id for c in chunks}
        for parent in parents:
            assert parent.child_ids is not None
            for child_id in parent.child_ids:
                assert child_id in all_ids

    def test_child_parent_id_is_bidirectional(self):
        chunks = HierarchicalChunker(chunk_size=100, chunk_overlap=50, parent_chunk_size=500).chunk(
            "word " * 200,
        )
        parents = [c for c in chunks if c.hierarchy_level == 0]
        for parent in parents:
            assert parent.child_ids is not None
            for child_id in parent.child_ids:
                child = next(c for c in chunks if c.id == child_id)
                assert child.parent_id == parent.id

    def test_hierarchy_levels_are_correct(self):
        chunks = HierarchicalChunker(chunk_size=100, chunk_overlap=50, parent_chunk_size=500).chunk(
            "word " * 200,
        )
        parent_levels = {c.hierarchy_level for c in chunks if c.hierarchy_level == 0}
        child_levels = {c.hierarchy_level for c in chunks if c.hierarchy_level > 0}
        assert parent_levels == {0}
        assert child_levels == {1}

    def test_get_parent_chunks(self):
        chunker = HierarchicalChunker(chunk_size=100, chunk_overlap=50, parent_chunk_size=500)
        chunks = chunker.chunk("word " * 200)
        parents = chunker.get_parent_chunks(chunks)
        assert len(parents) > 0
        assert all(c.hierarchy_level == 0 for c in parents)

    def test_get_child_chunks(self):
        chunker = HierarchicalChunker(chunk_size=100, chunk_overlap=50, parent_chunk_size=500)
        chunks = chunker.chunk("word " * 200)
        children = chunker.get_child_chunks(chunks)
        assert len(children) > 0
        assert all(c.hierarchy_level > 0 for c in children)

    def test_get_chunks_with_parent_context(self):
        chunker = HierarchicalChunker(chunk_size=100, chunk_overlap=50, parent_chunk_size=500)
        chunks = chunker.chunk("word " * 200)
        children = chunker.get_child_chunks(chunks)
        results = chunker.get_chunks_with_parent_context(children, chunks)
        assert len(results) == len(children)
        for result in results:
            assert result["parent"] is not None
            assert result["parent_content"] is not None

    def test_large_text_creates_multiple_parents(self):
        text = "word " * 1000
        chunks = HierarchicalChunker(chunk_size=100, chunk_overlap=50, parent_chunk_size=500).chunk(text)
        parents = [c for c in chunks if c.hierarchy_level == 0]
        assert len(parents) > 1

    def test_metadata_propagated_to_all_chunks(self):
        chunks = HierarchicalChunker(
            chunk_size=100, chunk_overlap=50, parent_chunk_size=500, metadata={"src": "test"},
        ).chunk("hello world")
        for chunk in chunks:
            meta = chunk.metadata
            assert meta is not None and meta["src"] == "test"

    def test_child_metadata_includes_parent_chunk_size(self):
        chunks = HierarchicalChunker(chunk_size=100, chunk_overlap=50, parent_chunk_size=500).chunk("hello world")
        children = [c for c in chunks if c.hierarchy_level > 0]
        for child in children:
            assert child.metadata is not None
            assert "parent_chunk_size" in child.metadata


class TestGetChunker:
    def test_character(self):
        assert isinstance(get_chunker("character"), CharacterChunker)

    def test_recursive(self):
        assert isinstance(get_chunker("recursive"), RecursiveChunker)

    def test_markdown(self):
        assert isinstance(get_chunker("markdown"), MarkdownChunker)

    def test_code(self):
        assert isinstance(get_chunker("code"), CodeChunker)

    def test_hierarchical(self):
        assert isinstance(get_chunker("hierarchical"), HierarchicalChunker)

    def test_case_insensitive(self):
        assert isinstance(get_chunker("Recursive"), RecursiveChunker)
        assert isinstance(get_chunker("CHARACTER"), CharacterChunker)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy: 'nonexistent'"):
            get_chunker("nonexistent")

    def test_passes_chunk_size(self):
        assert get_chunker("character", chunk_size=500).chunk_size == 500

    def test_passes_chunk_overlap(self):
        assert get_chunker("character", chunk_overlap=50).chunk_overlap == 50

    def test_passes_metadata(self):
        assert get_chunker("character", metadata={"k": "v"}).default_metadata == {"k": "v"}

    def test_markdown_kwargs(self):
        chunker = get_chunker("markdown", split_headers=["h1"], preserve_structure=False)
        assert isinstance(chunker, MarkdownChunker)
        assert chunker.split_headers == ["h1"]
        assert chunker.preserve_structure is False

    def test_code_kwargs(self):
        chunker = get_chunker("code", language="javascript", split_by="class")
        assert isinstance(chunker, CodeChunker)
        assert chunker.language == "javascript"
        assert chunker.split_by == "class"

    def test_hierarchical_kwargs(self):
        chunker = get_chunker("hierarchical", parent_chunk_size=2000, max_levels=3)
        assert isinstance(chunker, HierarchicalChunker)
        assert chunker.parent_chunk_size == 2000
        assert chunker.max_levels == 3

    def test_recursive_kwargs(self):
        chunker = get_chunker("recursive", separators=["\n", ". "], keep_separator=False)
        assert isinstance(chunker, RecursiveChunker)
        assert chunker.separators == ["\n", ". "]
        assert chunker.keep_separator is False

    def test_character_kwargs(self):
        chunker = get_chunker("character", respect_word_boundaries=False)
        assert isinstance(chunker, CharacterChunker)
        assert chunker.respect_word_boundaries is False


class TestListChunkers:
    def test_returns_all_five_strategies(self):
        strategies = list_chunkers()
        assert set(strategies) == {"character", "recursive", "markdown", "code", "hierarchical"}

    def test_count_matches_registry(self):
        assert len(list_chunkers()) == len(CHUNKER_REGISTRY)
        assert len(list_chunkers()) == 5


class TestChunkText:
    def test_returns_chunks_with_default_strategy(self):
        text = "Hello world. " * 50
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_markdown_strategy(self):
        chunks = chunk_text("# Title\nBody.", strategy="markdown")
        assert len(chunks) >= 1
        assert "# Title" in chunks[0].content

    def test_metadata_propagated(self):
        chunks = chunk_text("hello", strategy="character", metadata={"doc": "test"})
        meta = chunks[0].metadata
        assert meta is not None and meta["doc"] == "test"

    def test_empty_text(self):
        assert chunk_text("") == []

    def test_code_strategy_with_kwargs(self):
        chunks = chunk_text(
            "def foo():\n    pass\n",
            strategy="code",
            language="python",
        )
        assert len(chunks) >= 1
        meta = chunks[0].metadata
        assert meta is not None and meta["language"] == "python"
