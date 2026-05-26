from __future__ import annotations

import pytest

from src.rag.indexing.bm25 import BM25Index
from src.rag.indexing.contextual import ContextualEmbeddings
from src.rag.indexing.deduplication import ChunkDeduplicator
from src.rag.indexing.hierarchical import HierarchicalIndex


class TestBM25Tokenize:
    def test_lowercase_conversion(self):
        idx = BM25Index()
        assert idx.tokenize("Hello WORLD") == ["hello", "world"]

    def test_removes_punctuation(self):
        idx = BM25Index()
        assert idx.tokenize("hello, world!") == ["hello", "world"]

    def test_filters_single_char_tokens(self):
        idx = BM25Index()
        assert idx.tokenize("a big cat") == ["big", "cat"]

    def test_empty_string(self):
        idx = BM25Index()
        assert idx.tokenize("") == []

    def test_multiple_spaces(self):
        idx = BM25Index()
        assert idx.tokenize("hello   world") == ["hello", "world"]


class TestBM25AddDocument:
    def test_count_after_add(self):
        idx = BM25Index()
        idx.add_document("doc1", "hello world")
        assert idx.count() == 1

    def test_count_after_multiple_adds(self):
        idx = BM25Index()
        idx.add_document("doc1", "hello world")
        idx.add_document("doc2", "foo bar baz")
        assert idx.count() == 2

    def test_avg_doc_length_computed(self):
        idx = BM25Index()
        idx.add_document("doc1", "hello world")
        assert idx.get_stats()["avg_doc_length"] == pytest.approx(2.0)
        idx.add_document("doc2", "one two three four")
        assert idx.get_stats()["avg_doc_length"] == pytest.approx(3.0)

    def test_add_documents_batch(self):
        idx = BM25Index()
        docs = [
            {"id": "d1", "content": "hello world"},
            {"id": "d2", "content": "foo bar"},
            {"id": "d3", "content": "baz qux"},
        ]
        added = idx.add_documents(docs)
        assert added == 3
        assert idx.count() == 3

    def test_add_documents_skips_empty_content(self):
        idx = BM25Index()
        docs = [
            {"id": "d1", "content": "hello"},
            {"id": "d2", "content": ""},
            {"id": "d3", "content": "world"},
        ]
        added = idx.add_documents(docs)
        assert added == 2

    def test_add_documents_skips_missing_id(self):
        idx = BM25Index()
        docs = [
            {"content": "hello"},
            {"id": "d1", "content": "world"},
        ]
        added = idx.add_documents(docs)
        assert added == 1


class TestBM25Search:
    def test_search_returns_scored_results(self):
        idx = BM25Index()
        idx.add_document("doc1", "machine learning algorithms")
        idx.add_document("doc2", "deep learning neural networks")
        idx.add_document("doc3", "cooking recipes for dinner")
        results = idx.search("machine learning", n_results=3)
        assert len(results) == 2
        assert all("score" in r for r in results)
        assert all(r["score"] > 0 for r in results)
        assert results[0]["score"] >= results[1]["score"]

    def test_search_result_has_correct_fields(self):
        idx = BM25Index()
        idx.add_document("doc1", "hello world", metadata={"source": "test"})
        results = idx.search("hello", n_results=1)
        assert len(results) == 1
        r = results[0]
        assert r["id"] == "doc1"
        assert r["content"] == "hello world"
        assert r["metadata"] == {"source": "test"}
        assert isinstance(r["score"], float)

    def test_search_no_match_returns_empty(self):
        idx = BM25Index()
        idx.add_document("doc1", "hello world")
        results = idx.search("quantum physics")
        assert results == []

    def test_search_empty_index_returns_empty(self):
        idx = BM25Index()
        assert idx.search("anything") == []

    def test_search_multi_term_query_scores_higher_for_both_terms(self):
        idx = BM25Index()
        idx.add_document("doc1", "machine learning is great")
        idx.add_document("doc2", "machine tools in the shop")
        idx.add_document("doc3", "learning to code is fun")
        results = idx.search("machine learning", n_results=3)
        assert results[0]["id"] == "doc1"

    def test_search_respects_n_results(self):
        idx = BM25Index()
        for i in range(10):
            idx.add_document(f"doc{i}", f"document number {i} about testing")
        results = idx.search("testing", n_results=3)
        assert len(results) == 3


class TestBM25IDF:
    def test_idf_rare_term_higher(self):
        idx = BM25Index()
        idx.add_document("d1", "unique rareword common")
        idx.add_document("d2", "common word here")
        idx.add_document("d3", "another common document")
        idf_rare = idx._compute_idf("rareword")
        idf_common = idx._compute_idf("common")
        assert idf_rare > idf_common

    def test_idf_absent_term_zero(self):
        idx = BM25Index()
        idx.add_document("d1", "hello world")
        assert idx._compute_idf("nonexistent") == 0.0


class TestBM25DocLengthNormalization:
    def test_same_tf_shorter_doc_scores_higher(self):
        idx = BM25Index(k1=1.5, b=0.75)
        idx.add_document("short", "python")
        idx.add_document("long", "python filler filler filler filler filler")
        results = idx.search("python", n_results=2)
        short_result = next(r for r in results if r["id"] == "short")
        long_result = next(r for r in results if r["id"] == "long")
        assert short_result["score"] > long_result["score"]


class TestBM25DeleteDocument:
    def test_delete_existing(self):
        idx = BM25Index()
        idx.add_document("doc1", "hello world")
        assert idx.delete_document("doc1") is True
        assert idx.count() == 0

    def test_delete_nonexistent(self):
        idx = BM25Index()
        assert idx.delete_document("nope") is False

    def test_delete_updates_avg_length(self):
        idx = BM25Index()
        idx.add_document("doc1", "hello world")
        idx.add_document("doc2", "one two three four five")
        idx.delete_document("doc1")
        assert idx.get_stats()["avg_doc_length"] == pytest.approx(5.0)

    def test_delete_last_doc_avg_length_zero(self):
        idx = BM25Index()
        idx.add_document("doc1", "hello world")
        idx.delete_document("doc1")
        assert idx.get_stats()["avg_doc_length"] == 0.0

    def test_search_after_delete_excludes_doc(self):
        idx = BM25Index()
        idx.add_document("doc1", "hello world")
        idx.add_document("doc2", "hello python")
        idx.delete_document("doc1")
        results = idx.search("hello")
        assert len(results) == 1
        assert results[0]["id"] == "doc2"


class TestBM25Clear:
    def test_clear_resets_everything(self):
        idx = BM25Index()
        idx.add_document("d1", "hello")
        idx.add_document("d2", "world")
        idx.clear()
        assert idx.count() == 0
        assert idx.get_stats()["avg_doc_length"] == 0.0
        assert idx.get_stats()["unique_terms"] == 0


class TestBM25GetStats:
    def test_stats_fields(self):
        idx = BM25Index(k1=1.2, b=0.8)
        idx.add_document("d1", "hello world test")
        stats = idx.get_stats()
        assert stats["total_docs"] == 1
        assert stats["avg_doc_length"] == pytest.approx(3.0)
        assert stats["unique_terms"] == 3
        assert stats["k1"] == 1.2
        assert stats["b"] == 0.8


class TestHierarchicalIndexAddChunk:
    def test_add_parent_chunk(self):
        hi = HierarchicalIndex()
        hi.add_chunk("p1", "parent content", hierarchy_level=0)
        assert hi.count() == 1

    def test_add_child_chunk_with_parent(self):
        hi = HierarchicalIndex()
        hi.add_chunk("p1", "parent content", hierarchy_level=0)
        hi.add_chunk("c1", "child content", parent_id="p1", hierarchy_level=1)
        assert hi.count() == 2

    def test_get_chunk(self):
        hi = HierarchicalIndex()
        hi.add_chunk("p1", "parent content", metadata={"source": "doc"})
        chunk = hi.get_chunk("p1")
        assert chunk is not None
        assert chunk["content"] == "parent content"
        assert chunk["metadata"] == {"source": "doc"}

    def test_get_chunk_nonexistent(self):
        hi = HierarchicalIndex()
        assert hi.get_chunk("nope") is None


class TestHierarchicalIndexRelationships:
    def test_get_parent(self):
        hi = HierarchicalIndex()
        hi.add_chunk("p1", "parent text", hierarchy_level=0)
        hi.add_chunk("c1", "child text", parent_id="p1", hierarchy_level=1)
        parent = hi.get_parent("c1")
        assert parent is not None
        assert parent["id"] == "p1"

    def test_get_parent_of_root_returns_none(self):
        hi = HierarchicalIndex()
        hi.add_chunk("p1", "parent", hierarchy_level=0)
        assert hi.get_parent("p1") is None

    def test_get_children(self):
        hi = HierarchicalIndex()
        hi.add_chunk("p1", "parent", hierarchy_level=0)
        hi.add_chunk("c1", "child1", parent_id="p1", hierarchy_level=1)
        hi.add_chunk("c2", "child2", parent_id="p1", hierarchy_level=1)
        children = hi.get_children("p1")
        assert len(children) == 2
        assert {c["id"] for c in children} == {"c1", "c2"}

    def test_get_children_empty(self):
        hi = HierarchicalIndex()
        hi.add_chunk("p1", "parent", hierarchy_level=0)
        assert hi.get_children("p1") == []

    def test_count_parents_and_children(self):
        hi = HierarchicalIndex()
        hi.add_chunk("p1", "p", hierarchy_level=0)
        hi.add_chunk("c1", "c", parent_id="p1", hierarchy_level=1)
        hi.add_chunk("c2", "c", parent_id="p1", hierarchy_level=1)
        assert hi.count_parents() == 1
        assert hi.count_children() == 2


class TestHierarchicalIndexEnhanceResults:
    def test_enhance_adds_parent_content(self):
        hi = HierarchicalIndex()
        hi.add_chunk("p1", "full parent document", hierarchy_level=0)
        hi.add_chunk("c1", "child excerpt", parent_id="p1", hierarchy_level=1)
        results = [{"id": "c1", "score": 0.9}]
        enhanced = hi.enhance_results_with_context(results)
        assert len(enhanced) == 1
        assert enhanced[0]["parent_content"] == "full parent document"
        assert enhanced[0]["parent_id"] == "p1"

    def test_enhance_without_parent(self):
        hi = HierarchicalIndex(include_parent_context=False)
        hi.add_chunk("p1", "parent", hierarchy_level=0)
        hi.add_chunk("c1", "child", parent_id="p1", hierarchy_level=1)
        results = [{"id": "c1", "score": 0.9}]
        enhanced = hi.enhance_results_with_context(results)
        assert "parent_content" not in enhanced[0]

    def test_enhance_override_include_parent(self):
        hi = HierarchicalIndex(include_parent_context=True)
        hi.add_chunk("p1", "parent", hierarchy_level=0)
        hi.add_chunk("c1", "child", parent_id="p1", hierarchy_level=1)
        results = [{"id": "c1", "score": 0.9}]
        enhanced = hi.enhance_results_with_context(results, include_parent=False)
        assert "parent_content" not in enhanced[0]


class TestHierarchicalIndexDeleteByReference:
    def test_delete_by_reference(self):
        hi = HierarchicalIndex()
        hi.add_chunk(
            "p1", "parent", hierarchy_level=0,
            metadata={"reference_name": "doc_a"},
        )
        hi.add_chunk(
            "c1", "child", parent_id="p1", hierarchy_level=1,
            metadata={"reference_name": "doc_a"},
        )
        hi.add_chunk(
            "p2", "other parent", hierarchy_level=0,
            metadata={"reference_name": "doc_b"},
        )
        deleted = hi.delete_by_reference("doc_a")
        assert deleted == 1
        assert hi.count() == 1
        assert hi.get_chunk("p2") is not None

    def test_delete_by_reference_no_match(self):
        hi = HierarchicalIndex()
        hi.add_chunk("p1", "parent", hierarchy_level=0)
        assert hi.delete_by_reference("nonexistent") == 0

    def test_delete_by_reference_independent_chunks(self):
        hi = HierarchicalIndex()
        hi.add_chunk("a1", "a1", hierarchy_level=0, metadata={"reference_name": "ref_x"})
        hi.add_chunk("a2", "a2", hierarchy_level=0, metadata={"reference_name": "ref_x"})
        deleted = hi.delete_by_reference("ref_x")
        assert deleted == 2
        assert hi.count() == 0


class TestHierarchicalIndexClear:
    def test_clear(self):
        hi = HierarchicalIndex()
        hi.add_chunk("p1", "parent", hierarchy_level=0)
        hi.add_chunk("c1", "child", parent_id="p1", hierarchy_level=1)
        hi.clear()
        assert hi.count() == 0


class TestHierarchicalIndexGetStats:
    def test_stats(self):
        hi = HierarchicalIndex(include_parent_context=True)
        hi.add_chunk("p1", "p", hierarchy_level=0)
        hi.add_chunk("c1", "c", parent_id="p1", hierarchy_level=1)
        stats = hi.get_stats()
        assert stats["total_chunks"] == 2
        assert stats["parent_chunks"] == 1
        assert stats["child_chunks"] == 1
        assert stats["include_parent_context"] is True


class TestHierarchicalIndexGetChildEmbeddings:
    def test_returns_ids_and_embeddings(self):
        hi = HierarchicalIndex()
        hi.add_chunk("p1", "p", hierarchy_level=0)
        hi.add_chunk("c1", "c1", parent_id="p1", hierarchy_level=1, embedding=[0.1, 0.2])
        hi.add_chunk("c2", "c2", parent_id="p1", hierarchy_level=1, embedding=[0.3, 0.4])
        ids, embs = hi.get_child_embeddings()
        assert ids == ["c1", "c2"]
        assert embs == [[0.1, 0.2], [0.3, 0.4]]

    def test_skips_children_without_embeddings(self):
        hi = HierarchicalIndex()
        hi.add_chunk("p1", "p", hierarchy_level=0)
        hi.add_chunk("c1", "c1", parent_id="p1", hierarchy_level=1, embedding=[0.1])
        hi.add_chunk("c2", "c2", parent_id="p1", hierarchy_level=1)
        ids, embs = hi.get_child_embeddings()
        assert ids == ["c1"]
        assert embs == [[0.1]]


class TestContextualEmbeddingsPrepareChunk:
    def test_with_title_and_section(self):
        ce = ContextualEmbeddings()
        result = ce.prepare_chunk_for_embedding(
            chunk_content="body text",
            document_title="My Doc",
            section_header="Intro",
        )
        assert "Document: My Doc" in result
        assert "Section: Intro" in result
        assert "body text" in result

    def test_with_title_only(self):
        ce = ContextualEmbeddings()
        result = ce.prepare_chunk_for_embedding(
            chunk_content="body",
            document_title="Doc",
        )
        assert "Document: Doc" in result
        assert "Section:" not in result
        assert result.endswith("body")

    def test_no_context_returns_chunk_unchanged(self):
        ce = ContextualEmbeddings()
        result = ce.prepare_chunk_for_embedding(chunk_content="raw text")
        assert result == "raw text"

    def test_truncation_applied_to_long_title(self):
        ce = ContextualEmbeddings(max_context_length=30)
        result = ce.prepare_chunk_for_embedding(
            chunk_content="body",
            document_title="A" * 100,
            section_header="B" * 100,
        )
        context_part = result.split("\n\n")[0]
        assert len(context_part) <= 35


class TestContextualEmbeddingsBatch:
    def test_batch_preparation(self):
        ce = ContextualEmbeddings()
        chunks = [
            {"content": "first chunk", "metadata": {"header": "Section 1"}},
            {"content": "second chunk", "metadata": {}},
        ]
        results = ce.prepare_chunks_batch(chunks, document_title="My Doc")
        assert len(results) == 2
        assert "My Doc" in results[0]
        assert "Section 1" in results[0]
        assert "My Doc" in results[1]

    def test_batch_uses_per_chunk_title_from_metadata(self):
        ce = ContextualEmbeddings()
        chunks = [
            {
                "content": "chunk1",
                "metadata": {"document_title": "Custom Title"},
            },
            {"content": "chunk2", "metadata": {}},
        ]
        results = ce.prepare_chunks_batch(chunks, document_title="Fallback")
        assert "Custom Title" in results[0]
        assert "Fallback" in results[1]

    def test_empty_batch(self):
        ce = ContextualEmbeddings()
        assert ce.prepare_chunks_batch([]) == []


class TestContextualEmbeddingsTruncation:
    def test_short_text_unchanged(self):
        ce = ContextualEmbeddings()
        assert ce._truncate("short", 100) == "short"

    def test_empty_string(self):
        ce = ContextualEmbeddings()
        assert ce._truncate("", 100) == ""

    def test_long_text_truncated(self):
        ce = ContextualEmbeddings()
        result = ce._truncate("a " * 100, 20)
        assert len(result) <= 23
        assert result.endswith("...")


class TestChunkDeduplicatorExact:
    def test_first_content_not_duplicate(self):
        dd = ChunkDeduplicator(mode="exact")
        assert dd.is_duplicate("hello world") is False

    def test_identical_content_is_duplicate(self):
        dd = ChunkDeduplicator(mode="exact")
        dd.is_duplicate("hello world")
        assert dd.is_duplicate("hello world") is True

    def test_different_content_not_duplicate(self):
        dd = ChunkDeduplicator(mode="exact")
        dd.is_duplicate("hello")
        assert dd.is_duplicate("world") is False

    def test_compute_hash_deterministic(self):
        dd = ChunkDeduplicator(mode="exact")
        h1 = dd.compute_hash("test content")
        h2 = dd.compute_hash("test content")
        assert h1 == h2

    def test_compute_hash_different_for_different_content(self):
        dd = ChunkDeduplicator(mode="exact")
        h1 = dd.compute_hash("aaa")
        h2 = dd.compute_hash("bbb")
        assert h1 != h2

    def test_compute_hash_length(self):
        dd = ChunkDeduplicator(mode="exact")
        h = dd.compute_hash("test")
        assert len(h) == 16


class TestChunkDeduplicatorSimilarity:
    def test_identical_embedding_is_duplicate(self):
        dd = ChunkDeduplicator(mode="similarity", similarity_threshold=0.95)
        emb = [1.0, 0.0, 0.0]
        assert dd.is_duplicate("text a", emb) is False
        assert dd.is_duplicate("text b", emb) is True

    def test_orthogonal_embedding_not_duplicate(self):
        dd = ChunkDeduplicator(mode="similarity", similarity_threshold=0.95)
        assert dd.is_duplicate("a", [1.0, 0.0, 0.0]) is False
        assert dd.is_duplicate("b", [0.0, 1.0, 0.0]) is False

    def test_near_threshold(self):
        dd = ChunkDeduplicator(mode="similarity", similarity_threshold=0.90)
        assert dd.is_duplicate("a", [1.0, 0.0]) is False
        assert dd.is_duplicate("b", [0.95, 0.312]) is True


class TestChunkDeduplicatorFilterDuplicates:
    def test_filter_exact_removes_duplicates(self):

        class Chunk:
            def __init__(self, content):
                self.content = content

        dd = ChunkDeduplicator(mode="exact")
        chunks = [Chunk("hello"), Chunk("world"), Chunk("hello")]
        embeddings = [[0.1], [0.2], [0.3]]
        filtered_c, filtered_e = dd.filter_duplicates(chunks, embeddings)
        assert len(filtered_c) == 2
        assert filtered_c[0].content == "hello"
        assert filtered_c[1].content == "world"
        assert filtered_e == [[0.1], [0.2]]

    def test_filter_empty_input(self):
        dd = ChunkDeduplicator(mode="exact")
        c, e = dd.filter_duplicates([], [])
        assert c == []
        assert e == []


class TestChunkDeduplicatorClear:
    def test_clear_resets_state(self):
        dd = ChunkDeduplicator(mode="exact")
        dd.is_duplicate("hello")
        dd.clear()
        assert dd.is_duplicate("hello") is False


class TestChunkDeduplicatorGetStats:
    def test_stats_exact_mode(self):
        dd = ChunkDeduplicator(mode="exact", similarity_threshold=0.95)
        dd.is_duplicate("a")
        dd.is_duplicate("b")
        stats = dd.get_stats()
        assert stats["mode"] == "exact"
        assert stats["similarity_threshold"] == 0.95
        assert stats["seen_hashes"] == 2
        assert stats["seen_embeddings"] == 0

    def test_stats_similarity_mode(self):
        dd = ChunkDeduplicator(mode="similarity", similarity_threshold=0.9)
        dd.is_duplicate("a", [1.0, 0.0])
        stats = dd.get_stats()
        assert stats["seen_embeddings"] == 1
