from __future__ import annotations

from ffai.rag.indexing.bm25 import BM25Index


class TestBM25Init:
    def test_default_params(self):
        idx = BM25Index()
        assert idx.k1 == 1.5
        assert idx.b == 0.75
        assert idx.count() == 0

    def test_custom_params(self):
        idx = BM25Index(k1=2.0, b=0.5, epsilon=0.1)
        assert idx.k1 == 2.0
        assert idx.b == 0.5


class TestBM25Tokenize:
    def test_lowercase(self):
        idx = BM25Index()
        assert idx.tokenize("Hello World") == ["hello", "world"]

    def test_removes_punctuation(self):
        idx = BM25Index()
        assert idx.tokenize("it's a test!") == ["it", "test"]

    def test_filters_short_tokens(self):
        idx = BM25Index()
        assert idx.tokenize("I am a person") == ["am", "person"]

    def test_empty_string(self):
        idx = BM25Index()
        assert idx.tokenize("") == []


class TestBM25AddDocument:
    def test_single_document(self):
        idx = BM25Index()
        idx.add_document("d1", "the quick brown fox")
        assert idx.count() == 1

    def test_with_metadata(self):
        idx = BM25Index()
        idx.add_document("d1", "test content", metadata={"source": "file.txt"})
        stats = idx.get_stats()
        assert stats["total_docs"] == 1

    def test_duplicate_id_overwrites(self):
        idx = BM25Index()
        idx.add_document("d1", "first version")
        idx.add_document("d1", "second version longer text")
        assert idx.count() == 1
        results = idx.search("second version")
        assert len(results) == 1
        assert results[0]["content"] == "second version longer text"

    def test_duplicate_id_updates_term_freqs(self):
        idx = BM25Index()
        idx.add_document("d1", "apple apple apple")
        idx.add_document("d2", "banana banana")
        idx.add_document("d1", "orange orange")
        results = idx.search("apple")
        assert len(results) == 0
        results = idx.search("orange")
        assert len(results) == 1
        assert results[0]["id"] == "d1"

    def test_add_documents_batch(self):
        idx = BM25Index()
        docs = [
            {"id": "d1", "content": "first document"},
            {"id": "d2", "content": "second document"},
            {"id": "", "content": "skipped no id"},
        ]
        count = idx.add_documents(docs)
        assert count == 2
        assert idx.count() == 2


class TestBM25Search:
    def test_returns_relevant_results(self):
        idx = BM25Index()
        idx.add_document("d1", "machine learning algorithms")
        idx.add_document("d2", "cooking recipes for dinner")
        idx.add_document("d3", "deep learning neural networks")
        results = idx.search("machine learning")
        assert len(results) >= 1
        assert results[0]["id"] == "d1"
        assert results[0]["score"] > 0

    def test_respects_n_results(self):
        idx = BM25Index()
        for i in range(10):
            idx.add_document(f"d{i}", f"document number {i} about testing")
        results = idx.search("testing", n_results=3)
        assert len(results) == 3

    def test_empty_index_returns_empty(self):
        idx = BM25Index()
        assert idx.search("anything") == []

    def test_no_matching_terms_returns_empty(self):
        idx = BM25Index()
        idx.add_document("d1", "quantum physics")
        assert idx.search("cooking recipes") == []

    def test_results_sorted_by_score_descending(self):
        idx = BM25Index()
        idx.add_document("d1", "cat cat cat")
        idx.add_document("d2", "cat")
        idx.add_document("d3", "cat cat")
        results = idx.search("cat")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_results_include_content_and_metadata(self):
        idx = BM25Index()
        idx.add_document("d1", "hello world", metadata={"source": "greeting.txt"})
        results = idx.search("hello")
        assert results[0]["content"] == "hello world"
        assert results[0]["metadata"]["source"] == "greeting.txt"


class TestBM25Delete:
    def test_delete_existing_document(self):
        idx = BM25Index()
        idx.add_document("d1", "test document")
        assert idx.delete_document("d1") is True
        assert idx.count() == 0

    def test_delete_nonexistent_returns_false(self):
        idx = BM25Index()
        assert idx.delete_document("nope") is False

    def test_delete_by_metadata(self):
        idx = BM25Index()
        idx.add_document("d1", "doc one", metadata={"source": "a.txt"})
        idx.add_document("d2", "doc two", metadata={"source": "b.txt"})
        idx.add_document("d3", "doc three", metadata={"source": "a.txt"})
        deleted = idx.delete_by_metadata("source", "a.txt")
        assert deleted == 2
        assert idx.count() == 1

    def test_deleted_doc_excluded_from_search(self):
        idx = BM25Index()
        idx.add_document("d1", "unique keyword xyz")
        idx.add_document("d2", "other document")
        idx.delete_document("d1")
        results = idx.search("xyz")
        assert len(results) == 0


class TestBM25Clear:
    def test_clear_resets_index(self):
        idx = BM25Index()
        idx.add_document("d1", "test")
        idx.add_document("d2", "another")
        idx.clear()
        assert idx.count() == 0
        assert idx.get_stats()["avg_doc_length"] == 0.0
        assert idx.get_stats()["unique_terms"] == 0


class TestBM25Stats:
    def test_stats_reflect_index_state(self):
        idx = BM25Index()
        idx.add_document("d1", "hello world hello")
        stats = idx.get_stats()
        assert stats["total_docs"] == 1
        assert stats["avg_doc_length"] == 3.0
        assert stats["unique_terms"] == 2
        assert stats["k1"] == 1.5
        assert stats["b"] == 0.75
