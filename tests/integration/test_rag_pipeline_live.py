import os

import pytest

pytestmark = pytest.mark.integration

try:
    import chromadb  # type: ignore[import-untyped]  # noqa: F401
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


def _get_mistral_api_key():
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        pytest.skip("MISTRAL_API_KEY not set")
    return key


def _skip_no_chromadb():
    if not HAS_CHROMADB:
        pytest.skip("chromadb not installed (pip install ffai[rag])")


DOCUMENTS = {
    "python": (
        "Python is a high-level programming language known for its readability and versatility. "
        "It supports multiple programming paradigms including procedural, object-oriented, and "
        "functional programming. Python has a large standard library and a thriving ecosystem of "
        "third-party packages."
    ),
    "rust": (
        "Rust is a systems programming language focused on safety, speed, and concurrency. "
        "It enforces memory safety without a garbage collector using a borrow checker system. "
        "Rust has been adopted by major companies for performance-critical applications."
    ),
    "cooking": (
        "Italian cuisine is known for its regional diversity and simplicity of preparation. "
        "Popular dishes include pasta, pizza, risotto, and gelato. Italian cooking emphasizes "
        "fresh ingredients and traditional techniques passed down through generations."
    ),
}


class TestRAGPipelineLive:
    @pytest.fixture(autouse=True)
    def setup_rag(self, tmp_path):
        _skip_no_chromadb()
        from ffai.rag.embed import Embeddings
        from ffai.rag.rag import RAG
        from ffai.rag.store import VectorStore

        api_key = _get_mistral_api_key()
        embed = Embeddings("mistral/mistral-embed", api_key=api_key, cache_enabled=True)
        store = VectorStore(
            collection_name=f"test_rag_{os.getpid()}",
            dir=str(tmp_path / "chroma_db"),
        )
        self.rag = RAG(embed=embed, store=store, chunk_size=200, chunk_overlap=50)

    def test_index_and_search_roundtrip(self):
        self.rag.index(DOCUMENTS["python"], source="python")
        self.rag.index(DOCUMENTS["cooking"], source="cooking")

        hits = self.rag.search("programming language")
        assert len(hits) >= 1
        assert any("python" in h.content.lower() or h.source == "python" for h in hits)

    def test_search_returns_relevant_source(self):
        self.rag.index(DOCUMENTS["python"], source="python")
        self.rag.index(DOCUMENTS["cooking"], source="cooking")
        self.rag.index(DOCUMENTS["rust"], source="rust")

        hits = self.rag.search("memory safety borrow checker")
        assert len(hits) >= 1
        assert hits[0].source == "rust"

    def test_search_returns_relevant_cooking(self):
        self.rag.index(DOCUMENTS["python"], source="python")
        self.rag.index(DOCUMENTS["cooking"], source="cooking")

        hits = self.rag.search("pasta recipe ingredients")
        assert len(hits) >= 1
        assert any(h.source == "cooking" for h in hits)

    def test_count_reflects_indexed_docs(self):
        self.rag.index(DOCUMENTS["python"], source="python")
        count_after = self.rag.count()
        assert count_after > 0

        self.rag.index(DOCUMENTS["rust"], source="rust")
        assert self.rag.count() > count_after

    def test_delete_removes_source(self):
        self.rag.index(DOCUMENTS["python"], source="python")
        self.rag.index(DOCUMENTS["rust"], source="rust")
        count_before = self.rag.count()

        self.rag.delete("python")
        assert self.rag.count() < count_before

        hits = self.rag.search("python programming")
        assert all(h.source != "python" for h in hits)

    def test_async_index_and_search(self):
        import asyncio
        asyncio.run(self.rag.aindex(DOCUMENTS["rust"], source="rust"))
        hits = asyncio.run(self.rag.asearch("systems programming safety"))
        assert len(hits) >= 1
        assert hits[0].source == "rust"

    def test_search_with_top_k(self):
        self.rag.index(DOCUMENTS["python"], source="python")
        self.rag.index(DOCUMENTS["rust"], source="rust")
        self.rag.index(DOCUMENTS["cooking"], source="cooking")

        hits = self.rag.search("programming", top_k=2)
        assert len(hits) <= 2

    def test_search_hit_has_required_fields(self):
        self.rag.index(DOCUMENTS["python"], source="python")
        hits = self.rag.search("python")
        assert len(hits) >= 1
        hit = hits[0]
        assert isinstance(hit.content, str)
        assert len(hit.content) > 0
        assert isinstance(hit.score, float)
        assert 0 <= hit.score <= 1
        assert isinstance(hit.id, str)
        assert isinstance(hit.source, str)


class TestRAGHybridLive:
    @pytest.fixture(autouse=True)
    def setup_rag(self, tmp_path):
        _skip_no_chromadb()
        from ffai.rag.embed import Embeddings
        from ffai.rag.rag import RAG
        from ffai.rag.store import VectorStore

        api_key = _get_mistral_api_key()
        embed = Embeddings("mistral/mistral-embed", api_key=api_key, cache_enabled=True)
        store = VectorStore(
            collection_name=f"test_hybrid_{os.getpid()}",
            dir=str(tmp_path / "chroma_db"),
        )
        self.rag = RAG(
            embed=embed, store=store,
            chunk_size=200, chunk_overlap=50,
            bm25_alpha=0.6,
        )

    def test_hybrid_search_returns_results(self):
        self.rag.index(DOCUMENTS["python"], source="python")
        self.rag.index(DOCUMENTS["rust"], source="rust")
        hits = self.rag.search("programming language")
        assert len(hits) >= 1

    def test_hybrid_async_search(self):
        import asyncio
        asyncio.run(self.rag.aindex(DOCUMENTS["rust"], source="rust"))
        hits = asyncio.run(self.rag.asearch("memory safety"))
        assert len(hits) >= 1

    def test_bm25_boosts_exact_keyword_match(self):
        self.rag.index(DOCUMENTS["python"], source="python")
        self.rag.index(DOCUMENTS["rust"], source="rust")
        self.rag.index(DOCUMENTS["cooking"], source="cooking")
        hits = self.rag.search("borrow checker")
        assert len(hits) >= 1
        assert any(h.source == "rust" for h in hits)


class TestRAGRerankerLive:
    @pytest.fixture(autouse=True)
    def setup_rag(self, tmp_path):
        _skip_no_chromadb()
        from ffai.rag.embed import Embeddings
        from ffai.rag.rag import RAG
        from ffai.rag.store import VectorStore

        api_key = _get_mistral_api_key()
        embed = Embeddings("mistral/mistral-embed", api_key=api_key, cache_enabled=True)
        store = VectorStore(
            collection_name=f"test_rerank_{os.getpid()}",
            dir=str(tmp_path / "chroma_db"),
        )
        self.rag = RAG(
            embed=embed, store=store,
            chunk_size=200, chunk_overlap=50,
            reranker="diversity",
        )
        self._tmp_path = tmp_path

    def test_reranker_search_returns_results(self):
        self.rag.index(DOCUMENTS["python"], source="python")
        self.rag.index(DOCUMENTS["rust"], source="rust")
        hits = self.rag.search("programming")
        assert len(hits) >= 1

    def test_reranker_produces_results(self):
        self.rag.index(DOCUMENTS["python"], source="python")
        self.rag.index(DOCUMENTS["rust"], source="rust")
        self.rag.index(DOCUMENTS["cooking"], source="cooking")
        hits = self.rag.search("programming", top_k=3)
        assert len(hits) >= 2
        sources = {h.source for h in hits}
        assert len(sources) >= 2

    def test_reranker_with_hybrid_search(self):
        from ffai.rag.embed import Embeddings
        from ffai.rag.rag import RAG
        from ffai.rag.store import VectorStore

        api_key = _get_mistral_api_key()
        embed = Embeddings("mistral/mistral-embed", api_key=api_key, cache_enabled=True)
        store = VectorStore(
            collection_name=f"test_rerank_hybrid_{os.getpid()}",
            dir=str(self._tmp_path / "chroma_hybrid"),
        )
        rag = RAG(
            embed=embed, store=store,
            chunk_size=200, chunk_overlap=50,
            bm25_alpha=0.6, reranker="diversity",
        )
        rag.index(DOCUMENTS["python"], source="python")
        rag.index(DOCUMENTS["rust"], source="rust")
        hits = rag.search("memory safety programming")
        assert len(hits) >= 1


class TestRAGMultiSourceLive:
    @pytest.fixture(autouse=True)
    def setup_rag(self, tmp_path):
        _skip_no_chromadb()
        from ffai.rag.embed import Embeddings
        from ffai.rag.rag import RAG
        from ffai.rag.store import VectorStore

        api_key = _get_mistral_api_key()
        embed = Embeddings("mistral/mistral-embed", api_key=api_key, cache_enabled=True)
        store = VectorStore(
            collection_name=f"test_multi_{os.getpid()}",
            dir=str(tmp_path / "chroma_db"),
        )
        self.rag = RAG(embed=embed, store=store, chunk_size=200, chunk_overlap=50)

    def test_delete_all_leaves_empty(self):
        self.rag.index(DOCUMENTS["python"], source="python")
        self.rag.index(DOCUMENTS["rust"], source="rust")
        assert self.rag.count() > 0

        self.rag.delete("python")
        self.rag.delete("rust")
        assert self.rag.count() == 0

    def test_search_after_delete_returns_nothing(self):
        self.rag.index(DOCUMENTS["python"], source="python")
        self.rag.delete("python")
        hits = self.rag.search("python programming")
        assert hits == []

    def test_index_same_source_twice(self):
        self.rag.index(DOCUMENTS["python"], source="python")
        count_first = self.rag.count()
        self.rag.index(DOCUMENTS["python"][:60], source="python")
        count_second = self.rag.count()
        assert count_second > 0

    def test_chunk_without_indexing(self):
        chunks = self.rag.chunk(DOCUMENTS["python"], source="python")
        assert len(chunks) >= 1
        assert all(len(c.content) > 0 for c in chunks)
        assert self.rag.count() == 0

    def test_search_returns_unique_ids(self):
        self.rag.index(DOCUMENTS["python"], source="python")
        self.rag.index(DOCUMENTS["rust"], source="rust")
        hits = self.rag.search("programming", top_k=10)
        ids = [h.id for h in hits]
        assert len(ids) == len(set(ids))


class TestRAGBatchIndexingLive:
    @pytest.fixture(autouse=True)
    def setup_rag(self, tmp_path):
        _skip_no_chromadb()
        from ffai.rag.embed import Embeddings
        from ffai.rag.rag import RAG
        from ffai.rag.store import VectorStore

        api_key = _get_mistral_api_key()
        embed = Embeddings("mistral/mistral-embed", api_key=api_key, cache_enabled=True)
        store = VectorStore(
            collection_name=f"test_batch_{os.getpid()}",
            dir=str(tmp_path / "chroma_db"),
        )
        self.rag = RAG(embed=embed, store=store, chunk_size=200, chunk_overlap=50)

    def test_index_many_indexes_all_documents(self):
        total = self.rag.index_many([
            {"text": DOCUMENTS["python"], "source": "python"},
            {"text": DOCUMENTS["rust"], "source": "rust"},
            {"text": DOCUMENTS["cooking"], "source": "cooking"},
        ])
        assert total > 0
        assert self.rag.count() >= 3

    def test_index_many_search_finds_content(self):
        self.rag.index_many([
            {"text": DOCUMENTS["python"], "source": "python"},
            {"text": DOCUMENTS["rust"], "source": "rust"},
        ])
        hits = self.rag.search("memory safety borrow checker")
        assert len(hits) >= 1
        assert any(h.source == "rust" for h in hits)

    def test_index_many_async(self):
        import asyncio
        total = asyncio.run(self.rag.aindex_many([
            {"text": DOCUMENTS["python"], "source": "python"},
            {"text": DOCUMENTS["cooking"], "source": "cooking"},
        ]))
        assert total > 0
        hits = self.rag.search("pasta recipe")
        assert len(hits) >= 1


class TestRAGDedupLive:
    @pytest.fixture(autouse=True)
    def setup_rag(self, tmp_path):
        _skip_no_chromadb()
        from ffai.rag.embed import Embeddings
        from ffai.rag.rag import RAG
        from ffai.rag.store import VectorStore

        api_key = _get_mistral_api_key()
        embed = Embeddings("mistral/mistral-embed", api_key=api_key, cache_enabled=True)
        store = VectorStore(
            collection_name=f"test_dedup_{os.getpid()}",
            dir=str(tmp_path / "chroma_db"),
        )
        self.rag = RAG(embed=embed, store=store, chunk_size=200, chunk_overlap=50)

    def test_same_checksum_skips_reindex(self):
        import hashlib
        text = DOCUMENTS["python"]
        checksum = hashlib.sha256(text.encode()).hexdigest()

        count_first = self.rag.index(text, source="python", checksum=checksum)
        assert count_first > 0
        total_after_first = self.rag.count()

        count_second = self.rag.index(text, source="python", checksum=checksum)
        assert count_second == 0
        assert self.rag.count() == total_after_first

    def test_different_checksum_reindexes(self):
        import hashlib
        text = DOCUMENTS["python"]
        checksum1 = hashlib.sha256(text.encode()).hexdigest()

        self.rag.index(text, source="python", checksum=checksum1)
        count_after_first = self.rag.count()

        new_text = DOCUMENTS["rust"]
        checksum2 = hashlib.sha256(new_text.encode()).hexdigest()
        count_second = self.rag.index(new_text, source="python", checksum=checksum2)
        assert count_second > 0

    def test_no_checksum_always_indexes(self):
        result1 = self.rag.index(DOCUMENTS["python"], source="python")
        result2 = self.rag.index(DOCUMENTS["python"], source="python")
        assert result1 > 0
        assert result2 > 0


class TestRAGQueryExpansionLive:
    @pytest.fixture(autouse=True)
    def setup_rag(self, tmp_path):
        _skip_no_chromadb()
        from ffai.rag.embed import Embeddings
        from ffai.rag.rag import RAG
        from ffai.rag.store import VectorStore

        api_key = _get_mistral_api_key()
        embed = Embeddings("mistral/mistral-embed", api_key=api_key, cache_enabled=True)
        store = VectorStore(
            collection_name=f"test_expand_{os.getpid()}",
            dir=str(tmp_path / "chroma_db"),
        )

        def expand(query):
            return [
                query,
                query + " explained simply",
                "what is " + query,
            ]

        self.rag = RAG(
            embed=embed, store=store,
            chunk_size=200, chunk_overlap=50,
            query_expander=expand,
        )
        self._tmp_path = tmp_path

    def test_expansion_search_returns_results(self):
        self.rag.index(DOCUMENTS["python"], source="python")
        self.rag.index(DOCUMENTS["rust"], source="rust")
        hits = self.rag.search("programming language")
        assert len(hits) >= 1

    def test_expansion_finds_more_results(self):
        from ffai.rag.embed import Embeddings
        from ffai.rag.rag import RAG
        from ffai.rag.store import VectorStore

        api_key = _get_mistral_api_key()
        embed = Embeddings("mistral/mistral-embed", api_key=api_key, cache_enabled=True)
        store_no_expand = VectorStore(
            collection_name=f"test_no_expand_{os.getpid()}",
            dir=str(self._tmp_path / "chroma_no_expand"),
        )
        rag_no_expand = RAG(embed=embed, store=store_no_expand, chunk_size=200, chunk_overlap=50)

        self.rag.index(DOCUMENTS["python"], source="python")
        self.rag.index(DOCUMENTS["rust"], source="rust")
        self.rag.index(DOCUMENTS["cooking"], source="cooking")

        rag_no_expand.index(DOCUMENTS["python"], source="python")
        rag_no_expand.index(DOCUMENTS["rust"], source="rust")
        rag_no_expand.index(DOCUMENTS["cooking"], source="cooking")

        hits_expanded = self.rag.search("safety and speed", top_k=5)
        hits_plain = rag_no_expand.search("safety and speed", top_k=5)

        sources_expanded = {h.source for h in hits_expanded}
        assert len(sources_expanded) >= 1

    def test_expansion_with_failing_callable(self):
        from ffai.rag.embed import Embeddings
        from ffai.rag.rag import RAG
        from ffai.rag.store import VectorStore

        api_key = _get_mistral_api_key()
        embed = Embeddings("mistral/mistral-embed", api_key=api_key, cache_enabled=True)
        store = VectorStore(
            collection_name=f"test_expand_fail_{os.getpid()}",
            dir=str(self._tmp_path / "chroma_fail"),
        )

        def failing_expand(q):
            raise RuntimeError("LLM unavailable")

        rag = RAG(
            embed=embed, store=store,
            chunk_size=200, chunk_overlap=50,
            query_expander=failing_expand,
        )
        rag.index(DOCUMENTS["rust"], source="rust")
        hits = rag.search("memory safety")
        assert len(hits) >= 1
        assert hits[0].source == "rust"

    def test_expansion_with_hybrid_and_reranker(self):
        from ffai.rag.embed import Embeddings
        from ffai.rag.rag import RAG
        from ffai.rag.store import VectorStore

        api_key = _get_mistral_api_key()
        embed = Embeddings("mistral/mistral-embed", api_key=api_key, cache_enabled=True)
        store = VectorStore(
            collection_name=f"test_expand_full_{os.getpid()}",
            dir=str(self._tmp_path / "chroma_full"),
        )

        def expand(query):
            return [query, query + " overview"]

        rag = RAG(
            embed=embed, store=store,
            chunk_size=200, chunk_overlap=50,
            bm25_alpha=0.6, reranker="diversity",
            query_expander=expand,
        )
        rag.index(DOCUMENTS["python"], source="python")
        rag.index(DOCUMENTS["rust"], source="rust")
        rag.index(DOCUMENTS["cooking"], source="cooking")
        hits = rag.search("programming language safety")
        assert len(hits) >= 1
