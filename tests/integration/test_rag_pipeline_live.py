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
        from src.rag.embed import Embeddings
        from src.rag.rag import RAG
        from src.rag.store import VectorStore

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
        from src.rag.embed import Embeddings
        from src.rag.rag import RAG
        from src.rag.store import VectorStore

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
