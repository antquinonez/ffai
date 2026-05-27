import os

import pytest

from src.FFAI import FFAI
from src.rag.types import QueryResult

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


class TestRAGQueryLive:
    @pytest.fixture(autouse=True)
    def setup_rag(self, tmp_path):
        _skip_no_chromadb()
        from src.rag.embed import Embeddings
        from src.rag.rag import RAG
        from src.rag.store import VectorStore

        api_key = _get_mistral_api_key()
        embed = Embeddings("mistral/mistral-embed", api_key=api_key, cache_enabled=True)
        store = VectorStore(
            collection_name=f"test_rag_query_{os.getpid()}",
            dir=str(tmp_path / "chroma_db"),
        )
        self.rag = RAG(embed=embed, store=store, chunk_size=200, chunk_overlap=50)

    def test_query_with_mock_llm(self):
        self.rag.index(DOCUMENTS["python"], source="python")
        self.rag.index(DOCUMENTS["rust"], source="rust")
        self.rag.index(DOCUMENTS["cooking"], source="cooking")

        result = self.rag.query(
            "What programming language focuses on memory safety?",
            generate_fn=lambda prompt: "Rust",
        )
        assert isinstance(result, QueryResult)
        assert result.answer == "Rust"
        assert len(result.hits) >= 1
        assert "rust" in result.sources
        assert "memory safety" in result.prompt.lower() or "rust" in result.prompt.lower()

    def test_query_sources_match_search_results(self):
        self.rag.index(DOCUMENTS["python"], source="python")
        self.rag.index(DOCUMENTS["cooking"], source="cooking")

        result = self.rag.query(
            "Tell me about Italian food",
            generate_fn=lambda prompt: "Italian cuisine",
        )
        hit_sources = {h.source for h in result.hits if h.source}
        for source in result.sources:
            assert source in hit_sources

    def test_query_custom_template(self):
        self.rag.index(DOCUMENTS["python"], source="python")

        template = "BACKGROUND:\n{context}\n\nQUERY:\n{question}\n\nANSWER:"
        result = self.rag.query(
            "What is Python?",
            generate_fn=lambda prompt: "A programming language",
            prompt_template=template,
        )
        assert "BACKGROUND:" in result.prompt
        assert "QUERY:" in result.prompt
        assert "ANSWER:" in result.prompt

    def test_query_empty_index(self):
        result = self.rag.query(
            "What is Python?",
            generate_fn=lambda prompt: "I have no information about that.",
        )
        assert result.answer == "I have no information about that."
        assert result.hits == []
        assert result.sources == []

    def test_aquery_async_path(self):
        import asyncio

        self.rag.index(DOCUMENTS["rust"], source="rust")

        result = asyncio.run(self.rag.aquery(
            "What language uses a borrow checker?",
            generate_fn=lambda prompt: "Rust uses a borrow checker.",
        ))
        assert "borrow checker" in result.answer
        assert len(result.hits) >= 1
        assert "rust" in result.sources

    def test_query_with_hybrid_search(self, tmp_path):
        from src.rag.embed import Embeddings
        from src.rag.rag import RAG
        from src.rag.store import VectorStore

        api_key = _get_mistral_api_key()
        embed = Embeddings("mistral/mistral-embed", api_key=api_key, cache_enabled=True)
        store = VectorStore(
            collection_name=f"test_query_hybrid_{os.getpid()}",
            dir=str(tmp_path / "chroma_hybrid"),
        )
        rag = RAG(embed=embed, store=store, chunk_size=200, chunk_overlap=50, bm25_alpha=0.5)
        rag.index(DOCUMENTS["python"], source="python")
        rag.index(DOCUMENTS["rust"], source="rust")

        result = rag.query(
            "safety and performance",
            generate_fn=lambda prompt: "Rust",
        )
        assert result.answer == "Rust"
        assert len(result.hits) >= 1


class TestFFAIQueryLive:
    @pytest.fixture(autouse=True)
    def setup_ffai(self, tmp_path):
        _skip_no_chromadb()
        from dotenv import load_dotenv

        load_dotenv()
        from src.Clients.FFLiteLLMClient import FFLiteLLMClient
        from src.rag.embed import Embeddings
        from src.rag.rag import RAG
        from src.rag.store import VectorStore

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            pytest.skip("MISTRAL_API_KEY not set")

        client = FFLiteLLMClient(
            model_string="mistral/mistral-small-2503",
            api_key=api_key,
            temperature=0,
            max_tokens=100,
            system_instructions="Be concise. Answer in one sentence.",
        )
        embed = Embeddings("mistral/mistral-embed", api_key=api_key, cache_enabled=True)
        store = VectorStore(
            collection_name=f"test_ffai_query_{os.getpid()}",
            dir=str(tmp_path / "chroma_db"),
        )
        rag = RAG(embed=embed, store=store, chunk_size=200, chunk_overlap=50)
        self.ffai = FFAI(client=client, rag=rag)

    def test_ffai_query_returns_answer(self):
        assert self.ffai.rag is not None
        self.ffai.rag.index(DOCUMENTS["python"], source="python")
        self.ffai.rag.index(DOCUMENTS["rust"], source="rust")

        result = self.ffai.query("What is Python?")
        assert isinstance(result, QueryResult)
        assert len(result.answer) > 0
        assert len(result.hits) >= 1
        assert "python" in result.sources

    def test_ffai_query_without_rag_raises(self):
        from tests.conftest import ConcreteClient

        ffai = FFAI(client=ConcreteClient())
        with pytest.raises(ValueError, match="RAG is not configured"):
            ffai.query("test")

    def test_ffai_query_answer_references_context(self):
        assert self.ffai.rag is not None
        self.ffai.rag.index(DOCUMENTS["cooking"], source="cooking")

        result = self.ffai.query("What is Italian cuisine known for?")
        assert len(result.answer) > 0
        assert "cooking" in result.sources
        assert "prompt" in result.prompt.lower() or "italian" in result.answer.lower()

    def test_ffai_query_with_hybrid_and_reranker(self, tmp_path):
        from src.rag.embed import Embeddings
        from src.rag.rag import RAG
        from src.rag.store import VectorStore

        api_key = os.getenv("MISTRAL_API_KEY")
        embed = Embeddings("mistral/mistral-embed", api_key=api_key, cache_enabled=True)
        store = VectorStore(
            collection_name=f"test_ffai_query_hybrid_{os.getpid()}",
            dir=str(tmp_path / "chroma_hybrid"),
        )
        rag = RAG(
            embed=embed, store=store,
            chunk_size=200, chunk_overlap=50,
            bm25_alpha=0.5, reranker="diversity",
        )
        self.ffai.rag = rag
        rag.index(DOCUMENTS["python"], source="python")
        rag.index(DOCUMENTS["rust"], source="rust")
        rag.index(DOCUMENTS["cooking"], source="cooking")

        result = self.ffai.query("Which language focuses on memory safety?")
        assert isinstance(result, QueryResult)
        assert len(result.hits) >= 1
        assert len(result.sources) >= 1

    def test_ffai_query_custom_template(self):
        assert self.ffai.rag is not None
        self.ffai.rag.index(DOCUMENTS["rust"], source="rust")

        template = "FACTS:\n{context}\n\nASK: {question}\n\nRESPOND:"
        result = self.ffai.query(
            "What is Rust?",
            prompt_template=template,
        )
        assert "FACTS:" in result.prompt
        assert "ASK:" in result.prompt
        assert len(result.answer) > 0
