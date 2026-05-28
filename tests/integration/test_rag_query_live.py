import os

import pytest

from ffai.FFAI import FFAI
from ffai.rag.types import QueryResult

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
        from ffai.rag.embed import Embeddings
        from ffai.rag.rag import RAG
        from ffai.rag.store import VectorStore

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
        from ffai.rag.embed import Embeddings
        from ffai.rag.rag import RAG
        from ffai.rag.store import VectorStore

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
        from ffai.Clients.FFLiteLLMClient import FFLiteLLMClient
        from ffai.rag.embed import Embeddings
        from ffai.rag.rag import RAG
        from ffai.rag.store import VectorStore

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
        from ffai.rag.embed import Embeddings
        from ffai.rag.rag import RAG
        from ffai.rag.store import VectorStore

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

    def test_ffai_query_populates_metadata(self):
        assert self.ffai.rag is not None
        self.ffai.rag.index(DOCUMENTS["python"], source="python")

        result = self.ffai.query("What is Python?")
        assert isinstance(result, QueryResult)
        assert len(result.answer) > 0
        assert result.usage is not None
        assert result.cost_usd > 0
        assert result.duration_ms is not None
        assert result.duration_ms > 0

    def test_ffai_query_aquery_populates_metadata(self):
        import asyncio

        assert self.ffai.rag is not None
        self.ffai.rag.index(DOCUMENTS["rust"], source="rust")

        result = asyncio.run(self.ffai.aquery("What is Rust?"))
        assert isinstance(result, QueryResult)
        assert len(result.answer) > 0
        assert result.usage is not None
        assert result.cost_usd > 0
        assert result.duration_ms is not None
        assert result.duration_ms > 0


class TestRAGQueryMetadataLive:
    """L1: Verify GenerationResult metadata flows through RAG.query()."""

    @pytest.fixture(autouse=True)
    def setup_rag(self, tmp_path):
        _skip_no_chromadb()
        from ffai.rag.embed import Embeddings
        from ffai.rag.rag import RAG
        from ffai.rag.store import VectorStore

        api_key = _get_mistral_api_key()
        embed = Embeddings("mistral/mistral-embed", api_key=api_key, cache_enabled=True)
        store = VectorStore(
            collection_name=f"test_meta_{os.getpid()}",
            dir=str(tmp_path / "chroma_db"),
        )
        self.rag = RAG(embed=embed, store=store, chunk_size=200, chunk_overlap=50)

    def test_query_with_generation_result_carries_metadata(self):
        from ffai.rag.types import GenerationResult

        self.rag.index(DOCUMENTS["python"], source="python")

        def generate(prompt: str) -> GenerationResult:
            return GenerationResult(
                text="Python is a programming language.",
                usage={"prompt_tokens": 10, "completion_tokens": 20},
                cost_usd=0.001,
                duration_ms=250.0,
            )

        result = self.rag.query("What is Python?", generate_fn=generate)
        assert result.answer == "Python is a programming language."
        assert result.usage == {"prompt_tokens": 10, "completion_tokens": 20}
        assert result.cost_usd == 0.001
        assert result.duration_ms == 250.0

    def test_query_with_plain_string_has_no_metadata(self):
        self.rag.index(DOCUMENTS["rust"], source="rust")

        result = self.rag.query(
            "What is Rust?",
            generate_fn=lambda p: "A systems language.",
        )
        assert result.answer == "A systems language."
        assert result.usage is None
        assert result.cost_usd == 0.0
        assert result.duration_ms is None

    def test_query_with_default_generate_fn_carries_metadata(self):
        from ffai.rag.types import GenerationResult

        self.rag.index(DOCUMENTS["python"], source="python")

        call_count = 0

        def generate(prompt: str) -> GenerationResult:
            nonlocal call_count
            call_count += 1
            return GenerationResult(
                text="Answer from default fn.",
                usage={"tokens": 30},
                cost_usd=0.002,
                duration_ms=100.0,
            )

        self.rag.set_generate_fn(generate)
        result = self.rag.query("What is Python?")
        assert result.answer == "Answer from default fn."
        assert result.cost_usd == 0.002
        assert result.duration_ms == 100.0
        assert call_count == 1


class TestRAGBM25OnlyQueryLive:
    """BM25-only query without chromadb dependency."""

    @pytest.fixture(autouse=True)
    def setup_rag(self):
        self.rag = None

    def test_bm25_query_returns_answer(self):
        from ffai.rag.rag import RAG

        rag = RAG(embed="mistral/mistral-embed", bm25_alpha=0.6)
        rag.index(DOCUMENTS["python"], source="python")
        rag.index(DOCUMENTS["rust"], source="rust")

        result = rag.query(
            "memory safety borrow checker",
            generate_fn=lambda p: "Rust",
        )
        assert result.answer == "Rust"
        assert len(result.hits) >= 1
        assert "rust" in result.sources

    def test_bm25_query_filters_by_metadata(self):
        from ffai.rag.rag import RAG

        rag = RAG(embed="mistral/mistral-embed", bm25_alpha=0.6)
        rag.index(DOCUMENTS["python"], source="python")
        rag.index(DOCUMENTS["cooking"], source="cooking")

        result = rag.query(
            "pasta recipe",
            generate_fn=lambda p: "Italian food",
            source="cooking",
        )
        assert result.answer == "Italian food"
        assert all(h.source == "cooking" for h in result.hits)

    def test_bm25_query_empty_index(self):
        from ffai.rag.rag import RAG

        rag = RAG(embed="mistral/mistral-embed", bm25_alpha=0.6)
        result = rag.query(
            "anything",
            generate_fn=lambda p: "No data available.",
        )
        assert result.answer == "No data available."
        assert result.hits == []
        assert result.sources == []

    def test_bm25_query_with_generation_result_metadata(self):
        from ffai.rag.rag import RAG
        from ffai.rag.types import GenerationResult

        rag = RAG(embed="mistral/mistral-embed", bm25_alpha=0.6)
        rag.index(DOCUMENTS["rust"], source="rust")

        def generate(p: str) -> GenerationResult:
            return GenerationResult(
                text="Rust language",
                usage={"tokens": 5},
                cost_usd=0.0005,
                duration_ms=50.0,
            )

        result = rag.query("What is Rust?", generate_fn=generate)
        assert result.answer == "Rust language"
        assert result.usage == {"tokens": 5}
        assert result.cost_usd == 0.0005
        assert result.duration_ms == 50.0


class TestFFAIAutoWireLive:
    """L2/L3: Verify FFAI auto-wires ClientAdapter as generate_fn."""

    @pytest.fixture(autouse=True)
    def setup_ffai(self, tmp_path):
        _skip_no_chromadb()
        from dotenv import load_dotenv

        load_dotenv()
        from ffai.Clients.FFLiteLLMClient import FFLiteLLMClient
        from ffai.rag.embed import Embeddings
        from ffai.rag.rag import RAG
        from ffai.rag.store import VectorStore

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
            collection_name=f"test_autowire_{os.getpid()}",
            dir=str(tmp_path / "chroma_db"),
        )
        rag = RAG(embed=embed, store=store, chunk_size=200, chunk_overlap=50)
        self.ffai = FFAI(client=client, rag=rag)

    def test_auto_wire_query_without_explicit_generate_fn(self):
        assert self.ffai.rag is not None
        self.ffai.rag.index(DOCUMENTS["python"], source="python")
        self.ffai.rag.index(DOCUMENTS["rust"], source="rust")

        result = self.ffai.query("What is Python?")
        assert isinstance(result, QueryResult)
        assert len(result.answer) > 0
        assert len(result.hits) >= 1
        assert "python" in result.sources

    def test_auto_wire_carries_usage_metadata(self):
        assert self.ffai.rag is not None
        self.ffai.rag.index(DOCUMENTS["rust"], source="rust")

        result = self.ffai.query("What is Rust?")
        assert result.usage is not None
        assert result.cost_usd > 0
        assert result.duration_ms is not None
        assert result.duration_ms > 0

    def test_auto_wire_with_hybrid_and_reranker(self, tmp_path):
        from ffai.rag.embed import Embeddings
        from ffai.rag.rag import RAG
        from ffai.rag.store import VectorStore

        api_key = os.getenv("MISTRAL_API_KEY")
        embed = Embeddings("mistral/mistral-embed", api_key=api_key, cache_enabled=True)
        store = VectorStore(
            collection_name=f"test_autowire_hybrid_{os.getpid()}",
            dir=str(tmp_path / "chroma_hybrid"),
        )
        rag = RAG(
            embed=embed, store=store,
            chunk_size=200, chunk_overlap=50,
            bm25_alpha=0.5, reranker="diversity",
        )
        self.ffai.rag = rag
        from ffai.rag import ClientAdapter

        rag.set_generate_fn(ClientAdapter(self.ffai.client))
        rag.index(DOCUMENTS["python"], source="python")
        rag.index(DOCUMENTS["rust"], source="rust")
        rag.index(DOCUMENTS["cooking"], source="cooking")

        result = self.ffai.query("Which language uses a borrow checker?")
        assert isinstance(result, QueryResult)
        assert len(result.answer) > 0
        assert result.usage is not None
        assert result.cost_usd > 0


class TestFFAIFacadeLive:
    """L3: FFAI facade methods (index, search, delete, count)."""

    @pytest.fixture(autouse=True)
    def setup_ffai(self, tmp_path):
        _skip_no_chromadb()
        from dotenv import load_dotenv

        load_dotenv()
        from ffai.Clients.FFLiteLLMClient import FFLiteLLMClient
        from ffai.rag.embed import Embeddings
        from ffai.rag.rag import RAG
        from ffai.rag.store import VectorStore

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
            collection_name=f"test_facade_{os.getpid()}",
            dir=str(tmp_path / "chroma_db"),
        )
        rag = RAG(embed=embed, store=store, chunk_size=200, chunk_overlap=50)
        self.ffai = FFAI(client=client, rag=rag)

    def test_facade_index_and_count(self):
        n = self.ffai.index(DOCUMENTS["python"], source="python")
        assert n > 0
        assert self.ffai.count() == n

    def test_facade_search_returns_hits(self):
        self.ffai.index(DOCUMENTS["rust"], source="rust")
        hits = self.ffai.search("memory safety")
        assert len(hits) >= 1
        assert any(h.source == "rust" for h in hits)

    def test_facade_delete_and_verify(self):
        self.ffai.index(DOCUMENTS["python"], source="python")
        self.ffai.index(DOCUMENTS["rust"], source="rust")
        count_before = self.ffai.count()
        self.ffai.delete("python")
        assert self.ffai.count() < count_before

    def test_facade_query_after_index(self):
        self.ffai.index(DOCUMENTS["python"], source="python")
        result = self.ffai.query("What is Python?")
        assert isinstance(result, QueryResult)
        assert len(result.answer) > 0
        assert "python" in result.sources


class TestRAGAllowEmptyLive:
    """L1: allow_llm_on_empty with real search."""

    @pytest.fixture(autouse=True)
    def setup_rag(self):
        self.rag = None

    def test_allow_llm_on_empty_false_no_llm_call(self):
        from ffai.rag.rag import RAG

        rag = RAG(embed="mistral/mistral-embed", bm25_alpha=0.6)
        call_count = 0

        def counting_fn(p: str) -> str:
            nonlocal call_count
            call_count += 1
            return "should not be called"

        result = rag.query("nonexistent topic xyz", generate_fn=counting_fn, allow_llm_on_empty=False)
        assert result.answer == ""
        assert result.hits == []
        assert call_count == 0

    def test_allow_llm_on_empty_true_calls_llm(self):
        from ffai.rag.rag import RAG

        rag = RAG(embed="mistral/mistral-embed", bm25_alpha=0.6)
        result = rag.query(
            "nonexistent topic xyz",
            generate_fn=lambda p: "I don't know.",
            allow_llm_on_empty=True,
        )
        assert result.answer == "I don't know."
