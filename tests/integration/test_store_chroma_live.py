"""Integration tests for ChromaDB vector store backend.

Exercises the full VectorStoreBase contract against a real ChromaDB instance
using synthetic embeddings (no LLM calls required).

Run:
    pytest tests/integration/test_store_chroma_live.py -m integration -v
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.chroma]

try:
    import chromadb  # type: ignore[import-untyped]  # noqa: F401

    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


def _skip_no_chromadb():
    if not HAS_CHROMADB:
        pytest.skip("chromadb not installed (pip install ffai[rag])")


DIM = 8
VEC_A = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
VEC_B = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
VEC_C = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
VEC_NEAR_A = [0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class TestChromaStoreLiveCRUD:
    @pytest.fixture(autouse=True)
    def setup_store(self, tmp_path):
        _skip_no_chromadb()
        from ffai.rag.stores.chroma import ChromaVectorStore

        self.store = ChromaVectorStore(
            collection_name=f"test_chroma_{os.getpid()}_{uuid.uuid4().hex[:6]}",
            dir=str(tmp_path / "chroma_db"),
        )

    def test_name_property(self):
        assert self.store.name == "chroma"

    def test_count_starts_at_zero(self):
        assert self.store.count() == 0

    def test_aadd_returns_count(self):
        ids = [str(uuid.uuid4()) for _ in range(3)]
        count = asyncio.run(self.store.aadd(
            ids=ids,
            texts=["alpha", "beta", "gamma"],
            embeddings=[VEC_A, VEC_B, VEC_C],
            metadatas=[{"source": "doc1"}] * 3,
        ))
        assert count == 3
        assert self.store.count() == 3

    def test_asearch_returns_relevant_hits(self):
        ids = [str(uuid.uuid4()) for _ in range(2)]
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=["python programming", "rust systems"],
            embeddings=[VEC_A, VEC_B],
            metadatas=[{"source": "python_doc"}, {"source": "rust_doc"}],
        ))
        hits = asyncio.run(self.store.asearch(VEC_A, top_k=2))
        assert len(hits) == 2
        assert hits[0].content == "python programming"
        assert hits[0].source == "python_doc"
        assert hits[0].score > 0.0

    def test_asearch_with_where_filter(self):
        ids = [str(uuid.uuid4()) for _ in range(3)]
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=["a", "b", "c"],
            embeddings=[VEC_A, VEC_B, VEC_C],
            metadatas=[
                {"source": "doc_a"},
                {"source": "doc_b"},
                {"source": "doc_c"},
            ],
        ))
        hits = asyncio.run(self.store.asearch(VEC_A, top_k=10, where={"source": "doc_a"}))
        assert len(hits) == 1
        assert hits[0].source == "doc_a"

    def test_delete_by_source(self):
        ids = [str(uuid.uuid4()) for _ in range(2)]
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=["x", "y"],
            embeddings=[VEC_A, VEC_B],
            metadatas=[{"source": "to_delete"}, {"source": "keep"}],
        ))
        assert self.store.count() == 2
        self.store.delete_by_source("to_delete")
        assert self.store.count() == 1
        hits = asyncio.run(self.store.asearch(VEC_A, top_k=10))
        assert all(h.source != "to_delete" for h in hits)

    def test_delete_by_source_and_strategy(self):
        ids = [str(uuid.uuid4()) for _ in range(2)]
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=["a", "b"],
            embeddings=[VEC_A, VEC_B],
            metadatas=[
                {"source": "doc1", "chunking_strategy": "recursive"},
                {"source": "doc1", "chunking_strategy": "fixed"},
            ],
        ))
        assert self.store.count() == 2
        self.store.delete_by_source_and_strategy("doc1", "recursive")
        assert self.store.count() == 1
        hits = asyncio.run(self.store.asearch(VEC_A, top_k=10))
        assert all(h.metadata.get("chunking_strategy") != "recursive" for h in hits)

    def test_list_sources_sorted(self):
        ids = [str(uuid.uuid4()) for _ in range(3)]
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=["a", "b", "c"],
            embeddings=[VEC_A, VEC_B, VEC_C],
            metadatas=[{"source": "charlie"}, {"source": "alpha"}, {"source": "bravo"}],
        ))
        assert self.store.list_sources() == ["alpha", "bravo", "charlie"]

    def test_get_all_returns_all_documents(self):
        ids = [str(uuid.uuid4()) for _ in range(2)]
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=["doc one", "doc two"],
            embeddings=[VEC_A, VEC_B],
            metadatas=[{"source": "s1"}, {"source": "s2"}],
        ))
        all_docs = self.store.get_all()
        assert len(all_docs) == 2
        contents = {d["content"] for d in all_docs}
        assert contents == {"doc one", "doc two"}

    def test_needs_reindex_true_when_no_match(self):
        assert self.store.needs_reindex("nonexistent", "abc") is True

    def test_needs_reindex_false_when_checksum_matches(self):
        ids = [str(uuid.uuid4())]
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=["text"],
            embeddings=[VEC_A],
            metadatas=[{"source": "doc1", "chunking_strategy": "default", "document_checksum": "abc"}],
        ))
        assert self.store.needs_reindex("doc1", "abc") is False

    def test_needs_reindex_true_when_checksum_differs(self):
        ids = [str(uuid.uuid4())]
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=["text"],
            embeddings=[VEC_A],
            metadatas=[{"source": "doc1", "chunking_strategy": "default", "document_checksum": "old"}],
        ))
        assert self.store.needs_reindex("doc1", "new") is True

    def test_clear_resets_store(self):
        ids = [str(uuid.uuid4()) for _ in range(2)]
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=["a", "b"],
            embeddings=[VEC_A, VEC_B],
            metadatas=[{"source": "doc1"}, {"source": "doc1"}],
        ))
        assert self.store.count() == 2
        self.store.clear()
        assert self.store.count() == 0
        assert self.store.list_sources() == []


class TestChromaStoreLiveSearchRanking:
    @pytest.fixture(autouse=True)
    def setup_store(self, tmp_path):
        _skip_no_chromadb()
        from ffai.rag.stores.chroma import ChromaVectorStore

        self.store = ChromaVectorStore(
            collection_name=f"test_chroma_rank_{os.getpid()}_{uuid.uuid4().hex[:6]}",
            dir=str(tmp_path / "chroma_db"),
        )

    def test_closer_embedding_ranks_higher(self):
        ids = [str(uuid.uuid4()) for _ in range(3)]
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=["near", "medium", "far"],
            embeddings=[VEC_NEAR_A, VEC_B, VEC_C],
            metadatas=[{"source": "s1"}, {"source": "s2"}, {"source": "s3"}],
        ))
        hits = asyncio.run(self.store.asearch(VEC_A, top_k=3))
        assert len(hits) == 3
        assert hits[0].content == "near"
        assert hits[0].score > hits[1].score

    def test_top_k_limits_results(self):
        ids = [str(uuid.uuid4()) for _ in range(5)]
        vecs = []
        for i in range(5):
            v = [0.0] * DIM
            v[i % DIM] = 1.0
            vecs.append(v)
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=[f"text_{i}" for i in range(5)],
            embeddings=vecs,
            metadatas=[{"source": f"s{i}"} for i in range(5)],
        ))
        hits = asyncio.run(self.store.asearch(VEC_A, top_k=2))
        assert len(hits) == 2

    def test_upsert_replaces_existing_id(self):
        uid = str(uuid.uuid4())
        asyncio.run(self.store.aadd(
            ids=[uid],
            texts=["version 1"],
            embeddings=[VEC_A],
            metadatas=[{"source": "doc1"}],
        ))
        assert self.store.count() == 1
        self.store.clear()
        asyncio.run(self.store.aadd(
            ids=[uid],
            texts=["version 2"],
            embeddings=[VEC_B],
            metadatas=[{"source": "doc1"}],
        ))
        assert self.store.count() == 1
        all_docs = self.store.get_all()
        assert len(all_docs) == 1
        assert all_docs[0]["content"] == "version 2"
