"""Integration tests for Qdrant vector store backend.

Exercises the full VectorStoreBase contract across all Qdrant deployment
modes: local (file-based), in-memory (ephemeral), and server (Docker).

All tests use synthetic embeddings — no LLM calls required.

Run:
    pytest tests/integration/test_store_qdrant_live.py -m integration -v

    # Only local-mode tests
    pytest tests/integration/test_store_qdrant_live.py -m "integration and qdrant" -k Local

    # Only in-memory tests
    pytest tests/integration/test_store_qdrant_live.py -m "integration and qdrant" -k Memory

    # Only server-mode tests (requires Docker)
    docker compose -f docker-compose.dev.yaml up -d
    pytest tests/integration/test_store_qdrant_live.py -m "integration and qdrant" -k Server
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.qdrant]

try:
    import qdrant_client  # type: ignore[import-untyped]  # noqa: F401

    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False


def _skip_no_qdrant():
    if not HAS_QDRANT:
        pytest.skip("qdrant-client not installed (pip install qdrant-client)")


def _skip_no_server():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        sock.connect(("localhost", 6333))
        sock.close()
    except (ConnectionRefusedError, OSError):
        sock.close()
        pytest.skip(
            "Qdrant server not running on localhost:6333. "
            "Start with: docker compose -f docker-compose.dev.yaml up -d "
            "or pass --qdrant-server to auto-start"
        )


DIM = 8
VEC_A = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
VEC_B = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
VEC_C = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
VEC_NEAR_A = [0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _add_docs(store, n, source="doc1"):
    ids = [str(uuid.uuid4()) for _ in range(n)]
    vecs = [VEC_A, VEC_B, VEC_C]
    texts = [f"text_{i}" for i in range(n)]
    embs = [vecs[i % len(vecs)] for i in range(n)]
    metas = [{"source": source} for _ in range(n)]
    return asyncio.run(store.aadd(ids=ids, texts=texts, embeddings=embs, metadatas=metas))


# ──────────────────────────────────────────────
# Mode 1: Local (file-based, persistent)
# ──────────────────────────────────────────────


class TestQdrantLocalModeCRUD:
    @pytest.fixture(autouse=True)
    def setup_store(self, tmp_path):
        _skip_no_qdrant()
        from ffai.rag.stores.qdrant import QdrantVectorStore

        self.store = QdrantVectorStore(
            collection_name=f"test_local_{os.getpid()}_{uuid.uuid4().hex[:6]}",
            path=str(tmp_path / "qdrant_db"),
            embedding_dim=DIM,
        )

    def test_name_is_qdrant(self):
        assert self.store.name == "qdrant"

    def test_is_local_flag(self):
        assert self.store._is_local is True

    def test_count_starts_at_zero(self):
        assert self.store.count() == 0

    def test_aadd_returns_count(self):
        count = _add_docs(self.store, 3)
        assert count == 3
        assert self.store.count() == 3

    def test_aadd_with_non_uuid_ids(self):
        asyncio.run(self.store.aadd(
            ids=["chunk_0", "chunk_1"],
            texts=["hello", "world"],
            embeddings=[VEC_A, VEC_B],
            metadatas=[{"source": "doc1"}, {"source": "doc1"}],
        ))
        assert self.store.count() == 2

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
            metadatas=[{"source": "doc_a"}, {"source": "doc_b"}, {"source": "doc_c"}],
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
        self.store.delete_by_source_and_strategy("doc1", "recursive")
        assert self.store.count() == 1

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
        assert {d["content"] for d in all_docs} == {"doc one", "doc two"}

    def test_needs_reindex_true_when_no_match(self):
        assert self.store.needs_reindex("nonexistent", "abc") is True

    def test_needs_reindex_false_when_checksum_matches(self):
        asyncio.run(self.store.aadd(
            ids=[str(uuid.uuid4())],
            texts=["text"],
            embeddings=[VEC_A],
            metadatas=[{"source": "doc1", "chunking_strategy": "default", "document_checksum": "abc"}],
        ))
        assert self.store.needs_reindex("doc1", "abc") is False

    def test_needs_reindex_true_when_checksum_differs(self):
        asyncio.run(self.store.aadd(
            ids=[str(uuid.uuid4())],
            texts=["text"],
            embeddings=[VEC_A],
            metadatas=[{"source": "doc1", "chunking_strategy": "default", "document_checksum": "old"}],
        ))
        assert self.store.needs_reindex("doc1", "new") is True

    def test_clear_resets_store(self):
        _add_docs(self.store, 2)
        assert self.store.count() == 2
        self.store.clear()
        assert self.store.count() == 0
        assert self.store.list_sources() == []


class TestQdrantLocalModeSearch:
    @pytest.fixture(autouse=True)
    def setup_store(self, tmp_path):
        _skip_no_qdrant()
        from ffai.rag.stores.qdrant import QdrantVectorStore

        self.store = QdrantVectorStore(
            collection_name=f"test_local_rank_{os.getpid()}_{uuid.uuid4().hex[:6]}",
            path=str(tmp_path / "qdrant_db"),
            embedding_dim=DIM,
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
            ids=[uid], texts=["version 1"], embeddings=[VEC_A],
            metadatas=[{"source": "doc1"}],
        ))
        assert self.store.count() == 1
        asyncio.run(self.store.aadd(
            ids=[uid], texts=["version 2"], embeddings=[VEC_B],
            metadatas=[{"source": "doc1"}],
        ))
        assert self.store.count() == 1
        assert self.store.get_all()[0]["content"] == "version 2"


# ──────────────────────────────────────────────
# Mode 4: In-memory (ephemeral, data lost on exit)
# ──────────────────────────────────────────────


class TestQdrantMemoryModeCRUD:
    @pytest.fixture(autouse=True)
    def setup_store(self):
        _skip_no_qdrant()
        from ffai.rag.stores.qdrant import QdrantVectorStore

        self.store = QdrantVectorStore(
            collection_name=f"test_mem_{uuid.uuid4().hex[:6]}",
            location=":memory:",
            embedding_dim=DIM,
        )

    def test_is_local_flag(self):
        assert self.store._is_local is True

    def test_count_starts_at_zero(self):
        assert self.store.count() == 0

    def test_aadd_and_count(self):
        count = _add_docs(self.store, 3)
        assert count == 3
        assert self.store.count() == 3

    def test_asearch_returns_hits(self):
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

    def test_asearch_with_where_filter(self):
        ids = [str(uuid.uuid4()) for _ in range(2)]
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=["a", "b"],
            embeddings=[VEC_A, VEC_B],
            metadatas=[{"source": "doc_a"}, {"source": "doc_b"}],
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
        self.store.delete_by_source("to_delete")
        assert self.store.count() == 1

    def test_list_sources(self):
        ids = [str(uuid.uuid4()) for _ in range(2)]
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=["a", "b"],
            embeddings=[VEC_A, VEC_B],
            metadatas=[{"source": "beta"}, {"source": "alpha"}],
        ))
        assert self.store.list_sources() == ["alpha", "beta"]

    def test_needs_reindex(self):
        asyncio.run(self.store.aadd(
            ids=[str(uuid.uuid4())],
            texts=["text"],
            embeddings=[VEC_A],
            metadatas=[{"source": "doc1", "chunking_strategy": "default", "document_checksum": "abc"}],
        ))
        assert self.store.needs_reindex("doc1", "abc") is False
        assert self.store.needs_reindex("doc1", "xyz") is True

    def test_clear_resets(self):
        _add_docs(self.store, 2)
        self.store.clear()
        assert self.store.count() == 0


class TestQdrantMemoryModeSearch:
    @pytest.fixture(autouse=True)
    def setup_store(self):
        _skip_no_qdrant()
        from ffai.rag.stores.qdrant import QdrantVectorStore

        self.store = QdrantVectorStore(
            collection_name=f"test_mem_rank_{uuid.uuid4().hex[:6]}",
            location=":memory:",
            embedding_dim=DIM,
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
        assert hits[0].content == "near"
        assert hits[0].score > hits[1].score

    def test_top_k_limits_results(self):
        ids = [str(uuid.uuid4()) for _ in range(5)]
        vecs = [[0.0] * DIM for _ in range(5)]
        for i in range(5):
            vecs[i][i % DIM] = 1.0
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=[f"t{i}" for i in range(5)],
            embeddings=vecs,
            metadatas=[{"source": f"s{i}"} for i in range(5)],
        ))
        assert len(asyncio.run(self.store.asearch(VEC_A, top_k=2))) == 2

    def test_upsert_replaces_existing_id(self):
        uid = str(uuid.uuid4())
        asyncio.run(self.store.aadd(
            ids=[uid], texts=["v1"], embeddings=[VEC_A],
            metadatas=[{"source": "d1"}],
        ))
        asyncio.run(self.store.aadd(
            ids=[uid], texts=["v2"], embeddings=[VEC_B],
            metadatas=[{"source": "d1"}],
        ))
        assert self.store.count() == 1
        assert self.store.get_all()[0]["content"] == "v2"


# ──────────────────────────────────────────────
# Mode 2: Server (Docker, localhost:6333)
# ──────────────────────────────────────────────

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


class TestQdrantServerModeCRUD:
    @pytest.fixture(autouse=True)
    def setup_store(self, qdrant_server_available):
        _skip_no_qdrant()
        if not qdrant_server_available:
            pytest.skip(
                "Qdrant server not running on localhost:6333. "
                "Start with: docker compose -f docker-compose.dev.yaml up -d "
                "or pass --qdrant-server to auto-start"
            )
        from ffai.rag.stores.qdrant import QdrantVectorStore

        self.store = QdrantVectorStore(
            collection_name=f"test_server_{os.getpid()}_{uuid.uuid4().hex[:6]}",
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            embedding_dim=DIM,
        )

    def test_is_local_flag_is_false(self):
        assert self.store._is_local is False

    def test_count_starts_at_zero(self):
        assert self.store.count() == 0

    def test_aadd_and_count(self):
        count = _add_docs(self.store, 3)
        assert count == 3
        assert self.store.count() == 3

    def test_asearch_returns_hits(self):
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

    def test_asearch_with_where_filter(self):
        ids = [str(uuid.uuid4()) for _ in range(2)]
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=["a", "b"],
            embeddings=[VEC_A, VEC_B],
            metadatas=[{"source": "doc_a"}, {"source": "doc_b"}],
        ))
        hits = asyncio.run(self.store.asearch(VEC_A, top_k=10, where={"source": "doc_a"}))
        assert len(hits) == 1

    def test_delete_by_source(self):
        ids = [str(uuid.uuid4()) for _ in range(2)]
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=["x", "y"],
            embeddings=[VEC_A, VEC_B],
            metadatas=[{"source": "to_delete"}, {"source": "keep"}],
        ))
        self.store.delete_by_source("to_delete")
        assert self.store.count() == 1

    def test_list_sources(self):
        ids = [str(uuid.uuid4()) for _ in range(2)]
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=["a", "b"],
            embeddings=[VEC_A, VEC_B],
            metadatas=[{"source": "beta"}, {"source": "alpha"}],
        ))
        assert self.store.list_sources() == ["alpha", "beta"]

    def test_needs_reindex(self):
        asyncio.run(self.store.aadd(
            ids=[str(uuid.uuid4())],
            texts=["text"],
            embeddings=[VEC_A],
            metadatas=[{"source": "doc1", "chunking_strategy": "default", "document_checksum": "abc"}],
        ))
        assert self.store.needs_reindex("doc1", "abc") is False
        assert self.store.needs_reindex("doc1", "xyz") is True

    def test_clear_resets(self):
        _add_docs(self.store, 2)
        self.store.clear()
        assert self.store.count() == 0


class TestQdrantServerModeSearch:
    @pytest.fixture(autouse=True)
    def setup_store(self, qdrant_server_available):
        _skip_no_qdrant()
        if not qdrant_server_available:
            pytest.skip(
                "Qdrant server not running on localhost:6333. "
                "Start with: docker compose -f docker-compose.dev.yaml up -d "
                "or pass --qdrant-server to auto-start"
            )
        from ffai.rag.stores.qdrant import QdrantVectorStore

        self.store = QdrantVectorStore(
            collection_name=f"test_server_rank_{os.getpid()}_{uuid.uuid4().hex[:6]}",
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            embedding_dim=DIM,
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
        assert hits[0].content == "near"
        assert hits[0].score > hits[1].score

    def test_top_k_limits_results(self):
        ids = [str(uuid.uuid4()) for _ in range(5)]
        vecs = [[0.0] * DIM for _ in range(5)]
        for i in range(5):
            vecs[i][i % DIM] = 1.0
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=[f"t{i}" for i in range(5)],
            embeddings=vecs,
            metadatas=[{"source": f"s{i}"} for i in range(5)],
        ))
        assert len(asyncio.run(self.store.asearch(VEC_A, top_k=2))) == 2

    def test_upsert_replaces_existing_id(self):
        uid = str(uuid.uuid4())
        asyncio.run(self.store.aadd(
            ids=[uid], texts=["v1"], embeddings=[VEC_A],
            metadatas=[{"source": "d1"}],
        ))
        asyncio.run(self.store.aadd(
            ids=[uid], texts=["v2"], embeddings=[VEC_B],
            metadatas=[{"source": "d1"}],
        ))
        assert self.store.count() == 1
        assert self.store.get_all()[0]["content"] == "v2"

    def test_uses_async_client(self):
        import asyncio as _asyncio

        async def _check():
            ac = await self.store._get_async_client()
            return ac is not None

        assert _asyncio.run(_check()) is True


# ──────────────────────────────────────────────
# Mode 3: Qdrant Cloud (hosted)
# ──────────────────────────────────────────────
# Requires QDRANT_CLUSTER_ENDPOINT and QDRANT_KEY
# environment variables. Tests skip if not set.


class TestQdrantCloudModeCRUD:
    @pytest.fixture(autouse=True)
    def setup_store(self, qdrant_cloud_config):
        _skip_no_qdrant()
        from ffai.rag.stores.qdrant import QdrantVectorStore

        self.store = QdrantVectorStore(
            collection_name=f"test_cloud_{os.getpid()}_{uuid.uuid4().hex[:6]}",
            url=qdrant_cloud_config["url"],
            api_key=qdrant_cloud_config["api_key"],
            embedding_dim=DIM,
        )
        yield
        try:
            self.store._client.delete_collection(self.store._collection_name)
        except Exception:
            pass

    def test_is_local_flag_is_false(self):
        assert self.store._is_local is False

    def test_count_starts_at_zero(self):
        assert self.store.count() == 0

    def test_aadd_and_count(self):
        count = _add_docs(self.store, 3)
        assert count == 3
        assert self.store.count() == 3

    def test_asearch_returns_hits(self):
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

    def test_asearch_with_where_filter(self):
        ids = [str(uuid.uuid4()) for _ in range(2)]
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=["a", "b"],
            embeddings=[VEC_A, VEC_B],
            metadatas=[{"source": "doc_a"}, {"source": "doc_b"}],
        ))
        hits = asyncio.run(self.store.asearch(VEC_A, top_k=10, where={"source": "doc_a"}))
        assert len(hits) == 1

    def test_delete_by_source(self):
        ids = [str(uuid.uuid4()) for _ in range(2)]
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=["x", "y"],
            embeddings=[VEC_A, VEC_B],
            metadatas=[{"source": "to_delete"}, {"source": "keep"}],
        ))
        self.store.delete_by_source("to_delete")
        assert self.store.count() == 1

    def test_list_sources(self):
        ids = [str(uuid.uuid4()) for _ in range(2)]
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=["a", "b"],
            embeddings=[VEC_A, VEC_B],
            metadatas=[{"source": "beta"}, {"source": "alpha"}],
        ))
        assert self.store.list_sources() == ["alpha", "beta"]

    def test_needs_reindex(self):
        asyncio.run(self.store.aadd(
            ids=[str(uuid.uuid4())],
            texts=["text"],
            embeddings=[VEC_A],
            metadatas=[{"source": "doc1", "chunking_strategy": "default", "document_checksum": "abc"}],
        ))
        assert self.store.needs_reindex("doc1", "abc") is False
        assert self.store.needs_reindex("doc1", "xyz") is True

    def test_clear_resets(self):
        _add_docs(self.store, 2)
        self.store.clear()
        assert self.store.count() == 0


class TestQdrantCloudModeSearch:
    @pytest.fixture(autouse=True)
    def setup_store(self, qdrant_cloud_config):
        _skip_no_qdrant()
        from ffai.rag.stores.qdrant import QdrantVectorStore

        self.store = QdrantVectorStore(
            collection_name=f"test_cloud_rank_{os.getpid()}_{uuid.uuid4().hex[:6]}",
            url=qdrant_cloud_config["url"],
            api_key=qdrant_cloud_config["api_key"],
            embedding_dim=DIM,
        )
        yield
        try:
            self.store._client.delete_collection(self.store._collection_name)
        except Exception:
            pass

    def test_closer_embedding_ranks_higher(self):
        ids = [str(uuid.uuid4()) for _ in range(3)]
        asyncio.run(self.store.aadd(
            ids=ids,
            texts=["near", "medium", "far"],
            embeddings=[VEC_NEAR_A, VEC_B, VEC_C],
            metadatas=[{"source": "s1"}, {"source": "s2"}, {"source": "s3"}],
        ))
        hits = asyncio.run(self.store.asearch(VEC_A, top_k=3))
        assert hits[0].content == "near"
        assert hits[0].score > hits[1].score

    def test_upsert_replaces_existing_id(self):
        uid = str(uuid.uuid4())
        asyncio.run(self.store.aadd(
            ids=[uid], texts=["v1"], embeddings=[VEC_A],
            metadatas=[{"source": "d1"}],
        ))
        asyncio.run(self.store.aadd(
            ids=[uid], texts=["v2"], embeddings=[VEC_B],
            metadatas=[{"source": "d1"}],
        ))
        assert self.store.count() == 1
        assert self.store.get_all()[0]["content"] == "v2"

    def test_uses_async_client(self):
        import asyncio as _asyncio

        async def _check():
            ac = await self.store._get_async_client()
            return ac is not None

        assert _asyncio.run(_check()) is True


# ──────────────────────────────────────────────
# Cross-mode: get_store() factory
# ──────────────────────────────────────────────


class TestQdrantFactoryIntegration:
    def test_get_store_local_mode(self, tmp_path):
        _skip_no_qdrant()
        from ffai.rag.stores import get_store

        store = get_store(
            "qdrant",
            path=str(tmp_path / "factory_local"),
            embedding_dim=DIM,
        )
        assert store.name == "qdrant"
        assert store.count() == 0

    def test_get_store_memory_mode(self):
        _skip_no_qdrant()
        from ffai.rag.stores import get_store

        store = get_store(
            "qdrant",
            location=":memory:",
            embedding_dim=DIM,
            collection_name=f"factory_mem_{uuid.uuid4().hex[:6]}",
        )
        assert store.name == "qdrant"
        assert store.count() == 0

    def test_get_store_server_mode(self, qdrant_server_available):
        _skip_no_qdrant()
        if not qdrant_server_available:
            pytest.skip(
                "Qdrant server not running on localhost:6333. "
                "Start with: docker compose -f docker-compose.dev.yaml up -d "
                "or pass --qdrant-server to auto-start"
            )
        from ffai.rag.stores import get_store

        store = get_store(
            "qdrant",
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            embedding_dim=DIM,
            collection_name=f"factory_server_{os.getpid()}_{uuid.uuid4().hex[:6]}",
        )
        assert store.name == "qdrant"
        assert store.count() == 0

    def test_get_store_cloud_mode(self, qdrant_cloud_config):
        _skip_no_qdrant()
        from ffai.rag.stores import get_store

        col_name = f"factory_cloud_{os.getpid()}_{uuid.uuid4().hex[:6]}"
        store = get_store(
            "qdrant",
            url=qdrant_cloud_config["url"],
            api_key=qdrant_cloud_config["api_key"],
            embedding_dim=DIM,
            collection_name=col_name,
        )
        assert store.name == "qdrant"
        assert store.count() == 0
        store._client.delete_collection(col_name)  # type: ignore[reportAttributeAccessIssue]
