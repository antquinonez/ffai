from __future__ import annotations

import pytest

from ffai.rag.stores import (
    STORE_REGISTRY,
    VectorStoreBase,
    get_store,
    is_store_available,
    list_available_stores,
    list_stores,
)


class TestListStores:
    def test_returns_all_known_backend_names(self):
        names = list_stores()
        assert names == ["chroma", "pgvector", "qdrant", "sqlite_vss"]


class TestListAvailableStores:
    def test_returns_only_backends_with_deps_installed(self):
        available = list_available_stores()
        for name in available:
            assert is_store_available(name)

    def test_chroma_available_when_installed(self):
        if is_store_available("chroma"):
            assert "chroma" in list_available_stores()


class TestIsStoreAvailable:
    @pytest.mark.skipif(not is_store_available("chroma"), reason="chromadb not installed")
    def test_chroma_is_available(self):
        assert is_store_available("chroma") is True

    def test_pgvector_is_not_available_without_deps(self):
        if is_store_available("pgvector"):
            pytest.skip("pgvector deps installed")
        assert is_store_available("pgvector") is False

    @pytest.mark.skipif(not is_store_available("chroma"), reason="chromadb not installed")
    def test_case_insensitive(self):
        assert is_store_available("Chroma") is True


class TestGetStore:
    @pytest.mark.skipif(not is_store_available("chroma"), reason="chromadb not installed")
    def test_returns_chroma_vector_store(self):
        from ffai.rag.stores.chroma import ChromaVectorStore

        store = get_store("chroma", collection_name="test_col", dir="/tmp/test_stores")
        assert isinstance(store, ChromaVectorStore)
        assert isinstance(store, VectorStoreBase)
        assert store.name == "chroma"

    def test_unknown_backend_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown vector store backend"):
            get_store("nonexistent")

    def test_known_but_unavailable_raises_importerror(self):
        if is_store_available("pgvector"):
            pytest.skip("pgvector deps installed")
        with pytest.raises(ImportError, match="known but its dependency is not installed"):
            get_store("pgvector")

    @pytest.mark.skipif(not is_store_available("chroma"), reason="chromadb not installed")
    def test_case_insensitive_backend_name(self):
        from ffai.rag.stores.chroma import ChromaVectorStore

        store = get_store("Chroma", collection_name="test_col", dir="/tmp/test_stores")
        assert isinstance(store, ChromaVectorStore)


class TestStoreRegistry:
    @pytest.mark.skipif(not is_store_available("chroma"), reason="chromadb not installed")
    def test_registry_has_chroma_after_query(self):
        get_store("chroma", collection_name="test_col", dir="/tmp/test_stores")
        assert "chroma" in STORE_REGISTRY
        from ffai.rag.stores.chroma import ChromaVectorStore
        assert STORE_REGISTRY["chroma"] is ChromaVectorStore

    def test_importing_stores_does_not_fail(self):
        import ffai.rag.stores
        assert ffai.rag.stores is not None


class TestBackwardCompatImports:
    def test_from_store_import_vectorstore(self):
        from ffai.rag.store import VectorStore
        assert VectorStore is not None

    def test_from_store_import_chromadb_available(self):
        from ffai.rag.store import CHROMADB_AVAILABLE
        assert isinstance(CHROMADB_AVAILABLE, bool)

    def test_from_rag_import_vectorstore(self):
        from ffai.rag import VectorStore
        assert VectorStore is not None

    def test_from_rag_import_vectorstorebase(self):
        from ffai.rag import VectorStoreBase
        assert VectorStoreBase is not None

    def test_from_rag_import_get_store(self):
        from ffai.rag import get_store as gs
        assert callable(gs)

    def test_from_rag_import_list_stores(self):
        from ffai.rag import list_stores as ls
        assert ls() == ["chroma", "pgvector", "qdrant", "sqlite_vss"]
