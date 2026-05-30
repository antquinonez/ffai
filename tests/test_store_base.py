from __future__ import annotations

import pytest

from ffai.rag.stores.base import VectorStoreBase


class TestVectorStoreBaseCannotInstantiate:
    def test_raises_typeerror(self):
        with pytest.raises(TypeError):
            VectorStoreBase()


class TestIncompleteSubclass:
    def test_raises_typeerror_when_methods_missing(self):
        class Incomplete(VectorStoreBase):
            pass

        with pytest.raises(TypeError):
            Incomplete()


class TestCompleteSubclass:
    def test_instantiates_when_all_methods_implemented(self):
        class Complete(VectorStoreBase):
            @property
            def name(self) -> str:
                return "test"

            async def aadd(self, ids, texts, embeddings, metadatas):
                return 0

            async def asearch(self, query_embedding, top_k=5, where=None):
                return []

            def delete_by_source(self, source):
                pass

            def delete_by_source_and_strategy(self, source, strategy):
                pass

            def count(self):
                return 0

            def clear(self):
                pass

            def list_sources(self):
                return []

            def get_all(self):
                return []

            def needs_reindex(self, source, checksum, strategy="default"):
                return True

        store = Complete()
        assert store.name == "test"
        assert store.count() == 0
        assert store.list_sources() == []
        assert store.needs_reindex("x", "abc") is True


class TestVectorStoreBaseIsAbstract:
    def test_has_ten_abstract_methods(self):
        abstracts = {
            name
            for name in dir(VectorStoreBase)
            if getattr(getattr(VectorStoreBase, name, None), "__isabstractmethod__", False)
        }
        expected = {
            "aadd", "asearch", "clear", "count", "delete_by_source",
            "delete_by_source_and_strategy", "get_all", "list_sources",
            "name", "needs_reindex",
        }
        assert abstracts == expected
