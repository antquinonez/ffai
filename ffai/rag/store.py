"""Backward-compatibility re-export shim.

The VectorStore class now lives in :mod:`ffai.rag.stores.chroma`.
This module re-exports it under the old name so that existing imports
continue to work.
"""

from .stores.chroma import CHROMADB_AVAILABLE
from .stores.chroma import ChromaVectorStore as VectorStore

__all__ = ["CHROMADB_AVAILABLE", "VectorStore"]
