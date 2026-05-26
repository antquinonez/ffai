from __future__ import annotations

from .embeddings import FFEmbeddings
from .indexing import BM25Index, ChunkDeduplicator, ContextualEmbeddings, HierarchicalIndex
from .pipeline import RAGPipeline, format_results_for_prompt, normalize_scores
from .search import (
    CrossEncoderReranker,
    DiversityReranker,
    HybridSearch,
    NoopReranker,
    QueryExpander,
    RerankerBase,
    get_reranker,
    reciprocal_rank_fusion,
)
from .splitters import (
    CharacterChunker,
    ChunkerBase,
    CodeChunker,
    HierarchicalChunker,
    HierarchicalTextChunk,
    MarkdownChunker,
    RecursiveChunker,
    TextChunk,
    chunk_text,
    get_chunker,
    list_chunkers,
)

try:
    from .client import FFRAGClient, RAGClient
    from .vector_store import CHROMADB_AVAILABLE, FFVectorStore
except ImportError:
    RAGClient = None
    FFRAGClient = None
    FFVectorStore = None
    CHROMADB_AVAILABLE = False

if not CHROMADB_AVAILABLE:
    RAGClient = None
    FFRAGClient = None
    FFVectorStore = None

__all__ = [
    "CHROMADB_AVAILABLE",
    "BM25Index",
    "CharacterChunker",
    "ChunkDeduplicator",
    "ChunkerBase",
    "CodeChunker",
    "ContextualEmbeddings",
    "CrossEncoderReranker",
    "DiversityReranker",
    "FFEmbeddings",
    "FFRAGClient",
    "FFVectorStore",
    "HierarchicalChunker",
    "HierarchicalIndex",
    "HierarchicalTextChunk",
    "HybridSearch",
    "MarkdownChunker",
    "NoopReranker",
    "QueryExpander",
    "RAGClient",
    "RAGPipeline",
    "RecursiveChunker",
    "RerankerBase",
    "TextChunk",
    "chunk_text",
    "format_results_for_prompt",
    "get_chunker",
    "get_reranker",
    "list_chunkers",
    "normalize_scores",
    "reciprocal_rank_fusion",
]
