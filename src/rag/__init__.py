from .embed import Embeddings
from .format import format_hits
from .indexing import BM25Index, ChunkDeduplicator, ContextualEmbeddings, HierarchicalIndex
from .rag import RAG
from .search import (
    CrossEncoderReranker,
    DiversityReranker,
    HybridSearch,
    NoopReranker,
    QueryExpander,
    RerankerBase,
    fuse_search_results,
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
from .types import SearchHit

try:
    from .store import CHROMADB_AVAILABLE, VectorStore
except ImportError:
    VectorStore = None  # type: ignore[assignment,misc]
    CHROMADB_AVAILABLE = False

__all__ = [
    "CHROMADB_AVAILABLE",
    "RAG",
    "BM25Index",
    "CharacterChunker",
    "ChunkDeduplicator",
    "ChunkerBase",
    "CodeChunker",
    "ContextualEmbeddings",
    "CrossEncoderReranker",
    "DiversityReranker",
    "Embeddings",
    "HierarchicalChunker",
    "HierarchicalIndex",
    "HierarchicalTextChunk",
    "HybridSearch",
    "MarkdownChunker",
    "NoopReranker",
    "QueryExpander",
    "RecursiveChunker",
    "RerankerBase",
    "SearchHit",
    "TextChunk",
    "VectorStore",
    "chunk_text",
    "format_hits",
    "fuse_search_results",
    "get_chunker",
    "get_reranker",
    "list_chunkers",
    "reciprocal_rank_fusion",
]
