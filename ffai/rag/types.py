"""Define shared data classes used across the RAG subsystem."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchHit:
    """Single search result from a RAG query.

    Attributes:
        content: Matched chunk text.
        score: Relevance score (higher is better).
        source: Source document identifier.
        metadata: Additional metadata from indexing.
        parent_content: Parent chunk text for hierarchical context.
        id: Unique chunk identifier.

    """

    content: str
    score: float
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_content: str | None = None
    id: str = ""


@dataclass
class GenerationResult:
    """Result of a RAG generation call.

    Attributes:
        text: Generated answer text.
        usage: Provider-specific token usage object.
        cost_usd: Estimated cost in USD.
        duration_ms: Wall-clock generation duration in milliseconds.

    """

    text: str
    usage: Any | None = None
    cost_usd: float = 0.0
    duration_ms: float | None = None


@dataclass
class QueryResult:
    """Result of a RAG query combining search and generation.

    Attributes:
        answer: Generated answer text.
        hits: Search hits used as context.
        sources: Deduplicated source identifiers.
        prompt: Full prompt sent to the generation function.
        usage: Provider-specific token usage object.
        cost_usd: Estimated generation cost in USD.
        duration_ms: Generation duration in milliseconds.

    """

    answer: str
    hits: list[SearchHit]
    sources: list[str]
    prompt: str
    usage: Any | None = None
    cost_usd: float = 0.0
    duration_ms: float | None = None
