from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchHit:
    content: str
    score: float
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_content: str | None = None
    id: str = ""


@dataclass
class GenerationResult:
    text: str
    usage: Any | None = None
    cost_usd: float = 0.0
    duration_ms: float | None = None


@dataclass
class QueryResult:
    answer: str
    hits: list[SearchHit]
    sources: list[str]
    prompt: str
    usage: Any | None = None
    cost_usd: float = 0.0
    duration_ms: float | None = None
