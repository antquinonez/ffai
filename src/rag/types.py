from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchHit:
    content: str
    score: float
    source: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    parent_content: str | None = None
    id: str = ""
