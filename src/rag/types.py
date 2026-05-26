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
