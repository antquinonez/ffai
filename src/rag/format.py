"""Format search hits into a numbered, human-readable text string."""

from __future__ import annotations

from .types import SearchHit


def format_hits(
    hits: list[SearchHit],
    max_chars: int | None = None,
    include_parent: bool = True,
) -> str:
    if not hits:
        return ""

    parts: list[str] = []
    total = 0

    for i, hit in enumerate(hits, 1):
        parent = ""
        if include_parent and hit.parent_content:
            snippet = hit.parent_content[:200]
            suffix = "..." if len(hit.parent_content) > 200 else ""
            parent = f"\n[Parent context: {snippet}{suffix}]"

        line = f"[{i}] (source: {hit.source}, relevance: {hit.score:.2f})\n{hit.content}{parent}\n"

        if max_chars and total + len(line) > max_chars:
            break

        parts.append(line)
        total += len(line)

    return "".join(parts)
