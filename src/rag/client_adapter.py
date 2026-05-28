"""Adapt an FFAI client into a callable that returns a GenerationResult for RAG generation."""

from __future__ import annotations

import asyncio
from typing import Any

from .types import GenerationResult


class ClientAdapter:
    def __init__(self, client: Any, **kwargs: Any) -> None:
        self._client = client
        self._kwargs = kwargs

    def __call__(self, prompt: str) -> GenerationResult:
        text = self._client.generate_response(prompt=prompt, **self._kwargs)
        if asyncio.iscoroutine(text):
            text = asyncio.run(text)
        usage = getattr(self._client, "last_usage", None)
        cost_usd = getattr(self._client, "last_cost_usd", 0.0)
        duration = getattr(self._client, "last_duration_ms", None)
        return GenerationResult(
            text=text,
            usage=usage,
            cost_usd=cost_usd,
            duration_ms=duration,
        )
