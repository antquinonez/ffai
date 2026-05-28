"""Adapt an FFAI client into a callable that returns a GenerationResult for RAG generation."""

from __future__ import annotations

import asyncio
from typing import Any

from .types import GenerationResult


class ClientAdapter:
    """Wrap an FFAI client into a callable that returns a GenerationResult.

    Handles both synchronous and asynchronous ``generate_response`` methods,
    automatically awaiting coroutines when necessary.

    Args:
        client: An FFAI client instance with a ``generate_response`` method.
        **kwargs: Extra keyword arguments forwarded to ``generate_response``
            on every call.

    """

    def __init__(self, client: Any, **kwargs: Any) -> None:
        self._client = client
        self._kwargs = kwargs

    def __call__(self, prompt: str) -> GenerationResult:
        """Generate a response for the given prompt.

        Args:
            prompt: The prompt string to send to the client.

        Returns:
            A GenerationResult containing the response text, usage,
            cost, and duration metadata.

        """
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
