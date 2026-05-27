from __future__ import annotations

from typing import Any


class ClientAdapter:
    def __init__(self, client: Any, **kwargs: Any) -> None:
        self._client = client
        self._kwargs = kwargs

    def __call__(self, prompt: str) -> str:
        return self._client.generate_response(prompt=prompt, **self._kwargs)
