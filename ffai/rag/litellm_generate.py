import time
from collections.abc import Callable
from typing import Any

import litellm

from ..core.usage import TokenUsage
from .types import GenerationResult

_CONFLICTING_KEYS = frozenset({"model", "messages", "api_key", "temperature", "max_tokens"})


def litellm_generate_fn(
    model: str,
    api_key: str | None = None,
    temperature: float = 0.5,
    max_tokens: int = 1024,
    **kwargs: Any,
) -> Callable[[str], GenerationResult]:
    extra = {k: v for k, v in kwargs.items() if k not in _CONFLICTING_KEYS}

    def generate(prompt: str) -> GenerationResult:
        t0 = time.perf_counter()
        params: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **extra,
        }
        if api_key:
            params["api_key"] = api_key
        resp = litellm.completion(**params)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        raw_usage = getattr(resp, "usage", None)
        if raw_usage is not None:
            in_t = getattr(raw_usage, "prompt_tokens", 0) or 0
            out_t = getattr(raw_usage, "completion_tokens", 0) or 0
            usage = TokenUsage(
                input_tokens=int(in_t),
                output_tokens=int(out_t),
                total_tokens=int(in_t) + int(out_t),
            )
        else:
            usage = TokenUsage()

        try:
            cost = litellm.completion_cost(resp)
        except Exception:
            cost = 0.0

        text = resp.choices[0].message.content or ""  # type: ignore[reportAttributeAccessIssue]

        return GenerationResult(
            text=text,
            usage=usage,
            cost_usd=cost if cost is not None else 0.0,
            duration_ms=elapsed_ms,
        )

    return generate
