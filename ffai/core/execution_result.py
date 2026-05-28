# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Internal result type returned by ResponseExecutor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .usage import TokenUsage


@dataclass
class ExecutionResult:
    """Outcome of a single prompt execution, before history recording.

    Attributes:
        response: Raw LLM response string (or None on skip/failure).
        resolved_prompt: The fully interpolated prompt sent to the model.
        usage: Token usage from the API call, if available.
        cost_usd: Estimated cost in USD for this call.
        duration_ms: Wall-clock duration of the LLM call in milliseconds.
        status: Execution status -- ``"success"``, ``"skipped"``, or ``"failed"``.
        condition_trace: The resolved condition expression (when condition is used).
        condition_error: Error message if condition evaluation failed.
        parsed: Validated Pydantic model instance (when response_model is used).
        parsing_errors: Validation error strings if structured output parsing failed.

    """

    response: str | None = None
    resolved_prompt: str = ""
    model: str = ""
    usage: TokenUsage | None = None
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    status: str = "success"
    condition_trace: str | None = None
    condition_error: str | None = None
    parsed: Any = None
    parsing_errors: list[str] | None = None
