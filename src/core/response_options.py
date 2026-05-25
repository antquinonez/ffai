# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Frozen dataclass grouping optional parameters for generate_response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class ResponseOptions:
    """Configuration for a single ``generate_response`` call.

    All fields are optional.  Provider-specific parameters (temperature,
    max_tokens, tools, tool_choice) are NOT here -- they stay as ``**kwargs``
    on ``FFAI.generate_response()``.

    Attributes:
        model: Override model for this call.
        system_instructions: Override system instructions for this call.
        response_format: Response format hint (e.g. ``{"type": "json_object"}``).
        response_model: Pydantic BaseModel subclass for structured output.
        condition: Expression evaluated before execution.  If false, prompt
            is skipped with ``status="skipped"``.
        abort_condition: Stored for future DAG executor use.
        strict: If True, raise ValueError on unknown ``{{name.response}}``
            references.
        history: List of prompt names to include in conversation context.
        dependencies: Prompt names this call depends on.

    """

    model: str | None = None
    system_instructions: str | None = None
    response_format: str | dict | None = None
    response_model: type | None = None
    condition: str | None = None
    abort_condition: str | None = None
    strict: bool = False
    history: list[str] | None = None
    dependencies: list[str] | None = None

    _KNOWN_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "model",
            "system_instructions",
            "response_format",
            "response_model",
            "condition",
            "abort_condition",
            "strict",
            "history",
            "dependencies",
        }
    )

    _IGNORED_KEYS: ClassVar[frozenset[str]] = frozenset({"prompt", "prompt_name", "sequence"})

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResponseOptions:
        """Construct from an ``execute_graph`` prompt dict.

        Known option keys are mapped to fields.  ``prompt``, ``prompt_name``,
        and ``sequence`` are silently ignored.  All other unknown keys are
        silently ignored.

        Args:
            d: Dict with keys like ``prompt_name``, ``prompt``, ``model``,
                ``condition``, etc.

        Returns:
            ``ResponseOptions`` with matched fields set.

        """
        kwargs: dict[str, Any] = {}
        for key in cls._KNOWN_KEYS:
            if key in d and d[key] is not None:
                kwargs[key] = d[key]
        return cls(**kwargs)
