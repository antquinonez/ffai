# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Shared type definitions for FFAI."""

from __future__ import annotations

from typing import TypedDict

from typing_extensions import NotRequired


class Interaction(TypedDict):
    """Record of a single prompt-response exchange.

    Attributes:
        prompt: The user's input text.
        response: The model's output text.
        prompt_name: Optional name identifying the prompt template used.
        timestamp: Unix timestamp of the interaction.
        model: Model identifier used for generation.
        history: Optional prior conversation context.

    """

    prompt: str
    response: str
    prompt_name: NotRequired[str | None]
    timestamp: NotRequired[float]
    model: NotRequired[str | None]
    history: NotRequired[list[str] | None]


class PromptSpec(TypedDict, total=False):
    """Declarative specification for a single prompt within a batch or pipeline.

    All fields are optional (``total=False``) allowing partial specs to be
    merged with defaults at runtime.

    Attributes:
        sequence: Execution order within the batch (0-indexed).
        prompt_name: Human-readable name for the prompt.
        prompt: The prompt text.
        history: Prior conversation context to prepend.
        condition: Expression that must evaluate truthy for the prompt to run.
        abort_condition: Expression that triggers batch abort after this prompt.
        response_model: Pydantic model for structured output parsing.
        system_instructions: System-level instructions override.
        model: Model identifier override.

    """

    sequence: int
    prompt_name: str
    prompt: str
    history: list[str] | None
    condition: str | None
    abort_condition: str | None
    response_model: type | None
    system_instructions: str | None
    model: str | None
