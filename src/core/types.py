# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Shared type definitions for FFAI."""

from __future__ import annotations

from typing import TypedDict

from typing_extensions import NotRequired


class Interaction(TypedDict):
    prompt: str
    response: str
    prompt_name: NotRequired[str | None]
    timestamp: NotRequired[float]
    model: NotRequired[str | None]
    history: NotRequired[list[str] | None]


class PromptSpec(TypedDict, total=False):
    sequence: int
    prompt_name: str
    prompt: str
    history: list[str] | None
    condition: str | None
    abort_condition: str | None
    response_model: type | None
    system_instructions: str | None
    model: str | None
