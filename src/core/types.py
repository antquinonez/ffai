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
