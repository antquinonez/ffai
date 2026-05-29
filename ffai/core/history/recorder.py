# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Single point for writing interactions to all history stores."""

from __future__ import annotations

import logging
import time
from typing import Any

from ..response_context import ResponseContext
from .ordered import OrderedPromptHistory
from .permanent import PermanentHistory

logger = logging.getLogger(__name__)


class HistoryRecorder:
    """Records interactions to all 5 history stores in a single operation.

    Owns the raw ``history`` and ``clean_history`` lists and coordinates
    writes to ``PermanentHistory``, ``OrderedPromptHistory``, and
    ``ResponseContext``.  Callers invoke ``record()`` instead of manually
    writing to 5 separate stores.

    Args:
        context: The ResponseContext for prompt_attr_history recording.
        permanent_history: The PermanentHistory for chronological turns.
        ordered_history: The OrderedPromptHistory for named interactions.

    """

    def __init__(
        self,
        context: ResponseContext,
        permanent_history: PermanentHistory,
        ordered_history: OrderedPromptHistory,
    ) -> None:
        self.history: list[dict[str, Any]] = []
        self.clean_history: list[dict[str, Any]] = []
        self._context = context
        self._permanent = permanent_history
        self._ordered = ordered_history

    def record(
        self,
        prompt: str,
        response: Any,
        model: str,
        prompt_name: str | None = None,
        history: list[str] | None = None,
        status: str = "success",
        resolved_prompt: str | None = None,
        usage: Any = None,
    ) -> None:
        """Record an interaction to all 5 history stores.

        Args:
            prompt: The resolved prompt text.
            response: The cleaned response.
            model: Model identifier used.
            prompt_name: Logical name for the prompt.
            history: List of prompt names this call depends on.
            status: Execution status ("success", "skipped", "failed").
            resolved_prompt: The fully interpolated prompt sent to the model.
            usage: Token usage from the API call.
        """
        self._permanent.add_turn_user(prompt)
        self._permanent.add_turn_assistant(
            str(response) if response is not None else ""
        )

        interaction: dict[str, Any] = {
            "prompt": prompt,
            "response": response,
            "prompt_name": prompt_name,
            "timestamp": time.time(),
            "model": model,
            "history": history,
            "status": status,
            "resolved_prompt": resolved_prompt,
            "usage": usage,
        }

        self.history.append(interaction)
        self.clean_history.append(interaction)

        self._context.record(prompt, response, model, prompt_name, history)

        self._ordered.add_interaction(
            model=model,
            prompt=prompt,
            response=str(response) if response is not None else "",
            prompt_name=prompt_name,
            history=history,
        )
