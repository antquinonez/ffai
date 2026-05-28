# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Maintain an append-only chronological turn history with coalesced consecutive user turns."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any


class PermanentHistory:
    """Append-only chronological turn history with timestamps.

    Each turn stores a ``role`` (``"user"`` or ``"assistant"``),
    structured ``content``, and a per-turn ``timestamp``. Consecutive
    user turns are coalesced by appending content rather than creating
    a new entry.

    """

    def __init__(self) -> None:
        self.turns: list[dict[str, Any]] = []
        self.timestamp: float = time.time()

    def add_turn_assistant(self, content: str) -> None:
        """Append an assistant turn with the current timestamp.

        Args:
            content: The assistant's response text.

        """
        self.turns.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": content}],
                "timestamp": time.time(),
            }
        )

    def add_turn_user(self, content: str) -> None:
        """Append a user turn, coalescing with the previous user turn if adjacent.

        Args:
            content: The user's input text.

        """
        if self.turns and self.turns[-1]["role"] == "user":
            self.turns[-1]["content"][0]["text"] += "\n" + content
            self.turns[-1]["timestamp"] = time.time()
        else:
            self.turns.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": content}],
                    "timestamp": time.time(),
                }
            )

    def get_all_turns(self) -> list[dict[str, Any]]:
        """Return a deep copy of all turns with their timestamps.

        Returns:
            List of turn dictionaries.

        """
        return deepcopy(self.turns)

    def get_turns_since(self, timestamp: float) -> list[dict[str, Any]]:
        """Return turns that occurred after the specified timestamp.

        Args:
            timestamp: Unix timestamp cutoff (exclusive).

        Returns:
            List of turn dictionaries whose timestamp is greater than *timestamp*.

        """
        return [turn for turn in self.turns if turn["timestamp"] > timestamp]
