# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Manage API-facing message history for provider SDK calls."""

from __future__ import annotations

from typing import Any


class ConversationHistory:
    """API-facing message history for provider SDK calls.

    Unlike :class:`PermanentHistory`, turns here carry no timestamps
    and the history is intended to be cleared (e.g. when starting a
    new conversation context). The :meth:`get_turns` output is
    structured for direct injection into provider message APIs.

    """

    def __init__(self) -> None:
        self.turns: list[dict[str, Any]] = []

    def add_turn_assistant(self, content: str) -> None:
        """Append an assistant turn to the history.

        Args:
            content: The assistant's response text.

        """
        self.turns.append({"role": "assistant", "content": [{"type": "text", "text": content}]})

    def add_turn_user(self, content: str) -> None:
        """Append a user turn, coalescing with the previous user turn if adjacent.

        Args:
            content: The user's input text.

        """
        if self.turns and self.turns[-1]["role"] == "user":
            self.turns[-1]["content"][0]["text"] += "\n" + content
        else:
            self.turns.append({"role": "user", "content": [{"type": "text", "text": content}]})

    def get_turns(self) -> list[dict[str, Any]]:
        """Return all turns formatted for provider message APIs.

        Returns:
            List of message dictionaries with ``role`` and ``content`` keys.

        """
        result = []
        for turn in self.turns:
            if turn["role"] == "user":
                result.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": turn["content"][0]["text"]}],
                    }
                )
            else:
                result.append(turn)
        return result
