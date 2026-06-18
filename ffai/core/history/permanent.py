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
    structured ``content``, a per-turn ``timestamp``, and optional
    ``metadata``. Consecutive user turns are coalesced by appending
    content **unless** the caller passes non-``None`` metadata, in
    which case a new turn is always created (so distinct ``prompt_name``
    metadata is never silently merged).

    """

    def __init__(self) -> None:
        self.turns: list[dict[str, Any]] = []
        self.timestamp: float = time.time()

    def add_turn_assistant(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append an assistant turn with the current timestamp.

        Args:
            content: The assistant's response text.
            metadata: Optional caller metadata. When provided, stored on
                the turn dict under the ``"metadata"`` key. Defaults to
                an empty dict.

        """
        self.turns.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": content}],
                "timestamp": time.time(),
                "metadata": deepcopy(metadata) if metadata is not None else {},
            }
        )

    def add_turn_user(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a user turn, coalescing with the previous user turn when adjacent.

        Coalescing only happens when *metadata* is ``None``. If metadata
        is provided, a new turn is always created so distinct
        ``prompt_name`` metadata is never silently merged into the
        previous user turn.

        Args:
            content: The user's input text.
            metadata: Optional caller metadata. When non-``None``, forces
                creation of a new turn (no coalescing).

        """
        if (
            metadata is None
            and self.turns
            and self.turns[-1]["role"] == "user"
            and not self.turns[-1].get("metadata")
        ):
            self.turns[-1]["content"][0]["text"] += "\n" + content
            self.turns[-1]["timestamp"] = time.time()
            return

        self.turns.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": content}],
                "timestamp": time.time(),
                "metadata": deepcopy(metadata) if metadata is not None else {},
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
