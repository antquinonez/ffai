# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Client conversation history management with suspend/restore support."""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages the underlying client's conversation history.

    Handles:
    - Thread-safe suspend/restore for declarative history injection
    - Proxy access to the client's conversation with error handling
    - Message appending and conversation clearing

    Args:
        client: The AI client whose conversation history is managed.
        lock: Optional lock for thread-safe access.

    """

    def __init__(
        self,
        client: Any,
        lock: threading.Lock | None = None,
    ) -> None:
        self._client = client
        self._lock = lock or threading.Lock()

    @property
    def client(self) -> Any:
        return self._client

    @client.setter
    def client(self, value: Any) -> None:
        self._client = value

    def should_suspend(self, prompt: str, history: list[str] | None) -> bool:
        """Determine if client history should be suspended for this call.

        Suspension is needed when a ``history`` parameter is provided or
        the prompt contains ``{{...}}`` interpolation patterns.

        Args:
            prompt: The prompt text.
            history: The declarative history list.

        Returns:
            True if client history should be suspended.
        """
        has_interpolation = "{{" in prompt and "}}" in prompt
        return history is not None or has_interpolation

    def suspend(self, reason: str = "") -> list[dict[str, Any]] | None:
        """Save and clear the client's conversation history.

        Args:
            reason: Log message reason for suspension.

        Returns:
            Saved history if suspended, None if client lacks methods.
        """
        if not (
            hasattr(self._client, "get_conversation_history")
            and hasattr(self._client, "set_conversation_history")
        ):
            return None

        with self._lock:
            saved = self._client.get_conversation_history().copy()
            self._client.set_conversation_history([])
        logger.debug(f"Suspended client conversation history: {reason}")
        return saved

    def restore(self, saved: list[dict[str, Any]] | None) -> None:
        """Merge new messages back into the saved client history.

        Args:
            saved: Previously saved history from ``suspend()``.
        """
        if saved is None:
            return
        with self._lock:
            new_msgs = self._client.get_conversation_history()
            combined = list(saved) + list(new_msgs)
            self._client.set_conversation_history(combined)
            logger.debug(f"Restored client conversation history (+{len(new_msgs)} new messages)")

    def get_history(self) -> list[dict[str, Any]]:
        """Get the client's conversation history with error handling."""
        try:
            if hasattr(self._client, "get_conversation_history"):
                return self._client.get_conversation_history()
            logger.warning("Client does not support retrieving conversation history")
            return []
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e!s}")
            return []

    def set_history(self, history: list[dict[str, Any]]) -> bool:
        """Set the client's conversation history with error handling."""
        try:
            if hasattr(self._client, "set_conversation_history"):
                self._client.set_conversation_history(history)
                return True
            logger.warning("Client does not support setting conversation history")
            return False
        except Exception as e:
            logger.error(f"Error setting conversation history: {e!s}")
            return False

    def add_message(self, role: str, content: str, **kwargs: Any) -> bool:
        """Add a single message to the client's conversation history."""
        try:
            history = self.get_history()
            message = {"role": role, "content": content, **kwargs}
            history.append(message)
            return self.set_history(history)
        except Exception as e:
            logger.error(f"Error adding message to conversation history: {e!s}")
            return False

    def clear(self) -> None:
        """Clear the client's conversation history."""
        if hasattr(self._client, "clear_conversation"):
            self._client.clear_conversation()
