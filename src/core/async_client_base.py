# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Async abstract base class for AI client implementations.

Mirrors ``FFAIClientBase`` but with ``generate_response`` and ``clone``
as async methods.  In-memory operations (conversation history get/set,
clear) remain synchronous.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from .client_base import FFAIClientBase


class AsyncFFAIClientBase(FFAIClientBase):
    """Async variant of ``FFAIClientBase`` for use with ``execute_graph``.

    Subclasses must implement ``generate_response`` and ``clone`` as async
    methods.  All other methods (``clear_conversation``,
    ``get_conversation_history``, ``set_conversation_history``) remain
    synchronous because they operate on in-memory lists.

    """

    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the AI model.

        Args:
            prompt: The user prompt to send to the model.
            **kwargs: Additional model-specific parameters.

        Returns:
            The generated response string.

        """
        pass

    @abstractmethod
    async def clone(self) -> AsyncFFAIClientBase:
        """Create a fresh async clone of this client with empty history.

        Returns:
            New ``AsyncFFAIClientBase`` instance with same config, empty history.

        """
        pass
